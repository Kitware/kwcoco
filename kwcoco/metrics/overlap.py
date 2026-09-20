"""Geometry-overlap helpers used by experimental metric assignment.

This module keeps geometry conversion separate from assignment semantics.
Instance-mask evaluation has two independent choices:

* rasterization: how polygon / mask annotations become full-image RLE masks;
* overlap: which backend computes pairwise IoU between those RLE masks.

The distinction matters because COCO polygon rasterization is not generally
identical to generic image-space polygon filling.  ``rasterizer='coco'`` uses
COCO's ``frPoly`` convention through ``kwimage_ext`` when available, with
``pycocotools`` as a reference fallback.  ``rasterizer='kwimage'`` preserves
the historical kwimage-mask path for controlled comparisons.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _backend_ready_rle(rle: dict[str, Any]) -> dict[str, Any]:
    rle = dict(rle)
    counts = rle.get('counts', None)
    if isinstance(counts, str):
        rle['counts'] = counts.encode('ascii')
    rle['size'] = list(map(int, rle['size']))
    return rle


def _empty_uncompressed_rle(dims: tuple[int, int]):
    height, width = map(int, dims)
    return {'size': [height, width], 'counts': [height * width]}


def resolve_mask_iou_backend(backend: str = 'auto'):
    """Resolve an explicit batched mask-IoU backend and describe it."""
    if backend not in {'auto', 'kwimage_ext', 'pycocotools'}:
        raise KeyError(backend)

    errors = []
    if backend in {'auto', 'kwimage_ext'}:
        try:
            from kwimage_ext.structs._mask_backend import cython_mask
        except Exception as ex:
            errors.append(('kwimage_ext', ex))
            if backend == 'kwimage_ext':
                raise
        else:
            metadata = {'name': 'kwimage_ext'}
            if hasattr(cython_mask, 'backend_metadata'):
                try:
                    metadata['dispatch'] = cython_mask.backend_metadata()
                except Exception as ex:  # pragma: no cover - diagnostic only
                    metadata['dispatch_error'] = repr(ex)
            return cython_mask.iou, metadata

    if backend in {'auto', 'pycocotools'}:
        try:
            from pycocotools import mask as mask_utils
        except Exception as ex:
            errors.append(('pycocotools', ex))
            if backend == 'pycocotools':
                raise
        else:
            return mask_utils.iou, {'name': 'pycocotools'}

    detail = ', '.join(f'{name}: {ex!r}' for name, ex in errors)
    raise ImportError(
        'native mask IoU requires either kwimage_ext or pycocotools; '
        f'neither backend is available ({detail})'
    )


class _CocoRasterBackend:
    """Small adapter over COCO-style mask codec APIs."""

    def __init__(self, name, module, metadata):
        self.name = name
        self.module = module
        self.metadata = metadata

    def fr_poly(self, polygons, height, width):
        if self.name == 'kwimage_ext':
            return self.module.frPoly(polygons, height, width)
        return self.module.frPyObjects(polygons, height, width)

    def merge(self, rles):
        return self.module.merge(rles)

    def decode_one(self, rle):
        if self.name == 'kwimage_ext':
            arr = np.asarray(self.module.decode([rle]))
            if arr.ndim != 3 or arr.shape[2] != 1:
                raise ValueError(
                    'kwimage_ext decode returned unexpected shape: '
                    f'{arr.shape!r}'
                )
            return arr[:, :, 0]
        return np.asarray(self.module.decode(rle))

    def encode_one(self, mask):
        mask = np.asarray(mask, dtype=np.uint8, order='F')
        if self.name == 'kwimage_ext':
            return self.module.encode(mask)[0]
        return self.module.encode(np.asfortranarray(mask))

    def area(self, rles):
        return self.module.area(rles)

    def empty_rle(self, dims):
        height, width = map(int, dims)
        if self.name == 'kwimage_ext':
            return self.module.frUncompressedRLE(
                [_empty_uncompressed_rle(dims)], height, width)[0]
        return self.encode_one(np.zeros((height, width), dtype=np.uint8))


def resolve_coco_raster_backend(backend: str = 'auto'):
    """Resolve a COCO-compatible polygon/RLE rasterization backend.

    ``kwimage_ext`` is preferred because its Rust ``frPoly`` implementation is
    intended to preserve COCO mask semantics without requiring pycocotools at
    runtime.  ``pycocotools`` remains the reference fallback.
    """
    if backend not in {'auto', 'kwimage_ext', 'pycocotools'}:
        raise KeyError(backend)
    errors = []
    if backend in {'auto', 'kwimage_ext'}:
        try:
            from kwimage_ext.structs._mask_backend import cython_mask
        except Exception as ex:
            errors.append(('kwimage_ext', ex))
            if backend == 'kwimage_ext':
                raise
        else:
            metadata = {'name': 'kwimage_ext'}
            if hasattr(cython_mask, 'backend_metadata'):
                try:
                    metadata['dispatch'] = cython_mask.backend_metadata()
                except Exception as ex:  # pragma: no cover
                    metadata['dispatch_error'] = repr(ex)
            return _CocoRasterBackend(
                'kwimage_ext', cython_mask, metadata), metadata

    if backend in {'auto', 'pycocotools'}:
        try:
            from pycocotools import mask as mask_utils
        except Exception as ex:
            errors.append(('pycocotools', ex))
            if backend == 'pycocotools':
                raise
        else:
            metadata = {'name': 'pycocotools'}
            return _CocoRasterBackend(
                'pycocotools', mask_utils, metadata), metadata

    detail = ', '.join(f'{name}: {ex!r}' for name, ex in errors)
    raise ImportError(
        'COCO-compatible mask rasterization requires either kwimage_ext or '
        f'pycocotools; neither backend is available ({detail})'
    )


def _polygon_parts_and_holes(mpoly):
    """Extract flat exterior/hole coordinates from a kwimage MultiPolygon."""
    parts = []
    for poly in mpoly.data:
        exterior = np.asarray(poly.data['exterior'].data, dtype=float)
        exterior = exterior.reshape(-1)
        holes = []
        for interior in poly.data.get('interiors', []):
            hole = np.asarray(interior.data, dtype=float).reshape(-1)
            holes.append(hole)
        parts.append((exterior, holes))
    return parts


def _segmentation_to_coco_rle(
    segmentation: Any,
    dims: tuple[int, int],
    *,
    raster_backend: str = 'auto',
    _resolved_backend=None,
):
    """Rasterize geometry with COCO ``frPoly`` semantics.

    Ordinary polygons never pass through a generic image polygon filler.  Rich
    polygons with holes are built from COCO-rasterized exterior/interior masks
    and then encoded back to RLE.  Existing mask/RLE inputs remain exact.
    """
    import kwimage

    height, width = map(int, dims)
    seg = kwimage.Segmentation.coerce(segmentation, dims=(height, width))
    if seg is None:
        raise ValueError('unable to coerce segmentation')

    data = seg.data
    if isinstance(data, kwimage.Mask):
        return _backend_ready_rle(data.to_bytes_rle().data)

    if _resolved_backend is None:
        codec, metadata = resolve_coco_raster_backend(raster_backend)
    else:
        codec, metadata = _resolved_backend

    mpoly = data.to_multi_polygon()
    parts = _polygon_parts_and_holes(mpoly)
    has_holes = any(holes for _exterior, holes in parts)

    if not has_holes:
        polygons = [
            exterior.tolist()
            for exterior, _holes in parts
            if exterior.size >= 6 and exterior.size % 2 == 0
        ]
        if not polygons:
            return _backend_ready_rle(codec.empty_rle((height, width)))
        rles = codec.fr_poly(polygons, height, width)
        rle = rles[0] if len(rles) == 1 else codec.merge(rles)
        return _backend_ready_rle(rle)

    # Strict COCO polygon syntax cannot express holes.  Preserve COCO boundary
    # rasterization for every ring, then subtract holes in image space.
    canvas = np.zeros((height, width), dtype=bool, order='F')
    for exterior, holes in parts:
        if exterior.size < 6 or exterior.size % 2:
            continue
        outer_rle = codec.fr_poly([exterior.tolist()], height, width)[0]
        outer = codec.decode_one(outer_rle).astype(bool, copy=False)
        for hole in holes:
            if hole.size < 6 or hole.size % 2:
                continue
            hole_rle = codec.fr_poly([hole.tolist()], height, width)[0]
            hole_mask = codec.decode_one(hole_rle).astype(bool, copy=False)
            outer = outer & ~hole_mask
        canvas |= outer
    return _backend_ready_rle(codec.encode_one(canvas))


def segmentation_to_rle(
    segmentation: Any,
    dims: tuple[int, int],
    *,
    rasterizer: str = 'kwimage',
    raster_backend: str = 'auto',
    _resolved_backend=None,
):
    """Rasterize one kwimage-compatible segmentation to compressed RLE.

    Args:
        rasterizer:
            ``'kwimage'`` preserves the historical ``Segmentation.to_mask``
            path. ``'coco'`` uses COCO ``frPoly`` semantics. ``'auto'`` means
            COCO semantics when an explicit COCO raster backend is available,
            otherwise the kwimage path.
    """
    import kwimage

    if segmentation is None:
        raise ValueError('segmentation is required for segmentation IoU')
    dims = tuple(map(int, dims))
    if len(dims) != 2 or min(dims) <= 0:
        raise ValueError(f'invalid image dimensions: {dims!r}')
    if rasterizer not in {'auto', 'kwimage', 'coco'}:
        raise KeyError(rasterizer)

    if rasterizer in {'auto', 'coco'}:
        try:
            return _segmentation_to_coco_rle(
                segmentation,
                dims,
                raster_backend=raster_backend,
                _resolved_backend=_resolved_backend,
            )
        except ImportError:
            if rasterizer == 'coco':
                raise
            # ``auto`` is the only mode allowed to fall back to generic
            # kwimage rasterization when no COCO codec is present.

    seg = kwimage.Segmentation.coerce(segmentation, dims=dims)
    if seg is None:
        raise ValueError('unable to coerce segmentation')
    mask = seg.to_mask(dims=dims)
    rle = mask.to_bytes_rle().data
    rle = _backend_ready_rle(rle)
    if tuple(rle['size']) != dims:
        raise ValueError(
            'segmentation RLE canvas mismatch: '
            f"{tuple(rle['size'])!r} != {dims!r}"
        )
    return rle


def detections_to_rles(
    dets,
    dims: tuple[int, int],
    *,
    rasterizer: str = 'kwimage',
    raster_backend: str = 'auto',
):
    """Convert a Detections segmentation column to compressed RLEs once."""
    segmentations = dets.data.get('segmentations', None)
    if segmentations is None:
        raise ValueError('Detections are missing the segmentations column')
    if len(segmentations) != len(dets):
        raise ValueError('segmentation column length does not match detections')

    resolved = None
    if rasterizer in {'auto', 'coco'}:
        try:
            resolved = resolve_coco_raster_backend(raster_backend)
        except ImportError:
            if rasterizer == 'coco':
                raise

    rles = [
        segmentation_to_rle(
            seg,
            dims,
            rasterizer=rasterizer,
            raster_backend=raster_backend,
            _resolved_backend=resolved,
        )
        for seg in segmentations
    ]
    metadata = {
        'rasterizer': rasterizer,
        'raster_backend': None if resolved is None else resolved[1],
    }
    return rles, metadata


def rle_areas(rles, *, backend: str = 'auto'):
    """Measure areas of compressed COCO RLEs without decoding masks.

    Returns:
        tuple[np.ndarray, dict]:
            Per-mask areas and backend provenance.
    """
    codec, metadata = resolve_coco_raster_backend(backend)
    if not rles:
        return np.empty(0, dtype=float), metadata
    areas = np.asarray(codec.area(rles), dtype=float)
    if areas.ndim == 0:
        areas = areas.reshape(1)
    areas = areas.reshape(-1)
    if len(areas) != len(rles):
        raise ValueError(
            'RLE area backend returned unexpected shape: '
            f'{areas.shape!r} for {len(rles)} masks'
        )
    return areas, metadata



def mask_ignore_region_flags(
    pred_rles,
    ignore_rles,
    *,
    threshold: float,
    threshold_policy: str = 'strict',
    overlap_backend: str = 'auto',
    raster_backend: str = 'auto',
):
    """Mark predictions substantially covered by segmentation ignore regions.

    The ignore regions are merged before overlap is measured, so overlapping
    ignore annotations cannot double-count covered pixels.  Coverage is
    measured as intersection over *prediction* area, which is exactly the
    denominator used by COCO mask IoU when the truth mask is marked crowd.

    This helper implements kwcoco's class-based ignore-region semantics.  It
    is intentionally distinct from annotation-level COCO ``iscrowd`` matching.
    """
    if threshold_policy not in {'strict', 'inclusive'}:
        raise KeyError(threshold_policy)
    if not pred_rles:
        return np.zeros(0, dtype=bool), {
            'overlap_backend': None,
            'raster_backend': None,
        }
    if not ignore_rles:
        return np.zeros(len(pred_rles), dtype=bool), {
            'overlap_backend': None,
            'raster_backend': None,
        }

    codec, raster_meta = resolve_coco_raster_backend(raster_backend)
    union_rle = _backend_ready_rle(codec.merge(ignore_rles))
    overlap, overlap_meta = pairwise_segmentation_iou(
        pred_rles, [union_rle], iscrowd=[1], backend=overlap_backend
    )
    coverage = overlap[:, 0]
    if threshold_policy == 'strict':
        flags = coverage > threshold
    else:
        flags = coverage >= threshold
    metadata = {
        'overlap_backend': overlap_meta,
        'raster_backend': raster_meta,
        'coverage': coverage,
    }
    return flags, metadata

def pairwise_segmentation_iou(
    pred_rles,
    true_rles,
    *,
    iscrowd=None,
    backend: str = 'auto',
):
    """Compute batched prediction-vs-truth mask IoUs from compressed RLEs."""
    if iscrowd is None:
        iscrowd = np.zeros(len(true_rles), dtype=np.uint8)
    else:
        iscrowd = np.asarray(iscrowd, dtype=np.uint8)
    if len(iscrowd) != len(true_rles):
        raise ValueError('iscrowd length must equal number of truth masks')
    func, metadata = resolve_mask_iou_backend(backend)
    matrix = np.asarray(func(pred_rles, true_rles, iscrowd.tolist()), dtype=float)
    expected = (len(pred_rles), len(true_rles))
    if matrix.shape != expected:
        # Some C-extension empty-input paths historically returned 1-D empty
        # arrays.  Normalize only genuinely empty matrices; anything else is a
        # backend contract violation.
        if 0 in expected and matrix.size == 0:
            matrix = np.empty(expected, dtype=float)
        else:
            raise ValueError(
                f'mask IoU backend returned shape={matrix.shape!r}, '
                f'expected={expected!r}'
            )
    return matrix, metadata
