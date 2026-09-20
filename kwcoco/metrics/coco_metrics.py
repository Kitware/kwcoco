"""
Helpers for official MS-COCO reference metrics.

The functions in this module deliberately bridge to ``pycocotools``.  They
provide an ecosystem-compatible reference backend; they do not replace
kwcoco's native ``ConfusionVectors -> Measures`` metric architecture.  See
``dev/pycocotools_metrics_compare`` for the measured comparison suite used to
separate overlap, assignment, and accumulation differences.

This module contains the narrow compatibility layer needed when rich kwcoco
segmentations are passed to ``pycocotools``.  KWCoco permits polygon interiors
(holes), whereas the original MS-COCO polygon representation does not.  For
metric computation we preserve ordinary polygons as polygons and encode only
representations that cannot be expressed losslessly in strict MS-COCO polygon
form as RLE masks.
"""
from __future__ import annotations

from typing import Any

import numpy as np


COCO_DETECTION_STAT_KEYS = (
    'AP',
    'AP50',
    'AP75',
    'AP_small',
    'AP_medium',
    'AP_large',
    'AR1',
    'AR10',
    'AR100',
    'AR_small',
    'AR_medium',
    'AR_large',
)


COCO_IOU_THRESHOLDS = tuple(np.linspace(0.50, 0.95, 10).tolist())
COCO_RECALL_THRESHOLDS = tuple(np.linspace(0.0, 1.0, 101).tolist())
COCO_MAX_DETS = (1, 10, 100)
COCO_AREA_RANGES = {
    'all': (0.0, 1e10),
    'small': (0.0, float(32 ** 2)),
    'medium': (float(32 ** 2), float(96 ** 2)),
    'large': (float(96 ** 2), 1e10),
}


def _rle_json_safe(rle: dict[str, Any]) -> dict[str, Any]:
    """Normalize a pycocotools RLE into a JSON-compatible dictionary."""
    rle = dict(rle)
    counts = rle.get('counts', None)
    if isinstance(counts, bytes):
        rle['counts'] = counts.decode('ascii')
    if 'size' in rle:
        rle['size'] = list(map(int, rle['size']))
    return rle


def _normalize_strict_coco_polygons(segmentation: list) -> list[list[float]]:
    """Normalize polygon-like lists into strict MS-COCO polygon syntax.

    ``pycocotools._mask.frPyObjects`` dispatches based on list lengths.  In
    particular, an ``N x 2`` polygon with exactly four vertices can be
    mistaken for an ``xywh`` bbox because the outer Python list has length
    four.  KWCoco / kwimage's rich polygon representation can naturally
    produce this ``N x 2`` form after coercion.

    Strict MS-COCO polygon syntax is unambiguous: a segmentation is a list of
    polygons, and each polygon is one flat ``[x1, y1, x2, y2, ...]`` list.
    Normalize to that representation before crossing the pycocotools boundary.
    """
    if not segmentation:
        return []

    first = segmentation[0]
    if np.isscalar(first):
        # A single already-flattened polygon.
        polygons = [segmentation]
    else:
        arr = np.asarray(segmentation, dtype=object)
        # An N x 2 coordinate-list is one polygon, not N polygon parts.
        if arr.ndim == 2 and arr.shape[1] == 2:
            polygons = [segmentation]
        else:
            polygons = segmentation

    normalized = []
    for poly in polygons:
        poly_arr = np.asarray(poly, dtype=float)
        if poly_arr.ndim == 2:
            if poly_arr.shape[1] != 2:
                raise ValueError(
                    'Polygon coordinate arrays must have shape (N, 2), '
                    f'got {poly_arr.shape!r}'
                )
            poly_arr = poly_arr.reshape(-1)
        elif poly_arr.ndim != 1:
            raise ValueError(
                'Polygon coordinates must be a flat list or an (N, 2) array, '
                f'got shape={poly_arr.shape!r}'
            )
        if poly_arr.size % 2 != 0:
            raise ValueError(
                'Polygon coordinate list must contain x/y pairs, '
                f'got {poly_arr.size} values'
            )
        # Strict MS-COCO polygons require at least three vertices.  In
        # particular, pycocotools.frPyObjects() dispatches a 4-value list to
        # its bbox converter, which makes a degenerate 2-vertex polygon fail
        # with a misleading bbox type error.  Perturbation / clipping code can
        # legitimately collapse a polygon to fewer than three vertices.  Such
        # parts have no polygonal area and are therefore omitted at the strict
        # COCO boundary.
        if poly_arr.size < 6:
            continue
        normalized.append(poly_arr.tolist())
    return normalized


def _strict_segmentation_to_rle(segmentation: Any, dims: tuple[int, int]):
    """Coerce a strict COCO segmentation into a compressed RLE."""
    from pycocotools import mask as mask_utils

    height, width = map(int, dims)
    if isinstance(segmentation, list):
        if len(segmentation) == 0:
            empty = np.zeros((height, width), dtype=np.uint8, order='F')
            rle = mask_utils.encode(empty)
        else:
            polygons = _normalize_strict_coco_polygons(segmentation)
            if polygons:
                rles = mask_utils.frPyObjects(polygons, height, width)
                rle = mask_utils.merge(rles)
            else:
                # COCOeval cannot consume an empty polygon list because
                # annToRLE() eventually calls frPyObjects([]), which indexes
                # the first element.  Encode the equivalent empty mask
                # explicitly instead.
                empty = np.zeros((height, width), dtype=np.uint8, order='F')
                rle = mask_utils.encode(empty)
    elif isinstance(segmentation, dict):
        rle = dict(segmentation)
        counts = rle.get('counts', None)
        if isinstance(counts, list):
            rle = mask_utils.frPyObjects(rle, height, width)
        elif isinstance(counts, str):
            # The C-extension operates on byte strings internally.
            rle['counts'] = counts.encode('ascii')
    else:  # pragma: no cover - guarded by the public coercion helper
        raise TypeError(type(segmentation))
    return rle


def _multipolygon_has_holes(mpoly) -> bool:
    return any(len(poly.data.get('interiors', [])) for poly in mpoly.data)


def _multipolygon_with_holes_to_rle(mpoly, dims: tuple[int, int]):
    """
    Rasterize a kwcoco polygon-with-holes using pycocotools polygon semantics.

    The original MS-COCO polygon syntax cannot represent interiors.  Using
    pycocotools itself for the rasterization keeps the evaluation boundary
    aligned with the implementation that will consume the resulting RLE.
    """
    from pycocotools import mask as mask_utils

    height, width = map(int, dims)
    canvas = np.zeros((height, width), dtype=np.uint8, order='F')
    for poly in mpoly.data:
        exterior = np.asarray(poly.data['exterior'].data, dtype=float)
        exterior_coco = exterior.reshape(-1).tolist()
        outer_rles = mask_utils.frPyObjects([exterior_coco], height, width)
        outer = mask_utils.decode(mask_utils.merge(outer_rles)).astype(bool)
        for interior in poly.data.get('interiors', []):
            interior_xy = np.asarray(interior.data, dtype=float)
            interior_coco = interior_xy.reshape(-1).tolist()
            hole_rles = mask_utils.frPyObjects([interior_coco], height, width)
            hole = mask_utils.decode(mask_utils.merge(hole_rles)).astype(bool)
            outer[hole] = False
        canvas[outer] = 1
    rle = mask_utils.encode(np.asfortranarray(canvas))
    return _rle_json_safe(rle)


def coerce_pycocotools_segmentation(
    segmentation: Any,
    dims: tuple[int, int],
) -> tuple[list | dict[str, Any], float]:
    """
    Convert any kwimage/kwcoco segmentation into strict MS-COCO geometry.

    Args:
        segmentation:
            Any segmentation representation accepted by
            :class:`kwimage.Segmentation`.

        dims:
            ``(height, width)`` of the parent image.  Explicit dimensions are
            required so hole-containing polygons can be represented exactly as
            full-image RLE masks.

    Returns:
        Tuple[strict_segmentation, raster_area]:
            The first item is accepted by ``pycocotools.COCOeval``.  The second
            is the corresponding rasterized area used by COCO area-range
            filtering.

    Notes:
        * Strict MS-COCO polygons are kept as polygons.  This lets
          ``pycocotools`` perform its normal polygon rasterization.
        * RLE inputs remain RLE.
        * KWCoco polygons containing interiors are converted to RLE because the
          original MS-COCO polygon syntax cannot represent holes.
    """
    import kwimage
    from pycocotools import mask as mask_utils

    if segmentation is None:
        raise ValueError('segmentation is required for iou_type="segm"')

    height, width = map(int, dims)
    if height <= 0 or width <= 0:
        raise ValueError(f'invalid image dimensions: {dims!r}')

    seg = kwimage.Segmentation.coerce(segmentation, dims=(height, width))
    if seg is None:
        raise ValueError('unable to coerce segmentation')

    data = seg.data
    if isinstance(data, kwimage.Mask):
        strict = _rle_json_safe(data.to_bytes_rle().to_coco())
    else:
        mpoly = data.to_multi_polygon()
        if _multipolygon_has_holes(mpoly):
            strict = _multipolygon_with_holes_to_rle(
                mpoly, dims=(height, width)
            )
        else:
            strict = mpoly.to_coco(style='orig')
            strict = _normalize_strict_coco_polygons(strict)
            if not strict:
                # A transformed or perturbed polygon may collapse to a line
                # or point.  That is not a valid strict COCO polygon.  Preserve
                # the intended zero-area segmentation as an explicit empty RLE
                # instead of letting pycocotools misinterpret it as a bbox.
                empty = np.zeros((height, width), dtype=np.uint8, order='F')
                strict = _rle_json_safe(mask_utils.encode(empty))

    rle = _strict_segmentation_to_rle(strict, dims=(height, width))
    rle_dims = tuple(map(int, rle['size']))
    if rle_dims != (height, width):
        raise ValueError(
            'Segmentation canvas does not match image dimensions: '
            f'{rle_dims!r} != {(height, width)!r}'
        )
    area = float(mask_utils.area(rle))
    return strict, area


def summarize_pycocotools_eval(evaler) -> dict[str, Any]:
    """Return stable, named JSON-friendly values from a standard COCOeval."""
    stats = np.asarray(evaler.stats, dtype=float).ravel()
    if len(stats) != len(COCO_DETECTION_STAT_KEYS):
        raise ValueError(
            'Expected standard 12-element COCO detection stats, '
            f'got {len(stats)}. Custom IoU/area summarization should be '
            'reported separately from the standard COCO summary.'
        )
    summary = {
        key: float(value)
        for key, value in zip(COCO_DETECTION_STAT_KEYS, stats)
    }
    params = {
        'iou_thresholds': np.asarray(evaler.params.iouThrs).astype(float).tolist(),
        'recall_thresholds': np.asarray(evaler.params.recThrs).astype(float).tolist(),
        'max_dets': list(map(int, evaler.params.maxDets)),
        'area_ranges': [list(map(float, r)) for r in evaler.params.areaRng],
        'area_range_labels': list(evaler.params.areaRngLbl),
        'image_ids': list(map(int, evaler.params.imgIds)),
        'category_ids': list(map(int, evaler.params.catIds)),
    }
    return {
        'backend': 'pycocotools.COCOeval',
        'iou_type': str(evaler.params.iouType),
        'summary': summary,
        'stats': stats.tolist(),
        'params': params,
    }
