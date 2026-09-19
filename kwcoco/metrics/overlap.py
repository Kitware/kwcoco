"""Geometry-overlap helpers used by experimental metric assignment.

This module keeps overlap computation separate from assignment semantics.  The
initial segmentation path converts rich kwimage segmentations to compressed
RLE once per detection, then evaluates pairwise mask IoU through an explicit
backend.  ``kwimage_ext`` is preferred when available; ``pycocotools`` is the
reference fallback.
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


def segmentation_to_rle(segmentation: Any, dims: tuple[int, int]):
    """Rasterize one kwimage-compatible segmentation to compressed RLE."""
    import kwimage

    if segmentation is None:
        raise ValueError('segmentation is required for segmentation IoU')
    dims = tuple(map(int, dims))
    if len(dims) != 2 or min(dims) <= 0:
        raise ValueError(f'invalid image dimensions: {dims!r}')
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


def detections_to_rles(dets, dims: tuple[int, int]):
    """Convert a Detections segmentation column to compressed RLEs once."""
    segmentations = dets.data.get('segmentations', None)
    if segmentations is None:
        raise ValueError('Detections are missing the segmentations column')
    if len(segmentations) != len(dets):
        raise ValueError('segmentation column length does not match detections')
    return [segmentation_to_rle(seg, dims) for seg in segmentations]


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
    raise ImportError(f'no mask IoU backend is available ({detail})')


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
