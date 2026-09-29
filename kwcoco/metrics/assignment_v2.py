"""
Experimental detection assignment primitives.

This module is intentionally additive and independent from
:mod:`kwcoco.metrics.assignment`.  The goal is to factor detection metric
assignment into testable, composable pieces:

* candidate generation
* optional sparse spatial indexing
* reusable per-image candidate indexes
* threshold-specific greedy assignment
* confusion-vector row emission

The current public evaluator does not use this module yet.  The main design
constraint preserved from the existing evaluator is that assignment emits
confusion-vector rows with traceable true / predicted detection indexes.

Relationship to :mod:`kwcoco.metrics.assignment`
-------------------------------------------------

This module is an alternative assignment design, not a promise of row-for-row
compatibility with the original implementation.  The two implementations now
share important baseline behavior, including the ``compat='all'`` wrapper
default, strict ``iou > threshold`` matching, ancestor relabeling in
``compat='ancestors'`` mode, and bias-aware sparse candidate generation.

Known semantic and feature differences are intentional unless an integration
decision says otherwise:

* When ``classes`` is omitted, v2 uses ``true_dets.classes`` or
  ``pred_dets.classes`` when available.  V1 does not infer the hierarchy this
  way and may fall back to flat matching.
* In ``compat='all'`` mode, v2 preserves the predicted class of a matched row.
  V1 can relabel a finer prediction to an ancestor truth when an explicit
  hierarchy is supplied.  V2 reserves that coercion for
  ``compat='ancestors'``.
* V2 uses a stable descending score sort, so equal-score predictions retain
  their input order.  V1 reverses ``argsort`` output, and its tie order is not
  the same contract.
* V2 currently implements only ``prioritize='iou'`` and ordinary truth reuse
  policy ``'never'``. Segmentation mode additionally implements COCO-style
  reusable crowd truths. V1 also supports class-prioritized matching and
  least-used ordinary truth reuse.
* V2 emits an additional ``rank`` column and can reuse one candidate index for
  multiple IoU thresholds.

Consequently, v2 can produce different confusion rows for accepted inputs even
when neither implementation is incorrect.  Tests that compare them should
assert exact equality only over their shared semantic contract and should
separately characterize the documented differences.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, overload
import math
import warnings

import numpy as np
import ubelt as ub

from kwcoco.metrics.assignment import ASSIGN_COLUMNS
from kwcoco.metrics.assignment import USE_NEG_INF
from kwcoco.metrics.assignment import _filter_ignore_regions

if TYPE_CHECKING:
    import kwimage
    from kwcoco.category_tree import CategoryTree


V2_ASSIGN_COLUMNS = ASSIGN_COLUMNS + ['rank']


@functools.lru_cache(maxsize=3)
def _resolve_assignment_backend_cached(backend: str):
    """Cache immutable backend resolution state for repeated area batches."""
    if backend not in {'auto', 'python', 'kwimage_ext'}:
        raise KeyError(backend)
    if backend in {'auto', 'kwimage_ext'}:
        try:
            from kwimage_ext.algo import assignment as ext_assignment
            func = ext_assignment.coco_greedy_match
            metadata = ext_assignment.backend_metadata()
        except Exception as ex:
            if backend == 'kwimage_ext':
                raise ImportError(
                    'assignment_backend="kwimage_ext" requires a kwimage_ext '
                    'build with the coco_greedy_match Rust capability'
                ) from ex
        else:
            return 'kwimage_ext', func, tuple(sorted(metadata.items()))
    metadata = {
        'name': 'kwcoco.assignment_v2.python',
        'kind': 'python',
        'module': __name__,
    }
    return 'python', None, tuple(sorted(metadata.items()))


def resolve_assignment_backend(backend: str = 'auto'):
    """Resolve the optional compiled greedy-assignment backend.

    Returns:
        tuple: ``(key, function, metadata)`` where ``function`` is ``None``
        for the Python implementation.  ``auto`` only falls back when the
        kwimage_ext capability is absent; an explicitly requested Rust backend
        fails loudly instead of silently changing provenance.

    Notes:
        Backend discovery is cached because native COCO evaluation resolves the
        same kernel for many image/area batches.  A fresh metadata dictionary is
        returned on every call so result provenance cannot mutate the cache.
    """
    key, func, metadata_items = _resolve_assignment_backend_cached(backend)
    return key, func, dict(metadata_items)


@functools.lru_cache(maxsize=3)
def _resolve_assignment_grid_backend_cached(backend: str):
    """Resolve the optional area x threshold batched assignment kernel."""
    if backend not in {'auto', 'python', 'kwimage_ext'}:
        raise KeyError(backend)
    if backend in {'auto', 'kwimage_ext'}:
        try:
            from kwimage_ext.algo import assignment as ext_assignment
            func = ext_assignment.coco_greedy_match_grid
            metadata = ext_assignment.backend_metadata()
            capabilities = set(metadata.get('capabilities', []))
            if 'coco-assignment-grid' not in capabilities:
                raise ImportError(
                    'kwimage_ext is installed but lacks '
                    'coco-assignment-grid support'
                )
        except Exception as ex:
            if backend == 'kwimage_ext':
                raise ImportError(
                    'assignment_backend="kwimage_ext" requires a kwimage_ext '
                    'build with the coco_greedy_match_grid Rust capability'
                ) from ex
        else:
            metadata = dict(metadata)
            metadata['name'] = 'kwimage_ext.coco_greedy_match_grid'
            metadata['capability'] = 'coco-assignment-grid'
            return 'kwimage_ext', func, tuple(sorted(metadata.items()))
    metadata = {
        'name': 'kwcoco.assignment_v2.python_grid',
        'kind': 'python',
        'module': __name__,
        'capability': 'coco-assignment-grid',
    }
    return 'python', None, tuple(sorted(metadata.items()))


def resolve_assignment_grid_backend(backend: str = 'auto'):
    """Resolve the compact area x threshold assignment backend."""
    key, func, metadata_items = _resolve_assignment_grid_backend_cached(backend)
    return key, func, dict(metadata_items)


@dataclass
class BoxSpatialIndex:
    """
    Lightweight uniform-grid index over true boxes for candidate generation.

    This is not intended to be a universally optimal spatial index.  It is a
    small dependency-free structure that is only built when dense compatible
    class blocks look large enough to justify extra setup cost.

    Attributes:
        boxes_ltrb: true boxes as ``(x1, y1, x2, y2)``.
        cell_size: grid-cell size in image coordinate units.
        cell_to_idxs: maps integer grid cells to true-box indexes.
        max_cells_per_box: safety valve used during construction.
    """

    boxes_ltrb: np.ndarray
    cell_size: float
    cell_to_idxs: dict[tuple[int, int], list[int]]
    max_cells_per_box: int = 4096

    @classmethod
    def build(
        cls,
        boxes_ltrb: np.ndarray,
        cell_size: float | None = None,
        max_cells_per_box: int = 4096,
    ) -> 'BoxSpatialIndex | None':
        boxes_ltrb = np.asarray(boxes_ltrb, dtype=float)
        if len(boxes_ltrb) == 0:
            return None
        if cell_size is None:
            widths = np.maximum(0, boxes_ltrb[:, 2] - boxes_ltrb[:, 0])
            heights = np.maximum(0, boxes_ltrb[:, 3] - boxes_ltrb[:, 1])
            scales = np.maximum(widths, heights)
            scales = scales[np.isfinite(scales) & (scales > 0)]
            if len(scales) == 0:
                return None
            cell_size = float(np.median(scales))
        if not np.isfinite(cell_size) or cell_size <= 0:
            return None

        cell_to_idxs: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, box in enumerate(boxes_ltrb):
            cells = _box_cells(box, cell_size)
            if cells is None or len(cells) > max_cells_per_box:
                # If a pathological box would touch too many cells, do not use
                # this simple grid index for the block.
                return None
            for cell in cells:
                cell_to_idxs[cell].append(idx)
        return cls(
            boxes_ltrb=boxes_ltrb,
            cell_size=cell_size,
            cell_to_idxs=dict(cell_to_idxs),
            max_cells_per_box=max_cells_per_box,
        )

    def query(self, box_ltrb: np.ndarray, bias: float = 0.0) -> np.ndarray:
        """Return candidate true-box indexes whose grid cells overlap a box."""
        query_box = np.asarray(box_ltrb, dtype=float)
        if bias > 0:
            query_box = query_box + np.array([-bias, -bias, bias, bias])
        cells = _box_cells(query_box, self.cell_size)
        if cells is None or len(cells) > self.max_cells_per_box:
            return np.arange(len(self.boxes_ltrb), dtype=int)
        cand = set()
        for cell in cells:
            cand.update(self.cell_to_idxs.get(cell, []))
        if not cand:
            return np.empty(0, dtype=int)
        cand_idxs = np.array(sorted(cand), dtype=int)
        cand_boxes = self.boxes_ltrb[cand_idxs]
        overlap = _ltrb_intersects(box_ltrb[None, :], cand_boxes, bias=bias)[0]
        return cand_idxs[overlap]


def _box_cells(box_ltrb: np.ndarray, cell_size: float) -> list[tuple[int, int]] | None:
    x1, y1, x2, y2 = map(float, box_ltrb)
    if not all(map(np.isfinite, [x1, y1, x2, y2])):
        return None
    if x2 < x1 or y2 < y1:
        return None
    ix1 = math.floor(x1 / cell_size)
    iy1 = math.floor(y1 / cell_size)
    ix2 = math.floor(x2 / cell_size)
    iy2 = math.floor(y2 / cell_size)
    return [(ix, iy) for ix in range(ix1, ix2 + 1) for iy in range(iy1, iy2 + 1)]


def _boxes_to_ltrb(boxes) -> np.ndarray:
    """Coerce a kwimage Boxes object to a numpy LTRB array."""
    data = np.asarray(boxes.to_ltrb().data, dtype=float)
    if data.size == 0:
        # kwimage can represent an empty Boxes payload as shape ``(0,)``.
        # Downstream bbox candidate construction is matrix-oriented and must
        # retain the coordinate axis even when one side of an image is empty.
        data = data.reshape(0, 4)
    elif data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(
            'Expected boxes coercible to an Nx4 LTRB matrix, got '
            f'shape={data.shape!r}'
        )
    return data


def _boxes_to_xywh(boxes) -> np.ndarray:
    """Coerce a kwimage Boxes object to a numpy XYWH array."""
    data = np.asarray(boxes.to_xywh().data, dtype=float)
    if data.size == 0:
        data = data.reshape(0, 4)
    elif data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(
            'Expected boxes coercible to an Nx4 XYWH matrix, got '
            f'shape={data.shape!r}'
        )
    return data


def _ltrb_intersects(
    left: np.ndarray, right: np.ndarray, bias: float = 0.0
) -> np.ndarray:
    """Return a dense boolean matrix indicating box intersection."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) == 0 or len(right) == 0:
        return np.empty((len(left), len(right)), dtype=bool)
    return (
        (left[:, None, 0] < right[None, :, 2] + bias)
        & (left[:, None, 2] + bias > right[None, :, 0])
        & (left[:, None, 1] < right[None, :, 3] + bias)
        & (left[:, None, 3] + bias > right[None, :, 1])
    )


def _threshold_mask(values, threshold: float, policy: str):
    """Apply an explicit IoU threshold boundary convention."""
    if policy == 'strict':
        return values > threshold
    if policy == 'inclusive':
        return values >= threshold
    raise KeyError(policy)


def _candidate_order(txs, ious, *, coco_tie_break=False):
    """Order candidates by overlap with an explicit equal-IoU policy.

    COCOeval scans ground truths in stable input order and replaces the best
    match when a later ground truth has *equal* overlap.  Therefore the
    equivalent pre-sorted representation prefers the later truth index for
    equal IoU.  Generic assignment-v2 keeps its existing stable-earlier tie
    convention unless a COCO compatibility profile opts in.
    """
    txs = np.asarray(txs, dtype=int)
    ious = np.asarray(ious, dtype=float)
    if coco_tie_break:
        return np.lexsort((-txs, -ious))
    return np.argsort(-ious, kind='mergesort')


def _coco_bbox_overlaps(
    pred_ltrb: np.ndarray,
    true_ltrb: np.ndarray,
    true_crowd_flags: np.ndarray,
    *,
    pred_areas: np.ndarray | None = None,
    true_areas: np.ndarray | None = None,
) -> np.ndarray:
    """COCO continuous-coordinate bbox overlap with crowd semantics.

    For ordinary truths this is standard IoU.  For ``iscrowd`` truths,
    pycocotools defines overlap as intersection divided by detection area so
    multiple detections can be absorbed by the same crowd region.
    """
    pred_ltrb = np.asarray(pred_ltrb, dtype=float)
    true_ltrb = np.asarray(true_ltrb, dtype=float)
    crowd = np.asarray(true_crowd_flags, dtype=bool)
    shape = (len(pred_ltrb), len(true_ltrb))
    if 0 in shape:
        return np.empty(shape, dtype=float)

    px1 = pred_ltrb[:, 0][:, None]
    py1 = pred_ltrb[:, 1][:, None]
    px2 = pred_ltrb[:, 2][:, None]
    py2 = pred_ltrb[:, 3][:, None]
    tx1 = true_ltrb[:, 0][None, :]
    ty1 = true_ltrb[:, 1][None, :]
    tx2 = true_ltrb[:, 2][None, :]
    ty2 = true_ltrb[:, 3][None, :]

    iw = np.maximum(0.0, np.minimum(px2, tx2) - np.maximum(px1, tx1))
    ih = np.maximum(0.0, np.minimum(py2, ty2) - np.maximum(py1, ty1))
    inter = iw * ih

    if pred_areas is None:
        pred_areas = (
            np.maximum(0.0, pred_ltrb[:, 2] - pred_ltrb[:, 0])
            * np.maximum(0.0, pred_ltrb[:, 3] - pred_ltrb[:, 1])
        )
    if true_areas is None:
        true_areas = (
            np.maximum(0.0, true_ltrb[:, 2] - true_ltrb[:, 0])
            * np.maximum(0.0, true_ltrb[:, 3] - true_ltrb[:, 1])
        )
    parea = np.asarray(pred_areas, dtype=float)[:, None]
    tarea = np.asarray(true_areas, dtype=float)[None, :]
    union = parea + tarea - inter
    denom = np.where(crowd[None, :], parea, union)
    overlaps = np.zeros(shape, dtype=float)
    np.divide(inter, denom, out=overlaps, where=denom > 0)
    return overlaps


@dataclass
class AssignmentCandidateIndex:
    """
    Reusable per-image candidate graph for detection assignment.

    The index stores candidate true detections for each prediction, their IoUs,
    and a score rank for each prediction.  Building this once allows repeated
    greedy assignment queries for different IoU thresholds without recomputing
    pairwise overlaps or compatible-class groups.

    The candidate graph is deliberately sparse: only candidate pairs with
    pairs passing the configured ``threshold_policy`` at ``min_iou`` are retained.
    For COCO-style AP50:95, ``min_iou=0.5`` is sufficient for all threshold
    queries.
    """

    true_dets: 'kwimage.Detections'
    pred_dets: 'kwimage.Detections'
    classes: 'CategoryTree | None'
    compat: str
    prioritize: str
    bias: float
    min_iou: float
    pred_order: np.ndarray
    pred_rank: np.ndarray
    pred_category_rank: np.ndarray
    max_category_size: int
    pred_scores: np.ndarray
    pred_cxs: np.ndarray
    true_cxs: np.ndarray
    cx_to_matchable_txs: dict[int, np.ndarray]
    cand_txs: list[np.ndarray]
    cand_ious: list[np.ndarray]
    iou_type: str
    bbox_overlap_mode: str
    overlap_backend: dict[str, Any] | None
    segmentation_rasterizer: str
    raster_backend: dict[str, Any] | None
    threshold_policy: str
    annotation_ignore_policy: str
    true_crowd_flags: np.ndarray
    true_annotation_ignore_flags: np.ndarray
    pred_rles: list[dict[str, Any]] | None
    true_rles: list[dict[str, Any]] | None
    pred_areas: np.ndarray
    true_areas: np.ndarray
    area_backend: dict[str, Any] | None
    image_dims: tuple[int, int] | None
    build_stats: dict[str, Any]
    area_ignore_cache: dict[
        tuple[float, float], tuple[np.ndarray, np.ndarray]
    ]
    pred_order_cache: dict[tuple[str, int], np.ndarray]
    candidate_csr_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None

    @classmethod
    def build(
        cls,
        true_dets: 'kwimage.Detections',
        pred_dets: 'kwimage.Detections',
        *,
        classes: 'CategoryTree | None' = None,
        compat: str = 'mutex',
        prioritize: str = 'iou',
        bias: float = 0.0,
        min_iou: float = 0.5,
        spatial_threshold: int | None = 50_000,
        iou_type: str = 'bbox',
        bbox_overlap_mode: str = 'legacy',
        overlap_backend: str = 'auto',
        segmentation_rasterizer: str = 'auto',
        raster_backend: str = 'auto',
        threshold_policy: str = 'auto',
        annotation_ignore_policy: str = 'coco',
        image_dims: tuple[int, int] | None = None,
    ) -> 'AssignmentCandidateIndex':
        """
        Build a reusable candidate index.

        Args:
            true_dets: kwimage-style true detections.
            pred_dets: kwimage-style predicted detections.
            classes: optional CategoryTree / class lookup.
            compat: one of ``mutex``, ``ancestors``, or ``all``.
            prioritize: currently only ``iou`` is implemented.
            bias: IoU bias forwarded to kwimage Boxes.ious.
            min_iou: discard candidate pairs below this threshold.
            spatial_threshold: if a compatible bbox block has more than this
                many dense pairs, try the sparse grid candidate generator.
                Segmentation overlap currently uses one batched mask-IoU call
                per compatible class block.
            iou_type: ``'bbox'`` or experimental ``'segm'`` overlap.
            bbox_overlap_mode: ``'legacy'`` preserves the existing v2 bbox
                overlap / crowd behavior. ``'coco'`` uses continuous-coordinate
                COCO bbox overlap, including intersection-over-detection-area
                for crowd truths. Native COCO bbox scoring opts into this
                explicitly; it is not the generic v2 bbox default.
            overlap_backend: mask-IoU backend for ``iou_type='segm'``.
            segmentation_rasterizer: ``'kwimage'`` or COCO-compatible
                ``'coco'`` polygon rasterization.
            raster_backend: backend for COCO-compatible polygon rasterization.
            threshold_policy: ``'strict'`` uses ``IoU > threshold`` (legacy
                kwcoco semantics); ``'inclusive'`` uses ``IoU >= threshold``
                to match COCOeval's threshold boundary. ``'auto'`` selects
                strict legacy-bbox matching and inclusive matching for
                segmentation or the explicit COCO bbox profile.
            annotation_ignore_policy: annotation-level ignore semantics for
                segmentation mode. ``'coco'`` matches official COCOeval's
                crowd-only ignore behavior; ``'kwcoco'`` additionally honors
                an explicit truth ``ignore`` column; ``'none'`` disables both
                crowd reuse and annotation-level ignore.
            image_dims: required ``(height, width)`` canvas for segmentation
                rasterization.
        """
        if prioritize != 'iou':
            raise NotImplementedError(
                'assignment_v2 currently implements prioritize="iou" only'
            )
        if iou_type not in {'bbox', 'segm'}:
            raise KeyError(iou_type)
        if bbox_overlap_mode not in {'legacy', 'coco'}:
            raise KeyError(bbox_overlap_mode)
        if bbox_overlap_mode == 'coco' and bias != 0:
            raise ValueError(
                'bbox_overlap_mode="coco" requires bias=0 because COCO '
                'uses continuous-coordinate bbox geometry without pixel bias'
            )
        if iou_type != 'bbox' and bbox_overlap_mode != 'legacy':
            raise ValueError(
                'bbox_overlap_mode only applies when iou_type="bbox"'
            )
        if threshold_policy == 'auto':
            threshold_policy = (
                'inclusive'
                if iou_type == 'segm' or bbox_overlap_mode == 'coco'
                else 'strict'
            )
        if threshold_policy not in {'strict', 'inclusive'}:
            raise KeyError(threshold_policy)
        if annotation_ignore_policy not in {'none', 'coco', 'kwcoco'}:
            raise KeyError(annotation_ignore_policy)
        if segmentation_rasterizer not in {'auto', 'kwimage', 'coco'}:
            raise KeyError(segmentation_rasterizer)
        if iou_type == 'segm':
            if bias != 0:
                raise NotImplementedError(
                    'segmentation IoU does not define bbox-style bias'
                )
            if image_dims is None:
                raise ValueError(
                    'image_dims=(height, width) is required for segmentation IoU'
                )
            image_dims = (int(image_dims[0]), int(image_dims[1]))
        valid_compat_keys = {'ancestors', 'mutex', 'all'}
        if compat not in valid_compat_keys:
            raise KeyError(compat)

        if classes is None:
            classes = getattr(true_dets, 'classes', None)
        if classes is None:
            classes = getattr(pred_dets, 'classes', None)
        if classes is None and compat == 'ancestors':
            compat = 'mutex'

        pred_cxs = np.asarray(pred_dets.class_idxs, dtype=int)
        true_cxs = np.asarray(true_dets.class_idxs, dtype=int)
        if 'scores' in pred_dets.data:
            pred_scores = np.asarray(pred_dets.scores, dtype=float)
        else:
            pred_scores = np.ones(len(pred_dets), dtype=float)
        pred_order = np.argsort(-pred_scores, kind='mergesort')
        pred_rank: np.ndarray = np.full(
            len(pred_dets), fill_value=-1, dtype=int
        )
        pred_rank[pred_order] = np.arange(len(pred_order))
        pred_category_rank = np.full(len(pred_dets), fill_value=-1, dtype=int)
        category_counts: dict[int, int] = defaultdict(int)
        for px in pred_order:
            px = int(px)
            cx = int(pred_cxs[px])
            pred_category_rank[px] = category_counts[cx]
            category_counts[cx] += 1
        max_category_size = max(category_counts.values(), default=0)

        unique_tcxs = sorted(set(map(int, true_cxs.tolist())))
        unique_pcxs = sorted(set(map(int, pred_cxs.tolist())))
        cx_to_txs = {
            cx: np.where(true_cxs == cx)[0].astype(int) for cx in unique_tcxs
        }

        if compat == 'mutex':
            cx_to_matchable_cxs = {cx: [cx] for cx in unique_pcxs}
        elif compat == 'all':
            cx_to_matchable_cxs = {cx: unique_tcxs for cx in unique_pcxs}
        elif compat == 'ancestors':
            import networkx as nx

            cx_to_matchable_cxs = {}
            for cx in unique_pcxs:
                ancestors = sorted(
                    ub.take(
                        classes.node_to_idx,  # type: ignore
                        nx.ancestors(classes.graph, classes.idx_to_node[cx]),  # type: ignore
                    )
                )
                cx_to_matchable_cxs[cx] = sorted([cx] + ancestors)
        else:  # pragma: no cover
            raise KeyError(compat)

        cx_to_matchable_txs = {}
        for cx, matchable_cxs in cx_to_matchable_cxs.items():
            parts = [cx_to_txs.get(int(tc), np.empty(0, dtype=int)) for tc in matchable_cxs]
            if parts:
                txs = np.array(sorted(ub.flatten(parts)), dtype=int)
            else:
                txs = np.empty(0, dtype=int)
            cx_to_matchable_txs[int(cx)] = txs

        cand_txs: list[np.ndarray] = [
            np.empty(0, dtype=int) for _ in range(len(pred_dets))
        ]
        cand_ious: list[np.ndarray] = [
            np.empty(0, dtype=float) for _ in range(len(pred_dets))
        ]
        build_stats: dict[str, Any] = {
            'iou_type': iou_type,
            'bbox_overlap_mode': bbox_overlap_mode,
            'threshold_policy': threshold_policy,
            'segmentation_rasterizer': segmentation_rasterizer,
            'annotation_ignore_policy': annotation_ignore_policy,
            'dense_blocks': 0,
            'spatial_blocks': 0,
            'dense_pairs_considered': 0,
            'sparse_pairs_considered': 0,
            'mask_pairs_considered': 0,
            'pairs_kept': 0,
        }

        pred_ltrb = _boxes_to_ltrb(pred_dets.boxes)
        true_ltrb = _boxes_to_ltrb(true_dets.boxes)

        pred_rles = true_rles = None
        mask_iou_func = None
        overlap_metadata = None
        area_metadata = None
        raw_crowd_flags = np.asarray(
            true_dets.data.get('iscrowd', np.zeros(len(true_dets), dtype=bool)),
            dtype=bool,
        )
        raw_ignore_flags = np.asarray(
            true_dets.data.get('ignore', np.zeros(len(true_dets), dtype=bool)),
            dtype=bool,
        )
        coco_annotation_mode = (
            iou_type == 'segm'
            or (iou_type == 'bbox' and bbox_overlap_mode == 'coco')
        )
        if annotation_ignore_policy == 'none' or not coco_annotation_mode:
            true_crowd_flags = np.zeros(len(true_dets), dtype=bool)
            true_annotation_ignore_flags = np.zeros(len(true_dets), dtype=bool)
        elif annotation_ignore_policy == 'coco':
            # Match the current official COCOeval implementation: for bbox /
            # segm evaluation, annotation-level ignore is driven by iscrowd.
            true_crowd_flags = raw_crowd_flags.copy()
            true_annotation_ignore_flags = raw_crowd_flags.copy()
        else:  # kwcoco
            true_crowd_flags = raw_crowd_flags.copy()
            true_annotation_ignore_flags = raw_crowd_flags | raw_ignore_flags

        if iou_type == 'segm':
            if image_dims is None:  # narrowed above; keep this local for type safety
                raise AssertionError('segmentation image dimensions were not resolved')
            from kwcoco.metrics.overlap import detections_to_rles
            from kwcoco.metrics.overlap import rle_areas
            from kwcoco.metrics.overlap import resolve_mask_iou_backend

            # Resolve the required overlap capability before rasterization.
            # Without a native COCO mask backend, kwimage's dense-mask to
            # compressed-RLE conversion can otherwise leak a lower-level
            # NotImplementedError instead of our documented dependency error.
            mask_iou_func, overlap_metadata = resolve_mask_iou_backend(
                overlap_backend
            )
            pred_rles, pred_raster_meta = detections_to_rles(
                pred_dets, image_dims,
                rasterizer=segmentation_rasterizer,
                raster_backend=raster_backend,
            )
            true_rles, true_raster_meta = detections_to_rles(
                true_dets, image_dims,
                rasterizer=segmentation_rasterizer,
                raster_backend=raster_backend,
            )
            if pred_raster_meta != true_raster_meta:
                raise AssertionError('prediction/truth raster backends disagree')
            pred_areas, pred_area_meta = rle_areas(
                pred_rles, backend=raster_backend
            )
            true_areas, true_area_meta = rle_areas(
                true_rles, backend=raster_backend
            )
            if pred_area_meta != true_area_meta:
                raise AssertionError('prediction/truth area backends disagree')
            area_metadata = pred_area_meta
            build_stats['overlap_backend'] = overlap_metadata
            build_stats['rasterization'] = pred_raster_meta
            build_stats['area_backend'] = area_metadata
        else:
            if bbox_overlap_mode == 'coco':
                # Preserve the original COCO XYWH width / height values for
                # area calculations.  Reconstructing them as ``x2 - x1`` can
                # move an exact 32^2 / 96^2 boundary across an area slice when
                # the origin is fractional due to floating-point cancellation.
                pred_xywh = _boxes_to_xywh(pred_dets.boxes)
                true_xywh = _boxes_to_xywh(true_dets.boxes)
                pred_widths = np.maximum(0, pred_xywh[:, 2])
                pred_heights = np.maximum(0, pred_xywh[:, 3])
                true_widths = np.maximum(0, true_xywh[:, 2])
                true_heights = np.maximum(0, true_xywh[:, 3])
                area_name = 'numpy_coco_xywh_area'
            else:
                pred_widths = np.maximum(0, pred_ltrb[:, 2] - pred_ltrb[:, 0])
                pred_heights = np.maximum(0, pred_ltrb[:, 3] - pred_ltrb[:, 1])
                true_widths = np.maximum(0, true_ltrb[:, 2] - true_ltrb[:, 0])
                true_heights = np.maximum(0, true_ltrb[:, 3] - true_ltrb[:, 1])
                area_name = 'numpy_bbox_area'
            pred_areas = pred_widths * pred_heights
            true_areas = true_widths * true_heights
            area_metadata = {
                'name': area_name,
                'kind': 'bbox_area',
            }
            build_stats['area_backend'] = area_metadata
            if bbox_overlap_mode == 'coco':
                overlap_metadata = {
                    'name': 'numpy_coco_bbox',
                    'kind': 'bbox_iou',
                    'crowd': 'intersection_over_detection_area',
                }
                build_stats['overlap_backend'] = overlap_metadata

        for pred_cx in unique_pcxs:
            pred_idxs = np.where(pred_cxs == pred_cx)[0].astype(int)
            true_idxs = cx_to_matchable_txs.get(int(pred_cx), np.empty(0, dtype=int))
            if len(pred_idxs) == 0 or len(true_idxs) == 0:
                continue
            dense_pair_count = int(len(pred_idxs) * len(true_idxs))
            if iou_type == 'segm':
                assert pred_rles is not None
                assert true_rles is not None
                assert mask_iou_func is not None
                block_pred_rles = [pred_rles[int(px)] for px in pred_idxs]
                block_true_rles = [true_rles[int(tx)] for tx in true_idxs]
                crowd = true_crowd_flags[true_idxs].astype(np.uint8).tolist()
                ious = np.asarray(
                    mask_iou_func(block_pred_rles, block_true_rles, crowd),
                    dtype=float,
                )
                expected = (len(pred_idxs), len(true_idxs))
                if ious.shape != expected:
                    if ious.size == 0 and 0 in expected:
                        ious = np.empty(expected, dtype=float)
                    else:
                        raise ValueError(
                            'mask IoU backend returned unexpected shape: '
                            f'{ious.shape!r} != {expected!r}'
                        )
                build_stats['dense_blocks'] += 1
                build_stats['mask_pairs_considered'] += dense_pair_count
                for px, row in zip(pred_idxs, ious):
                    keep = _threshold_mask(row, min_iou, threshold_policy)
                    if np.any(keep):
                        txs = true_idxs[keep]
                        iou_vals = np.asarray(row[keep], dtype=float)
                        order = _candidate_order(
                            txs, iou_vals, coco_tie_break=True
                        )
                        cand_txs[int(px)] = txs[order]
                        cand_ious[int(px)] = iou_vals[order]
                        build_stats['pairs_kept'] += int(len(order))
                continue

            use_spatial = (
                spatial_threshold is not None
                and dense_pair_count > spatial_threshold
                and min_iou > 0
            )
            if use_spatial:
                if bbox_overlap_mode == 'coco':
                    used = _populate_sparse_coco_bbox_candidates(
                        pred_ltrb,
                        true_ltrb,
                        true_crowd_flags,
                        pred_areas,
                        true_areas,
                        pred_idxs,
                        true_idxs,
                        min_iou=min_iou,
                        threshold_policy=threshold_policy,
                        cand_txs=cand_txs,
                        cand_ious=cand_ious,
                        build_stats=build_stats,
                    )
                else:
                    used = _populate_sparse_candidates(
                        pred_dets,
                        true_dets,
                        pred_ltrb,
                        true_ltrb,
                        pred_idxs,
                        true_idxs,
                        bias=bias,
                        min_iou=min_iou,
                        cand_txs=cand_txs,
                        cand_ious=cand_ious,
                        build_stats=build_stats,
                    )
                if used:
                    continue

            build_stats['dense_blocks'] += 1
            build_stats['dense_pairs_considered'] += dense_pair_count
            if bbox_overlap_mode == 'coco':
                ious = _coco_bbox_overlaps(
                    pred_ltrb[pred_idxs],
                    true_ltrb[true_idxs],
                    true_crowd_flags[true_idxs],
                    pred_areas=pred_areas[pred_idxs],
                    true_areas=true_areas[true_idxs],
                )
            else:
                ious = pred_dets.boxes[pred_idxs].ious(
                    true_dets.boxes[true_idxs], bias=bias
                )
            for px, row in zip(pred_idxs, ious):
                keep = _threshold_mask(row, min_iou, threshold_policy)
                if np.any(keep):
                    txs = true_idxs[keep]
                    iou_vals = np.asarray(row[keep], dtype=float)
                    order = _candidate_order(
                        txs,
                        iou_vals,
                        coco_tie_break=(bbox_overlap_mode == 'coco'),
                    )
                    cand_txs[int(px)] = txs[order]
                    cand_ious[int(px)] = iou_vals[order]
                    build_stats['pairs_kept'] += int(len(order))

        return cls(
            true_dets=true_dets,
            pred_dets=pred_dets,
            classes=classes,
            compat=compat,
            prioritize=prioritize,
            bias=bias,
            min_iou=float(min_iou),
            pred_order=pred_order,
            pred_rank=pred_rank,
            pred_category_rank=pred_category_rank,
            max_category_size=max_category_size,
            pred_scores=pred_scores,
            pred_cxs=pred_cxs,
            true_cxs=true_cxs,
            cx_to_matchable_txs=cx_to_matchable_txs,
            cand_txs=cand_txs,
            cand_ious=cand_ious,
            iou_type=iou_type,
            bbox_overlap_mode=bbox_overlap_mode,
            overlap_backend=overlap_metadata,
            segmentation_rasterizer=segmentation_rasterizer,
            raster_backend=build_stats.get('rasterization', {}).get('raster_backend'),
            threshold_policy=threshold_policy,
            annotation_ignore_policy=annotation_ignore_policy,
            true_crowd_flags=true_crowd_flags,
            true_annotation_ignore_flags=true_annotation_ignore_flags,
            pred_rles=pred_rles,
            true_rles=true_rles,
            pred_areas=pred_areas,
            true_areas=true_areas,
            area_backend=area_metadata,
            image_dims=image_dims,
            build_stats=build_stats,
            area_ignore_cache={},
            pred_order_cache={},
            candidate_csr_cache=None,
        )

    def _candidate_csr(self):
        """Return a lazily flattened CSR view of the sparse candidates."""
        cached = self.candidate_csr_cache
        if cached is None:
            counts = np.fromiter(
                (len(txs) for txs in self.cand_txs),
                dtype=np.int64,
                count=len(self.cand_txs),
            )
            offsets = np.empty(len(counts) + 1, dtype=np.int64)
            offsets[0] = 0
            np.cumsum(counts, out=offsets[1:])
            if offsets[-1]:
                flat_txs = np.concatenate(self.cand_txs).astype(
                    np.int64, copy=False
                )
                flat_ious = np.concatenate(self.cand_ious).astype(
                    np.float64, copy=False
                )
            else:
                flat_txs = np.empty(0, dtype=np.int64)
                flat_ious = np.empty(0, dtype=np.float64)
            cached = (
                np.ascontiguousarray(offsets),
                np.ascontiguousarray(flat_txs),
                np.ascontiguousarray(flat_ious),
            )
            self.candidate_csr_cache = cached
        return cached

    def assign(
        self,
        iou_thresh: float,
        *,
        bg_weight: float = 1.0,
        bg_cidx: int = -1,
        ignore_classes: str | Iterable[str] | None = 'ignore',
        max_dets: int | None = None,
        max_dets_scope: str = 'image',
        area_range: tuple[float, float] | None = None,
        truth_reuse_policy: str = 'never',
    ) -> dict[str, list[int | float]]:
        """Run greedy assignment for one IoU threshold.

        Args:
            max_dets_scope:
                ``'image'`` preserves the original v2 behavior. ``'category'``
                applies the cap independently to each prediction category,
                matching the normal COCO ``useCats=1`` evaluation scope.
            area_range:
                Optional inclusive ``(min_area, max_area)`` range. Truths
                outside the range remain matchable ignored candidates, and an
                unmatched prediction outside the range receives zero weight.
        """
        if truth_reuse_policy != 'never':
            raise NotImplementedError(
                'assignment_v2 currently implements truth_reuse_policy="never" only'
            )
        if iou_thresh < self.min_iou:
            raise ValueError(
                f'iou_thresh={iou_thresh!r} is below cached min_iou={self.min_iou!r}'
            )

        if max_dets_scope not in {'image', 'category'}:
            raise KeyError(max_dets_scope)

        true_available = np.ones(len(self.true_dets), dtype=bool)  # type: ignore
        true_nmatches = np.zeros(len(self.true_dets), dtype=int)  # type: ignore
        pred_order = self._pred_order_for(max_dets, max_dets_scope)

        true_area_ignore_flags, pred_area_ignore_flags = self._area_ignore_flags(
            area_range
        )

        true_ignore_flags = np.zeros(len(self.true_dets), dtype=bool)  # type: ignore
        pred_ignore_flags = np.zeros(len(self.pred_dets), dtype=bool)  # type: ignore
        if ignore_classes is not None:
            if self.iou_type == 'segm':
                true_ignore_flags, pred_ignore_flags = self._mask_ignore_regions(
                    iou_thresh=iou_thresh, ignore_classes=ignore_classes
                )
            else:
                true_ignore_flags, pred_ignore_flags = _filter_ignore_regions(
                    self.true_dets,
                    self.pred_dets,
                    ioaa_thresh=iou_thresh,
                    ignore_classes=ignore_classes,
                )
            true_available[true_ignore_flags] = False
            pred_order = pred_order[~pred_ignore_flags[pred_order]]

        annotation_ignore_flags = (
            self.true_annotation_ignore_flags | true_area_ignore_flags
        )
        crowd_flags = self.true_crowd_flags
        true_cxs = self.true_cxs
        pred_cxs = self.pred_cxs
        pred_scores = self.pred_scores
        cand_txs = self.cand_txs
        cand_ious = self.cand_ious
        pred_rank = self.pred_rank
        true_weights = self.true_dets.data.get('weights', None)  # type: ignore
        inclusive = self.threshold_policy == 'inclusive'

        y_pred: list = []
        y_true: list = []
        y_score: list = []
        y_weight: list = []
        y_iou: list = []
        y_pxs: list = []
        y_txs: list = []
        y_rank: list = []

        for px in pred_order:
            px = int(px)
            pred_cx = int(pred_cxs[px])
            score = float(pred_scores[px])
            txs = cand_txs[px]
            ious = cand_ious[px]
            chosen_tx = -1
            chosen_iou = -1.0
            if len(txs):
                # Candidate arrays are already sorted by descending overlap.
                # Scanning them directly is substantially cheaper than
                # allocating several tiny boolean numpy arrays per prediction,
                # and it naturally stops once overlap falls below threshold.
                ignored_fallback_tx = -1
                ignored_fallback_iou = -1.0
                for cand_pos in range(len(txs)):
                    cand_iou = float(ious[cand_pos])
                    if inclusive:
                        if cand_iou < iou_thresh:
                            break
                    else:
                        if cand_iou <= iou_thresh:
                            break
                    tx = int(txs[cand_pos])
                    if true_ignore_flags[tx]:
                        continue
                    if not true_available[tx] and not crowd_flags[tx]:
                        continue
                    if annotation_ignore_flags[tx]:
                        if ignored_fallback_tx < 0:
                            ignored_fallback_tx = tx
                            ignored_fallback_iou = cand_iou
                        continue
                    chosen_tx = tx
                    chosen_iou = cand_iou
                    break
                if chosen_tx < 0 and ignored_fallback_tx >= 0:
                    chosen_tx = ignored_fallback_tx
                    chosen_iou = ignored_fallback_iou

            if chosen_tx >= 0:
                if not crowd_flags[chosen_tx]:
                    true_available[chosen_tx] = False
                true_nmatches[chosen_tx] += 1
                true_cx = int(true_cxs[chosen_tx])
                if annotation_ignore_flags[chosen_tx]:
                    weight = 0.0
                elif true_weights is not None:
                    weight = float(true_weights[chosen_tx])
                else:
                    weight = 1.0
                assigned_pred_cx = pred_cx
                if self.compat == 'ancestors':
                    # Preserve v1 hierarchical scoring: a finer prediction
                    # matched to an ancestor truth is scored as that ancestor.
                    assigned_pred_cx = true_cx
                y_pred.append(assigned_pred_cx)
                y_true.append(true_cx)
                y_score.append(score)
                y_weight.append(weight)
                y_iou.append(chosen_iou)
                y_pxs.append(px)
                y_txs.append(chosen_tx)
                y_rank.append(int(pred_rank[px]))
            else:
                y_pred.append(pred_cx)
                y_true.append(bg_cidx)
                y_score.append(score)
                weight = 0.0 if pred_area_ignore_flags[px] else float(bg_weight)
                y_weight.append(weight)
                y_iou.append(-1.0)
                y_pxs.append(px)
                y_txs.append(-1)
                y_rank.append(int(pred_rank[px]))

        unused_flags = true_nmatches == 0
        unused_flags = unused_flags & np.logical_not(annotation_ignore_flags)
        if ignore_classes is not None:
            unused_flags = unused_flags & np.logical_not(true_ignore_flags)
        unused_txs = np.where(unused_flags)[0]
        if true_weights is not None:
            unused_y_weight = np.asarray(true_weights)[unused_txs].tolist()
        else:
            unused_y_weight = [1.0] * len(unused_txs)
        y_pred.extend([-1] * len(unused_txs))
        y_true.extend(self.true_cxs[unused_txs].tolist())
        y_score.extend([-np.inf if USE_NEG_INF else 0.0] * len(unused_txs))
        y_weight.extend(unused_y_weight)
        y_iou.extend([-1.0] * len(unused_txs))
        y_pxs.extend([-1] * len(unused_txs))
        y_txs.extend(unused_txs.tolist())
        y_rank.extend([-1] * len(unused_txs))

        return {
            'pred': y_pred,
            'true': y_true,
            'score': y_score,
            'weight': y_weight,
            'iou': y_iou,
            'txs': y_txs,
            'pxs': y_pxs,
            'rank': y_rank,
        }

    def _pred_order_for(
        self,
        max_dets: int | None,
        max_dets_scope: str,
    ) -> np.ndarray:
        """Return the stable score order after a cached maxDet cap."""
        if max_dets is None or not np.isfinite(max_dets):
            return self.pred_order
        max_dets = int(max_dets)
        if max_dets < 0:
            raise ValueError('max_dets must be nonnegative')
        key = (max_dets_scope, max_dets)
        cached = self.pred_order_cache.get(key, None)
        if cached is not None:
            return cached
        if max_dets_scope == 'image':
            order = self.pred_order[:max_dets]
        elif max_dets_scope == 'category':
            if max_dets >= self.max_category_size:
                order = self.pred_order
            else:
                flags = self.pred_category_rank[self.pred_order] < max_dets
                order = self.pred_order[flags]
        else:  # guarded by assign, useful for direct internal calls too
            raise KeyError(max_dets_scope)
        self.pred_order_cache[key] = order
        return order

    def _area_ignore_flags(
        self,
        area_range: tuple[float, float] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return cached truth/prediction area-ignore masks."""
        if area_range is None:
            return (
                np.zeros(len(self.true_dets), dtype=bool),
                np.zeros(len(self.pred_dets), dtype=bool),
            )
        if len(area_range) != 2:
            raise ValueError('area_range must be a (min_area, max_area) pair')
        min_area, max_area = map(float, area_range)
        if not np.isfinite(min_area) or min_area < 0:
            raise ValueError(f'invalid minimum area: {min_area!r}')
        if np.isnan(max_area) or max_area < min_area:
            raise ValueError(f'invalid maximum area: {max_area!r}')
        key = (min_area, max_area)
        cached = self.area_ignore_cache.get(key, None)
        if cached is None:
            cached = (
                (self.true_areas < min_area) | (self.true_areas > max_area),
                (self.pred_areas < min_area) | (self.pred_areas > max_area),
            )
            self.area_ignore_cache[key] = cached
        return cached

    def truncate_max_dets(
        self,
        rows: dict[str, list[int | float]],
        max_dets: int,
        *,
        max_dets_scope: str = 'category',
        bg_cidx: int = -1,
    ) -> dict[str, list[int | float]]:
        """Derive a smaller maxDet assignment from a larger-prefix result.

        Greedy matching is score ordered, so matches made by detections inside
        a smaller score prefix are independent of detections that come later.
        Removing the later rows is therefore equivalent to rerunning matching,
        except that a regular truth matched only by a removed detection must be
        restored as a false-negative row.
        """
        max_dets = int(max_dets)
        if max_dets < 0:
            raise ValueError('max_dets must be nonnegative')
        pxs = np.asarray(rows.get('pxs', []), dtype=int)
        txs = np.asarray(rows.get('txs', []), dtype=int)
        weights = np.asarray(rows.get('weight', []), dtype=float)
        if not (len(pxs) == len(txs) == len(weights)):
            raise ValueError('assignment rows are missing aligned trace columns')

        pred_rows = pxs >= 0
        if max_dets_scope == 'image':
            pred_keep = np.zeros(len(rows.get('pred', [])), dtype=bool)
            pred_keep[pred_rows] = self.pred_rank[pxs[pred_rows]] < max_dets
        elif max_dets_scope == 'category':
            pred_keep = np.zeros(len(rows.get('pred', [])), dtype=bool)
            pred_keep[pred_rows] = (
                self.pred_category_rank[pxs[pred_rows]] < max_dets
            )
        else:
            raise KeyError(max_dets_scope)
        # Direct assignment always emits score-ranked prediction rows first,
        # then false-negative rows in truth-index order. Rebuild that same row
        # shape instead of simply filtering the larger result so the derived
        # view remains row-for-row comparable to a direct rerun.
        result = {
            key: [
                value for value, flag in zip(values, pred_keep.tolist()) if flag
            ]
            for key, values in rows.items()
        }

        kept_match_txs = set(
            txs[pred_keep & (txs >= 0) & (weights > 0)].tolist()
        )
        fn_weights = {
            int(tx): float(weight)
            for tx, weight, is_pred in zip(txs, weights, pred_rows)
            if not is_pred and tx >= 0
        }
        dropped_regular_matches = pred_rows & ~pred_keep & (txs >= 0) & (weights > 0)
        for row_idx in np.where(dropped_regular_matches)[0]:
            tx = int(txs[row_idx])
            if tx not in kept_match_txs:
                fn_weights[tx] = float(weights[row_idx])

        if fn_weights:
            fn_score = -np.inf if USE_NEG_INF else 0.0
            for tx in sorted(fn_weights):
                for key in result:
                    if key == 'pred':
                        value = bg_cidx
                    elif key == 'true':
                        value = int(self.true_cxs[tx])
                    elif key == 'score':
                        value = fn_score
                    elif key == 'weight':
                        value = fn_weights[tx]
                    elif key == 'iou':
                        value = -1.0
                    elif key == 'txs':
                        value = tx
                    elif key in {'pxs', 'rank'}:
                        value = -1
                    else:
                        raise KeyError(
                            f'Cannot synthesize truncated assignment column {key!r}'
                        )
                    result[key].append(value)
        return result

    def _mask_ignore_regions(
        self,
        *,
        iou_thresh: float,
        ignore_classes: str | Iterable[str],
    ):
        """Compute kwcoco class-based ignore flags in mask geometry.

        This intentionally differs from annotation-level COCO crowd handling.
        Ignore-class truths are excluded from ordinary matching, and predictions
        sufficiently covered by the union of those masks are removed from the
        confusion rows.
        """
        import kwarray
        from kwcoco.metrics.overlap import mask_ignore_region_flags

        true_flags = np.zeros(len(self.true_dets), dtype=bool)
        pred_flags = np.zeros(len(self.pred_dets), dtype=bool)
        if isinstance(ignore_classes, str):
            ignore_classes = {ignore_classes}
        else:
            ignore_classes = set(ignore_classes)

        classes = self.true_dets.classes
        if classes is None or not ignore_classes:
            return true_flags, pred_flags

        normalized = set()
        for name in ignore_classes:
            if name in classes:
                normalized.add(name)
            else:
                lowered = str(name).lower()
                matches = [c for c in classes if str(c).lower() == lowered]
                if matches:
                    normalized.add(matches[0])
        ignore_cidxs = [classes.index(c) for c in normalized]
        if ignore_cidxs:
            true_flags = kwarray.isect_flags(self.true_cxs, ignore_cidxs)

        if np.any(true_flags) and len(self.pred_dets):
            if self.pred_rles is None or self.true_rles is None:
                raise AssertionError('segmentation ignore requires cached RLEs')
            ignore_rles = [
                self.true_rles[int(tx)] for tx in np.where(true_flags)[0]
            ]
            overlap_backend = 'auto'
            if self.overlap_backend is not None:
                overlap_backend = self.overlap_backend.get('name', 'auto')
            raster_backend = 'auto'
            if self.raster_backend is not None:
                raster_backend = self.raster_backend.get('name', 'auto')
            pred_flags, _meta = mask_ignore_region_flags(
                self.pred_rles,
                ignore_rles,
                threshold=iou_thresh,
                # Preserve historical kwcoco ignore-region semantics: IOAA
                # must be strictly greater than the threshold even when the
                # instance-assignment boundary uses COCO-inclusive IoU.
                threshold_policy='strict',
                overlap_backend=overlap_backend,
                raster_backend=raster_backend,
            )
        return true_flags, pred_flags


    def _match_coco_grid_python(
        self,
        thresholds,
        pred_order,
        annotation_ignore_grid,
        true_ignore_flags,
    ):
        """Reference compact matcher over area x threshold slices."""
        thresholds = np.asarray(thresholds, dtype=float)
        pred_order = np.asarray(pred_order, dtype=int)
        annotation_ignore_grid = np.asarray(annotation_ignore_grid, dtype=bool)
        true_ignore_flags = np.asarray(true_ignore_flags, dtype=bool)
        num_area = len(annotation_ignore_grid)
        num_thresh = len(thresholds)
        num_pred = len(pred_order)
        num_true = len(self.true_dets)  # type: ignore
        match_txs = np.full(
            (num_area, num_thresh, num_pred), -1, dtype=np.int64
        )
        match_ious = np.full(match_txs.shape, -1.0, dtype=float)
        true_nmatches = np.zeros(
            (num_area, num_thresh, num_true), dtype=np.int64
        )
        crowd_flags = self.true_crowd_flags
        inclusive = self.threshold_policy == 'inclusive'
        cand_txs = self.cand_txs
        cand_ious = self.cand_ious

        for ai in range(num_area):
            annotation_ignore = annotation_ignore_grid[ai]
            for ti, threshold in enumerate(thresholds):
                true_available = ~true_ignore_flags.copy()
                for oi, px_ in enumerate(pred_order):
                    px = int(px_)
                    chosen_tx = -1
                    chosen_iou = -1.0
                    fallback_tx = -1
                    fallback_iou = -1.0
                    txs = cand_txs[px]
                    ious = cand_ious[px]
                    for ci in range(len(txs)):
                        cand_iou = float(ious[ci])
                        if inclusive:
                            if cand_iou < threshold:
                                break
                        else:
                            if cand_iou <= threshold:
                                break
                        tx = int(txs[ci])
                        if true_ignore_flags[tx]:
                            continue
                        if not true_available[tx] and not crowd_flags[tx]:
                            continue
                        if annotation_ignore[tx]:
                            if fallback_tx < 0:
                                fallback_tx = tx
                                fallback_iou = cand_iou
                            continue
                        chosen_tx = tx
                        chosen_iou = cand_iou
                        break
                    if chosen_tx < 0 and fallback_tx >= 0:
                        chosen_tx = fallback_tx
                        chosen_iou = fallback_iou
                    if chosen_tx >= 0:
                        if not crowd_flags[chosen_tx]:
                            true_available[chosen_tx] = False
                        true_nmatches[ai, ti, chosen_tx] += 1
                        match_txs[ai, ti, oi] = chosen_tx
                        match_ious[ai, ti, oi] = chosen_iou
        return match_txs, match_ious, true_nmatches

    def match_coco_grid(
        self,
        thresholds,
        area_ranges,
        *,
        assignment_backend='auto',
        max_dets=None,
        max_dets_scope='category',
    ):
        """Return compact COCO match state for all area/IoU slices.

        This path intentionally does not materialize confusion rows.  It is
        restricted to COCO-compatible scoring where class-based ignore regions
        are not requested; diagnostic/native-extension paths continue to use
        :meth:`assign_many`.
        """
        thresholds = np.asarray(list(map(float, thresholds)), dtype=float)
        if len(thresholds) == 0:
            raise ValueError('thresholds must not be empty')
        if np.any(thresholds < self.min_iou):
            raise ValueError(
                'threshold below cached min_iou: '
                f'{thresholds.min()!r} < {self.min_iou!r}'
            )
        area_ranges = list(area_ranges)
        pred_order = self._pred_order_for(max_dets, max_dets_scope)
        true_ignore_flags = np.zeros(len(self.true_dets), dtype=bool)  # type: ignore
        annotation_rows = []
        pred_area_rows = []
        for area_range in area_ranges:
            true_area_ignore, pred_area_ignore = self._area_ignore_flags(area_range)
            annotation_rows.append(
                self.true_annotation_ignore_flags | true_area_ignore
            )
            pred_area_rows.append(pred_area_ignore[pred_order])
        if annotation_rows:
            annotation_ignore_grid = np.stack(annotation_rows, axis=0)
            pred_area_ignore_grid = np.stack(pred_area_rows, axis=0)
        else:
            annotation_ignore_grid = np.empty(
                (0, len(self.true_dets)), dtype=bool  # type: ignore
            )
            pred_area_ignore_grid = np.empty((0, len(pred_order)), dtype=bool)

        backend_key, rust_func, backend_meta = resolve_assignment_grid_backend(
            assignment_backend
        )
        if backend_key == 'kwimage_ext':
            offsets, flat_txs, flat_ious = self._candidate_csr()
            match_txs, match_ious, true_nmatches = rust_func(
                offsets,
                flat_txs,
                flat_ious,
                pred_order,
                thresholds,
                true_ignore_flags,
                annotation_ignore_grid,
                self.true_crowd_flags,
                inclusive=self.threshold_policy == 'inclusive',
            )
            match_txs = np.asarray(match_txs, dtype=np.int64)
            match_ious = np.asarray(match_ious, dtype=float)
            true_nmatches = np.asarray(true_nmatches, dtype=np.int64)
        else:
            match_txs, match_ious, true_nmatches = self._match_coco_grid_python(
                thresholds,
                pred_order,
                annotation_ignore_grid,
                true_ignore_flags,
            )

        expected_match_shape = (
            len(area_ranges), len(thresholds), len(pred_order)
        )
        expected_truth_shape = (
            len(area_ranges), len(thresholds), len(self.true_dets)  # type: ignore
        )
        if match_txs.shape != expected_match_shape:
            raise ValueError(
                f'compact matcher returned {match_txs.shape!r}, '
                f'expected {expected_match_shape!r}'
            )
        if match_ious.shape != expected_match_shape:
            raise ValueError(
                f'compact matcher IoUs returned {match_ious.shape!r}, '
                f'expected {expected_match_shape!r}'
            )
        if true_nmatches.shape != expected_truth_shape:
            raise ValueError(
                f'compact matcher truth counts returned {true_nmatches.shape!r}, '
                f'expected {expected_truth_shape!r}'
            )

        return {
            'pred_order': np.asarray(pred_order, dtype=np.int64),
            'pred_cxs': np.asarray(self.pred_cxs[pred_order], dtype=np.int32),
            'pred_scores': np.asarray(self.pred_scores[pred_order], dtype=float),
            'pred_category_rank': np.asarray(
                self.pred_category_rank[pred_order], dtype=np.int32
            ),
            'match_txs': match_txs,
            'match_ious': match_ious,
            'true_nmatches': true_nmatches,
            'true_cxs': np.asarray(self.true_cxs, dtype=np.int32),
            'annotation_ignore_flags': annotation_ignore_grid,
            'pred_area_ignore_flags': pred_area_ignore_grid,
            'backend': backend_key,
            'backend_metadata': backend_meta,
        }

    def _materialize_match_matrices(
        self,
        thresholds,
        pred_order,
        match_txs,
        match_ious,
        true_nmatches,
        *,
        bg_weight,
        bg_cidx,
        true_ignore_flags,
        annotation_ignore_flags,
        pred_area_ignore_flags,
    ):
        """Convert compact batched match matrices into v2 assignment rows."""
        thresholds = list(map(float, thresholds))
        pred_order = np.asarray(pred_order, dtype=int)
        match_txs = np.asarray(match_txs, dtype=np.int64)
        match_ious = np.asarray(match_ious, dtype=float)
        true_nmatches = np.asarray(true_nmatches, dtype=np.int64)
        expected_matches = (len(thresholds), len(pred_order))
        expected_truth = (len(thresholds), len(self.true_dets))  # type: ignore
        if match_txs.shape != expected_matches or match_ious.shape != expected_matches:
            raise ValueError(
                'assignment backend returned unexpected match shape: '
                f'{match_txs.shape!r}, {match_ious.shape!r} != {expected_matches!r}'
            )
        if true_nmatches.shape != expected_truth:
            raise ValueError(
                'assignment backend returned unexpected truth-count shape: '
                f'{true_nmatches.shape!r} != {expected_truth!r}'
            )

        pred_cxs = self.pred_cxs[pred_order]
        pred_scores = self.pred_scores[pred_order]
        pred_ranks = self.pred_rank[pred_order]
        pred_area_ignored = pred_area_ignore_flags[pred_order]
        true_weights = self.true_dets.data.get('weights', None)  # type: ignore
        if true_weights is not None:
            true_weights = np.asarray(true_weights, dtype=float)
        fn_score = -np.inf if USE_NEG_INF else 0.0
        results = {}

        for ti, threshold in enumerate(thresholds):
            row_txs = match_txs[ti]
            matched = row_txs >= 0
            y_pred = np.asarray(pred_cxs, dtype=int).copy()
            y_true = np.full(len(pred_order), bg_cidx, dtype=int)
            y_score = np.asarray(pred_scores, dtype=float).copy()
            y_weight = np.full(len(pred_order), float(bg_weight), dtype=float)
            y_iou = np.asarray(match_ious[ti], dtype=float).copy()
            y_pxs = np.asarray(pred_order, dtype=int).copy()
            y_txs = np.asarray(row_txs, dtype=int).copy()
            y_rank = np.asarray(pred_ranks, dtype=int).copy()

            if np.any(matched):
                matched_pos = np.where(matched)[0]
                matched_txs = row_txs[matched]
                matched_true_cxs = self.true_cxs[matched_txs]
                y_true[matched] = matched_true_cxs
                if self.compat == 'ancestors':
                    y_pred[matched] = matched_true_cxs
                if true_weights is None:
                    y_weight[matched] = 1.0
                else:
                    y_weight[matched] = true_weights[matched_txs]
                ignored_matches = annotation_ignore_flags[matched_txs]
                if np.any(ignored_matches):
                    y_weight[matched_pos[ignored_matches]] = 0.0

            unmatched = ~matched
            if np.any(unmatched):
                area_ignored_fp = unmatched & pred_area_ignored
                y_weight[area_ignored_fp] = 0.0

            unused_flags = true_nmatches[ti] == 0
            unused_flags &= np.logical_not(annotation_ignore_flags)
            unused_flags &= np.logical_not(true_ignore_flags)
            unused_txs = np.where(unused_flags)[0]
            if len(unused_txs):
                if true_weights is None:
                    fn_weights = np.ones(len(unused_txs), dtype=float)
                else:
                    fn_weights = true_weights[unused_txs]
                y_pred = np.concatenate([
                    y_pred,
                    np.full(len(unused_txs), bg_cidx, dtype=int),
                ])
                y_true = np.concatenate([y_true, self.true_cxs[unused_txs]])
                y_score = np.concatenate([
                    y_score,
                    np.full(len(unused_txs), fn_score, dtype=float),
                ])
                y_weight = np.concatenate([y_weight, fn_weights])
                y_iou = np.concatenate([
                    y_iou,
                    np.full(len(unused_txs), -1.0, dtype=float),
                ])
                y_pxs = np.concatenate([
                    y_pxs,
                    np.full(len(unused_txs), -1, dtype=int),
                ])
                y_txs = np.concatenate([y_txs, unused_txs])
                y_rank = np.concatenate([
                    y_rank,
                    np.full(len(unused_txs), -1, dtype=int),
                ])

            results[threshold] = {
                'pred': y_pred.tolist(),
                'true': y_true.tolist(),
                'score': y_score.tolist(),
                'weight': y_weight.tolist(),
                'iou': y_iou.tolist(),
                'txs': y_txs.tolist(),
                'pxs': y_pxs.tolist(),
                'rank': y_rank.tolist(),
            }
        return results

    def _assign_many_kwimage_ext(
        self,
        thresholds,
        rust_func,
        *,
        bg_weight,
        bg_cidx,
        ignore_classes,
        max_dets,
        max_dets_scope,
        area_range,
        truth_reuse_policy,
    ):
        """Run threshold-batched matching through kwimage_ext Rust."""
        if truth_reuse_policy != 'never':
            raise NotImplementedError(
                'Rust assignment currently implements truth_reuse_policy="never" only'
            )
        if ignore_classes is not None:
            raise NotImplementedError(
                'Rust batched assignment does not yet implement threshold-dependent '
                'kwcoco ignore-class regions; use assignment_backend="python"'
            )
        pred_order = self._pred_order_for(max_dets, max_dets_scope)
        true_area_ignore_flags, pred_area_ignore_flags = self._area_ignore_flags(
            area_range
        )
        true_ignore_flags = np.zeros(len(self.true_dets), dtype=bool)  # type: ignore
        annotation_ignore_flags = (
            self.true_annotation_ignore_flags | true_area_ignore_flags
        )
        offsets, flat_txs, flat_ious = self._candidate_csr()
        match_txs, match_ious, true_nmatches = rust_func(
            offsets,
            flat_txs,
            flat_ious,
            pred_order,
            np.asarray(thresholds, dtype=np.float64),
            true_ignore_flags,
            annotation_ignore_flags,
            self.true_crowd_flags,
            inclusive=self.threshold_policy == 'inclusive',
        )
        return self._materialize_match_matrices(
            thresholds,
            pred_order,
            match_txs,
            match_ious,
            true_nmatches,
            bg_weight=bg_weight,
            bg_cidx=bg_cidx,
            true_ignore_flags=true_ignore_flags,
            annotation_ignore_flags=annotation_ignore_flags,
            pred_area_ignore_flags=pred_area_ignore_flags,
        )

    def assign_many(
        self,
        iou_thresh: Iterable[float],
        *,
        assignment_backend: str = 'auto',
        **kwargs: Any,
    ) -> dict[float, dict[str, list[int | float]]]:
        """Run repeated threshold queries from the same candidate index.

        ``assignment_backend='auto'`` uses the optional kwimage_ext Rust
        threshold-batched kernel when available and semantics permit it.
        ``ignore_classes`` currently requires the Python reference path because
        kwcoco ignore-region membership itself depends on the IoU threshold.
        """
        thresholds = [float(t) for t in iou_thresh]
        for threshold in thresholds:
            if threshold < self.min_iou:
                raise ValueError(
                    f'iou_thresh={threshold!r} is below cached min_iou={self.min_iou!r}'
                )
        requested_backend = assignment_backend
        key, func, _metadata = resolve_assignment_backend(assignment_backend)
        ignore_classes = kwargs.get('ignore_classes', 'ignore')
        if key == 'kwimage_ext' and ignore_classes is not None:
            if requested_backend == 'kwimage_ext':
                raise NotImplementedError(
                    'assignment_backend="kwimage_ext" does not yet support '
                    'threshold-dependent ignore_classes'
                )
            key = 'python'
            func = None
        if key == 'python':
            return {
                threshold: self.assign(threshold, **kwargs)
                for threshold in thresholds
            }
        assert func is not None
        rust_kwargs = {
            'bg_weight': kwargs.pop('bg_weight', 1.0),
            'bg_cidx': kwargs.pop('bg_cidx', -1),
            'ignore_classes': kwargs.pop('ignore_classes', 'ignore'),
            'max_dets': kwargs.pop('max_dets', None),
            'max_dets_scope': kwargs.pop('max_dets_scope', 'image'),
            'area_range': kwargs.pop('area_range', None),
            'truth_reuse_policy': kwargs.pop('truth_reuse_policy', 'never'),
        }
        if kwargs:
            raise TypeError(
                'Unexpected assignment kwargs for Rust backend: '
                f'{sorted(kwargs)!r}'
            )
        return self._assign_many_kwimage_ext(
            thresholds,
            func,
            **rust_kwargs,
        )



def _populate_sparse_coco_bbox_candidates(
    pred_ltrb: np.ndarray,
    true_ltrb: np.ndarray,
    true_crowd_flags: np.ndarray,
    pred_areas: np.ndarray,
    true_areas: np.ndarray,
    pred_idxs: np.ndarray,
    true_idxs: np.ndarray,
    *,
    min_iou: float,
    threshold_policy: str,
    cand_txs: list[np.ndarray],
    cand_ious: list[np.ndarray],
    build_stats: dict[str, int],
) -> bool:
    """Populate a sparse COCO bbox candidate block without changing overlap.

    The spatial index is only a geometric prefilter for positive-overlap box
    pairs.  The retained value is still computed by ``_coco_bbox_overlaps`` so
    crowd truths use intersection / detection-area exactly as the dense path.
    """
    true_local_ltrb = true_ltrb[true_idxs]
    index = BoxSpatialIndex.build(true_local_ltrb)
    if index is None:
        return False
    # A successfully built index is sufficient to make the sparse result
    # authoritative, even when every prediction is disjoint from every truth.
    # Falling back to a dense block in that case would defeat the optimization.
    used_any = True
    build_stats['spatial_blocks'] += 1
    for px in pred_idxs:
        # COCO bbox geometry has no pixel-coordinate bias.
        local_txs = index.query(pred_ltrb[int(px)], bias=0.0)
        if len(local_txs) == 0:
            continue
        global_txs = true_idxs[local_txs]
        build_stats['sparse_pairs_considered'] += int(len(global_txs))
        row = _coco_bbox_overlaps(
            pred_ltrb[[int(px)]],
            true_ltrb[global_txs],
            true_crowd_flags[global_txs],
            pred_areas=pred_areas[[int(px)]],
            true_areas=true_areas[global_txs],
        )[0]
        keep = _threshold_mask(row, min_iou, threshold_policy)
        if np.any(keep):
            txs = global_txs[keep]
            iou_vals = np.asarray(row[keep], dtype=float)
            order = _candidate_order(txs, iou_vals, coco_tie_break=True)
            cand_txs[int(px)] = txs[order]
            cand_ious[int(px)] = iou_vals[order]
            build_stats['pairs_kept'] += int(len(order))
        used_any = True
    return used_any

def _populate_sparse_candidates(
    pred_dets,
    true_dets,
    pred_ltrb: np.ndarray,
    true_ltrb: np.ndarray,
    pred_idxs: np.ndarray,
    true_idxs: np.ndarray,
    *,
    bias: float,
    min_iou: float,
    cand_txs: list[np.ndarray],
    cand_ious: list[np.ndarray],
    build_stats: dict[str, int],
) -> bool:
    true_local_ltrb = true_ltrb[true_idxs]
    index = BoxSpatialIndex.build(true_local_ltrb)
    if index is None:
        return False
    used_any = False
    build_stats['spatial_blocks'] += 1
    for px in pred_idxs:
        local_txs = index.query(pred_ltrb[int(px)], bias=bias)
        if len(local_txs) == 0:
            continue
        global_txs = true_idxs[local_txs]
        build_stats['sparse_pairs_considered'] += int(len(global_txs))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='invalid .* true_divide')
            row = pred_dets.boxes[[int(px)]].ious(true_dets.boxes[global_txs], bias=bias)[0]
        keep = row > min_iou
        if np.any(keep):
            txs = global_txs[keep]
            iou_vals = np.asarray(row[keep], dtype=float)
            order = np.argsort(-iou_vals, kind='mergesort')
            cand_txs[int(px)] = txs[order]
            cand_ious[int(px)] = iou_vals[order]
            build_stats['pairs_kept'] += int(len(order))
        used_any = True
    return used_any


@overload
def assign_confusion_vectors_v2(
    true_dets: 'kwimage.Detections',
    pred_dets: 'kwimage.Detections',
    *,
    iou_thresh: float = 0.5,
    bg_weight: float = 1.0,
    bg_cidx: int = -1,
    bias: float = 0.0,
    classes: 'CategoryTree | None' = None,
    compat: str = 'all',
    prioritize: str = 'iou',
    ignore_classes: str | Iterable[str] | None = 'ignore',
    max_dets: int | None = None,
    max_dets_scope: str = 'image',
    area_range: tuple[float, float] | None = None,
    truth_reuse_policy: str = 'never',
    min_iou: float | None = None,
    spatial_threshold: int | None = 50_000,
    iou_type: str = 'bbox',
    bbox_overlap_mode: str = 'legacy',
    overlap_backend: str = 'auto',
    segmentation_rasterizer: str = 'auto',
    raster_backend: str = 'auto',
    threshold_policy: str = 'auto',
    annotation_ignore_policy: str = 'coco',
    image_dims: tuple[int, int] | None = None,
) -> dict[str, list[int | float]]: ...


@overload
def assign_confusion_vectors_v2(
    true_dets: 'kwimage.Detections',
    pred_dets: 'kwimage.Detections',
    *,
    iou_thresh: Iterable[float],
    bg_weight: float = 1.0,
    bg_cidx: int = -1,
    bias: float = 0.0,
    classes: 'CategoryTree | None' = None,
    compat: str = 'all',
    prioritize: str = 'iou',
    ignore_classes: str | Iterable[str] | None = 'ignore',
    max_dets: int | None = None,
    max_dets_scope: str = 'image',
    area_range: tuple[float, float] | None = None,
    truth_reuse_policy: str = 'never',
    min_iou: float | None = None,
    spatial_threshold: int | None = 50_000,
    iou_type: str = 'bbox',
    bbox_overlap_mode: str = 'legacy',
    overlap_backend: str = 'auto',
    segmentation_rasterizer: str = 'auto',
    raster_backend: str = 'auto',
    threshold_policy: str = 'auto',
    annotation_ignore_policy: str = 'coco',
    image_dims: tuple[int, int] | None = None,
) -> dict[float, dict[str, list[int | float]]]: ...


def assign_confusion_vectors_v2(
    true_dets: 'kwimage.Detections',
    pred_dets: 'kwimage.Detections',
    *,
    iou_thresh: float | Iterable[float] = 0.5,
    bg_weight: float = 1.0,
    bg_cidx: int = -1,
    bias: float = 0.0,
    classes: 'CategoryTree | None' = None,
    compat: str = 'all',
    prioritize: str = 'iou',
    ignore_classes: str | Iterable[str] | None = 'ignore',
    max_dets: int | None = None,
    max_dets_scope: str = 'image',
    area_range: tuple[float, float] | None = None,
    truth_reuse_policy: str = 'never',
    min_iou: float | None = None,
    spatial_threshold: int | None = 50_000,
    iou_type: str = 'bbox',
    bbox_overlap_mode: str = 'legacy',
    overlap_backend: str = 'auto',
    segmentation_rasterizer: str = 'auto',
    raster_backend: str = 'auto',
    threshold_policy: str = 'auto',
    annotation_ignore_policy: str = 'coco',
    image_dims: tuple[int, int] | None = None,
) -> dict[str, list[int | float]] | dict[float, dict[str, list[int | float]]]:
    """
    Convenience wrapper around :class:`AssignmentCandidateIndex`.

    This accepts a similar high-level input shape to the original assignment
    function, but it does not promise row-for-row equivalence.  See the module
    documentation for the known semantic and feature differences.
    """
    if isinstance(iou_thresh, Iterable):
        iou_thresh_list = [float(t) for t in iou_thresh]
        multi_iou = True
    else:
        iou_thresh_list = [float(iou_thresh)]
        multi_iou = False
    if min_iou is None:
        min_iou = min(iou_thresh_list) if iou_thresh_list else 0.0
    index = AssignmentCandidateIndex.build(
        true_dets,
        pred_dets,
        classes=classes,
        compat=compat,
        prioritize=prioritize,
        bias=bias,
        min_iou=float(min_iou),
        spatial_threshold=spatial_threshold,
        iou_type=iou_type,
        bbox_overlap_mode=bbox_overlap_mode,
        overlap_backend=overlap_backend,
        segmentation_rasterizer=segmentation_rasterizer,
        raster_backend=raster_backend,
        threshold_policy=threshold_policy,
        annotation_ignore_policy=annotation_ignore_policy,
        image_dims=image_dims,
    )
    results = index.assign_many(
        iou_thresh_list,
        bg_weight=bg_weight,
        bg_cidx=bg_cidx,
        ignore_classes=ignore_classes,
        max_dets=max_dets,
        max_dets_scope=max_dets_scope,
        area_range=area_range,
        truth_reuse_policy=truth_reuse_policy,
    )
    if multi_iou:
        return results
    else:
        return results[iou_thresh_list[0]]
