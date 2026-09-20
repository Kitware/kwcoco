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
    return np.asarray(boxes.to_ltrb().data, dtype=float)


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
    pred_scores: np.ndarray
    pred_cxs: np.ndarray
    true_cxs: np.ndarray
    cx_to_matchable_txs: dict[int, np.ndarray]
    cand_txs: list[np.ndarray]
    cand_ious: list[np.ndarray]
    iou_type: str
    overlap_backend: dict[str, Any] | None
    segmentation_rasterizer: str
    raster_backend: dict[str, Any] | None
    threshold_policy: str
    annotation_ignore_policy: str
    true_crowd_flags: np.ndarray
    true_annotation_ignore_flags: np.ndarray
    pred_rles: list[dict[str, Any]] | None
    true_rles: list[dict[str, Any]] | None
    image_dims: tuple[int, int] | None
    build_stats: dict[str, Any]

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
            overlap_backend: mask-IoU backend for ``iou_type='segm'``.
            segmentation_rasterizer: ``'kwimage'`` or COCO-compatible
                ``'coco'`` polygon rasterization.
            raster_backend: backend for COCO-compatible polygon rasterization.
            threshold_policy: ``'strict'`` uses ``IoU > threshold`` (legacy
                kwcoco semantics); ``'inclusive'`` uses ``IoU >= threshold``
                to match COCOeval's threshold boundary. ``'auto'`` selects
                strict bbox matching and inclusive segmentation matching.
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
        if threshold_policy == 'auto':
            threshold_policy = 'inclusive' if iou_type == 'segm' else 'strict'
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
            image_dims = tuple(map(int, image_dims))
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
        raw_crowd_flags = np.asarray(
            true_dets.data.get('iscrowd', np.zeros(len(true_dets), dtype=bool)),
            dtype=bool,
        )
        raw_ignore_flags = np.asarray(
            true_dets.data.get('ignore', np.zeros(len(true_dets), dtype=bool)),
            dtype=bool,
        )
        if annotation_ignore_policy == 'none' or iou_type != 'segm':
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
            from kwcoco.metrics.overlap import detections_to_rles
            from kwcoco.metrics.overlap import resolve_mask_iou_backend

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
            mask_iou_func, overlap_metadata = resolve_mask_iou_backend(
                overlap_backend
            )
            build_stats['overlap_backend'] = overlap_metadata
            build_stats['rasterization'] = pred_raster_meta

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
                        order = np.argsort(-iou_vals, kind='mergesort')
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
            ious = pred_dets.boxes[pred_idxs].ious(true_dets.boxes[true_idxs], bias=bias)
            for px, row in zip(pred_idxs, ious):
                keep = _threshold_mask(row, min_iou, threshold_policy)
                if np.any(keep):
                    txs = true_idxs[keep]
                    iou_vals = np.asarray(row[keep], dtype=float)
                    order = np.argsort(-iou_vals, kind='mergesort')
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
            pred_scores=pred_scores,
            pred_cxs=pred_cxs,
            true_cxs=true_cxs,
            cx_to_matchable_txs=cx_to_matchable_txs,
            cand_txs=cand_txs,
            cand_ious=cand_ious,
            iou_type=iou_type,
            overlap_backend=overlap_metadata,
            segmentation_rasterizer=segmentation_rasterizer,
            raster_backend=build_stats.get('rasterization', {}).get('raster_backend'),
            threshold_policy=threshold_policy,
            annotation_ignore_policy=annotation_ignore_policy,
            true_crowd_flags=true_crowd_flags,
            true_annotation_ignore_flags=true_annotation_ignore_flags,
            pred_rles=pred_rles,
            true_rles=true_rles,
            image_dims=image_dims,
            build_stats=build_stats,
        )

    def assign(
        self,
        iou_thresh: float,
        *,
        bg_weight: float = 1.0,
        bg_cidx: int = -1,
        ignore_classes: str | Iterable[str] | None = 'ignore',
        max_dets: int | None = None,
        truth_reuse_policy: str = 'never',
    ) -> dict[str, list[int | float]]:
        """Run greedy assignment for one IoU threshold."""
        if truth_reuse_policy != 'never':
            raise NotImplementedError(
                'assignment_v2 currently implements truth_reuse_policy="never" only'
            )
        if iou_thresh < self.min_iou:
            raise ValueError(
                f'iou_thresh={iou_thresh!r} is below cached min_iou={self.min_iou!r}'
            )

        true_available = np.ones(len(self.true_dets), dtype=bool)  # type: ignore
        true_nmatches = np.zeros(len(self.true_dets), dtype=int)  # type: ignore
        pred_order = self.pred_order
        if max_dets is not None and np.isfinite(max_dets):
            pred_order = pred_order[: int(max_dets)]

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

        annotation_ignore_flags = self.true_annotation_ignore_flags
        crowd_flags = self.true_crowd_flags

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
            pred_cx = int(self.pred_cxs[px])
            score = float(self.pred_scores[px])
            txs = self.cand_txs[px]
            ious = self.cand_ious[px]
            chosen_tx = -1
            chosen_iou = -1.0
            if len(txs):
                threshold_valid = _threshold_mask(
                    ious, iou_thresh, self.threshold_policy
                )
                reusable = crowd_flags[txs]
                available = (true_available[txs] | reusable) & ~true_ignore_flags[txs]
                valid = threshold_valid & available
                if np.any(valid):
                    # COCOeval sorts ignored truths after regular truths.  The
                    # candidate arrays here are sorted by descending overlap,
                    # so explicitly prefer the best non-ignored candidate and
                    # only fall back to an ignored/crowd truth when necessary.
                    normal_valid = valid & ~annotation_ignore_flags[txs]
                    choose_from = normal_valid if np.any(normal_valid) else valid
                    chosen_pos = int(np.where(choose_from)[0][0])
                    chosen_tx = int(txs[chosen_pos])
                    chosen_iou = float(ious[chosen_pos])

            if chosen_tx >= 0:
                if not crowd_flags[chosen_tx]:
                    true_available[chosen_tx] = False
                true_nmatches[chosen_tx] += 1
                true_cx = int(self.true_cxs[chosen_tx])
                if annotation_ignore_flags[chosen_tx]:
                    weight = 0.0
                elif 'weights' in self.true_dets.data:  # type: ignore
                    weight = float(self.true_dets.weights[chosen_tx])  # type: ignore
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
                y_rank.append(int(self.pred_rank[px]))
            else:
                y_pred.append(pred_cx)
                y_true.append(bg_cidx)
                y_score.append(score)
                y_weight.append(float(bg_weight))
                y_iou.append(-1.0)
                y_pxs.append(px)
                y_txs.append(-1)
                y_rank.append(int(self.pred_rank[px]))

        unused_flags = true_nmatches == 0
        unused_flags = unused_flags & np.logical_not(annotation_ignore_flags)
        if ignore_classes is not None:
            unused_flags = unused_flags & np.logical_not(true_ignore_flags)
        unused_txs = np.where(unused_flags)[0]
        if 'weights' in self.true_dets.data:  # type: ignore
            unused_y_weight = self.true_dets.weights[unused_txs].tolist()  # type: ignore
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
        if not ub.iterable(ignore_classes):
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

    def assign_many(
        self,
        iou_thresh: Iterable[float],
        **kwargs: Any,
    ) -> dict[float, dict[str, list[int | float]]]:
        """Run repeated threshold queries from the same candidate index."""
        return {float(t): self.assign(float(t), **kwargs) for t in iou_thresh}


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
    truth_reuse_policy: str = 'never',
    min_iou: float | None = None,
    spatial_threshold: int | None = 50_000,
    iou_type: str = 'bbox',
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
    truth_reuse_policy: str = 'never',
    min_iou: float | None = None,
    spatial_threshold: int | None = 50_000,
    iou_type: str = 'bbox',
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
    truth_reuse_policy: str = 'never',
    min_iou: float | None = None,
    spatial_threshold: int | None = 50_000,
    iou_type: str = 'bbox',
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
        truth_reuse_policy=truth_reuse_policy,
    )
    if multi_iou:
        return results
    else:
        return results[iou_thresh_list[0]]
