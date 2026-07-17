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
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import math
import warnings

import numpy as np
import ubelt as ub

from kwcoco.metrics.assignment import ASSIGN_COLUMNS
from kwcoco.metrics.assignment import USE_NEG_INF
from kwcoco.metrics.assignment import _filter_ignore_regions


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

    def query(self, box_ltrb: np.ndarray) -> np.ndarray:
        """Return candidate true-box indexes whose grid cells overlap a box."""
        cells = _box_cells(box_ltrb, self.cell_size)
        if cells is None or len(cells) > self.max_cells_per_box:
            return np.arange(len(self.boxes_ltrb), dtype=int)
        cand = set()
        for cell in cells:
            cand.update(self.cell_to_idxs.get(cell, []))
        if not cand:
            return np.empty(0, dtype=int)
        cand_idxs = np.array(sorted(cand), dtype=int)
        cand_boxes = self.boxes_ltrb[cand_idxs]
        overlap = _ltrb_intersects(box_ltrb[None, :], cand_boxes)[0]
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


def _ltrb_intersects(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return a dense boolean matrix indicating box intersection."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) == 0 or len(right) == 0:
        return np.empty((len(left), len(right)), dtype=bool)
    return (
        (left[:, None, 0] < right[None, :, 2])
        & (left[:, None, 2] > right[None, :, 0])
        & (left[:, None, 1] < right[None, :, 3])
        & (left[:, None, 3] > right[None, :, 1])
    )


@dataclass
class AssignmentCandidateIndex:
    """
    Reusable per-image candidate graph for detection assignment.

    The index stores candidate true detections for each prediction, their IoUs,
    and a score rank for each prediction.  Building this once allows repeated
    greedy assignment queries for different IoU thresholds without recomputing
    pairwise overlaps or compatible-class groups.

    The candidate graph is deliberately sparse: only candidate pairs with
    ``iou >= min_iou`` are retained.  For COCO-style AP50:95, ``min_iou=0.5``
    is sufficient for all threshold queries.
    """

    true_dets: object
    pred_dets: object
    classes: object | None
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
    build_stats: dict[str, int]

    @classmethod
    def build(
        cls,
        true_dets,
        pred_dets,
        *,
        classes=None,
        compat: str = 'mutex',
        prioritize: str = 'iou',
        bias: float = 0.0,
        min_iou: float = 0.5,
        spatial_threshold: int | None = 50_000,
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
            spatial_threshold: if a compatible block has more than this many
                dense pairs, try the sparse grid candidate generator.  Use
                ``None`` to always use dense compatible blocks.
        """
        if prioritize != 'iou':
            raise NotImplementedError(
                'assignment_v2 currently implements prioritize="iou" only'
            )
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
        pred_rank = np.full(len(pred_dets), fill_value=-1, dtype=int)
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

        cand_txs = [np.empty(0, dtype=int) for _ in range(len(pred_dets))]
        cand_ious = [np.empty(0, dtype=float) for _ in range(len(pred_dets))]
        build_stats = {
            'dense_blocks': 0,
            'spatial_blocks': 0,
            'dense_pairs_considered': 0,
            'sparse_pairs_considered': 0,
            'pairs_kept': 0,
        }

        pred_ltrb = _boxes_to_ltrb(pred_dets.boxes)
        true_ltrb = _boxes_to_ltrb(true_dets.boxes)

        for pred_cx in unique_pcxs:
            pred_idxs = np.where(pred_cxs == pred_cx)[0].astype(int)
            true_idxs = cx_to_matchable_txs.get(int(pred_cx), np.empty(0, dtype=int))
            if len(pred_idxs) == 0 or len(true_idxs) == 0:
                continue
            dense_pair_count = int(len(pred_idxs) * len(true_idxs))
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
                keep = row >= min_iou
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
            build_stats=build_stats,
        )

    def assign(
        self,
        iou_thresh: float,
        *,
        bg_weight: float = 1.0,
        bg_cidx: int = -1,
        ignore_classes='ignore',
        max_dets: int | None = None,
        truth_reuse_policy: str = 'never',
    ) -> dict[str, list]:
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
            true_ignore_flags, pred_ignore_flags = _filter_ignore_regions(
                self.true_dets,
                self.pred_dets,
                ioaa_thresh=iou_thresh,
                ignore_classes=ignore_classes,
            )
            true_available[true_ignore_flags] = False
            pred_order = pred_order[~pred_ignore_flags[pred_order]]

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
                valid = (ious >= iou_thresh) & true_available[txs]
                if np.any(valid):
                    # Candidate arrays are already sorted by descending IoU.
                    chosen_pos = int(np.where(valid)[0][0])
                    chosen_tx = int(txs[chosen_pos])
                    chosen_iou = float(ious[chosen_pos])

            if chosen_tx >= 0:
                true_available[chosen_tx] = False
                true_nmatches[chosen_tx] += 1
                true_cx = int(self.true_cxs[chosen_tx])
                if 'weights' in self.true_dets.data:  # type: ignore
                    weight = float(self.true_dets.weights[chosen_tx])  # type: ignore
                else:
                    weight = 1.0
                y_pred.append(pred_cx)
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
        if ignore_classes is not None:
            unused_flags = unused_flags & ~true_ignore_flags
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

    def assign_many(
        self,
        iou_thresh: list[float] | np.ndarray,
        **kwargs,
    ) -> dict[float, dict[str, list]]:
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
        local_txs = index.query(pred_ltrb[int(px)])
        if len(local_txs) == 0:
            continue
        global_txs = true_idxs[local_txs]
        build_stats['sparse_pairs_considered'] += int(len(global_txs))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='invalid .* true_divide')
            row = pred_dets.boxes[[int(px)]].ious(true_dets.boxes[global_txs], bias=bias)[0]
        keep = row >= min_iou
        if np.any(keep):
            txs = global_txs[keep]
            iou_vals = np.asarray(row[keep], dtype=float)
            order = np.argsort(-iou_vals, kind='mergesort')
            cand_txs[int(px)] = txs[order]
            cand_ious[int(px)] = iou_vals[order]
            build_stats['pairs_kept'] += int(len(order))
        used_any = True
    return used_any


def assign_confusion_vectors_v2(
    true_dets,
    pred_dets,
    *,
    iou_thresh=0.5,
    bg_weight: float = 1.0,
    bg_cidx: int = -1,
    bias: float = 0.0,
    classes=None,
    compat: str = 'mutex',
    prioritize: str = 'iou',
    ignore_classes='ignore',
    max_dets: int | None = None,
    truth_reuse_policy: str = 'never',
    min_iou: float | None = None,
    spatial_threshold: int | None = 50_000,
):
    """
    Convenience wrapper around :class:`AssignmentCandidateIndex`.

    This mirrors the old assignment function at a high level, but exposes the
    candidate-index implementation through a standalone module.
    """
    iou_thresh_list = [iou_thresh] if not ub.iterable(iou_thresh) else list(iou_thresh)  # type: ignore
    if min_iou is None:
        min_iou = min(map(float, iou_thresh_list)) if iou_thresh_list else 0.0  # type: ignore
    index = AssignmentCandidateIndex.build(
        true_dets,
        pred_dets,
        classes=classes,
        compat=compat,
        prioritize=prioritize,
        bias=bias,
        min_iou=float(min_iou),  # type: ignore
        spatial_threshold=spatial_threshold,
    )
    results = index.assign_many(
        iou_thresh_list,  # type: ignore
        bg_weight=bg_weight,
        bg_cidx=bg_cidx,
        ignore_classes=ignore_classes,
        max_dets=max_dets,
        truth_reuse_policy=truth_reuse_policy,
    )
    if ub.iterable(iou_thresh):
        return results
    else:
        return results[float(iou_thresh_list[0])]  # type: ignore
