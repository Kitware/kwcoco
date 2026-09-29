"""
COCO-compatible metric helpers.

This module contains two deliberately separate pieces of infrastructure:

* the strict ``pycocotools`` bridge used for the official COCO reference path;
* the small native COCO accumulator used by ``DetectionMetrics.score_native_coco``.

The official bridge remains the reference implementation.  The native path
exists so reusable kwcoco assignment rows can be accumulated with the same
stable score ordering, precision envelope, and 101 recall samples without
routing through generic ``ConfusionVectors -> Measures`` behavior.

KWCoco permits polygon interiors (holes), whereas the original MS-COCO polygon
representation does not.  At the strict reference boundary ordinary polygons
remain polygons and representations that cannot be expressed losslessly are
encoded as exact RLE masks.  See ``dev/pycocotools_metrics_compare`` for the
comparison suite that keeps overlap, assignment, and accumulation differences
independently observable.
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


NATIVE_COCO_IMPLEMENTATION = 'kwcoco.native_coco'
OFFICIAL_COCO_IMPLEMENTATION = 'pycocotools.COCOeval'
COCO_DETECTION_RESULT_SCHEMA = 'coco_detection_summary_v1'
COCO_EVAL_TENSOR_SCHEMA = 'coco_detection_eval_tensor_v1'


def accumulate_coco_assignment_rows(
    rows,
    classes,
    *,
    category_idxs=None,
    ignore_classes=None,
    recall_thresholds=COCO_RECALL_THRESHOLDS,
    with_details=False,
):
    """Accumulate one COCO assignment slice from explicit kwcoco rows.

    Args:
        rows (Mapping[str, Sequence]):
            Assignment rows containing at least ``true``, ``pred``, ``score``,
            and ``weight``.  A zero row weight means the row is ignored by the
            COCO slice; positive weights are treated as one event.  Native COCO
            scoring normalizes kwcoco truth weights before assignment, so this
            binary interpretation is only carrying COCO ignore state.

        classes:
            Category sequence / ``CategoryTree`` used by the assignment rows.

        category_idxs (Sequence[int] | None):
            Optional subset of class indexes to expose on the COCO category
            axis. Assignment rows remain in the original class-index space;
            the returned tensors are compacted to this ordered subset. This
            is used by :class:`CocoEvaluator` to exclude its historical
            synthetic background bookkeeping class while preserving a real
            COCO category literally named ``background`` when one is declared.

        ignore_classes (Iterable[str] | str | None):
            Optional kwcoco extension: classes excluded from native summary
            aggregation.  Matching ignore-region behavior is handled earlier
            by assignment.

        recall_thresholds (Sequence[float]):
            Recall samples.  The standard COCO setting has 101 values from
            zero through one inclusive.

        with_details (bool):
            Include per-class scalar diagnostics.

    Returns:
        dict:
            ``mAP`` / ``mAR`` and the number of classes that contain scored
            ground truth in this slice.

    Notes:
        COCOeval uses a stable descending mergesort for detection scores.  This
        matters for equal-score TP/FP rows.  The helper intentionally lives
        here instead of changing historical generic ``Measures`` tie behavior.
    """
    true = np.asarray(rows.get('true', []), dtype=np.int32)
    pred = np.asarray(rows.get('pred', []), dtype=np.int32)
    score = np.asarray(rows.get('score', []), dtype=float)
    weight = np.asarray(rows.get('weight', []), dtype=float)
    lengths = {len(true), len(pred), len(score), len(weight)}
    if len(lengths) != 1:
        raise ValueError(
            'COCO assignment columns must have equal length: '
            f'true={len(true)}, pred={len(pred)}, score={len(score)}, '
            f'weight={len(weight)}'
        )

    recall_thresholds = np.asarray(recall_thresholds, dtype=float)
    if ignore_classes is None:
        ignored_names = set()
    elif isinstance(ignore_classes, str):
        ignored_names = {ignore_classes.casefold()}
    else:
        ignored_names = {str(name).casefold() for name in ignore_classes}

    # Assignment rows use weight=0 for COCO ignore state (crowd, out-of-range,
    # explicit annotation ignore, or configured ignore regions).  COCO itself
    # has no fractional truth-weight semantics, so positive rows count once.
    scored = weight > 0

    if category_idxs is None:
        category_idxs = list(range(len(classes)))
    else:
        category_idxs = list(map(int, category_idxs))
        if len(set(category_idxs)) != len(category_idxs):
            raise ValueError('category_idxs must be unique')
        invalid = [cx for cx in category_idxs if cx < 0 or cx >= len(classes)]
        if invalid:
            raise ValueError(
                f'category_idxs contains out-of-range indexes: {invalid!r}'
            )

    num_classes = len(category_idxs)
    num_recalls = len(recall_thresholds)
    precision_tensor = -np.ones((num_recalls, num_classes), dtype=float)
    recall_tensor = -np.ones(num_classes, dtype=float)
    score_tensor = -np.ones((num_recalls, num_classes), dtype=float)
    realpos_tensor = np.zeros(num_classes, dtype=int)

    ap_values = []
    recall_values = []
    perclass = {}
    for out_cx, cx in enumerate(category_idxs):
        node = classes[cx]
        name = str(node)
        if name.casefold() in ignored_names:
            continue

        true_flags = true == cx
        realpos_total = int(np.count_nonzero(true_flags & scored))
        realpos_tensor[out_cx] = realpos_total
        if realpos_total <= 0:
            continue

        pred_flags = pred == cx
        cls_score = score[pred_flags]
        cls_scored = scored[pred_flags]
        cls_is_tp = true[pred_flags] == cx
        if len(cls_score) == 0:
            ap = 0.0
            recall = 0.0
            sampled = np.zeros(num_recalls, dtype=float)
            sampled_scores = np.zeros(num_recalls, dtype=float)
        else:
            # Sorting negated scores preserves original row order for ties.
            order = np.argsort(-cls_score, kind='mergesort')
            cls_scored = cls_scored[order]
            cls_is_tp = cls_is_tp[order]

            tp = np.cumsum(cls_scored & cls_is_tp, dtype=float)
            fp = np.cumsum(cls_scored & ~cls_is_tp, dtype=float)
            rc = tp / realpos_total
            pr = tp / (tp + fp + np.spacing(1))
            if len(pr) > 1:
                pr = np.maximum.accumulate(pr[::-1])[::-1]

            inds = np.searchsorted(rc, recall_thresholds, side='left')
            sampled = np.zeros(num_recalls, dtype=float)
            sampled_scores = np.zeros(num_recalls, dtype=float)
            valid = inds < len(pr)
            sampled[valid] = pr[inds[valid]]
            sampled_scores[valid] = cls_score[order][inds[valid]]
            ap = float(sampled.mean())
            recall = float(rc[-1])

        precision_tensor[:, out_cx] = sampled
        recall_tensor[out_cx] = recall
        score_tensor[:, out_cx] = sampled_scores
        ap_values.append(ap)
        recall_values.append(recall)
        if with_details:
            perclass[name] = {
                'class_idx': int(cx),
                'category_axis_idx': int(out_cx),
                'ap': ap,
                'recall': recall,
                'realpos_total': realpos_total,
            }

    if ap_values:
        mean_ap = float(np.mean(ap_values))
        mean_recall = float(np.mean(recall_values))
    else:
        mean_ap = -1.0
        mean_recall = -1.0
    info = {
        'mAP': mean_ap,
        'mAR': mean_recall,
        'num_valid_classes': len(ap_values),
        'precision': precision_tensor,
        'recall': recall_tensor,
        'scores': score_tensor,
        'realpos_total': realpos_tensor,
    }
    if with_details:
        info['perclass'] = perclass
    return info


def accumulate_coco_compact_matches(
    batches,
    classes,
    *,
    thresholds,
    max_dets,
    category_idxs=None,
    recall_thresholds=COCO_RECALL_THRESHOLDS,
):
    """Accumulate COCO tensors directly from compact per-image match arrays.

    This is the fast counterpart to :func:`accumulate_coco_assignment_rows`.
    It deliberately consumes the matcher state *before* confusion rows are
    materialized.  That avoids the expensive round-trip

    ``NumPy match arrays -> Python row lists -> NumPy accumulator arrays``

    on the normal COCO scoring path while keeping explicit rows available to
    callers that request diagnostics.

    Each batch is a mapping with these fields:

    ``pred_cxs``
        Prediction category indexes in stable score order.
    ``pred_scores``
        Scores aligned with ``pred_cxs``.
    ``pred_category_rank``
        Zero-based stable score rank within each prediction category.
    ``match_txs``
        ``[T, P]`` matched truth indexes, ``-1`` for unmatched predictions.
    ``true_cxs``
        Truth category indexes local to the image.
    ``annotation_ignore_flags``
        Truth ignore state for the current area slice.  This already includes
        COCO crowd and area-range ignore semantics.
    ``pred_area_ignore_flags``
        Prediction area-ignore state aligned with the prediction order.

    Args:
        batches (Sequence[Mapping]): Per-image compact match state.
        classes: Category sequence / ``CategoryTree``.
        thresholds (Sequence[float]): Matching thresholds represented by the
            first axis of every ``match_txs`` array.
        max_dets (Sequence[int]): Per-image/per-category score prefixes to
            accumulate from the largest-prefix matching result.
        category_idxs (Sequence[int] | None): Category-axis subset.
        recall_thresholds (Sequence[float]): COCO recall samples.

    Returns:
        dict: Tensor fragments with shapes ``[T,R,K,M]`` for precision/scores
        and ``[T,K,M]`` for recall/real-positive counts.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    recall_thresholds = np.asarray(recall_thresholds, dtype=float)
    max_dets = list(map(int, max_dets))
    if category_idxs is None:
        category_idxs = list(range(len(classes)))
    else:
        category_idxs = list(map(int, category_idxs))

    T = len(thresholds)
    R = len(recall_thresholds)
    K = len(category_idxs)
    M = len(max_dets)
    precision = -np.ones((T, R, K, M), dtype=float)
    recall = -np.ones((T, K, M), dtype=float)
    scores = -np.ones((T, R, K, M), dtype=float)
    realpos = np.zeros((T, K, M), dtype=int)

    if batches:
        pred_cxs_parts = []
        pred_scores_parts = []
        pred_rank_parts = []
        pred_area_ignore_parts = []
        matched_true_cx_parts = [[] for _ in range(T)]
        matched_ignore_parts = [[] for _ in range(T)]
        realpos_by_class = np.zeros(len(classes), dtype=np.int64)

        for batch in batches:
            pred_cxs = np.asarray(batch['pred_cxs'], dtype=np.int32)
            pred_scores = np.asarray(batch['pred_scores'], dtype=float)
            pred_rank = np.asarray(batch['pred_category_rank'], dtype=np.int32)
            pred_area_ignore = np.asarray(
                batch['pred_area_ignore_flags'], dtype=bool
            )
            match_txs = np.asarray(batch['match_txs'], dtype=np.int64)
            true_cxs = np.asarray(batch['true_cxs'], dtype=np.int32)
            annotation_ignore = np.asarray(
                batch['annotation_ignore_flags'], dtype=bool
            )
            if match_txs.shape != (T, len(pred_cxs)):
                raise ValueError(
                    'compact match shape mismatch: '
                    f'{match_txs.shape!r} != {(T, len(pred_cxs))!r}'
                )
            if not (
                len(pred_cxs)
                == len(pred_scores)
                == len(pred_rank)
                == len(pred_area_ignore)
            ):
                raise ValueError('compact prediction columns must align')
            if len(true_cxs) != len(annotation_ignore):
                raise ValueError('compact truth columns must align')

            eligible_txs = np.where(~annotation_ignore)[0]
            if len(eligible_txs):
                realpos_by_class += np.bincount(
                    true_cxs[eligible_txs], minlength=len(classes)
                )

            pred_cxs_parts.append(pred_cxs)
            pred_scores_parts.append(pred_scores)
            pred_rank_parts.append(pred_rank)
            pred_area_ignore_parts.append(pred_area_ignore)

            for ti in range(T):
                row_txs = match_txs[ti]
                matched = row_txs >= 0
                row_true_cx = np.full(len(row_txs), -1, dtype=np.int32)
                row_ignore = np.zeros(len(row_txs), dtype=bool)
                if np.any(matched):
                    matched_txs = row_txs[matched]
                    row_true_cx[matched] = true_cxs[matched_txs]
                    row_ignore[matched] = annotation_ignore[matched_txs]
                matched_true_cx_parts[ti].append(row_true_cx)
                matched_ignore_parts[ti].append(row_ignore)

        pred_cxs = (
            np.concatenate(pred_cxs_parts)
            if pred_cxs_parts else np.empty(0, dtype=np.int32)
        )
        pred_scores = (
            np.concatenate(pred_scores_parts)
            if pred_scores_parts else np.empty(0, dtype=float)
        )
        pred_rank = (
            np.concatenate(pred_rank_parts)
            if pred_rank_parts else np.empty(0, dtype=np.int32)
        )
        pred_area_ignore = (
            np.concatenate(pred_area_ignore_parts)
            if pred_area_ignore_parts else np.empty(0, dtype=bool)
        )
        matched_true_cx = np.stack([
            np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)
            for parts in matched_true_cx_parts
        ], axis=0)
        matched_ignore = np.stack([
            np.concatenate(parts) if parts else np.empty(0, dtype=bool)
            for parts in matched_ignore_parts
        ], axis=0)
    else:
        pred_cxs = np.empty(0, dtype=np.int32)
        pred_scores = np.empty(0, dtype=float)
        pred_rank = np.empty(0, dtype=np.int32)
        pred_area_ignore = np.empty(0, dtype=bool)
        matched_true_cx = np.empty((T, 0), dtype=np.int32)
        matched_ignore = np.empty((T, 0), dtype=bool)
        realpos_by_class = np.zeros(len(classes), dtype=np.int64)

    for mi, max_det in enumerate(max_dets):
        prefix = pred_rank < int(max_det)
        for out_cx, cx in enumerate(category_idxs):
            realpos_total = int(realpos_by_class[cx])
            realpos[:, out_cx, mi] = realpos_total
            if realpos_total <= 0:
                continue

            cls_flags = prefix & (pred_cxs == cx)
            cls_score = pred_scores[cls_flags]
            cls_area_ignore = pred_area_ignore[cls_flags]
            if len(cls_score):
                # COCO stable global descending score order.  Input batch order
                # is already normalized image order, so mergesort preserves the
                # required image/prediction tie order.
                order = np.argsort(-cls_score, kind='mergesort')
                sorted_score = cls_score[order]
            else:
                order = np.empty(0, dtype=int)
                sorted_score = cls_score

            for ti in range(T):
                if len(cls_score) == 0:
                    sampled = np.zeros(R, dtype=float)
                    sampled_scores = np.zeros(R, dtype=float)
                    final_recall = 0.0
                else:
                    cls_true_cx = matched_true_cx[ti, cls_flags]
                    cls_ignored = matched_ignore[ti, cls_flags]
                    matched = cls_true_cx >= 0
                    scored = ~cls_ignored
                    scored &= ~(~matched & cls_area_ignore)
                    is_tp = cls_true_cx == cx

                    scored = scored[order]
                    is_tp = is_tp[order]
                    tp = np.cumsum(scored & is_tp, dtype=float)
                    fp = np.cumsum(scored & ~is_tp, dtype=float)
                    rc = tp / realpos_total
                    pr = tp / (tp + fp + np.spacing(1))
                    if len(pr) > 1:
                        pr = np.maximum.accumulate(pr[::-1])[::-1]
                    inds = np.searchsorted(
                        rc, recall_thresholds, side='left'
                    )
                    sampled = np.zeros(R, dtype=float)
                    sampled_scores = np.zeros(R, dtype=float)
                    valid = inds < len(pr)
                    sampled[valid] = pr[inds[valid]]
                    sampled_scores[valid] = sorted_score[inds[valid]]
                    final_recall = float(rc[-1])

                precision[ti, :, out_cx, mi] = sampled
                recall[ti, out_cx, mi] = final_recall
                scores[ti, :, out_cx, mi] = sampled_scores

    return {
        'precision': precision,
        'recall': recall,
        'scores': scores,
        'realpos_total': realpos,
    }


def accumulate_coco_compact_grid(
    batches,
    classes,
    *,
    thresholds,
    max_dets,
    num_areas,
    category_idxs=None,
    recall_thresholds=COCO_RECALL_THRESHOLDS,
):
    """Accumulate a full area-grid COCO tensor from compact image matches.

    This is the production companion to
    :func:`accumulate_coco_compact_matches`.  The matcher has already shared
    candidate geometry across area ranges and IoU thresholds; this accumulator
    preserves that sharing instead of splitting the result back into separate
    area streams. Prediction columns are concatenated once and stable score
    orders are computed once per category / maxDet pair, then reused for every
    area range and threshold.

    Each batch contains ``match_txs`` with shape ``[A,T,P]``,
    ``annotation_ignore_flags`` with shape ``[A,Ntrue]``, and
    ``pred_area_ignore_flags`` with shape ``[A,P]`` in addition to the common
    compact prediction/truth columns documented by
    :func:`accumulate_coco_compact_matches`.

    Returns:
        dict: ``precision`` / ``scores`` with shape ``[T,R,K,A,M]`` and
        ``recall`` / ``realpos_total`` with shape ``[T,K,A,M]``.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    recall_thresholds = np.asarray(recall_thresholds, dtype=float)
    max_dets = list(map(int, max_dets))
    num_areas = int(num_areas)
    if num_areas < 0:
        raise ValueError('num_areas must be nonnegative')
    if category_idxs is None:
        category_idxs = list(range(len(classes)))
    else:
        category_idxs = list(map(int, category_idxs))

    T = len(thresholds)
    R = len(recall_thresholds)
    K = len(category_idxs)
    A = num_areas
    M = len(max_dets)
    precision = -np.ones((T, R, K, A, M), dtype=float)
    recall = -np.ones((T, K, A, M), dtype=float)
    scores = -np.ones((T, R, K, A, M), dtype=float)
    realpos = np.zeros((T, K, A, M), dtype=int)

    pred_cxs_parts = []
    pred_scores_parts = []
    pred_rank_parts = []
    normalized_batches = []
    for batch in batches:
        pred_cxs = np.asarray(batch['pred_cxs'], dtype=np.int32)
        pred_scores = np.asarray(batch['pred_scores'], dtype=float)
        pred_rank = np.asarray(batch['pred_category_rank'], dtype=np.int32)
        match_txs = np.asarray(batch['match_txs'], dtype=np.int64)
        true_cxs = np.asarray(batch['true_cxs'], dtype=np.int32)
        annotation_ignore = np.asarray(
            batch['annotation_ignore_flags'], dtype=bool
        )
        pred_area_ignore = np.asarray(
            batch['pred_area_ignore_flags'], dtype=bool
        )
        expected_match = (A, T, len(pred_cxs))
        expected_truth = (A, len(true_cxs))
        expected_pred_ignore = (A, len(pred_cxs))
        if match_txs.shape != expected_match:
            raise ValueError(
                'compact grid match shape mismatch: '
                f'{match_txs.shape!r} != {expected_match!r}'
            )
        if annotation_ignore.shape != expected_truth:
            raise ValueError(
                'compact grid truth-ignore shape mismatch: '
                f'{annotation_ignore.shape!r} != {expected_truth!r}'
            )
        if pred_area_ignore.shape != expected_pred_ignore:
            raise ValueError(
                'compact grid prediction-ignore shape mismatch: '
                f'{pred_area_ignore.shape!r} != {expected_pred_ignore!r}'
            )
        if not (len(pred_cxs) == len(pred_scores) == len(pred_rank)):
            raise ValueError('compact prediction columns must align')
        pred_cxs_parts.append(pred_cxs)
        pred_scores_parts.append(pred_scores)
        pred_rank_parts.append(pred_rank)
        normalized_batches.append({
            'match_txs': match_txs,
            'true_cxs': true_cxs,
            'annotation_ignore_flags': annotation_ignore,
            'pred_area_ignore_flags': pred_area_ignore,
        })

    pred_cxs = (
        np.concatenate(pred_cxs_parts)
        if pred_cxs_parts else np.empty(0, dtype=np.int32)
    )
    pred_scores = (
        np.concatenate(pred_scores_parts)
        if pred_scores_parts else np.empty(0, dtype=float)
    )
    pred_rank = (
        np.concatenate(pred_rank_parts)
        if pred_rank_parts else np.empty(0, dtype=np.int32)
    )

    # Stable global score orders are area/threshold invariant.  Cache the
    # selected global prediction indexes once, then reuse them throughout the
    # A x T tensor loops.
    score_order_cache = {}
    for mi, max_det in enumerate(max_dets):
        prefix = pred_rank < int(max_det)
        for out_cx, cx in enumerate(category_idxs):
            global_idxs = np.where(prefix & (pred_cxs == cx))[0]
            if len(global_idxs):
                local_order = np.argsort(
                    -pred_scores[global_idxs], kind='mergesort'
                )
                sorted_global_idxs = global_idxs[local_order]
                sorted_scores = pred_scores[sorted_global_idxs]
            else:
                sorted_global_idxs = np.empty(0, dtype=np.int64)
                sorted_scores = np.empty(0, dtype=float)
            score_order_cache[(out_cx, mi)] = (
                sorted_global_idxs,
                sorted_scores,
            )

    for ai in range(A):
        realpos_by_class = np.zeros(len(classes), dtype=np.int64)
        pred_area_parts = []
        matched_true_parts = [[] for _ in range(T)]
        matched_ignore_parts = [[] for _ in range(T)]

        for batch in normalized_batches:
            match_txs = batch['match_txs'][ai]
            true_cxs = batch['true_cxs']
            annotation_ignore = batch['annotation_ignore_flags'][ai]
            pred_area_parts.append(batch['pred_area_ignore_flags'][ai])

            eligible_txs = np.where(~annotation_ignore)[0]
            if len(eligible_txs):
                realpos_by_class += np.bincount(
                    true_cxs[eligible_txs], minlength=len(classes)
                )

            for ti in range(T):
                row_txs = match_txs[ti]
                matched = row_txs >= 0
                row_true_cx = np.full(len(row_txs), -1, dtype=np.int32)
                row_ignore = np.zeros(len(row_txs), dtype=bool)
                if np.any(matched):
                    matched_txs = row_txs[matched]
                    row_true_cx[matched] = true_cxs[matched_txs]
                    row_ignore[matched] = annotation_ignore[matched_txs]
                matched_true_parts[ti].append(row_true_cx)
                matched_ignore_parts[ti].append(row_ignore)

        pred_area_ignore = (
            np.concatenate(pred_area_parts)
            if pred_area_parts else np.empty(0, dtype=bool)
        )
        matched_true_cx = np.stack([
            np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)
            for parts in matched_true_parts
        ], axis=0)
        matched_ignore = np.stack([
            np.concatenate(parts) if parts else np.empty(0, dtype=bool)
            for parts in matched_ignore_parts
        ], axis=0)

        for mi, _max_det in enumerate(max_dets):
            for out_cx, cx in enumerate(category_idxs):
                realpos_total = int(realpos_by_class[cx])
                realpos[:, out_cx, ai, mi] = realpos_total
                if realpos_total <= 0:
                    continue

                sorted_global_idxs, sorted_score = score_order_cache[
                    (out_cx, mi)
                ]
                if len(sorted_global_idxs):
                    cls_area_ignore = pred_area_ignore[sorted_global_idxs]
                else:
                    cls_area_ignore = np.empty(0, dtype=bool)

                for ti in range(T):
                    if len(sorted_global_idxs) == 0:
                        sampled = np.zeros(R, dtype=float)
                        sampled_scores = np.zeros(R, dtype=float)
                        final_recall = 0.0
                    else:
                        cls_true_cx = matched_true_cx[
                            ti, sorted_global_idxs
                        ]
                        cls_ignored = matched_ignore[
                            ti, sorted_global_idxs
                        ]
                        matched = cls_true_cx >= 0
                        scored = ~cls_ignored
                        scored &= ~(~matched & cls_area_ignore)
                        is_tp = cls_true_cx == cx

                        tp = np.cumsum(scored & is_tp, dtype=float)
                        fp = np.cumsum(scored & ~is_tp, dtype=float)
                        rc = tp / realpos_total
                        pr = tp / (tp + fp + np.spacing(1))
                        if len(pr) > 1:
                            pr = np.maximum.accumulate(pr[::-1])[::-1]
                        inds = np.searchsorted(
                            rc, recall_thresholds, side='left'
                        )
                        sampled = np.zeros(R, dtype=float)
                        sampled_scores = np.zeros(R, dtype=float)
                        valid = inds < len(pr)
                        sampled[valid] = pr[inds[valid]]
                        sampled_scores[valid] = sorted_score[inds[valid]]
                        final_recall = float(rc[-1])

                    precision[ti, :, out_cx, ai, mi] = sampled
                    recall[ti, out_cx, ai, mi] = final_recall
                    scores[ti, :, out_cx, ai, mi] = sampled_scores

    return {
        'precision': precision,
        'recall': recall,
        'scores': scores,
        'realpos_total': realpos,
    }

def snapshot_pycocotools_eval(evaler) -> dict[str, Any]:
    """Copy the accumulated COCOeval tensors into a stable kwcoco record.

    The arrays mirror ``COCOeval.eval`` exactly:

    * precision: ``[T, R, K, A, M]``
    * recall: ``[T, K, A, M]``
    * scores: ``[T, R, K, A, M]``

    Keeping this representation explicit gives the native implementation a
    stronger differential target than the 12 scalar summary values alone.
    """
    eval_info = evaler.eval
    params = evaler.params
    return {
        'schema': COCO_EVAL_TENSOR_SCHEMA,
        'counts': list(map(int, eval_info['counts'])),
        'precision': np.asarray(eval_info['precision'], dtype=float).copy(),
        'recall': np.asarray(eval_info['recall'], dtype=float).copy(),
        'scores': np.asarray(eval_info['scores'], dtype=float).copy(),
        'iou_thresholds': np.asarray(params.iouThrs, dtype=float).tolist(),
        'recall_thresholds': np.asarray(params.recThrs, dtype=float).tolist(),
        'category_ids': list(map(int, params.catIds)),
        'image_ids': list(map(int, params.imgIds)),
        'area_ranges': [list(map(float, r)) for r in params.areaRng],
        'area_range_labels': list(params.areaRngLbl),
        'max_dets': list(map(int, params.maxDets)),
    }


def summarize_coco_eval_tensor(eval_info: dict[str, Any]) -> dict[str, float]:
    """Compute the standard 12 detection statistics from a COCO eval tensor.

    This mirrors the selection semantics of ``COCOeval.summarize`` while
    operating on the stable tensor record used by the native differential
    harness. Missing requested standard slices are reported as ``-1`` rather
    than guessed from neighboring parameters.
    """
    precision = np.asarray(eval_info['precision'], dtype=float)
    recall = np.asarray(eval_info['recall'], dtype=float)
    iou_thresholds = np.asarray(eval_info['iou_thresholds'], dtype=float)
    area_labels = list(eval_info['area_range_labels'])
    max_dets = list(map(int, eval_info['max_dets']))

    def _mean_valid(values):
        values = np.asarray(values, dtype=float)
        valid = values > -1
        if not np.any(valid):
            return -1.0
        return float(np.mean(values[valid]))

    def _area_index(label):
        try:
            return area_labels.index(label)
        except ValueError:
            return None

    def _maxdet_index(value):
        try:
            return max_dets.index(int(value))
        except ValueError:
            return None

    def _threshold_indices(value=None):
        if value is None:
            return np.arange(len(iou_thresholds), dtype=int)
        idxs = np.where(iou_thresholds == float(value))[0]
        return idxs.astype(int)

    def _ap(*, iou=None, area='all', max_det=100):
        ai = _area_index(area)
        mi = _maxdet_index(max_det)
        ti = _threshold_indices(iou)
        if ai is None or mi is None or len(ti) == 0:
            return -1.0
        return _mean_valid(precision[ti, :, :, ai, mi])

    def _ar(*, area='all', max_det=100):
        ai = _area_index(area)
        mi = _maxdet_index(max_det)
        if ai is None or mi is None:
            return -1.0
        return _mean_valid(recall[:, :, ai, mi])

    return {
        'AP': _ap(area='all', max_det=100),
        'AP50': _ap(iou=0.50, area='all', max_det=100),
        'AP75': _ap(iou=0.75, area='all', max_det=100),
        'AP_small': _ap(area='small', max_det=100),
        'AP_medium': _ap(area='medium', max_det=100),
        'AP_large': _ap(area='large', max_det=100),
        'AR1': _ar(area='all', max_det=1),
        'AR10': _ar(area='all', max_det=10),
        'AR100': _ar(area='all', max_det=100),
        'AR_small': _ar(area='small', max_det=100),
        'AR_medium': _ar(area='medium', max_det=100),
        'AR_large': _ar(area='large', max_det=100),
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
    """Return a stable JSON-friendly record from an accumulated COCOeval.

    The record is defined from the accumulated tensor rather than depending on
    ``COCOeval.summarize()`` having been called first.  This keeps the official
    wrapper usable for nonstandard IoU / recall / area / maxDet grids while the
    familiar 12 named statistics retain COCO's slice-selection semantics.
    Missing standard slices are reported as ``-1``.
    """
    eval_record = snapshot_pycocotools_eval(evaler)
    summary = summarize_coco_eval_tensor(eval_record)
    stats = [summary[key] for key in COCO_DETECTION_STAT_KEYS]
    params = {
        'iou_thresholds': eval_record['iou_thresholds'],
        'recall_thresholds': eval_record['recall_thresholds'],
        'max_dets': eval_record['max_dets'],
        'area_ranges': eval_record['area_ranges'],
        'area_range_labels': eval_record['area_range_labels'],
        'area_range_map': {
            label: list(bounds) for label, bounds in zip(
                eval_record['area_range_labels'], eval_record['area_ranges']
            )
        },
        'image_ids': list(map(int, evaler.params.imgIds)),
        'category_ids': eval_record['category_ids'],
    }
    return {
        'backend': OFFICIAL_COCO_IMPLEMENTATION,
        'implementation': OFFICIAL_COCO_IMPLEMENTATION,
        'result_schema': COCO_DETECTION_RESULT_SCHEMA,
        'official': True,
        'reference_implementation': OFFICIAL_COCO_IMPLEMENTATION,
        'iou_type': str(evaler.params.iouType),
        'summary': summary,
        'stats': stats,
        'params': params,
    }
