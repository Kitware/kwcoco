#!/usr/bin/env python3
"""Phase 2.4 / 3: native COCO tensor conformance for bbox and segm.

This runner upgrades the Phase-2.3 summary comparison into a differential
conformance suite against the official pycocotools package.  It validates:

* standard 12-value AP / AR summaries;
* row-level assignments at representative IoU thresholds;
* every populated precision / recall / score tensor cell;
* absent-category / absent-slice masks;
* explicit bbox crowd intersection-over-detection-area semantics;
* stable ties, maxDets cutoffs, area boundaries, empty images/categories;
* multiple deterministic randomized fixtures.

The native bbox path exercised here is explicit COCO compatibility.  It does
not change ``CocoEvaluator.evaluate()`` or assignment-v2's legacy bbox default.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time
import traceback

import numpy as np

from compare_metrics import _assignment_diff
from compare_metrics import make_case
from compare_metrics import timed_call
from run_phase23_native_coco import CUSTOM_CASES as PHASE23_CUSTOM_CASES
from run_phase23_native_coco import _base_dsets
from run_phase23_native_coco import _poly


PROFILES = {
    'quick': {
        'nimgs': 4,
        'image_size': (96, 96),
        'random_seeds': 2,
    },
    'full': {
        'nimgs': 32,
        'image_size': (256, 256),
        'random_seeds': 8,
    },
    'stress': {
        'nimgs': 96,
        'image_size': (512, 512),
        'random_seeds': 16,
    },
}
DEBUG_THRESHOLDS = [0.50, 0.75, 0.95]
IOU_TYPES = ('bbox', 'segm')
TENSOR_KEYS = ('precision', 'recall', 'scores')
CUSTOM_PARAMETER_GRID = {
    'iou_thresholds': [0.50, 0.625, 0.75, 0.90],
    'recall_thresholds': np.linspace(0.0, 1.0, 17).tolist(),
    'area_ranges': {
        'all': (0.0, 1e10),
        'small': (0.0, float(32 ** 2)),
        'medium': (float(32 ** 2), float(96 ** 2)),
        'large': (float(96 ** 2), 1e10),
        'tiny': (0.0, float(16 ** 2)),
    },
    'max_dets': [1, 5, 10, 17, 100],
}
CUSTOM_GRID_CASES = {
    'mixed', 'accumulator_curve', 'area_boundaries', 'random_mixed_seed_0',
}


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command(command, cwd=None):
    try:
        proc = subprocess.run(
            command, cwd=cwd, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except Exception as ex:
        return {'returncode': None, 'error': repr(ex), 'output': None}
    return {'returncode': proc.returncode, 'output': proc.stdout.strip()}


def environment_info():
    info = {
        'python': sys.version,
        'platform': platform.platform(),
        'executable': sys.executable,
        'packages': {
            'numpy': np.__version__,
            'kwcoco': _version('kwcoco'),
            'kwimage': _version('kwimage'),
            'kwimage_ext': _version('kwimage_ext'),
            'pycocotools': _version('pycocotools'),
        },
        'backends': {},
    }
    try:
        from kwcoco.metrics.overlap import resolve_mask_iou_backend
        _func, meta = resolve_mask_iou_backend('auto')
    except Exception as ex:
        info['backends']['mask_iou'] = {'status': 'error', 'error': repr(ex)}
    else:
        info['backends']['mask_iou'] = {'status': 'ok', **meta}
    try:
        from kwcoco.metrics.overlap import resolve_coco_raster_backend
        _codec, meta = resolve_coco_raster_backend('auto')
    except Exception as ex:
        info['backends']['coco_raster'] = {'status': 'error', 'error': repr(ex)}
    else:
        info['backends']['coco_raster'] = {'status': 'ok', **meta}
    return info


def source_state():
    import kwcoco
    repo = Path(kwcoco.__file__).resolve().parent.parent
    return {
        'repo': os.fspath(repo),
        'head': _command(['git', 'rev-parse', 'HEAD'], cwd=repo),
        'status': _command(['git', 'status', '--short'], cwd=repo),
        'diff_stat': _command(['git', 'diff', '--stat'], cwd=repo),
    }


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Path):
        return os.fspath(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    raise TypeError(type(obj))


def make_threshold_boundary_case():
    true, pred = _base_dsets(dims=(64, 64))
    # Same-height boxes shifted by one third of their width have IoU exactly
    # 0.5 in continuous coordinates: intersection=8*12, union=16*12.
    true_box = [8, 8, 12, 12]
    pred_box = [12, 8, 12, 12]
    true.add_annotation(
        id=1, image_id=1, category_id=1, bbox=true_box,
        segmentation=_poly(8, 8, 20, 20))
    pred.add_annotation(
        id=11, image_id=1, category_id=1, bbox=pred_box,
        segmentation=_poly(12, 8, 24, 20), score=.9)
    return true, pred, {'kind': 'iou_exactly_0.5'}


def make_area_boundary_case():
    true, pred = _base_dsets(dims=(160, 160), gids=(1, 2))
    # Exact COCO boundaries: 32^2 belongs to both small/medium, 96^2 belongs
    # to both medium/large because COCO uses inclusive bounds.
    for gid, side in [(1, 32), (2, 96)]:
        box = [8, 8, side, side]
        seg = _poly(8, 8, 8 + side, 8 + side)
        true.add_annotation(
            id=gid, image_id=gid, category_id=1, bbox=box, segmentation=seg)
        pred.add_annotation(
            id=10 + gid, image_id=gid, category_id=1, bbox=box,
            segmentation=seg, score=.9 - gid * .01)
    return true, pred, {'kind': 'inclusive_area_boundaries'}


def make_bbox_crowd_ioa_case():
    true, pred = _base_dsets(dims=(128, 128))
    # Detection is wholly inside a much larger crowd. Ordinary IoU is small,
    # while COCO crowd IoA is 1.0 and must suppress the detection.
    true.add_annotation(
        id=1, image_id=1, category_id=1,
        bbox=[8, 8, 96, 96], segmentation=_poly(8, 8, 104, 104),
        iscrowd=1)
    pred.add_annotation(
        id=11, image_id=1, category_id=1,
        bbox=[32, 32, 12, 12], segmentation=_poly(32, 32, 44, 44),
        score=.99)
    return true, pred, {'kind': 'bbox_crowd_ioa_not_union_iou'}


def make_maxdet_tie_case():
    true, pred = _base_dsets(dims=(96, 96))
    true.add_annotation(
        id=1, image_id=1, category_id=1,
        bbox=[4, 4, 12, 12], segmentation=_poly(4, 4, 16, 16))
    # 100 FPs then the TP, all with the same score. Stable source order must
    # keep the TP at rank 101 and maxDets=100 must truncate it.
    for idx in range(100):
        x = 48 + (idx % 3)
        y = 48 + ((idx // 3) % 3)
        pred.add_annotation(
            id=1000 + idx, image_id=1, category_id=1,
            bbox=[x, y, 8, 8], segmentation=_poly(x, y, x + 8, y + 8),
            score=.5)
    pred.add_annotation(
        id=2000, image_id=1, category_id=1,
        bbox=[4, 4, 12, 12], segmentation=_poly(4, 4, 16, 16), score=.5)
    return true, pred, {'kind': 'stable_tie_at_maxdet_cutoff'}


def make_duplicate_detection_case():
    true, pred = _base_dsets(dims=(64, 64))
    box = [8, 8, 20, 20]
    seg = _poly(8, 8, 28, 28)
    true.add_annotation(
        id=1, image_id=1, category_id=1, bbox=box, segmentation=seg)
    for aid, score in [(11, .9), (12, .8), (13, .7)]:
        pred.add_annotation(
            id=aid, image_id=1, category_id=1, bbox=box,
            segmentation=seg, score=score)
    return true, pred, {'kind': 'duplicate_detections'}


def make_equal_iou_truth_tie_case():
    true, pred = _base_dsets(dims=(64, 64))
    box = [8, 8, 20, 20]
    seg = _poly(8, 8, 28, 28)
    true.add_annotation(
        id=1, image_id=1, category_id=1, bbox=box, segmentation=seg)
    true.add_annotation(
        id=2, image_id=1, category_id=1, bbox=box, segmentation=seg)
    pred.add_annotation(
        id=11, image_id=1, category_id=1, bbox=box,
        segmentation=seg, score=.9)
    return true, pred, {'kind': 'equal_iou_later_truth_tie_break'}



def make_fractional_polygon_case():
    true, pred = _base_dsets(dims=(80, 80))
    true_seg = [[8.25, 8.75, 38.50, 9.10, 37.85, 41.20, 8.60, 40.65]]
    pred_seg = [[9.05, 8.40, 39.10, 9.45, 38.35, 41.55, 9.15, 40.20]]
    true.add_annotation(
        id=1, image_id=1, category_id=1,
        bbox=[8.25, 8.75, 30.25, 32.45], segmentation=true_seg)
    pred.add_annotation(
        id=11, image_id=1, category_id=1,
        bbox=[9.05, 8.40, 30.05, 33.15], segmentation=pred_seg, score=.93)
    return true, pred, {'kind': 'fractional_polygon_rasterization'}


def make_rich_hole_case():
    true, pred = _base_dsets(dims=(64, 64))
    rich = {
        'exterior': [[8, 8], [56, 8], [56, 56], [8, 56]],
        'interiors': [
            [[24, 24], [40, 24], [40, 40], [24, 40]],
        ],
    }
    for dset, aid, score in [(true, 1, None), (pred, 11, .91)]:
        kw = {
            'id': aid,
            'image_id': 1,
            'category_id': 1,
            'bbox': [8, 8, 48, 48],
            'segmentation': rich,
        }
        if score is not None:
            kw['score'] = score
        dset.add_annotation(**kw)
    return true, pred, {'kind': 'rich_polygon_hole_exact_rle_boundary'}


def make_rle_geometry_case():
    from pycocotools import mask as mask_utils

    true, pred = _base_dsets(dims=(72, 72))
    true_mask = np.zeros((72, 72), dtype=np.uint8, order='F')
    pred_mask = np.zeros((72, 72), dtype=np.uint8, order='F')
    true_mask[10:46, 14:52] = 1
    pred_mask[12:48, 15:53] = 1

    def encode(mask):
        rle = dict(mask_utils.encode(np.asfortranarray(mask)))
        if isinstance(rle['counts'], bytes):
            rle['counts'] = rle['counts'].decode('ascii')
        rle['size'] = list(map(int, rle['size']))
        return rle

    true.add_annotation(
        id=1, image_id=1, category_id=1,
        bbox=[14, 10, 38, 36], segmentation=encode(true_mask))
    pred.add_annotation(
        id=11, image_id=1, category_id=1,
        bbox=[15, 12, 38, 36], segmentation=encode(pred_mask), score=.88)
    return true, pred, {'kind': 'compressed_rle_geometry'}

def make_explicit_ignore_extension_case():
    true, pred = _base_dsets(dims=(64, 64))
    box = [8, 8, 20, 20]
    seg = _poly(8, 8, 28, 28)
    # `ignore` is a kwcoco extension for this metric family, not an official
    # COCO detection-ignore signal. Strict reference mode must still score it.
    true.add_annotation(
        id=1, image_id=1, category_id=1, bbox=box, segmentation=seg,
        ignore=1)
    pred.add_annotation(
        id=11, image_id=1, category_id=1, bbox=box, segmentation=seg,
        score=.9)
    return true, pred, {'kind': 'kwcoco_explicit_ignore_not_official_coco'}


def make_negative_score_case():
    true, pred = _base_dsets(dims=(64, 64))
    box = [8, 8, 20, 20]
    seg = _poly(8, 8, 28, 28)
    true.add_annotation(
        id=1, image_id=1, category_id=1, bbox=box, segmentation=seg)
    # A higher-scoring negative-confidence FP keeps the precision curve
    # nontrivial while forcing valid sampled score thresholds below zero.
    pred.add_annotation(
        id=11, image_id=1, category_id=1, bbox=[36, 36, 16, 16],
        segmentation=_poly(36, 36, 52, 52), score=-0.1)
    pred.add_annotation(
        id=12, image_id=1, category_id=1, bbox=box,
        segmentation=seg, score=-0.2)
    return true, pred, {'kind': 'negative_detection_scores'}


def make_empty_truth_case():
    true, pred = _base_dsets(dims=(64, 64))
    pred.add_annotation(
        id=11, image_id=1, category_id=1, bbox=[8, 8, 20, 20],
        segmentation=_poly(8, 8, 28, 28), score=.9)
    return true, pred, {'kind': 'all_predictions_no_truth'}


def make_empty_prediction_case():
    true, pred = _base_dsets(dims=(64, 64))
    true.add_annotation(
        id=1, image_id=1, category_id=1, bbox=[8, 8, 20, 20],
        segmentation=_poly(8, 8, 28, 28))
    return true, pred, {'kind': 'all_truth_no_predictions'}


PHASE3_CUSTOM_CASES = {
    **PHASE23_CUSTOM_CASES,
    'threshold_boundary': make_threshold_boundary_case,
    'area_boundaries': make_area_boundary_case,
    'bbox_crowd_ioa': make_bbox_crowd_ioa_case,
    'maxdet_tie': make_maxdet_tie_case,
    'duplicate_detections': make_duplicate_detection_case,
    'equal_iou_truth_tie': make_equal_iou_truth_tie_case,
    'fractional_polygon': make_fractional_polygon_case,
    'rich_hole': make_rich_hole_case,
    'rle_geometry': make_rle_geometry_case,
    'explicit_ignore_extension': make_explicit_ignore_extension_case,
    'negative_scores': make_negative_score_case,
    'empty_truth': make_empty_truth_case,
    'empty_prediction': make_empty_prediction_case,
}
BASE_CASES = ['perfect', 'box_noise', 'class_noise', 'mask_only_shift', 'mixed']


def _case_data(name, profile, seed):
    if name in PHASE3_CUSTOM_CASES:
        return PHASE3_CUSTOM_CASES[name]()
    if name.startswith('random_mixed_seed_'):
        return make_case('mixed', profile, seed)
    return make_case(name, profile, seed)


def _tensor_diff(native_eval, official_eval, tol):
    result = {}
    if native_eval['schema'] != official_eval['schema']:
        result['schema_mismatch'] = [
            native_eval['schema'], official_eval['schema']]
    native_counts = list(map(int, native_eval['counts']))
    official_counts = list(map(int, official_eval['counts']))
    result['native_counts'] = native_counts
    result['official_counts'] = official_counts
    result['count_shape_match'] = native_counts == official_counts

    parameter_mismatches = {}
    exact_keys = [
        'category_ids', 'image_ids', 'area_range_labels', 'max_dets',
    ]
    numeric_keys = [
        'iou_thresholds', 'recall_thresholds', 'area_ranges',
    ]
    for key in exact_keys:
        n = native_eval.get(key, None)
        o = official_eval.get(key, None)
        if n != o:
            parameter_mismatches[key] = {'native': n, 'official': o}
    for key in numeric_keys:
        n = np.asarray(native_eval.get(key, []), dtype=float)
        o = np.asarray(official_eval.get(key, []), dtype=float)
        if n.shape != o.shape or not np.allclose(n, o, atol=0, rtol=0):
            parameter_mismatches[key] = {
                'native': n.tolist(), 'official': o.tolist()}
    result['parameter_mismatches'] = parameter_mismatches

    for key in TENSOR_KEYS:
        n = np.asarray(native_eval[key], dtype=float)
        o = np.asarray(official_eval[key], dtype=float)
        info = {
            'native_shape': list(n.shape),
            'official_shape': list(o.shape),
        }
        if n.shape != o.shape:
            info.update({
                'shape_match': False,
                'absent_mask_mismatches': None,
                'num_over_tolerance': None,
                'max_abs_delta': None,
            })
            result[key] = info
            continue
        info['shape_match'] = True
        if key == 'scores':
            # COCO does not require confidence scores to be nonnegative. The
            # score tensor shares the precision tensor's population mask, so
            # use that mask instead of confusing a valid negative score with
            # COCOeval's -1 sentinel for an absent category/slice.
            n_absent = np.asarray(native_eval['precision'], dtype=float) < 0
            o_absent = np.asarray(official_eval['precision'], dtype=float) < 0
        else:
            n_absent = n < 0
            o_absent = o < 0
        absent_delta = n_absent != o_absent
        info['absent_mask_mismatches'] = int(absent_delta.sum())
        shared = ~(n_absent | o_absent)
        if np.any(shared):
            delta = np.abs(n[shared] - o[shared])
            max_abs = float(delta.max())
            num_bad = int(np.count_nonzero(delta > tol))
            if num_bad:
                full_delta = np.full(n.shape, -np.inf, dtype=float)
                full_delta[shared] = np.abs(n[shared] - o[shared])
                worst = np.unravel_index(int(np.argmax(full_delta)), n.shape)
                info['worst_index'] = list(map(int, worst))
                info['worst_native'] = float(n[worst])
                info['worst_official'] = float(o[worst])
            else:
                info['worst_index'] = None
        else:
            max_abs = 0.0
            num_bad = 0
            info['worst_index'] = None
        info['num_over_tolerance'] = num_bad
        info['max_abs_delta'] = max_abs
        result[key] = info
    return result

def _geometry_compare(
    dmet,
    iou_type,
    tol,
    *,
    parameter_profile='standard',
    parameter_overrides=None,
    compare_assignment=True,
    assignment_backend='auto',
):
    score_kw = dict(parameter_overrides or {})
    native, native_wall = timed_call(
        lambda: dmet.score_native_coco(
            iou_type=iou_type,
            overlap_backend='auto',
            raster_backend='auto',
            annotation_ignore_policy='coco',
            assignment_backend=assignment_backend,
            with_eval=True,
            **score_kw,
        ), repeat=1)
    official, official_wall = timed_call(
        lambda: dmet.score_pycocotools(
            iou_type=iou_type,
            bbox_annotation_policy='coco',
            with_eval=True,
            with_confusion=False,
            verbose=0,
            **score_kw,
        ), repeat=1)

    assignment = {}
    if compare_assignment:
        row_kw = {
            'iou_thresh': DEBUG_THRESHOLDS,
            'iou_type': iou_type,
            'compat': 'mutex',
            'ignore_classes': None,
            'threshold_policy': 'inclusive',
            'annotation_ignore_policy': 'coco',
            'max_dets': 100,
            'max_dets_scope': 'category',
            'area_range': (0.0, 1e10),
        }
        if iou_type == 'segm':
            row_kw.update({
                'overlap_backend': 'auto',
                'segmentation_rasterizer': 'coco',
                'raster_backend': 'auto',
            })
        else:
            row_kw['bbox_overlap_mode'] = 'coco'
        native_rows = dmet.confusion_vectors_v2(**row_kw)
        official_rows = dmet.score_pycocotools(
            iou_type=iou_type,
            bbox_annotation_policy='coco',
            iou_thresholds=DEBUG_THRESHOLDS,
            with_confusion=True,
            verbose=0,
        )['iou_to_cfsn_vecs']
        assignment = {
            str(t): _assignment_diff(native_rows[t], official_rows[t])
            for t in DEBUG_THRESHOLDS
        }

    deltas = {
        key: float(native['summary'][key] - official['summary'][key])
        for key in native['summary']
    }
    abs_deltas = {key: abs(value) for key, value in deltas.items()}
    tensor = _tensor_diff(native['eval'], official['eval'], tol=tol)

    return {
        'parameter_profile': parameter_profile,
        'native': {
            key: native[key]
            for key in [
                'backend', 'implementation', 'result_schema', 'official',
                'reference_implementation', 'iou_type', 'assignment',
                'accumulator', 'summary', 'params', 'provenance', 'timing'
            ]
        },
        'official': {
            key: official[key]
            for key in [
                'backend', 'implementation', 'result_schema', 'official',
                'reference_implementation', 'iou_type', 'summary', 'params'
            ]
        },
        'delta': deltas,
        'absolute_delta': abs_deltas,
        'tensor': tensor,
        'assignment': assignment,
        'timing': {
            'native_wall': native_wall,
            'official_wall': official_wall,
        },
    }

def compare_case(case_name, profile, seed, tol, assignment_backend='auto'):
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    if case_name.startswith('random_mixed_seed_'):
        seed = int(case_name.rsplit('_', 1)[1])
    true_dset, pred_dset, spec = _case_data(case_name, profile, seed)
    dmet, dmet_timing = timed_call(
        lambda: DetectionMetrics.from_coco(true_dset, pred_dset), repeat=1)
    for gid, weights in spec.get('dmet_truth_weights', {}).items():
        dmet.gid_to_true_dets[gid].data['weights'] = np.asarray(
            weights, dtype=float)

    geometry = {}
    parameter_profiles = {}
    for iou_type in IOU_TYPES:
        geometry[iou_type] = _geometry_compare(
            dmet, iou_type, tol, parameter_profile='standard',
            assignment_backend=assignment_backend)
        if case_name in CUSTOM_GRID_CASES:
            custom = dict(CUSTOM_PARAMETER_GRID)
            gids = sorted(
                set(dmet.gid_to_true_dets) | set(dmet.gid_to_pred_dets)
            )
            if len(gids) > 1:
                # Exercise COCOeval's image-id normalization with a subset,
                # duplicate, and deliberately reversed source order.
                custom['gids'] = [gids[-1], gids[0], gids[-1]]
            parameter_profiles.setdefault(iou_type, {})['custom_grid'] = (
                _geometry_compare(
                    dmet,
                    iou_type,
                    tol,
                    parameter_profile='custom_grid',
                    parameter_overrides=custom,
                    compare_assignment=False,
                    assignment_backend=assignment_backend,
                )
            )

    return {
        'case': case_name,
        'seed': seed,
        'spec': spec,
        'dataset': {
            'n_images': len(true_dset.imgs),
            'n_true': len(true_dset.anns),
            'n_pred': len(pred_dset.anns),
        },
        'geometry': geometry,
        'parameter_profiles': parameter_profiles,
        'timing': {'detection_metrics_build': dmet_timing},
    }

def _case_plan(profile):
    names = BASE_CASES + list(PHASE3_CUSTOM_CASES)
    for seed in range(int(profile['random_seeds'])):
        names.append(f'random_mixed_seed_{seed}')
    # Preserve order while eliminating duplicates from the phase23 custom map.
    return list(dict.fromkeys(names))


def build_report(mode, tol, assignment_backend='auto'):
    profile = dict(PROFILES[mode])
    cases = _case_plan(profile)
    report = {
        'schema_version': 1,
        'campaign': 'phase24_phase3_coco_conformance',
        'mode': mode,
        'created_unix': time.time(),
        'environment': environment_info(),
        'source_state': source_state(),
        'profile': profile,
        'iou_types': list(IOU_TYPES),
        'assignment_backend': assignment_backend,
        'cases': cases,
        'experiments': [],
    }
    for idx, case_name in enumerate(cases):
        print(f'Running Phase-3 COCO conformance case {case_name!r}')
        try:
            result = compare_case(
                case_name, profile, seed=idx, tol=tol,
                assignment_backend=assignment_backend)
        except Exception as ex:
            report['experiments'].append({
                'name': case_name,
                'status': 'error',
                'error': repr(ex),
                'traceback': traceback.format_exc(),
            })
        else:
            report['experiments'].append({
                'name': case_name,
                'status': 'ok',
                'result': result,
            })
    return report


def _iter_comparisons(result):
    for iou_type, comparison in result['geometry'].items():
        yield iou_type, 'standard', comparison
    for iou_type, profiles in result.get('parameter_profiles', {}).items():
        for profile_name, comparison in profiles.items():
            yield iou_type, profile_name, comparison


def parity_failures(report, tol):
    failures = []
    for exp in report['experiments']:
        if exp['status'] != 'ok':
            failures.append(f"{exp['name']}: runner error")
            continue
        result = exp['result']
        for iou_type, profile_name, geo in _iter_comparisons(result):
            prefix = f"{exp['name']}[{iou_type}/{profile_name}]"
            native = geo['native']
            official = geo['official']
            if native['result_schema'] != official['result_schema']:
                failures.append(f'{prefix}: result schema mismatch')
            if native['official'] is not False or official['official'] is not True:
                failures.append(f'{prefix}: official/native identity mismatch')
            if native['reference_implementation'] != official['implementation']:
                failures.append(f'{prefix}: reference implementation mismatch')
            if native['iou_type'] != iou_type or official['iou_type'] != iou_type:
                failures.append(f'{prefix}: iou_type identity mismatch')
            shared_param_keys = [
                'iou_thresholds', 'recall_thresholds', 'max_dets',
                'area_ranges', 'area_range_labels', 'area_range_map',
                'image_ids', 'category_ids',
            ]
            for key in shared_param_keys:
                if native['params'].get(key) != official['params'].get(key):
                    failures.append(
                        f'{prefix}: public parameter metadata mismatch for {key}'
                    )
            provenance = native.get('provenance', {})
            if iou_type == 'bbox':
                if provenance.get('overlap_backends') != ['numpy_coco_bbox']:
                    failures.append(f'{prefix}: bbox overlap provenance mismatch')
                if provenance.get('area_backends') != ['numpy_coco_xywh_area']:
                    failures.append(f'{prefix}: bbox area provenance mismatch')
            else:
                for key in ['overlap_backends', 'raster_backends', 'area_backends']:
                    if not provenance.get(key):
                        failures.append(
                            f'{prefix}: unresolved segmentation provenance {key}'
                        )
            bad_metrics = {
                key: value for key, value in geo['absolute_delta'].items()
                if value > tol
            }
            if bad_metrics:
                failures.append(
                    f'{prefix}: summary deltas exceed {tol}: {bad_metrics}')
            for threshold, diff in geo['assignment'].items():
                ndiff = diff['num_native_only'] + diff['num_official_only']
                if ndiff:
                    failures.append(
                        f'{prefix}: assignment delta @{threshold}: {ndiff}')
            tensor = geo['tensor']
            if tensor.get('schema_mismatch'):
                failures.append(f'{prefix}: eval tensor schema mismatch')
            if not tensor['count_shape_match']:
                failures.append(f'{prefix}: eval tensor count mismatch')
            if tensor['parameter_mismatches']:
                failures.append(
                    f"{prefix}: tensor parameter metadata mismatch: "
                    f"{sorted(tensor['parameter_mismatches'])}")
            for key in TENSOR_KEYS:
                info = tensor[key]
                if not info['shape_match']:
                    failures.append(f'{prefix}: {key} shape mismatch')
                    continue
                if info['absent_mask_mismatches']:
                    failures.append(
                        f"{prefix}: {key} absent-mask mismatches="
                        f"{info['absent_mask_mismatches']}")
                if info['num_over_tolerance']:
                    failures.append(
                        f"{prefix}: {key} cells over tolerance="
                        f"{info['num_over_tolerance']} max="
                        f"{info['max_abs_delta']}")
    return failures

def summary_lines(report):
    lines = [
        '# kwcoco Phase-2.4 / Phase-3 COCO conformance', '',
        f"Mode: `{report['mode']}`",
        f"Assignment backend: `{report.get('assignment_backend', 'auto')}`", '',
        '| case | geom/profile | status | AP delta | assignment delta @.50 | '
        'precision max delta | recall max delta | score max delta | native/COCO wall |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for exp in report['experiments']:
        if exp['status'] != 'ok':
            lines.append(f"| `{exp['name']}` | - | error | | | | | | |")
            continue
        for iou_type, profile_name, geo in _iter_comparisons(exp['result']):
            diff50 = geo['assignment'].get('0.5', None)
            ndiff = '' if diff50 is None else (
                diff50['num_native_only'] + diff50['num_official_only'])
            ratio = (
                geo['timing']['native_wall']['median_seconds'] /
                geo['timing']['official_wall']['median_seconds']
            )
            tensor = geo['tensor']
            lines.append(
                f"| `{exp['name']}` | `{iou_type}/{profile_name}` | ok | "
                f"{geo['absolute_delta']['AP']:.3g} | {ndiff} | "
                f"{tensor['precision']['max_abs_delta']:.3g} | "
                f"{tensor['recall']['max_abs_delta']:.3g} | "
                f"{tensor['scores']['max_abs_delta']:.3g} | {ratio:.2f}x |"
            )
    lines.extend([
        '',
        'Parity is evaluated against the installed official '
        '`pycocotools.COCOeval` implementation. The JSON report records '
        'summary deltas, row-assignment deltas, tensor parameter metadata, '
        'absent-mask mismatches, tensor max absolute deltas / counts over '
        'tolerance, backend provenance, and timing for bbox and segmentation '
        'independently.',
        '',
        'Selected fixtures also run a nonstandard IoU / recall / area / maxDet '
        'tensor grid plus a normalized image subset. This guards against '
        'accidental dependence on the standard 10x101x4x3 COCO shape.',
        '',
        'The native bbox path is an explicit COCO profile and does not change '
        'the historical `CocoEvaluator.evaluate()` bbox behavior.',
    ])
    return lines

def write_bundle(report, output_dpath):
    dpath = Path(output_dpath)
    dpath.mkdir(parents=True, exist_ok=True)
    mode = report['mode']
    jf = dpath / f'phase3_coco_conformance_{mode}_results.json'
    mf = dpath / f'phase3_coco_conformance_{mode}_summary.md'
    jf.write_text(json.dumps(report, indent=2, default=_json_default) + '\n')
    mf.write_text('\n'.join(summary_lines(report)) + '\n')
    bundle = dpath / f'kwcoco_metrics_phase3_coco_conformance_{mode}.tar.gz'
    with tarfile.open(bundle, 'w:gz') as tar:
        tar.add(jf, arcname=jf.name)
        tar.add(mf, arcname=mf.name)
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=sorted(PROFILES), default='quick')
    parser.add_argument(
        '--assignment-backend', choices=['auto', 'python', 'kwimage_ext'],
        default='auto', help='native greedy assignment kernel',
    )
    parser.add_argument(
        '--output-dpath', type=Path,
        default=Path('_metric_compare_phase3_coco_conformance'))
    parser.add_argument(
        '--assert-parity', action='store_true',
        help='exit nonzero on summary, assignment, or tensor disagreement')
    parser.add_argument(
        '--parity-tol', type=float, default=1e-12,
        help='absolute tolerance used by --assert-parity')
    args = parser.parse_args(argv)

    report = build_report(
        args.mode, tol=args.parity_tol,
        assignment_backend=args.assignment_backend)
    bundle = write_bundle(report, args.output_dpath)
    print('Phase-2.4 / Phase-3 COCO conformance suite complete')
    print(f'  bundle: {bundle}')
    if args.assert_parity:
        failures = parity_failures(report, args.parity_tol)
        if failures:
            print('PARITY FAILURES:')
            for failure in failures:
                print(f'  - {failure}')
            return 1
        print(f'  parity: ok (absolute tolerance={args.parity_tol:g})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
