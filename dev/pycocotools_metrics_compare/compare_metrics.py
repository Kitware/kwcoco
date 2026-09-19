#!/usr/bin/env python3
"""
Measured comparison of kwcoco, COCOeval, and mask-IoU backends.

This is development / benchmarking code.  It intentionally uses a few private
backend hooks so we can compare the exact low-level C mask kernels.  Production
metric APIs should not depend on those private hooks.

See ``README.md`` in this directory for the comparison matrix and rationale.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np


PROFILES = {
    'smoke': {
        'nimgs': 4,
        'image_size': (96, 96),
        'kernel_limit': 24,
        'pixel_global_limit': 1_000_000,
    },
    'small': {
        'nimgs': 32,
        'image_size': (256, 256),
        'kernel_limit': 96,
        'pixel_global_limit': 10_000_000,
    },
    'medium': {
        'nimgs': 128,
        'image_size': (384, 384),
        'kernel_limit': 192,
        'pixel_global_limit': 5_000_000,
    },
}


CASE_SPECS = {
    'perfect': {
        'box_noise': 0.0,
        'cls_noise': 0.0,
        'score_noise': 0.05,
        'n_fp': 0,
        'n_fn': 0,
        'mask_shift': (0.0, 0.0),
    },
    'box_noise': {
        'box_noise': 3.0,
        'cls_noise': 0.0,
        'score_noise': 0.20,
        'n_fp': (0, 2),
        'n_fn': (0, 1),
        'mask_shift': (0.0, 0.0),
    },
    'class_noise': {
        'box_noise': 1.0,
        'cls_noise': 0.20,
        'score_noise': 0.30,
        'n_fp': (0, 1),
        'n_fn': (0, 1),
        'mask_shift': (0.0, 0.0),
    },
    'mask_only_shift': {
        'box_noise': 0.0,
        'cls_noise': 0.0,
        'score_noise': 0.05,
        'n_fp': 0,
        'n_fn': 0,
        # Deliberately make bbox and mask quality disagree.
        'mask_shift': (12.0, 7.0),
    },
    'mixed': {
        'box_noise': 4.0,
        'cls_noise': 0.10,
        'score_noise': 0.35,
        'n_fp': (0, 3),
        'n_fn': (0, 2),
        'mask_shift': (3.0, -2.0),
    },
}


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_info():
    import kwcoco
    import kwimage

    info = {
        'python': sys.version,
        'platform': platform.platform(),
        'executable': sys.executable,
        'pid': os.getpid(),
        'packages': {
            'kwcoco': getattr(kwcoco, '__version__', None),
            'kwimage': getattr(kwimage, '__version__', None),
            'kwimage_ext': _package_version('kwimage_ext'),
            'pycocotools': _package_version('pycocotools'),
            'numpy': np.__version__,
        },
        'mask_backends': {},
    }

    # Check the extension directly.  Do not ask kwimage's dispatcher to select
    # the backend here because that intentionally emits a warning when the
    # optional extension is absent.  Backend absence is a normal benchmark
    # state and should be represented in the report, not as a noisy failure.
    try:
        from kwimage_ext.structs._mask_backend import cython_mask as _kwext_mask  # NOQA
    except Exception as ex:
        info['mask_backends']['kwimage_ext'] = {
            'available': False,
            'status': 'unavailable',
            'error': repr(ex),
        }
    else:
        info['mask_backends']['kwimage_ext'] = {
            'available': True,
            'status': 'available',
            'implementation': 'kwimage_ext.structs._mask_backend.cython_mask',
        }

    try:
        from pycocotools import _mask as _pycoco_mask  # NOQA
    except Exception as ex:
        info['mask_backends']['pycocotools'] = {
            'available': False,
            'status': 'unavailable',
            'error': repr(ex),
        }
    else:
        info['mask_backends']['pycocotools'] = {
            'available': True,
            'status': 'available',
            'implementation': 'pycocotools._mask',
        }

    if info['mask_backends']['kwimage_ext']['available']:
        dispatch = 'kwimage_ext'
    elif info['mask_backends']['pycocotools']['available']:
        dispatch = 'pycocotools-fallback'
    else:
        dispatch = None
    info['kwimage_mask_dispatch_expected'] = dispatch
    return info


def timed_call(func, repeat=1):
    """Run a callable repeatedly and retain the fastest/median timings."""
    times = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - start)
    return result, {
        'repeat': repeat,
        'min_seconds': min(times),
        'median_seconds': statistics.median(times),
        'max_seconds': max(times),
        'all_seconds': times,
    }


def _jsonify_scalar(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _translate_prediction_masks(pred_dset, offset):
    import kwimage

    dx, dy = offset
    if dx == 0 and dy == 0:
        return pred_dset
    transform = kwimage.Affine.translate((dx, dy))
    for ann in pred_dset.anns.values():
        segmentation = ann.get('segmentation', None)
        if segmentation is None:
            continue
        seg = kwimage.Segmentation.coerce(segmentation)
        if seg is None:
            continue
        shifted = seg.warp(transform)
        ann['segmentation'] = shifted.to_coco(style='new')
        # The bbox is intentionally not updated.  This is a controlled test
        # axis that separates mask quality from bbox quality.
    return pred_dset


def make_case(case_name, profile, seed):
    import kwcoco
    from kwcoco.demo.perterb import perterb_coco

    spec = dict(CASE_SPECS[case_name])
    nimgs = profile['nimgs']
    image_size = tuple(profile['image_size'])

    true_dset = kwcoco.CocoDataset.demo(
        f'shapes{nimgs}',
        image_size=image_size,
        rng=seed,
        verbose=0,
        use_cache=False,
    )
    perturb_kw = {
        'rng': seed + 1,
        'box_noise': spec['box_noise'],
        'cls_noise': spec['cls_noise'],
        'score_noise': spec['score_noise'],
        'n_fp': spec['n_fp'],
        'n_fn': spec['n_fn'],
        'with_probs': False,
        'verbose': 0,
    }
    pred_dset = perterb_coco(true_dset, **perturb_kw)
    pred_dset = _translate_prediction_masks(pred_dset, spec['mask_shift'])
    return true_dset, pred_dset, spec


def _assignment_set(cfsn):
    """Normalize matched assignments into stable image/local-index tuples."""
    data = cfsn.data
    gids = np.asarray(data['gid'])
    txs = np.asarray(data['txs'])
    pxs = np.asarray(data['pxs'])
    flags = (txs >= 0) & (pxs >= 0)
    return {
        (int(g), int(p), int(t))
        for g, p, t in zip(gids[flags], pxs[flags], txs[flags])
    }


def _assignment_diff(native_cfsn, official_cfsn):
    native = _assignment_set(native_cfsn)
    official = _assignment_set(official_cfsn)
    return {
        'num_native_matches': len(native),
        'num_official_matches': len(official),
        'num_shared_matches': len(native & official),
        'num_native_only': len(native - official),
        'num_official_only': len(official - native),
        'native_only_examples': sorted(native - official)[:20],
        'official_only_examples': sorted(official - native)[:20],
    }


def _native_measures_from_cfsn(cfsn):
    ovr = cfsn.binarize_ovr()
    info = ovr.measures(ap_method='pycocotools')
    return {
        'mAP': float(info['mAP']),
        'mAUC': float(info['mAUC']),
    }


def compare_metric_layers(true_dset, pred_dset, repeat=1):
    """Compare assignment and accumulation layers for bbox and masks."""
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dmet, build_timing = timed_call(
        lambda: DetectionMetrics.from_coco(true_dset, pred_dset), repeat=repeat
    )

    native_cfsn, native_assign_timing = timed_call(
        lambda: dmet.confusion_vectors(
            iou_thresh=0.5,
            compat='mutex',
            workers=0,
            track_probs=False,
        ),
        repeat=repeat,
    )
    native_measures, native_measure_timing = timed_call(
        lambda: _native_measures_from_cfsn(native_cfsn), repeat=repeat
    )

    official_bbox, official_bbox_timing = timed_call(
        lambda: dmet.score_pycocotools(
            iou_type='bbox',
            with_evaler=False,
            with_confusion=False,
            verbose=0,
        ),
        repeat=repeat,
    )
    official_bbox_debug, official_bbox_debug_timing = timed_call(
        lambda: dmet.score_pycocotools(
            iou_type='bbox',
            with_evaler=False,
            with_confusion=True,
            verbose=0,
        ),
        repeat=1,
    )
    official_bbox_cfsn = official_bbox_debug['iou_to_cfsn_vecs'][0.5]
    official_bbox_native_measures = _native_measures_from_cfsn(
        official_bbox_cfsn
    )

    official_segm, official_segm_timing = timed_call(
        lambda: dmet.score_pycocotools(
            iou_type='segm',
            with_evaler=False,
            with_confusion=False,
            verbose=0,
        ),
        repeat=repeat,
    )
    official_segm_debug, official_segm_debug_timing = timed_call(
        lambda: dmet.score_pycocotools(
            iou_type='segm',
            with_evaler=False,
            with_confusion=True,
            verbose=0,
        ),
        repeat=1,
    )
    official_segm_cfsn = official_segm_debug['iou_to_cfsn_vecs'][0.5]
    official_segm_native_measures = _native_measures_from_cfsn(
        official_segm_cfsn
    )

    bbox_triplet = {
        'native_native_mAP50': native_measures['mAP'],
        'official_native_mAP50': official_bbox_native_measures['mAP'],
        'official_official_AP50': float(
            official_bbox['summary']['AP50']
        ),
    }
    bbox_triplet['delta_assignment'] = (
        bbox_triplet['native_native_mAP50']
        - bbox_triplet['official_native_mAP50']
    )
    bbox_triplet['delta_accumulation'] = (
        bbox_triplet['official_native_mAP50']
        - bbox_triplet['official_official_AP50']
    )
    bbox_triplet['delta_end_to_end'] = (
        bbox_triplet['native_native_mAP50']
        - bbox_triplet['official_official_AP50']
    )

    segm_pair = {
        'official_native_mAP50': official_segm_native_measures['mAP'],
        'official_official_AP50': float(
            official_segm['summary']['AP50']
        ),
        'native_native_mAP50': None,
        'native_native_status': 'phase2-not-implemented',
    }
    segm_pair['delta_accumulation'] = (
        segm_pair['official_native_mAP50']
        - segm_pair['official_official_AP50']
    )

    return {
        'dataset_build_timing': build_timing,
        'bbox': {
            'metrics': bbox_triplet,
            'assignment_diff': _assignment_diff(
                native_cfsn, official_bbox_cfsn
            ),
            'timing': {
                'native_assignment': native_assign_timing,
                'native_measures': native_measure_timing,
                'official_cocoeval': official_bbox_timing,
                'official_with_confusion_extraction': official_bbox_debug_timing,
            },
        },
        'segm': {
            'metrics': segm_pair,
            'timing': {
                'official_cocoeval': official_segm_timing,
                'official_with_confusion_extraction': official_segm_debug_timing,
            },
        },
    }


def _backend_ready_rle(rle):
    rle = dict(rle)
    counts = rle.get('counts', None)
    if isinstance(counts, str):
        rle['counts'] = counts.encode('ascii')
    rle['size'] = list(map(int, rle['size']))
    return rle


def _collect_rles(dset, limit):
    from kwcoco.metrics.coco_metrics import (
        coerce_pycocotools_segmentation,
        _strict_segmentation_to_rle,
    )

    rles = []
    for gid in dset.images():
        img = dset.imgs[gid]
        dims = (int(img['height']), int(img['width']))
        for ann in dset.annots(gid=gid).objs:
            segmentation = ann.get('segmentation', None)
            if segmentation is None:
                continue
            strict, _area = coerce_pycocotools_segmentation(
                segmentation, dims
            )
            rle = _strict_segmentation_to_rle(strict, dims)
            rles.append(_backend_ready_rle(rle))
            if len(rles) >= limit:
                return rles
    return rles


def compare_mask_kernels(true_dset, pred_dset, limit=64, repeat=3):
    """Compare explicit mask-IoU backends on identical compressed RLEs.

    Each backend has its own result row.  In particular, the kwimage extension
    row remains present with ``status='skipped'`` when the optional extension
    is unavailable.  We never silently relabel a pycocotools fallback as a
    kwimage_ext measurement.
    """
    true_rles = _collect_rles(true_dset, limit)
    pred_rles = _collect_rles(pred_dset, limit)
    if not true_rles or not pred_rles:
        return {
            'status': 'no-rles',
            'backends': {},
            'comparisons': {},
        }

    iscrowd = np.zeros(len(true_rles), dtype=np.uint8)
    report = {
        'status': 'measured',
        'shape': [len(pred_rles), len(true_rles)],
        'num_pairs': int(len(pred_rles) * len(true_rles)),
        'backends': {},
        'comparisons': {},
    }

    py_result = None
    try:
        from pycocotools import _mask as pycoco_mask
    except Exception as ex:
        report['backends']['pycocotools'] = {
            'status': 'skipped',
            'available': False,
            'reason': 'backend-unavailable',
            'error': repr(ex),
        }
    else:
        py_result, py_timing = timed_call(
            lambda: np.asarray(
                pycoco_mask.iou(pred_rles, true_rles, iscrowd)
            ),
            repeat=repeat,
        )
        report['backends']['pycocotools'] = {
            'status': 'measured',
            'available': True,
            'implementation': 'pycocotools._mask',
            'timing': py_timing,
        }

    kw_result = None
    try:
        from kwimage_ext.structs._mask_backend import cython_mask as kw_backend
    except Exception as ex:
        report['backends']['kwimage_ext'] = {
            'status': 'skipped',
            'available': False,
            'reason': 'backend-unavailable',
            'error': repr(ex),
        }
    else:
        kw_result, kw_timing = timed_call(
            lambda: np.asarray(
                kw_backend.iou(pred_rles, true_rles, iscrowd)
            ),
            repeat=repeat,
        )
        report['backends']['kwimage_ext'] = {
            'status': 'measured',
            'available': True,
            'implementation': 'kwimage_ext.structs._mask_backend.cython_mask',
            'timing': kw_timing,
        }

    compare_key = 'kwimage_ext_vs_pycocotools'
    if kw_result is None or py_result is None:
        missing = []
        if kw_result is None:
            missing.append('kwimage_ext')
        if py_result is None:
            missing.append('pycocotools')
        report['status'] = 'partial'
        report['comparisons'][compare_key] = {
            'status': 'skipped',
            'reason': 'missing-backend',
            'missing': missing,
        }
    else:
        absdiff = np.abs(kw_result - py_result)
        kw_timing = report['backends']['kwimage_ext']['timing']
        py_timing = report['backends']['pycocotools']['timing']
        report['comparisons'][compare_key] = {
            'status': 'measured',
            'max_abs_diff': float(absdiff.max(initial=0)),
            'num_exact_differences': int(np.count_nonzero(kw_result != py_result)),
            'num_close_differences_1e12': int(
                np.count_nonzero(
                    ~np.isclose(kw_result, py_result, rtol=0, atol=1e-12)
                )
            ),
            'speed_ratio_pycoco_over_kwimage_ext': (
                py_timing['median_seconds'] / kw_timing['median_seconds']
                if kw_timing['median_seconds']
                else None
            ),
        }
    return report



def _instance_saliency_arrays(dset, gids, prediction=False):
    """
    Rasterize instance annotations into saliency truth / score images.

    For truth, all instance masks are unioned into a boolean foreground mask.
    For predictions, each instance mask is filled with its annotation score and
    overlapping predictions take the maximum score.  This is a controlled
    bridge from instance predictions to the dense saliency problem; it is not
    intended to make pixel AP equivalent to instance AP.
    """
    import kwimage

    arrays = []
    for gid in gids:
        img = dset.imgs[gid]
        dims = (int(img['height']), int(img['width']))
        if prediction:
            canvas = np.zeros(dims, dtype=np.float32)
        else:
            canvas = np.zeros(dims, dtype=bool)
        for ann in dset.annots(gid=gid).objs:
            segmentation = ann.get('segmentation', None)
            if segmentation is None:
                continue
            seg = kwimage.Segmentation.coerce(segmentation, dims=dims)
            if seg is None:
                continue
            mask = seg.to_mask(dims=dims).to_c_mask().data.astype(bool)
            if prediction:
                score = float(ann.get('score', 1.0))
                canvas[mask] = np.maximum(canvas[mask], score)
            else:
                canvas[mask] = True
        arrays.append(canvas)
    return arrays


def _pixel_measure_summary(measures):
    summary = measures.summary()
    # ``summary['max_f1']`` is intentionally a human-readable label such as
    # ``'f1=1.00@0.49'``.  Use the structured maximizer API for machine output.
    max_f1_info = measures.maximized_thresholds().get('f1', {})
    return {
        'ap': float(summary['ap']),
        'auc': float(summary['auc']),
        'max_f1': float(max_f1_info.get('metric_value', np.nan)),
        'max_f1_thresh': float(max_f1_info.get('thresh', np.nan)),
        'realpos_total': float(measures['realpos_total']),
        'realneg_total': float(measures['realneg_total']),
    }


def compare_dense_saliency_accumulation(
    true_dset,
    pred_dset,
    thresh_bins=1024,
    global_pixel_limit=10_000_000,
):
    """
    Exercise the confusion accumulation pattern used by segmentation_metrics.

    Compares a direct global BinaryConfusionVectors reduction against the
    bounded-memory per-image Measures -> MeasureCombiner strategy.  The latter
    mirrors the scalable accumulation pattern in ``segmentation_metrics.py``.
    """
    import kwarray
    from kwcoco.metrics.confusion_measures import MeasureCombiner
    from kwcoco.metrics.confusion_vectors import BinaryConfusionVectors

    gids = sorted(set(true_dset.imgs) & set(pred_dset.imgs))
    true_arrays, true_raster_timing = timed_call(
        lambda: _instance_saliency_arrays(true_dset, gids, prediction=False)
    )
    pred_arrays, pred_raster_timing = timed_call(
        lambda: _instance_saliency_arrays(pred_dset, gids, prediction=True)
    )
    total_pixels = int(sum(a.size for a in true_arrays))

    left_bin_edges = np.linspace(0, 1, thresh_bins)

    def _quantize(scores):
        idx = np.searchsorted(left_bin_edges, scores)
        idx = np.clip(idx, 0, len(left_bin_edges) - 1)
        return left_bin_edges[idx]

    def _online_accumulate():
        combiner = MeasureCombiner(thresh_bins=thresh_bins)
        for truth, probs in zip(true_arrays, pred_arrays):
            pred_score = _quantize(probs.ravel())
            data = kwarray.DataFrameArray({
                'is_true': truth.ravel(),
                'pred_score': pred_score,
                'weight': np.ones(truth.size, dtype=np.float32),
            })
            binvecs = BinaryConfusionVectors(data)
            combiner.submit(binvecs.measures())
            if combiner.queue_size > 5:
                combiner.combine()
        return combiner.finalize()

    online_measures, online_timing = timed_call(_online_accumulate)
    result = {
        'total_pixels': total_pixels,
        'thresh_bins': thresh_bins,
        'truth_raster_timing': true_raster_timing,
        'prediction_raster_timing': pred_raster_timing,
        'online_measure_combiner': {
            'summary': _pixel_measure_summary(online_measures),
            'timing': online_timing,
        },
    }

    if total_pixels <= global_pixel_limit:
        def _global_accumulate(quantized):
            truth = np.concatenate([a.ravel() for a in true_arrays])
            probs = np.concatenate([a.ravel() for a in pred_arrays])
            if quantized:
                probs = _quantize(probs)
            data = kwarray.DataFrameArray({
                'is_true': truth,
                'pred_score': probs,
                'weight': np.ones(len(truth), dtype=np.float32),
            })
            return BinaryConfusionVectors(data).measures()

        global_raw, global_raw_timing = timed_call(
            lambda: _global_accumulate(quantized=False)
        )
        global_quant, global_quant_timing = timed_call(
            lambda: _global_accumulate(quantized=True)
        )
        raw_summary = _pixel_measure_summary(global_raw)
        quant_summary = _pixel_measure_summary(global_quant)
        online_summary = result['online_measure_combiner']['summary']
        result['direct_global'] = {
            'status': 'measured',
            'unquantized': {
                'summary': raw_summary,
                'timing': global_raw_timing,
            },
            'quantized': {
                'summary': quant_summary,
                'timing': global_quant_timing,
            },
            'delta_quantized_minus_unquantized': {
                key: quant_summary[key] - raw_summary[key]
                for key in ['ap', 'auc', 'max_f1']
            },
            'delta_online_minus_quantized_global': {
                key: online_summary[key] - quant_summary[key]
                for key in ['ap', 'auc', 'max_f1']
            },
        }
    else:
        result['direct_global'] = {
            'status': 'skipped-pixel-limit',
            'global_pixel_limit': int(global_pixel_limit),
        }
    return result

def dataset_summary(dset):
    return {
        'n_images': dset.n_images,
        'n_annots': dset.n_annots,
        'n_categories': dset.n_cats,
    }


def run_case(case_name, profile, seed, repeat, kernel_repeat):
    true_dset, pred_dset, spec = make_case(case_name, profile, seed)
    result = {
        'name': case_name,
        'spec': spec,
        'truth': dataset_summary(true_dset),
        'pred': dataset_summary(pred_dset),
    }
    result['metrics'] = compare_metric_layers(
        true_dset, pred_dset, repeat=repeat
    )
    result['mask_kernels'] = compare_mask_kernels(
        true_dset,
        pred_dset,
        limit=profile['kernel_limit'],
        repeat=kernel_repeat,
    )
    result['dense_saliency'] = compare_dense_saliency_accumulation(
        true_dset,
        pred_dset,
        thresh_bins=1024,
        global_pixel_limit=profile['pixel_global_limit'],
    )
    return result


def _print_summary(report):
    print('\nMetric decomposition at IoU=0.5')
    header = (
        'case',
        'bbox native/native',
        'bbox official/native',
        'bbox official/official',
        'segm official/native',
        'segm official/official',
    )
    print(' | '.join(header))
    print(' | '.join(['---'] * len(header)))
    for case in report['cases']:
        bbox = case['metrics']['bbox']['metrics']
        segm = case['metrics']['segm']['metrics']
        row = [
            case['name'],
            f"{bbox['native_native_mAP50']:.6f}",
            f"{bbox['official_native_mAP50']:.6f}",
            f"{bbox['official_official_AP50']:.6f}",
            f"{segm['official_native_mAP50']:.6f}",
            f"{segm['official_official_AP50']:.6f}",
        ]
        print(' | '.join(row))

    print('\nMask kernel comparison')
    for case in report['cases']:
        info = case['mask_kernels']
        print(f"  {case['name']}:")
        for backend_name, backend_info in info.get('backends', {}).items():
            status = backend_info.get('status', 'unknown')
            if status == 'measured':
                median = backend_info['timing']['median_seconds']
                print(
                    f"    {backend_name}: measured "
                    f"(median={median:.6g}s)"
                )
            else:
                reason = backend_info.get('reason', 'unavailable')
                print(f"    {backend_name}: SKIPPED ({reason})")
        comparison = info.get('comparisons', {}).get(
            'kwimage_ext_vs_pycocotools', {}
        )
        if comparison:
            print(
                '    kwimage_ext_vs_pycocotools: ' +
                json.dumps(comparison, default=_jsonify_scalar)
            )

    print('\nDense saliency confusion accumulation')
    for case in report['cases']:
        info = case['dense_saliency']
        online = info['online_measure_combiner']['summary']
        direct = info['direct_global']
        print(
            f"  {case['name']}: online AP={online['ap']:.6f}, "
            f"direct={direct.get('quantized', {}).get('summary', {}).get('ap', direct['status'])}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=sorted(PROFILES), default='smoke')
    parser.add_argument('--cases', nargs='*', default=list(CASE_SPECS))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--kernel-repeat', type=int, default=3)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args(argv)

    unknown = set(args.cases) - set(CASE_SPECS)
    if unknown:
        raise KeyError(f'Unknown cases: {sorted(unknown)}')

    profile = dict(PROFILES[args.profile])
    report = {
        'schema_version': 1,
        'profile_name': args.profile,
        'profile': profile,
        'seed': args.seed,
        'environment': environment_info(),
        'cases': [],
    }

    for case_index, case_name in enumerate(args.cases):
        print(f'Running case {case_name!r}')
        case_result = run_case(
            case_name,
            profile,
            seed=args.seed + case_index * 1009,
            repeat=args.repeat,
            kernel_repeat=args.kernel_repeat,
        )
        report['cases'].append(case_result)

    _print_summary(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as file:
            json.dump(report, file, indent=2, default=_jsonify_scalar)
            file.write('\n')
        print(f'Wrote {args.output}')
    return report


if __name__ == '__main__':
    main()
