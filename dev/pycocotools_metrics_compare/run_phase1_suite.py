#!/usr/bin/env python3
"""
Phase-1 metric comparison suite.

This is measurement code, not a production evaluator.  It extends the baseline
``compare_metrics.py`` experiment with:

* native assignment v1 versus experimental assignment_v2 comparisons;
* reusable multi-IoU candidate-index timing for assignment_v2;
* controlled image-count, resolution, and object-density sweeps;
* deterministic adversarial matching fixtures;
* explicit backend / source provenance;
* a single uploadable result bundle.

The goal is to characterize behavior before native instance-mask assignment is
implemented.
"""
from __future__ import annotations

import argparse
import collections
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

# Keep this development module importable both as a direct script and through
# importlib from tests.  dev/ is intentionally not a package.
SCRIPT_DPATH = Path(__file__).resolve().parent
if os.fspath(SCRIPT_DPATH) not in sys.path:
    sys.path.insert(0, os.fspath(SCRIPT_DPATH))
import compare_metrics as baseline


IOU_THRESHOLDS = tuple(float(x) for x in np.arange(0.50, 0.951, 0.05))


QUICK_SWEEPS = {
    'images': [4, 16, 64],
    'resolution': [96, 256, 512],
    'tracks': [2, 8, 32],
}

FULL_SWEEPS = {
    'images': [4, 16, 64, 256],
    'resolution': [96, 256, 512, 1024],
    'tracks': [2, 8, 32, 128],
}


RANDOM_CASES = ('perfect', 'box_noise', 'class_noise', 'mask_only_shift', 'mixed')
ADVERSARIAL_CASES = (
    'iou_050_boundary',
    'score_tie_duplicate',
    'one_pred_two_truths',
    'empty_prediction',
    'empty_truth',
    'hole_mismatch',
)


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return os.fspath(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    raise TypeError(type(value))


def _run_text_command(command, cwd=None):
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except Exception as ex:
        return {'returncode': None, 'output': None, 'error': repr(ex)}
    return {'returncode': proc.returncode, 'output': proc.stdout.strip()}


def source_state():
    import kwcoco

    repo_dpath = Path(kwcoco.__file__).resolve().parent.parent
    return {
        'repo_dpath': os.fspath(repo_dpath),
        'git_head': _run_text_command(['git', 'rev-parse', 'HEAD'], cwd=repo_dpath),
        'git_status': _run_text_command(['git', 'status', '--short'], cwd=repo_dpath),
        'git_diff_stat': _run_text_command(['git', 'diff', '--stat'], cwd=repo_dpath),
    }


def _memory_snapshot():
    info = {}
    try:
        import psutil
    except Exception:
        pass
    else:
        try:
            info['rss_bytes'] = int(psutil.Process().memory_info().rss)
        except Exception:
            pass
    try:
        import resource
    except Exception:
        pass
    else:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        value = float(usage.ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        if platform.system() == 'Darwin':
            info['maxrss_bytes'] = int(value)
        else:
            info['maxrss_bytes'] = int(value * 1024)
    return info


def timed_with_memory(func, repeat=1):
    times = []
    result = None
    before = _memory_snapshot()
    for _ in range(repeat):
        start = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - start)
    after = _memory_snapshot()
    timing = {
        'repeat': int(repeat),
        'min_seconds': float(min(times)),
        'median_seconds': float(np.median(times)),
        'max_seconds': float(max(times)),
        'all_seconds': [float(t) for t in times],
        'memory_before': before,
        'memory_after': after,
    }
    for key in ['rss_bytes', 'maxrss_bytes']:
        if key in before and key in after:
            timing[key + '_delta'] = int(after[key] - before[key])
    return result, timing


def _coerce_confusion_data(accum):
    import kwarray

    data = {}
    for key, values in accum.items():
        dtype = None
        if key in {'iou', 'score', 'weight'}:
            dtype = np.float32
        elif key in {'pxs', 'txs', 'gid', 'pred', 'true', 'rank'}:
            dtype = np.int32
        try:
            data[key] = np.asarray(values, dtype=dtype)
        except (TypeError, ValueError):
            data[key] = np.asarray(values)
    if not data:
        data = {
            key: np.empty(0, dtype=np.int32)
            for key in ['gid', 'pred', 'true', 'txs', 'pxs']
        }
        for key in ['score', 'weight', 'iou']:
            data[key] = np.empty(0, dtype=np.float32)
    return kwarray.DataFrameArray(data)


def _confusion_vectors_v2(
    dmet,
    iou_thresholds,
    *,
    spatial_threshold=50_000,
    compat='mutex',
):
    """Build dataset-level ConfusionVectors with assignment_v2."""
    import ubelt as ub
    from kwcoco.metrics.assignment_v2 import AssignmentCandidateIndex
    from kwcoco.metrics.confusion_vectors import ConfusionVectors

    thresholds = [float(t) for t in iou_thresholds]
    min_iou = min(thresholds)
    gids = sorted(dmet.gid_to_true_dets)
    threshold_to_accum = {t: ub.ddict(list) for t in thresholds}
    aggregate_stats = collections.Counter()

    for gid in gids:
        true_dets = dmet.true_detections(gid)
        pred_dets = dmet.pred_detections(gid)
        index = AssignmentCandidateIndex.build(
            true_dets,
            pred_dets,
            classes=dmet.classes,
            compat=compat,
            prioritize='iou',
            bias=0.0,
            min_iou=min_iou,
            spatial_threshold=spatial_threshold,
        )
        aggregate_stats.update(index.build_stats)
        threshold_to_y = index.assign_many(
            thresholds,
            bg_weight=1.0,
            bg_cidx=-1,
            ignore_classes='ignore',
            max_dets=None,
            truth_reuse_policy='never',
        )
        for threshold, y in threshold_to_y.items():
            accum = threshold_to_accum[threshold]
            row_count = len(y['pred'])
            for key, values in y.items():
                accum[key].extend(values)
            accum['gid'].extend([gid] * row_count)

    threshold_to_cfsn = {}
    for threshold, accum in threshold_to_accum.items():
        cfsn_data = _coerce_confusion_data(accum)
        threshold_to_cfsn[threshold] = ConfusionVectors(
            cfsn_data,
            classes=dmet.classes,
            probs=None,
        )
    return threshold_to_cfsn, dict(aggregate_stats)


def _row_counter(cfsn):
    data = cfsn.data
    keys = ['gid', 'txs', 'pxs', 'true', 'pred']
    arrays = [np.asarray(data[key]) for key in keys]
    return collections.Counter(
        tuple(int(v) for v in row)
        for row in zip(*arrays)
    )


def _row_diff(left, right, example_limit=20):
    left_rows = _row_counter(left)
    right_rows = _row_counter(right)
    left_only = left_rows - right_rows
    right_only = right_rows - left_rows
    return {
        'num_left_rows': int(sum(left_rows.values())),
        'num_right_rows': int(sum(right_rows.values())),
        'num_left_only': int(sum(left_only.values())),
        'num_right_only': int(sum(right_only.values())),
        'left_only_examples': list(left_only.elements())[:example_limit],
        'right_only_examples': list(right_only.elements())[:example_limit],
        'exact_multiset_match': not left_only and not right_only,
    }


def compare_assignment_v1_v2(true_dset, pred_dset, repeat=1):
    """Compare native assignment implementations without changing semantics."""
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dmet, build_timing = timed_with_memory(
        lambda: DetectionMetrics.from_coco(true_dset, pred_dset),
        repeat=repeat,
    )

    def _v1_multi():
        return dmet.confusion_vectors(
            iou_thresh=IOU_THRESHOLDS,
            compat='mutex',
            prioritize='iou',
            workers=0,
            track_probs=False,
            truth_reuse_policy=False,
        )

    def _v2_multi_dense():
        return _confusion_vectors_v2(
            dmet,
            IOU_THRESHOLDS,
            spatial_threshold=None,
            compat='mutex',
        )

    def _v2_multi_auto():
        return _confusion_vectors_v2(
            dmet,
            IOU_THRESHOLDS,
            spatial_threshold=50_000,
            compat='mutex',
        )

    v1_map, v1_timing = timed_with_memory(_v1_multi, repeat=repeat)
    (v2_dense_map, v2_dense_stats), v2_dense_timing = timed_with_memory(
        _v2_multi_dense, repeat=repeat
    )
    (v2_auto_map, v2_auto_stats), v2_auto_timing = timed_with_memory(
        _v2_multi_auto, repeat=repeat
    )

    threshold_diffs = {}
    for threshold in IOU_THRESHOLDS:
        threshold_diffs[str(threshold)] = {
            'v1_vs_v2_dense': _row_diff(
                v1_map[threshold], v2_dense_map[threshold]
            ),
            'v2_dense_vs_auto': _row_diff(
                v2_dense_map[threshold], v2_auto_map[threshold]
            ),
        }

    t50 = 0.5
    v1_measures = baseline._native_measures_from_cfsn(v1_map[t50])
    v2_dense_measures = baseline._native_measures_from_cfsn(v2_dense_map[t50])
    v2_auto_measures = baseline._native_measures_from_cfsn(v2_auto_map[t50])

    return {
        'dataset_build_timing': build_timing,
        'iou_thresholds': list(IOU_THRESHOLDS),
        'metrics_at_0_5': {
            'v1': v1_measures,
            'v2_dense': v2_dense_measures,
            'v2_auto': v2_auto_measures,
        },
        'timing': {
            'v1_multi_iou': v1_timing,
            'v2_dense_multi_iou': v2_dense_timing,
            'v2_auto_multi_iou': v2_auto_timing,
        },
        'v2_build_stats': {
            'dense': v2_dense_stats,
            'auto': v2_auto_stats,
        },
        'threshold_diffs': threshold_diffs,
    }


def _new_dataset_with_schema(width=128, height=128):
    import kwcoco

    dset = kwcoco.CocoDataset()
    cid = dset.add_category(name='object')
    gid = dset.add_image(
        name='synthetic',
        file_name='synthetic.png',
        width=int(width),
        height=int(height),
    )
    return dset, gid, cid


def _rect_seg(x, y, w, h):
    return {
        'exterior': [
            [float(x), float(y)],
            [float(x + w), float(y)],
            [float(x + w), float(y + h)],
            [float(x), float(y + h)],
        ],
        'interiors': [],
    }


def _add_rect(dset, gid, cid, bbox, score=None, segmentation=None, **extra):
    x, y, w, h = map(float, bbox)
    if segmentation is None:
        segmentation = _rect_seg(x, y, w, h)
    ann = {
        'image_id': gid,
        'category_id': cid,
        'bbox': [x, y, w, h],
        'segmentation': segmentation,
        'area': float(w * h),
        **extra,
    }
    if score is not None:
        ann['score'] = float(score)
    return dset.add_annotation(**ann)


def make_adversarial_case(name):
    import kwcoco

    true, gid, cid = _new_dataset_with_schema()
    pred = kwcoco.CocoDataset()
    pred.add_category(name='object', id=cid)
    pred.add_image(
        name='synthetic', file_name='synthetic.png', width=128, height=128, id=gid
    )

    if name == 'iou_050_boundary':
        _add_rect(true, gid, cid, [10, 10, 20, 20])
        _add_rect(pred, gid, cid, [10, 10, 10, 20], score=0.9)
    elif name == 'score_tie_duplicate':
        _add_rect(true, gid, cid, [20, 20, 30, 30])
        _add_rect(pred, gid, cid, [20, 20, 30, 30], score=0.9)
        _add_rect(pred, gid, cid, [20, 20, 30, 30], score=0.9)
    elif name == 'one_pred_two_truths':
        _add_rect(true, gid, cid, [20, 20, 30, 30])
        _add_rect(true, gid, cid, [24, 20, 30, 30])
        _add_rect(pred, gid, cid, [22, 20, 30, 30], score=0.9)
    elif name == 'empty_prediction':
        _add_rect(true, gid, cid, [20, 20, 30, 30])
    elif name == 'empty_truth':
        _add_rect(pred, gid, cid, [20, 20, 30, 30], score=0.9)
    elif name == 'hole_mismatch':
        exterior = [
            [20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]
        ]
        hole = [
            [40.0, 40.0], [60.0, 40.0], [60.0, 60.0], [40.0, 60.0]
        ]
        true_seg = {'exterior': exterior, 'interiors': [hole]}
        pred_seg = {'exterior': exterior, 'interiors': []}
        _add_rect(
            true, gid, cid, [20, 20, 60, 60], segmentation=true_seg
        )
        _add_rect(
            pred, gid, cid, [20, 20, 60, 60], segmentation=pred_seg, score=0.9
        )
    else:
        raise KeyError(name)

    true._build_index()
    pred._build_index()
    return true, pred, {'builder': 'handcrafted', 'case': name}


def make_vidshapes_case(nimgs, image_size, num_tracks, seed, case_name='mixed'):
    import kwcoco
    from kwcoco.demo.perterb import perterb_coco

    spec = dict(baseline.CASE_SPECS[case_name])
    true_dset = kwcoco.CocoDataset.demo(
        'vidshapes1',
        num_frames=int(nimgs),
        num_tracks=int(num_tracks),
        image_size=tuple(image_size),
        render=False,
        rng=seed,
        verbose=0,
    )
    pred_dset = perterb_coco(
        true_dset,
        rng=seed + 1,
        box_noise=spec['box_noise'],
        cls_noise=spec['cls_noise'],
        score_noise=spec['score_noise'],
        n_fp=spec['n_fp'],
        n_fn=spec['n_fn'],
        with_probs=False,
        verbose=0,
    )
    pred_dset = baseline._translate_prediction_masks(
        pred_dset, spec['mask_shift']
    )
    spec.update({
        'builder': 'vidshapes',
        'nimgs': int(nimgs),
        'num_tracks': int(num_tracks),
        'image_size': list(map(int, image_size)),
    })
    return true_dset, pred_dset, spec


def evaluate_dataset_pair(
    true_dset,
    pred_dset,
    spec,
    *,
    repeat=1,
    kernel_repeat=2,
    include_reference=True,
    include_kernels=False,
    include_dense=False,
    pixel_global_limit=1_000_000,
):
    result = {
        'spec': spec,
        'truth': baseline.dataset_summary(true_dset),
        'pred': baseline.dataset_summary(pred_dset),
    }
    result['assignment_v1_v2'] = compare_assignment_v1_v2(
        true_dset, pred_dset, repeat=repeat
    )
    if include_reference:
        result['reference_metrics'] = baseline.compare_metric_layers(
            true_dset, pred_dset, repeat=repeat
        )
    if include_kernels:
        result['mask_kernels'] = baseline.compare_mask_kernels(
            true_dset, pred_dset, limit=96, repeat=kernel_repeat
        )
    if include_dense:
        result['dense_saliency'] = baseline.compare_dense_saliency_accumulation(
            true_dset,
            pred_dset,
            thresh_bins=1024,
            global_pixel_limit=pixel_global_limit,
        )
    return result


def _safe_experiment(name, func):
    print(f'\n=== {name} ===', flush=True)
    started = time.time()
    try:
        payload = func()
    except Exception as ex:
        traceback_text = traceback.format_exc()
        print(traceback_text, flush=True)
        return {
            'name': name,
            'status': 'error',
            'elapsed_seconds': time.time() - started,
            'error': {
                'type': type(ex).__name__,
                'message': str(ex),
                'repr': repr(ex),
                'traceback': traceback_text,
            },
        }
    else:
        return {
            'name': name,
            'status': 'ok',
            'elapsed_seconds': time.time() - started,
            'result': payload,
        }


def _random_case_experiment(case_name, profile, seed, repeat, kernel_repeat):
    def work():
        true_dset, pred_dset, spec = baseline.make_case(
            case_name, profile, seed
        )
        return evaluate_dataset_pair(
            true_dset,
            pred_dset,
            spec,
            repeat=repeat,
            kernel_repeat=kernel_repeat,
            include_reference=True,
            include_kernels=True,
            include_dense=True,
            pixel_global_limit=profile['pixel_global_limit'],
        )
    return work


def _adversarial_experiment(case_name, repeat):
    def work():
        true_dset, pred_dset, spec = make_adversarial_case(case_name)
        return evaluate_dataset_pair(
            true_dset,
            pred_dset,
            spec,
            repeat=repeat,
            include_reference=True,
            include_kernels=False,
            include_dense=False,
        )
    return work


def _scaling_shape_experiment(case_name, profile, seed, repeat):
    def work():
        true_dset, pred_dset, spec = baseline.make_case(
            case_name, profile, seed
        )
        spec = dict(spec)
        spec['profile'] = dict(profile)
        return evaluate_dataset_pair(
            true_dset,
            pred_dset,
            spec,
            repeat=repeat,
            include_reference=True,
            include_kernels=False,
            include_dense=False,
        )
    return work


def _density_experiment(nimgs, image_size, tracks, seed, repeat):
    def work():
        true_dset, pred_dset, spec = make_vidshapes_case(
            nimgs=nimgs,
            image_size=image_size,
            num_tracks=tracks,
            seed=seed,
            case_name='mixed',
        )
        return evaluate_dataset_pair(
            true_dset,
            pred_dset,
            spec,
            repeat=repeat,
            include_reference=True,
            include_kernels=False,
            include_dense=False,
        )
    return work


def _summary_lines(report):
    lines = [
        '# kwcoco metric Phase-1 comparison summary',
        '',
        f"Mode: `{report['mode']}`",
        f"Schema: `{report['schema_version']}`",
        '',
        'Upload the generated `.tar.gz` bundle for analysis.',
        '',
        '## Experiment status',
        '',
        '| Experiment | Status | Seconds |',
        '| --- | --- | ---: |',
    ]
    for item in report['experiments']:
        lines.append(
            f"| `{item['name']}` | {item['status']} | "
            f"{item['elapsed_seconds']:.3f} |"
        )
    lines.extend([
        '',
        '## Assignment observations',
        '',
        '| Experiment | v1/v2 exact @0.50 | v1 sec | v2 dense sec | v2 auto sec |',
        '| --- | --- | ---: | ---: | ---: |',
    ])
    for item in report['experiments']:
        if item['status'] != 'ok':
            continue
        info = item['result'].get('assignment_v1_v2')
        if not info:
            continue
        diff = info['threshold_diffs'].get('0.5')
        if diff is None:
            continue
        exact = diff['v1_vs_v2_dense']['exact_multiset_match']
        timing = info['timing']
        lines.append(
            f"| `{item['name']}` | {exact} | "
            f"{timing['v1_multi_iou']['median_seconds']:.6f} | "
            f"{timing['v2_dense_multi_iou']['median_seconds']:.6f} | "
            f"{timing['v2_auto_multi_iou']['median_seconds']:.6f} |"
        )
    return lines


def build_suite(mode, repeat, kernel_repeat):
    if mode == 'quick':
        sweep = QUICK_SWEEPS
        baseline_profile = dict(baseline.PROFILES['smoke'])
    elif mode == 'full':
        sweep = FULL_SWEEPS
        baseline_profile = dict(baseline.PROFILES['small'])
    else:
        raise KeyError(mode)

    report = {
        'schema_version': 2,
        'mode': mode,
        'created_unix': time.time(),
        'environment': baseline.environment_info(),
        'source_state': source_state(),
        'config': {
            'repeat': int(repeat),
            'kernel_repeat': int(kernel_repeat),
            'iou_thresholds': list(IOU_THRESHOLDS),
            'sweeps': sweep,
        },
        'experiments': [],
    }

    seed = 0
    for case_name in RANDOM_CASES:
        name = f'baseline/{case_name}'
        report['experiments'].append(_safe_experiment(
            name,
            _random_case_experiment(
                case_name,
                baseline_profile,
                seed=seed,
                repeat=repeat,
                kernel_repeat=kernel_repeat,
            ),
        ))
        seed += 1009

    for case_name in ADVERSARIAL_CASES:
        name = f'adversarial/{case_name}'
        report['experiments'].append(_safe_experiment(
            name,
            _adversarial_experiment(case_name, repeat=repeat),
        ))

    # Image-count sweep: keep resolution fixed.
    for nimgs in sweep['images']:
        profile = {
            'nimgs': int(nimgs),
            'image_size': (256, 256),
            'kernel_limit': 0,
            'pixel_global_limit': 0,
        }
        name = f'scaling/images_{nimgs}'
        report['experiments'].append(_safe_experiment(
            name,
            _scaling_shape_experiment(
                'mixed', profile, seed=30000 + int(nimgs), repeat=repeat
            ),
        ))

    # Resolution sweep: this exercises mask conversion / COCOeval geometry cost.
    for size in sweep['resolution']:
        profile = {
            'nimgs': 8 if mode == 'quick' else 16,
            'image_size': (int(size), int(size)),
            'kernel_limit': 0,
            'pixel_global_limit': 0,
        }
        name = f'scaling/resolution_{size}'
        report['experiments'].append(_safe_experiment(
            name,
            _scaling_shape_experiment(
                'mixed', profile, seed=40000 + int(size), repeat=repeat
            ),
        ))

    # Object-density sweep: vidshapes exposes explicit track count.
    for tracks in sweep['tracks']:
        name = f'scaling/tracks_{tracks}'
        report['experiments'].append(_safe_experiment(
            name,
            _density_experiment(
                nimgs=8 if mode == 'quick' else 16,
                image_size=(256, 256),
                tracks=int(tracks),
                seed=50000 + int(tracks),
                repeat=repeat,
            ),
        ))

    return report


def write_bundle(report, output_dpath):
    output_dpath = Path(output_dpath)
    output_dpath.mkdir(parents=True, exist_ok=True)
    mode = report['mode']
    result_fpath = output_dpath / f'phase1_{mode}_results.json'
    summary_fpath = output_dpath / f'phase1_{mode}_summary.md'
    source_fpath = output_dpath / f'phase1_{mode}_source_state.json'

    result_fpath.write_text(
        json.dumps(report, indent=2, default=_json_default) + '\n'
    )
    summary_fpath.write_text('\n'.join(_summary_lines(report)) + '\n')
    source_fpath.write_text(
        json.dumps(report['source_state'], indent=2, default=_json_default) + '\n'
    )

    bundle_fpath = output_dpath / f'kwcoco_metrics_phase1_{mode}.tar.gz'
    with tarfile.open(bundle_fpath, 'w:gz') as tar:
        for fpath in [result_fpath, summary_fpath, source_fpath]:
            tar.add(fpath, arcname=fpath.name)
    return {
        'result_fpath': result_fpath,
        'summary_fpath': summary_fpath,
        'source_fpath': source_fpath,
        'bundle_fpath': bundle_fpath,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    parser.add_argument('--repeat', type=int, default=None)
    parser.add_argument('--kernel-repeat', type=int, default=None)
    parser.add_argument(
        '--output-dpath',
        type=Path,
        default=Path('_metric_compare_phase1'),
    )
    args = parser.parse_args(argv)

    if args.repeat is None:
        repeat = 1 if args.mode == 'quick' else 3
    else:
        repeat = args.repeat
    if args.kernel_repeat is None:
        kernel_repeat = 2 if args.mode == 'quick' else 5
    else:
        kernel_repeat = args.kernel_repeat

    report = build_suite(
        mode=args.mode,
        repeat=repeat,
        kernel_repeat=kernel_repeat,
    )
    paths = write_bundle(report, args.output_dpath)
    print('\nPhase-1 suite complete')
    for key, path in paths.items():
        print(f'  {key}: {path}')
    errors = [x for x in report['experiments'] if x['status'] != 'ok']
    print(f'  experiments: {len(report["experiments"])}')
    print(f'  errors: {len(errors)}')
    if errors:
        print('  Error experiments were retained in the bundle:')
        for item in errors:
            print(f'    - {item["name"]}: {item["error"]["type"]}: {item["error"]["message"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
