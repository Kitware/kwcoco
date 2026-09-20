#!/usr/bin/env python3
"""Profile native COCO scoring against the official pycocotools reference.

This is a performance diagnostic, not a conformance gate.  The Phase-3
conformance runner proves semantics; this script isolates scorer runtime after
``DetectionMetrics`` construction and supplies repeat/warmup timing plus
optional cProfile, line-profiler, and tracemalloc views.

The synthetic pressure cases are intentionally different from the conformance
corpus:

* ``maxdet-pressure`` creates many predictions in one category so native work
  performed beyond COCO's largest requested maxDet becomes visible.
* ``candidate-heavy`` creates many mutually-overlapping detections so the
  greedy assignment inner loop has longer candidate lists.

Examples:
    python dev/pycocotools_metrics_compare/profile_native_coco.py \
        --scenario perfect --dataset-mode full --iou-type bbox \
        --with-eval --repeat 7 --warmup 2

    python dev/pycocotools_metrics_compare/profile_native_coco.py \
        --scenario maxdet-pressure --iou-type bbox --with-eval \
        --num-objects 1000 --repeat 7 --warmup 2 --profile cprofile

    # Optional dependency:
    # uv pip install line-profiler
    python dev/pycocotools_metrics_compare/profile_native_coco.py \
        --scenario perfect --dataset-mode full --iou-type bbox \
        --with-eval --repeat 5 --profile line
"""
from __future__ import annotations

import argparse
import cProfile
import gc
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import pstats
import statistics
import sys
import time
import tracemalloc

import numpy as np

from run_phase3_coco_conformance import PROFILES
from run_phase3_coco_conformance import _case_data
from run_phase3_coco_conformance import _tensor_diff


SCENARIOS = {
    'perfect',
    'mixed',
    'maxdet-pressure',
    'candidate-heavy',
}
IOU_TYPES = {'bbox', 'segm'}


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _json_default(obj):
    if isinstance(obj, Path):
        return os.fspath(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    raise TypeError(type(obj))


def _rect_poly(x, y, w, h):
    return [
        float(x), float(y),
        float(x + w), float(y),
        float(x + w), float(y + h),
        float(x), float(y + h),
    ]


def _pressure_case(kind, num_objects, seed=0):
    """Construct one-image high-object-count cases outside the parity corpus."""
    import kwcoco

    rng = np.random.RandomState(seed)
    if kind == 'maxdet-pressure':
        num_true = min(100, max(1, num_objects // 4))
        num_pred = num_objects
        side = int(math.ceil(math.sqrt(num_true)))
        cell = 18
        box_wh = 10
        width = max(256, side * cell + 16)
        height = max(256, side * cell + 16)
    elif kind == 'candidate-heavy':
        num_true = num_objects
        num_pred = num_objects
        width = height = 256
        box_wh = 96
    else:  # pragma: no cover
        raise KeyError(kind)

    true = kwcoco.CocoDataset()
    pred = kwcoco.CocoDataset()
    for dset in [true, pred]:
        dset.add_category('object', id=1)
        dset.add_image('image-1', id=1, width=width, height=height)

    truth_boxes = []
    for tx in range(num_true):
        if kind == 'maxdet-pressure':
            row, col = divmod(tx, side)
            x = 8 + col * cell
            y = 8 + row * cell
            w = h = box_wh
        else:
            # Long candidate lists: every box occupies nearly the same region,
            # with tiny deterministic offsets to avoid a completely degenerate
            # fixture while retaining heavy mutual overlap.
            x = 64 + (tx % 5) * 0.2
            y = 64 + (tx % 7) * 0.2
            w = h = box_wh
        bbox = [float(x), float(y), float(w), float(h)]
        truth_boxes.append(bbox)
        true.add_annotation(
            id=tx + 1,
            image_id=1,
            category_id=1,
            bbox=bbox,
            segmentation=[_rect_poly(x, y, w, h)],
        )

    for px in range(num_pred):
        if px < num_true:
            x, y, w, h = truth_boxes[px]
        elif kind == 'maxdet-pressure':
            # Low-score extras are deliberately outside the top maxDet=100.
            base = truth_boxes[px % num_true]
            x, y, w, h = base
            x += 1.5 + (px % 3) * 0.25
            y += 1.0 + (px % 5) * 0.15
        else:
            x = 64 + (px % 11) * 0.15
            y = 64 + (px % 13) * 0.15
            w = h = box_wh
        # Strictly descending by construction.  A tiny deterministic jitter on
        # extras avoids benchmarking score-tie handling instead of maxDet work.
        score = 1.0 - (px / max(1, num_pred + 1))
        score -= float(rng.uniform(0, 1e-9))
        pred.add_annotation(
            id=px + 1,
            image_id=1,
            category_id=1,
            bbox=[float(x), float(y), float(w), float(h)],
            segmentation=[_rect_poly(x, y, w, h)],
            score=float(score),
        )

    spec = {
        'kind': kind,
        'num_truth': num_true,
        'num_predictions': num_pred,
        'image_size': [height, width],
        'seed': seed,
    }
    return true, pred, spec


def _make_scenario(name, dataset_mode, num_objects, seed):
    if name in {'perfect', 'mixed'}:
        profile = PROFILES[dataset_mode]
        true, pred, spec = _case_data(name, profile, seed)
        spec = dict(spec)
        spec['dataset_mode'] = dataset_mode
        return true, pred, spec
    return _pressure_case(name, num_objects=num_objects, seed=seed)


def _dataset_counts(dset):
    return {
        'images': len(dset.imgs),
        'annotations': len(dset.anns),
        'categories': len(dset.cats),
    }


def _native_call(dmet, iou_type, with_eval, assignment_backend='auto'):
    kwargs = {
        'iou_type': iou_type,
        'with_eval': with_eval,
        'assignment_backend': assignment_backend,
    }
    if iou_type == 'segm':
        kwargs.update({
            'overlap_backend': 'auto',
            'raster_backend': 'auto',
        })
    return dmet.score_native_coco(**kwargs)


def _official_call(dmet, iou_type, with_eval):
    return dmet.score_pycocotools(
        iou_type=iou_type,
        with_eval=with_eval,
        bbox_annotation_policy='coco',
        verbose=0,
    )


def _time_call(func, warmup, repeat):
    for _ in range(warmup):
        func()
    samples = []
    result = None
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        result = func()
        samples.append(time.perf_counter() - start)
    return result, {
        'warmup': int(warmup),
        'repeat': int(repeat),
        'min_seconds': float(min(samples)),
        'median_seconds': float(statistics.median(samples)),
        'mean_seconds': float(statistics.mean(samples)),
        'max_seconds': float(max(samples)),
        'all_seconds': list(map(float, samples)),
    }


def _measure_tracemalloc(func):
    gc.collect()
    tracemalloc.start()
    try:
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        'elapsed_seconds_with_tracemalloc': float(elapsed),
        'python_current_bytes': int(current),
        'python_peak_bytes': int(peak),
        'note': 'tracemalloc observes Python-managed allocations, not full C/Rust RSS',
    }


def _write_cprofile(func, output_base):
    profile = cProfile.Profile()
    profile.enable()
    func()
    profile.disable()

    binary_fpath = output_base.with_suffix('.prof')
    text_fpath = output_base.with_suffix('.cprofile.txt')
    profile.dump_stats(os.fspath(binary_fpath))
    with text_fpath.open('w') as stream:
        stream.write('=== cumulative time ===\n')
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats(
            'cumulative'
        ).print_stats(80)
        stream.write('\n=== internal time ===\n')
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats(
            'tottime'
        ).print_stats(80)
    return {
        'binary': os.fspath(binary_fpath),
        'text': os.fspath(text_fpath),
    }


def _line_profile_targets():
    from kwcoco.metrics.assignment_v2 import AssignmentCandidateIndex
    from kwcoco.metrics.assignment_v2 import _coco_bbox_overlaps
    from kwcoco.metrics.assignment_v2 import _populate_sparse_coco_bbox_candidates
    from kwcoco.metrics.coco_metrics import accumulate_coco_assignment_rows
    from kwcoco.metrics.detect_metrics import DetectionMetrics
    from kwcoco.metrics.overlap import detections_to_rles
    from kwcoco.metrics.overlap import rle_areas

    return [
        DetectionMetrics.score_native_coco,
        AssignmentCandidateIndex.build.__func__,
        AssignmentCandidateIndex.assign,
        AssignmentCandidateIndex.assign_many,
        _coco_bbox_overlaps,
        _populate_sparse_coco_bbox_candidates,
        accumulate_coco_assignment_rows,
        detections_to_rles,
        rle_areas,
    ]


def _write_line_profile(func, output_base):
    try:
        import line_profiler
    except ImportError as ex:
        raise RuntimeError(
            'line-profiler is optional. Install it into this environment with '
            '`uv pip install line-profiler` and rerun with --profile line/both.'
        ) from ex

    lp = line_profiler.LineProfiler()
    for target in _line_profile_targets():
        lp.add_function(target)
    wrapped = lp(func)
    wrapped()
    text_fpath = output_base.with_suffix('.line_profile.txt')
    with text_fpath.open('w') as stream:
        lp.print_stats(stream=stream, output_unit=1e-6)
    return {'text': os.fspath(text_fpath)}


def _parity_info(native, official, with_eval, tol=1e-12):
    keys = list(native['summary'])
    deltas = {
        key: abs(float(native['summary'][key]) - float(official['summary'][key]))
        for key in keys
    }
    info = {
        'max_summary_abs_delta': max(deltas.values(), default=0.0),
        'summary_deltas': deltas,
    }
    if with_eval:
        tensor = _tensor_diff(native['eval'], official['eval'], tol)
        info['tensor'] = tensor
        tensor_bad = []
        for key in ['precision', 'recall', 'scores']:
            part = tensor[key]
            if (
                not part.get('shape_match', False)
                or part.get('absent_mask_mismatches', 0)
                or part.get('num_over_tolerance', 0)
            ):
                tensor_bad.append(key)
        info['tensor_bad_keys'] = tensor_bad
        info['tensor_parameter_mismatches'] = tensor['parameter_mismatches']
    return info


def _internal_native_timing(native):
    timing = dict(native.get('timing', {}))
    total = float(timing.get('total_seconds', 0.0))
    phase_keys = [
        'candidate_build_seconds',
        'assignment_seconds',
        'accumulation_seconds',
    ]
    timing['phase_fractions'] = {
        key: (float(timing.get(key, 0.0)) / total if total else 0.0)
        for key in phase_keys
    }
    num_images = int(timing.get('candidate_index_builds', 0))
    num_slices = int(timing.get('assignment_slices', 0))
    num_thresholds = len(native.get('params', {}).get('iou_thresholds', []))
    timing['inferred_assignment_calls'] = int(timing.get(
        'assignment_calls', num_images * num_slices * num_thresholds
    ))
    return timing


def _environment():
    return {
        'python': sys.version,
        'executable': sys.executable,
        'packages': {
            name: _version(name)
            for name in ['kwcoco', 'kwimage', 'kwimage_ext', 'pycocotools', 'line-profiler']
        },
    }


def _summary_markdown(report):
    native_wall = report['timing']['native_wall']
    official_wall = report['timing']['official_wall']
    ratio = native_wall['median_seconds'] / official_wall['median_seconds']
    internal = report['native_internal_timing']
    lines = [
        '# Native COCO performance profile',
        '',
        f"- Scenario: `{report['scenario']}`",
        f"- Geometry: `{report['iou_type']}`",
        f"- Assignment backend requested: `{report['assignment_backend']}`",
        f"- Assignment backend resolved: `{report['assignment_backend_resolved']}`",
        f"- Full tensor: `{report['with_eval']}`",
        f"- Truth annotations: {report['dataset']['truth']['annotations']}",
        f"- Prediction annotations: {report['dataset']['prediction']['annotations']}",
        f"- Native median: {native_wall['median_seconds']:.6f}s",
        f"- Official median: {official_wall['median_seconds']:.6f}s",
        f"- Native / official: {ratio:.3f}x",
        f"- Candidate build: {internal.get('candidate_build_seconds', 0):.6f}s",
        f"- Candidate predictions before/after maxDet cap: "
        f"{internal.get('candidate_predictions_before_maxdet', 0)} / "
        f"{internal.get('candidate_predictions_after_maxdet', 0)}",
        f"- Assignment: {internal.get('assignment_seconds', 0):.6f}s",
        f"- Accumulation: {internal.get('accumulation_seconds', 0):.6f}s",
        f"- Assignment slices: {internal.get('assignment_slices', 0)}",
        f"- Assignment base slices: {internal.get('assignment_base_slices', internal.get('assignment_slices', 0))}",
        f"- Assignment threshold queries: {internal.get('inferred_assignment_calls', 0)}",
        f"- Assignment backend batches: {internal.get('assignment_batches', 0)}",
        f"- Compact match path: {internal.get('compact_match_path', False)}",
        f"- Max summary delta: {report['parity']['max_summary_abs_delta']:.3g}",
    ]
    if report['with_eval']:
        lines.append(
            f"- Tensor mismatches: {report['parity'].get('tensor_bad_keys', [])}"
        )
    memory = report.get('memory', None)
    if memory:
        lines += [
            f"- Native Python peak allocation: {memory['native']['python_peak_bytes'] / (1024 ** 2):.2f} MiB",
            f"- Official Python peak allocation: {memory['official']['python_peak_bytes'] / (1024 ** 2):.2f} MiB",
        ]
    profiles = report.get('profiles', {})
    if profiles:
        lines += ['', '## Profiler artifacts', '']
        for kind, paths in profiles.items():
            lines.append(f'- `{kind}`: `{paths}`')
    lines += [
        '',
        'Timing runs are unprofiled. cProfile / line-profiler / tracemalloc are',
        'separate diagnostic executions so their overhead does not contaminate',
        'the reported repeat timing.',
    ]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=sorted(SCENARIOS), default='perfect')
    parser.add_argument('--dataset-mode', choices=sorted(PROFILES), default='full')
    parser.add_argument('--iou-type', choices=sorted(IOU_TYPES), default='bbox')
    parser.add_argument(
        '--assignment-backend',
        choices=['auto', 'python', 'kwimage_ext'],
        default='auto',
        help='Greedy assignment kernel to profile',
    )
    parser.add_argument('--with-eval', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--num-objects', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument(
        '--profile', choices=['none', 'cprofile', 'line', 'both'], default='none'
    )
    parser.add_argument('--memory', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        '--output-dpath', type=Path, default=Path('_metric_profile_native_coco')
    )
    args = parser.parse_args(argv)

    if args.num_objects <= 0:
        parser.error('--num-objects must be positive')
    if args.repeat <= 0 or args.warmup < 0:
        parser.error('--repeat must be positive and --warmup nonnegative')

    true, pred, spec = _make_scenario(
        args.scenario, args.dataset_mode, args.num_objects, args.seed
    )
    from kwcoco.metrics.detect_metrics import DetectionMetrics
    dmet = DetectionMetrics.from_coco(true, pred)

    native_func = lambda: _native_call(
        dmet, args.iou_type, args.with_eval, args.assignment_backend)
    official_func = lambda: _official_call(dmet, args.iou_type, args.with_eval)

    native, native_wall = _time_call(native_func, args.warmup, args.repeat)
    official, official_wall = _time_call(official_func, args.warmup, args.repeat)
    parity = _parity_info(native, official, args.with_eval)

    report = {
        'schema_version': 1,
        'campaign': 'native_coco_performance_profile',
        'scenario': args.scenario,
        'dataset_mode': args.dataset_mode,
        'iou_type': args.iou_type,
        'assignment_backend': args.assignment_backend,
        'assignment_backend_resolved': native['params'].get('assignment_backend'),
        'with_eval': bool(args.with_eval),
        'spec': spec,
        'dataset': {
            'truth': _dataset_counts(true),
            'prediction': _dataset_counts(pred),
        },
        'environment': _environment(),
        'timing': {
            'native_wall': native_wall,
            'official_wall': official_wall,
            'native_over_official_median': (
                native_wall['median_seconds'] / official_wall['median_seconds']
            ),
        },
        'native_internal_timing': _internal_native_timing(native),
        'parity': parity,
    }

    args.output_dpath.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{args.scenario}_{args.dataset_mode}_{args.iou_type}_"
        f"{args.assignment_backend}_{'eval' if args.with_eval else 'summary'}"
    )
    output_base = args.output_dpath / stem

    if args.memory:
        report['memory'] = {
            'native': _measure_tracemalloc(native_func),
            'official': _measure_tracemalloc(official_func),
        }

    profiles = {}
    if args.profile in {'cprofile', 'both'}:
        profiles['cprofile'] = _write_cprofile(
            native_func, output_base.with_name(output_base.name + '_native')
        )
    if args.profile in {'line', 'both'}:
        profiles['line'] = _write_line_profile(
            native_func, output_base.with_name(output_base.name + '_native')
        )
    if profiles:
        report['profiles'] = profiles

    json_fpath = output_base.with_suffix('.json')
    md_fpath = output_base.with_suffix('.md')
    json_fpath.write_text(json.dumps(report, indent=2, default=_json_default) + '\n')
    md_fpath.write_text(_summary_markdown(report))

    print(_summary_markdown(report), end='')
    print(f'JSON: {json_fpath}')
    print(f'Markdown: {md_fpath}')
    if parity['max_summary_abs_delta'] > 1e-12:
        raise SystemExit('summary parity failed during performance profile')
    if args.with_eval:
        if parity.get('tensor_bad_keys') or parity.get('tensor_parameter_mismatches'):
            raise SystemExit('tensor parity failed during performance profile')


if __name__ == '__main__':
    main()
