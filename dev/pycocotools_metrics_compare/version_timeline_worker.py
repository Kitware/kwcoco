#!/usr/bin/env python3
"""Worker for cross-version historical KWCoco metric benchmarking.

This file intentionally uses only APIs that were already present in KWCoco
0.9.0.  The parent runner selects the KWCoco source tree with ``PYTHONPATH`` and
executes this worker in a fresh process for every measured revision.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.metadata
import inspect
import io
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _capabilities():
    from kwcoco.coco_evaluator import CocoEvaluator
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    score_pct = inspect.signature(DetectionMetrics.score_pycocotools)
    score_pct_params = score_pct.parameters
    score_native = getattr(DetectionMetrics, 'score_native_coco', None)
    eval_pct = getattr(CocoEvaluator, 'evaluate_pycocotools', None)
    eval_native = getattr(CocoEvaluator, 'evaluate_native_coco', None)

    stat_keys = None
    try:
        from kwcoco.metrics.coco_metrics import COCO_DETECTION_STAT_KEYS
    except Exception:
        pass
    else:
        stat_keys = list(COCO_DETECTION_STAT_KEYS)

    return {
        'historical_evaluator': hasattr(CocoEvaluator, 'evaluate'),
        'pycocotools_bbox': hasattr(DetectionMetrics, 'score_pycocotools'),
        'pycocotools_iou_type_argument': 'iou_type' in score_pct_params,
        'pycocotools_full_eval_tensor': 'with_eval' in score_pct_params,
        'dataset_official_coco_api': eval_pct is not None,
        'native_coco_api': score_native is not None and eval_native is not None,
        'native_assignment_backend_argument': (
            score_native is not None
            and 'assignment_backend' in inspect.signature(score_native).parameters
        ),
        'named_coco_detection_stats': stat_keys,
    }


def _quiet_call(call):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        result = call()
    return result


def _make_evaluator(truth_fpath, pred_fpath):
    from kwcoco.coco_evaluator import CocoEvaluator

    evaluator = CocoEvaluator({
        'true_dataset': os.fspath(truth_fpath),
        'pred_dataset': os.fspath(pred_fpath),
    })
    # Verbosity is evaluator state, not part of the historical CocoEvalConfig.
    # Keep this out of the config dictionary so the same worker is valid for
    # both scriptconfig-based releases and the current kwconf implementation.
    if hasattr(evaluator, '_verbose'):
        evaluator._verbose = 0
    return evaluator


def _timing_summary(warmup_samples, samples, warmup, repeat):
    return {
        'warmup': int(warmup),
        'repeat': int(repeat),
        'warmup_seconds': list(map(float, warmup_samples)),
        'samples_seconds': list(map(float, samples)),
        'min_seconds': float(min(samples)) if samples else None,
        'median_seconds': float(statistics.median(samples)) if samples else None,
        'mean_seconds': float(statistics.mean(samples)) if samples else None,
        'max_seconds': float(max(samples)) if samples else None,
        'stdev_seconds': (
            float(statistics.stdev(samples)) if len(samples) > 1 else 0.0
        ),
    }


def _time_call(call, warmup, repeat):
    warmup_samples = []
    for _ in range(warmup):
        gc.collect()
        start = time.perf_counter()
        _quiet_call(call)
        warmup_samples.append(time.perf_counter() - start)

    samples = []
    result = None
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        result = _quiet_call(call)
        samples.append(time.perf_counter() - start)

    if result is None:
        result = _quiet_call(call)
    return result, _timing_summary(warmup_samples, samples, warmup, repeat)


def _time_evaluator(truth_fpath, pred_fpath, warmup, repeat):
    # Prepared scope: dataset coercion / class rectification happens outside the
    # timed region, while _build_dmet, assignment, and accumulation remain in.
    prepared = _make_evaluator(truth_fpath, pred_fpath)
    _quiet_call(prepared._ensure_init)
    result, prepared_timing = _time_call(prepared.evaluate, warmup, repeat)

    # End-to-end scope: include evaluator construction and dataset initialization.
    def fresh_call():
        return _make_evaluator(truth_fpath, pred_fpath).evaluate()

    _fresh_result, end_to_end_timing = _time_call(fresh_call, warmup, repeat)
    return result, {
        'prepared_evaluator': prepared_timing,
        'end_to_end': end_to_end_timing,
    }


def _metric_state(results):
    """Strip non-metric metadata while retaining complete curves/counts."""
    raw = results.__json__()
    state = {}
    for result_key, item in raw.items():
        if not isinstance(item, dict):
            continue
        keep = {}
        for key in ['nocls_measures', 'ovr_measures']:
            if key in item:
                keep[key] = item[key]
        state[result_key] = keep
    return state


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--truth-coco', type=Path, required=True)
    parser.add_argument('--pred-coco', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--source-label', required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args(argv)

    import kwcoco

    result, timing = _time_evaluator(
        args.truth_coco, args.pred_coco, args.warmup, args.repeat)

    record = {
        'source_label': args.source_label,
        'kwcoco_version': getattr(kwcoco, '__version__', None),
        'kwcoco_file': os.path.realpath(kwcoco.__file__),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'dependencies': {
            name: _version(name)
            for name in [
                'numpy', 'ubelt', 'kwarray', 'kwimage', 'kwimage_ext',
                'pycocotools', 'scriptconfig', 'kwconf',
            ]
        },
        'capabilities': _capabilities(),
        'timing': timing,
        'metric_state': _metric_state(result),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(record, indent=2, allow_nan=True) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
