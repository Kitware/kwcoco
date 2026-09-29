#!/usr/bin/env python3
"""Isolated worker for release-scale native COCO benchmarking.

This module is intentionally launched in a fresh Python process by
``run_release_benchmark.py``.  Keeping each backend isolated avoids allocator
carry-over and makes process RSS measurements meaningful.  The parent runner
owns cross-backend comparisons and report generation.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time
import tracemalloc

import numpy as np


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


def _rect_poly(x, y, w, h, inset=0.0):
    """Return a flattened fractional COCO polygon for a rectangle."""
    x1 = float(x + inset)
    y1 = float(y + inset)
    x2 = float(x + w - inset)
    y2 = float(y + h - inset)
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _synthetic_dsets(spec):
    """Build a deterministic scalable detection / instance-mask workload."""
    import kwcoco

    seed = int(spec.get('seed', 0))
    rng = np.random.RandomState(seed)
    n_images = int(spec['n_images'])
    n_classes = int(spec.get('n_classes', 5))
    truth_per_image = int(spec.get('truth_per_image', 8))
    pred_per_image = int(spec.get('pred_per_image', 24))
    image_size = int(spec.get('image_size', 512))
    crowd_fraction = float(spec.get('crowd_fraction', 0.02))
    negative_fraction = float(spec.get('negative_fraction', 0.05))
    class_error_fraction = float(spec.get('class_error_fraction', 0.08))
    miss_fraction = float(spec.get('miss_fraction', 0.12))
    jitter = float(spec.get('jitter', 0.08))
    dense = bool(spec.get('dense', False))

    true = kwcoco.CocoDataset()
    pred = kwcoco.CocoDataset()
    for dset in [true, pred]:
        for cid in range(1, n_classes + 1):
            dset.add_category(name=f'class_{cid}', id=cid)

    next_true_aid = 1
    next_pred_aid = 1
    for gid in range(1, n_images + 1):
        name = f'image_{gid:07d}'
        for dset in [true, pred]:
            dset.add_image(
                id=gid, file_name=name, width=image_size, height=image_size)

        is_negative = rng.uniform() < negative_fraction
        local_truth = []
        n_true = 0 if is_negative else truth_per_image
        for tx in range(n_true):
            if dense:
                # Concentrate boxes in a central region to increase candidate
                # list lengths while keeping exact geometry deterministic.
                base_x = image_size * 0.30 + (tx % 7) * 2.25
                base_y = image_size * 0.30 + (tx % 5) * 2.75
                side = float(rng.choice([48.0, 72.0, 112.0]))
                w = side * float(rng.uniform(0.85, 1.15))
                h = side * float(rng.uniform(0.85, 1.15))
                x = min(base_x + rng.uniform(-1.0, 1.0), image_size - w - 1)
                y = min(base_y + rng.uniform(-1.0, 1.0), image_size - h - 1)
            else:
                # Span COCO small / medium / large ranges. Fractional origins
                # exercise the strict XYWH area path validated by Phase 3.
                side = float(rng.choice([18.0, 36.0, 64.0, 104.0, 140.0]))
                w = side * float(rng.uniform(0.75, 1.25))
                h = side * float(rng.uniform(0.75, 1.25))
                w = min(w, image_size * 0.42)
                h = min(h, image_size * 0.42)
                x = float(rng.uniform(0.25, max(0.26, image_size - w - 0.25)))
                y = float(rng.uniform(0.25, max(0.26, image_size - h - 0.25)))
            cid = int(rng.randint(1, n_classes + 1))
            iscrowd = int(rng.uniform() < crowd_fraction)
            bbox = [float(x), float(y), float(w), float(h)]
            seg = [_rect_poly(x, y, w, h)]
            true.add_annotation(
                id=next_true_aid,
                image_id=gid,
                category_id=cid,
                bbox=bbox,
                segmentation=seg,
                iscrowd=iscrowd,
            )
            local_truth.append((bbox, cid, iscrowd))
            next_true_aid += 1

        preds = []
        for tx, (bbox, cid, iscrowd) in enumerate(local_truth):
            if rng.uniform() < miss_fraction:
                continue
            x, y, w, h = bbox
            dx = float(rng.normal(0, max(0.15, w * jitter)))
            dy = float(rng.normal(0, max(0.15, h * jitter)))
            dw = float(rng.normal(0, max(0.10, w * jitter * 0.5)))
            dh = float(rng.normal(0, max(0.10, h * jitter * 0.5)))
            pw = max(2.0, min(image_size * 0.5, w + dw))
            ph = max(2.0, min(image_size * 0.5, h + dh))
            px = min(max(0.0, x + dx), image_size - pw)
            py = min(max(0.0, y + dy), image_size - ph)
            pred_cid = cid
            if rng.uniform() < class_error_fraction:
                pred_cid = 1 + (cid % n_classes)
            quality = math.exp(-0.5 * ((dx / max(w, 1)) ** 2 + (dy / max(h, 1)) ** 2))
            score = float(np.clip(0.52 + 0.45 * quality + rng.normal(0, 0.025), 0.01, 0.999))
            # Slightly different polygon inset makes segm non-identical to bbox
            # while preserving a deterministic polygon-only canonical dataset.
            inset = float(max(0.0, rng.normal(0.25, 0.15)))
            preds.append((
                score, pred_cid, [px, py, pw, ph],
                [_rect_poly(px, py, pw, ph, inset=inset)]))

        while len(preds) < pred_per_image:
            side = float(rng.choice([14.0, 30.0, 58.0, 96.0]))
            w = side * float(rng.uniform(0.8, 1.2))
            h = side * float(rng.uniform(0.8, 1.2))
            x = float(rng.uniform(0.0, max(0.01, image_size - w)))
            y = float(rng.uniform(0.0, max(0.01, image_size - h)))
            cid = int(rng.randint(1, n_classes + 1))
            score = float(rng.beta(2.0, 5.0) * 0.85)
            preds.append((score, cid, [x, y, w, h], [_rect_poly(x, y, w, h)]))

        # Preserve stable source ordering while allowing maxDet pressure.
        preds = preds[:pred_per_image]
        for score, cid, bbox, seg in preds:
            pred.add_annotation(
                id=next_pred_aid,
                image_id=gid,
                category_id=cid,
                bbox=list(map(float, bbox)),
                segmentation=seg,
                score=float(score),
            )
            next_pred_aid += 1

    return true, pred


def _load_dsets(spec):
    import kwcoco

    kind = spec.get('kind', 'synthetic')
    if kind == 'synthetic':
        return _synthetic_dsets(spec)
    if kind == 'files':
        true = kwcoco.CocoDataset.coerce(spec['truth_coco'])
        pred = kwcoco.CocoDataset.coerce(spec['pred_coco'])
        return true, pred
    raise KeyError(kind)


def _dataset_counts(true, pred):
    return {
        'images': len(true.imgs),
        'truth_annotations': len(true.anns),
        'prediction_annotations': len(pred.anns),
        'categories': len(true.cats),
    }


def _native_kwargs(iou_type, backend, with_eval):
    assignment_backend = {
        'native-python': 'python',
        'native-rust': 'kwimage_ext',
        'native-auto': 'auto',
    }[backend]
    kwargs = {
        'iou_type': iou_type,
        'with_eval': with_eval,
        'assignment_backend': assignment_backend,
    }
    if iou_type == 'segm':
        kwargs.update(overlap_backend='auto', raster_backend='auto')
    return kwargs


def _make_call(true, pred, backend, iou_type, scope, with_eval):
    if scope == 'core':
        from kwcoco.metrics.detect_metrics import DetectionMetrics
        setup_start = time.perf_counter()
        dmet = DetectionMetrics.from_coco(true, pred)
        setup_seconds = time.perf_counter() - setup_start
        if backend == 'official':
            def call():
                return dmet.score_pycocotools(
                    iou_type=iou_type,
                    with_eval=with_eval,
                    bbox_annotation_policy='coco',
                    verbose=0,
                )
        else:
            kwargs = _native_kwargs(iou_type, backend, with_eval)
            def call():
                return dmet.score_native_coco(**kwargs)
        return call, setup_seconds

    if scope == 'evaluator':
        from kwcoco.coco_evaluator import CocoEvaluator
        # The public API path deliberately rebuilds its internal metric object
        # per call. This is the end-to-end user-facing measurement.
        setup_seconds = 0.0
        if backend == 'official':
            def call():
                evaluator = CocoEvaluator({
                    'true_dataset': true,
                    'pred_dataset': pred,
                })
                return evaluator.evaluate_pycocotools(
                    iou_type,
                    with_eval=with_eval,
                )[iou_type]
        else:
            kwargs = _native_kwargs(iou_type, backend, with_eval)
            def call():
                evaluator = CocoEvaluator({
                    'true_dataset': true,
                    'pred_dataset': pred,
                })
                return evaluator.evaluate_native_coco(
                    iou_type,
                    **{k: v for k, v in kwargs.items() if k != 'iou_type'},
                )[iou_type]
        return call, setup_seconds

    raise KeyError(scope)


class _RssSampler:
    """Low-overhead process RSS sampler including C/Rust allocations."""
    def __init__(self, interval=0.01):
        import psutil
        self.process = psutil.Process(os.getpid())
        self.interval = float(interval)
        self._stop = threading.Event()
        self._thread = None
        self.samples = []
        self.peak = 0

    def _sample_once(self):
        try:
            rss = int(self.process.memory_info().rss)
        except Exception:
            return
        self.samples.append(rss)
        self.peak = max(self.peak, rss)

    def _run(self):
        while not self._stop.wait(self.interval):
            self._sample_once()

    def start(self):
        self._sample_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample_once()


def _rss_now():
    import psutil
    return int(psutil.Process(os.getpid()).memory_info().rss)


def _quiet_call(call, quiet=True):
    if not quiet:
        return call()
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return call()


def _time_call(call, warmup, repeat, quiet=True):
    warmup_samples = []
    result = None
    for _ in range(warmup):
        result = None
        gc.collect()
        start = time.perf_counter()
        result = _quiet_call(call, quiet=quiet)
        warmup_samples.append(time.perf_counter() - start)
    samples = []
    for _ in range(repeat):
        result = None
        gc.collect()
        start = time.perf_counter()
        result = _quiet_call(call, quiet=quiet)
        samples.append(time.perf_counter() - start)
    return result, {
        'warmup': int(warmup),
        'repeat': int(repeat),
        'warmup_seconds': list(map(float, warmup_samples)),
        'samples_seconds': list(map(float, samples)),
        'min_seconds': float(min(samples)),
        'median_seconds': float(statistics.median(samples)),
        'mean_seconds': float(statistics.mean(samples)),
        'max_seconds': float(max(samples)),
        'stdev_seconds': float(statistics.stdev(samples)) if len(samples) > 1 else 0.0,
    }


def _memory_call(call, sample_interval, python_allocations, quiet=True):
    gc.collect()
    baseline = _rss_now()
    sampler = _RssSampler(sample_interval)
    sampler.start()
    trace_peak = None
    if python_allocations:
        tracemalloc.start()
    start = time.perf_counter()
    try:
        result = _quiet_call(call, quiet=quiet)
    finally:
        elapsed = time.perf_counter() - start
        if python_allocations:
            _current, trace_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        sampler.stop()
    final = _rss_now()
    peak = max(sampler.peak, baseline, final)
    return result, {
        'elapsed_seconds': float(elapsed),
        'baseline_rss_bytes': int(baseline),
        'peak_rss_bytes': int(peak),
        'peak_rss_delta_bytes': int(max(0, peak - baseline)),
        'final_rss_bytes': int(final),
        'sample_interval_seconds': float(sample_interval),
        'num_rss_samples': len(sampler.samples),
        'python_peak_tracemalloc_bytes': (
            None if trace_peak is None else int(trace_peak)
        ),
        'tracemalloc_note': (
            'Python allocations only; RSS is the primary metric because C/Rust '
            'allocations are not fully visible to tracemalloc.'
        ),
    }


def _result_metadata(result):
    summary = {
        str(k): float(v) for k, v in result.get('summary', {}).items()
    }
    return {
        'summary': summary,
        'backend': result.get('backend'),
        'implementation': result.get('implementation'),
        'official': result.get('official'),
        'iou_type': result.get('iou_type'),
        'accumulator': result.get('accumulator'),
        'params': result.get('params'),
        'provenance': result.get('provenance'),
        'timing': result.get('timing'),
    }


def _write_eval_npz(path, result):
    eval_record = result.get('eval')
    if eval_record is None:
        raise AssertionError('parity phase requires with_eval=True')
    arrays = {
        'precision': np.asarray(eval_record['precision']),
        'recall': np.asarray(eval_record['recall']),
        'scores': np.asarray(eval_record['scores']),
    }
    # ``realpos_total`` is a native kwcoco diagnostic tensor, not part of the
    # official COCOeval tensor snapshot.  Preserve it when it genuinely exists,
    # but never serialize ``None`` because ``np.asarray(None)`` creates an
    # object-dtype array that cannot be loaded with NumPy's safe default
    # ``allow_pickle=False``.
    realpos_total = eval_record.get('realpos_total', None)
    if realpos_total is not None:
        arrays['realpos_total'] = np.asarray(realpos_total)
    object_keys = [key for key, arr in arrays.items() if arr.dtype.hasobject]
    if object_keys:
        raise TypeError(
            'release benchmark parity artifacts must contain only numeric '
            f'arrays; object dtype found for {object_keys!r}'
        )
    np.savez_compressed(path, **arrays)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec-json', type=Path, required=True)
    parser.add_argument(
        '--backend',
        choices=['official', 'native-python', 'native-rust', 'native-auto'],
        required=True)
    parser.add_argument('--iou-type', choices=['bbox', 'segm'], required=True)
    parser.add_argument('--scope', choices=['core', 'evaluator'], default='core')
    parser.add_argument('--eval-mode', choices=['summary', 'tensor'], default='tensor')
    parser.add_argument('--phase', choices=['timing', 'memory', 'parity'], required=True)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--memory-sample-interval', type=float, default=0.01)
    parser.add_argument('--python-allocations', action='store_true')
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-npz', type=Path)
    parser.add_argument('--show-output', action='store_true')
    args = parser.parse_args(argv)

    spec = json.loads(args.spec_json.read_text())
    total_start = time.perf_counter()
    rss_process_start = _rss_now()
    dataset_start = time.perf_counter()
    true, pred = _load_dsets(spec)
    dataset_seconds = time.perf_counter() - dataset_start
    rss_after_dataset = _rss_now()
    counts = _dataset_counts(true, pred)

    with_eval = args.eval_mode == 'tensor' or args.phase == 'parity'
    call, setup_seconds = _make_call(
        true, pred, args.backend, args.iou_type, args.scope, with_eval)
    rss_before_score = _rss_now()

    if args.phase == 'timing':
        result, measurement = _time_call(
            call, args.warmup, args.repeat, quiet=not args.show_output)
    elif args.phase == 'memory':
        result, measurement = _memory_call(
            call,
            sample_interval=args.memory_sample_interval,
            python_allocations=args.python_allocations,
            quiet=not args.show_output,
        )
    elif args.phase == 'parity':
        start = time.perf_counter()
        result = _quiet_call(call, quiet=not args.show_output)
        measurement = {'elapsed_seconds': float(time.perf_counter() - start)}
        if args.output_npz is None:
            raise ValueError('--output-npz is required for parity phase')
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        _write_eval_npz(args.output_npz, result)
    else:  # pragma: no cover
        raise KeyError(args.phase)

    record = {
        'schema': 'kwcoco_release_benchmark_worker_v1',
        'spec': spec,
        'backend_label': args.backend,
        'iou_type': args.iou_type,
        'scope': args.scope,
        'eval_mode': args.eval_mode,
        'phase': args.phase,
        'counts': counts,
        'dataset_seconds': float(dataset_seconds),
        'metric_setup_seconds': float(setup_seconds),
        'rss_process_start_bytes': int(rss_process_start),
        'rss_after_dataset_bytes': int(rss_after_dataset),
        'rss_before_score_bytes': int(rss_before_score),
        'measurement': measurement,
        'result': _result_metadata(result),
        'worker_total_seconds': float(time.perf_counter() - total_start),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(record, indent=2, default=_json_default) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
