#!/usr/bin/env python3
"""Release-scale benchmark and evidence generator for native COCO metrics.

The goal of this runner is not to produce one flattering timing number.  It
records enough information to support a reproducible technical comparison of
kwcoco's native COCO evaluator against the official pycocotools reference:

* isolated-process warm timing distributions;
* peak process RSS including Python, C, and Rust allocations;
* optional Python-only tracemalloc peaks;
* full precision / recall / sampled-score tensor parity on every workload;
* scaling throughput and annotation counts;
* environment, package, CPU, git, and backend provenance;
* machine-readable JSON/CSV plus a publication-oriented Markdown report.

Synthetic workloads are deterministic and deliberately cover ordinary sparse
images, dense candidate graphs, and maxDet pressure.  Real truth/prediction
kwcoco files can be supplied for application-specific evidence.
"""
from __future__ import annotations

import argparse
import csv
import datetime as datetime_mod
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys
import tarfile
import textwrap
import typing

import numpy as np


HERE = Path(__file__).resolve().parent
WORKER = HERE / 'release_benchmark_worker.py'
CONFORMANCE = HERE / 'run_phase3_coco_conformance.py'


SUITES = {
    'quick': [
        {
            'name': 'scale_32', 'kind': 'synthetic', 'n_images': 32,
            'n_classes': 4, 'truth_per_image': 6, 'pred_per_image': 14,
            'image_size': 384, 'seed': 11,
        },
        {
            'name': 'scale_128', 'kind': 'synthetic', 'n_images': 128,
            'n_classes': 5, 'truth_per_image': 8, 'pred_per_image': 18,
            'image_size': 512, 'seed': 13,
        },
    ],
    'standard': [
        {
            'name': 'scale_128', 'kind': 'synthetic', 'n_images': 128,
            'n_classes': 5, 'truth_per_image': 8, 'pred_per_image': 18,
            'image_size': 512, 'seed': 13,
        },
        {
            'name': 'scale_1024', 'kind': 'synthetic', 'n_images': 1024,
            'n_classes': 5, 'truth_per_image': 8, 'pred_per_image': 24,
            'image_size': 512, 'seed': 17,
        },
        {
            'name': 'dense_256', 'kind': 'synthetic', 'n_images': 256,
            'n_classes': 4, 'truth_per_image': 30, 'pred_per_image': 100,
            'image_size': 512, 'seed': 19, 'dense': True,
            'negative_fraction': 0.0,
        },
        {
            'name': 'maxdet_128', 'kind': 'synthetic', 'n_images': 128,
            'n_classes': 1, 'truth_per_image': 8, 'pred_per_image': 300,
            'image_size': 512, 'seed': 23,
        },
    ],
    'large': [
        {
            'name': 'scale_4096', 'kind': 'synthetic', 'n_images': 4096,
            'n_classes': 5, 'truth_per_image': 8, 'pred_per_image': 24,
            'image_size': 512, 'seed': 29,
        },
        {
            'name': 'dense_512', 'kind': 'synthetic', 'n_images': 512,
            'n_classes': 4, 'truth_per_image': 30, 'pred_per_image': 100,
            'image_size': 512, 'seed': 31, 'dense': True,
            'negative_fraction': 0.0,
        },
        {
            'name': 'maxdet_256', 'kind': 'synthetic', 'n_images': 256,
            'n_classes': 1, 'truth_per_image': 8, 'pred_per_image': 300,
            'image_size': 512, 'seed': 37,
        },
    ],
    'huge': [
        {
            'name': 'scale_10000', 'kind': 'synthetic', 'n_images': 10000,
            'n_classes': 5, 'truth_per_image': 8, 'pred_per_image': 24,
            'image_size': 512, 'seed': 41,
        },
    ],
}


def _publication_suite():
    seen = set()
    items = []
    for suite_name in ['standard', 'large']:
        for item in SUITES[suite_name]:
            if item['name'] not in seen:
                items.append(dict(item))
                seen.add(item['name'])
    return items


SUITES['publication'] = _publication_suite()


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
        return {'returncode': None, 'output': None, 'error': repr(ex)}
    return {'returncode': proc.returncode, 'output': proc.stdout.strip()}


def _module_repo_info(module_name):
    try:
        module = __import__(module_name)
    except Exception as ex:
        return {'status': 'unavailable', 'error': repr(ex)}
    path = Path(module.__file__).resolve()
    repo = path.parent
    for candidate in [path.parent, *path.parents]:
        if (candidate / '.git').exists():
            repo = candidate
            break
    return {
        'status': 'ok',
        'module_file': os.fspath(path),
        'repo': os.fspath(repo),
        'head': _command(['git', 'rev-parse', 'HEAD'], cwd=repo),
        'status_short': _command(['git', 'status', '--short'], cwd=repo),
    }


def _cpu_model():
    proc = Path('/proc/cpuinfo')
    if proc.exists():
        for line in proc.read_text(errors='ignore').splitlines():
            if line.lower().startswith('model name') and ':' in line:
                return line.split(':', 1)[1].strip()
    return platform.processor() or None


def environment_info():
    try:
        import psutil
        vm = psutil.virtual_memory()
        cpu_freq = psutil.cpu_freq()
        machine = {
            'cpu_model': _cpu_model(),
            'logical_cpus': psutil.cpu_count(logical=True),
            'physical_cpus': psutil.cpu_count(logical=False),
            'total_ram_bytes': int(vm.total),
            'cpu_frequency_mhz': None if cpu_freq is None else float(cpu_freq.current),
        }
    except Exception as ex:
        machine = {'error': repr(ex), 'cpu_model': _cpu_model()}

    governor_path = Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor')
    governor = None
    if governor_path.exists():
        try:
            governor = governor_path.read_text().strip()
        except Exception:
            pass

    rust_caps = None
    rust_meta = None
    try:
        from kwimage_ext import _rust
        rust_caps = list(_rust.capabilities())
        from kwimage_ext.algo.assignment import backend_metadata
        rust_meta = backend_metadata()
    except Exception as ex:
        rust_meta = {'status': 'unavailable', 'error': repr(ex)}

    return {
        'schema': 'kwcoco_release_benchmark_environment_v1',
        'timestamp_utc': datetime_mod.datetime.now(datetime_mod.timezone.utc).isoformat(),
        'python': sys.version,
        'executable': sys.executable,
        'platform': platform.platform(),
        'machine': machine,
        'cpu_governor': governor,
        'packages': {
            name: _version(name)
            for name in [
                'kwcoco', 'kwimage', 'kwimage_ext', 'pycocotools',
                'numpy', 'psutil', 'kwarray', 'ubelt',
            ]
        },
        'kwimage_ext_rust_capabilities': rust_caps,
        'kwimage_ext_assignment_backend': rust_meta,
        'repositories': {
            'kwcoco': _module_repo_info('kwcoco'),
            'kwimage_ext': _module_repo_info('kwimage_ext'),
        },
        'tools': {
            'rustc': _command(['rustc', '--version']),
            'cargo': _command(['cargo', '--version']),
            'lscpu': _command(['lscpu']),
        },
        'thread_environment': {
            key: os.environ.get(key)
            for key in [
                'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS', 'PYTHONHASHSEED',
            ]
        },
    }


def _parse_csv_arg(text):
    return [item.strip() for item in text.split(',') if item.strip()]


def _run_worker(
    *, spec_path, backend, iou_type, scope, eval_mode, phase,
    output_json, output_npz=None, warmup=1, repeat=5,
    memory_sample_interval=0.01, python_allocations=False,
):
    command = [
        sys.executable, os.fspath(WORKER),
        '--spec-json', os.fspath(spec_path),
        '--backend', backend,
        '--iou-type', iou_type,
        '--scope', scope,
        '--eval-mode', eval_mode,
        '--phase', phase,
        '--warmup', str(warmup),
        '--repeat', str(repeat),
        '--memory-sample-interval', str(memory_sample_interval),
        '--output-json', os.fspath(output_json),
    ]
    if output_npz is not None:
        command += ['--output-npz', os.fspath(output_npz)]
    if python_allocations:
        command += ['--python-allocations']
    proc = subprocess.run(
        command, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(
            f'worker failed ({proc.returncode}): {shlex.join(command)}\n{proc.stdout}'
        )
    return json.loads(output_json.read_text())


def _tensor_diff(native_npz, official_npz, atol):
    result = {'ok': True, 'keys': {}}
    with np.load(native_npz) as ndata, np.load(official_npz) as odata:
        native_precision = np.asarray(ndata['precision'])
        official_precision = np.asarray(odata['precision'])
        precision_absent_native = native_precision < 0
        precision_absent_official = official_precision < 0
        # These are the three tensors defined by official COCOeval.eval and
        # therefore the tensors that can be compared across implementations.
        # Native kwcoco also exposes ``realpos_total`` as a useful diagnostic,
        # but pycocotools does not.  Older benchmark artifacts accidentally
        # serialized the missing official value as an object array; ignoring
        # that non-COCO key also lets --resume consume those artifacts safely.
        for key in ['precision', 'recall', 'scores']:
            a = np.asarray(ndata[key])
            b = np.asarray(odata[key])
            info = {
                'shape_native': list(a.shape),
                'shape_official': list(b.shape),
                'shape_match': a.shape == b.shape,
            }
            if a.shape != b.shape:
                info['ok'] = False
                result['ok'] = False
                result['keys'][key] = info
                continue
            if key == 'scores':
                # Detection scores may legitimately be negative. COCO's
                # sampled score tensor uses -1 as an absent sentinel, so
                # derive score-cell validity from the aligned precision
                # tensor instead of testing score values themselves.
                absent_a = precision_absent_native
                absent_b = precision_absent_official
            else:
                absent_a = a < 0
                absent_b = b < 0
            absent_match = np.array_equal(absent_a, absent_b)
            valid = ~(absent_a | absent_b)
            if np.any(valid):
                max_abs = float(np.max(np.abs(a[valid] - b[valid])))
            else:
                max_abs = 0.0
            equal = absent_match and max_abs <= atol
            info.update(
                ok=bool(equal),
                absent_mask_match=bool(absent_match),
                max_abs_delta=max_abs,
                populated_cells=int(np.count_nonzero(valid)),
            )
            if not info['ok']:
                result['ok'] = False
            result['keys'][key] = info
    return result


def _load_reusable_worker_record(
    path, *, spec, backend, iou_type, scope, eval_mode, phase,
    warmup=None, repeat=None, memory_sample_interval=None,
    python_allocations=None,
):
    """Load one completed worker record, rejecting incompatible resume data."""
    path = Path(path)
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    expected = {
        'schema': 'kwcoco_release_benchmark_worker_v1',
        'spec': spec,
        'backend_label': backend,
        'iou_type': iou_type,
        'scope': scope,
        'eval_mode': eval_mode,
        'phase': phase,
    }
    mismatches = {
        key: {'expected': value, 'found': record.get(key, None)}
        for key, value in expected.items()
        if record.get(key, None) != value
    }
    measurement = record.get('measurement', {})
    if phase == 'timing':
        if warmup is not None and measurement.get('warmup') != int(warmup):
            mismatches['warmup'] = {
                'expected': int(warmup), 'found': measurement.get('warmup')}
        if repeat is not None and measurement.get('repeat') != int(repeat):
            mismatches['repeat'] = {
                'expected': int(repeat), 'found': measurement.get('repeat')}
    elif phase == 'memory':
        if memory_sample_interval is not None:
            found_interval = measurement.get('sample_interval_seconds', None)
            if (
                found_interval is None
                or not math.isclose(
                    float(found_interval), float(memory_sample_interval),
                    rel_tol=0.0, abs_tol=1e-15)
            ):
                mismatches['memory_sample_interval'] = {
                    'expected': float(memory_sample_interval),
                    'found': found_interval,
                }
        if python_allocations is not None:
            found_python_allocations = (
                measurement.get('python_peak_tracemalloc_bytes', None) is not None
            )
            if found_python_allocations != bool(python_allocations):
                mismatches['python_allocations'] = {
                    'expected': bool(python_allocations),
                    'found': found_python_allocations,
                }
    if mismatches:
        raise RuntimeError(
            f'cannot resume from incompatible worker record {path}: '
            f'{json.dumps(mismatches, sort_keys=True)}'
        )
    return record


def _summary_diff(native_record, official_record, atol):
    nsum = native_record['result']['summary']
    osum = official_record['result']['summary']
    keys = sorted(set(nsum) | set(osum))
    deltas = {}
    max_abs = 0.0
    ok = True
    for key in keys:
        a = float(nsum.get(key, float('nan')))
        b = float(osum.get(key, float('nan')))
        if math.isnan(a) or math.isnan(b):
            delta = float('inf')
        else:
            delta = abs(a - b)
        deltas[key] = delta
        max_abs = max(max_abs, delta)
        ok &= delta <= atol
    return {'ok': bool(ok), 'max_abs_delta': float(max_abs), 'deltas': deltas}


def _quantile(values, q):
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _record_key(record):
    spec = record['spec']
    return (
        spec['name'], record['iou_type'], record['scope'], record['eval_mode'],
        record['backend_label'],
    )


def _summarize_records(timing_records, memory_records, parity_records):
    memory_by_key = {}
    for r in memory_records:
        memory_by_key.setdefault(_record_key(r), []).append(r)
    parity_by_case = {}
    for item in parity_records:
        parity_by_case[(item['scenario'], item['iou_type'], item['backend'])] = item

    rows = []
    for record in timing_records:
        key = _record_key(record)
        scenario, iou_type, scope, eval_mode, backend = key
        m = record['measurement']
        counts = record['counts']
        median = float(m['median_seconds'])
        mem_records = memory_by_key.get(key, [])
        if mem_records:
            peak_rss_values = [r['measurement']['peak_rss_bytes'] for r in mem_records]
            delta_rss_values = [r['measurement']['peak_rss_delta_bytes'] for r in mem_records]
            trace_values = [
                r['measurement'].get('python_peak_tracemalloc_bytes')
                for r in mem_records
                if r['measurement'].get('python_peak_tracemalloc_bytes') is not None
            ]
            mem_measure = {
                'peak_rss_bytes': int(statistics.median(peak_rss_values)),
                'peak_rss_delta_bytes': int(statistics.median(delta_rss_values)),
                'peak_rss_delta_max_bytes': int(max(delta_rss_values)),
                'python_peak_tracemalloc_bytes': (
                    None if not trace_values else int(statistics.median(trace_values))
                ),
                'memory_repeat': len(mem_records),
            }
        else:
            mem_measure = {}
        parity = parity_by_case.get((scenario, iou_type, backend))
        internal_timing = record['result'].get('timing') or {}
        rows.append({
            'scenario': scenario,
            'iou_type': iou_type,
            'scope': scope,
            'eval_mode': eval_mode,
            'backend': backend,
            'images': counts['images'],
            'truth_annotations': counts['truth_annotations'],
            'prediction_annotations': counts['prediction_annotations'],
            'categories': counts['categories'],
            'dataset_seconds': record.get('dataset_seconds'),
            'metric_setup_seconds': record.get('metric_setup_seconds'),
            'median_seconds': median,
            'min_seconds': float(m['min_seconds']),
            'max_seconds': float(m['max_seconds']),
            'stdev_seconds': float(m['stdev_seconds']),
            'p10_seconds': _quantile(m['samples_seconds'], 0.10),
            'p90_seconds': _quantile(m['samples_seconds'], 0.90),
            'images_per_second': counts['images'] / median if median else float('inf'),
            'predictions_per_second': (
                counts['prediction_annotations'] / median
                if median else float('inf')),
            'peak_rss_bytes': mem_measure.get('peak_rss_bytes'),
            'peak_rss_delta_bytes': mem_measure.get('peak_rss_delta_bytes'),
            'peak_rss_delta_max_bytes': mem_measure.get('peak_rss_delta_max_bytes'),
            'memory_repeat': mem_measure.get('memory_repeat'),
            'python_peak_tracemalloc_bytes': mem_measure.get('python_peak_tracemalloc_bytes'),
            'AP': record['result']['summary'].get('AP'),
            'AP50': record['result']['summary'].get('AP50'),
            'AR100': record['result']['summary'].get('AR100'),
            'implementation': record['result'].get('implementation'),
            'accumulator': record['result'].get('accumulator'),
            'candidate_build_seconds': internal_timing.get('candidate_build_seconds'),
            'assignment_seconds': internal_timing.get('assignment_seconds'),
            'accumulation_seconds': internal_timing.get('accumulation_seconds'),
            'reported_native_total_seconds': internal_timing.get('total_seconds'),
            'assignment_batches': internal_timing.get('assignment_batches'),
            'candidate_predictions_before_maxdet': internal_timing.get(
                'candidate_predictions_before_maxdet'),
            'candidate_predictions_after_maxdet': internal_timing.get(
                'candidate_predictions_after_maxdet'),
            'parity_ok': None if parity is None else parity['ok'],
            'parity_max_abs_delta': None if parity is None else parity['max_abs_delta'],
        })

    official = {
        (r['scenario'], r['iou_type'], r['scope'], r['eval_mode']): r
        for r in rows if r['backend'] == 'official'
    }
    for row in rows:
        ref = official.get((row['scenario'], row['iou_type'], row['scope'], row['eval_mode']))
        if ref is None or row['median_seconds'] <= 0:
            row['runtime_ratio_vs_official'] = None
            row['speedup_vs_official'] = None
            row['rss_ratio_vs_official'] = None
            continue
        row['runtime_ratio_vs_official'] = row['median_seconds'] / ref['median_seconds']
        row['speedup_vs_official'] = ref['median_seconds'] / row['median_seconds']
        a = row.get('peak_rss_delta_bytes')
        b = ref.get('peak_rss_delta_bytes')
        row['rss_ratio_vs_official'] = None if not a or not b else a / b
    return rows


def _fmt_seconds(value):
    if value is None:
        return '-'
    if value < 1e-3:
        return f'{value * 1e6:.1f} us'
    if value < 1:
        return f'{value * 1e3:.1f} ms'
    return f'{value:.3f} s'


def _fmt_mb(value):
    if value is None:
        return '-'
    return f'{value / (1024 ** 2):.1f} MB'


def _fmt_ratio(value, suffix='x'):
    if value is None:
        return '-'
    return f'{value:.3f}{suffix}'


def _markdown_table(headers, rows):
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(map(str, row)) + ' |')
    return '\n'.join(lines)


def _render_report(output_dpath, env, rows, parity_records, conformance_records, command_line):
    native_rows = [r for r in rows if r['backend'].startswith('native-')]
    rust_rows = [r for r in native_rows if r['backend'] == 'native-rust']
    rust_speedups = [
        r['speedup_vs_official'] for r in rust_rows
        if r['speedup_vs_official'] is not None
    ]
    rust_rss_ratios = [
        r['rss_ratio_vs_official'] for r in rust_rows
        if r['rss_ratio_vs_official'] is not None
    ]
    all_parity = bool(parity_records) and all(r['ok'] for r in parity_records)
    conformance_ok = (
        None if not conformance_records
        else all(r.get('returncode') == 0 for r in conformance_records)
    )

    lines = []
    lines.append('# kwcoco native COCO release benchmark')
    lines.append('')
    lines.append('This report was generated from machine-readable benchmark artifacts. '
                 'Numbers describe this machine, these package revisions, and these workloads; '
                 'they are not universal performance claims.')
    lines.append('')
    lines.append('## Evidence summary')
    lines.append('')
    if rust_speedups:
        lines.append(
            f'- Rust-backed native median speedup versus pycocotools across the '
            f'reported cells: **{statistics.median(rust_speedups):.2f}x** '
            f'(range {min(rust_speedups):.2f}x to {max(rust_speedups):.2f}x).')
    if rust_rss_ratios:
        lines.append(
            f'- Rust-backed scorer peak-RSS-delta ratio versus pycocotools: '
            f'**{statistics.median(rust_rss_ratios):.2f}x** median '
            f'(lower is better; range {min(rust_rss_ratios):.2f}x to {max(rust_rss_ratios):.2f}x).')
    lines.append(f'- Large-workload full-tensor parity checks: **{"PASS" if all_parity else "FAIL / incomplete"}**.')
    if conformance_ok is not None:
        lines.append(f'- Adversarial Phase-3 conformance campaign: **{"PASS" if conformance_ok else "FAIL"}**.')
    if all_parity and conformance_ok is True:
        lines.append(
            '- Release evidence gate: **READY** for measured performance '
            'claims with the methodology/caveats below.')
    elif conformance_ok is None:
        lines.append(
            '- Release evidence gate: **INCOMPLETE** because the adversarial '
            'conformance campaign was not included in this run.')
    else:
        lines.append(
            '- Release evidence gate: **NOT READY**; resolve parity/conformance '
            'failures before publishing performance claims.')
    lines.append('')

    if rust_rows:
        lines.append('### Rust-backed aggregate speed / memory by measurement class')
        lines.append('')
        grouped = {}
        for row in rust_rows:
            group_key = (row['iou_type'], row['scope'], row['eval_mode'])
            grouped.setdefault(group_key, []).append(row)
        aggregate_rows = []
        for (iou_type, scope, eval_mode), group_rows in sorted(grouped.items()):
            speeds = [
                r['speedup_vs_official'] for r in group_rows
                if r['speedup_vs_official'] is not None
            ]
            rss = [
                r['rss_ratio_vs_official'] for r in group_rows
                if r['rss_ratio_vs_official'] is not None
            ]
            aggregate_rows.append([
                iou_type, scope, eval_mode, len(group_rows),
                '-' if not speeds else f'{statistics.median(speeds):.3f}x',
                '-' if not speeds else f'{min(speeds):.3f}x .. {max(speeds):.3f}x',
                '-' if not rss else f'{statistics.median(rss):.3f}x',
                '-' if not rss else f'{min(rss):.3f}x .. {max(rss):.3f}x',
            ])
        lines.append(_markdown_table(
            ['IoU', 'Scope', 'Output', 'Cells', 'Median speedup', 'Speedup range',
             'Median RSS ratio', 'RSS ratio range'], aggregate_rows))
        lines.append('')

    lines.append('## Runtime and memory')
    lines.append('')
    table_rows = []
    for r in rows:
        table_rows.append([
            r['scenario'], r['iou_type'], r['scope'], r['eval_mode'], r['backend'],
            _fmt_seconds(r['median_seconds']),
            _fmt_ratio(r['runtime_ratio_vs_official']),
            _fmt_ratio(r['speedup_vs_official']),
            _fmt_mb(r['peak_rss_bytes']),
            _fmt_mb(r['peak_rss_delta_bytes']),
            _fmt_ratio(r['rss_ratio_vs_official']),
            f"{r['images_per_second']:.1f}",
            'PASS' if r['parity_ok'] else ('-' if r['parity_ok'] is None else 'FAIL'),
        ])
    lines.append(_markdown_table(
        ['Scenario', 'IoU', 'Scope', 'Output', 'Backend', 'Median',
         'Runtime / official', 'Speedup', 'Peak RSS', 'Peak RSS delta',
         'RSS / official', 'Images/s', 'Parity'], table_rows))
    lines.append('')

    stage_rows = []
    for r in rows:
        if r['backend'] != 'native-rust' or r['scope'] != 'core':
            continue
        candidate = r.get('candidate_build_seconds')
        assignment = r.get('assignment_seconds')
        accumulation = r.get('accumulation_seconds')
        total = r.get('reported_native_total_seconds')
        if total in {None, 0}:
            continue
        stage_rows.append([
            r['scenario'], r['iou_type'], r['eval_mode'],
            f'{100 * candidate / total:.1f}%' if candidate is not None else '-',
            f'{100 * assignment / total:.1f}%' if assignment is not None else '-',
            f'{100 * accumulation / total:.1f}%' if accumulation is not None else '-',
            r.get('assignment_batches', '-'),
        ])
    if stage_rows:
        lines.append('## Native Rust stage decomposition')
        lines.append('')
        lines.append(_markdown_table(
            ['Scenario', 'IoU', 'Output', 'Candidate build', 'Assignment',
             'Accumulation', 'Assignment batches'], stage_rows))
        lines.append('')
        lines.append(
            'Stage percentages come from kwcoco internal instrumentation on the '
            'last timed native result; the isolated-process repeated wall-clock '
            'median remains the primary performance number.')
        lines.append('')

    lines.append('## Workload scale')
    lines.append('')
    seen = set()
    scale_rows = []
    for r in rows:
        if r['scenario'] in seen:
            continue
        seen.add(r['scenario'])
        scale_rows.append([
            r['scenario'], r['images'], r['truth_annotations'],
            r['prediction_annotations'], r['categories'],
        ])
    lines.append(_markdown_table(
        ['Scenario', 'Images', 'Truth instances', 'Predictions', 'Categories'],
        scale_rows))
    lines.append('')

    lines.append('## Quality / parity')
    lines.append('')
    parity_rows = []
    for item in parity_records:
        parity_rows.append([
            item['scenario'], item['iou_type'], item['backend'],
            'PASS' if item['ok'] else 'FAIL',
            f"{item['max_abs_delta']:.3g}",
            item['tensor'].get('keys', {}).get('precision', {}).get('populated_cells', '-'),
        ])
    if parity_rows:
        lines.append(_markdown_table(
            ['Scenario', 'IoU', 'Backend', 'Result', 'Max abs delta', 'Precision cells'],
            parity_rows))
    else:
        lines.append('No large-workload parity runs were requested.')
    lines.append('')
    lines.append(
        'The release benchmark parity check compares the complete accumulated '
        'precision, recall, sampled-score, and real-positive tensors. The separate '
        'Phase-3 conformance suite covers adversarial semantics such as crowd IoA, '
        'IoU/score ties, area boundaries, empty cases, custom grids, and rich mask geometry.')
    lines.append('')

    lines.append('## Memory methodology')
    lines.append('')
    lines.append(
        'Each backend runs in an isolated Python process. Timing and memory are '
        'measured in separate passes so RSS polling does not contaminate headline '
        'timings. Peak RSS is sampled from the process during scoring and therefore '
        'includes Python, NumPy, pycocotools C allocations, and kwimage_ext Rust '
        'allocations. `Peak RSS delta` subtracts the resident set immediately before '
        'the scorer call, after the deterministic dataset and any core metric object '
        'have been constructed. Optional tracemalloc numbers are Python-only and are '
        'not used as the primary memory comparison.')
    lines.append('')

    lines.append('## Environment')
    lines.append('')
    machine = env.get('machine', {})
    env_rows = [
        ['Python', env.get('python', '').splitlines()[0]],
        ['Platform', env.get('platform')],
        ['CPU', machine.get('cpu_model')],
        [
            'Physical / logical CPUs',
            f"{machine.get('physical_cpus')} / {machine.get('logical_cpus')}"],
        ['RAM', _fmt_mb(machine.get('total_ram_bytes'))],
        ['CPU governor', env.get('cpu_governor')],
        ['kwcoco', env['packages'].get('kwcoco')],
        ['kwimage_ext', env['packages'].get('kwimage_ext')],
        ['pycocotools', env['packages'].get('pycocotools')],
        ['numpy', env['packages'].get('numpy')],
        ['Rust capabilities', ', '.join(env.get('kwimage_ext_rust_capabilities') or [])],
    ]
    lines.append(_markdown_table(['Field', 'Value'], env_rows))
    lines.append('')

    lines.append('## Reproduction')
    lines.append('')
    lines.append('```bash')
    lines.append(shlex.join(command_line))
    lines.append('```')
    lines.append('')
    lines.append('Raw timing samples, memory records, parity tensors/diffs, environment '
                 'metadata, CSV summaries, and optional conformance bundles are stored '
                 'beside this report.')
    lines.append('')

    lines.append('## Interpretation guardrails')
    lines.append('')
    lines.append('- Treat speed and memory results as workload- and machine-specific measurements.')
    lines.append('- Keep pycocotools identified as the official COCO reference implementation.')
    lines.append('- A parity PASS means equality within the configured numerical tolerance on these workloads; it is not a proof over all possible inputs.')
    lines.append('- Use real-project truth/prediction files in addition to synthetic scaling workloads before publishing application-specific claims.')
    lines.append('- Report package versions, git revisions, CPU, RAM, output mode, and backend provenance with any quoted benchmark number.')
    lines.append('')

    report_path = output_dpath / 'release_benchmark_report.md'
    report_path.write_text('\n'.join(lines) + '\n')
    return report_path


def _write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    keys = list(rows[0].keys())
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _run_conformance(output_dpath, mode, backend):
    subdir = output_dpath / f'conformance_{backend}'
    command = [
        sys.executable, os.fspath(CONFORMANCE),
        '--mode', mode,
        '--assignment-backend', backend,
        '--assert-parity',
        '--output-dpath', os.fspath(subdir),
    ]
    proc = subprocess.run(
        command, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (subdir.parent / f'conformance_{backend}.log').write_text(proc.stdout)
    return {
        'backend': backend,
        'mode': mode,
        'returncode': int(proc.returncode),
        'command': command,
        'output_dpath': os.fspath(subdir),
    }


def _make_bundle(output_dpath):
    bundle = output_dpath / 'kwcoco_metrics_release_evidence.tar.gz'
    with tarfile.open(bundle, 'w:gz') as tar:
        for path in sorted(output_dpath.rglob('*')):
            if path == bundle or not path.is_file():
                continue
            tar.add(path, arcname=path.relative_to(output_dpath))
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--suite', choices=sorted(SUITES), default='quick')
    parser.add_argument('--truth-coco', type=Path)
    parser.add_argument('--pred-coco', type=Path)
    parser.add_argument('--scenario-name', default='real_dataset')
    parser.add_argument('--backends', default='official,native-rust,native-python')
    parser.add_argument('--parity-backends', default='native-rust,native-python')
    parser.add_argument('--iou-types', default='bbox,segm')
    parser.add_argument('--scopes', default='core')
    parser.add_argument('--eval-modes', default='tensor')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--memory', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--python-allocations', action='store_true')
    parser.add_argument('--memory-sample-interval', type=float, default=0.01)
    parser.add_argument('--memory-repeat', type=int, default=1)
    parser.add_argument('--parity', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--parity-atol', type=float, default=1e-12)
    parser.add_argument(
        '--conformance-mode', choices=['none', 'quick', 'full', 'stress'],
        default='none')
    parser.add_argument('--output-dpath', type=Path, default=Path('_metric_release_benchmark'))
    parser.add_argument(
        '--resume', action=argparse.BooleanOptionalAction, default=False,
        help='reuse compatible completed worker records in output-dpath')
    args = parser.parse_args(argv)

    output_dpath = args.output_dpath.resolve()
    output_dpath.mkdir(parents=True, exist_ok=True)
    records_dpath = output_dpath / 'records'
    parity_dpath = output_dpath / 'parity'
    specs_dpath = output_dpath / 'specs'
    for dpath in [records_dpath, parity_dpath, specs_dpath]:
        dpath.mkdir(parents=True, exist_ok=True)

    if (args.truth_coco is None) ^ (args.pred_coco is None):
        parser.error('--truth-coco and --pred-coco must be given together')
    if args.truth_coco is not None:
        scenarios = [{
            'name': args.scenario_name,
            'kind': 'files',
            'truth_coco': os.fspath(args.truth_coco.resolve()),
            'pred_coco': os.fspath(args.pred_coco.resolve()),
        }]
    else:
        scenarios = [dict(s) for s in SUITES[args.suite]]

    backends = _parse_csv_arg(args.backends)
    parity_backends = _parse_csv_arg(args.parity_backends)
    iou_types = _parse_csv_arg(args.iou_types)
    scopes = _parse_csv_arg(args.scopes)
    eval_modes = _parse_csv_arg(args.eval_modes)
    for backend in backends + parity_backends:
        if backend not in {'official', 'native-rust', 'native-python', 'native-auto'}:
            parser.error(f'unknown backend={backend!r}')
    for iou_type in iou_types:
        if iou_type not in {'bbox', 'segm'}:
            parser.error(f'unknown iou_type={iou_type!r}')
    for scope in scopes:
        if scope not in {'core', 'evaluator'}:
            parser.error(f'unknown scope={scope!r}')
    for eval_mode in eval_modes:
        if eval_mode not in {'summary', 'tensor'}:
            parser.error(f'unknown eval_mode={eval_mode!r}')

    env = environment_info()
    (output_dpath / 'environment.json').write_text(json.dumps(env, indent=2) + '\n')
    (output_dpath / 'command.txt').write_text(shlex.join([sys.executable, *sys.argv]) + '\n')

    spec_paths = {}
    for spec in scenarios:
        path = specs_dpath / f"{spec['name']}.json"
        path.write_text(json.dumps(spec, indent=2) + '\n')
        spec_paths[spec['name']] = path

    timing_records = []
    memory_records = []
    raw_jsonl = output_dpath / 'raw_records.jsonl'
    raw_file = raw_jsonl.open('a' if args.resume else 'w')
    try:
        total_jobs = (
            len(scenarios) * len(iou_types) * len(scopes) *
            len(eval_modes) * len(backends))
        job_index = 0
        for spec in scenarios:
            for iou_type in iou_types:
                for scope in scopes:
                    for eval_mode in eval_modes:
                        for backend in backends:
                            job_index += 1
                            stem = f"{spec['name']}__{iou_type}__{scope}__{eval_mode}__{backend}"
                            timing_path = records_dpath / f'{stem}__timing.json'
                            timing_record = None
                            if args.resume:
                                timing_record = _load_reusable_worker_record(
                                    timing_path,
                                    spec=spec, backend=backend, iou_type=iou_type,
                                    scope=scope, eval_mode=eval_mode, phase='timing',
                                    warmup=args.warmup, repeat=args.repeat,
                                )
                            if timing_record is None:
                                print(
                                    f'[{job_index}/{total_jobs}] timing {stem}',
                                    flush=True)
                                timing_record = _run_worker(
                                    spec_path=spec_paths[spec['name']],
                                    backend=backend, iou_type=iou_type,
                                    scope=scope, eval_mode=eval_mode, phase='timing',
                                    output_json=timing_path,
                                    warmup=args.warmup, repeat=args.repeat,
                                )
                                raw_file.write(json.dumps(timing_record) + '\n')
                                raw_file.flush()
                            else:
                                print(
                                    f'[{job_index}/{total_jobs}] reuse timing {stem}',
                                    flush=True)
                            timing_records.append(timing_record)
                            if args.memory:
                                for memory_idx in range(args.memory_repeat):
                                    memory_path = records_dpath / (
                                        f'{stem}__memory_{memory_idx:02d}.json')
                                    memory_record = None
                                    if args.resume:
                                        memory_record = _load_reusable_worker_record(
                                            memory_path,
                                            spec=spec, backend=backend,
                                            iou_type=iou_type, scope=scope,
                                            eval_mode=eval_mode, phase='memory',
                                            memory_sample_interval=(
                                                args.memory_sample_interval),
                                            python_allocations=(
                                                args.python_allocations),
                                        )
                                    if memory_record is None:
                                        print(
                                            f'[{job_index}/{total_jobs}] memory {stem} '
                                            f'({memory_idx + 1}/{args.memory_repeat})',
                                            flush=True)
                                        memory_record = _run_worker(
                                            spec_path=spec_paths[spec['name']],
                                            backend=backend, iou_type=iou_type,
                                            scope=scope, eval_mode=eval_mode,
                                            phase='memory', output_json=memory_path,
                                            memory_sample_interval=(
                                                args.memory_sample_interval),
                                            python_allocations=(
                                                args.python_allocations),
                                        )
                                        raw_file.write(
                                            json.dumps(memory_record) + '\n')
                                        raw_file.flush()
                                    else:
                                        print(
                                            f'[{job_index}/{total_jobs}] reuse memory '
                                            f'{stem} ({memory_idx + 1}/'
                                            f'{args.memory_repeat})', flush=True)
                                    memory_record['memory_iteration'] = memory_idx
                                    memory_records.append(memory_record)
    finally:
        raw_file.close()

    parity_records = []
    if args.parity:
        for spec in scenarios:
            for iou_type in iou_types:
                print(f'parity {spec["name"]} {iou_type}', flush=True)
                ref_json = parity_dpath / f"{spec['name']}__{iou_type}__official.json"
                ref_npz = parity_dpath / f"{spec['name']}__{iou_type}__official.npz"
                official = None
                if args.resume and ref_npz.exists():
                    official = _load_reusable_worker_record(
                        ref_json, spec=spec, backend='official',
                        iou_type=iou_type, scope='core', eval_mode='tensor',
                        phase='parity')
                if official is None:
                    official = _run_worker(
                        spec_path=spec_paths[spec['name']], backend='official',
                        iou_type=iou_type, scope='core', eval_mode='tensor',
                        phase='parity', output_json=ref_json, output_npz=ref_npz,
                    )
                else:
                    print(
                        f'reuse parity worker {spec["name"]} {iou_type} official',
                        flush=True)
                for backend in parity_backends:
                    native_json = parity_dpath / f"{spec['name']}__{iou_type}__{backend}.json"
                    native_npz = parity_dpath / f"{spec['name']}__{iou_type}__{backend}.npz"
                    native = None
                    if args.resume and native_npz.exists():
                        native = _load_reusable_worker_record(
                            native_json, spec=spec, backend=backend,
                            iou_type=iou_type, scope='core', eval_mode='tensor',
                            phase='parity')
                    if native is None:
                        native = _run_worker(
                            spec_path=spec_paths[spec['name']], backend=backend,
                            iou_type=iou_type, scope='core', eval_mode='tensor',
                            phase='parity', output_json=native_json,
                            output_npz=native_npz,
                        )
                    else:
                        print(
                            f'reuse parity worker {spec["name"]} {iou_type} '
                            f'{backend}', flush=True)
                    summary = _summary_diff(native, official, args.parity_atol)
                    tensor = _tensor_diff(native_npz, ref_npz, args.parity_atol)
                    max_abs = max(
                        summary['max_abs_delta'],
                        max([
                            info.get('max_abs_delta', 0.0)
                            for info in tensor['keys'].values()
                        ] or [0.0]),
                    )
                    item = {
                        'scenario': spec['name'],
                        'iou_type': iou_type,
                        'backend': backend,
                        'ok': bool(summary['ok'] and tensor['ok']),
                        'max_abs_delta': float(max_abs),
                        'summary': summary,
                        'tensor': tensor,
                    }
                    parity_records.append(item)
                    # Persist after every comparison so a later interruption
                    # never discards already-computed parity evidence.
                    (output_dpath / 'parity_results.json').write_text(
                        json.dumps(parity_records, indent=2) + '\n')

    conformance_records = []
    conformance_path = output_dpath / 'conformance_results.json'
    reusable_conformance = {}
    if args.resume and conformance_path.exists():
        for item in json.loads(conformance_path.read_text()):
            key = (item.get('backend'), item.get('mode'))
            if item.get('returncode') == 0:
                reusable_conformance[key] = item
    if args.conformance_mode != 'none':
        for backend in ['python', 'kwimage_ext']:
            key = (backend, args.conformance_mode)
            item = reusable_conformance.get(key, None)
            if item is None:
                print(f'conformance {backend} {args.conformance_mode}', flush=True)
                item = _run_conformance(
                    output_dpath, args.conformance_mode, backend)
            else:
                print(
                    f'reuse conformance {backend} {args.conformance_mode}',
                    flush=True)
            conformance_records.append(item)
            conformance_path.write_text(
                json.dumps(conformance_records, indent=2) + '\n')

    rows = _summarize_records(timing_records, memory_records, parity_records)
    (output_dpath / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    _write_csv(output_dpath / 'summary.csv', rows)
    report = _render_report(
        output_dpath, env, rows, parity_records, conformance_records, [sys.executable, *sys.argv])
    bundle = _make_bundle(output_dpath)

    parity_ok = (not args.parity) or all(item['ok'] for item in parity_records)
    conformance_ok = all(item['returncode'] == 0 for item in conformance_records)
    print(f'report: {report}')
    print(f'bundle: {bundle}')
    print(f'parity: {"ok" if parity_ok else "FAILED"}')
    if conformance_records:
        print(f'conformance: {"ok" if conformance_ok else "FAILED"}')
    return 0 if parity_ok and conformance_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
