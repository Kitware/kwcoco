#!/usr/bin/env python3
"""Cross-version evidence for KWCoco's historical detection metric contract.

This complements ``run_release_benchmark.py``.  The release benchmark compares
current strict/native COCO implementations against the official pycocotools
reference.  This runner instead compares the *historical KWCoco public
``CocoEvaluator.evaluate()`` contract* across git revisions using exactly the
same frozen truth/prediction JSON and the same Python dependency environment.

The default comparison is the released v0.9.0 tag against the current worktree.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORKER = HERE / 'version_timeline_worker.py'


def _run(command, *, cwd=None, env=None, check=True):
    return subprocess.run(
        list(map(os.fspath, command)), cwd=cwd, env=env, check=check,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _git_revision(root, ref):
    proc = _run(['git', 'rev-parse', ref], cwd=root)
    return proc.stdout.strip()


def _export_ref(root, ref, dpath):
    """Export a committed git revision without modifying caller worktrees."""
    dpath.mkdir(parents=True, exist_ok=True)
    archive_fpath = dpath.parent / f'{dpath.name}.tar'
    with open(archive_fpath, 'wb') as file:
        proc = subprocess.run(
            ['git', 'archive', '--format=tar', ref], cwd=root,
            check=False, stdout=file, stderr=subprocess.PIPE)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode('utf8', errors='replace'))
    with tarfile.open(archive_fpath, 'r') as tar:
        try:
            # The archive was just produced from this trusted local git repo,
            # but use the restrictive extraction policy when Python supports it.
            tar.extractall(dpath, filter='data')
        except TypeError:
            # Python < 3.12 has no extraction filter argument.
            tar.extractall(dpath)
    archive_fpath.unlink()
    return dpath


def _stable_jitter(seed):
    # Small deterministic pseudo-random value without depending on numpy or the
    # kwcoco version being measured.
    value = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
    return ((value / 2 ** 32) - 0.5) * 2.0


def make_frozen_coco_pair(n_images=128, n_classes=4, truth_per_image=4, seed=0):
    """Create one deterministic bbox workload using only builtin Python."""
    categories = [
        {'id': cid, 'name': f'class_{cid}'}
        for cid in range(1, n_classes + 1)
    ]
    images = [
        {'id': gid, 'file_name': f'image_{gid:06d}.png', 'width': 1024, 'height': 768}
        for gid in range(1, n_images + 1)
    ]
    truth_anns = []
    pred_anns = []
    taid = 1
    paid = 1
    for gid in range(1, n_images + 1):
        for objx in range(truth_per_image):
            cid = 1 + ((gid + objx) % n_classes)
            base = seed + gid * 101 + objx * 17
            x = float((gid * 37 + objx * 83) % 850)
            y = float((gid * 53 + objx * 47) % 620)
            w = float(30 + ((gid * 11 + objx * 7) % 100))
            h = float(25 + ((gid * 13 + objx * 5) % 90))
            truth_anns.append({
                'id': taid, 'image_id': gid, 'category_id': cid,
                'bbox': [x, y, w, h], 'area': w * h,
            })
            taid += 1

            # Mostly-correct prediction with deterministic localization noise.
            dx = _stable_jitter(base) * 5.0
            dy = _stable_jitter(base + 1) * 5.0
            dw = _stable_jitter(base + 2) * 2.0
            dh = _stable_jitter(base + 3) * 2.0
            pcid = cid if ((gid + objx) % 11) else 1 + (cid % n_classes)
            score = 0.55 + 0.44 * (((base * 2654435761) & 0xFFFF) / 0xFFFF)
            pred_anns.append({
                'id': paid, 'image_id': gid, 'category_id': pcid,
                'bbox': [x + dx, y + dy, max(2.0, w + dw), max(2.0, h + dh)],
                'score': float(score),
            })
            paid += 1

        # Add false positives and leave deterministic negative pressure.
        if gid % 3 == 0:
            cid = 1 + (gid % n_classes)
            x = float((gid * 71) % 900)
            y = float((gid * 29) % 650)
            w = float(20 + (gid % 60))
            h = float(18 + (gid % 50))
            pred_anns.append({
                'id': paid, 'image_id': gid, 'category_id': cid,
                'bbox': [x, y, w, h],
                'score': float(0.15 + (gid % 50) / 100.0),
            })
            paid += 1

    truth = {
        'info': {'description': 'kwcoco historical metric version benchmark'},
        'images': images,
        'categories': categories,
        'annotations': truth_anns,
    }
    pred = {
        'info': {'description': 'kwcoco historical metric version benchmark predictions'},
        'images': images,
        'categories': categories,
        'annotations': pred_anns,
    }
    return truth, pred


def _write_workload(output_dpath, n_images, n_classes, truth_per_image, seed):
    truth, pred = make_frozen_coco_pair(
        n_images=n_images, n_classes=n_classes,
        truth_per_image=truth_per_image, seed=seed)
    workload_dpath = output_dpath / 'workload'
    workload_dpath.mkdir(parents=True, exist_ok=True)
    truth_fpath = workload_dpath / 'truth.kwcoco.json'
    pred_fpath = workload_dpath / 'pred.kwcoco.json'
    truth_fpath.write_text(json.dumps(truth, indent=2) + '\n')
    pred_fpath.write_text(json.dumps(pred, indent=2) + '\n')
    counts = {
        'images': len(truth['images']),
        'truth_annotations': len(truth['annotations']),
        'prediction_annotations': len(pred['annotations']),
        'categories': len(truth['categories']),
    }
    (workload_dpath / 'counts.json').write_text(json.dumps(counts, indent=2) + '\n')
    return truth_fpath, pred_fpath, counts


def _iter_numeric_diffs(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a) | set(b))
        for key in keys:
            child = f'{path}.{key}' if path else str(key)
            if key not in a or key not in b:
                yield {'path': child, 'kind': 'missing', 'a': a.get(key), 'b': b.get(key)}
            else:
                yield from _iter_numeric_diffs(a[key], b[key], child)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield {'path': path, 'kind': 'length', 'a': len(a), 'b': len(b)}
        for idx, (aa, bb) in enumerate(zip(a, b)):
            yield from _iter_numeric_diffs(aa, bb, f'{path}[{idx}]')
        return
    numeric = isinstance(a, (int, float)) and not isinstance(a, bool)
    numeric = numeric and isinstance(b, (int, float)) and not isinstance(b, bool)
    if numeric:
        af = float(a)
        bf = float(b)
        if math.isnan(af) and math.isnan(bf):
            return
        delta = abs(af - bf)
        if delta != 0:
            yield {'path': path, 'kind': 'numeric', 'a': af, 'b': bf, 'abs_delta': delta}
    elif a != b:
        yield {'path': path, 'kind': 'value', 'a': a, 'b': b}


def compare_metric_states(a, b, atol=0.0, max_examples=20):
    diffs = list(_iter_numeric_diffs(a, b))
    numeric = [d for d in diffs if d['kind'] == 'numeric']
    structural = [d for d in diffs if d['kind'] != 'numeric']
    max_abs = max([d['abs_delta'] for d in numeric], default=0.0)
    bad_numeric = [d for d in numeric if d['abs_delta'] > atol]
    bad = structural + bad_numeric
    return {
        'ok': not bad,
        'atol': float(atol),
        'max_abs_delta': float(max_abs),
        'num_numeric_differences': len(numeric),
        'num_structural_differences': len(structural),
        'num_differences_over_tolerance': len(bad),
        'examples': bad[:max_examples],
    }


def _worker_env(source_root):
    env = os.environ.copy()
    old = env.get('PYTHONPATH')
    env['PYTHONPATH'] = os.fspath(source_root) + (os.pathsep + old if old else '')
    return env


def _run_worker(source_root, label, truth_fpath, pred_fpath, output_json, warmup, repeat):
    command = [
        sys.executable, WORKER,
        '--truth-coco', truth_fpath,
        '--pred-coco', pred_fpath,
        '--warmup', str(warmup),
        '--repeat', str(repeat),
        '--source-label', label,
        '--output-json', output_json,
    ]
    proc = _run(command, cwd=ROOT, env=_worker_env(source_root), check=False)
    if proc.returncode:
        return {
            'status': 'error',
            'source_label': label,
            'source_root': os.fspath(source_root),
            'command': list(map(os.fspath, command)),
            'output': proc.stdout,
        }
    record = json.loads(output_json.read_text())
    record['status'] = 'ok'
    record['source_root'] = os.fspath(source_root)
    return record


def _fmt(value, digits=4):
    if value is None:
        return '-'
    return f'{value:.{digits}f}'


def _capability_text(value):
    if isinstance(value, list):
        return ', '.join(map(str, value)) if value else '-'
    return 'yes' if value else 'no'


def _render_report(output_dpath, records, parity, counts, baseline_label):
    lines = [
        '# KWCoco metric version timeline evidence',
        '',
        'This report compares the historical `CocoEvaluator.evaluate()` contract '
        'across source revisions on one frozen bbox workload. All revisions are '
        'run with the same Python executable and dependency environment; only the '
        '`kwcoco` source tree changes.',
        '',
        '## Workload',
        '',
        '| field | value |',
        '| --- | ---: |',
    ]
    for key, value in counts.items():
        lines.append(f'| {key} | {value} |')

    lines += ['', '## Historical evaluator runtime', '']
    baseline = next(
        (r for r in records
         if r['source_label'] == baseline_label and r['status'] == 'ok'),
        None,
    )
    for scope in ['prepared_evaluator', 'end_to_end']:
        lines += [
            f'### {scope}', '',
            '| source | kwcoco | median s | speedup vs baseline | status |',
            '| --- | --- | ---: | ---: | --- |',
        ]
        baseline_time = (
            None if baseline is None
            else baseline['timing'][scope]['median_seconds']
        )
        for record in records:
            if record['status'] != 'ok':
                lines.append(
                    f"| {record['source_label']} | - | - | - | ERROR |")
                continue
            median = record['timing'][scope]['median_seconds']
            speedup = (
                None if not baseline_time or not median
                else baseline_time / median
            )
            lines.append(
                f"| {record['source_label']} | {record['kwcoco_version']} | "
                f"{_fmt(median, 6)} | {_fmt(speedup, 3)}x | ok |")
        lines.append('')

    lines += ['', '## Historical score compatibility', '',
              '| candidate | baseline | parity | max abs delta | differing values |',
              '| --- | --- | --- | ---: | ---: |']
    for item in parity:
        lines.append(
            f"| {item['candidate']} | {item['baseline']} | "
            f"{'PASS' if item['comparison']['ok'] else 'FAIL'} | "
            f"{item['comparison']['max_abs_delta']:.3g} | "
            f"{item['comparison']['num_differences_over_tolerance']} |")

    capabilities = sorted({
        key for record in records if record['status'] == 'ok'
        for key in record['capabilities']
    })
    lines += ['', '## Capability matrix', '',
              '| capability | ' + ' | '.join(r['source_label'] for r in records) + ' |',
              '| --- | ' + ' | '.join(['---'] * len(records)) + ' |']
    for capability in capabilities:
        values = []
        for record in records:
            if record['status'] != 'ok':
                values.append('ERROR')
            else:
                values.append(_capability_text(record['capabilities'].get(capability)))
        lines.append('| ' + capability + ' | ' + ' | '.join(values) + ' |')

    errors = [r for r in records if r['status'] != 'ok']
    if errors:
        lines += ['', '## Worker errors', '']
        for record in errors:
            lines += [
                f"### {record['source_label']}", '', '```text',
                record['output'].rstrip(), '```', '',
            ]

    lines += [
        '',
        '## Interpretation',
        '',
        '* Historical parity covers the complete serialized `nocls_measures` and '
        '  `ovr_measures` trees, including threshold curves and counts, not just mAP.',
        '* Runtime numbers are an A/B measurement of the historical evaluator only. '
        '  They do not measure the new strict native COCO evaluator.',
        '* Use `run_release_benchmark.py` for current native Python/Rust versus '
        '  official pycocotools bbox+segm timing, memory, and full tensor parity.',
        '* A cross-version timing claim is only meaningful when the worker records '
        '  show the same Python and shared numerical dependency versions.',
    ]
    report_fpath = output_dpath / 'version_timeline_report.md'
    report_fpath.write_text('\n'.join(lines) + '\n')
    return report_fpath


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--refs', default='v0.9.0,current',
                        help='comma-separated git refs; "current" is the worktree')
    parser.add_argument('--baseline', default='v0.9.0')
    parser.add_argument('--n-images', type=int, default=128)
    parser.add_argument('--n-classes', type=int, default=4)
    parser.add_argument('--truth-per-image', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--atol', type=float, default=0.0)
    parser.add_argument('--output-dpath', type=Path,
                        default=Path('_metric_version_timeline'))
    args = parser.parse_args(argv)

    refs = [p.strip() for p in args.refs.split(',') if p.strip()]
    if args.baseline not in refs:
        parser.error('--baseline must be present in --refs')

    output_dpath = args.output_dpath.absolute()
    output_dpath.mkdir(parents=True, exist_ok=True)
    truth_fpath, pred_fpath, counts = _write_workload(
        output_dpath, args.n_images, args.n_classes,
        args.truth_per_image, args.seed)

    sources_dpath = output_dpath / 'sources'
    records_dpath = output_dpath / 'records'
    sources_dpath.mkdir(exist_ok=True)
    records_dpath.mkdir(exist_ok=True)

    records = []
    revisions = {}
    for ref in refs:
        label = ref
        if ref == 'current':
            source_root = ROOT
            try:
                revision = _git_revision(ROOT, 'HEAD')
            except Exception:
                revision = None
        else:
            revision = _git_revision(ROOT, ref)
            safe = ref.replace('/', '_').replace(os.sep, '_')
            source_root = sources_dpath / safe
            if source_root.exists():
                shutil.rmtree(source_root)
            _export_ref(ROOT, ref, source_root)
        revisions[label] = revision
        output_json = records_dpath / f"{label.replace('/', '_')}.json"
        print(f'benchmark {label}', flush=True)
        record = _run_worker(
            source_root, label, truth_fpath, pred_fpath,
            output_json, args.warmup, args.repeat)
        record['revision'] = revision
        records.append(record)

    good = {r['source_label']: r for r in records if r['status'] == 'ok'}
    parity = []
    baseline = good.get(args.baseline)
    if baseline is not None:
        for label, record in good.items():
            if label == args.baseline:
                continue
            comparison = compare_metric_states(
                baseline['metric_state'], record['metric_state'], atol=args.atol)
            parity.append({
                'baseline': args.baseline,
                'candidate': label,
                'comparison': comparison,
            })

    summary = {
        'schema': 'kwcoco_metric_version_timeline_v1',
        'baseline': args.baseline,
        'refs': refs,
        'revisions': revisions,
        'workload_counts': counts,
        'records': records,
        'parity': parity,
    }
    (output_dpath / 'version_timeline.json').write_text(
        json.dumps(summary, indent=2, allow_nan=True) + '\n')
    report = _render_report(output_dpath, records, parity, counts, args.baseline)
    print(f'report: {report}')

    failed_workers = any(r['status'] != 'ok' for r in records)
    failed_parity = any(not p['comparison']['ok'] for p in parity)
    return 1 if failed_workers or failed_parity else 0


if __name__ == '__main__':
    raise SystemExit(main())
