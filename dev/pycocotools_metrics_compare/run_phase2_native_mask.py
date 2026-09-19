#!/usr/bin/env python3
"""Phase-2 comparison of native kwcoco instance-mask assignment vs COCOeval.

This runner exercises the experimental ``assignment_v2`` segmentation path.
It keeps assignment and accumulation comparisons separate:

* kwcoco mask assignment -> kwcoco Measures
* COCOeval mask assignment -> kwcoco Measures
* COCOeval mask assignment -> COCOeval accumulation

The native path intentionally refuses crowd / ignore-region semantics for now;
those are measured and designed separately instead of silently using bbox-only
ignore behavior.
"""
from __future__ import annotations

import argparse
import importlib
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

from compare_metrics import CASE_SPECS
from compare_metrics import _assignment_diff
from compare_metrics import _native_measures_from_cfsn
from compare_metrics import make_case
from compare_metrics import timed_call


PROFILES = {
    'quick': {'nimgs': 4, 'image_size': (96, 96)},
    'full': {'nimgs': 32, 'image_size': (256, 256)},
}
THRESHOLDS = [0.50, 0.75, 0.95]
CASES = ['perfect', 'box_noise', 'class_noise', 'mask_only_shift', 'mixed']


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
        'overlap_backend': None,
    }
    try:
        from kwcoco.metrics.overlap import resolve_mask_iou_backend
        _func, meta = resolve_mask_iou_backend('auto')
    except Exception as ex:
        info['overlap_backend'] = {'status': 'error', 'error': repr(ex)}
    else:
        info['overlap_backend'] = {'status': 'ok', **meta}
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


def compare_case(case_name, profile, seed=0):
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    true_dset, pred_dset, spec = make_case(case_name, profile, seed)
    dmet, build_timing = timed_call(
        lambda: DetectionMetrics.from_coco(true_dset, pred_dset), repeat=1)

    native, native_timing = timed_call(
        lambda: dmet.confusion_vectors_v2(
            iou_thresh=THRESHOLDS,
            iou_type='segm',
            overlap_backend='auto',
            compat='mutex',
            ignore_classes=None,
            spatial_threshold=None,
        ),
        repeat=1,
    )

    official_debug, official_debug_timing = timed_call(
        lambda: dmet.score_pycocotools(
            iou_type='segm',
            iou_thresholds=THRESHOLDS,
            with_confusion=True,
            with_evaler=False,
            verbose=0,
        ),
        repeat=1,
    )
    official_standard, official_standard_timing = timed_call(
        lambda: dmet.score_pycocotools(
            iou_type='segm',
            with_confusion=False,
            with_evaler=False,
            verbose=0,
        ),
        repeat=1,
    )

    threshold_rows = {}
    official_cfsn_map = official_debug['iou_to_cfsn_vecs']
    for threshold in THRESHOLDS:
        native_cfsn = native[threshold]
        official_cfsn = official_cfsn_map[threshold]
        threshold_rows[str(threshold)] = {
            'assignment_diff': _assignment_diff(native_cfsn, official_cfsn),
            'native_assignment_native_measures': _native_measures_from_cfsn(
                native_cfsn),
            'official_assignment_native_measures': _native_measures_from_cfsn(
                official_cfsn),
        }

    native50 = threshold_rows['0.5']['native_assignment_native_measures']['mAP']
    off_native50 = threshold_rows['0.5']['official_assignment_native_measures']['mAP']
    off_official50 = float(official_standard['summary']['AP50'])
    decomposition = {
        'native_native_mAP50': native50,
        'official_native_mAP50': off_native50,
        'official_official_AP50': off_official50,
        'delta_assignment': native50 - off_native50,
        'delta_accumulation': off_native50 - off_official50,
        'delta_end_to_end': native50 - off_official50,
    }

    return {
        'case': case_name,
        'spec': spec,
        'dataset': {
            'n_images': len(true_dset.imgs),
            'n_true': len(true_dset.anns),
            'n_pred': len(pred_dset.anns),
        },
        'decomposition': decomposition,
        'thresholds': threshold_rows,
        'timing': {
            'detection_metrics_build': build_timing,
            'native_mask_assignment': native_timing,
            'official_mask_assignment_extract': official_debug_timing,
            'official_standard_cocoeval': official_standard_timing,
        },
    }


def build_report(mode):
    profile = dict(PROFILES[mode])
    report = {
        'schema_version': 1,
        'mode': mode,
        'created_unix': time.time(),
        'environment': environment_info(),
        'source_state': source_state(),
        'profile': profile,
        'thresholds': THRESHOLDS,
        'experiments': [],
    }
    for idx, case_name in enumerate(CASES):
        print(f'Running native-mask case {case_name!r}')
        try:
            result = compare_case(case_name, profile, seed=idx)
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


def summary_lines(report):
    lines = [
        '# kwcoco Phase-2 native instance-mask summary', '',
        f"Mode: `{report['mode']}`", '',
        '## Overlap backend', '', '```json',
        json.dumps(report['environment']['overlap_backend'], indent=2),
        '```', '',
        '| case | status | native/native mAP50 | official/native mAP50 | COCO AP50 | native-only matches @.50 | official-only matches @.50 | native sec |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for exp in report['experiments']:
        if exp['status'] != 'ok':
            lines.append(f"| `{exp['name']}` | error | | | | | | |")
            continue
        r = exp['result']
        d = r['decomposition']
        a = r['thresholds']['0.5']['assignment_diff']
        t = r['timing']['native_mask_assignment']['median_seconds']
        lines.append(
            f"| `{exp['name']}` | ok | {d['native_native_mAP50']:.6f} | "
            f"{d['official_native_mAP50']:.6f} | {d['official_official_AP50']:.6f} | "
            f"{a['num_native_only']} | {a['num_official_only']} | {t:.6f} |"
        )
    return lines


def write_bundle(report, output_dpath):
    dpath = Path(output_dpath)
    dpath.mkdir(parents=True, exist_ok=True)
    mode = report['mode']
    json_fpath = dpath / f'phase2_native_mask_{mode}_results.json'
    md_fpath = dpath / f'phase2_native_mask_{mode}_summary.md'
    json_fpath.write_text(json.dumps(report, indent=2, default=_json_default) + '\n')
    md_fpath.write_text('\n'.join(summary_lines(report)) + '\n')
    bundle = dpath / f'kwcoco_metrics_phase2_native_mask_{mode}.tar.gz'
    with tarfile.open(bundle, 'w:gz') as tar:
        tar.add(json_fpath, arcname=json_fpath.name)
        tar.add(md_fpath, arcname=md_fpath.name)
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=sorted(PROFILES), default='quick')
    parser.add_argument('--output-dpath', type=Path,
                        default=Path('_metric_compare_phase2_native_mask'))
    args = parser.parse_args(argv)
    report = build_report(args.mode)
    bundle = write_bundle(report, args.output_dpath)
    print('Phase-2 native-mask suite complete')
    print(f'  bundle: {bundle}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
