#!/usr/bin/env python3
"""Phase-2.1: localize native-vs-COCO mask assignment differences.

This experiment varies only two native semantics while holding the assignment
engine and mask-IoU kernel fixed:

* polygon rasterization: generic kwimage vs COCO ``frPoly`` convention;
* IoU threshold boundary: strict ``>`` vs inclusive ``>=``.

COCO-compatible rasterization is evaluated through both kwimage_ext (normally
Rust) and pycocotools so the Rust polygon codec is tested independently of the
assignment algorithm.
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
NATIVE_VARIANTS = {
    'kwimage_strict': {
        'segmentation_rasterizer': 'kwimage',
        'raster_backend': 'auto',
        'threshold_policy': 'strict',
    },
    'coco_rust_strict': {
        'segmentation_rasterizer': 'coco',
        'raster_backend': 'kwimage_ext',
        'threshold_policy': 'strict',
    },
    'coco_rust_inclusive': {
        'segmentation_rasterizer': 'coco',
        'raster_backend': 'kwimage_ext',
        'threshold_policy': 'inclusive',
    },
    'coco_pycoco_inclusive': {
        'segmentation_rasterizer': 'coco',
        'raster_backend': 'pycocotools',
        'threshold_policy': 'inclusive',
    },
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
    for name in ['kwimage_ext', 'pycocotools']:
        try:
            from kwcoco.metrics.overlap import resolve_coco_raster_backend
            _codec, meta = resolve_coco_raster_backend(name)
        except Exception as ex:
            info['backends']['raster_' + name] = {
                'status': 'unavailable', 'error': repr(ex)}
        else:
            info['backends']['raster_' + name] = {'status': 'ok', **meta}
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


def _assignment_examples(diff):
    examples = set()
    examples.update(map(tuple, diff['native_only_examples']))
    examples.update(map(tuple, diff['official_only_examples']))
    return sorted(examples)


def _rle_exact(a, b):
    ac = a.get('counts')
    bc = b.get('counts')
    if isinstance(ac, str):
        ac = ac.encode('ascii')
    if isinstance(bc, str):
        bc = bc.encode('ascii')
    return list(map(int, a['size'])) == list(map(int, b['size'])) and ac == bc


def rasterization_parity(dmet):
    """Compare Rust and pycocotools COCO-rasterized RLEs detection-by-detection."""
    from kwcoco.metrics.overlap import detections_to_rles

    rows = []
    exact = 0
    total = 0
    mismatches = []
    for gid in sorted(dmet.gid_to_true_dets):
        dims = dmet.gid_to_image_dims[gid]
        for role, dets in [
            ('true', dmet.true_detections(gid)),
            ('pred', dmet.pred_detections(gid)),
        ]:
            rust, _ = detections_to_rles(
                dets, dims, rasterizer='coco', raster_backend='kwimage_ext')
            ref, _ = detections_to_rles(
                dets, dims, rasterizer='coco', raster_backend='pycocotools')
            for idx, (a, b) in enumerate(zip(rust, ref)):
                total += 1
                same = _rle_exact(a, b)
                exact += int(same)
                if not same and len(mismatches) < 20:
                    mismatches.append({'gid': int(gid), 'role': role, 'index': idx})
            rows.append({'gid': int(gid), 'role': role, 'count': len(rust)})
    return {
        'total_rles': total,
        'exact_rles': exact,
        'num_rle_mismatches': total - exact,
        'mismatch_examples': mismatches,
        'groups': rows,
    }


def pair_diagnostics(dmet, examples):
    """Record pair IoUs under both rasterization conventions for mismatches."""
    from kwcoco.metrics.overlap import detections_to_rles
    from kwcoco.metrics.overlap import pairwise_segmentation_iou

    grouped = {}
    for gid, px, tx in examples:
        grouped.setdefault(int(gid), []).append((int(px), int(tx)))
    result = []
    for gid, pairs in sorted(grouped.items()):
        dims = dmet.gid_to_image_dims[gid]
        pred = dmet.pred_detections(gid)
        true = dmet.true_detections(gid)
        representations = {}
        for label, rasterizer, raster_backend in [
            ('kwimage', 'kwimage', 'auto'),
            ('coco_rust', 'coco', 'kwimage_ext'),
            ('coco_pycoco', 'coco', 'pycocotools'),
        ]:
            pred_rles, _ = detections_to_rles(
                pred, dims, rasterizer=rasterizer, raster_backend=raster_backend)
            true_rles, _ = detections_to_rles(
                true, dims, rasterizer=rasterizer, raster_backend=raster_backend)
            representations[label] = (pred_rles, true_rles)
        for px, tx in pairs:
            row = {'gid': gid, 'px': px, 'tx': tx, 'ious': {}}
            for label, (pred_rles, true_rles) in representations.items():
                mat, _ = pairwise_segmentation_iou(
                    [pred_rles[px]], [true_rles[tx]], backend='kwimage_ext')
                row['ious'][label] = float(mat[0, 0])
            result.append(row)
    return result


def compare_case(case_name, profile, seed=0):
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    true_dset, pred_dset, spec = make_case(case_name, profile, seed)
    dmet, build_timing = timed_call(
        lambda: DetectionMetrics.from_coco(true_dset, pred_dset), repeat=1)

    official_debug, official_debug_timing = timed_call(
        lambda: dmet.score_pycocotools(
            iou_type='segm', iou_thresholds=THRESHOLDS,
            with_confusion=True, with_evaler=False, verbose=0),
        repeat=1,
    )
    official_standard, official_standard_timing = timed_call(
        lambda: dmet.score_pycocotools(
            iou_type='segm', with_confusion=False,
            with_evaler=False, verbose=0),
        repeat=1,
    )

    variants = {}
    all_examples = set()
    for label, opts in NATIVE_VARIANTS.items():
        try:
            native, timing = timed_call(
                lambda opts=opts: dmet.confusion_vectors_v2(
                    iou_thresh=THRESHOLDS,
                    iou_type='segm', overlap_backend='kwimage_ext',
                    compat='mutex', ignore_classes=None,
                    spatial_threshold=None, **opts),
                repeat=1,
            )
        except Exception as ex:
            variants[label] = {
                'status': 'error', 'error': repr(ex),
                'traceback': traceback.format_exc(),
            }
            continue
        threshold_rows = {}
        for threshold in THRESHOLDS:
            native_cfsn = native[threshold]
            official_cfsn = official_debug['iou_to_cfsn_vecs'][threshold]
            diff = _assignment_diff(native_cfsn, official_cfsn)
            all_examples.update(_assignment_examples(diff))
            threshold_rows[str(threshold)] = {
                'assignment_diff': diff,
                'native_measures': _native_measures_from_cfsn(native_cfsn),
                'official_assignment_native_measures': _native_measures_from_cfsn(
                    official_cfsn),
            }
        variants[label] = {
            'status': 'ok', 'options': opts,
            'thresholds': threshold_rows, 'timing': timing,
        }

    raster_parity = None
    try:
        raster_parity = rasterization_parity(dmet)
    except Exception as ex:
        raster_parity = {'status': 'error', 'error': repr(ex),
                         'traceback': traceback.format_exc()}

    diagnostics = []
    if all_examples:
        try:
            diagnostics = pair_diagnostics(dmet, sorted(all_examples))
        except Exception as ex:
            diagnostics = [{'status': 'error', 'error': repr(ex),
                            'traceback': traceback.format_exc()}]

    return {
        'case': case_name,
        'spec': spec,
        'dataset': {
            'n_images': len(true_dset.imgs),
            'n_true': len(true_dset.anns),
            'n_pred': len(pred_dset.anns),
        },
        'official': {
            'AP50': float(official_standard['summary']['AP50']),
            'assignment_extract_timing': official_debug_timing,
            'standard_timing': official_standard_timing,
        },
        'variants': variants,
        'rasterization_parity': raster_parity,
        'pair_diagnostics': diagnostics,
        'detection_metrics_build': build_timing,
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
        print(f'Running rasterization case {case_name!r}')
        try:
            result = compare_case(case_name, profile, seed=idx)
        except Exception as ex:
            report['experiments'].append({
                'name': case_name, 'status': 'error', 'error': repr(ex),
                'traceback': traceback.format_exc(),
            })
        else:
            report['experiments'].append({
                'name': case_name, 'status': 'ok', 'result': result,
            })
    return report


def summary_lines(report):
    lines = [
        '# kwcoco Phase-2.1 rasterization / threshold summary', '',
        f"Mode: `{report['mode']}`", '',
        '| case | variant | matches @.50 native/off/shared | native-only | official-only | native mAP50 | Rust/pycoco RLE mismatches |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for exp in report['experiments']:
        if exp['status'] != 'ok':
            lines.append(f"| `{exp['name']}` | ERROR | | | | | |")
            continue
        result = exp['result']
        raster_mismatch = result['rasterization_parity'].get(
            'num_rle_mismatches', 'error')
        for label, variant in result['variants'].items():
            if variant['status'] != 'ok':
                lines.append(
                    f"| `{exp['name']}` | `{label}` | error | | | | {raster_mismatch} |")
                continue
            row = variant['thresholds']['0.5']
            diff = row['assignment_diff']
            matches = (
                f"{diff['num_native_matches']}/"
                f"{diff['num_official_matches']}/"
                f"{diff['num_shared_matches']}")
            map50 = row['native_measures']['mAP']
            lines.append(
                f"| `{exp['name']}` | `{label}` | {matches} | "
                f"{diff['num_native_only']} | {diff['num_official_only']} | "
                f"{map50:.6f} | {raster_mismatch} |")
    return lines


def write_bundle(report, output_dpath):
    dpath = Path(output_dpath)
    dpath.mkdir(parents=True, exist_ok=True)
    mode = report['mode']
    json_fpath = dpath / f'phase21_rasterization_{mode}_results.json'
    md_fpath = dpath / f'phase21_rasterization_{mode}_summary.md'
    json_fpath.write_text(json.dumps(report, indent=2, default=_json_default) + '\n')
    md_fpath.write_text('\n'.join(summary_lines(report)) + '\n')
    bundle = dpath / f'kwcoco_metrics_phase21_rasterization_{mode}.tar.gz'
    with tarfile.open(bundle, 'w:gz') as tar:
        tar.add(json_fpath, arcname=json_fpath.name)
        tar.add(md_fpath, arcname=md_fpath.name)
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=sorted(PROFILES), default='quick')
    parser.add_argument('--output-dpath', type=Path,
                        default=Path('_metric_compare_phase21_rasterization'))
    args = parser.parse_args(argv)
    report = build_report(args.mode)
    bundle = write_bundle(report, args.output_dpath)
    print('Phase-2.1 rasterization suite complete')
    print(f'  bundle: {bundle}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
