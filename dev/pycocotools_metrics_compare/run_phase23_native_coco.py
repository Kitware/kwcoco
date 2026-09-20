#!/usr/bin/env python3
"""Phase-2.3: compare native COCO-shaped mask summaries with COCOeval.

This runner exercises the dataset-level native instance-mask evaluator after
area-range and maxDets semantics have been applied during assignment.  It keeps
row-level assignment comparisons separate from AP/AR accumulation differences.
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


PROFILES = {
    'quick': {'nimgs': 4, 'image_size': (96, 96)},
    'full': {'nimgs': 32, 'image_size': (256, 256)},
}
BASE_CASES = ['perfect', 'box_noise', 'class_noise', 'mask_only_shift', 'mixed']
DEBUG_THRESHOLDS = [0.50, 0.75, 0.95]


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


def _poly(x1, y1, x2, y2):
    return [[x1, y1, x2, y1, x2, y2, x1, y2]]


def _base_dsets(*, dims=(128, 128), categories=('object',), gids=(1,)):
    import kwcoco
    height, width = dims
    true = kwcoco.CocoDataset()
    pred = kwcoco.CocoDataset()
    for dset in [true, pred]:
        for cid, name in enumerate(categories, start=1):
            dset.add_category(name=name, id=cid)
        for gid in gids:
            dset.add_image(
                id=gid, file_name=f'im{gid}.png', width=width, height=height)
    return true, pred


def make_area_priority_case():
    true, pred = _base_dsets()
    # The small truth has slightly lower IoU than the medium truth, but must
    # still outrank it when the small-area slice is evaluated.
    true.add_annotation(
        id=1, image_id=1, category_id=1,
        bbox=[16, 12, 30, 30], segmentation=_poly(16, 12, 46, 42))
    true.add_annotation(
        id=2, image_id=1, category_id=1,
        bbox=[10, 10, 34, 34], segmentation=_poly(10, 10, 44, 44))
    pred.add_annotation(
        id=11, image_id=1, category_id=1,
        bbox=[12, 12, 30, 30], segmentation=_poly(12, 12, 42, 42),
        score=.99)
    return true, pred, {'kind': 'area_priority'}


def make_maxdets_case():
    true, pred = _base_dsets(categories=('a', 'b'))
    for cid, box in [(1, [8, 8, 18, 18]), (2, [72, 72, 18, 18])]:
        x, y, w, h = box
        true.add_annotation(
            id=cid, image_id=1, category_id=cid, bbox=box,
            segmentation=_poly(x, y, x + w, y + h))
        pred.add_annotation(
            id=10 + cid, image_id=1, category_id=cid, bbox=box,
            segmentation=_poly(x, y, x + w, y + h),
            score=.99 - cid * .01)
    return true, pred, {'kind': 'max_dets_per_category'}


def make_empty_image_case():
    true, pred = _base_dsets(gids=(1, 2))
    box1 = [8, 8, 20, 20]
    true.add_annotation(
        id=1, image_id=1, category_id=1, bbox=box1,
        segmentation=_poly(8, 8, 28, 28))
    pred.add_annotation(
        id=11, image_id=1, category_id=1, bbox=box1,
        segmentation=_poly(8, 8, 28, 28), score=.90)
    # A higher-score false positive on a truth-empty image.
    pred.add_annotation(
        id=12, image_id=2, category_id=1, bbox=[70, 70, 20, 20],
        segmentation=_poly(70, 70, 90, 90), score=.99)
    return true, pred, {'kind': 'negative_image'}


def make_crowd_case():
    true, pred = _base_dsets()
    true.add_annotation(
        id=1, image_id=1, category_id=1,
        bbox=[12, 12, 24, 24], segmentation=_poly(12, 12, 36, 36))
    true.add_annotation(
        id=2, image_id=1, category_id=1,
        bbox=[60, 60, 48, 48], segmentation=_poly(60, 60, 108, 108),
        iscrowd=1)
    for aid, box, score in [
        (11, [12, 12, 24, 24], .99),
        (12, [64, 64, 12, 12], .95),
        (13, [82, 82, 12, 12], .90),
    ]:
        x, y, w, h = box
        pred.add_annotation(
            id=aid, image_id=1, category_id=1, bbox=box,
            segmentation=_poly(x, y, x + w, y + h), score=score)
    return true, pred, {'kind': 'crowd'}


CUSTOM_CASES = {
    'area_priority': make_area_priority_case,
    'max_dets': make_maxdets_case,
    'empty_images': make_empty_image_case,
    'crowd': make_crowd_case,
}


def _case_data(name, profile, seed):
    if name in CUSTOM_CASES:
        return CUSTOM_CASES[name]()
    return make_case(name, profile, seed)


def compare_case(case_name, profile, seed=0):
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    true_dset, pred_dset, spec = _case_data(case_name, profile, seed)
    dmet, dmet_timing = timed_call(
        lambda: DetectionMetrics.from_coco(true_dset, pred_dset), repeat=1)

    native, native_wall = timed_call(
        lambda: dmet.score_native_coco(
            iou_type='segm', overlap_backend='auto', raster_backend='auto'),
        repeat=1)
    official, official_wall = timed_call(
        lambda: dmet.score_pycocotools(
            iou_type='segm', with_confusion=False, verbose=0),
        repeat=1)

    native_rows = dmet.confusion_vectors_v2(
        iou_thresh=DEBUG_THRESHOLDS,
        iou_type='segm', compat='mutex', ignore_classes=None,
        overlap_backend='auto', segmentation_rasterizer='coco',
        raster_backend='auto', threshold_policy='inclusive')
    official_rows = dmet.score_pycocotools(
        iou_type='segm', iou_thresholds=DEBUG_THRESHOLDS,
        with_confusion=True, verbose=0)['iou_to_cfsn_vecs']
    assignment = {
        str(t): _assignment_diff(native_rows[t], official_rows[t])
        for t in DEBUG_THRESHOLDS
    }

    deltas = {
        key: float(native['summary'][key] - official['summary'][key])
        for key in native['summary']
    }
    abs_deltas = {key: abs(value) for key, value in deltas.items()}
    return {
        'case': case_name,
        'spec': spec,
        'dataset': {
            'n_images': len(true_dset.imgs),
            'n_true': len(true_dset.anns),
            'n_pred': len(pred_dset.anns),
        },
        'native': {
            'summary': native['summary'],
            'params': native['params'],
            'provenance': native['provenance'],
            'timing': native['timing'],
        },
        'official': {
            'summary': official['summary'],
            'params': official['params'],
        },
        'delta': deltas,
        'absolute_delta': abs_deltas,
        'assignment': assignment,
        'timing': {
            'detection_metrics_build': dmet_timing,
            'native_wall': native_wall,
            'official_wall': official_wall,
        },
    }


def build_report(mode):
    profile = dict(PROFILES[mode])
    cases = BASE_CASES + list(CUSTOM_CASES)
    report = {
        'schema_version': 1,
        'mode': mode,
        'created_unix': time.time(),
        'environment': environment_info(),
        'source_state': source_state(),
        'profile': profile,
        'cases': cases,
        'experiments': [],
    }
    for idx, case_name in enumerate(cases):
        print(f'Running Phase-2.3 case {case_name!r}')
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
        '# kwcoco Phase-2.3 native COCO-shaped mask summary', '',
        f"Mode: `{report['mode']}`", '',
        '| case | status | native AP | COCO AP | |delta AP| | native AR100 | '
        'COCO AR100 | assignment diff @.50 | build sec | assign sec | accum sec |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for exp in report['experiments']:
        if exp['status'] != 'ok':
            lines.append(f"| `{exp['name']}` | error | | | | | | | | | |")
            continue
        r = exp['result']
        ns = r['native']['summary']
        osum = r['official']['summary']
        diff50 = r['assignment']['0.5']
        ndiff = diff50['num_native_only'] + diff50['num_official_only']
        timing = r['native']['timing']
        lines.append(
            f"| `{exp['name']}` | ok | {ns['AP']:.6f} | {osum['AP']:.6f} | "
            f"{r['absolute_delta']['AP']:.6f} | {ns['AR100']:.6f} | "
            f"{osum['AR100']:.6f} | {ndiff} | "
            f"{timing['candidate_build_seconds']:.6f} | "
            f"{timing['assignment_seconds']:.6f} | "
            f"{timing['accumulation_seconds']:.6f} |"
        )
    lines.extend([
        '',
        'The JSON report contains all 12 named AP/AR deltas, row-level '
        'assignment disagreement counts at IoU 0.50/0.75/0.95, backend '
        'provenance, and timing decomposition.',
    ])
    return lines


def write_bundle(report, output_dpath):
    dpath = Path(output_dpath)
    dpath.mkdir(parents=True, exist_ok=True)
    mode = report['mode']
    jf = dpath / f'phase23_native_coco_{mode}_results.json'
    mf = dpath / f'phase23_native_coco_{mode}_summary.md'
    jf.write_text(json.dumps(report, indent=2, default=_json_default) + '\n')
    mf.write_text('\n'.join(summary_lines(report)) + '\n')
    bundle = dpath / f'kwcoco_metrics_phase23_native_coco_{mode}.tar.gz'
    with tarfile.open(bundle, 'w:gz') as tar:
        tar.add(jf, arcname=jf.name)
        tar.add(mf, arcname=mf.name)
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=sorted(PROFILES), default='quick')
    parser.add_argument(
        '--output-dpath', type=Path,
        default=Path('_metric_compare_phase23_native_coco'))
    args = parser.parse_args(argv)
    report = build_report(args.mode)
    bundle = write_bundle(report, args.output_dpath)
    print('Phase-2.3 native COCO-shaped mask suite complete')
    print(f'  bundle: {bundle}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
