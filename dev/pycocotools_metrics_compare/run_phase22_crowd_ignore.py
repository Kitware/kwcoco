#!/usr/bin/env python3
"""Phase-2.2: validate crowd and ignore semantics for native mask assignment.

This suite keeps three concepts separate:

* COCO crowd semantics: crowd masks use detection-area overlap, may be reused,
  and matches to them carry zero metric weight.
* COCO-compatible mask defaults: COCO polygon rasterization and inclusive IoU
  thresholds.
* kwcoco ignore-class semantics: ignore-category masks suppress predictions by
  intersection-over-prediction-area after the ignore masks are unioned.

The first group is compared row-for-row with official COCOeval.  The ignore
class and explicit annotation-ignore extensions are reported independently
because they are intentionally richer than current official COCOeval behavior.
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
from compare_metrics import _native_measures_from_cfsn
from compare_metrics import timed_call


THRESHOLDS = [0.50, 0.75, 0.95]


def _poly(x1, y1, x2, y2):
    return [[x1, y1, x2, y1, x2, y2, x1, y2]]


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


def _base_dsets():
    import kwcoco
    true = kwcoco.CocoDataset()
    pred = kwcoco.CocoDataset()
    for dset in [true, pred]:
        dset.add_category(name='object', id=1)
        dset.add_category(name='ignore', id=2)
        dset.add_image(id=1, file_name='im.png', width=64, height=64)
    return true, pred


def make_crowd_reuse_case():
    true, pred = _base_dsets()
    true.add_annotation(
        id=1, image_id=1, category_id=1,
        bbox=[4, 4, 10, 10], segmentation=_poly(4, 4, 14, 14), iscrowd=0)
    true.add_annotation(
        id=2, image_id=1, category_id=1,
        bbox=[28, 28, 26, 26], segmentation=_poly(28, 28, 54, 54), iscrowd=1)
    for aid, box, score in [
        (11, [4, 4, 10, 10], .99),
        (12, [30, 30, 8, 8], .95),
        (13, [40, 40, 8, 8], .90),
        (14, [52, 4, 6, 6], .80),
    ]:
        x, y, w, h = box
        pred.add_annotation(
            id=aid, image_id=1, category_id=1, bbox=box,
            segmentation=_poly(x, y, x + w, y + h), score=score)
    return true, pred


def make_regular_vs_crowd_case():
    true, pred = _base_dsets()
    true.add_annotation(
        id=1, image_id=1, category_id=1,
        bbox=[10, 10, 12, 12], segmentation=_poly(10, 10, 22, 22), iscrowd=0)
    true.add_annotation(
        id=2, image_id=1, category_id=1,
        bbox=[6, 6, 24, 24], segmentation=_poly(6, 6, 30, 30), iscrowd=1)
    pred.add_annotation(
        id=11, image_id=1, category_id=1,
        bbox=[11, 11, 10, 10], segmentation=_poly(11, 11, 21, 21), score=.99)
    return true, pred


def make_exact_threshold_case():
    import kwimage
    true, pred = _base_dsets()
    dims = (64, 64)
    tmask = np.zeros(dims, dtype=np.uint8)
    pmask = np.zeros(dims, dtype=np.uint8)
    tmask[8:24, 8:24] = 1
    pmask[8:24, 8:16] = 1
    trle = kwimage.Mask(tmask, 'c_mask').to_bytes_rle().data
    prle = kwimage.Mask(pmask, 'c_mask').to_bytes_rle().data
    true.add_annotation(
        id=1, image_id=1, category_id=1,
        bbox=[8, 8, 16, 16], segmentation=trle, iscrowd=0)
    pred.add_annotation(
        id=11, image_id=1, category_id=1,
        bbox=[8, 8, 8, 16], segmentation=prle, score=.99)
    return true, pred


def make_ignore_class_case():
    true, pred = _base_dsets()
    true.add_annotation(
        id=1, image_id=1, category_id=2,
        bbox=[10, 10, 16, 20], segmentation=_poly(10, 10, 26, 30))
    true.add_annotation(
        id=2, image_id=1, category_id=2,
        bbox=[20, 10, 16, 20], segmentation=_poly(20, 10, 36, 30))
    pred.add_annotation(
        id=11, image_id=1, category_id=1,
        bbox=[14, 14, 18, 12], segmentation=_poly(14, 14, 32, 26), score=.9)
    return true, pred


def make_explicit_ignore_case():
    true, pred = _base_dsets()
    true.add_annotation(
        id=1, image_id=1, category_id=1,
        bbox=[8, 8, 20, 20], segmentation=_poly(8, 8, 28, 28),
        iscrowd=0, ignore=True)
    pred.add_annotation(
        id=11, image_id=1, category_id=1,
        bbox=[8, 8, 20, 20], segmentation=_poly(8, 8, 28, 28), score=.99)
    return true, pred


def _weight_rows(cfsn):
    data = cfsn.data
    return [
        {
            'px': int(px), 'tx': int(tx), 'weight': float(w),
            'true': int(y), 'pred': int(p), 'score': float(s),
        }
        for px, tx, w, y, p, s in zip(
            data['pxs'], data['txs'], data['weight'],
            data['true'], data['pred'], data['score'])
    ]


def compare_official_case(name, maker):
    from kwcoco.metrics.detect_metrics import DetectionMetrics
    true, pred = maker()
    dmet = DetectionMetrics.from_coco(true, pred)
    native, native_timing = timed_call(
        lambda: dmet.confusion_vectors_v2(
            iou_thresh=THRESHOLDS, iou_type='segm', compat='mutex',
            ignore_classes=None, spatial_threshold=None), repeat=1)
    official, official_timing = timed_call(
        lambda: dmet.score_pycocotools(
            iou_type='segm', iou_thresholds=THRESHOLDS,
            with_confusion=True, verbose=0), repeat=1)
    rows = {}
    for t in THRESHOLDS:
        nc = native[t]
        oc = official['iou_to_cfsn_vecs'][t]
        rows[str(t)] = {
            'assignment_diff': _assignment_diff(nc, oc),
            'native_measures': _native_measures_from_cfsn(nc),
            'official_native_measures': _native_measures_from_cfsn(oc),
            'native_rows': _weight_rows(nc),
            'official_rows': _weight_rows(oc),
        }
    return {
        'case': name,
        'kind': 'official_parity',
        'thresholds': rows,
        'timing': {'native': native_timing, 'official': official_timing},
    }


def compare_ignore_class_extension():
    from kwcoco.metrics.detect_metrics import DetectionMetrics
    true, pred = make_ignore_class_case()
    dmet = DetectionMetrics.from_coco(true, pred)
    ignored = dmet.confusion_vectors_v2(
        iou_thresh=0.5, iou_type='segm', compat='mutex',
        ignore_classes='ignore', spatial_threshold=None)
    raw = dmet.confusion_vectors_v2(
        iou_thresh=0.5, iou_type='segm', compat='mutex',
        ignore_classes=None, spatial_threshold=None)
    return {
        'case': 'ignore_class_union',
        'kind': 'kwcoco_extension',
        'ignored_rows': _weight_rows(ignored),
        'without_ignore_rows': _weight_rows(raw),
    }


def compare_explicit_ignore_extension():
    from kwcoco.metrics.detect_metrics import DetectionMetrics
    true, pred = make_explicit_ignore_case()
    dmet = DetectionMetrics.from_coco(true, pred)
    coco = dmet.confusion_vectors_v2(
        iou_thresh=0.5, iou_type='segm', compat='mutex',
        ignore_classes=None, annotation_ignore_policy='coco',
        spatial_threshold=None)
    native = dmet.confusion_vectors_v2(
        iou_thresh=0.5, iou_type='segm', compat='mutex',
        ignore_classes=None, annotation_ignore_policy='kwcoco',
        spatial_threshold=None)
    return {
        'case': 'explicit_annotation_ignore',
        'kind': 'kwcoco_extension',
        'coco_policy_rows': _weight_rows(coco),
        'kwcoco_policy_rows': _weight_rows(native),
    }


def build_report():
    experiments = []
    for name, maker in [
        ('crowd_reuse', make_crowd_reuse_case),
        ('regular_vs_crowd', make_regular_vs_crowd_case),
        ('exact_iou_050', make_exact_threshold_case),
    ]:
        print(f'Running {name!r}')
        try:
            result = compare_official_case(name, maker)
        except Exception as ex:
            experiments.append({
                'name': name, 'status': 'error', 'error': repr(ex),
                'traceback': traceback.format_exc()})
        else:
            experiments.append({'name': name, 'status': 'ok', 'result': result})
    for func in [compare_ignore_class_extension, compare_explicit_ignore_extension]:
        name = func.__name__.replace('compare_', '')
        print(f'Running {name!r}')
        try:
            result = func()
        except Exception as ex:
            experiments.append({
                'name': name, 'status': 'error', 'error': repr(ex),
                'traceback': traceback.format_exc()})
        else:
            experiments.append({'name': name, 'status': 'ok', 'result': result})
    return {
        'schema_version': 1,
        'created_unix': time.time(),
        'environment': environment_info(),
        'source_state': source_state(),
        'thresholds': THRESHOLDS,
        'experiments': experiments,
    }


def summary_lines(report):
    lines = [
        '# kwcoco Phase-2.2 crowd / ignore summary', '',
        '## Official parity', '',
        '| case | threshold | native/off/shared | native-only | official-only | native mAP | official-assignment mAP |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: |',
    ]
    for exp in report['experiments']:
        if exp['status'] != 'ok':
            lines.append(f"| `{exp['name']}` | ERROR | | | | | |")
            continue
        r = exp['result']
        if r['kind'] != 'official_parity':
            continue
        for t, row in r['thresholds'].items():
            d = row['assignment_diff']
            lines.append(
                f"| `{r['case']}` | {t} | {d['num_native_matches']}/{d['num_official_matches']}/{d['num_shared_matches']} | "
                f"{d['num_native_only']} | {d['num_official_only']} | "
                f"{row['native_measures']['mAP']:.6f} | {row['official_native_measures']['mAP']:.6f} |")
    lines.extend(['', '## Native extensions', ''])
    for exp in report['experiments']:
        if exp['status'] == 'ok' and exp['result']['kind'] == 'kwcoco_extension':
            lines.append(f"* `{exp['result']['case']}` recorded in JSON")
    return lines


def write_bundle(report, output_dpath):
    dpath = Path(output_dpath)
    dpath.mkdir(parents=True, exist_ok=True)
    jf = dpath / 'phase22_crowd_ignore_results.json'
    mf = dpath / 'phase22_crowd_ignore_summary.md'
    jf.write_text(json.dumps(report, indent=2, default=_json_default) + '\n')
    mf.write_text('\n'.join(summary_lines(report)) + '\n')
    bundle = dpath / 'kwcoco_metrics_phase22_crowd_ignore.tar.gz'
    with tarfile.open(bundle, 'w:gz') as tar:
        tar.add(jf, arcname=jf.name)
        tar.add(mf, arcname=mf.name)
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-dpath', type=Path,
        default=Path('_metric_compare_phase22_crowd_ignore'))
    args = parser.parse_args(argv)
    report = build_report()
    bundle = write_bundle(report, args.output_dpath)
    print('Phase-2.2 crowd/ignore suite complete')
    print(f'  bundle: {bundle}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
