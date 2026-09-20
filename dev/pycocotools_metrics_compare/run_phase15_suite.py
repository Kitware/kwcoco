#!/usr/bin/env python3
"""
Phase-1.5 backend comparison suite for kwcoco / kwimage acceleration kernels.

This suite is intentionally low-level.  It measures the numerical agreement
and throughput of box-IoU and COCO-RLE operations independently from detector
assignment and AP accumulation.  Run the exact same command in environments
with different ``kwimage_ext`` installations (absent, legacy Cython, Rust) and
compare the resulting bundles.
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
import statistics
import subprocess
import sys
import tarfile
import time
import traceback

import numpy as np


THRESHOLDS = (0.50, 0.75, 0.95)

PROFILES = {
    'quick': {
        'box_sizes': [8, 32, 128, 384],
        'mask_counts': [4, 16, 48],
        'mask_shape': (128, 128),
        'target_seconds': 0.04,
        'max_repeat': 200,
        'seed': 0,
    },
    'full': {
        'box_sizes': [8, 32, 128, 384, 768, 1536],
        'mask_counts': [4, 16, 48, 128, 256],
        'mask_shape': (256, 256),
        'target_seconds': 0.20,
        'max_repeat': 500,
        'seed': 0,
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
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
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
        rust = importlib.import_module('kwimage_ext._rust')
    except Exception as ex:
        info['backends']['kwimage_ext_rust'] = {
            'available': False,
            'error': repr(ex),
        }
    else:
        caps = []
        try:
            caps = list(rust.capabilities())
        except Exception:
            pass
        info['backends']['kwimage_ext_rust'] = {
            'available': True,
            'module_file': getattr(rust, '__file__', None),
            'capabilities': caps,
        }

    try:
        boxes = importlib.import_module(
            'kwimage_ext.structs._boxes_backend.cython_boxes')
    except Exception as ex:
        info['backends']['kwimage_ext_boxes_public'] = {
            'available': False, 'error': repr(ex)}
    else:
        item = {'available': True, 'module_file': getattr(boxes, '__file__', None)}
        if hasattr(boxes, 'backend_metadata'):
            try:
                item['dispatch'] = boxes.backend_metadata()
            except Exception as ex:
                item['dispatch_error'] = repr(ex)
        elif str(getattr(boxes, '__file__', '')).endswith(('.so', '.pyd', '.dylib')):
            item['dispatch'] = {'kind': 'legacy-direct-extension'}
        info['backends']['kwimage_ext_boxes_public'] = item

    try:
        masks = importlib.import_module(
            'kwimage_ext.structs._mask_backend.cython_mask')
    except Exception as ex:
        info['backends']['kwimage_ext_masks_public'] = {
            'available': False, 'error': repr(ex)}
    else:
        item = {'available': True, 'module_file': getattr(masks, '__file__', None)}
        if hasattr(masks, 'backend_metadata'):
            try:
                item['dispatch'] = masks.backend_metadata()
            except Exception as ex:
                item['dispatch_error'] = repr(ex)
        elif str(getattr(masks, '__file__', '')).endswith(('.so', '.pyd', '.dylib')):
            item['dispatch'] = {'kind': 'legacy-direct-extension'}
        info['backends']['kwimage_ext_masks_public'] = item

    try:
        importlib.import_module('pycocotools.mask')
    except Exception as ex:
        info['backends']['pycocotools_mask'] = {
            'available': False, 'error': repr(ex)}
    else:
        info['backends']['pycocotools_mask'] = {'available': True}
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


def adaptive_timing(func, target_seconds, max_repeat):
    """Calibrate repeat count so wrapper overhead is not the entire result."""
    # Warmup also catches lazy imports / one-time allocation.
    result = func()
    start = time.perf_counter()
    func()
    first = max(time.perf_counter() - start, 1e-9)
    repeat = max(1, int(math.ceil(target_seconds / first)))
    repeat = min(repeat, max_repeat)
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = func()
        samples.append(time.perf_counter() - start)
    return result, {
        'repeat': repeat,
        'min_seconds': min(samples),
        'median_seconds': statistics.median(samples),
        'mean_seconds': statistics.mean(samples),
        'max_seconds': max(samples),
        'total_seconds': sum(samples),
    }


def numpy_box_iou(boxes, query, bias=0.0):
    boxes = np.asarray(boxes, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    if len(boxes) == 0 or len(query) == 0:
        return np.empty((len(boxes), len(query)), dtype=np.float32)
    tl = np.maximum(boxes[:, None, 0:2], query[None, :, 0:2])
    br = np.minimum(boxes[:, None, 2:4], query[None, :, 2:4])
    wh = np.maximum(br - tl + bias, 0)
    inter = wh[..., 0] * wh[..., 1]
    a1 = ((boxes[:, 2] - boxes[:, 0] + bias) *
          (boxes[:, 3] - boxes[:, 1] + bias))[:, None]
    a2 = ((query[:, 2] - query[:, 0] + bias) *
          (query[:, 3] - query[:, 1] + bias))[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        out = inter / (a1 + a2 - inter)
    return np.nan_to_num(out).astype(np.float32)


def random_ltrb(rng, n, canvas=2048):
    xy1 = rng.uniform(0, canvas * 0.8, size=(n, 2))
    wh = rng.uniform(1, canvas * 0.2, size=(n, 2))
    boxes = np.concatenate([xy1, xy1 + wh], axis=1)
    return boxes.astype(np.float32)


def compare_arrays(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return {'shape_equal': False, 'left_shape': a.shape, 'right_shape': b.shape}
    if a.size:
        absdiff = np.abs(a.astype(float) - b.astype(float))
        max_abs = float(absdiff.max())
    else:
        max_abs = 0.0
    result = {
        'shape_equal': True,
        'exact_equal': bool(np.array_equal(a, b)),
        'allclose_1e_12': bool(np.allclose(a, b, rtol=1e-12, atol=1e-12)),
        'max_abs_diff': max_abs,
    }
    if a.shape == b.shape:
        for threshold in THRESHOLDS:
            result[f'ge_{threshold}_disagreements'] = int(
                np.count_nonzero((a >= threshold) != (b >= threshold)))
    return result


def box_benchmarks(profile):
    rng = np.random.RandomState(profile['seed'])
    rows = []
    backends = {}
    try:
        public = importlib.import_module(
            'kwimage_ext.structs._boxes_backend.cython_boxes')
        backends['kwimage_ext_public'] = public.bbox_ious_c
    except Exception:
        pass
    try:
        rust = importlib.import_module('kwimage_ext._rust')
        if hasattr(rust, 'bbox_ious_c'):
            backends['kwimage_ext_rust_direct'] = rust.bbox_ious_c
    except Exception:
        pass
    try:
        import kwimage
    except Exception:
        pass
    else:
        def kwimage_auto(a, b, bias=0.0):
            return kwimage.Boxes(a, 'ltrb').ious(kwimage.Boxes(b, 'ltrb'), bias=bias)
        backends['kwimage_auto'] = kwimage_auto

    for n in profile['box_sizes']:
        boxes = random_ltrb(rng, n)
        query = random_ltrb(rng, n)
        reference, ref_timing = adaptive_timing(
            lambda: numpy_box_iou(boxes, query, bias=0.0),
            profile['target_seconds'], profile['max_repeat'])
        item = {
            'n': n,
            'pairs': n * n,
            'reference': {'backend': 'numpy', 'timing': ref_timing},
            'backends': {},
        }
        for name, func in backends.items():
            try:
                got, timing = adaptive_timing(
                    lambda func=func: func(boxes, query, bias=0.0),
                    profile['target_seconds'], profile['max_repeat'])
            except Exception as ex:
                item['backends'][name] = {
                    'status': 'error', 'error': repr(ex),
                    'traceback': traceback.format_exc(),
                }
            else:
                item['backends'][name] = {
                    'status': 'ok',
                    'timing': timing,
                    'parity': compare_arrays(got, reference),
                }
        rows.append(item)
    return rows


def make_masks(rng, count, shape, complexity):
    h, w = shape
    masks = np.zeros((h, w, count), dtype=np.uint8, order='F')
    if complexity == 'rectangles':
        for i in range(count):
            x1 = rng.randint(0, max(1, w - 8))
            y1 = rng.randint(0, max(1, h - 8))
            x2 = rng.randint(x1 + 1, w + 1)
            y2 = rng.randint(y1 + 1, h + 1)
            masks[y1:y2, x1:x2, i] = 1
    elif complexity == 'fragmented':
        for i in range(count):
            for _ in range(24):
                x = rng.randint(0, w)
                y = rng.randint(0, h)
                rx = rng.randint(1, max(2, w // 20))
                ry = rng.randint(1, max(2, h // 20))
                masks[max(0, y - ry):min(h, y + ry + 1),
                      max(0, x - rx):min(w, x + rx + 1), i] = 1
    elif complexity == 'sparse_noise':
        masks[...] = (rng.rand(h, w, count) < 0.02)
    else:
        raise KeyError(complexity)
    return np.asfortranarray(masks)


def _mask_backends():
    result = {}
    try:
        public = importlib.import_module(
            'kwimage_ext.structs._mask_backend.cython_mask')
        result['kwimage_ext_public'] = public
    except Exception:
        pass
    try:
        rust = importlib.import_module('kwimage_ext._rust')
        if hasattr(rust, 'iou'):
            result['kwimage_ext_rust_direct'] = rust
    except Exception:
        pass
    try:
        pycoco = importlib.import_module('pycocotools.mask')
        result['pycocotools'] = pycoco
    except Exception:
        pass
    return result


def mask_benchmarks(profile):
    rng = np.random.RandomState(profile['seed'] + 991)
    backends = _mask_backends()
    if 'pycocotools' not in backends:
        return {'status': 'skipped', 'reason': 'pycocotools unavailable', 'rows': []}
    pycoco = backends['pycocotools']
    rows = []
    for complexity in ['rectangles', 'fragmented', 'sparse_noise']:
        for count in profile['mask_counts']:
            masks = make_masks(rng, count, profile['mask_shape'], complexity)
            reference_rles = pycoco.encode(masks)
            crowd = [0] * count
            reference, ref_timing = adaptive_timing(
                lambda: pycoco.iou(reference_rles, reference_rles, crowd),
                profile['target_seconds'], profile['max_repeat'])
            row = {
                'complexity': complexity,
                'shape': profile['mask_shape'],
                'count': count,
                'pairs': count * count,
                'reference': {'backend': 'pycocotools', 'timing': ref_timing},
                'backends': {},
            }
            for name, backend in backends.items():
                if name == 'pycocotools':
                    continue
                try:
                    got, timing = adaptive_timing(
                        lambda backend=backend: backend.iou(
                            reference_rles, reference_rles, crowd),
                        profile['target_seconds'], profile['max_repeat'])
                except Exception as ex:
                    row['backends'][name] = {
                        'status': 'error', 'error': repr(ex),
                        'traceback': traceback.format_exc(),
                    }
                else:
                    row['backends'][name] = {
                        'status': 'ok',
                        'timing': timing,
                        'parity': compare_arrays(got, reference),
                    }

            # Codec parity is measured on the same pixels, separately from IoU.
            public = backends.get('kwimage_ext_public')
            if public is not None:
                try:
                    ext_rles = public.encode(masks)
                    encoded_equal = all(
                        a['counts'] == b['counts'] and a['size'] == b['size']
                        for a, b in zip(ext_rles, reference_rles)
                    )
                    decoded_equal = np.array_equal(
                        public.decode(ext_rles), pycoco.decode(reference_rles))
                    area_equal = np.array_equal(
                        public.area(ext_rles), pycoco.area(reference_rles))
                    bbox_equal = np.allclose(
                        public.toBbox(ext_rles), pycoco.toBbox(reference_rles))
                    row['codec_parity'] = {
                        'compressed_counts_equal': bool(encoded_equal),
                        'decode_equal': bool(decoded_equal),
                        'area_equal': bool(area_equal),
                        'bbox_equal': bool(bbox_equal),
                    }
                except Exception as ex:
                    row['codec_parity'] = {
                        'status': 'error', 'error': repr(ex),
                        'traceback': traceback.format_exc(),
                    }
            rows.append(row)
    return {'status': 'ok', 'rows': rows}


def edge_case_checks():
    checks = []
    try:
        mask = importlib.import_module(
            'kwimage_ext.structs._mask_backend.cython_mask')
    except Exception as ex:
        return [{'name': 'mask_backend_import', 'status': 'skipped', 'error': repr(ex)}]

    def run(name, func):
        try:
            value = func()
        except Exception as ex:
            checks.append({'name': name, 'status': 'error', 'error': repr(ex),
                           'traceback': traceback.format_exc()})
        else:
            checks.append({'name': name, 'status': 'ok', 'value': value})

    run('one_dim_bbox_iou', lambda: np.asarray(mask.iou(
        np.array([0, 0, 10, 10], dtype=float),
        np.array([0, 0, 10, 10], dtype=float), [0])).tolist())
    run('iscrowd_length_validation', lambda: _expect_error(
        lambda: mask.iou(np.array([[0, 0, 10, 10]], dtype=float),
                         np.array([[0, 0, 10, 10]], dtype=float), [])))
    run('nx2_quad_frpyobjects', lambda: int(mask.area([
        mask.frPyObjects(np.array([[1, 1], [1, 8], [8, 8], [8, 1]], dtype=float), 16, 16)
    ])[0]))
    run('degenerate_polygon', lambda: int(mask.area([
        mask.frPoly([[1, 1, 3, 3]], 16, 16)[0]
    ])[0]))
    return checks


def _expect_error(func):
    try:
        func()
    except Exception as ex:
        return {'raised': True, 'type': type(ex).__name__, 'message': str(ex)}
    return {'raised': False}


def summary_lines(report):
    lines = [
        '# kwcoco metrics Phase-1.5 backend summary', '',
        f"Mode: `{report['mode']}`", '',
        '## Backend provenance', '', '```json',
        json.dumps(report['environment']['backends'], indent=2, default=_json_default),
        '```', '',
        '## Box IoU', '',
        '| N | backend | median sec | max abs diff |',
        '| ---: | --- | ---: | ---: |',
    ]
    for row in report['box_benchmarks']:
        for name, item in row['backends'].items():
            if item['status'] == 'ok':
                lines.append(
                    f"| {row['n']} | `{name}` | "
                    f"{item['timing']['median_seconds']:.8f} | "
                    f"{item['parity']['max_abs_diff']:.3g} |")
    lines.extend(['', '## Mask IoU', '',
                  '| kind | count | backend | median sec | max abs diff |',
                  '| --- | ---: | --- | ---: | ---: |'])
    for row in report['mask_benchmarks'].get('rows', []):
        for name, item in row['backends'].items():
            if item['status'] == 'ok':
                lines.append(
                    f"| `{row['complexity']}` | {row['count']} | `{name}` | "
                    f"{item['timing']['median_seconds']:.8f} | "
                    f"{item['parity']['max_abs_diff']:.3g} |")
    return lines


def build_report(mode):
    profile = dict(PROFILES[mode])
    return {
        'schema_version': 1,
        'mode': mode,
        'created_unix': time.time(),
        'environment': environment_info(),
        'source_state': source_state(),
        'profile': profile,
        'box_benchmarks': box_benchmarks(profile),
        'mask_benchmarks': mask_benchmarks(profile),
        'edge_cases': edge_case_checks(),
    }


def write_bundle(report, output_dpath):
    dpath = Path(output_dpath)
    dpath.mkdir(parents=True, exist_ok=True)
    mode = report['mode']
    json_fpath = dpath / f'phase15_{mode}_results.json'
    md_fpath = dpath / f'phase15_{mode}_summary.md'
    json_fpath.write_text(json.dumps(report, indent=2, default=_json_default) + '\n')
    md_fpath.write_text('\n'.join(summary_lines(report)) + '\n')
    bundle = dpath / f'kwcoco_metrics_phase15_{mode}.tar.gz'
    with tarfile.open(bundle, 'w:gz') as tar:
        tar.add(json_fpath, arcname=json_fpath.name)
        tar.add(md_fpath, arcname=md_fpath.name)
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=sorted(PROFILES), default='quick')
    parser.add_argument('--output-dpath', type=Path,
                        default=Path('_metric_compare_phase15'))
    args = parser.parse_args(argv)
    report = build_report(args.mode)
    bundle = write_bundle(report, args.output_dpath)
    print('Phase-1.5 suite complete')
    print(f'  bundle: {bundle}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
