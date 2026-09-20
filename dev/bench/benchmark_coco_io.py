#!/usr/bin/env python3
"""Profile KWCoco manifest construction, validation, serialization, and ZIP I/O.

This benchmark is intentionally standalone and non-mutating.  Given one or
more existing manifests it writes all comparison artifacts into a temporary
work directory.  Synthetic datasets are also available for construction
benchmarks.

Examples:
    python dev/bench/benchmark_coco_io.py \
        /data/path/train_positive_tiles.kwcoco.zip \
        /data/path/train_negative_tiles.kwcoco.zip

    python dev/bench/benchmark_coco_io.py --synthetic
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import zipfile

import psutil

import kwcoco


class PeakRSS:
    """Sample incremental RSS while an operation runs."""

    def __init__(self, interval=0.002):
        self.interval = interval
        self.proc = psutil.Process()
        self.base = 0
        self.peak = 0
        self._stop = threading.Event()

    def __enter__(self):
        self.base = self.proc.memory_info().rss
        self.peak = self.base

        def _sample():
            while not self._stop.wait(self.interval):
                self.peak = max(self.peak, self.proc.memory_info().rss)

        self.thread = threading.Thread(target=_sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)

    @property
    def delta_mb(self):
        return (self.peak - self.base) / (1024 ** 2)


def measure(label, func, *, meta=None):
    """Measure wall time, CPU time, incremental peak RSS, and output bytes."""
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    with PeakRSS() as rss:
        value = func()
    row = {
        'stage': label,
        'wall_s': time.perf_counter() - wall0,
        'cpu_s': time.process_time() - cpu0,
        'peak_rss_delta_mb': rss.delta_mb,
    }
    if meta:
        row.update(meta)
    if isinstance(value, (str, bytes, bytearray)):
        row['bytes'] = len(value)
    elif isinstance(value, (str, os.PathLike, Path)):
        path = Path(value)
        if path.exists():
            row['bytes'] = path.stat().st_size
    print(json.dumps(row), flush=True)
    return value, row


def _validation_config(**enabled):
    config = {
        'schema': False,
        'unique': False,
        'missing': False,
        'corrupted': False,
        'channels': False,
        'img_attrs': False,
        'annot_attrs': False,
        'verbose': 0,
    }
    config.update(enabled)
    return config


def _zip_bytes(data, fpath, compression, compresslevel=None):
    kwargs = {'compression': compression}
    if compresslevel is not None:
        kwargs['compresslevel'] = compresslevel
    with zipfile.ZipFile(fpath, 'w', **kwargs) as zfile:
        zfile.writestr('data.kwcoco.json', data)
    return fpath


def benchmark_loaded_dataset(
    dset, label, workdir, check_missing=True, include_lzma=False
):
    """Benchmark validation and output paths for one already loaded dataset."""
    meta = {
        'dataset': label,
        'n_images': len(dset.dataset.get('images', [])),
        'n_annots': len(dset.dataset.get('annotations', [])),
    }
    rows = []

    auto_config = _validation_config(schema=True)
    jsonschema_config = _validation_config(schema=True)
    jsonschema_config['schema_backend'] = 'jsonschema'
    validation_cases = [
        ('validate_schema_auto', auto_config),
        ('validate_schema_jsonschema', jsonschema_config),
        ('validate_unique', _validation_config(unique=True)),
        ('validate_channels', _validation_config(channels=True)),
        ('validate_annot_attrs', _validation_config(annot_attrs=True)),
    ]
    try:
        import fastjsonschema  # NOQA
    except ImportError:
        pass
    else:
        fast_config = _validation_config(schema=True)
        fast_config['schema_backend'] = 'fastjsonschema'
        validation_cases.insert(1, ('validate_schema_fastjsonschema', fast_config))

    for stage, config in validation_cases:
        _, row = measure(stage, lambda c=config: dset.validate(**c), meta=meta)
        rows.append(row)

    if check_missing:
        _, row = measure(
            'validate_missing',
            lambda: dset.validate(**_validation_config(missing=True)),
            meta=meta,
        )
        rows.append(row)

    text, row = measure('json_dumps_to_text', dset.dumps, meta=meta)
    rows.append(row)
    payload, row = measure('utf8_encode', lambda: text.encode('utf8'), meta=meta)
    rows.append(row)

    raw_fpath = workdir / f'{label}.kwcoco.json'

    def write_raw_payload():
        raw_fpath.write_bytes(payload)
        return raw_fpath

    _, row = measure('filesystem_write_json_bytes', write_raw_payload, meta=meta)
    rows.append(row)

    fsync_fpath = workdir / f'{label}.fsync.kwcoco.json'

    def write_raw_payload_fsync():
        with fsync_fpath.open('wb') as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        return fsync_fpath

    _, row = measure(
        'filesystem_write_json_bytes_fsync', write_raw_payload_fsync, meta=meta
    )
    rows.append(row)

    _, row = measure(
        'reload_json', lambda: kwcoco.CocoDataset(raw_fpath), meta=meta
    )
    rows.append(row)

    codec_cases = [
        ('zip_deflate_6', zipfile.ZIP_DEFLATED, 6),
        ('zip_deflate_1', zipfile.ZIP_DEFLATED, 1),
        ('zip_stored', zipfile.ZIP_STORED, None),
    ]
    if include_lzma:
        # LZMA can take minutes on large manifests, so it is deliberately
        # excluded from the default benchmark loop.
        codec_cases.append(('zip_lzma', zipfile.ZIP_LZMA, None))
    codec_paths = {}
    for name, codec, level in codec_cases:
        fpath = workdir / f'{label}.{name}.kwcoco.zip'
        codec_paths[name] = fpath
        _, row = measure(
            name,
            lambda p=fpath, c=codec, lev=level: _zip_bytes(payload, p, c, lev),
            meta=meta,
        )
        rows.append(row)

    for name, fpath in codec_paths.items():
        _, row = measure(
            f'reload_{name}',
            lambda p=fpath: kwcoco.CocoDataset(p),
            meta=meta,
        )
        rows.append(row)

    default_zip = workdir / f'{label}.default.kwcoco.zip'

    def dump_default_zip_core():
        # Use a caller-owned file object so this benchmark isolates KWCoco's
        # serialization / compression path from the safer.open wrapper.
        with default_zip.open('wb') as file:
            dset.dump(file, compress=True)
        return default_zip

    _, row = measure(
        'cocodataset_dump_default_zip_core',
        dump_default_zip_core,
        meta=meta,
    )
    rows.append(row)

    _, row = measure(
        'reload_default_zip',
        lambda: kwcoco.CocoDataset(default_zip),
        meta=meta,
    )
    rows.append(row)
    return rows


def make_synthetic_tables(n_images, n_annots):
    images = [
        {
            'id': i + 1,
            'name': f'tile_{i:06d}',
            'file_name': f'../_cache/sha256/{i % 256:02x}/{i:064x}.jpg',
            'width': 800,
            'height': 800,
            'channels': 'red|green|blue',
            'source_image_id': (i % 4000) + 1,
            'scale': [1.0, 0.75, 0.5][i % 3],
        }
        for i in range(n_images)
    ]
    annotations = []
    for i in range(n_annots):
        x = float(i % 731)
        y = float((i * 7) % 719)
        w = float(20 + (i % 100))
        h = float(18 + (i % 80))
        annotations.append(
            {
                'id': i + 1,
                'image_id': (i % n_images) + 1,
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'segmentation': [[x, y, x + w, y, x + w, y + h, x, y + h]],
            }
        )
    return {
        'info': [{'description': 'synthetic kwcoco I/O benchmark'}],
        'licenses': [],
        'categories': [{'id': 1, 'name': 'positive'}],
        'videos': [],
        'images': images,
        'annotations': annotations,
    }


def benchmark_construction(data, label):
    meta = {
        'dataset': label,
        'n_images': len(data['images']),
        'n_annots': len(data['annotations']),
    }
    rows = []

    _, row = measure(
        'construct_from_raw_tables_and_build_index',
        lambda: kwcoco.CocoDataset(data),
        meta=meta,
    )
    rows.append(row)

    def construct_bulk():
        dset = kwcoco.CocoDataset()
        dset.add_category(name='positive', id=1)
        dset.add_images(data['images'])
        dset.add_annotations(data['annotations'])
        return dset

    _, row = measure('construct_with_bulk_add', construct_bulk, meta=meta)
    rows.append(row)

    def construct_single():
        dset = kwcoco.CocoDataset()
        dset.add_category(name='positive', id=1)
        for img in data['images']:
            item = dict(img)
            gid = item.pop('id')
            file_name = item.pop('file_name', None)
            dset.add_image(file_name=file_name, id=gid, **item)
        for ann in data['annotations']:
            item = dict(ann)
            aid = item.pop('id')
            image_id = item.pop('image_id')
            category_id = item.pop('category_id', None)
            bbox = item.pop('bbox', None)
            segmentation = item.pop('segmentation', None)
            dset.add_annotation(
                image_id=image_id,
                category_id=category_id,
                bbox=bbox,
                segmentation=segmentation,
                id=aid,
                **item,
            )
        return dset

    _, row = measure('construct_with_single_add', construct_single, meta=meta)
    rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifests', nargs='*', type=Path)
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='also benchmark 124834/0 and 62417/100000 synthetic datasets',
    )
    parser.add_argument(
        '--skip-missing',
        action='store_true',
        help='skip filesystem existence validation for real manifests',
    )
    parser.add_argument(
        '--include-lzma',
        action='store_true',
        help='include the intentionally slow LZMA comparison case',
    )
    parser.add_argument(
        '--workdir',
        type=Path,
        default=None,
        help='parent directory for temporary outputs; use a /data path to measure that filesystem',
    )
    parser.add_argument(
        '--output', type=Path, default=None, help='optional JSONL summary path'
    )
    args = parser.parse_args()

    temp_parent = None
    if args.workdir is not None:
        temp_parent = args.workdir.expanduser().resolve()
        temp_parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    with tempfile.TemporaryDirectory(
        prefix='kwcoco-io-bench-', dir=temp_parent
    ) as tmp:
        workdir = Path(tmp)
        for manifest in args.manifests:
            manifest = manifest.expanduser().resolve()
            dset, row = measure(
                'load_manifest',
                lambda p=manifest: kwcoco.CocoDataset(p),
                meta={'dataset': manifest.name},
            )
            all_rows.append(row)
            all_rows.extend(
                benchmark_loaded_dataset(
                    dset,
                    manifest.stem.replace('.', '_'),
                    workdir,
                    check_missing=not args.skip_missing,
                    include_lzma=args.include_lzma,
                )
            )

        if args.synthetic:
            for label, n_images, n_annots in [
                ('negative125k', 124_834, 0),
                ('positive62k100k', 62_417, 100_000),
            ]:
                data, row = measure(
                    'construct_synthetic_raw_tables',
                    lambda ni=n_images, na=n_annots: make_synthetic_tables(ni, na),
                    meta={'dataset': label},
                )
                all_rows.append(row)
                all_rows.extend(benchmark_construction(data, label))
                dset = kwcoco.CocoDataset(data)
                all_rows.extend(
                    benchmark_loaded_dataset(
                        dset,
                        label,
                        workdir,
                        check_missing=False,
                        include_lzma=args.include_lzma,
                    )
                )
                del dset, data
                gc.collect()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w') as file:
            for row in all_rows:
                file.write(json.dumps(row) + '\n')
        print(f'wrote summary: {args.output}')


if __name__ == '__main__':
    main()
