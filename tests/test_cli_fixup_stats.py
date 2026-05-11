import sys
import types

import pytest


def test_find_corrupted_assets_auxiliary_iteration(monkeypatch, tmp_path):
    from kwcoco.cli import coco_fixup
    import kwcoco

    class DummyJob:
        def __init__(self, func, args, kwargs):
            self._func = func
            self._args = args
            self._kwargs = kwargs
            self.input_info = None

        def result(self):
            return self._func(*self._args, **self._kwargs)

    class DummyPool:
        def __init__(self, *args, **kwargs):
            self._jobs = []

        def submit(self, func, *args, **kwargs):
            job = DummyJob(func, args, kwargs)
            self._jobs.append(job)
            return job

        def as_completed(self, desc=None, progkw=None):
            return list(self._jobs)

    def fake_corruption_check(*args, **kwargs):
        return {'failed': False}

    monkeypatch.setattr(coco_fixup.ub, 'JobPool', DummyPool)
    monkeypatch.setattr(
        'kwcoco._helpers._image_corruption_check', fake_corruption_check
    )

    dset = kwcoco.CocoDataset()
    dset.bundle_dpath = tmp_path
    dset.dataset['images'] = [
        {
            'id': 1,
            'file_name': 'img.png',
            'auxiliary': [
                {'file_name': 'aux1.png'},
                {'file_name': 'aux2.png'},
            ],
        }
    ]

    corrupted = coco_fixup.find_corrupted_assets(
        dset, check_aux=True, workers=0, corrupted_assets='full'
    )
    assert corrupted == []


def test_coco_fixup_missing_src():
    from kwcoco.cli.coco_fixup import CocoFixup

    with pytest.raises(ValueError, match='must specify source'):
        CocoFixup.main(cmdline=0, src=None, dst='unused.json')


def test_coco_fixup_invalid_corrupted_assets(tmp_path):
    from kwcoco.cli import coco_fixup
    import kwcoco

    dset = kwcoco.CocoDataset()
    dset.bundle_dpath = tmp_path
    dset.dataset['images'] = []

    with pytest.raises(
        ValueError, match="corrupted_assets must be 'only_shape' or 'full'"
    ):
        coco_fixup.find_corrupted_assets(dset, corrupted_assets='bad')


def test_coco_fixup_missing_osgeo(monkeypatch, tmp_path):
    from kwcoco.cli.coco_fixup import CocoFixup
    import kwcoco

    dset = kwcoco.CocoDataset()
    dset.fpath = tmp_path / 'input.json'
    dset.dump()
    dst = tmp_path / 'fixed.json'

    fake_osgeo = types.ModuleType('osgeo')
    monkeypatch.setitem(sys.modules, 'osgeo', fake_osgeo)
    monkeypatch.delitem(sys.modules, 'osgeo.gdal', raising=False)

    CocoFixup.main(
        cmdline=0,
        src=dset.fpath,
        dst=dst,
        missing_assets=False,
        corrupted_assets=False,
    )

    assert dst.exists()


def test_coco_stats_missing_src():
    from kwcoco.cli.coco_stats import CocoStatsCLI

    with pytest.raises(ValueError, match='must specify source'):
        CocoStatsCLI.main(cmdline=0, src=None)


def test_coco_stats_does_not_clobber_pandas_option(tmp_path):
    import pandas as pd
    from kwcoco.cli.coco_stats import CocoStatsCLI
    import kwcoco

    dset = kwcoco.CocoDataset()
    dset.fpath = tmp_path / 'input.json'
    dset.dump()

    original_value = pd.get_option('max_colwidth')
    pd.set_option('max_colwidth', 123)
    try:
        CocoStatsCLI.main(
            cmdline=0,
            src=str(dset.fpath),
            format='json',
            extended=False,
            catfreq=False,
            basic=True,
        )
        assert pd.get_option('max_colwidth') == 123
    finally:
        pd.set_option('max_colwidth', original_value)
