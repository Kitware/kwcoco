import runpy
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    'modname',
    [
        'kwcoco.cli.coco_eval',
        'kwcoco.cli.coco_toydata',
        'kwcoco.cli.coco_union',
    ],
)
def test_cli_modules_support_direct_help(modname):
    info = subprocess.run(
        [sys.executable, '-m', modname, '--help'],
        text=True,
        capture_output=True,
    )
    assert info.returncode == 0, info.stderr


def test_python_m_kwcoco_propagates_exit_code(monkeypatch):
    import kwcoco.cli.__main__ as cli_main

    monkeypatch.setattr(cli_main, 'main', lambda: 7)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module('kwcoco', run_name='__main__')
    assert exc_info.value.code == 7


def test_coco_eval_auto_output_path(monkeypatch, tmp_path):
    from kwcoco.cli import coco_eval

    class FakeResults:
        def __init__(self):
            self.dump_fpath = None

        def dump(self, fpath, indent=None):
            self.dump_fpath = fpath

        def dump_figures(self, *args, **kwargs):
            raise AssertionError('draw=False should disable figure output')

    results = FakeResults()

    class FakeEvaluator:
        def __init__(self, config):
            self.config = config

        def _init(self):
            pass

        def evaluate(self):
            return results

    monkeypatch.setattr(coco_eval.coco_evaluator, 'CocoEvaluator', FakeEvaluator)

    out_dpath = tmp_path / 'metrics'
    coco_eval.main(
        cmdline=0,
        true_dataset='unused-truth.json',
        pred_dataset='unused-pred.json',
        out_dpath=out_dpath,
        out_fpath='auto',
        draw=False,
    )

    assert results.dump_fpath == out_dpath / 'metrics.json'


def test_finish_install_reports_missing_executables(monkeypatch, capsys):
    from kwcoco.cli import finish_install

    monkeypatch.setattr(finish_install.ub, 'find_exe', lambda exe: None)
    finish_install.request_system_executables(['definitely-not-installed'])
    captured = capsys.readouterr()
    assert 'definitely-not-installed' in captured.out


def test_plot_stats_builds_cache_data_once(monkeypatch, tmp_path):
    import kwcoco
    from kwcoco.cli import coco_plot_stats

    class FakeDataset:
        fpath = tmp_path / 'data.kwcoco.json'

        def _cached_hashid(self):
            return 'fake-hash'

    fake_dset = FakeDataset()
    monkeypatch.setattr(
        kwcoco.CocoDataset, 'coerce', classmethod(lambda cls, src: fake_dset)
    )

    calls = []

    def fake_build_stats_data(dset):
        calls.append(dset)
        return {}, {}, {}, None

    class FakeCacher:
        def __init__(self, *args, **kwargs):
            pass

        def tryload(self):
            return None

        def save(self, data):
            self.data = data

    class FakePlots:
        def __init__(self, plots_dpath, tables_data, nonsaved_data, options, dpi):
            self.tables_data = tables_data

    monkeypatch.setattr(coco_plot_stats, 'build_stats_data', fake_build_stats_data)
    monkeypatch.setattr(coco_plot_stats.ub, 'Cacher', FakeCacher)
    monkeypatch.setattr(coco_plot_stats, 'Plots', FakePlots)

    coco_plot_stats.prep_plots('unused', tmp_path / 'plots')
    assert calls == [fake_dset]


def test_plot_stats_preserves_asymmetric_box_dimensions(monkeypatch):
    import kwcoco
    from kwcoco.cli import coco_plot_stats

    dset = kwcoco.CocoDataset()
    gid = dset.add_image(file_name='unused.png', width=100, height=100)
    dset.add_annotation(image_id=gid, bbox=[0, 0, 10, 20])

    monkeypatch.setattr(coco_plot_stats, 'geospatial_stats', lambda *args: None)

    def fake_polygon_shape_stats(data):
        data = data.copy()
        data['sseg_rt_area'] = data['box_rt_area']
        data['obox_major'] = data['box_height']
        data['obox_minor'] = data['box_width']
        return data

    monkeypatch.setattr(
        coco_plot_stats, 'polygon_shape_stats', fake_polygon_shape_stats
    )

    _, _, _, dataframes = coco_plot_stats.build_stats_data(dset)
    row = dataframes['perannot_data'].iloc[0]
    assert row['box_width'] == 10
    assert row['box_height'] == 20


def test_plot_stats_stops_process_context_once(monkeypatch, tmp_path):
    from kwcoco.cli import coco_plot_stats
    import kwutil.process_context

    class Config(dict):
        __getattr__ = dict.__getitem__

        def to_dict(self):
            return dict(self)

    config = Config(
        dst_dpath=tmp_path,
        dst_fpath=tmp_path / 'stats.json',
        with_process_context=True,
        src='unused',
        options=None,
        dpi=72,
        plots=[],
    )

    class FakePlots:
        tables_data = {}
        scalar_stats = {}
        plot_functions = {}

    stop_calls = []

    class FakeProcessContext:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            stop_calls.append(None)
            return {}

    monkeypatch.setattr(
        coco_plot_stats,
        'prep_plots',
        lambda *args, **kw: FakePlots(),
    )
    monkeypatch.setattr(
        kwutil.process_context, 'ProcessContext', FakeProcessContext
    )

    coco_plot_stats.run(config)
    assert len(stop_calls) == 1
