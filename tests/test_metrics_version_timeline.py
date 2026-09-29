"""Contract tests for the historical metric version benchmark."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'dev' / 'pycocotools_metrics_compare' / 'run_version_timeline.py'


def _load_runner():
    spec = importlib.util.spec_from_file_location('run_version_timeline', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_version_timeline_frozen_workload_is_deterministic():
    runner = _load_runner()
    a = runner.make_frozen_coco_pair(n_images=3, n_classes=2, truth_per_image=2, seed=3)
    b = runner.make_frozen_coco_pair(n_images=3, n_classes=2, truth_per_image=2, seed=3)
    assert a == b
    truth, pred = a
    assert len(truth['annotations']) == 6
    assert len(pred['annotations']) == 7


def test_version_timeline_metric_state_diff():
    runner = _load_runner()
    baseline = {
        'result': {
            'nocls_measures': {'ap': 0.5, 'thresholds': [0.1, 0.2]},
            'ovr_measures': {'perclass': {'a': {'ap': 0.75}}},
        }
    }
    same = {
        'result': {
            'nocls_measures': {'ap': 0.5, 'thresholds': [0.1, 0.2]},
            'ovr_measures': {'perclass': {'a': {'ap': 0.75}}},
        }
    }
    close = {
        'result': {
            'nocls_measures': {'ap': 0.5 + 1e-13, 'thresholds': [0.1, 0.2]},
            'ovr_measures': {'perclass': {'a': {'ap': 0.75}}},
        }
    }
    assert runner.compare_metric_states(baseline, same)['ok']
    assert not runner.compare_metric_states(baseline, close, atol=0.0)['ok']
    assert runner.compare_metric_states(baseline, close, atol=1e-12)['ok']


def test_version_timeline_worker_uses_historical_evaluator_config(monkeypatch):
    import sys
    import types

    worker_fpath = (
        ROOT / 'dev' / 'pycocotools_metrics_compare' / 'version_timeline_worker.py'
    )
    spec = importlib.util.spec_from_file_location('version_timeline_worker', worker_fpath)
    worker = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(worker)

    class FakeCocoEvaluator:
        def __init__(self, config):
            # These are the only cross-version constructor keys the worker needs.
            assert set(config) == {'true_dataset', 'pred_dataset'}
            self.config = config
            self._verbose = 1

    fake_package = types.ModuleType('kwcoco')
    fake_package.__path__ = []
    fake_module = types.ModuleType('kwcoco.coco_evaluator')
    fake_module.CocoEvaluator = FakeCocoEvaluator
    monkeypatch.setitem(sys.modules, 'kwcoco', fake_package)
    monkeypatch.setitem(sys.modules, 'kwcoco.coco_evaluator', fake_module)

    evaluator = worker._make_evaluator(Path('truth.json'), Path('pred.json'))
    assert evaluator.config == {
        'true_dataset': 'truth.json',
        'pred_dataset': 'pred.json',
    }
    assert evaluator._verbose == 0
