"""Fast contract tests for the release-scale metric benchmark harness."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COMPARE_DPATH = ROOT / 'dev' / 'pycocotools_metrics_compare'


def _load_script(name):
    path = COMPARE_DPATH / name
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_release_benchmark_synthetic_counts():
    worker = _load_script('release_benchmark_worker.py')
    spec = {
        'name': 'unit',
        'kind': 'synthetic',
        'n_images': 4,
        'n_classes': 3,
        'truth_per_image': 2,
        'pred_per_image': 5,
        'image_size': 128,
        'negative_fraction': 0.0,
        'miss_fraction': 0.0,
        'seed': 0,
    }
    true, pred = worker._synthetic_dsets(spec)
    counts = worker._dataset_counts(true, pred)
    assert counts == {
        'images': 4,
        'truth_annotations': 8,
        'prediction_annotations': 20,
        'categories': 3,
    }
    assert all('segmentation' in ann for ann in true.anns.values())
    assert all('score' in ann for ann in pred.anns.values())


def test_release_tensor_diff_allows_negative_scores(tmp_path):
    runner = _load_script('run_release_benchmark.py')
    precision = np.array([[[[[0.75]]], [[[0.50]]]]], dtype=float)
    recall = np.array([[[[1.0]]]], dtype=float)
    # Negative detection confidences are valid. Cell presence is defined by
    # precision, not by whether the sampled score itself is negative.
    scores = np.array([[[[[-0.2]]], [[[-0.7]]]]], dtype=float)
    realpos = np.array([[[[1]]]], dtype=int)
    native = tmp_path / 'native.npz'
    official = tmp_path / 'official.npz'
    for path in [native, official]:
        np.savez_compressed(
            path,
            precision=precision,
            recall=recall,
            scores=scores,
            realpos_total=realpos,
        )
    diff = runner._tensor_diff(native, official, atol=1e-12)
    assert diff['ok']
    assert diff['keys']['scores']['populated_cells'] == 2
