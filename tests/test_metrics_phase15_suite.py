"""Structural tests for the low-level Phase-1.5 backend benchmark."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    fpath = repo / 'dev/pycocotools_metrics_compare/run_phase15_suite.py'
    spec = importlib.util.spec_from_file_location('kwcoco_metric_phase15_suite', fpath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_numpy_box_iou_known_case():
    mod = _load_module()
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    got = mod.numpy_box_iou(a, a, bias=0.0)
    assert np.allclose(got, [[1.0]])


def test_mask_generators_are_fortran_and_nonempty():
    mod = _load_module()
    rng = np.random.RandomState(0)
    for kind in ['rectangles', 'fragmented', 'sparse_noise']:
        masks = mod.make_masks(rng, 3, (64, 80), kind)
        assert masks.shape == (64, 80, 3)
        assert masks.flags['F_CONTIGUOUS']
        assert masks.sum() > 0


def test_compare_arrays_threshold_diagnostics():
    mod = _load_module()
    left = np.array([[0.5, 0.75]])
    right = np.array([[0.499, 0.75]])
    info = mod.compare_arrays(left, right)
    assert info['ge_0.5_disagreements'] == 1
    assert info['ge_0.75_disagreements'] == 0
