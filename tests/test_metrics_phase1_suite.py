"""Small structural tests for the dev Phase-1 metric comparison suite."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_phase1_module():
    import kwcoco

    repo_dpath = Path(kwcoco.__file__).resolve().parent.parent
    fpath = repo_dpath / 'dev/pycocotools_metrics_compare/run_phase1_suite.py'
    spec = importlib.util.spec_from_file_location('kwcoco_metric_phase1_suite', fpath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phase1_adversarial_fixture_shapes():
    module = _load_phase1_module()

    true, pred, _ = module.make_adversarial_case('iou_050_boundary')
    assert true.n_images == pred.n_images == 1
    assert true.n_annots == pred.n_annots == 1

    true, pred, _ = module.make_adversarial_case('score_tie_duplicate')
    assert true.n_annots == 1
    assert pred.n_annots == 2

    true, pred, _ = module.make_adversarial_case('empty_prediction')
    assert true.n_annots == 1
    assert pred.n_annots == 0

    true, pred, _ = module.make_adversarial_case('empty_truth')
    assert true.n_annots == 0
    assert pred.n_annots == 1


def test_phase1_hole_fixture_preserves_interior():
    module = _load_phase1_module()
    true, pred, _ = module.make_adversarial_case('hole_mismatch')
    true_ann = next(iter(true.anns.values()))
    pred_ann = next(iter(pred.anns.values()))
    assert len(true_ann['segmentation']['interiors']) == 1
    assert len(pred_ann['segmentation']['interiors']) == 0
