import numpy as np
import pytest


pycocotools = pytest.importorskip('pycocotools')


def _make_single_instance_dsets(true_seg, pred_seg, *, dims=(64, 64)):
    import kwcoco

    height, width = dims
    true = kwcoco.CocoDataset()
    pred = kwcoco.CocoDataset()
    for dset in [true, pred]:
        dset.add_category('thing', id=1)
        dset.add_image(
            file_name='image.png', id=1, width=width, height=height
        )

    bbox = [8, 8, 48, 48]
    true.add_annotation(
        image_id=1,
        category_id=1,
        bbox=bbox,
        segmentation=true_seg,
    )
    pred.add_annotation(
        image_id=1,
        category_id=1,
        bbox=bbox,
        segmentation=pred_seg,
        score=1.0,
    )
    return true, pred


def _outer_square():
    return [[8, 8, 56, 8, 56, 56, 8, 56]]


def test_official_bbox_and_segm_metrics_can_disagree():
    """Same boxes but very different masks should separate the two metrics."""
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    true_seg = _outer_square()
    pred_seg = [[8, 8, 18, 8, 18, 18, 8, 18]]
    true, pred = _make_single_instance_dsets(true_seg, pred_seg)

    dmet = DetectionMetrics.from_coco(true, pred)
    bbox_info = dmet.score_pycocotools(iou_type='bbox')
    segm_info = dmet.score_pycocotools(iou_type='segm')

    assert bbox_info['backend'] == 'pycocotools.COCOeval'
    assert bbox_info['iou_type'] == 'bbox'
    assert segm_info['iou_type'] == 'segm'
    assert bbox_info['summary']['AP50'] > 0.99
    assert segm_info['summary']['AP50'] < 0.01

    # Ordinary polygons should remain polygons at the strict COCO boundary.
    pred_coco, true_coco, _, _ = dmet._to_coco(iou_type='segm')
    assert isinstance(next(iter(true_coco.anns.values()))['segmentation'], list)
    assert isinstance(next(iter(pred_coco.anns.values()))['segmentation'], list)

    # Keep the historical output available while exposing unambiguous names.
    assert bbox_info['mAP'] == bbox_info['summary']['AP50']
    assert set(bbox_info['summary']) == {
        'AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
        'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large',
    }


def test_kwcoco_new_style_quadrilateral_is_not_misread_as_bbox():
    """A four-vertex N x 2 polygon must not trigger pycocotools bbox dispatch."""
    from pycocotools import mask as mask_utils
    from kwcoco.metrics.coco_metrics import coerce_pycocotools_segmentation

    # This is the no-hole rich/new-style representation emitted by kwimage
    # after perturb/warp operations.  Exactly four vertices is the important
    # regression: pycocotools dispatches raw Python lists by their length.
    rich_quad = {
        'exterior': [[8, 8], [56, 8], [56, 56], [8, 56]],
        'interiors': [],
    }
    strict, area = coerce_pycocotools_segmentation(rich_quad, (64, 64))
    assert isinstance(strict, list)

    # The bridge must normalize whatever kwimage emitted into the strict COCO
    # polygon shape before asking frPyObjects to rasterize it.
    from kwcoco.metrics.coco_metrics import _strict_segmentation_to_rle
    rle = _strict_segmentation_to_rle(strict, (64, 64))
    decoded = mask_utils.decode(rle)
    assert decoded[16, 16] == 1
    assert decoded[4, 4] == 0
    assert area > 0

    # Exercise the same end-to-end metric boundary used by the comparison
    # suite.  Rich no-hole polygons are a common output of perturb/warp code.
    from kwcoco.metrics.detect_metrics import DetectionMetrics
    true, pred = _make_single_instance_dsets(rich_quad, rich_quad)
    dmet = DetectionMetrics.from_coco(true, pred)
    info = dmet.score_pycocotools(iou_type='segm')
    assert info['summary']['AP50'] > 0.99


def test_kwcoco_polygon_hole_is_preserved_at_strict_coco_boundary():
    """Rich kwcoco polygon interiors must not be filled by strict COCO export."""
    from pycocotools import mask as mask_utils
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    rich_poly = {
        'exterior': [[8, 8], [56, 8], [56, 56], [8, 56]],
        'interiors': [
            [[24, 24], [40, 24], [40, 40], [24, 40]],
        ],
    }
    true, pred = _make_single_instance_dsets(rich_poly, rich_poly)

    dmet = DetectionMetrics.from_coco(true, pred)
    segm_info = dmet.score_pycocotools(iou_type='segm')
    assert segm_info['summary']['AP50'] > 0.99

    pred_coco, true_coco, _, _ = dmet._to_coco(iou_type='segm')
    for dset in [true_coco, pred_coco]:
        strict_seg = next(iter(dset.anns.values()))['segmentation']
        # MS-COCO polygons cannot encode holes, so this must cross the strict
        # boundary as an exact RLE mask.
        assert isinstance(strict_seg, dict)
        rle = dict(strict_seg)
        if isinstance(rle['counts'], str):
            rle['counts'] = rle['counts'].encode('ascii')
        mask = mask_utils.decode(rle)
        assert mask[12, 12] == 1
        assert mask[32, 32] == 0


def test_coco_evaluator_official_multi_geometry_api():
    from kwcoco.coco_evaluator import CocoEvaluator

    true_seg = _outer_square()
    pred_seg = [[8, 8, 18, 8, 18, 18, 8, 18]]
    true, pred = _make_single_instance_dsets(true_seg, pred_seg)

    evaluator = CocoEvaluator({
        'true_dataset': true,
        'pred_dataset': pred,
    })
    official = evaluator.evaluate_pycocotools(['bbox', 'segm'])

    assert set(official) == {'bbox', 'segm'}
    assert official['bbox']['summary']['AP50'] > 0.99
    assert official['segm']['summary']['AP50'] < 0.01
    assert official['bbox']['backend'] == 'pycocotools.COCOeval'
    assert official['bbox']['implementation'] == 'pycocotools.COCOeval'
    assert official['bbox']['official'] is True
    assert official['bbox']['result_schema'] == 'coco_detection_summary_v1'
    assert official['bbox']['reference_implementation'] == 'pycocotools.COCOeval'
    assert official['bbox']['iou_type'] == 'bbox'
    assert official['segm']['iou_type'] == 'segm'


def test_default_pycocotools_mode_remains_bbox():
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    true, pred = _make_single_instance_dsets(_outer_square(), _outer_square())
    dmet = DetectionMetrics.from_coco(true, pred)
    implicit = dmet.score_pycocotools()
    explicit = dmet.score_pycocotools(iou_type='bbox')
    np.testing.assert_allclose(implicit['stats'], explicit['stats'])
    assert implicit['iou_type'] == explicit['iou_type'] == 'bbox'


def test_rle_input_remains_mask_exact_through_metric_bridge():
    from pycocotools import mask as mask_utils
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[7:43, 11:51] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle = dict(rle)
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('ascii')
    rle['size'] = list(rle['size'])

    true, pred = _make_single_instance_dsets(rle, rle)
    dmet = DetectionMetrics.from_coco(true, pred)
    info = dmet.score_pycocotools(iou_type='segm')
    assert info['summary']['AP50'] > 0.99

    pred_coco, true_coco, _, _ = dmet._to_coco(iou_type='segm')
    for dset in [true_coco, pred_coco]:
        strict = dict(next(iter(dset.anns.values()))['segmentation'])
        if isinstance(strict['counts'], str):
            strict['counts'] = strict['counts'].encode('ascii')
        decoded = mask_utils.decode(strict)
        np.testing.assert_array_equal(decoded, mask)



def test_degenerate_two_vertex_polygon_becomes_empty_rle():
    """Degenerate polygon parts must not be mis-dispatched as COCO bboxes."""
    from pycocotools import mask as mask_utils
    from kwcoco.metrics.coco_metrics import coerce_pycocotools_segmentation

    # A collapsed polygon can arise after perturbation / clipping.  Four
    # coordinate values are especially dangerous because pycocotools treats
    # a 4-value list as an xywh bbox rather than polygon coordinates.
    degenerate = [[8, 8, 18, 18]]
    strict, area = coerce_pycocotools_segmentation(degenerate, (64, 64))
    assert isinstance(strict, dict)
    rle = dict(strict)
    if isinstance(rle['counts'], str):
        rle['counts'] = rle['counts'].encode('ascii')
    decoded = mask_utils.decode(rle)
    assert decoded.sum() == 0
    assert area == 0


def test_demo_class_noise_segmentation_metric_does_not_crash():
    """Regression for the comparison-suite class-noise perturbation path."""
    import kwcoco
    from kwcoco.demo.perterb import perterb_coco
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    true = kwcoco.CocoDataset.demo(
        'shapes4', image_size=(128, 128), rng=0, verbose=0, use_cache=False
    )
    pred = perterb_coco(
        true,
        rng=1,
        box_noise=1.0,
        cls_noise=0.20,
        score_noise=0.30,
        n_fp=(0, 1),
        n_fn=(0, 1),
        with_probs=False,
        verbose=0,
    )
    dmet = DetectionMetrics.from_coco(true, pred)
    info = dmet.score_pycocotools(iou_type='segm', verbose=0)
    assert 'summary' in info
    assert np.isfinite(info['summary']['AP']) or info['summary']['AP'] == -1


def test_official_mask_assignment_can_use_native_kwcoco_measures():
    """COCO mask assignment should remain inspectable via kwcoco Measures."""
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    true_seg = _outer_square()
    pred_seg = [[8, 8, 18, 8, 18, 18, 8, 18]]
    true, pred = _make_single_instance_dsets(true_seg, pred_seg)

    dmet = DetectionMetrics.from_coco(true, pred)
    info = dmet.score_pycocotools(
        iou_type='segm',
        with_confusion=True,
        iou_thresholds=None,
    )
    cfsn = info['iou_to_cfsn_vecs'][0.5]
    native_measures = cfsn.binarize_ovr().measures(
        ap_method='pycocotools'
    )

    # This case is deliberately a mask miss despite identical boxes.  The
    # important contract is that official mask assignment can be inspected
    # and accumulated by the native kwcoco Measures machinery.  Native
    # accumulation is not expected to be numerically identical to COCOeval,
    # which is exactly what the comparison suite is designed to measure.
    perfect_true, perfect_pred = _make_single_instance_dsets(true_seg, true_seg)
    perfect_dmet = DetectionMetrics.from_coco(perfect_true, perfect_pred)
    perfect_info = perfect_dmet.score_pycocotools(
        iou_type='segm',
        with_confusion=True,
        iou_thresholds=None,
    )
    perfect_cfsn = perfect_info['iou_to_cfsn_vecs'][0.5]
    perfect_native = perfect_cfsn.binarize_ovr().measures(
        ap_method='pycocotools'
    )

    assert np.isfinite(native_measures['mAP'])
    assert native_measures['mAP'] < perfect_native['mAP']
    assert perfect_native['mAP'] > 0.99
    assert info['summary']['AP50'] < 0.01


def test_score_pycocotools_same_length_iou_threshold_override():
    """Explicit IoU vectors are compared by value, not only by length."""
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    true_seg = _outer_square()
    true, pred = _make_single_instance_dsets(true_seg, true_seg)
    dmet = DetectionMetrics.from_coco(true, pred)

    # Passing the exact standard vector explicitly is still a standard COCO
    # evaluation and must retain the stable named summary surface.
    standard = np.arange(0.50, 0.96, 0.05)
    standard_info = dmet.score_pycocotools(
        iou_type='segm', iou_thresholds=standard, with_evaler=True, verbose=0
    )
    assert np.allclose(standard_info['evaler'].params.iouThrs, standard)
    assert 'summary' in standard_info
    assert standard_info['result_schema'] == 'coco_detection_summary_v1'

    # A different vector with the same length must actually replace COCOeval's
    # defaults.  The previous conditional accidentally ignored this case.
    custom = np.linspace(0.10, 0.55, len(standard))
    custom_info = dmet.score_pycocotools(
        iou_type='segm', iou_thresholds=custom, with_evaler=True, verbose=0
    )
    assert np.allclose(custom_info['evaler'].params.iouThrs, custom)

    # Even a small explicit change is a real parameter override. Do not use a
    # tolerance test to decide whether the caller's grid should be installed.
    near_standard = standard.copy()
    near_standard[-1] += 1e-12
    near_info = dmet.score_pycocotools(
        iou_type='segm', iou_thresholds=near_standard,
        with_evaler=True, verbose=0
    )
    assert np.array_equal(near_info['evaler'].params.iouThrs, near_standard)
