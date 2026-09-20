"""Focused tests for experimental native instance-mask assignment."""


def _same_box_disjoint_masks():
    import kwimage
    import numpy as np

    dims = (20, 20)
    true_mask = np.zeros(dims, dtype=np.uint8)
    pred_mask = np.zeros(dims, dtype=np.uint8)
    true_mask[2:10, 2:10] = 1
    pred_mask[10:18, 10:18] = 1
    box_data = np.array([[0, 0, 20, 20]], dtype=float)
    true = kwimage.Detections(
        boxes=kwimage.Boxes(box_data.copy(), 'xywh'),
        class_idxs=np.array([0]),
        segmentations=[kwimage.Mask(true_mask, 'c_mask')],
    )
    pred = kwimage.Detections(
        boxes=kwimage.Boxes(box_data.copy(), 'xywh'),
        class_idxs=np.array([0]),
        scores=np.array([0.9]),
        segmentations=[kwimage.Mask(pred_mask, 'c_mask')],
    )
    return dims, true, pred


def test_segmentation_assignment_separates_box_and_mask_geometry():
    from kwcoco.metrics.assignment_v2 import assign_confusion_vectors_v2

    dims, true, pred = _same_box_disjoint_masks()
    bbox_y = assign_confusion_vectors_v2(
        true, pred,
        iou_thresh=0.5,
        compat='mutex',
        ignore_classes=None,
        spatial_threshold=None,
        iou_type='bbox',
    )
    segm_y = assign_confusion_vectors_v2(
        true, pred,
        iou_thresh=0.5,
        compat='mutex',
        ignore_classes=None,
        spatial_threshold=None,
        iou_type='segm',
        image_dims=dims,
    )
    assert bbox_y['txs'] == [0]
    assert bbox_y['pxs'] == [0]
    assert segm_y['txs'] == [-1, 0]
    assert segm_y['pxs'] == [0, -1]


def test_segmentation_assignment_reuses_overlap_for_multiple_thresholds():
    from kwcoco.metrics.assignment_v2 import AssignmentCandidateIndex

    dims, true, pred = _same_box_disjoint_masks()
    index = AssignmentCandidateIndex.build(
        true, pred,
        compat='mutex',
        min_iou=0.0,
        iou_type='segm',
        image_dims=dims,
        spatial_threshold=None,
    )
    result = index.assign_many([0.0, 0.5], ignore_classes=None)
    assert set(result) == {0.0, 0.5}
    assert index.build_stats['iou_type'] == 'segm'
    assert index.build_stats['mask_pairs_considered'] == 1


def test_detection_metrics_v2_mask_path_on_coco_data():
    import kwcoco
    from kwcoco.demo.perterb import perterb_coco
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    true = kwcoco.CocoDataset.demo(
        'shapes2', image_size=(64, 64), rng=0, verbose=0, use_cache=False)
    pred = perterb_coco(
        true, rng=1, box_noise=0, cls_noise=0, score_noise=0,
        n_fp=0, n_fn=0, with_probs=False, verbose=0)
    dmet = DetectionMetrics.from_coco(true, pred)
    cfsn = dmet.confusion_vectors_v2(
        iou_thresh=0.5,
        iou_type='segm',
        compat='mutex',
        ignore_classes=None,
        spatial_threshold=None,
    )
    assert len(cfsn.data['gid']) >= 1
    assert 'iou' in cfsn.data


def _half_overlap_rle_case():
    import kwimage
    import numpy as np

    dims = (16, 16)
    true_mask = np.zeros(dims, dtype=np.uint8)
    pred_mask = np.zeros(dims, dtype=np.uint8)
    true_mask[2:10, 2:10] = 1
    pred_mask[2:10, 2:6] = 1
    # Prediction is a strict subset of truth with area 32 / 64 => IoU 0.5.
    box_data = np.array([[0, 0, 16, 16]], dtype=float)
    true = kwimage.Detections(
        boxes=kwimage.Boxes(box_data.copy(), 'xywh'),
        class_idxs=np.array([0]),
        segmentations=[kwimage.Mask(true_mask, 'c_mask')],
    )
    pred = kwimage.Detections(
        boxes=kwimage.Boxes(box_data.copy(), 'xywh'),
        class_idxs=np.array([0]),
        scores=np.array([0.9]),
        segmentations=[kwimage.Mask(pred_mask, 'c_mask')],
    )
    return dims, true, pred


def test_segmentation_threshold_policy_is_explicit():
    from kwcoco.metrics.assignment_v2 import AssignmentCandidateIndex

    dims, true, pred = _half_overlap_rle_case()
    strict = AssignmentCandidateIndex.build(
        true, pred,
        compat='mutex', min_iou=0.5,
        iou_type='segm', image_dims=dims,
        segmentation_rasterizer='coco',
        threshold_policy='strict',
        spatial_threshold=None,
    ).assign(0.5, ignore_classes=None)
    inclusive = AssignmentCandidateIndex.build(
        true, pred,
        compat='mutex', min_iou=0.5,
        iou_type='segm', image_dims=dims,
        segmentation_rasterizer='coco',
        threshold_policy='inclusive',
        spatial_threshold=None,
    ).assign(0.5, ignore_classes=None)
    assert strict['txs'] == [-1, 0]
    assert inclusive['txs'] == [0]


def test_coco_rasterizer_kwimage_ext_matches_pycocotools():
    import numpy as np
    import pytest
    from kwcoco.metrics.overlap import segmentation_to_rle

    pytest.importorskip('pycocotools')
    pytest.importorskip('kwimage_ext')

    dims = (40, 48)
    # Fractional quadrilateral chosen to exercise COCO's subpixel polygon
    # rasterization rather than generic cv2-style fill semantics.
    polygon = [[
        3.2, 4.7,
        31.6, 6.1,
        29.4, 28.8,
        5.1, 30.2,
    ]]
    rust_rle = segmentation_to_rle(
        polygon, dims,
        rasterizer='coco', raster_backend='kwimage_ext')
    ref_rle = segmentation_to_rle(
        polygon, dims,
        rasterizer='coco', raster_backend='pycocotools')

    from pycocotools import mask as mask_utils
    rust_mask = mask_utils.decode(rust_rle)
    ref_mask = mask_utils.decode(ref_rle)
    assert np.array_equal(rust_mask, ref_mask)
    assert float(mask_utils.area(rust_rle)) == float(mask_utils.area(ref_rle))
