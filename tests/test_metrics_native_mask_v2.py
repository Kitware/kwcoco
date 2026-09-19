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
