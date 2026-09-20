"""Crowd and ignore semantics for experimental native mask assignment."""


def _poly(x1, y1, x2, y2):
    return [[x1, y1, x2, y1, x2, y2, x1, y2]]


def _base_dsets():
    import kwcoco

    true = kwcoco.CocoDataset()
    pred = kwcoco.CocoDataset()
    for dset in [true, pred]:
        dset.add_category(name='object', id=1)
        dset.add_category(name='ignore', id=2)
        dset.add_image(id=1, file_name='im.png', width=64, height=64)
    return true, pred



def _require_native_mask_backend():
    import pytest
    try:
        from kwcoco.metrics.overlap import resolve_mask_iou_backend
        from kwcoco.metrics.overlap import resolve_coco_raster_backend
        resolve_mask_iou_backend('auto')
        resolve_coco_raster_backend('auto')
    except Exception as ex:
        pytest.skip(f'no native mask backend available: {ex!r}')

def _assignment_set(cfsn):
    import numpy as np

    data = cfsn.data
    gids = np.asarray(data['gid'])
    txs = np.asarray(data['txs'])
    pxs = np.asarray(data['pxs'])
    flags = (txs >= 0) & (pxs >= 0)
    return {
        (int(g), int(p), int(t))
        for g, p, t in zip(gids[flags], pxs[flags], txs[flags])
    }


def test_native_mask_crowd_reuse_matches_cocoeval():
    import pytest
    pytest.importorskip('pycocotools')

    from kwcoco.metrics.detect_metrics import DetectionMetrics

    true, pred = _base_dsets()
    # One regular truth and one large crowd truth of the same class.
    true.add_annotation(
        id=1, image_id=1, category_id=1,
        bbox=[4, 4, 10, 10], segmentation=_poly(4, 4, 14, 14), iscrowd=0)
    true.add_annotation(
        id=2, image_id=1, category_id=1,
        bbox=[28, 28, 26, 26], segmentation=_poly(28, 28, 54, 54), iscrowd=1)

    # Correct regular detection, two detections wholly inside the crowd, and FP.
    pred.add_annotation(
        id=11, image_id=1, category_id=1,
        bbox=[4, 4, 10, 10], segmentation=_poly(4, 4, 14, 14), score=.99)
    pred.add_annotation(
        id=12, image_id=1, category_id=1,
        bbox=[30, 30, 8, 8], segmentation=_poly(30, 30, 38, 38), score=.95)
    pred.add_annotation(
        id=13, image_id=1, category_id=1,
        bbox=[40, 40, 8, 8], segmentation=_poly(40, 40, 48, 48), score=.90)
    pred.add_annotation(
        id=14, image_id=1, category_id=1,
        bbox=[52, 4, 6, 6], segmentation=_poly(52, 4, 58, 10), score=.80)

    dmet = DetectionMetrics.from_coco(true, pred)
    native = dmet.confusion_vectors_v2(
        iou_thresh=0.5, iou_type='segm', compat='mutex',
        ignore_classes=None, spatial_threshold=None)
    official = dmet.score_pycocotools(
        iou_type='segm', iou_thresholds=[0.5],
        with_confusion=True, verbose=0)['iou_to_cfsn_vecs'][0.5]

    assert _assignment_set(native) == _assignment_set(official)
    data = native.data
    # Both crowd-contained predictions should survive as traceable matches but
    # carry zero metric weight.  Crowd truth itself must not become an FN.
    crowd_rows = [
        i for i, tx in enumerate(data['txs'])
        if int(tx) == 1 and int(data['pxs'][i]) >= 0
    ]
    assert len(crowd_rows) == 2
    assert all(float(data['weight'][i]) == 0 for i in crowd_rows)
    assert not any(int(tx) == 1 and int(px) < 0
                   for tx, px in zip(data['txs'], data['pxs']))


def test_native_mask_prefers_regular_truth_over_better_crowd_match():
    import pytest
    pytest.importorskip('pycocotools')

    from kwcoco.metrics.detect_metrics import DetectionMetrics

    true, pred = _base_dsets()
    # Prediction is fully inside crowd (crowd IOA=1), but also overlaps a
    # regular truth above threshold. COCO prefers the regular nonignored truth.
    true.add_annotation(
        id=1, image_id=1, category_id=1,
        bbox=[10, 10, 12, 12], segmentation=_poly(10, 10, 22, 22), iscrowd=0)
    true.add_annotation(
        id=2, image_id=1, category_id=1,
        bbox=[6, 6, 24, 24], segmentation=_poly(6, 6, 30, 30), iscrowd=1)
    pred.add_annotation(
        id=11, image_id=1, category_id=1,
        bbox=[11, 11, 10, 10], segmentation=_poly(11, 11, 21, 21), score=.99)

    dmet = DetectionMetrics.from_coco(true, pred)
    native = dmet.confusion_vectors_v2(
        iou_thresh=0.5, iou_type='segm', compat='mutex',
        ignore_classes=None, spatial_threshold=None)
    official = dmet.score_pycocotools(
        iou_type='segm', iou_thresholds=[0.5],
        with_confusion=True, verbose=0)['iou_to_cfsn_vecs'][0.5]
    assert _assignment_set(native) == _assignment_set(official)
    matched = [
        (int(px), int(tx), float(w))
        for px, tx, w in zip(native.data['pxs'], native.data['txs'], native.data['weight'])
        if int(px) >= 0 and int(tx) >= 0
    ]
    assert matched == [(0, 0, 1.0)]


def test_native_mask_ignore_class_uses_union_coverage():
    _require_native_mask_backend()
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    true, pred = _base_dsets()
    # Two overlapping ignore regions cover one object prediction.  They are
    # merged before IOA, so overlap cannot be double-counted.
    true.add_annotation(
        id=1, image_id=1, category_id=2,
        bbox=[10, 10, 16, 20], segmentation=_poly(10, 10, 26, 30))
    true.add_annotation(
        id=2, image_id=1, category_id=2,
        bbox=[20, 10, 16, 20], segmentation=_poly(20, 10, 36, 30))
    pred.add_annotation(
        id=11, image_id=1, category_id=1,
        bbox=[14, 14, 18, 12], segmentation=_poly(14, 14, 32, 26), score=.9)

    dmet = DetectionMetrics.from_coco(true, pred)
    ignored = dmet.confusion_vectors_v2(
        iou_thresh=0.5, iou_type='segm', compat='mutex',
        ignore_classes='ignore', spatial_threshold=None)
    kept = dmet.confusion_vectors_v2(
        iou_thresh=0.5, iou_type='segm', compat='mutex',
        ignore_classes=None, spatial_threshold=None)

    assert len(ignored.data['pxs']) == 0
    assert len(kept.data['pxs']) == 3  # one FP plus two unmatched ignore-class truths


def test_explicit_annotation_ignore_is_kwcoco_extension_not_coco_default():
    _require_native_mask_backend()
    import kwimage
    import numpy as np
    from kwcoco.metrics.assignment_v2 import AssignmentCandidateIndex

    dims = (32, 32)
    mask = np.zeros(dims, dtype=np.uint8)
    mask[4:20, 4:20] = 1
    box = kwimage.Boxes(np.array([[4, 4, 16, 16]], dtype=float), 'xywh')
    true = kwimage.Detections(
        boxes=box.copy(), class_idxs=np.array([0]),
        segmentations=[kwimage.Mask(mask, 'c_mask')])
    true.data['ignore'] = np.array([True])
    true.data['iscrowd'] = np.array([False])
    pred = kwimage.Detections(
        boxes=box.copy(), class_idxs=np.array([0]), scores=np.array([.9]),
        segmentations=[kwimage.Mask(mask, 'c_mask')])

    coco = AssignmentCandidateIndex.build(
        true, pred, iou_type='segm', image_dims=dims,
        annotation_ignore_policy='coco').assign(0.5, ignore_classes=None)
    native = AssignmentCandidateIndex.build(
        true, pred, iou_type='segm', image_dims=dims,
        annotation_ignore_policy='kwcoco').assign(0.5, ignore_classes=None)
    assert coco['weight'] == [1.0]
    assert native['weight'] == [0.0]


def test_segmentation_auto_defaults_are_coco_compatible():
    _require_native_mask_backend()
    import kwimage
    import numpy as np
    from kwcoco.metrics.assignment_v2 import AssignmentCandidateIndex

    dims = (16, 16)
    true_mask = np.zeros(dims, dtype=np.uint8)
    pred_mask = np.zeros(dims, dtype=np.uint8)
    true_mask[2:10, 2:10] = 1
    pred_mask[2:10, 2:6] = 1
    boxes = kwimage.Boxes(np.array([[0, 0, 16, 16]], dtype=float), 'xywh')
    true = kwimage.Detections(
        boxes=boxes, class_idxs=np.array([0]),
        segmentations=[kwimage.Mask(true_mask, 'c_mask')])
    pred = kwimage.Detections(
        boxes=boxes, class_idxs=np.array([0]), scores=np.array([.9]),
        segmentations=[kwimage.Mask(pred_mask, 'c_mask')])
    index = AssignmentCandidateIndex.build(
        true, pred, compat='mutex', min_iou=0.5,
        iou_type='segm', image_dims=dims, spatial_threshold=None)
    result = index.assign(0.5, ignore_classes=None)
    assert index.threshold_policy == 'inclusive'
    # Exact RLE input remains unchanged regardless of rasterization route.
    assert result['txs'] == [0]
