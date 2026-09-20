"""Phase-2.3 tests for native COCO-shaped instance-mask evaluation."""
import numpy as np
import pytest


pycocotools = pytest.importorskip('pycocotools')


def _rle(mask):
    from pycocotools import mask as mask_utils

    encoded = dict(mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))))
    if isinstance(encoded['counts'], bytes):
        encoded['counts'] = encoded['counts'].decode('ascii')
    encoded['size'] = list(map(int, encoded['size']))
    return encoded


def _bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if not len(xs):
        return [0, 0, 0, 0]
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return [x1, y1, x2 - x1, y2 - y1]


def _empty_pair(dims=(64, 64), categories=('thing',)):
    import kwcoco

    height, width = dims
    true = kwcoco.CocoDataset()
    pred = kwcoco.CocoDataset()
    for dset in [true, pred]:
        for cid, name in enumerate(categories, start=1):
            dset.add_category(name=name, id=cid)
        dset.add_image(
            id=1, file_name='im1.png', width=width, height=height
        )
    return true, pred


def _add_mask_ann(dset, mask, *, aid, cid=1, score=None, bbox=None, **kw):
    if bbox is None:
        bbox = _bbox_from_mask(mask)
    ann = {
        'id': aid,
        'image_id': 1,
        'category_id': cid,
        'bbox': bbox,
        'segmentation': _rle(mask),
        **kw,
    }
    if score is not None:
        ann['score'] = score
    dset.add_annotation(**ann)


def _build_area_priority_index(*, crowd=False):
    from kwcoco.metrics.assignment_v2 import AssignmentCandidateIndex
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (32, 32)
    true, pred = _empty_pair(dims=dims)

    pred_mask = np.zeros(dims, dtype=np.uint8)
    pred_mask[4:14, 4:14] = 1  # area 100

    in_range = np.zeros(dims, dtype=np.uint8)
    in_range[4:14, 7:17] = 1  # area 100, IoU ~= .538

    out_of_range = np.zeros(dims, dtype=np.uint8)
    out_of_range[4:16, 4:16] = 1  # area 144, IoU ~= .694 or crowd IOA=1

    _add_mask_ann(true, in_range, aid=1)
    _add_mask_ann(true, out_of_range, aid=2, iscrowd=int(crowd))
    _add_mask_ann(pred, pred_mask, aid=11, score=.99)

    dmet = DetectionMetrics.from_coco(true, pred)
    index = AssignmentCandidateIndex.build(
        dmet.true_detections(1),
        dmet.pred_detections(1),
        classes=dmet.classes,
        compat='mutex',
        min_iou=.5,
        iou_type='segm',
        overlap_backend='pycocotools',
        segmentation_rasterizer='coco',
        raster_backend='pycocotools',
        threshold_policy='inclusive',
        image_dims=dims,
    )
    return index


@pytest.mark.parametrize('crowd', [False, True])
def test_area_range_valid_truth_beats_higher_overlap_ignored_truth(crowd):
    """Out-of-range truth must not steal a detection from in-range truth."""
    index = _build_area_priority_index(crowd=crowd)

    # The adversarial condition is real: without area semantics, tx=1 is the
    # first candidate because it has the higher overlap.
    first_tx = int(index.cand_txs[0][0])
    assert index.true_areas[first_tx] == 144.0
    assert sorted(index.true_areas.tolist()) == [100.0, 144.0]

    rows = index.assign(
        .5,
        ignore_classes=None,
        area_range=(0, 110),
    )
    assert len(rows['txs']) == 1
    assert index.true_areas[rows['txs'][0]] == 100.0
    assert rows['pxs'] == [0]
    assert rows['weight'] == [1.0]


def test_area_range_ignored_match_and_unmatched_detection_have_zero_weight():
    from kwcoco.metrics.assignment_v2 import AssignmentCandidateIndex
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (32, 32)
    true, pred = _empty_pair(dims=dims)
    large = np.zeros(dims, dtype=np.uint8)
    large[4:16, 4:16] = 1  # area 144
    small = np.zeros(dims, dtype=np.uint8)
    small[5:15, 5:15] = 1  # area 100, matches large
    _add_mask_ann(true, large, aid=1)
    _add_mask_ann(pred, small, aid=11, score=.9)

    dmet = DetectionMetrics.from_coco(true, pred)
    index = AssignmentCandidateIndex.build(
        dmet.true_detections(1), dmet.pred_detections(1),
        classes=dmet.classes, compat='mutex', min_iou=.5,
        iou_type='segm', overlap_backend='pycocotools',
        segmentation_rasterizer='coco', raster_backend='pycocotools',
        threshold_policy='inclusive', image_dims=dims,
    )
    rows = index.assign(.5, ignore_classes=None, area_range=(0, 110))
    assert rows['txs'] == [0]
    assert rows['weight'] == [0.0]

    true2, pred2 = _empty_pair(dims=dims)
    _add_mask_ann(pred2, large, aid=11, score=.9)
    dmet2 = DetectionMetrics.from_coco(true2, pred2)
    index2 = AssignmentCandidateIndex.build(
        dmet2.true_detections(1), dmet2.pred_detections(1),
        classes=dmet2.classes, compat='mutex', min_iou=.5,
        iou_type='segm', overlap_backend='pycocotools',
        segmentation_rasterizer='coco', raster_backend='pycocotools',
        threshold_policy='inclusive', image_dims=dims,
    )
    rows2 = index2.assign(.5, ignore_classes=None, area_range=(0, 110))
    assert rows2['txs'] == [-1]
    assert rows2['weight'] == [0.0]


def test_native_coco_uses_mask_area_not_bbox_area():
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (128, 128)
    true, pred = _empty_pair(dims=dims)
    mask = np.zeros(dims, dtype=np.uint8)
    mask[10:26, 10:26] = 1  # mask area 256 => small
    deliberately_large_bbox = [0, 0, 100, 100]
    _add_mask_ann(
        true, mask, aid=1, bbox=deliberately_large_bbox
    )
    _add_mask_ann(
        pred, mask, aid=11, score=.9, bbox=deliberately_large_bbox
    )

    dmet = DetectionMetrics.from_coco(true, pred)
    native = dmet.score_native_coco(
        overlap_backend='pycocotools', raster_backend='pycocotools'
    )
    official = dmet.score_pycocotools(iou_type='segm')

    assert native['summary']['AP_small'] > .99
    assert official['summary']['AP_small'] > .99
    assert native['summary']['AP_medium'] == -1.0
    assert native['params']['segmentation_rasterizer'] == 'coco'


@pytest.mark.parametrize(
    'side, expected_labels',
    [
        (32, ('small', 'medium')),
        (96, ('medium', 'large')),
    ],
)
def test_native_coco_area_boundaries_are_inclusive(side, expected_labels):
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (128, 128)
    true, pred = _empty_pair(dims=dims)
    mask = np.zeros(dims, dtype=np.uint8)
    mask[8:8 + side, 8:8 + side] = 1
    _add_mask_ann(true, mask, aid=1)
    _add_mask_ann(pred, mask, aid=11, score=.9)

    dmet = DetectionMetrics.from_coco(true, pred)
    native = dmet.score_native_coco(
        overlap_backend='pycocotools', raster_backend='pycocotools'
    )
    for label in expected_labels:
        assert native['summary'][f'AP_{label}'] > .99


def test_max_dets_category_scope_matches_coco_usecats_shape():
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (64, 64)
    true, pred = _empty_pair(dims=dims, categories=('a', 'b'))
    mask_a = np.zeros(dims, dtype=np.uint8)
    mask_b = np.zeros(dims, dtype=np.uint8)
    mask_a[4:16, 4:16] = 1
    mask_b[32:44, 32:44] = 1
    _add_mask_ann(true, mask_a, aid=1, cid=1)
    _add_mask_ann(true, mask_b, aid=2, cid=2)
    _add_mask_ann(pred, mask_a, aid=11, cid=1, score=.99)
    _add_mask_ann(pred, mask_b, aid=12, cid=2, score=.98)

    dmet = DetectionMetrics.from_coco(true, pred)
    image_cap = dmet.confusion_vectors_v2(
        iou_thresh=.5, iou_type='segm',
        overlap_backend='pycocotools', raster_backend='pycocotools',
        segmentation_rasterizer='coco', ignore_classes=None,
        max_dets=1, max_dets_scope='image',
    )
    category_cap = dmet.confusion_vectors_v2(
        iou_thresh=.5, iou_type='segm',
        overlap_backend='pycocotools', raster_backend='pycocotools',
        segmentation_rasterizer='coco', ignore_classes=None,
        max_dets=1, max_dets_scope='category',
    )
    assert int((image_cap.data['pxs'] >= 0).sum()) == 1
    assert int((category_cap.data['pxs'] >= 0).sum()) == 2

    native = dmet.score_native_coco(
        overlap_backend='pycocotools', raster_backend='pycocotools'
    )
    assert native['summary']['AR1'] > .99
    assert native['params']['max_dets_scope'] == 'image-category'


def test_native_coco_max_dets_is_score_ranked():
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (64, 64)
    true, pred = _empty_pair(dims=dims)
    truth = np.zeros(dims, dtype=np.uint8)
    false = np.zeros(dims, dtype=np.uint8)
    truth[8:24, 8:24] = 1
    false[36:52, 36:52] = 1
    _add_mask_ann(true, truth, aid=1)
    _add_mask_ann(pred, false, aid=11, score=.99)
    _add_mask_ann(pred, truth, aid=12, score=.90)

    dmet = DetectionMetrics.from_coco(true, pred)
    native = dmet.score_native_coco(
        overlap_backend='pycocotools', raster_backend='pycocotools'
    )
    assert native['summary']['AR1'] == 0.0
    assert native['summary']['AR10'] > .99


def test_native_coco_crowd_and_max_dets_interact_in_score_order():
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (96, 96)
    true, pred = _empty_pair(dims=dims)
    ordinary = np.zeros(dims, dtype=np.uint8)
    crowd = np.zeros(dims, dtype=np.uint8)
    ordinary[8:24, 8:24] = 1
    crowd[48:80, 48:80] = 1
    crowd_hit = np.zeros(dims, dtype=np.uint8)
    crowd_hit[52:68, 52:68] = 1

    _add_mask_ann(true, ordinary, aid=1)
    _add_mask_ann(true, crowd, aid=2, iscrowd=1)
    _add_mask_ann(pred, crowd_hit, aid=11, score=.99)
    _add_mask_ann(pred, ordinary, aid=12, score=.90)

    dmet = DetectionMetrics.from_coco(true, pred)
    native = dmet.score_native_coco(
        overlap_backend='pycocotools', raster_backend='pycocotools'
    )
    official = dmet.score_pycocotools(iou_type='segm')
    assert native['summary']['AR1'] == 0.0
    assert native['summary']['AR10'] > .99
    assert native['summary']['AR1'] == pytest.approx(
        official['summary']['AR1'], abs=1e-9
    )
    assert native['summary']['AR10'] == pytest.approx(
        official['summary']['AR10'], abs=1e-9
    )


def test_native_coco_empty_prediction_and_truth_cases():
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (64, 64)
    mask = np.zeros(dims, dtype=np.uint8)
    mask[8:24, 8:24] = 1

    true, pred = _empty_pair(dims=dims)
    _add_mask_ann(true, mask, aid=1)
    dmet = DetectionMetrics.from_coco(true, pred)
    native = dmet.score_native_coco(
        overlap_backend='pycocotools', raster_backend='pycocotools'
    )
    assert native['summary']['AR100'] == 0.0

    true2, pred2 = _empty_pair(dims=dims)
    _add_mask_ann(pred2, mask, aid=11, score=.9)
    dmet2 = DetectionMetrics.from_coco(true2, pred2)
    native2 = dmet2.score_native_coco(
        overlap_backend='pycocotools', raster_backend='pycocotools'
    )
    assert native2['summary']['AP'] == -1.0
    assert native2['summary']['AR100'] == -1.0


def test_native_coco_negative_images_contribute_false_positives():
    import kwcoco
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (64, 64)
    true = kwcoco.CocoDataset()
    pred = kwcoco.CocoDataset()
    for dset in [true, pred]:
        dset.add_category(name='thing', id=1)
        for gid in [1, 2]:
            dset.add_image(
                id=gid, file_name=f'im{gid}.png', width=64, height=64
            )

    mask1 = np.zeros(dims, dtype=np.uint8)
    mask1[8:24, 8:24] = 1
    mask2 = np.zeros(dims, dtype=np.uint8)
    mask2[32:48, 32:48] = 1
    true.add_annotation(
        id=1, image_id=1, category_id=1, bbox=_bbox_from_mask(mask1),
        segmentation=_rle(mask1)
    )
    pred.add_annotation(
        id=11, image_id=1, category_id=1, bbox=_bbox_from_mask(mask1),
        segmentation=_rle(mask1), score=.90
    )
    # Higher-scored false positive on an image with no truth.
    pred.add_annotation(
        id=12, image_id=2, category_id=1, bbox=_bbox_from_mask(mask2),
        segmentation=_rle(mask2), score=.99
    )

    dmet = DetectionMetrics.from_coco(true, pred)
    native = dmet.score_native_coco(
        overlap_backend='pycocotools', raster_backend='pycocotools'
    )
    assert 0 < native['summary']['AP50'] < 1


def test_native_coco_reuses_one_candidate_index_per_image(monkeypatch):
    import kwcoco
    from kwcoco.metrics.assignment_v2 import AssignmentCandidateIndex
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (64, 64)
    true = kwcoco.CocoDataset()
    pred = kwcoco.CocoDataset()
    for dset in [true, pred]:
        dset.add_category(name='thing', id=1)
        for gid in [1, 2]:
            dset.add_image(
                id=gid, file_name=f'im{gid}.png', width=64, height=64
            )
    for gid, offset in [(1, 4), (2, 32)]:
        mask = np.zeros(dims, dtype=np.uint8)
        mask[offset:offset + 12, offset:offset + 12] = 1
        bbox = _bbox_from_mask(mask)
        true.add_annotation(
            id=gid, image_id=gid, category_id=1, bbox=bbox,
            segmentation=_rle(mask)
        )
        pred.add_annotation(
            id=10 + gid, image_id=gid, category_id=1, bbox=bbox,
            segmentation=_rle(mask), score=.9
        )

    original_build = AssignmentCandidateIndex.build
    calls = []

    def counted_build(*args, **kwargs):
        calls.append(1)
        return original_build(*args, **kwargs)

    monkeypatch.setattr(AssignmentCandidateIndex, 'build', counted_build)
    dmet = DetectionMetrics.from_coco(true, pred)
    info = dmet.score_native_coco(
        overlap_backend='pycocotools', raster_backend='pycocotools'
    )
    assert len(calls) == 2
    assert info['timing']['candidate_index_builds'] == 2


def test_native_coco_perfect_summary_and_evaluator_surface():
    from kwcoco.coco_evaluator import CocoEvaluator
    from kwcoco.metrics.coco_metrics import COCO_DETECTION_STAT_KEYS
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (64, 64)
    true, pred = _empty_pair(dims=dims)
    mask = np.zeros(dims, dtype=np.uint8)
    mask[8:24, 8:24] = 1
    _add_mask_ann(true, mask, aid=1)
    _add_mask_ann(pred, mask, aid=11, score=.9)

    dmet = DetectionMetrics.from_coco(true, pred)
    native = dmet.score_native_coco(
        overlap_backend='pycocotools', raster_backend='pycocotools'
    )
    official = dmet.score_pycocotools(iou_type='segm')
    assert tuple(native['summary']) == COCO_DETECTION_STAT_KEYS
    assert native['backend'] == 'kwcoco'
    assert native['assignment'] == 'assignment_v2'
    assert native['summary']['AP'] > .99
    assert native['summary']['AR100'] > .99
    assert native['summary']['AP'] == pytest.approx(
        official['summary']['AP'], abs=1e-9
    )

    evaluator = CocoEvaluator({
        'true_dataset': true,
        'pred_dataset': pred,
    })
    wrapped = evaluator.evaluate_native_coco(
        'segm', overlap_backend='pycocotools', raster_backend='pycocotools'
    )
    assert wrapped['segm']['backend'] == 'kwcoco'
    assert wrapped['segm']['summary']['AP50'] > .99
    with pytest.raises(KeyError):
        evaluator.evaluate_native_coco('bbox')
