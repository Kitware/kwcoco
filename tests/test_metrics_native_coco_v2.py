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


def test_native_coco_score_ties_preserve_coco_stable_order():
    """COCO uses stable descending score order during accumulation."""
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

    mask = np.zeros(dims, dtype=np.uint8)
    mask[8:24, 8:24] = 1
    false_mask = np.zeros(dims, dtype=np.uint8)
    false_mask[36:52, 36:52] = 1
    true.add_annotation(
        id=1, image_id=1, category_id=1,
        bbox=_bbox_from_mask(mask), segmentation=_rle(mask)
    )
    # Equal scores are intentional.  With gids traversed in sorted order,
    # COCO accumulation sees the TP first and must preserve that tie order.
    pred.add_annotation(
        id=11, image_id=1, category_id=1,
        bbox=_bbox_from_mask(mask), segmentation=_rle(mask), score=.5
    )
    pred.add_annotation(
        id=12, image_id=2, category_id=1,
        bbox=_bbox_from_mask(false_mask), segmentation=_rle(false_mask),
        score=.5
    )

    dmet = DetectionMetrics.from_coco(true, pred)
    native = dmet.score_native_coco(
        overlap_backend='pycocotools', raster_backend='pycocotools'
    )
    official = dmet.score_pycocotools(iou_type='segm')
    assert native['summary']['AP50'] > .99
    assert native['summary']['AP50'] == pytest.approx(
        official['summary']['AP50'], abs=1e-12
    )
    assert native['summary']['AP'] == pytest.approx(
        official['summary']['AP'], abs=1e-12
    )


def test_native_coco_maxdets_100_truncates_before_matching():
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (64, 64)
    true, pred = _empty_pair(dims=dims)
    truth = np.zeros(dims, dtype=np.uint8)
    false = np.zeros(dims, dtype=np.uint8)
    truth[4:16, 4:16] = 1
    false[40:52, 40:52] = 1
    _add_mask_ann(true, truth, aid=1)
    for idx in range(100):
        _add_mask_ann(
            pred, false, aid=1000 + idx, score=1.0 - idx * .001
        )
    _add_mask_ann(pred, truth, aid=2000, score=.5)

    dmet = DetectionMetrics.from_coco(true, pred)
    native = dmet.score_native_coco(
        overlap_backend='pycocotools', raster_backend='pycocotools'
    )
    official = dmet.score_pycocotools(iou_type='segm')
    assert native['summary']['AR100'] == 0.0
    assert native['summary']['AR100'] == pytest.approx(
        official['summary']['AR100'], abs=1e-12
    )
    assert native['summary']['AP'] == pytest.approx(
        official['summary']['AP'], abs=1e-12
    )


def test_native_coco_ignores_kwcoco_truth_weights_for_coco_compatibility():
    """Arbitrary kwcoco truth weights are not part of COCOeval semantics."""
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

    mask = np.zeros(dims, dtype=np.uint8)
    mask[8:24, 8:24] = 1
    for gid, weight in [(1, 0.01), (2, 100.0)]:
        true.add_annotation(
            id=gid, image_id=gid, category_id=1,
            bbox=_bbox_from_mask(mask), segmentation=_rle(mask), weight=weight
        )
    pred.add_annotation(
        id=11, image_id=1, category_id=1,
        bbox=_bbox_from_mask(mask), segmentation=_rle(mask), score=.9
    )

    dmet = DetectionMetrics.from_coco(true, pred)
    # Force the kwcoco-only extension onto the metric inputs. This keeps the
    # regression independent of whether kwimage preserves arbitrary COCO
    # annotation fields while coercing detections.
    dmet.gid_to_true_dets[1].data['weights'] = np.asarray([0.01])
    dmet.gid_to_true_dets[2].data['weights'] = np.asarray([100.0])
    native = dmet.score_native_coco(
        overlap_backend='pycocotools', raster_backend='pycocotools'
    )
    official = dmet.score_pycocotools(iou_type='segm')
    assert native['params']['truth_weight_policy'] == 'coco-unweighted'
    assert native['summary']['AP'] == pytest.approx(
        official['summary']['AP'], abs=1e-12
    )
    assert native['summary']['AR100'] == pytest.approx(
        official['summary']['AR100'], abs=1e-12
    )


def test_native_coco_scores_category_literally_named_background():
    """COCO category names are labels; ``background`` is not magic here."""
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (64, 64)
    true, pred = _empty_pair(dims=dims, categories=('background',))
    mask = np.zeros(dims, dtype=np.uint8)
    mask[8:24, 8:24] = 1
    _add_mask_ann(true, mask, aid=1)
    _add_mask_ann(pred, mask, aid=11, score=.9)

    dmet = DetectionMetrics.from_coco(true, pred)
    native = dmet.score_native_coco(
        overlap_backend='pycocotools', raster_backend='pycocotools'
    )
    official = dmet.score_pycocotools(iou_type='segm')
    assert native['summary']['AP'] > .99
    assert native['summary']['AP'] == pytest.approx(
        official['summary']['AP'], abs=1e-12
    )


def test_coco_evaluator_distinguishes_synthetic_and_declared_background():
    """Strict COCO axes come from declared truth categories, not bg hacks."""
    from kwcoco.coco_evaluator import CocoEvaluator

    dims = (64, 64)
    true, pred = _empty_pair(dims=dims, categories=('background',))
    mask = np.zeros(dims, dtype=np.uint8)
    mask[8:24, 8:24] = 1
    _add_mask_ann(true, mask, aid=1)
    _add_mask_ann(pred, mask, aid=11, score=.9)

    evaluator = CocoEvaluator({
        'true_dataset': true,
        'pred_dataset': pred,
    })
    native = evaluator.evaluate_native_coco(
        ('bbox', 'segm'), with_eval=True,
        overlap_backend='pycocotools', raster_backend='pycocotools',
    )
    official = evaluator.evaluate_pycocotools(
        ('bbox', 'segm'), with_eval=True,
    )
    for iou_type in ['bbox', 'segm']:
        assert native[iou_type]['eval']['counts'][2] == 1
        assert official[iou_type]['eval']['counts'][2] == 1
        assert native[iou_type]['eval']['category_names'] == ['background']
        _assert_eval_tensor_parity(native[iou_type], official[iou_type])


@pytest.mark.parametrize('empty_side', ['truth', 'prediction'])
def test_native_coco_bbox_empty_side_matches_official(empty_side):
    """Empty per-image bbox inputs are valid COCO evaluation inputs."""
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (64, 64)
    true, pred = _empty_pair(dims=dims)
    mask = np.zeros(dims, dtype=np.uint8)
    mask[8:24, 8:24] = 1
    if empty_side != 'truth':
        _add_mask_ann(true, mask, aid=1)
    if empty_side != 'prediction':
        _add_mask_ann(pred, mask, aid=11, score=.9)

    dmet = DetectionMetrics.from_coco(true, pred)
    native = dmet.score_native_coco(iou_type='bbox', with_eval=True)
    official = dmet.score_pycocotools(
        iou_type='bbox', bbox_annotation_policy='coco', with_eval=True)
    _assert_eval_tensor_parity(native, official)


def test_native_coco_auto_backend_falls_back_to_pycocotools(monkeypatch):
    """``auto`` must remain usable when the optional Rust backend is absent."""
    import sys
    from kwcoco.metrics.overlap import resolve_coco_raster_backend
    from kwcoco.metrics.overlap import resolve_mask_iou_backend

    for key in list(sys.modules):
        if key == 'kwimage_ext' or key.startswith('kwimage_ext.'):
            monkeypatch.setitem(sys.modules, key, None)
    monkeypatch.setitem(sys.modules, 'kwimage_ext', None)

    _iou_func, overlap_meta = resolve_mask_iou_backend('auto')
    _codec, raster_meta = resolve_coco_raster_backend('auto')
    assert overlap_meta['name'] == 'pycocotools'
    assert raster_meta['name'] == 'pycocotools'


def test_native_coco_auto_backend_error_names_install_choices(monkeypatch):
    import sys
    from kwcoco.metrics.overlap import resolve_coco_raster_backend
    from kwcoco.metrics.overlap import resolve_mask_iou_backend

    for prefix in ['kwimage_ext', 'pycocotools']:
        for key in list(sys.modules):
            if key == prefix or key.startswith(prefix + '.'):
                monkeypatch.setitem(sys.modules, key, None)
        monkeypatch.setitem(sys.modules, prefix, None)

    for resolver in [resolve_mask_iou_backend, resolve_coco_raster_backend]:
        with pytest.raises(ImportError) as exc:
            resolver('auto')
        message = str(exc.value)
        assert 'kwimage_ext' in message
        assert 'pycocotools' in message


def test_native_coco_auto_backend_reports_actual_provenance():
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (64, 64)
    true, pred = _empty_pair(dims=dims)
    mask = np.zeros(dims, dtype=np.uint8)
    mask[8:24, 8:24] = 1
    _add_mask_ann(true, mask, aid=1)
    _add_mask_ann(pred, mask, aid=11, score=.9)

    dmet = DetectionMetrics.from_coco(true, pred)
    native = dmet.score_native_coco(
        overlap_backend='auto', raster_backend='auto'
    )
    provenance = native['provenance']
    assert provenance['requested_overlap_backend'] == 'auto'
    assert provenance['requested_raster_backend'] == 'auto'
    assert len(provenance['overlap_backends']) == 1
    assert len(provenance['raster_backends']) == 1
    assert len(provenance['area_backends']) == 1
    assert provenance['overlap_backends'][0] in {
        'kwimage_ext', 'pycocotools'
    }
    assert provenance['raster_backends'][0] in {
        'kwimage_ext', 'pycocotools'
    }
    assert provenance['area_backends'][0] in {
        'kwimage_ext', 'pycocotools'
    }



def test_native_coco_bbox_provenance_uses_coco_xywh_area():
    """Strict COCO bbox provenance names the XYWH area implementation."""
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (64, 64)
    true, pred = _empty_pair(dims=dims)
    mask = np.zeros(dims, dtype=np.uint8)
    mask[8:24, 8:24] = 1
    _add_mask_ann(true, mask, aid=1)
    _add_mask_ann(pred, mask, aid=11, score=.9)

    dmet = DetectionMetrics.from_coco(true, pred)
    native = dmet.score_native_coco(iou_type='bbox')
    provenance = native['provenance']
    assert provenance['overlap_backends'] == ['numpy_coco_bbox']
    assert provenance['area_backends'] == ['numpy_coco_xywh_area']


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
    assert native['result_schema'] == official['result_schema']
    assert official['official'] is True
    assert official['implementation'] == 'pycocotools.COCOeval'
    assert native['backend'] == 'kwcoco'
    assert native['implementation'] == 'kwcoco.native_coco'
    assert native['result_schema'] == 'coco_detection_summary_v1'
    assert native['official'] is False
    assert native['reference_implementation'] == 'pycocotools.COCOeval'
    assert native['assignment'] == 'assignment_v2'
    assert native['accumulator'] == 'compact_matches->coco_tensor'
    assert len(native['params']['recall_thresholds']) == 101
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
        'segm', overlap_backend='pycocotools', raster_backend='pycocotools',
        with_details=True,
    )
    assert wrapped['segm']['backend'] == 'kwcoco'
    assert wrapped['segm']['summary']['AP50'] > .99
    detail = wrapped['segm']['details']['all:maxDets=100']['0.5']
    assert detail['perclass']['thing']['ap'] > .99
    assert detail['perclass']['thing']['realpos_total'] == 1
    bbox_wrapped = evaluator.evaluate_native_coco('bbox', with_eval=True)
    assert bbox_wrapped['bbox']['summary']['AP50'] > .99
    assert bbox_wrapped['bbox']['params']['bbox_overlap_mode'] == 'coco'
    assert bbox_wrapped['bbox']['eval']['counts'] == [10, 101, 1, 4, 3]


def _assert_eval_tensor_parity(native, official, atol=1e-12):
    native_eval = native['eval']
    official_eval = official['eval']
    assert native_eval['schema'] == official_eval['schema']
    assert native_eval['counts'] == official_eval['counts']
    for key in ['category_ids', 'image_ids', 'area_range_labels', 'max_dets']:
        assert native_eval[key] == official_eval[key]
    for key in ['iou_thresholds', 'recall_thresholds', 'area_ranges']:
        assert np.array_equal(
            np.asarray(native_eval[key]), np.asarray(official_eval[key])
        )
    shared_param_keys = [
        'iou_thresholds', 'recall_thresholds', 'max_dets',
        'area_ranges', 'area_range_labels', 'area_range_map',
        'image_ids', 'category_ids',
    ]
    for key in shared_param_keys:
        assert native['params'][key] == official['params'][key]
    for key in ['precision', 'recall', 'scores']:
        n = np.asarray(native_eval[key])
        o = np.asarray(official_eval[key])
        assert n.shape == o.shape
        if key == 'scores':
            n_absent = np.asarray(native_eval['precision']) < 0
            o_absent = np.asarray(official_eval['precision']) < 0
        else:
            n_absent = n < 0
            o_absent = o < 0
        assert np.array_equal(n_absent, o_absent)
        valid = ~(n_absent | o_absent)
        assert np.allclose(n[valid], o[valid], atol=atol, rtol=0)


@pytest.mark.parametrize('iou_type', ['bbox', 'segm'])
def test_native_coco_full_eval_tensor_matches_official(iou_type):
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (64, 64)
    true, pred = _empty_pair(dims=dims, categories=('a', 'b'))
    mask_a = np.zeros(dims, dtype=np.uint8)
    mask_b = np.zeros(dims, dtype=np.uint8)
    mask_a[4:20, 4:20] = 1
    mask_b[32:48, 32:48] = 1
    _add_mask_ann(true, mask_a, aid=1, cid=1)
    _add_mask_ann(true, mask_b, aid=2, cid=2)
    _add_mask_ann(pred, mask_a, aid=11, cid=1, score=.9)
    # Keep a false positive ahead of the second TP so precision and score
    # sampling are nontrivial rather than all ones.
    false = np.zeros(dims, dtype=np.uint8)
    false[4:20, 40:56] = 1
    _add_mask_ann(pred, false, aid=12, cid=2, score=.85)
    _add_mask_ann(pred, mask_b, aid=13, cid=2, score=.8)

    dmet = DetectionMetrics.from_coco(true, pred)
    native = dmet.score_native_coco(
        iou_type=iou_type,
        overlap_backend='pycocotools',
        raster_backend='pycocotools',
        with_eval=True,
    )
    official = dmet.score_pycocotools(
        iou_type=iou_type,
        bbox_annotation_policy='coco',
        with_eval=True,
    )
    _assert_eval_tensor_parity(native, official)
    assert native['summary'] == pytest.approx(official['summary'], abs=1e-12)


def test_native_coco_bbox_crowd_uses_detection_area_overlap():
    """BBox crowd overlap is IoA, not ordinary union IoU."""
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (128, 128)
    true, pred = _empty_pair(dims=dims)
    crowd = np.zeros(dims, dtype=np.uint8)
    crowd[8:104, 8:104] = 1
    inside = np.zeros(dims, dtype=np.uint8)
    inside[32:44, 32:44] = 1
    _add_mask_ann(true, crowd, aid=1, iscrowd=1)
    _add_mask_ann(pred, inside, aid=11, score=.99)

    dmet = DetectionMetrics.from_coco(true, pred)
    native = dmet.score_native_coco(iou_type='bbox', with_eval=True)
    official = dmet.score_pycocotools(
        iou_type='bbox', bbox_annotation_policy='coco', with_eval=True)
    _assert_eval_tensor_parity(native, official)
    assert native['summary']['AP'] == -1.0
    assert native['summary']['AR100'] == -1.0
    assert native['provenance']['overlap_backends'] == ['numpy_coco_bbox']


def test_official_bbox_historical_and_strict_crowd_policies_are_separate():
    """The direct compatibility mode must not silently become strict COCO."""
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (128, 128)
    true, pred = _empty_pair(dims=dims)
    crowd = np.zeros(dims, dtype=np.uint8)
    crowd[8:104, 8:104] = 1
    inside = np.zeros(dims, dtype=np.uint8)
    inside[32:44, 32:44] = 1
    _add_mask_ann(true, crowd, aid=1, iscrowd=1)
    _add_mask_ann(pred, inside, aid=11, score=.99)

    dmet = DetectionMetrics.from_coco(true, pred)
    strict = dmet.score_pycocotools(
        iou_type='bbox', bbox_annotation_policy='coco', with_eval=True)
    historical = dmet.score_pycocotools(
        iou_type='bbox', bbox_annotation_policy='historical', with_eval=True)

    # Strict COCO treats the only truth as crowd/ignored. The historical
    # kwcoco direct wrapper intentionally flattened bbox crowd to an ordinary
    # truth, preserving its long-standing compatibility behavior.
    assert strict['summary']['AP'] == -1.0
    assert strict['summary']['AR100'] == -1.0
    assert historical['summary']['AP'] == pytest.approx(0.0)
    assert historical['summary']['AR100'] == pytest.approx(0.0)


def test_native_coco_truth_equal_iou_prefers_later_truth_like_cocoeval():
    """COCO's >= scan means a later truth wins an equal-IoU tie."""
    from kwcoco.metrics.detect_metrics import DetectionMetrics
    dims = (64, 64)
    true, pred = _empty_pair(dims=dims)
    mask = np.zeros(dims, dtype=np.uint8)
    mask[8:24, 8:24] = 1
    # Two identical truths make IoU exactly equal; truth index 1 is later.
    _add_mask_ann(true, mask, aid=1)
    _add_mask_ann(true, mask, aid=2)
    _add_mask_ann(pred, mask, aid=11, score=.9)
    dmet = DetectionMetrics.from_coco(true, pred)

    for iou_type in ['bbox', 'segm']:
        row_kw = dict(
            iou_thresh=.5,
            iou_type=iou_type,
            compat='mutex',
            ignore_classes=None,
            threshold_policy='inclusive',
            annotation_ignore_policy='coco',
            max_dets=100,
            max_dets_scope='category',
            area_range=(0.0, 1e10),
        )
        if iou_type == 'bbox':
            row_kw['bbox_overlap_mode'] = 'coco'
        else:
            row_kw.update({
                'overlap_backend': 'pycocotools',
                'segmentation_rasterizer': 'coco',
                'raster_backend': 'pycocotools',
            })
        native_rows = dmet.confusion_vectors_v2(**row_kw)
        official_rows = dmet.score_pycocotools(
            iou_type=iou_type,
            bbox_annotation_policy='coco',
            iou_thresholds=[.5],
            with_confusion=True,
        )['iou_to_cfsn_vecs'][.5]
        native_txs = np.asarray(native_rows.data['txs'])
        native_pxs = np.asarray(native_rows.data['pxs'])
        official_txs = np.asarray(official_rows.data['txs'])
        official_pxs = np.asarray(official_rows.data['pxs'])
        native_pairs = sorted(zip(
            native_pxs[(native_pxs >= 0) & (native_txs >= 0)].tolist(),
            native_txs[(native_pxs >= 0) & (native_txs >= 0)].tolist(),
        ))
        official_pairs = sorted(zip(
            official_pxs[(official_pxs >= 0) & (official_txs >= 0)].tolist(),
            official_txs[(official_pxs >= 0) & (official_txs >= 0)].tolist(),
        ))
        assert native_pairs == official_pairs == [(0, 1)]


def test_native_coco_unsorted_gid_subset_matches_coco_normalization():
    import kwcoco
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (64, 64)
    true = kwcoco.CocoDataset()
    pred = kwcoco.CocoDataset()
    for dset in [true, pred]:
        dset.add_category(name='thing', id=1)
        for gid in [1, 2]:
            dset.add_image(id=gid, file_name=f'im{gid}.png', width=64, height=64)
    mask = np.zeros(dims, dtype=np.uint8)
    mask[8:24, 8:24] = 1
    true.add_annotation(
        id=1, image_id=1, category_id=1,
        bbox=_bbox_from_mask(mask), segmentation=_rle(mask))
    pred.add_annotation(
        id=11, image_id=1, category_id=1,
        bbox=_bbox_from_mask(mask), segmentation=_rle(mask), score=.5)
    pred.add_annotation(
        id=12, image_id=2, category_id=1,
        bbox=[36, 36, 16, 16], segmentation=_rle(np.pad(
            np.ones((16, 16), dtype=np.uint8), ((36, 12), (36, 12)))),
        score=.5)

    dmet = DetectionMetrics.from_coco(true, pred)
    native = dmet.score_native_coco(
        iou_type='segm', gids=[2, 1, 2],
        overlap_backend='pycocotools', raster_backend='pycocotools',
        with_eval=True)
    # COCOeval internally unique-sorts image ids. Native normalization should
    # therefore be equivalent to the ordinary all-images reference.
    official = dmet.score_pycocotools(iou_type='segm', with_eval=True)
    _assert_eval_tensor_parity(native, official)


@pytest.mark.parametrize('iou_type', ['bbox', 'segm'])
def test_native_coco_strict_policy_ignores_kwcoco_ignore_extension(iou_type):
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (64, 64)
    true, pred = _empty_pair(dims=dims)
    mask = np.zeros(dims, dtype=np.uint8)
    mask[8:28, 8:28] = 1
    _add_mask_ann(true, mask, aid=1, ignore=1)
    _add_mask_ann(pred, mask, aid=11, score=.9)
    dmet = DetectionMetrics.from_coco(true, pred)

    native = dmet.score_native_coco(
        iou_type=iou_type, annotation_ignore_policy='coco',
        overlap_backend='pycocotools', raster_backend='pycocotools',
        with_eval=True)
    official = dmet.score_pycocotools(
        iou_type=iou_type, bbox_annotation_policy='coco', with_eval=True)
    _assert_eval_tensor_parity(native, official)
    assert native['summary']['AP'] == pytest.approx(1.0)


@pytest.mark.parametrize('iou_type', ['bbox', 'segm'])
def test_native_coco_negative_scores_tensor_matches_official(iou_type):
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (64, 64)
    true, pred = _empty_pair(dims=dims)
    truth = np.zeros(dims, dtype=np.uint8)
    truth[8:28, 8:28] = 1
    false = np.zeros(dims, dtype=np.uint8)
    false[36:52, 36:52] = 1
    _add_mask_ann(true, truth, aid=1)
    _add_mask_ann(pred, false, aid=11, score=-0.1)
    _add_mask_ann(pred, truth, aid=12, score=-0.2)

    dmet = DetectionMetrics.from_coco(true, pred)
    native = dmet.score_native_coco(
        iou_type=iou_type, overlap_backend='pycocotools',
        raster_backend='pycocotools', with_eval=True)
    official = dmet.score_pycocotools(
        iou_type=iou_type, bbox_annotation_policy='coco', with_eval=True)
    _assert_eval_tensor_parity(native, official)
    valid_scores = np.asarray(native['eval']['scores'])[
        np.asarray(native['eval']['precision']) >= 0]
    assert np.any(valid_scores < 0)


@pytest.mark.parametrize('iou_type', ['bbox', 'segm'])
def test_native_coco_custom_parameter_grid_matches_official(iou_type):
    """Tensor parity is not hard-coded to the standard 10x101x4x3 grid."""
    import kwcoco
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (128, 128)
    true = kwcoco.CocoDataset()
    pred = kwcoco.CocoDataset()
    for dset in [true, pred]:
        dset.add_category(name='a', id=1)
        dset.add_category(name='b', id=2)
        for gid in [1, 2, 3]:
            dset.add_image(
                id=gid, file_name=f'im{gid}.png', width=dims[1], height=dims[0])

    masks = []
    for x, y, side in [(4, 4, 12), (40, 8, 24), (12, 60, 40)]:
        mask = np.zeros(dims, dtype=np.uint8)
        mask[y:y + side, x:x + side] = 1
        masks.append(mask)
    for gid, mask in enumerate(masks, start=1):
        cid = 1 if gid != 2 else 2
        _add_mask_ann(true, mask, aid=gid, cid=cid, image_id=gid)
        _add_mask_ann(pred, mask, aid=10 + gid, cid=cid, image_id=gid,
                      score=.95 - gid * .05)
    # Add a higher-scoring FP so sampled precision / score thresholds are not
    # degenerate. It lives on an included image and category.
    false = np.zeros(dims, dtype=np.uint8)
    false[90:102, 90:102] = 1
    _add_mask_ann(pred, false, aid=20, cid=1, image_id=1, score=.99)

    dmet = DetectionMetrics.from_coco(true, pred)
    params = {
        'iou_thresholds': [.50, .625, .75, .90],
        'recall_thresholds': np.linspace(0, 1, 17),
        'area_ranges': {
            'all': (0.0, 1e10),
            'small': (0.0, float(32 ** 2)),
            'medium': (float(32 ** 2), float(96 ** 2)),
            'large': (float(96 ** 2), 1e10),
            'tiny': (0.0, float(16 ** 2)),
        },
        'max_dets': [1, 5, 10, 17, 100],
        # Repeated / unsorted ids must normalize like COCOeval.
        'gids': [3, 1, 3],
    }
    native = dmet.score_native_coco(
        iou_type=iou_type,
        overlap_backend='pycocotools',
        raster_backend='pycocotools',
        with_eval=True,
        **params,
    )
    official = dmet.score_pycocotools(
        iou_type=iou_type,
        bbox_annotation_policy='coco',
        with_eval=True,
        **params,
    )
    assert native['eval']['counts'] == [4, 17, 2, 5, 5]
    _assert_eval_tensor_parity(native, official)
    assert native['summary'] == pytest.approx(official['summary'], abs=1e-12)


@pytest.mark.parametrize('iou_type', ['bbox', 'segm'])
def test_coco_evaluator_custom_tensor_grid_matches_official(iou_type):
    """Dataset-level wrappers forward the same custom COCO parameter grid."""
    from kwcoco.coco_evaluator import CocoEvaluator

    dims = (64, 64)
    true, pred = _empty_pair(dims=dims)
    mask = np.zeros(dims, dtype=np.uint8)
    mask[8:28, 8:28] = 1
    _add_mask_ann(true, mask, aid=1)
    _add_mask_ann(pred, mask, aid=11, score=.9)

    evaluator = CocoEvaluator({
        'true_dataset': true,
        'pred_dataset': pred,
    })
    params = {
        'iou_thresholds': [.5, .7, .9],
        'recall_thresholds': [0.0, .25, .5, .75, 1.0],
        'area_ranges': {
            'all': (0.0, 1e10),
            'smallish': (0.0, 30 ** 2),
        },
        'max_dets': [1, 7],
        'gids': [1, 1],
        'with_eval': True,
    }
    native = evaluator.evaluate_native_coco(iou_type, **params)[iou_type]
    official = evaluator.evaluate_pycocotools(iou_type, **params)[iou_type]
    assert native['eval']['counts'] == [3, 5, 1, 2, 2]
    _assert_eval_tensor_parity(native, official)

def test_coco_bbox_profile_auto_threshold_is_inclusive():
    """Opting into the COCO bbox profile also opts into the COCO boundary."""
    import kwimage
    from kwcoco.metrics.assignment_v2 import AssignmentCandidateIndex

    true = kwimage.Detections(
        boxes=kwimage.Boxes([[0, 0, 12, 12]], 'xywh'),
        class_idxs=np.array([0]),
    )
    pred = kwimage.Detections(
        boxes=kwimage.Boxes([[4, 0, 12, 12]], 'xywh'),
        class_idxs=np.array([0]),
        scores=np.array([.9]),
    )
    common = dict(
        classes=['thing'], compat='mutex', iou_type='bbox', min_iou=.5,
    )
    coco = AssignmentCandidateIndex.build(
        true, pred, bbox_overlap_mode='coco', threshold_policy='auto', **common)
    legacy = AssignmentCandidateIndex.build(
        true, pred, bbox_overlap_mode='legacy', threshold_policy='auto', **common)
    assert coco.threshold_policy == 'inclusive'
    assert legacy.threshold_policy == 'strict'
    assert coco.assign(.5)['txs'][0] == 0
    assert legacy.assign(.5)['txs'][0] == -1


def test_coco_bbox_profile_rejects_pixel_bias():
    """The explicit COCO bbox profile cannot silently accept legacy bias."""
    import kwimage
    from kwcoco.metrics.assignment_v2 import AssignmentCandidateIndex

    true = kwimage.Detections(
        boxes=kwimage.Boxes([[0, 0, 10, 10]], 'xywh'),
        class_idxs=np.array([0]),
    )
    pred = true.copy()
    with pytest.raises(ValueError, match='requires bias=0'):
        AssignmentCandidateIndex.build(
            true, pred,
            classes=['thing'],
            compat='mutex',
            iou_type='bbox',
            bbox_overlap_mode='coco',
            bias=1,
        )


def test_native_coco_bbox_threshold_one_matches_cocoeval_cap():
    """COCOeval compares t=1.0 at min(t, 1 - 1e-10)."""
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    true, pred = _empty_pair(dims=(64, 64))
    true_box = [8.0, 8.0, 20.0, 20.0]
    # IoU is below exact one, but above COCO's 1 - 1e-10 comparison cap.
    pred_box = [8.0, 8.0, 20.0 - 1e-9, 20.0]
    true.add_annotation(
        id=1, image_id=1, category_id=1, bbox=true_box,
        segmentation=[[8, 8, 28, 8, 28, 28, 8, 28]])
    pred.add_annotation(
        id=11, image_id=1, category_id=1, bbox=pred_box,
        segmentation=[[8, 8, 28, 8, 28, 28, 8, 28]], score=.9)
    dmet = DetectionMetrics.from_coco(true, pred)
    params = {'iou_thresholds': [1.0]}
    native = dmet.score_native_coco(
        iou_type='bbox', with_eval=True, **params)
    official = dmet.score_pycocotools(
        iou_type='bbox', bbox_annotation_policy='coco', with_eval=True,
        **params)
    _assert_eval_tensor_parity(native, official)
    assert native['summary']['AP'] == pytest.approx(1.0)


def test_coco_bbox_sparse_candidate_path_matches_dense_path():
    """Spatial pruning must not change COCO crowd overlap or tie semantics."""
    from kwcoco.metrics.assignment_v2 import AssignmentCandidateIndex
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (128, 128)
    true, pred = _empty_pair(dims=dims)
    masks = []
    for x, y, side in [(4, 4, 20), (48, 8, 24), (70, 70, 40)]:
        mask = np.zeros(dims, dtype=np.uint8)
        mask[y:y + side, x:x + side] = 1
        masks.append(mask)
    _add_mask_ann(true, masks[0], aid=1)
    _add_mask_ann(true, masks[1], aid=2)
    _add_mask_ann(true, masks[2], aid=3, iscrowd=1)
    _add_mask_ann(pred, masks[0], aid=11, score=.95)
    _add_mask_ann(pred, masks[1], aid=12, score=.90)
    inside_crowd = np.zeros(dims, dtype=np.uint8)
    inside_crowd[80:90, 80:90] = 1
    _add_mask_ann(pred, inside_crowd, aid=13, score=.85)

    dmet = DetectionMetrics.from_coco(true, pred)
    common = dict(
        classes=dmet.classes,
        compat='mutex',
        iou_type='bbox',
        bbox_overlap_mode='coco',
        threshold_policy='inclusive',
        annotation_ignore_policy='coco',
        min_iou=.5,
    )
    dense = AssignmentCandidateIndex.build(
        dmet.true_detections(1), dmet.pred_detections(1),
        spatial_threshold=None, **common)
    sparse = AssignmentCandidateIndex.build(
        dmet.true_detections(1), dmet.pred_detections(1),
        spatial_threshold=1, **common)
    assert sparse.build_stats['spatial_blocks'] > 0
    for dense_txs, sparse_txs, dense_iou, sparse_iou in zip(
        dense.cand_txs, sparse.cand_txs, dense.cand_ious, sparse.cand_ious
    ):
        assert np.array_equal(dense_txs, sparse_txs)
        np.testing.assert_allclose(dense_iou, sparse_iou, atol=0, rtol=0)
    for threshold in [.5, .75, .95]:
        d = dense.assign(
            threshold, max_dets=100, max_dets_scope='category',
            area_range=(0.0, 1e10), ignore_classes=None)
        s = sparse.assign(
            threshold, max_dets=100, max_dets_scope='category',
            area_range=(0.0, 1e10), ignore_classes=None)
        assert d == s


def test_coco_bbox_fractional_origin_preserves_xywh_area_boundaries():
    """COCO bbox area must use authored XYWH width/height, not x2-x1."""
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    true, pred = _empty_pair(dims=(256, 256))
    bbox = [64.2, 64.4, 96.0, 96.0]
    poly = [[
        bbox[0], bbox[1],
        bbox[0] + bbox[2], bbox[1],
        bbox[0] + bbox[2], bbox[1] + bbox[3],
        bbox[0], bbox[1] + bbox[3],
    ]]
    true.add_annotation(
        id=1, image_id=1, category_id=1, bbox=bbox, segmentation=poly)
    pred.add_annotation(
        id=11, image_id=1, category_id=1, bbox=bbox,
        segmentation=poly, score=.9)

    dmet = DetectionMetrics.from_coco(true, pred)
    native = dmet.score_native_coco(iou_type='bbox', with_eval=True)
    official = dmet.score_pycocotools(
        iou_type='bbox', bbox_annotation_policy='coco', with_eval=True)
    _assert_eval_tensor_parity(native, official)
    assert native['summary']['AP_medium'] == pytest.approx(1.0)
    assert native['summary']['AP_large'] == pytest.approx(1.0)


def test_assignment_v2_truncate_max_dets_matches_direct_assignment():
    """Derived smaller maxDet prefixes must equal direct greedy matching."""
    import kwimage
    from kwcoco.metrics.assignment_v2 import AssignmentCandidateIndex

    true = kwimage.Detections(
        boxes=kwimage.Boxes([
            [0, 0, 10, 10],
            [20, 0, 10, 10],
            [40, 0, 10, 10],
        ], 'xywh'),
        class_idxs=np.array([0, 0, 0]),
    )
    pred = kwimage.Detections(
        boxes=kwimage.Boxes([
            [50, 50, 10, 10],  # highest-score FP
            [0, 0, 10, 10],    # TP only visible at maxDet >= 2
            [40, 0, 10, 10],   # TP only visible at maxDet >= 3
        ], 'xywh'),
        class_idxs=np.array([0, 0, 0]),
        scores=np.array([.99, .90, .80]),
    )
    index = AssignmentCandidateIndex.build(
        true, pred,
        classes=['thing'],
        compat='mutex',
        iou_type='bbox',
        bbox_overlap_mode='coco',
        threshold_policy='inclusive',
        annotation_ignore_policy='coco',
        min_iou=.5,
    )
    large = index.assign(
        .5, max_dets=3, max_dets_scope='category',
        area_range=(0.0, 1e10), ignore_classes=None)
    for max_det in [0, 1, 2, 3]:
        direct = index.assign(
            .5, max_dets=max_det, max_dets_scope='category',
            area_range=(0.0, 1e10), ignore_classes=None)
        if max_det == 3:
            derived = large
        else:
            derived = index.truncate_max_dets(
                large, max_det, max_dets_scope='category')
        assert derived == direct


def test_native_coco_reuses_largest_maxdet_assignment_prefix():
    """Full standard tensors should match only four area bases per threshold."""
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    true, pred = _empty_pair(dims=(64, 64))
    bbox = [8.0, 8.0, 16.0, 16.0]
    poly = [[8, 8, 24, 8, 24, 24, 8, 24]]
    true.add_annotation(
        id=1, image_id=1, category_id=1, bbox=bbox, segmentation=poly)
    pred.add_annotation(
        id=11, image_id=1, category_id=1, bbox=bbox,
        segmentation=poly, score=.9)

    dmet = DetectionMetrics.from_coco(true, pred)
    result = dmet.score_native_coco(iou_type='bbox', with_eval=True)
    timing = result['timing']
    assert timing['assignment_slices'] == 12
    assert timing['assignment_base_slices'] == 4
    assert timing['assignment_calls'] == 4 * 10


def test_native_coco_caps_predictions_before_candidate_geometry():
    """Candidate geometry should not include detections beyond largest maxDet."""
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    true, pred = _empty_pair(dims=(128, 128))
    truth_bbox = [8.0, 8.0, 16.0, 16.0]
    poly = [[8, 8, 24, 8, 24, 24, 8, 24]]
    true.add_annotation(
        id=1, image_id=1, category_id=1,
        bbox=truth_bbox, segmentation=poly)
    for px in range(5):
        x = 8.0 if px == 0 else 48.0 + px
        bbox = [x, 8.0, 16.0, 16.0]
        pred.add_annotation(
            id=11 + px, image_id=1, category_id=1,
            bbox=bbox,
            segmentation=[[x, 8, x + 16, 8, x + 16, 24, x, 24]],
            score=1.0 - px * .1,
        )

    dmet = DetectionMetrics.from_coco(true, pred)
    params = {'max_dets': [1, 2]}
    native = dmet.score_native_coco(
        iou_type='bbox', with_eval=True, **params)
    official = dmet.score_pycocotools(
        iou_type='bbox', bbox_annotation_policy='coco', with_eval=True,
        **params)
    _assert_eval_tensor_parity(native, official)
    timing = native['timing']
    assert timing['candidate_predictions_before_maxdet'] == 5
    assert timing['candidate_predictions_after_maxdet'] == 2


@pytest.mark.parametrize('iou_type', ['bbox', 'segm'])
def test_native_coco_kwimage_ext_assignment_backend_matches_python(iou_type):
    """Rust only replaces the greedy kernel; the full tensors stay identical."""
    from kwcoco.metrics.assignment_v2 import resolve_assignment_grid_backend
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    try:
        resolve_assignment_grid_backend('kwimage_ext')
    except ImportError:
        pytest.skip('kwimage_ext coco_greedy_match Rust capability is unavailable')

    dims = (96, 96)
    true, pred = _empty_pair(dims=dims)
    mask1 = np.zeros(dims, dtype=np.uint8)
    mask1[8:32, 8:32] = 1
    mask2 = np.zeros(dims, dtype=np.uint8)
    mask2[40:72, 44:76] = 1
    _add_mask_ann(true, mask1, aid=1)
    _add_mask_ann(true, mask2, aid=2, iscrowd=1)
    _add_mask_ann(pred, mask1, aid=11, score=.95)
    shifted = np.roll(mask2, 2, axis=1)
    _add_mask_ann(pred, shifted, aid=12, score=.90)
    _add_mask_ann(pred, mask2, aid=13, score=.85)

    dmet = DetectionMetrics.from_coco(true, pred)
    common = {
        'iou_type': iou_type,
        'with_eval': True,
    }
    if iou_type == 'segm':
        common.update({
            'overlap_backend': 'pycocotools',
            'raster_backend': 'pycocotools',
        })
    python_result = dmet.score_native_coco(
        assignment_backend='python', **common)
    rust_result = dmet.score_native_coco(
        assignment_backend='kwimage_ext', **common)
    assert python_result['summary'] == rust_result['summary']
    for key in ['precision', 'recall', 'scores', 'realpos_total']:
        np.testing.assert_array_equal(
            python_result['eval'][key], rust_result['eval'][key])
    assert rust_result['params']['assignment_backend'] == 'kwimage_ext'
    assert (
        rust_result['provenance']['resolved_assignment_backend']['kind']
        == 'rust'
    )
    assert rust_result['timing']['compact_match_path'] is True
    assert rust_result['accumulator'] == 'compact_matches->coco_tensor'
    assert rust_result['timing']['assignment_batches'] == (
        rust_result['timing']['candidate_index_builds']
    )


def test_coco_evaluator_forwards_assignment_backend():
    from kwcoco.coco_evaluator import CocoEvaluator
    from kwcoco.metrics.assignment_v2 import resolve_assignment_grid_backend

    try:
        resolve_assignment_grid_backend('kwimage_ext')
    except ImportError:
        pytest.skip('kwimage_ext coco_greedy_match Rust capability is unavailable')

    dims = (64, 64)
    true, pred = _empty_pair(dims=dims)
    mask = np.zeros(dims, dtype=np.uint8)
    mask[8:24, 8:24] = 1
    _add_mask_ann(true, mask, aid=1)
    _add_mask_ann(pred, mask, aid=11, score=.9)
    evaluator = CocoEvaluator({'true_dataset': true, 'pred_dataset': pred})
    result = evaluator.evaluate_native_coco(
        'bbox', assignment_backend='kwimage_ext', with_eval=True)['bbox']
    assert result['summary']['AP50'] == pytest.approx(1.0)
    assert result['params']['assignment_backend'] == 'kwimage_ext'


@pytest.mark.parametrize('iou_type', ['bbox', 'segm'])
def test_native_coco_compact_path_matches_explicit_row_reference(iou_type):
    """The fast compact path must equal the inspectable explicit-row path."""
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    dims = (128, 128)
    true, pred = _empty_pair(dims=dims)
    mask1 = np.zeros(dims, dtype=np.uint8)
    mask1[8:40, 8:40] = 1
    mask2 = np.zeros(dims, dtype=np.uint8)
    mask2[56:104, 60:108] = 1
    _add_mask_ann(true, mask1, aid=1)
    _add_mask_ann(true, mask2, aid=2, iscrowd=1)
    _add_mask_ann(pred, mask1, aid=11, score=.95)
    shifted = np.roll(mask2, 3, axis=1)
    _add_mask_ann(pred, shifted, aid=12, score=.90)
    fp = np.zeros(dims, dtype=np.uint8)
    fp[2:12, 90:100] = 1
    _add_mask_ann(pred, fp, aid=13, score=.85)

    dmet = DetectionMetrics.from_coco(true, pred)
    common = {'iou_type': iou_type, 'with_eval': True}
    if iou_type == 'segm':
        common.update({
            'overlap_backend': 'pycocotools',
            'raster_backend': 'pycocotools',
        })
    compact = dmet.score_native_coco(
        assignment_backend='python', with_details=False, **common)
    explicit = dmet.score_native_coco(
        assignment_backend='python', with_details=True, **common)
    assert compact['summary'] == explicit['summary']
    assert compact['accumulator'] == 'compact_matches->coco_tensor'
    assert explicit['accumulator'] == 'assignment_rows->coco_tensor'
    for key in ['precision', 'recall', 'scores', 'realpos_total']:
        np.testing.assert_array_equal(
            compact['eval'][key], explicit['eval'][key])


def test_compact_grid_batches_all_standard_area_ranges_per_image():
    """Full COCO tensors cross the compiled boundary once per image."""
    from kwcoco.metrics.detect_metrics import DetectionMetrics

    true, pred = _empty_pair(dims=(64, 64))
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    _add_mask_ann(true, mask, aid=1)
    _add_mask_ann(pred, mask, aid=11, score=.9)
    dmet = DetectionMetrics.from_coco(true, pred)
    result = dmet.score_native_coco(
        iou_type='bbox', assignment_backend='python', with_eval=True)
    timing = result['timing']
    assert timing['compact_match_path'] is True
    assert timing['assignment_base_slices'] == 4
    assert timing['assignment_calls'] == 4 * 10
    assert timing['assignment_batches'] == 1


def test_strict_coco_evaluator_preserves_category_id_collisions_in_legacy_names():
    """Strict COCO category ids must survive legacy name normalization collisions."""
    import kwcoco
    from kwcoco.coco_evaluator import CocoEvaluator

    true = kwcoco.CocoDataset()
    true.add_category('Foo Bar', id=1)
    true.add_category('foo_bar', id=2)
    true.add_image(id=1, file_name='img.png', width=64, height=64)
    true.add_annotation(
        id=1, image_id=1, category_id=1,
        bbox=[4, 4, 12, 12],
        segmentation=[[4, 4, 16, 4, 16, 16, 4, 16]],
    )
    true.add_annotation(
        id=2, image_id=1, category_id=2,
        bbox=[32, 32, 16, 16],
        segmentation=[[32, 32, 48, 32, 48, 48, 32, 48]],
    )
    pred = true.copy()
    for ann in pred.anns.values():
        ann['score'] = .9 if ann['category_id'] == 1 else .8

    evaluator = CocoEvaluator({
        'true_dataset': true,
        'pred_dataset': pred,
    })
    native = evaluator.evaluate_native_coco(
        'bbox', with_eval=True, assignment_backend='python')['bbox']
    official = evaluator.evaluate_pycocotools(
        'bbox', with_eval=True)['bbox']
    assert native['eval']['counts'][2] == 2
    assert native['eval']['category_ids'] == [0, 1]
    assert native['eval']['category_names'] == ['Foo Bar', 'foo_bar']
    _assert_eval_tensor_parity(native, official)
    assert native['summary']['AP50'] == pytest.approx(1.0)
