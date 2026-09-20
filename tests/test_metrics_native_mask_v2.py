"""Focused tests for experimental native instance-mask assignment."""


def _require_native_mask_backend():
    """Skip tests that exercise optional native mask-codec capabilities."""
    import pytest
    from kwcoco.metrics.overlap import resolve_coco_raster_backend
    from kwcoco.metrics.overlap import resolve_mask_iou_backend

    try:
        resolve_mask_iou_backend('auto')
        resolve_coco_raster_backend('auto')
    except ImportError as ex:
        pytest.skip(str(ex))


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
    _require_native_mask_backend()
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
    _require_native_mask_backend()
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
    _require_native_mask_backend()
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
    _require_native_mask_backend()
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


def test_dense_kwimage_mask_uses_resolved_coco_codec():
    """Dense masks should encode through kwcoco's resolved codec adapter."""
    import kwimage
    import numpy as np
    from kwcoco.metrics.overlap import _segmentation_to_coco_rle

    class DummyCodec:
        def __init__(self):
            self.seen = None

        def encode_one(self, mask):
            self.seen = np.asarray(mask)
            return {'size': list(mask.shape), 'counts': b'dummy'}

    codec = DummyCodec()
    mask = kwimage.Mask(np.ones((3, 4), dtype=np.uint8), 'c_mask')
    rle = _segmentation_to_coco_rle(
        mask, (3, 4), _resolved_backend=(codec, {'name': 'dummy'}))
    assert rle == {'size': [3, 4], 'counts': b'dummy'}
    assert codec.seen.flags['C_CONTIGUOUS']
    assert codec.seen.sum() == 12


def test_missing_mask_backend_raises_dependency_error_first(monkeypatch):
    """Do not leak kwimage's internal RLE NotImplementedError to callers."""
    import pytest
    import kwcoco.metrics.overlap as overlap
    from kwcoco.metrics.assignment_v2 import AssignmentCandidateIndex

    dims, true, pred = _same_box_disjoint_masks()

    def _missing_backend(_backend='auto'):
        raise ImportError('sentinel missing native mask backend')

    monkeypatch.setattr(overlap, 'resolve_mask_iou_backend', _missing_backend)
    with pytest.raises(ImportError, match='sentinel missing native mask backend'):
        AssignmentCandidateIndex.build(
            true, pred,
            compat='mutex',
            min_iou=0.5,
            iou_type='segm',
            image_dims=dims,
            spatial_threshold=None,
        )
