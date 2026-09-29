"""
Tests for the experimental assignment_v2 module.
"""
import pytest


def _row_signature(y):
    """Normalize assignment rows for small deterministic comparisons."""
    rows = list(zip(y['pred'], y['true'], y['txs'], y['pxs']))
    return sorted(rows)


def test_assignment_v2_matches_v1_on_simple_mutex_case():
    from kwcoco.metrics.assignment import _assign_confusion_vectors
    from kwcoco.metrics.assignment_v2 import assign_confusion_vectors_v2
    import kwimage
    import numpy as np

    true_dets = kwimage.Detections(
        boxes=kwimage.Boxes(np.array([
            [0, 0, 10, 10],
            [30, 30, 10, 10],
        ]), 'xywh'),
        class_idxs=np.array([0, 0]),
        weights=np.array([1.0, 1.0]),
    )
    pred_dets = kwimage.Detections(
        boxes=kwimage.Boxes(np.array([
            [0, 0, 10, 10],
            [100, 100, 10, 10],
        ]), 'xywh'),
        class_idxs=np.array([0, 0]),
        scores=np.array([0.9, 0.8]),
    )

    y1 = _assign_confusion_vectors(
        true_dets,
        pred_dets,
        iou_thresh=0.5,
        compat='mutex',
        ignore_classes=None,
    )
    y2 = assign_confusion_vectors_v2(
        true_dets,
        pred_dets,
        iou_thresh=0.5,
        compat='mutex',
        ignore_classes=None,
        spatial_threshold=None,
    )

    assert _row_signature(y2) == _row_signature(y1)


def test_assignment_v2_reuses_index_for_threshold_specific_assignment():
    from kwcoco.metrics.assignment_v2 import AssignmentCandidateIndex
    import kwimage
    import numpy as np

    true_dets = kwimage.Detections(
        boxes=kwimage.Boxes(np.array([[0, 0, 10, 10]]), 'xywh'),
        class_idxs=np.array([0]),
    )
    pred_dets = kwimage.Detections(
        boxes=kwimage.Boxes(np.array([
            [4, 0, 10, 10],  # IoU ~= 0.43, but higher score
            [0, 0, 10, 10],  # IoU = 1.0, but lower score
        ]), 'xywh'),
        class_idxs=np.array([0, 0]),
        scores=np.array([0.9, 0.8]),
    )

    index = AssignmentCandidateIndex.build(
        true_dets,
        pred_dets,
        compat='mutex',
        min_iou=0.3,
        spatial_threshold=None,
    )

    low = index.assign(0.3, ignore_classes=None)
    high = index.assign(0.5, ignore_classes=None)

    # At low threshold the high-scoring partial overlap gets the truth.
    assert low['txs'][0] == 0
    assert low['pxs'][0] == 0

    # At high threshold the partial overlap is ineligible, so the lower-score
    # perfect box gets the truth.
    assert high['txs'][0] == -1
    assert high['pxs'][0] == 0
    assert high['txs'][1] == 0
    assert high['pxs'][1] == 1


def test_assignment_v2_sparse_and_dense_candidate_indexes_agree():
    from kwcoco.metrics.assignment_v2 import AssignmentCandidateIndex
    import kwimage
    import numpy as np

    true_dets = kwimage.Detections(
        boxes=kwimage.Boxes(np.array([
            [0, 0, 10, 10],
            [100, 100, 10, 10],
            [200, 200, 10, 10],
        ]), 'xywh'),
        class_idxs=np.array([0, 0, 0]),
    )
    pred_dets = kwimage.Detections(
        boxes=kwimage.Boxes(np.array([
            [0, 0, 10, 10],
            [100, 100, 10, 10],
            [250, 250, 10, 10],
        ]), 'xywh'),
        class_idxs=np.array([0, 0, 0]),
        scores=np.array([0.9, 0.8, 0.7]),
    )

    dense = AssignmentCandidateIndex.build(
        true_dets,
        pred_dets,
        compat='mutex',
        min_iou=0.5,
        spatial_threshold=None,
    )
    sparse = AssignmentCandidateIndex.build(
        true_dets,
        pred_dets,
        compat='mutex',
        min_iou=0.5,
        spatial_threshold=1,
    )

    assert sparse.build_stats['spatial_blocks'] >= 1
    assert _row_signature(sparse.assign(0.5, ignore_classes=None)) == _row_signature(
        dense.assign(0.5, ignore_classes=None)
    )


def test_assignment_v2_coerces_ancestor_matches_like_v1():
    import networkx as nx
    import numpy as np
    import kwcoco
    import kwimage

    from kwcoco.metrics.assignment import _assign_confusion_vectors
    from kwcoco.metrics.assignment_v2 import assign_confusion_vectors_v2

    graph = nx.DiGraph()
    graph.add_edge('animal', 'dog')
    classes = kwcoco.CategoryTree(graph)
    animal_cx = classes.index('animal')
    dog_cx = classes.index('dog')

    true_dets = kwimage.Detections(
        boxes=kwimage.Boxes(np.array([[0, 0, 10, 10]]), 'xywh'),
        class_idxs=np.array([animal_cx]),
        classes=classes,
    )
    pred_dets = kwimage.Detections(
        boxes=kwimage.Boxes(np.array([[0, 0, 10, 10]]), 'xywh'),
        class_idxs=np.array([dog_cx]),
        scores=np.array([0.9]),
        classes=classes,
    )

    y1 = _assign_confusion_vectors(
        true_dets,
        pred_dets,
        classes=classes,
        compat='ancestors',
        iou_thresh=0.5,
        ignore_classes=None,
    )
    y2 = assign_confusion_vectors_v2(
        true_dets,
        pred_dets,
        classes=classes,
        compat='ancestors',
        iou_thresh=0.5,
        ignore_classes=None,
        spatial_threshold=None,
    )

    assert _row_signature(y2) == _row_signature(y1)
    assert y2['pred'] == [animal_cx]
    assert y2['true'] == [animal_cx]


def test_assignment_v2_sparse_bias_matches_dense():
    import kwimage
    import numpy as np

    from kwcoco.metrics.assignment_v2 import AssignmentCandidateIndex

    true_dets = kwimage.Detections(
        boxes=kwimage.Boxes(np.array([[1, 0, 1, 1]]), 'xywh'),
        class_idxs=np.array([0]),
    )
    pred_dets = kwimage.Detections(
        boxes=kwimage.Boxes(
            np.array([
                [0, 0, 1, 1],
                [1, 0, 1, 1],
            ]),
            'xywh',
        ),
        class_idxs=np.array([0, 0]),
        scores=np.array([0.9, 0.8]),
    )

    dense = AssignmentCandidateIndex.build(
        true_dets,
        pred_dets,
        spatial_threshold=None,
        compat='mutex',
        bias=1,
        min_iou=0.3,
    )
    sparse = AssignmentCandidateIndex.build(
        true_dets,
        pred_dets,
        spatial_threshold=1,
        compat='mutex',
        bias=1,
        min_iou=0.3,
    )

    dense_y = dense.assign(0.3, ignore_classes=None)
    sparse_y = sparse.assign(0.3, ignore_classes=None)
    assert sparse.build_stats['spatial_blocks'] >= 1
    assert _row_signature(sparse_y) == _row_signature(dense_y)
    assert sparse_y['txs'][0] == 0
    assert sparse_y['pxs'][0] == 0


def test_assignment_v2_exact_iou_threshold_matches_v1():
    import kwimage
    import numpy as np

    from kwcoco.metrics.assignment import _assign_confusion_vectors
    from kwcoco.metrics.assignment_v2 import assign_confusion_vectors_v2

    true_dets = kwimage.Detections(
        boxes=kwimage.Boxes(np.array([[0, 0, 2, 1]]), 'xywh'),
        class_idxs=np.array([0]),
    )
    pred_dets = kwimage.Detections(
        boxes=kwimage.Boxes(np.array([[0, 0, 1, 1]]), 'xywh'),
        class_idxs=np.array([0]),
        scores=np.array([0.9]),
    )

    y1 = _assign_confusion_vectors(
        true_dets,
        pred_dets,
        compat='mutex',
        iou_thresh=0.5,
        ignore_classes=None,
    )
    y2 = assign_confusion_vectors_v2(
        true_dets,
        pred_dets,
        compat='mutex',
        iou_thresh=0.5,
        ignore_classes=None,
        spatial_threshold=None,
    )

    assert _row_signature(y2) == _row_signature(y1)
    assert y2['txs'] == [-1, 0]
    assert y2['pxs'] == [0, -1]


def test_assignment_v2_default_compat_matches_v1():
    import kwimage
    import numpy as np

    from kwcoco.metrics.assignment import _assign_confusion_vectors
    from kwcoco.metrics.assignment_v2 import assign_confusion_vectors_v2

    true_dets = kwimage.Detections(
        boxes=kwimage.Boxes(np.array([[0, 0, 10, 10]]), 'xywh'),
        class_idxs=np.array([0]),
    )
    pred_dets = kwimage.Detections(
        boxes=kwimage.Boxes(np.array([[0, 0, 10, 10]]), 'xywh'),
        class_idxs=np.array([1]),
        scores=np.array([0.9]),
    )

    y1 = _assign_confusion_vectors(
        true_dets,
        pred_dets,
        iou_thresh=0.5,
        ignore_classes=None,
    )
    y2 = assign_confusion_vectors_v2(
        true_dets,
        pred_dets,
        iou_thresh=0.5,
        ignore_classes=None,
        spatial_threshold=None,
    )

    assert _row_signature(y2) == _row_signature(y1)
    assert y2['txs'] == [0]
    assert y2['pxs'] == [0]


@pytest.mark.parametrize('empty_side', ['truth', 'prediction'])
def test_assignment_v2_coco_bbox_accepts_empty_box_arrays(empty_side):
    """COCO bbox construction must preserve the Nx4 axis for empty images."""
    import kwimage
    import numpy as np

    from kwcoco.metrics.assignment_v2 import AssignmentCandidateIndex

    empty_boxes = kwimage.Boxes([], 'xywh')
    one_box = kwimage.Boxes([[0, 0, 10, 10]], 'xywh')
    empty_classes = np.empty(0, dtype=int)
    one_class = np.array([0], dtype=int)

    if empty_side == 'truth':
        true_dets = kwimage.Detections(
            boxes=empty_boxes,
            class_idxs=empty_classes,
        )
        pred_dets = kwimage.Detections(
            boxes=one_box,
            class_idxs=one_class,
            scores=np.array([0.9]),
        )
    else:
        true_dets = kwimage.Detections(
            boxes=one_box,
            class_idxs=one_class,
        )
        pred_dets = kwimage.Detections(
            boxes=empty_boxes,
            class_idxs=empty_classes,
            scores=np.empty(0, dtype=float),
        )

    index = AssignmentCandidateIndex.build(
        true_dets,
        pred_dets,
        classes=['thing'],
        compat='mutex',
        iou_type='bbox',
        bbox_overlap_mode='coco',
        threshold_policy='inclusive',
        min_iou=0.5,
    )
    assert index.true_areas.shape == (len(true_dets),)
    assert index.pred_areas.shape == (len(pred_dets),)

    rows = index.assign(0.5, ignore_classes=None)
    assert len(rows['txs']) == 1
    if empty_side == 'truth':
        assert rows['txs'] == [-1]
        assert rows['pxs'] == [0]
    else:
        assert rows['txs'] == [0]
        assert rows['pxs'] == [-1]


def test_assignment_v2_kwimage_ext_batched_kernel_matches_python():
    """The optional Rust kernel must be row-for-row identical to Python."""
    import numpy as np
    import kwimage

    from kwcoco.metrics.assignment_v2 import AssignmentCandidateIndex
    from kwcoco.metrics.assignment_v2 import resolve_assignment_backend

    try:
        resolve_assignment_backend('kwimage_ext')
    except ImportError:
        pytest.skip('kwimage_ext coco_greedy_match Rust capability is unavailable')

    true = kwimage.Detections(
        boxes=kwimage.Boxes([
            [0, 0, 20, 20],
            [8, 0, 20, 20],
            [40, 0, 16, 16],
        ], 'xywh'),
        class_idxs=np.array([0, 0, 0]),
    )
    true.data['iscrowd'] = np.array([0, 1, 0])
    pred = kwimage.Detections(
        boxes=kwimage.Boxes([
            [0, 0, 20, 20],
            [9, 0, 18, 20],
            [40, 0, 16, 16],
            [100, 100, 8, 8],
        ], 'xywh'),
        class_idxs=np.array([0, 0, 0, 0]),
        scores=np.array([.95, .90, .85, .80]),
    )
    index = AssignmentCandidateIndex.build(
        true,
        pred,
        classes=['thing'],
        compat='mutex',
        iou_type='bbox',
        bbox_overlap_mode='coco',
        threshold_policy='inclusive',
        annotation_ignore_policy='coco',
        min_iou=.5,
        spatial_threshold=None,
    )
    kwargs = {
        'ignore_classes': None,
        'max_dets': 3,
        'max_dets_scope': 'category',
        'area_range': (0.0, 20 ** 2),
        'truth_reuse_policy': 'never',
    }
    thresholds = [.5, .75, .95]
    py_rows = index.assign_many(
        thresholds, assignment_backend='python', **kwargs)
    rust_rows = index.assign_many(
        thresholds, assignment_backend='kwimage_ext', **kwargs)
    assert rust_rows == py_rows

    offsets, flat_txs, flat_ious = index._candidate_csr()
    assert offsets.shape == (len(pred) + 1,)
    assert offsets[-1] == len(flat_txs) == len(flat_ious)


def test_assignment_v2_auto_keeps_python_for_ignore_classes():
    """Threshold-dependent native ignore regions are still Python-owned."""
    import numpy as np
    import kwimage

    from kwcoco.metrics.assignment_v2 import AssignmentCandidateIndex

    classes = ['thing', 'ignore']
    true = kwimage.Detections(
        boxes=kwimage.Boxes([[0, 0, 10, 10]], 'xywh'),
        class_idxs=np.array([1]),
        classes=classes,
    )
    pred = kwimage.Detections(
        boxes=kwimage.Boxes([[0, 0, 10, 10]], 'xywh'),
        class_idxs=np.array([0]),
        scores=np.array([.9]),
        classes=classes,
    )
    index = AssignmentCandidateIndex.build(
        true, pred, classes=classes, min_iou=.5, spatial_threshold=None)
    auto = index.assign_many(
        [.5, .75], assignment_backend='auto', ignore_classes='ignore')
    python = index.assign_many(
        [.5, .75], assignment_backend='python', ignore_classes='ignore')
    assert auto == python


def test_assignment_backend_resolution_metadata_is_not_shared_mutable_state():
    from kwcoco.metrics.assignment_v2 import resolve_assignment_backend

    key1, func1, meta1 = resolve_assignment_backend('python')
    meta1['caller_mutation'] = True
    key2, func2, meta2 = resolve_assignment_backend('python')
    assert key1 == key2 == 'python'
    assert func1 is func2 is None
    assert 'caller_mutation' not in meta2
