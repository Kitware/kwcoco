"""
Tests for the experimental assignment_v2 module.
"""


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

    build_kwargs = {
        'compat': 'mutex',
        'bias': 1,
        'min_iou': 0.3,
    }
    dense = AssignmentCandidateIndex.build(
        true_dets,
        pred_dets,
        spatial_threshold=None,
        **build_kwargs,
    )
    sparse = AssignmentCandidateIndex.build(
        true_dets,
        pred_dets,
        spatial_threshold=1,
        **build_kwargs,
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
