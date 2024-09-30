
def test_truth_reuse_policy():
    """
    Create multiple predicted bounding boxes that overlap the same truth
    and test that they are both assigned to the same box when
    truth_reuse_policy is least-used.
    """
    from kwcoco.metrics.assignment import _assign_confusion_vectors
    import numpy as np
    import pandas as pd
    import ubelt as ub
    import kwimage

    true_boxes = kwimage.Boxes(np.array([
        [50,  50, 23, 31],
        [51,  51, 23, 31],
    ]), 'cxywh')

    pred_boxes = kwimage.Boxes(np.array([
        [50,  50, 23, 31],
        [50,  50, 23, 31],
        [50,  50, 23, 31],
        [50,  50, 23, 31],
        [500,  500, 23, 31],
    ]), 'cxywh')

    true_dets = kwimage.Detections(
        boxes=true_boxes,
        weights=np.array([1] * len(true_boxes)),
        class_idxs=np.array([0] * len(true_boxes))
    )

    pred_dets = kwimage.Detections(
        boxes=pred_boxes,
        scores=np.array([0.5] * len(pred_boxes)),
        class_idxs=np.array([0] * len(pred_boxes)),
    )

    true_dets.boxes.ious(pred_dets.boxes)

    bg_weight = 1.0
    compat = 'all'
    iou_thresh = 0.1
    bias = 0.0

    y1 = _assign_confusion_vectors(true_dets, pred_dets, bias=bias,
                                   bg_weight=bg_weight, iou_thresh=iou_thresh,
                                   compat=compat, truth_reuse_policy='never')
    y1 = pd.DataFrame(y1)
    print(y1)
    # Test the policy
    truth_usage1 = ub.dict_hist(y1['txs'])
    truth_usage1.pop(-1, None)
    assert max(truth_usage1.values()) <= 1, (
        'Truth only allowed to match a maximum of one time')

    y2 = _assign_confusion_vectors(true_dets, pred_dets, bias=bias,
                                   bg_weight=bg_weight, iou_thresh=iou_thresh,
                                   compat=compat, truth_reuse_policy='least_used')
    y2 = pd.DataFrame(y2)
    print(y2)

    truth_usage2 = ub.dict_hist(y2['txs'])
    truth_usage2.pop(-1, None)

    if truth_usage2[0] != 2 or truth_usage2[1] != 2:
        raise AssertionError(
            'Truth only allowed to match multiple times, but unused objects '
            'should be matched before increasing another the usage of a '
            'different true box')
