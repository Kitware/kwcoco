from kwcoco.coco_dataset import CocoDataset
import numba
import numpy as np


@numba.njit(cache=True)
def box_area(boxes):
    out = np.empty((boxes.shape[0],))
    for i in range(boxes.shape[0]):
        out[i] = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
    return out


@numba.njit(cache=True)
def box_iou(b1, b2):
    '''
    
    Args:
        b1: np.ndarray (N, 4)
            Bounding boxes in (x1, y1, x2, y2) format.
        b2: np.ndarray (M, 4)
            Bounding boxes in (x1, y1, x2, y2) format.

    Returns:
        np.ndarray (N, M) with box intersection-over-union for each pair between b1 and b2.
    '''
    out = np.empty((b1.shape[0], b2.shape[0]))

    area2 = box_area(b2)

    for i in range(b1.shape[0]):
        a1 = (b1[i, 2] - b1[i, 0]) * (b1[i, 3] - b1[i, 1])
        for j in range(b2.shape[0]):
            a2 = area2[j]

            x1 = max(b1[i, 0], b2[j, 0])
            y1 = max(b1[i, 1], b2[j, 1])
            x2 = min(b1[i, 2], b2[j, 2])
            y2 = min(b1[i, 3], b2[j, 3])

            w = max(0, x2 - x1)
            h = max(0, y2 - y1)

            inter = w * h
            union = a1 + a2 - inter
            iou = inter / union
            out[i, j] = iou
    return out


# TODO: I think there is an approach where the full IOU matrix never needs to be computed,
# only a single row at a time, keeping track of which truth boxes have been matched, and
# probably being able to stop if all truths have been matched. Likely to be more efficient
# due to less memory allocation and ability to reuse a single row vector. Does add some
# additional overhead in keeping track of some stuff though.
@numba.njit(cache=True)
def assign_confusion_vectors(
    pred_gid: np.ndarray,
    pred_cid: np.ndarray,
    pred_boxes: np.ndarray,
    scores: np.ndarray,
    truth_gid: np.ndarray,
    truth_cid: np.ndarray,
    truth_boxes: np.ndarray,
    iou_thres: float=0.5,
    mutex: bool=True,
) -> np.ndarray:
    '''Fast confusion vector assignment.

    Note: this function assumes that the predictions and ground truth are both sorted
    by gid, and will break if that assumption is violated.
    '''
    res = []
    pstart, tstart = 0, 0
    pend, tend = 1, 1
    while True:
        while pend < pred_gid.shape[0] and pred_gid[pend] == pred_gid[pstart]:
            pend += 1
        while tend < truth_gid.shape[0] and truth_gid[tend] == truth_gid[tstart]:
            tend += 1

        # handle case where we've exhausted all predictions but there are
        # still ground-truth images left
        if pstart < pred_gid.shape[0]:
            pgid = pred_gid[pstart]
            pcid = pred_cid[pstart:pend]
            pbox = pred_boxes[pstart:pend]
            s = scores[pstart:pend]
        else:
            pgid = -1
            pcid = None
            pbox = None
            s = None

        tgid = truth_gid[tstart]
        tcid = truth_cid[tstart:tend]
        tbox = truth_boxes[tstart:tend]

        if pgid == tgid:
            # Computes full IOU matrix between predictions and truths in the image
            # and then ignores matches with different cids if mutex = True
            sord = np.argsort(s)[::-1]
            iou = box_iou(pbox, tbox)

            for i in sord:
                best, best_iou = -1, 0
                for j in range(iou.shape[1]):
                    if mutex and pcid[i] != tcid[j]:
                        iou[i, j] = 0
                    elif best_iou < iou[i, j] > iou_thres:
                        best = j
                        best_iou = iou[i, j]
                if best != -1:
                    # Once a match is identified, suppress any further matches for the
                    # pred or truth in this pair.
                    # multiple predictions cannot match the same ground-truth box
                    for j in range(0, iou.shape[0]):
                        if j != i:
                            iou[j, best] = -1.0
                    # a single prediction cannot match multiple ground-truth boxes
                    # (this is necessary for correctly identifying false negatives)
                    for j in range(0, iou.shape[1]):
                        if j != best:
                            iou[i, j] = -1.0

                    # true positive
                    res.append([pcid[i], tcid[best], s[i], best_iou, best, i, pgid])
                else:
                    # false positive
                    res.append([pcid[i], -1.0, s[i], -1.0, -1.0, i, pgid])
                
            # false negatives
            for j in range(iou.shape[1]):
                false_neg = True
                for i in range(iou.shape[0]):
                    if iou[i, j] > iou_thres:
                        false_neg = False
                        break
                if false_neg:
                    res.append([-1.0, tcid[j], -np.inf, -1.0, j, -1.0, pgid])
            
            pstart = pend
            tstart = tend
        # no truth annotations for this image, all preds are false positives
        elif 0 <= pgid < tgid:
            for i in range(pbox.shape[0]):
                # false positive
                res.append([pcid[i], -1.0, s[i], -1.0, -1.0, i, pgid])
            pstart = pend
        # no predictions for this image, all truths are false negatives
        else: # pgid > tgid:
            for i in range(tbox.shape[0]):
                # false negative
                res.append([-1.0, tcid[i], -np.inf, -1.0, i, -1.0, tgid])
            tstart = tend

        if pstart >= pred_gid.shape[0] and tstart >= truth_gid.shape[0]:
            break
    
    res = np.array(res)
    return res


def confusion_vectors(truth: CocoDataset, pred: CocoDataset, iou_thres: float=0.5, mutex: bool=True):
    '''Create confusion vectors from truth and pred CocoDatasets.
    '''
    pred_anns = pred.annots()
    truth_anns = truth.annots()

    # get the prediction data and order by (image_id, annotation_id)
    pred_gid = np.array(pred_anns.gids)
    pred_cid = np.array(pred_anns.cids)
    pred_boxes = pred_anns.boxes.to_ltrb().data
    scores = np.array(pred_anns.get('score'))
    psrt = np.lexsort((pred_anns.ids, pred_gid))
    pred_gid, pred_cid, pred_boxes, scores = (
        pred_gid[psrt], pred_cid[psrt], pred_boxes[psrt], scores[psrt]
    )

    # get the truth data and order by (image_id, annotation_id)
    truth_gid = np.array(truth_anns.gids)
    truth_cid = np.array(truth_anns.cids)
    truth_boxes = truth_anns.boxes.to_ltrb().data
    tsrt = np.lexsort((truth_anns.ids, truth_gid))
    truth_gid, truth_cid, truth_boxes = truth_gid[tsrt], truth_cid[tsrt], truth_boxes[tsrt]

    # fast confusion vector assignment
    cv = assign_confusion_vectors(
        pred_gid,
        pred_cid,
        pred_boxes,
        scores,
        truth_gid,
        truth_cid,
        truth_boxes,
        iou_thres=iou_thres,
        mutex=mutex
    )

    # convert to pandas dataframe
    import pandas as pd
    cols = ['pred', 'true', 'score', 'iou', 'txs', 'pxs', 'gid']
    dt = [np.int32, np.int32, np.float32, np.float32, np.int32, np.int32, np.int32]
    cv = pd.DataFrame({k: v.astype(d) for k, v, d in zip(cols, cv.T, dt)}, columns=cols)
    return cv


def match_ids_by_filename(truth, pred):
    '''Ensure that filenames and image ids match up between truth and prediction files.
    '''
    name2id = {x['file_name'].rsplit('/', 1)[-1]: x['id'] for x in truth.imgs.values()}
    g2a = pred.index.gid_to_aids.copy()
    for x in pred.dataset['images']:
        gid = x['id']
        new_gid = name2id[x['file_name'].rsplit('/', 1)[-1]]
        for aid in g2a[gid]:
            pred.anns[aid]['image_id'] = new_gid
        x['id'] = new_gid
    pred = CocoDataset.from_data(pred.dataset)
    return pred


def calculate_map(
    truth: CocoDataset,
    pred: CocoDataset,
    iou_thres: float=0.5,
    mutex: bool=True,
    min_score: float=0.0,
):
    from kwcoco.metrics import ConfusionVectors
    import kwarray

    # Remap pred image IDs to match truth based on filename
    pred = match_ids_by_filename(truth, pred)
    
    # Filter out any predictions with score < min_score
    to_remove = [aid for aid in pred.anns if pred.anns[aid]['score'] < min_score]
    if len(to_remove) > 0:
        removed = pred.remove_annotations(to_remove)
        print(f'Removed {removed["annotations"]} predictions with score < {min_score}')

    # Compute confusion vectors
    cv = confusion_vectors(truth, pred, iou_thres, mutex)

    # Convert to form usable by kwcoco metrics
    cv.insert(4, 'weight', 1.0)
    # shift to zero-based index for cat ids
    cv.loc[cv.true > -1, 'true'] -= 1
    cv.loc[cv.pred > -1, 'pred'] -= 1

    # Use kwcoco to compute metrics
    confv = ConfusionVectors(kwarray.DataFrameArray(cv), [x['name'] for x in pred.cats.values()])
    perclass = confv.binarize_ovr()
    map = perclass.measures()['mAP']

    return map


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    help = {
        'truth': 'Path to ground-truth coco annotation file.',
        'pred': 'Path to prediction coco annotation file.',
        'iou': 'IoU threshold for matching predictions to ground-truth boxes.',
        'mutex': (
            'If True, only allow matches between predictions and truths with the same category label. '
            'If False, allow matches between predictions and truths regardless of category.'
        ),
        'min-score': 'Minimum confidence score; any predictions with a lower confidence will be ignored.',
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('truth', type=Path, help=help['truth'])
    parser.add_argument('pred', type=Path, help=help['pred'])
    parser.add_argument('-i', '--iou', type=float, default=0.5, help=help['iou'])
    parser.add_argument('--mutex', action=argparse.BooleanOptionalAction, default=True, help=help['mutex'])
    parser.add_argument('--min-score', type=float, default=0.0, help=help['min-score'])

    args = parser.parse_args()

    truth = CocoDataset(args.truth)
    pred = CocoDataset(args.pred)
    mAP = calculate_map(truth, pred, args.iou, args.mutex, args.min_score)
    print(f'mAP: {mAP}')