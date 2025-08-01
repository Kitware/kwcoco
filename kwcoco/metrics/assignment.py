"""
TODO:
    - [ ] _fast_pdist_priority: Look at absolute difference in sibling entropy
        when deciding whether to go up or down in the tree.

    - [ ] medschool applications true-pred matching (applicant proposing) fast
        algorithm.

    - [ ] Maybe looping over truth rather than pred is faster? but it makes you
        have to combine pred score / ious, which is weird.

    - [x] preallocate ndarray and use hstack to build confusion vectors?
        - doesn't help

    - [ ] relevant classes / classes / classes-of-interest we care about needs
        to be a first class member of detection metrics.

    - [ ] Add parameter that allows one prediction to "match" to more than one
        truth object. (example: we have a duck detector problem and all the
        ducks in a row are annotated as separate object, and we only care about
        getting the group)
"""
import warnings
import networkx as nx
import numpy as np
import ubelt as ub


USE_NEG_INF = True


def _assign_confusion_vectors(true_dets, pred_dets, bg_weight=1.0,
                              iou_thresh=0.5, bg_cidx=-1, bias=0.0, classes=None,
                              compat='all', prioritize='iou',
                              ignore_classes='ignore',
                              max_dets=None,
                              truth_reuse_policy='never',
                              ):
    """
    Create confusion vectors for detections by assigning to ground true boxes

    Given predictions and truth for an image return (y_pred, y_true,
    y_score), which is suitable for sklearn classification metrics

    Args:
        true_dets (Detections):
            groundtruth with boxes, class_idxs, and weights

        pred_dets (Detections):
            predictions with boxes, class_idxs, and scores

        iou_thresh (float, default=0.5):
            bounding box overlap iou threshold required for assignment

        bias (float, default=0.0):
            for computing bounding box overlap, either 1 or 0

        gids (List[int], default=None):
            which subset of images ids to compute confusion metrics on. If
            not specified all images are used.

        compat (str, default='all'):
            can be ('ancestors' | 'mutex' | 'all'). Determines which pred
            boxes are allowed to match which true boxes. If 'mutex', then
            pred boxes can only match true boxes of the same class. If
            'ancestors', then pred boxes can match true boxes that match or
            have a coarser label. If 'all', then any pred can match any
            true, regardless of its category label.

        prioritize (str, default='iou'):
            can be ('iou' | 'class' | 'correct'). Determines which box to
            assign to if multiple true boxes overlap a predicted box.  if
            prioritize is iou, then the true box with maximum iou (above
            iou_thresh) will be chosen.  If prioritize is class, then it will
            prefer matching a compatible class above a higher iou. If
            prioritize is correct, then ancestors of the true class are
            preferred over descendants of the true class, over unrelated
            classes.

        bg_cidx (int, default=-1):
            The index of the background class.  The index used in the truth
            column when a predicted bounding box does not match any true
            bounding box.

        classes (List[str] | kwcoco.CategoryTree):
            mapping from class indices to class names. Can also contain class
            hierarchy information.

        ignore_classes (str | List[str]):
            class name(s) indicating ignore regions

        max_dets (int): maximum number of detections to consider

        truth_reuse_policy (str | bool):
            Defaults to "never".
            * "never", which means only a single predicted detection
               is allowed to match a true detection.
            * "least_used" means that a predicted box will be allowed to
              match a true object, but it will prioritize unused boxes
              before reusing one.

    TODO:
        - [ ] This is a bottleneck function. An implementation in C / C++ /
              Cython / Rust would likely improve the overall system.

        - [ ] Implement crowd truth. Allow multiple predictions to match any
              truth object marked as "iscrowd".

    Returns:
        dict: with relevant confusion vectors. This keys of this dict can be
            interpreted as columns of a data frame. The `txs` / `pxs` columns
            represent the indexes of the true / predicted annotations that were
            assigned as matching. Additionally each row also contains the true
            and predicted class index, the predicted score, the true weight and
            the iou of the true and predicted boxes. A `txs` value of -1 means
            that the predicted box was not assigned to a true annotation and a
            `pxs` value of -1 means that the true annotation was not assigned to
            any predicted annotation.

    Example:
        >>> # xdoctest: +REQUIRES(module:pandas)
        >>> from kwcoco.metrics.assignment import _assign_confusion_vectors
        >>> import pandas as pd
        >>> import kwimage
        >>> # Given a raw numpy representation construct Detection wrappers
        >>> true_dets = kwimage.Detections(
        >>>     boxes=kwimage.Boxes(np.array([
        >>>         [ 0,  0, 10, 10], [10,  0, 20, 10],
        >>>         [10,  0, 20, 10], [20,  0, 30, 10]]), 'tlbr'),
        >>>     weights=np.array([1, 0, .9, 1]),
        >>>     class_idxs=np.array([0, 0, 1, 2]))
        >>> pred_dets = kwimage.Detections(
        >>>     boxes=kwimage.Boxes(np.array([
        >>>         [6, 2, 20, 10], [3,  2, 9, 7],
        >>>         [3,  9, 9, 7],  [3,  2, 9, 7],
        >>>         [2,  6, 7, 7],  [20,  0, 30, 10]]), 'tlbr'),
        >>>     scores=np.array([.5, .5, .5, .5, .5, .5]),
        >>>     class_idxs=np.array([0, 0, 1, 2, 0, 1]))
        >>> bg_weight = 1.0
        >>> compat = 'all'
        >>> iou_thresh = 0.5
        >>> bias = 0.0
        >>> y = _assign_confusion_vectors(true_dets, pred_dets, bias=bias,
        >>>                               bg_weight=bg_weight, iou_thresh=iou_thresh,
        >>>                               compat=compat)
        >>> y = pd.DataFrame(y)
        >>> print(y)  # xdoc: +IGNORE_WANT
           pred  true  score  weight     iou  txs  pxs
        0     1     2 0.5000  1.0000  1.0000    3    5
        1     0    -1 0.5000  1.0000 -1.0000   -1    4
        2     2    -1 0.5000  1.0000 -1.0000   -1    3
        3     1    -1 0.5000  1.0000 -1.0000   -1    2
        4     0    -1 0.5000  1.0000 -1.0000   -1    1
        5     0     0 0.5000  0.0000  0.6061    1    0
        6    -1     0 0.0000  1.0000 -1.0000    0   -1
        7    -1     1 0.0000  0.9000 -1.0000    2   -1

    Example:
        >>> # xdoctest: +REQUIRES(module:pandas)
        >>> from kwcoco.metrics.assignment import _assign_confusion_vectors
        >>> import pandas as pd
        >>> import ubelt as ub
        >>> from kwcoco.metrics import DetectionMetrics
        >>> dmet = DetectionMetrics.demo(nimgs=1, nclasses=8,
        >>>                              nboxes=(0, 20), n_fp=20,
        >>>                              box_noise=.2, cls_noise=.3)
        >>> gid = ub.peek(dmet.gid_to_pred_dets)
        >>> true_dets = dmet.true_detections(gid)
        >>> pred_dets = dmet.pred_detections(gid)
        >>> y = _assign_confusion_vectors(true_dets, pred_dets,
        >>>                               classes=dmet.classes,
        >>>                               compat='all', prioritize='class')
        >>> y = pd.DataFrame(y)
        >>> print(y)  # xdoc: +IGNORE_WANT
        >>> y = _assign_confusion_vectors(true_dets, pred_dets,
        >>>                               classes=dmet.classes,
        >>>                               compat='ancestors', iou_thresh=.5)
        >>> y = pd.DataFrame(y)
        >>> print(y)  # xdoc: +IGNORE_WANT

    Example:
        >>> # xdoctest: +REQUIRES(module:pandas)
        >>> from kwcoco.metrics.assignment import _assign_confusion_vectors
        >>> import pandas as pd
        >>> import ubelt as ub
        >>> from kwcoco.metrics import DetectionMetrics
        >>> dmet = DetectionMetrics.demo(nimgs=1, nclasses=8,
        >>>                              nboxes=(0, 20), n_fp=20,
        >>>                              box_noise=.2, cls_noise=.3)
        >>> classes = dmet.classes
        >>> gid = ub.peek(dmet.gid_to_pred_dets)
        >>> true_dets = dmet.true_detections(gid)
        >>> pred_dets = dmet.pred_detections(gid)
        >>> y = _assign_confusion_vectors(true_dets, pred_dets,
        >>>                               classes=dmet.classes,
        >>>                               compat='all', truth_reuse_policy='least_used')
        >>> y = pd.DataFrame(y)
        >>> print(y)  # xdoc: +IGNORE_WANT

    Ignore:
        from xinspect.dynamic_kwargs import get_func_kwargs
        globals().update(get_func_kwargs(_assign_confusion_vectors))
    """
    import kwarray
    valid_compat_keys = {'ancestors', 'mutex', 'all'}
    if compat not in valid_compat_keys:
        raise KeyError(compat)
    if classes is None and compat == 'ancestors':
        compat = 'mutex'

    if compat == 'mutex':
        prioritize = 'iou'

    # Group true boxes by class
    # Keep track which true boxes are available / not assigned
    unique_tcxs, tgroupxs = kwarray.group_indices(true_dets.class_idxs)
    cx_to_txs = dict(zip(unique_tcxs, tgroupxs))

    unique_pcxs = np.array(sorted(set(pred_dets.class_idxs)))

    if classes is None:
        import kwcoco
        # Build mutually exclusive category tree
        all_cxs = sorted(set(map(int, unique_pcxs)) | set(map(int, unique_tcxs)))
        all_cxs = list(range(max(all_cxs) + 1))
        classes = kwcoco.CategoryTree.from_mutex(all_cxs)

    cx_to_ancestors = classes.idx_to_ancestor_idxs()

    if prioritize == 'iou':
        pdist_priority = None  # TODO: cleanup
    else:
        pdist_priority = _fast_pdist_priority(classes, prioritize)

    if compat == 'mutex':
        # assume classes are mutually exclusive if hierarchy is not given
        cx_to_matchable_cxs = {cx: [cx] for cx in unique_pcxs}
    elif compat == 'ancestors':
        cx_to_matchable_cxs = {
            cx: sorted([cx] + sorted(ub.take(
                classes.node_to_idx,
                nx.ancestors(classes.graph, classes.idx_to_node[cx]))))
            for cx in unique_pcxs
        }
    elif compat == 'all':
        cx_to_matchable_cxs = {cx: unique_tcxs for cx in unique_pcxs}
    else:
        raise KeyError(compat)

    if compat == 'all':
        # In this case simply run the full pairwise iou
        common_true_idxs = np.arange(len(true_dets))
        cx_to_matchable_txs = {cx: common_true_idxs for cx in unique_pcxs}
        common_ious = pred_dets.boxes.ious(true_dets.boxes, bias=bias)
        # common_ious = pred_dets.boxes.ious(true_dets.boxes, impl='c', bias=bias)
        iou_lookup = dict(enumerate(common_ious))
    else:
        # For each pred-category find matchable true-indices
        cx_to_matchable_txs = {}
        for cx, compat_cx in cx_to_matchable_cxs.items():
            matchable_cxs = cx_to_matchable_cxs[cx]
            compat_txs = ub.take(cx_to_txs, matchable_cxs, default=[])
            compat_txs = np.array(sorted(ub.flatten(compat_txs)), dtype=int)
            cx_to_matchable_txs[cx] = compat_txs

        # Batch up the IOU pre-computation between compatible truths / preds
        iou_lookup = {}
        unique_pred_cxs, pgroupxs = kwarray.group_indices(pred_dets.class_idxs)
        for cx, pred_idxs in zip(unique_pred_cxs, pgroupxs):
            true_idxs = cx_to_matchable_txs[cx]
            ious = pred_dets.boxes[pred_idxs].ious(
                true_dets.boxes[true_idxs], bias=bias)
            _px_to_iou = dict(zip(pred_idxs, ious))
            iou_lookup.update(_px_to_iou)

    iou_thresh_list = (
        [iou_thresh] if not ub.iterable(iou_thresh) else iou_thresh)

    iou_thresh_to_y = {}
    for iou_thresh_ in iou_thresh_list:
        isvalid_lookup = {px: ious > iou_thresh_ for px, ious in iou_lookup.items()}

        y =  _critical_loop(true_dets, pred_dets, iou_lookup, isvalid_lookup,
                            cx_to_matchable_txs, bg_weight, prioritize, iou_thresh_,
                            pdist_priority, cx_to_ancestors, bg_cidx,
                            ignore_classes=ignore_classes, max_dets=max_dets,
                            truth_reuse_policy=truth_reuse_policy)
        iou_thresh_to_y[iou_thresh_] = y

    if ub.iterable(iou_thresh):
        return iou_thresh_to_y
    else:
        return y


def _critical_loop(true_dets, pred_dets, iou_lookup, isvalid_lookup,
                   cx_to_matchable_txs, bg_weight, prioritize, iou_thresh_,
                   pdist_priority, cx_to_ancestors, bg_cidx, ignore_classes,
                   max_dets, truth_reuse_policy):
    """
    Args:
        true_dets (Detections):
        pred_dets (Detections):
        iou_lookup (Dict[int, ndarray]):
        isvalid_lookup (Dict[int, ndarray]):
        cx_to_matchable_txs (Dict[int64, ndarray]):
        bg_weight (float):
        prioritize (str):
        iou_thresh_ (float):
        pdist_priority (ndarray):
        cx_to_ancestors (Dict[int, set[int]]):
        bg_cidx (int):
        ignore_classes (str):
        max_dets (NoneType):
        truth_reuse_policy (bool):

    Returns:
        Dict[str, ndarray]

    Ignore:
        keys = 'true_dets, pred_dets, iou_lookup, isvalid_lookup, cx_to_matchable_txs, bg_weight, prioritize, iou_thresh_, pdist_priority, cx_to_ancestors, bg_cidx, ignore_classes, max_dets, truth_reuse_policy'.split(', ')
        lut = globals()
        from xdev.introspect import gen_docstr_from_context, generate_typeannot
        gen_docstr_from_context(keys, lut)

    Note:
        * Preallocating numpy arrays does not help
        * It might be useful to code this critical loop up in C / Cython / Py03
        * Could numba help? (I'm having an issue with cmath)
    """
    # Note:
    import kwarray

    # Keep track of which true items have been used
    true_available = np.ones(len(true_dets), dtype=bool)
    true_nmatches = np.zeros(len(true_dets), dtype=int)

    # sort predictions by descending score
    if 'scores' in pred_dets.data:
        _scores = pred_dets.scores
    else:
        _scores = np.ones(len(pred_dets))

    _pred_sortx = _scores.argsort()[::-1]
    _pred_cxs = pred_dets.class_idxs.take(_pred_sortx, axis=0)
    _pred_scores = _scores.take(_pred_sortx, axis=0)

    if max_dets is not None and np.isfinite(max_dets):
        # for pycocoutils compat, probably not the most efficient way of
        # handling this
        _pred_sortx = _pred_sortx[0:max_dets]
        _pred_cxs = _pred_cxs[0:max_dets]
        _pred_scores = _pred_scores[0:max_dets]

    if ignore_classes is not None:
        # FIXME: does this use the iou threshold correctly?
        # iou_thresh is being used as iooa not iou to determine which
        # pred regions are ignored.
        true_ignore_flags, pred_ignore_flags = _filter_ignore_regions(
            true_dets, pred_dets, ioaa_thresh=iou_thresh_, ignore_classes=ignore_classes)

        # Remove ignored predicted regions from assignment consideration
        _pred_keep_flags = ~pred_ignore_flags[_pred_sortx]
        _pred_sortx = _pred_sortx[_pred_keep_flags]
        _pred_cxs = _pred_cxs[_pred_keep_flags]
        _pred_scores = _pred_scores[_pred_keep_flags]

        # Remove ignored truth regions from assignment consideration
        true_available[true_ignore_flags] = False

    y_pred = []
    y_true = []
    y_score = []
    y_weight = []
    y_iou = []
    y_pxs = []
    y_txs = []

    if not truth_reuse_policy:
        # Original default
        truth_reuse_policy = 'never'
        # TODO: allow an option where multiple predicted boxes are allowed to
        # match the same true box. This is important for cases where true
        # instances are not clearly defined and it is ambiguous if an object
        # should be annotated as one or multiple instances.
        # TODO: also in this case, we need to loop over all available true objects
        # and check if they intersect a predicted object, so we can allow
        # multiple truth objects to match a single predicted object.

    # Greedy assignment. For each predicted detection box.
    # Allow it to match the truth of compatible classes.
    _iter = zip(_pred_sortx, _pred_cxs, _pred_scores)
    # px, pred_cx, score  = next(_iter)
    for px, pred_cx, score in _iter:
        # px, pred_cx, score  = next(_iter)
        # Find compatible truth indices
        true_idxs = cx_to_matchable_txs[pred_cx]

        # Filter out any truth that has already been marked as unavailable
        available = true_available[true_idxs]
        available_true_idxs = true_idxs[available]

        ovmax = -np.inf
        ovidx = None
        weight = bg_weight
        tx = -1  # we will set this to the index of the assigned gt

        if len(available_true_idxs):
            # First grab all candidate available true boxes and lookup precomputed
            # ious between this pred and true_idxs
            cand_true_idxs = available_true_idxs

            if prioritize == 'iou':
                # simply match the true box with the highest iou (that is also
                # considered matchable)

                cand_ious = iou_lookup[px].compress(available)

                if truth_reuse_policy == 'never':
                    # In this case we never will match the same truth box
                    # twice. These are already marked as unavailable so
                    # we just choose the highest iou.
                    ovidx = cand_ious.argmax()
                elif truth_reuse_policy == 'least_used':
                    # In this case we are allowed to reuse boxes, but
                    # we will always choose the least used true match.
                    # boxes are never marked as unavailable in this case
                    # so we use a lex sort to pick the least used but still
                    # valid match with the highest iou
                    cand_nmatches = true_nmatches.compress(available)
                    cand_validmatch = cand_ious > iou_thresh_
                    ovidx = kwarray.arglexmax([
                        cand_ious,
                        -cand_nmatches,
                        cand_validmatch,
                    ])
                else:
                    raise KeyError(truth_reuse_policy)
                ovmax = cand_ious[ovidx]
                if ovmax > iou_thresh_:
                    tx = cand_true_idxs[ovidx]

            elif prioritize == 'correct' or prioritize == 'class':

                if truth_reuse_policy != 'never':
                    raise NotImplementedError(truth_reuse_policy)

                # Choose which (if any) of the overlapping true boxes to match
                # If there are any correct matches above the overlap threshold
                # choose to match that.
                # Flag any available true box that overlaps
                overlap_flags = isvalid_lookup[px][available]

                if overlap_flags.any():
                    cand_ious = iou_lookup[px][available]
                    cand_true_cxs = true_dets.class_idxs[cand_true_idxs]
                    cand_true_idxs = cand_true_idxs[overlap_flags]
                    cand_true_cxs = cand_true_cxs[overlap_flags]
                    cand_ious = cand_ious[overlap_flags]

                    # Choose candidate with highest priority
                    # (prefer finer-grained correct classes over higher overlap,
                    #  but choose highest overlap in a tie).
                    cand_class_priority = pdist_priority[pred_cx][cand_true_cxs]

                    # ovidx = ub.argmax(zip(cand_class_priority, cand_ious))
                    ovidx = kwarray.arglexmax([cand_ious, cand_class_priority])

                    ovmax = cand_ious[ovidx]
                    tx = cand_true_idxs[ovidx]
            else:
                raise KeyError(prioritize)

        if tx > -1:
            # If the prediction matched a true object, mark the assignment
            # as either a true or false positive
            # tx = available_true_idxs[ovidx]
            if truth_reuse_policy == 'never':
                true_available[tx] = False  # mark this true box as unavailable
            true_nmatches[tx] += 1  # indicate the number of times a true box is used.

            if 'weights' in true_dets.data:
                weight = true_dets.weights[tx]
            else:
                weight = 1.0
            true_cx = true_dets.class_idxs[tx]
            # If the prediction is a finer-grained category than the truth
            # change the prediction to match the truth (because it is
            # compatible). This is the key to hierarchical scoring.
            if pred_cx is not None and true_cx in cx_to_ancestors[pred_cx]:
                pred_cx = true_cx

            y_pred.append(pred_cx)
            y_true.append(true_cx)
            y_score.append(score)
            y_weight.append(weight)
            y_iou.append(ovmax)
            y_pxs.append(px)
            y_txs.append(tx)
        else:
            # Assign this prediction to the background
            # Mark this prediction as a false positive
            y_pred.append(pred_cx)
            y_true.append(bg_cidx)
            y_score.append(score)
            y_weight.append(bg_weight)
            y_iou.append(-1)
            y_pxs.append(px)
            y_txs.append(tx)

    # All pred boxes have been assigned to a truth box or the background.
    # Mark available true boxes we failed to predict as false negatives
    bg_px = -1
    unused_txs = np.where(true_nmatches == 0)[0]
    n = len(unused_txs)

    unused_y_true = true_dets.class_idxs[unused_txs].tolist()
    if 'weights' in true_dets.data:
        unused_y_weight = true_dets.weights[unused_txs].tolist()
    else:
        unused_y_weight = [1.0] * n

    y_pred.extend([-1] * n)
    y_true.extend(unused_y_true)
    if USE_NEG_INF:
        y_score.extend([-np.inf] * n)
    else:
        y_score.extend([0] * n)
    y_iou.extend([-1] * n)
    y_weight.extend(unused_y_weight)
    y_pxs.extend([bg_px] * n)
    y_txs.extend(unused_txs.tolist())

    y = {
        'pred': y_pred,
        'true': y_true,
        'score': y_score,
        'weight': y_weight,
        'iou': y_iou,
        'txs': y_txs,  # index into the original true box for this row
        'pxs': y_pxs,  # index into the original pred box for this row
    }
    # val_lens = ub.map_vals(len, y)
    # print('val_lens = {!r}'.format(val_lens))
    # assert ub.allsame(val_lens.values())
    return y


def _fast_pdist_priority(classes, prioritize, _cache={}):
    """
    Custom priority computation. Needs some vetting.

    This is the priority used when deciding which prediction to assign to which
    truth.

    TODO:
        - [ ] Look at absolute difference in sibling entropy when deciding
              whether to go up or down in the tree.
    """
    #  Note: distances to ancestors will be negative and distances
    #  to descendants will be positive. Prefer matching ancestors
    #  over descendants.
    key = ub.hash_data('\n'.join(list(map(str, classes))), hasher='sha1')
    # key = ub.urepr(classes.__json__())
    if key not in _cache:
        # classes = kwcoco.CategoryTree.from_json(classes)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='invalid .* less')
            warnings.filterwarnings('ignore', message='invalid .* greater_equal')
            # Get basic distance between nodes
            pdist = classes.idx_pairwise_distance()
            pdist_priority = np.array(pdist, dtype=np.float32, copy=True)
            if prioritize == 'correct':
                # Prioritizes all ancestors first, and then descendants
                # afterwards, nodes off the direct lineage are ignored.
                valid_vals = pdist_priority[np.isfinite(pdist_priority)]
                maxval = (valid_vals.max() - valid_vals.min()) + 1
                is_ancestor = (pdist_priority >= 0)
                is_descend = (pdist_priority < 0)
                # Prioritize ALL ancestors first
                pdist_priority[is_ancestor] = (
                    2 * maxval - pdist_priority[is_ancestor])
                # Prioritize ALL descendants next
                pdist_priority[is_descend] = (
                    maxval + pdist_priority[is_descend])
                pdist_priority[np.isnan(pdist_priority)] = -np.inf
            elif prioritize == 'class':
                # Prioritizes the exact match first, and then it alternates
                # between ancestors and desendants based on distance to self
                pdist_priority[pdist_priority < -1] += .5
                pdist_priority = np.abs(pdist_priority)
                pdist_priority[np.isnan(pdist_priority)] = np.inf
                pdist_priority = 1 / (pdist_priority + 1)
            else:
                raise KeyError(prioritize)
        _cache[key] = pdist_priority
    pdist_priority = _cache[key]
    return pdist_priority


def _filter_ignore_regions(true_dets, pred_dets, ioaa_thresh=0.5,
                           ignore_classes='ignore'):
    """
    Determine which true and predicted detections should be ignored.

    Args:

        true_dets (Detections)

        pred_dets (Detections)

        ioaa_thresh (float): intersection over other area thresh for ignoring
            a region.

    Returns:
        Tuple[ndarray, ndarray]: flags indicating which true and predicted
            detections should be ignored.

    Example:
        >>> from kwcoco.metrics.assignment import *  # NOQA
        >>> from kwcoco.metrics.assignment import _filter_ignore_regions
        >>> import kwimage
        >>> pred_dets = kwimage.Detections.random(classes=['a', 'b', 'c'])
        >>> true_dets = kwimage.Detections.random(
        >>>     segmentations=True, classes=['a', 'b', 'c', 'ignore'])
        >>> ignore_classes = {'ignore', 'b'}
        >>> ioaa_thresh = 0.5
        >>> print('true_dets = {!r}'.format(true_dets))
        >>> print('pred_dets = {!r}'.format(pred_dets))
        >>> flags1, flags2 = _filter_ignore_regions(
        >>>     true_dets, pred_dets, ioaa_thresh=ioaa_thresh, ignore_classes=ignore_classes)
        >>> print('flags1 = {!r}'.format(flags1))
        >>> print('flags2 = {!r}'.format(flags2))

        >>> flags3, flags4 = _filter_ignore_regions(
        >>>     true_dets, pred_dets, ioaa_thresh=ioaa_thresh,
        >>>     ignore_classes={c.upper() for c in ignore_classes})
        >>> assert np.all(flags1 == flags3)
        >>> assert np.all(flags2 == flags4)
    """
    true_ignore_flags = np.zeros(len(true_dets), dtype=bool)
    pred_ignore_flags = np.zeros(len(pred_dets), dtype=bool)

    if not ub.iterable(ignore_classes):
        ignore_classes = {ignore_classes}

    def _normalize_catname(name, classes):
        if classes is None:
            return name
        if name in classes:
            return name
        for cname in classes:
            if cname.lower() == name.lower():
                return cname
        return name

    ignore_classes = {_normalize_catname(c, true_dets.classes)
                      for c in ignore_classes}

    if true_dets.classes is not None:
        ignore_classes = ignore_classes & set(true_dets.classes)

    # Filter out true detections labeled as "ignore"
    if true_dets.classes is not None and ignore_classes:
        import kwarray
        ignore_cidxs = [true_dets.classes.index(c) for c in ignore_classes]
        true_ignore_flags = kwarray.isect_flags(
            true_dets.class_idxs, ignore_cidxs)

        if np.any(true_ignore_flags) and len(pred_dets):
            ignore_dets = true_dets.compress(true_ignore_flags)

            pred_boxes = pred_dets.data['boxes']
            ignore_boxes = ignore_dets.data['boxes']
            ignore_sseg = ignore_dets.data.get('segmentations', None)

            # Determine which predicted boxes are inside the ignore regions
            # note: using sum over max is deliberate here.
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='invalid .* less')
                warnings.filterwarnings('ignore', message='invalid .* greater_equal')
                warnings.filterwarnings('ignore', message='invalid .* true_divide')
                ignore_overlap = (pred_boxes.isect_area(ignore_boxes) /
                                  pred_boxes.area).clip(0, 1).sum(axis=1)
                ignore_overlap = np.nan_to_num(ignore_overlap)

            ignore_idxs = np.where(ignore_overlap > ioaa_thresh)[0]

            if ignore_sseg is not None:
                from shapely.ops import unary_union
                # If the ignore region has segmentations further refine our
                # estimate of which predictions should be ignored.
                ignore_sseg = ignore_sseg.to_polygon_list()
                box_polys = ignore_boxes.to_polygons()
                ignore_polys = [
                    bp if p is None else p
                    for bp, p in zip(box_polys, ignore_sseg.data)
                ]

                # FIXME: to to_shapely method can break, not sure if this is
                # the right way to fix this

                ignore_regions = []
                for p in ignore_polys:
                    try:
                        ignore_regions.append(p.to_shapely())
                    except Exception:
                        pass
                # ignore_regions = [p.to_shapely() for p in ignore_polys]
                ignore_region = unary_union(ignore_regions).buffer(0)

                cand_pred = pred_boxes.take(ignore_idxs)

                # Refine overlap estimates
                cand_regions = cand_pred.to_shapely()
                for idx, pred_region in zip(ignore_idxs, cand_regions):
                    try:
                        isect = ignore_region.intersection(pred_region)
                        overlap = (isect.area / pred_region.area)
                        ignore_overlap[idx] = overlap
                    except Exception as ex:
                        warnings.warn('ex = {!r}'.format(ex))
            pred_ignore_flags = ignore_overlap > ioaa_thresh
    return true_ignore_flags, pred_ignore_flags

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/kwcoco/metrics/assignment.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
