import ubelt as ub
import numpy as np


def perterb_coco(coco_dset, **kwargs):
    """
    Perterbs a coco dataset

    Example:
        >>> # xdoctest: +REQUIRES(module:ndsampler)
        >>> import kwcoco
        >>> true_dset = kwcoco.CocoDataset.demo('shapes8')
        >>> kwargs = {
        >>>     'box_noise': 0.5,
        >>>     'n_fp': 3,
        >>> }
        >>> pred_dset = perterb_coco(true_dset, **kwargs)

    """
    import kwimage
    import kwarray
    # Parse kwargs
    rng = kwarray.ensure_rng(kwargs.get('rng', 0))

    box_noise = kwargs.get('box_noise', 0)
    cls_noise = kwargs.get('cls_noise', 0)

    null_pred = kwargs.get('null_pred', False)
    with_probs = kwargs.get('with_probs', False)

    # specify an amount of overlap between true and false scores
    score_noise = kwargs.get('score_noise', 0.2)

    # Build random variables
    from kwarray import distributions
    DiscreteUniform = distributions.DiscreteUniform.seeded(rng=rng)
    def _parse_arg(key, default):
        value = kwargs.get(key, default)
        try:
            low, high = value
            return (low, high + 1)
        except Exception:
            return (0, value + 1)
    n_fp_RV = DiscreteUniform(*_parse_arg('n_fp', 0))
    n_fn_RV = DiscreteUniform(*_parse_arg('n_fn', 0))

    box_noise_RV = distributions.Normal(0, box_noise, rng=rng)
    cls_noise_RV = distributions.Bernoulli(cls_noise, rng=rng)

    # the values of true and false scores starts off with no overlap and
    # the overlap increases as the score noise increases.
    def _interp(v1, v2, alpha):
        return v1 * alpha + (1 - alpha) * v2
    mid = 0.5
    # true_high = 2.0
    true_high = 1.0
    true_low   = _interp(0, mid, score_noise)
    false_high = _interp(true_high, mid - 1e-3, score_noise)
    true_mean  = _interp(0.5, .8, score_noise)
    false_mean = _interp(0.5, .2, score_noise)

    true_score_RV = distributions.TruncNormal(
        mean=true_mean, std=.5, low=true_low, high=true_high, rng=rng)
    false_score_RV = distributions.TruncNormal(
        mean=false_mean, std=.5, low=0, high=false_high, rng=rng)

    # Create the category hierarcy
    classes = coco_dset.object_categories()

    frgnd_cx_RV = distributions.DiscreteUniform(0, len(classes), rng=rng)

    new_dset = coco_dset.copy()
    remove_aids = []
    false_anns = []

    for gid in coco_dset.imgs.keys():
        # Sample random variables
        n_fp_ = n_fp_RV()
        n_fn_ = n_fn_RV()

        true_annots = coco_dset.annots(gid=gid)
        aids = true_annots.aids
        for aid in aids:
            # Perterb box coordinates
            ann = new_dset.anns[aid]
            ann['bbox'] = (np.array(ann['bbox']) + box_noise_RV(4)).tolist()

            ann['score'] = true_score_RV()

            if cls_noise_RV():
                # Perterb class predictions
                ann['category_id'] = classes.idx_to_id[frgnd_cx_RV()]

        # Drop true positive boxes
        if n_fn_:
            import kwarray
            drop_idxs = kwarray.shuffle(np.arange(len(aids)), rng=rng)[0:n_fn_]
            remove_aids.extend(list(ub.take(aids, drop_idxs)))

        # Add false positive boxes
        if n_fp_:
            try:
                img = coco_dset.imgs[gid]
                scale = (img['width'], img['height'])
            except KeyError:
                scale = 100
            false_boxes = kwimage.Boxes.random(num=n_fp_, scale=scale,
                                               rng=rng, format='cxywh')
            false_cxs = frgnd_cx_RV(n_fp_)
            false_scores = false_score_RV(n_fp_)
            false_dets = kwimage.Detections(
                boxes=false_boxes,
                class_idxs=false_cxs,
                scores=false_scores,
                classes=classes,
            )
            for ann in list(false_dets.to_coco('new')):
                ann['category_id'] = classes.node_to_id[ann.pop('category_name')]
                ann['image_id'] = gid
                false_anns.append(ann)

        # Transform the scores for the assigned class into a predicted
        # probability for each class. (Currently a bit hacky).
        # class_probs = _demo_construct_probs(
        #     pred_cxs, pred_scores, classes, rng,
        #     hacked=kwargs.get('hacked', 0))

        # Hack in the probs
        if with_probs:
            raise NotImplementedError

        if null_pred:
            raise NotImplementedError

    new_dset.remove_annotations(remove_aids)
    for ann in false_anns:
        new_dset.add_annotation(**ann)
    return new_dset


def _demo_construct_probs(pred_cxs, pred_scores, classes, rng, hacked=1):
    """
    Constructs random probabilities for demo data
    """
    # Setup probs such that the assigned class receives a probability
    # equal-(ish) to the assigned score.
    # Its a bit tricky to setup hierarchical probs such that we get the
    # scores in the right place. We punt and just make probs
    # conditional. The right thing to do would be to do this, and then
    # perterb ancestor categories such that the probability evenetually
    # converges on the right value at that specific classes depth.
    import torch

    # Ensure probs
    pred_scores2 = pred_scores.clip(0, 1.0)

    class_energy = rng.rand(len(pred_scores2), len(classes)).astype(np.float32)
    for p, x, s in zip(class_energy, pred_cxs, pred_scores2):
        p[x] = s

    if hacked:
        # HACK! All that nice work we did is too slow for doctests
        return class_energy

    class_energy = torch.Tensor(class_energy)
    cond_logprobs = classes.conditional_log_softmax(class_energy, dim=1)
    cond_probs = torch.exp(cond_logprobs).numpy()

    # I was having a difficult time getting this right, so an
    # inefficient per-item non-vectorized implementation it is.
    # Note: that this implementation takes 70% of the time in this function
    # and is a bottleneck for the doctests. A vectorized implementation would
    # be nice.
    idx_to_ancestor_idxs = classes.idx_to_ancestor_idxs()
    idx_to_groups = {idx: group for group in classes.idx_groups for idx in group}

    def set_conditional_score(row, cx, score, idx_to_groups):
        group_cxs = np.array(idx_to_groups[cx])
        flags = group_cxs == cx
        group_row = row[group_cxs]
        # Ensure that that heriarchical probs sum to 1
        current = group_row[~flags]
        other = current * (1 - score) / current.sum()
        other = np.nan_to_num(other)
        group_row[~flags] = other
        group_row[flags] = score
        row[group_cxs] = group_row

    for row, cx, score in zip(cond_probs, pred_cxs, pred_scores2):
        set_conditional_score(row, cx, score, idx_to_groups)
        for ancestor_cx in idx_to_ancestor_idxs[cx]:
            if ancestor_cx != cx:
                # Hack all parent probs to 1.0 so conditional probs
                # turn into real probs.
                set_conditional_score(row, ancestor_cx, 1.0, idx_to_groups)
                # TODO: could add a fudge factor here so the
                # conditional prob is higher than score, but parent
                # probs are less than 1.0

                # TODO: could also maximize entropy of descendant nodes
                # so classes.decision2 would stop at this node

    # For each level the conditional probs must sum to 1
    if cond_probs.size > 0:
        for idxs in classes.idx_groups:
            level = cond_probs[:, idxs]
            totals = level.sum(axis=1)
            assert level.shape[1] == 1 or np.allclose(totals, 1.0), str(level) + ' : ' + str(totals)

    cond_logprobs = torch.Tensor(cond_probs).log()
    class_probs = classes._apply_logprob_chain_rule(cond_logprobs, dim=1).exp().numpy()
    class_probs = class_probs.reshape(-1, len(classes))
    # print([p[x] for p, x in zip(class_probs, pred_cxs)])
    # print(pred_scores2)
    return class_probs
