import ubelt as ub
import numpy as np


def perterb_coco(coco_dset, **kwargs):
    """
    Perterbs a coco dataset

    Args:
        rng (int, default=0):
        box_noise (int, default=0):
        cls_noise (int, default=0):
        null_pred (bool, default=False):
        with_probs (bool, default=False):
        score_noise (float, default=0.2):
        hacked (int, default=1):

    Example:
        >>> from kwcoco.demo.perterb import *  # NOQA
        >>> from kwcoco.demo.perterb import _demo_construct_probs
        >>> import kwcoco
        >>> coco_dset = true_dset = kwcoco.CocoDataset.demo('shapes8')
        >>> kwargs = {
        >>>     'box_noise': 0.5,
        >>>     'n_fp': 3,
        >>>     'with_probs': 1,
        >>>     'with_heatmaps': 1,
        >>> }
        >>> pred_dset = perterb_coco(true_dset, **kwargs)
        >>> pred_dset._check_json_serializable()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> gid = 1
        >>> canvas = true_dset.delayed_load(gid).finalize()
        >>> canvas = true_dset.annots(gid=gid).detections.draw_on(canvas, color='green')
        >>> canvas = pred_dset.annots(gid=gid).detections.draw_on(canvas, color='blue')
        >>> kwplot.imshow(canvas)

    Ignore:
        import xdev
        from kwcoco.demo.perterb import perterb_coco  # NOQA
        defaultkw = xdev.get_func_kwargs(perterb_coco)
        for k, v in defaultkw.items():
            desc = ''
            print('{} ({}, default={}): {}'.format(k, type(v).__name__, v, desc))

    """
    import kwimage
    import kwarray
    # Parse kwargs
    rng = kwarray.ensure_rng(kwargs.get('rng', 0))

    box_noise = kwargs.get('box_noise', 0)
    cls_noise = kwargs.get('cls_noise', 0)

    null_pred = kwargs.get('null_pred', False)
    with_probs = kwargs.get('with_probs', False)
    with_heatmaps = kwargs.get('with_heatmaps', False)

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
            return (value, value + 1)
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
    false_low = 0.0
    true_low   = _interp(0, mid, score_noise)
    false_high = _interp(true_high, mid - 1e-3, score_noise)
    true_mean  = _interp(0.5, .8, score_noise)
    false_mean = _interp(0.5, .2, score_noise)

    true_score_RV = distributions.TruncNormal(
        mean=true_mean, std=.5, low=true_low, high=true_high, rng=rng)
    false_score_RV = distributions.TruncNormal(
        mean=false_mean, std=.5, low=false_low, high=false_high, rng=rng)

    # Create the category hierarcy
    classes = coco_dset.object_categories()

    cids = coco_dset.cats.keys()
    cidxs = [classes.id_to_idx[c] for c in cids]

    frgnd_cx_RV = distributions.CategoryUniform(cidxs, rng=rng)

    new_dset = coco_dset.copy()
    remove_aids = []
    false_anns = []

    index_invalidated = False

    for gid in coco_dset.imgs.keys():
        # Sample random variables
        n_fp_ = n_fp_RV()
        n_fn_ = n_fn_RV()

        true_annots = coco_dset.annots(gid=gid)
        aids = true_annots.aids
        for aid in aids:
            # Perterb box coordinates
            ann = new_dset.anns[aid]

            old_bbox = np.array(ann['bbox'])
            new_bbox = (old_bbox + box_noise_RV(4)).tolist()

            new_x, new_y, new_w, new_h = new_bbox
            allow_neg_boxes = 0
            if not allow_neg_boxes:
                new_w = max(new_w, 0)
                new_h = max(new_h, 0)

            old_cxywh = kwimage.Boxes([old_bbox], 'xywh').to_cxywh()
            new_cxywh = kwimage.Boxes([new_bbox], 'xywh').to_cxywh()

            old_sseg = kwimage.Segmentation.coerce(ann['segmentation'])

            # Compute the transform of the box so we can modify the
            # other attributes (TODO: we could use a random affine transform
            # for everything)
            offset = new_cxywh.data[0, 0:2] - old_cxywh.data[0, 0:2]
            scale = new_cxywh.data[0, 2:4] / old_cxywh.data[0, 2:4]
            old_to_new = kwimage.Affine.coerce(offset=offset, scale=scale)
            new_sseg = old_sseg.warp(old_to_new)

            # Overwrite the data
            ann['segmentation'] = new_sseg.to_coco(style='new')
            ann['bbox'] = [new_x, new_y, new_w, new_h]
            ann['score'] = float(true_score_RV(1)[0])

            if cls_noise_RV():
                # Perterb class predictions
                ann['category_id'] = classes.idx_to_id[frgnd_cx_RV()]
                index_invalidated = True

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
                x, y, w, h = ann['bbox']
                sseg = kwimage.MultiPolygon.random().scale((w, h)).translate((x, y))
                ann['segmentation'] = sseg
                false_anns.append(ann)

        if null_pred:
            raise NotImplementedError

    if index_invalidated:
        new_dset.index.clear()
        new_dset._build_index()
    new_dset.remove_annotations(remove_aids)

    for ann in false_anns:
        new_dset.add_annotation(**ann)

    # Hack in the probs
    if with_probs:
        annots = new_dset.annots()
        pred_cids = annots.lookup('category_id')
        pred_cxs = np.array([classes.id_to_idx[cid] for cid in pred_cids])
        pred_scores = np.array(annots.lookup('score'))
        # Transform the scores for the assigned class into a predicted
        # probability for each class. (Currently a bit hacky).
        pred_probs = _demo_construct_probs(
            pred_cxs, pred_scores, classes, rng,
            hacked=kwargs.get('hacked', 1))

        for aid, prob in zip(annots.aids, pred_probs):
            new_dset.anns[aid]['prob'] = prob.tolist()

    # Hack in the per-class heatmaps
    if with_heatmaps:
        for gid in new_dset.images():
            annots = new_dset.annots(gid=gid)
            img = new_dset.index.imgs[gid]
            w = img['width']
            h = img['height']
            c = len(classes)
            # Build up basic prob masks
            heatmaps = np.zeros((c, h, w), dtype=np.float32)
            for ann in annots.objs:
                poly = kwimage.Segmentation.coerce(ann['segmentation']).to_multi_polygon()
                cid = ann['category_id']
                cidx = classes.id_to_idx[cid]
                probs = heatmaps[cidx]
                poly.fill(probs, 1)

            chan_datas = []
            # Add lots of noise to the data
            dims = (h, w)
            for cidx in range(len(classes)):
                chan_data = heatmaps[cidx]
                chan_data += (rng.randn(*dims) * 0.1)
                chan_data + chan_data.clip(0, 1)
                chan_data = kwimage.gaussian_blur(chan_data, sigma=1.2)
                chan_data = chan_data.clip(0, 1)
                mask = rng.randn(*dims)
                chan_data = chan_data * ((kwimage.fourier_mask(chan_data, mask)[..., 0]) + .5)
                chan_data += (rng.randn(*dims) * 0.1)
                chan_data = chan_data.clip(0, 1)
                chan_datas.append(chan_data)
            hwc_probs = np.stack(chan_datas, axis=2)

            coco_img = new_dset.coco_image(gid)
            chanspec = '|'.join(list(classes))
            # heatmap_fpath = dummy_heatmap_dpath / 'dummy_heatmap_{}.tif'.format(img['id'])
            # kwimage.imwrite(heatmap_fpath, hwc_probs, backend='gdal', compress='NONE',
            #                 blocksize=96)
            coco_img.add_auxiliary_item(
                # file_name=str(heatmap_fpath),
                imdata=hwc_probs,
                channels=chanspec,
            )

    return new_dset


def _demo_construct_probs(pred_cxs, pred_scores, classes, rng, hacked=1):
    """
    Constructs random probabilities for demo data

    Example:
        >>> import kwcoco
        >>> import kwarray
        >>> rng = kwarray.ensure_rng(0)
        >>> classes = kwcoco.CategoryTree.coerce(10)
        >>> hacked = 1
        >>> pred_cxs = rng.randint(0, 10, 10)
        >>> pred_scores = rng.rand(10)
        >>> probs = _demo_construct_probs(pred_cxs, pred_scores, classes, rng, hacked)
        >>> probs.sum(axis=1)
    """
    # Setup probs such that the assigned class receives a probability
    # equal-(ish) to the assigned score.
    # Its a bit tricky to setup hierarchical probs such that we get the
    # scores in the right place. We punt and just make probs
    # conditional. The right thing to do would be to do this, and then
    # perterb ancestor categories such that the probability evenetually
    # converges on the right value at that specific classes depth.
    # import torch

    # Ensure probs
    pred_scores2 = pred_scores.clip(0, 1.0)

    class_energy = rng.rand(len(pred_scores2), len(classes)).astype(np.float32)

    is_mutex = 0
    if hasattr(classes, 'is_mutex') and classes.is_mutex():
        is_mutex = 1

    if isinstance(classes, (list, tuple)):
        is_mutex = 1

    if is_mutex:
        class_energy = class_energy / class_energy.sum(axis=1, keepdims=True)
        for p, x, s in zip(class_energy, pred_cxs, pred_scores2):
            # ensure sum to 1 when classes are known mutex
            rest = p[0:x].sum() + p[x + 1:].sum()
            if s <= 1:
                p[:] = p * ((1 - s) / rest)
            p[x] = s
    else:
        for p, x, s in zip(class_energy, pred_cxs, pred_scores2):
            p[x] = s

    if hacked:
        # HACK! All that nice work we did is too slow for doctests
        return class_energy

    raise AssertionError('must be hacked')

    # class_energy = torch.Tensor(class_energy)
    # cond_logprobs = classes.conditional_log_softmax(class_energy, dim=1)
    # cond_probs = torch.exp(cond_logprobs).numpy()

    # # I was having a difficult time getting this right, so an
    # # inefficient per-item non-vectorized implementation it is.
    # # Note: that this implementation takes 70% of the time in this function
    # # and is a bottleneck for the doctests. A vectorized implementation would
    # # be nice.
    # idx_to_ancestor_idxs = classes.idx_to_ancestor_idxs()
    # idx_to_groups = {idx: group for group in classes.idx_groups for idx in group}

    # def set_conditional_score(row, cx, score, idx_to_groups):
    #     group_cxs = np.array(idx_to_groups[cx])
    #     flags = group_cxs == cx
    #     group_row = row[group_cxs]
    #     # Ensure that that heriarchical probs sum to 1
    #     current = group_row[~flags]
    #     other = current * (1 - score) / current.sum()
    #     other = np.nan_to_num(other)
    #     group_row[~flags] = other
    #     group_row[flags] = score
    #     row[group_cxs] = group_row

    # for row, cx, score in zip(cond_probs, pred_cxs, pred_scores2):
    #     set_conditional_score(row, cx, score, idx_to_groups)
    #     for ancestor_cx in idx_to_ancestor_idxs[cx]:
    #         if ancestor_cx != cx:
    #             # Hack all parent probs to 1.0 so conditional probs
    #             # turn into real probs.
    #             set_conditional_score(row, ancestor_cx, 1.0, idx_to_groups)
    #             # TODO: could add a fudge factor here so the
    #             # conditional prob is higher than score, but parent
    #             # probs are less than 1.0

    #             # TODO: could also maximize entropy of descendant nodes
    #             # so classes.decision2 would stop at this node

    # # For each level the conditional probs must sum to 1
    # if cond_probs.size > 0:
    #     for idxs in classes.idx_groups:
    #         level = cond_probs[:, idxs]
    #         totals = level.sum(axis=1)
    #         assert level.shape[1] == 1 or np.allclose(totals, 1.0), str(level) + ' : ' + str(totals)

    # cond_logprobs = torch.Tensor(cond_probs).log()
    # class_probs = classes._apply_logprob_chain_rule(cond_logprobs, dim=1).exp().numpy()
    # class_probs = class_probs.reshape(-1, len(classes))
    # # print([p[x] for p, x in zip(class_probs, pred_cxs)])
    # # print(pred_scores2)
    # return class_probs
