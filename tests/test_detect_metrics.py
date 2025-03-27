def test_with_one_image_and_no_truth():
    """
    Test case where there is no truth in an image.
    """
    from kwcoco.metrics.detect_metrics import DetectionMetrics
    import kwimage
    import kwarray

    rng = kwarray.ensure_rng(1428142639)

    classes = ['class_1', 'class_2', 'class_3', 'class_4', 'class_5']

    # This test case has only one image with no truth
    true_dets = kwimage.Detections.random(0, classes=classes, rng=rng)

    # But we do make predictions
    pred_dets = kwimage.Detections.random(3, classes=classes, rng=rng)

    dmet = DetectionMetrics()
    dmet.add_predictions(pred_dets, imgname='image1')
    dmet.add_truth(true_dets, imgname='image1')

    # Use our internal scoring to compute per-class and total scores
    kwcoco_scores = dmet.score_kwcoco()
    print(kwcoco_scores)

    # Ultimately the scores are derived from confusion vectors, which are
    # entirely transparent in our scoring system. These are nicely viewed
    # in a pandas table:
    cfsn_vecs = dmet.confusion_vectors()
    cfsn_table = cfsn_vecs.data.pandas()
    # For efficiency columns store index based information. These are
    # documented, but as a reminder the columns are:
    # pred - the predicted index (-1 for not assigned to a truth object)
    # true - the truth index (-1 for not assigned to a predicted object)
    # score - the score of the prediction
    # weight - the weight of the assigned pair
    # txs - the truth index of the detection within the image this came from
    # pxs - the predicted index of the detection within the image this came from
    # gid - the image id that this came from
    print(cfsn_table)

    # The details of how confusion vectors are translated into scores
    # are determined by
    # 1) how they are "binarized" into a classification problem and
    # 2) how scores are measured from that binary classification problem

    # The first way of binarizing confusion vectors is by ignoring class, this
    # is useful for determening if can detect anything at all. Note:
    # the weights column in the confusion vectors can be assigned to modify the
    # impact each sample has on the final score.
    binvecs_classless = cfsn_vecs.binarize_classless()
    classless_scores = binvecs_classless.measures(stabalize_thresh=0)
    print(classless_scores)

    # The other way to binairze is on a per-class basis, where this generates
    # "C" binary confusion vectors, one for each class type.
    binvecs_ovr = cfsn_vecs.binarize_ovr()
    perclass_scores = binvecs_ovr.measures(stabalize_thresh=0)
    print(perclass_scores)

    # Note, when there is no truth the "stabalization threshold" has a big
    # impact This inserts dummy points into the curve if there are fewer than
    # `thresh` (default 7) data points. This prevents divide by zeros, but
    # also means that when you have no examples of something and you
    # correctly predict no examples of it you get a perfect score. It also
    # means you get a non-zero score if you have no examples of something and
    # you predict some examples of it.

    # The following code inspects behavior at different stabalization
    # thresholds. There are no tests, as we may change behavior in the future.
    rows = []
    for stabalize_thresh in [0, 1, 3, 5, 7, 10, 100]:
        classless_scores = binvecs_classless.measures(stabalize_thresh=stabalize_thresh)
        ovr_scores = binvecs_ovr.measures(stabalize_thresh=stabalize_thresh)
        row = {
            'stabalize_thresh': stabalize_thresh,
        }
        metric_names = ['ap', 'auc', 'max_f1']
        catname = 'classless'
        for metric in metric_names:
            row[f'{catname}_{metric}'] = classless_scores[metric]
        row['mAP'] = ovr_scores['mAP']
        row['mAUC'] = ovr_scores['mAUC']
        for catname, measures in ovr_scores['perclass'].items():
            for metric in metric_names:
                row[f'{catname}_{metric}'] = measures[metric]
        rows.append(row)
    import pandas as pd
    df = pd.DataFrame(rows)
    print(df.T.to_string())


def test_with_multiple_images_and_some_have_no_truth():
    """
    Test case where there is no truth in an image.
    """
    from kwcoco.metrics.detect_metrics import DetectionMetrics
    import kwimage
    import kwarray

    rng = kwarray.ensure_rng(4329012312)

    dmet = DetectionMetrics()

    p_keep = 0.5
    max_fp = 10
    max_tp = 10
    scale_distri = kwarray.distributions.Normal(mean=1, std=0.1, rng=rng)

    # Generate multiple truth / predictions
    num_images = 10
    for image_idx in range(num_images):

        # For every even numbered image, force it to be empty, otherwise
        # use a random number of truth objects.
        num_truth = 0 if image_idx % 2 == 0 else rng.randint(max_tp)
        true_dets = kwimage.Detections.random(num_truth)

        # Generate random prediction by randomly droping truth, perterbing
        # them, and adding false positives
        keep_flags = (rng.rand(len(true_dets)) > p_keep)
        transform = kwimage.Affine.scale(scale_distri.sample())
        true_subset = true_dets.compress(keep_flags).warp(transform)
        num_fp = rng.randint(max_fp)
        false_positive = kwimage.Detections.random(num_fp)
        pred_dets = kwimage.Detections.concatenate([true_subset, false_positive])

        image_name = f'image_{image_idx:04d}'

        dmet.add_predictions(pred_dets, imgname=image_name)
        dmet.add_truth(true_dets, imgname=image_name)

    scores = dmet.score_kwcoco()
    print(scores)
