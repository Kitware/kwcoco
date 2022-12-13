

def bench_pop_info():
    from kwcoco.metrics.confusion_vectors import BinaryConfusionVectors  # NOQA
    import numpy  as np
    import kwarray
    classes = ['class1', 'class2']
    rng = kwarray.ensure_rng(0)
    true = (rng.rand(16384) * len(classes)).astype(int)
    probs = rng.rand(len(true), len(classes))
    import kwcoco
    cfsn_vecs = kwcoco.metrics.ConfusionVectors.from_arrays(true=true, probs=probs, classes=classes)
    binvecs = cfsn_vecs.binarize_ovr()['class1']
    info = binvecs.measures()

    tp = np.array(info['tp_count'])
    tn = np.array(info['tn_count'])
    fp = np.array(info['fp_count'])
    fn = np.array(info['fn_count'])
    ppv = np.array(info['ppv'])
    tpr = np.array(info['tpr'])
    tnr = np.array(info['tnr'])
    npv = np.array(info['npv'])
    fpr = np.array(info['fpr'])
    real_pos = fn + tp  # number of real positives
    p_denom = real_pos.copy()
    p_denom[p_denom == 0] = 1
    fnr = fn / p_denom  # miss-rate
    fdr  = 1 - ppv  # false discovery rate
    fmr  = 1 - npv  # false ommision rate (for)

    import timerit
    ti = timerit.Timerit(10000, bestof=10, verbose=2)
    for timer in ti.reset('acc-v1'):
        with timer:
            acc = (tp + tn) / (tp + tn + fp + fn)

    for timer in ti.reset('acc-v2'):
        with timer:
            tp_add_tn = tp + tn
            acc = tp_add_tn / (tp_add_tn + fp + fn)

    for timer in ti.reset('acc-v3'):
        with timer:
            tp_add_tn = tp + tn
            fp_add_fn = fp + fn
            acc = tp_add_tn / (tp_add_tn + fp_add_fn)

    import numexpr
    for timer in ti.reset('acc-numexpr'):
        with timer:
            numexpr.evaluate('(tp + tn) / (tp + tn + fp + fn)')

    import timerit
    ti = timerit.Timerit(10000, bestof=10, verbose=2)
    for timer in ti.reset('mcc'):
        mcc_numer = (tp * tn) - (fp * fn)
        mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc_denom[np.isnan(mcc_denom) | (mcc_denom == 0)] = 1
        mcc = mcc_numer / mcc_denom

    import timerit
    ti = timerit.Timerit(10000, bestof=10, verbose=2)
    for timer in ti.reset('mcc'):
        tp_add_tn = tp + tn
        fp_add_fn = fp + fn
        tn_add_fp = (tn + fp)
        mcc_numer = (tp * tn) - (fp * fn)
        mcc_denom = np.sqrt((tp + fp) * (tp + fn) * tn_add_fp * (tn + fn))
        mcc_denom[np.isnan(mcc_denom) | (mcc_denom == 0)] = 1
        mcc = mcc_numer / mcc_denom

    import timerit
    ti = timerit.Timerit(10000, bestof=10, verbose=2)
    for timer in ti.reset('mcc'):
        mcc = np.sqrt(ppv * tpr * tnr * npv) - np.sqrt(fdr * fnr * fpr * fmr)

    import timerit
    ti = timerit.Timerit(10000, bestof=10, verbose=2)
    for timer in ti.reset('mcc'):
        p1 = ppv * tpr * tnr * npv
        p2 = fdr * fnr * fpr * fmr
        np.sqrt(p1, out=p1)
        np.sqrt(p2, out=p2)
        mcc = np.subtract(p1, p2, out=p1)

    import numexpr
    for timer in ti.reset('acc-numexpr'):
        with timer:
            numexpr.evaluate('sqrt(ppv * tpr * tnr * npv) - sqrt(fdr * fnr * fpr * fmr)')


