# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
import numpy as np
import ubelt as ub
import os


# TODO : should we use locale or a ASCII_ONLY environ?
# For now lets just expose this environ, even if it is not standard
# ideally we would find a standard environ like NO_COLOR to accomplish this
# https://stackoverflow.com/questions/3425294/how-to-detect-the-os-default-language-in-python
ASCII_ONLY = os.environ.get('ASCII_ONLY', '')


def classification_report(y_true, y_pred, target_names=None,
                          sample_weight=None, verbose=False,
                          remove_unsupported=False, log=None,
                          ascii_only=False):
    """
    Computes a classification report which is a collection of various metrics
    commonly used to evaulate classification quality. This can handle binary
    and multiclass settings.

    Note that this function does not accept probabilities or scores and must
    instead act on final decisions. See ovr_classification_report for a
    probability based report function using a one-vs-rest strategy.

    This emulates the bm(cm) Matlab script written by David Powers that is used
    for computing bookmaker, markedness, and various other scores.

    References:
        https://csem.flinders.edu.au/research/techreps/SIE07001.pdf
        https://www.mathworks.com/matlabcentral/fileexchange/5648-bm-cm-?requestedDomain=www.mathworks.com
        Jurman, Riccadonna, Furlanello, (2012). A Comparison of MCC and CEN
            Error Measures in MultiClass Prediction

    Args:
        y_true (array): true labels for each item
        y_pred (array): predicted labels for each item
        target_names (List): mapping from label to category name
        sample_weight (ndarray): weight for each item
        verbose (False): print if True
        log (callable): print or logging function
        remove_unsupported (bool, default=False): removes categories that have
            no support.
        ascii_only (bool, default=False): if True dont use unicode characters.
            if the environ ASCII_ONLY is present this is forced to True and
            cannot be undone.

    Example:
        >>> # xdoctest: +IGNORE_WANT
        >>> # xdoctest: +REQUIRES(module:sklearn)
        >>> # xdoctest: +REQUIRES(module:pandas)
        >>> y_true = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]
        >>> y_pred = [1, 2, 1, 3, 1, 2, 2, 3, 2, 2, 3, 3, 2, 3, 3, 3, 1, 3]
        >>> target_names = None
        >>> sample_weight = None
        >>> report = classification_report(y_true, y_pred, verbose=0, ascii_only=1)
        >>> print(report['confusion'])
        pred  1  2  3  Σr
        real
        1     3  1  1   5
        2     0  4  1   5
        3     1  1  6   8
        Σp    4  6  8  18
        >>> print(report['metrics'])
        metric    precision  recall    fpr  markedness  bookmaker    mcc  support
        class
        1            0.7500  0.6000 0.0769      0.6071     0.5231 0.5635        5
        2            0.6667  0.8000 0.1538      0.5833     0.6462 0.6139        5
        3            0.7500  0.7500 0.2000      0.5500     0.5500 0.5500        8
        combined     0.7269  0.7222 0.1530      0.5751     0.5761 0.5758       18

    Example:
        >>> # xdoctest: +IGNORE_WANT
        >>> # xdoctest: +REQUIRES(module:sklearn)
        >>> # xdoctest: +REQUIRES(module:pandas)
        >>> from kwcoco.metrics.clf_report import *  # NOQA
        >>> y_true = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]
        >>> y_pred = [1, 2, 1, 3, 1, 2, 2, 3, 2, 2, 3, 3, 2, 3, 3, 3, 1, 3]
        >>> target_names = None
        >>> sample_weight = None
        >>> logs = []
        >>> report = classification_report(y_true, y_pred, verbose=1, ascii_only=True, log=logs.append)
        >>> print('\n'.join(logs))

    Ignore:
        >>> size = 100
        >>> rng = np.random.RandomState(0)
        >>> p_classes = np.array([.90, .05, .05][0:2])
        >>> p_classes = p_classes / p_classes.sum()
        >>> p_wrong   = np.array([.03, .01, .02][0:2])
        >>> y_true = testdata_ytrue(p_classes, p_wrong, size, rng)
        >>> rs = []
        >>> for x in range(17):
        >>>     p_wrong += .05
        >>>     y_pred = testdata_ypred(y_true, p_wrong, rng)
        >>>     report = classification_report(y_true, y_pred, verbose='hack')
        >>>     rs.append(report)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> import pandas as pd
        >>> df = pd.DataFrame(rs).drop(['raw'], axis=1)
        >>> delta = df.subtract(df['target'], axis=0)
        >>> sqrd_error = np.sqrt((delta ** 2).sum(axis=0))
        >>> print('Error')
        >>> print(sqrd_error.sort_values())
        >>> ys = df.to_dict(orient='list')
        >>> kwplot.multi_plot(ydata_list=ys)
    """
    import pandas as pd
    import scipy as sp  # NOQA
    import sklearn.metrics
    from sklearn.preprocessing import LabelEncoder

    if ASCII_ONLY:
        ascii_only = True

    if verbose or log:
        if log is None:
            log = print

    if target_names is None:
        unique_labels = np.unique(np.hstack([y_true, y_pred]))
        if len(unique_labels) == 1 and (unique_labels[0] == 0 or unique_labels[0] == 1):
            target_names = np.array([False, True])
            y_true_ = y_true
            y_pred_ = y_pred
        else:
            lb = LabelEncoder()
            lb.fit(unique_labels)
            y_true_ = lb.transform(y_true)
            y_pred_ = lb.transform(y_pred)
            target_names = lb.classes_
    else:
        y_true_ = y_true
        y_pred_ = y_pred

    # Real data is on the rows,
    # Pred data is on the cols.

    cm = sklearn.metrics.confusion_matrix(
        y_true_, y_pred_, sample_weight=sample_weight,
        labels=np.arange(len(target_names)))
    confusion = cm  # NOQA

    k = len(cm)  # number of classes
    N = cm.sum()  # number of examples

    real_total = cm.sum(axis=1)
    pred_total = cm.sum(axis=0)

    # the number of "positive" cases **per class**
    n_pos = real_total  # NOQA
    # the number of times a class was predicted.
    n_neg = N - n_pos  # NOQA

    # number of true positives per class
    n_tps = np.diag(cm)
    # number of true negatives per class
    n_fps = (cm - np.diagflat(np.diag(cm))).sum(axis=0)

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='invalid .* true_divide')
        warnings.filterwarnings('ignore', message='divide by zero')
        warnings.filterwarnings('ignore', message='invalid value encountered in double_scalars')

        tprs = n_tps / real_total  # true pos rate (recall)
        tpas = n_tps / pred_total  # true pos accuracy (precision)

        unused = (real_total + pred_total) == 0

        fprs = n_fps / n_neg  # false pose rate
        fprs[unused] = np.nan

        rprob = real_total / N
        pprob = pred_total / N

        # if len(cm) == 2:
        #     [[A, B],
        #      [C, D]] = cm
        #     (A * D - B * C) / np.sqrt((A + C) * (B + D) * (A + B) * (C + D))

        # bookmaker is analogous to recall, but unbiased by class frequency
        rprob_mat = np.tile(rprob, [k, 1]).T - (1 - np.eye(k))
        bmcm = cm.T / rprob_mat
        bms = np.sum(bmcm.T, axis=0) / N

        # markedness is analogous to precision, but unbiased by class frequency
        pprob_mat = np.tile(pprob, [k, 1]).T - (1 - np.eye(k))
        mkcm = cm / pprob_mat
        mks = np.sum(mkcm.T, axis=0) / N

        mccs = np.sign(bms) * np.sqrt(np.abs(bms * mks))

        import scipy
        # https://en.wikipedia.org/wiki/F1_score
        # f1_scores = scipy.stats.hmean(np.hstack([
        #     tpas[:, None],
        #     tprs[:, None]
        # ]), axis=1)
        f1_scores = 2 * (tpas * tprs) / (tpas + tprs)
        g1_scores = scipy.stats.gmean(np.hstack([
            tpas[:, None],
            tprs[:, None]
        ]), axis=1)

    perclass_data = ub.odict([
        ('precision', tpas),
        ('recall', tprs),
        ('fpr', fprs),
        ('markedness', mks),
        ('bookmaker', bms),
        ('mcc', mccs),
        ('f1', f1_scores),
        ('g1', g1_scores),
        ('support', real_total),
    ])

    tpa = np.nansum(tpas * rprob)
    tpr = np.nansum(tprs * rprob)

    fpr = np.nansum(fprs * rprob)

    mk = np.nansum(mks * rprob)
    bm = np.nansum(bms * pprob)

    # The simple mean seems to do the best
    mccs_ = mccs[~np.isnan(mccs)]
    if len(mccs_) == 0:
        mcc_combo = np.nan
    else:
        mcc_combo = np.nanmean(mccs_)

    combined_data = ub.odict([
        ('precision', tpa),
        ('recall', tpr),
        ('fpr', fpr),
        ('markedness', mk),
        ('bookmaker', bm),
        # ('mcc', np.sign(bm) * np.sqrt(np.abs(bm * mk))),
        ('mcc', mcc_combo),
        # np.sign(bm) * np.sqrt(np.abs(bm * mk))),
        ('f1', np.nanmean(f1_scores)),
        ('g1', np.nanmean(g1_scores)),
        ('support', real_total.sum()),
    ])

    # # Not sure how to compute this. Should it agree with the sklearn impl?
    # if verbose == 'hack':
    #     verbose = False
    #     mcc_known = sklearn.metrics.matthews_corrcoef(
    #         y_true, y_pred, sample_weight=sample_weight)
    #     mcc_raw = np.sign(bm) * np.sqrt(np.abs(bm * mk))

    #     def gmean(x, w=None):
    #         if w is None:
    #             return sp.stats.gmean(x)
    #         return np.exp(np.nansum(w * np.log(x)) / np.nansum(w))

    #     def hmean(x, w=None):
    #         if w is None:
    #             return sp.stats.hmean(x)
    #         return 1 / (np.nansum(w * (1 / x)) / np.nansum(w))

    #     def amean(x, w=None):
    #         if w is None:
    #             return np.mean(x)
    #         return np.nansum(w * x) / np.nansum(w)

    #     report = {
    #         'target': mcc_known,
    #         'raw': mcc_raw,
    #     }

    #     means = {
    #         'a': amean,
    #         # 'h': hmean,
    #         'g': gmean,
    #     }
    #     weights = {
    #         'p': pprob,
    #         'r': rprob,
    #         '': None,
    #     }
    #     for mean_key, mean in means.items():
    #         for w_key, w in weights.items():
    #             # Hack of very wrong items
    #             if mean_key == 'g':
    #                 if w_key in ['r', 'p', '']:
    #                     continue
    #             if mean_key == 'g':
    #                 if w_key in ['r']:
    #                     continue
    #             m = mean(mccs, w)
    #             r_key = '{} {}'.format(mean_key, w_key)
    #             report[r_key] = m
    #             # log(r_key)
    #             # log(np.abs(m - mcc_known))

    #     return report

    index = pd.Index(target_names, name='class')

    perclass_df = pd.DataFrame(perclass_data, index=index)

    if remove_unsupported:
        perclass_df = perclass_df[perclass_df['support'] > 0]

    # combined_df = pd.DataFrame(combined_data, index=['ave/sum'])
    combined_df = pd.DataFrame(combined_data, index=['combined'])

    metric_df = pd.concat([perclass_df, combined_df])
    metric_df.index.name = 'class'
    metric_df.columns.name = 'metric'

    pred_id = ['%s' % m for m in target_names]
    real_id = ['%s' % m for m in target_names]
    confusion_df = pd.DataFrame(confusion, columns=pred_id, index=real_id)

    if ascii_only :
        sum_glyph = 'sum-'
    else:
        sum_glyph = 'Σ'

    sum_real_key = sum_glyph + 'r'
    sum_pred_key = sum_glyph + 'p'

    confusion_df = confusion_df.append(pd.DataFrame(
        [confusion.sum(axis=0)], columns=pred_id, index=[sum_pred_key]))

    confusion_df[sum_real_key] = np.hstack([confusion.sum(axis=1), [0]])
    confusion_df.index.name = 'real'
    confusion_df.columns.name = 'pred'

    _residual = (confusion_df - np.floor(confusion_df)).values
    _thresh = 1e-6
    if np.all(_residual < _thresh):
        confusion_df = confusion_df.astype(int)
    confusion_df.iloc[(-1, -1)] = N
    _residual = (confusion_df - np.floor(confusion_df)).values
    if np.all(_residual < _thresh):
        confusion_df = confusion_df.astype(int)

    if verbose:
        if ascii_only:
            times_glyph = 'x'
        else:
            times_glyph = '×'

        cfsm_str = confusion_df.to_string(float_format=lambda x: '%.1f' % (x,))
        log('Confusion Matrix (real ' + times_glyph + ' pred) :\n' + ub.indent(cfsm_str))

        # ut.cprint('\nExtended Report', 'turquoise')
        float_precision = 2
        float_format = '%.' + str(float_precision) + 'f'
        ext_report = metric_df.to_string(float_format=float_format)
        log('\nEvaluation Metric Report:' + '\n' + ub.indent(ext_report))

    report = {
        'metrics': metric_df,
        'confusion': confusion_df,
    }

    # TODO: What is the difference between sklearn multiclass-MCC
    # and BM * MK MCC?

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='invalid value encountered in double_scalars')
            warnings.filterwarnings('ignore', message='Mean of empty slice')

            mcc = sklearn.metrics.matthews_corrcoef(
                y_true, y_pred, sample_weight=sample_weight)
            # mcc = matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight)
            # These scales are chosen somewhat arbitrarily in the context of a
            # computer vision application with relatively reasonable quality data
            # https://stats.stackexchange.com/questions/118219/how-to-interpret
            mcc_significance_scales = ub.odict([
                (1.0, 'perfect'),
                (0.9, 'very strong'),
                (0.7, 'strong'),
                (0.5, 'significant'),
                (0.3, 'moderate'),
                (0.2, 'weak'),
                (0.0, 'negligible'),
            ])
            for k, v in mcc_significance_scales.items():
                if np.abs(mcc) >= k:
                    if verbose:
                        log('classifier correlation is %s' % (v,))
                    break
            if verbose:
                float_precision = 2
                log(('MCC\' = %.' + str(float_precision) + 'f') % (mcc,))
            report['mcc'] = mcc
    except ValueError:
        report['mcc'] = None
    return report


def ovr_classification_report(mc_y_true, mc_probs, target_names=None,
                              sample_weight=None, metrics=None, verbose=0,
                              remove_unsupported=False, log=None):
    """
    One-vs-rest classification report

    Args:
        mc_y_true (ndarray[int]): multiclass truth labels (integer label
            format). Shape [N].

        mc_probs (ndarray): multiclass probabilities for each class.
            Shape [N x C].

        target_names (Dict[int, str] : mapping from int label to string name

        sample_weight (ndarray): weight for each item. Shape [N].

        metrics (List[str]): names of metrics to compute


    Example:
        >>> # xdoctest: +IGNORE_WANT
        >>> # xdoctest: +REQUIRES(module:sklearn)
        >>> # xdoctest: +REQUIRES(module:pandas)
        >>> from kwcoco.metrics.clf_report import *  # NOQA
        >>> y_true = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0]
        >>> y_probs = np.random.rand(len(y_true), max(y_true) + 1)
        >>> target_names = None
        >>> sample_weight = None
        >>> verbose = True
        >>> report = ovr_classification_report(y_true, y_probs)
        >>> print(report['ave'])
        auc     0.6541
        ap      0.6824
        kappa   0.0963
        mcc     0.1002
        brier   0.2214
        dtype: float64
        >>> print(report['ovr'])
             auc     ap  kappa    mcc  brier  support  weight
        0 0.6062 0.6161 0.0526 0.0598 0.2608        8  0.4444
        1 0.5846 0.6014 0.0000 0.0000 0.2195        5  0.2778
        2 0.8000 0.8693 0.2623 0.2652 0.1602        5  0.2778

    Ignore:
        >>> y_true = [1, 1, 1]
        >>> y_probs = np.random.rand(len(y_true), 3)
        >>> target_names = None
        >>> sample_weight = None
        >>> verbose = True
        >>> report = ovr_classification_report(y_true, y_probs)
        >>> print(report['ovr'])

    """
    import pandas as pd
    import sklearn.metrics

    if metrics is None:
        metrics = ['auc', 'ap', 'mcc', 'f1', 'brier', 'kappa']

    n_classes = mc_probs.shape[1]
    ohvec_true = np.eye(n_classes, dtype=np.uint8)[mc_y_true]

    # Preallocate common datas
    bin_probs = np.empty((len(mc_probs), 2), dtype=mc_probs.dtype)

    import kwimage  # TODO: move to kwarray
    import kwarray
    # Map everything onto 0-1 range
    ranked_cidxs = kwarray.argmaxima(mc_probs, 2, axis=1)
    ranked_scores = np.array([a[x] for a, x in zip(mc_probs, ranked_cidxs)])  # probably better numpy way to do this

    mc_scores = kwimage.normalize(mc_probs, mode='linear')
    total_scores = mc_scores.sum(axis=1, keepdims=0)
    # max_scores = mc_scores.max(axis=1, keepdims=0)

    class_metrics = ub.odict()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')
        warnings.filterwarnings('ignore', message='invalid value encountered in double_scalars')
        warnings.filterwarnings('ignore', message='divide by zero')
        warnings.filterwarnings('ignore', message='due to no true nor predicted samples')
        warnings.filterwarnings('ignore', message='ill-defined')

        for cidx in range(n_classes):
            k_metrics = ub.odict()

            class_score = mc_scores.T[cidx]
            is_other = (ranked_cidxs != cidx)
            other_score = np.array([a[f][0] for a, f in zip(ranked_scores, is_other)])

            # HEURISTIC:
            # We need to compute a score or "probability" of other
            # is there a better way to do this?
            # other_prob = total_scores - class_prob
            # class_prob = class_score
            # other_prob = max_scores
            # other_prob = (max_scores - class_score)
            class_prob = class_score / total_scores
            other_prob = other_score / total_scores
            # other_prob = (max_scores - class_score) / total_scores

            # Consider each class a one-vs-rest problem
            # Populate the first column

            bin_probs[:, 1] = class_prob
            bin_probs[:, 0] = other_prob

            # Index of the true class
            k_true = ohvec_true.T[cidx]
            # Index of the predicted class
            k_pred = np.argmax(bin_probs, axis=1)  # NOTE: ASSUME MUTEX CLASSES

            # Probabilities for the true class for each label
            bin_truth = np.eye(2)[k_true]
            true_probs = (bin_probs * bin_truth).sum(axis=1)

            if 'auc' in metrics:
                try:
                    k_metrics['auc'] = sklearn.metrics.roc_auc_score(
                        bin_truth, bin_probs, sample_weight=sample_weight)
                except ValueError:
                    k_metrics['auc'] = np.nan

            if 'ap' in metrics:
                k_metrics['ap'] = sklearn.metrics.average_precision_score(
                    bin_truth, bin_probs, sample_weight=sample_weight,
                    # zero_division=1,
                    # np.nan
                )

            if 'kappa' in metrics:
                k_metrics['kappa'] = sklearn.metrics.cohen_kappa_score(
                    k_true, k_pred, labels=[0, 1], sample_weight=sample_weight)

            if 'mcc' in metrics:
                k_metrics['mcc'] = sklearn.metrics.matthews_corrcoef(
                    k_true, k_pred, sample_weight=sample_weight)

            if 'f1' in metrics:
                k_metrics['f1'] = sklearn.metrics.fbeta_score(
                    k_true, k_pred, beta=1.0, sample_weight=sample_weight,
                    zero_division=0,
                    # zero_division=1,
                    # zero_division=np.nan,
                )

            if 'brier' in metrics:
                # Get the probablity of the real class for each example
                rprobs = np.clip(true_probs / total_scores, 0, 1)
                rwants = np.ones(len(rprobs))
                # Use custom brier implemention until sklearn is fixed.
                mse = (rwants - rprobs) ** 2
                if sample_weight is None:
                    k_metrics['brier'] = mse.mean()
                else:
                    k_metrics['brier'] = (mse * sample_weight).sum() / sample_weight.sum()
                # NOTE: There is a bug here (but bug is in sklearn 0.19.1)
                # brier = sklearn.metrics.brier_score_loss(rwants, rprobs)

            if sample_weight is None:
                k_metrics['support'] = k_true.sum()
            else:
                k_metrics['support'] = (sample_weight * k_true).sum()

            key = cidx if target_names is None else target_names[cidx]
            class_metrics[key] = k_metrics

    ovr_metrics = pd.DataFrame.from_dict(class_metrics, orient='index')

    if remove_unsupported:
        ovr_metrics = ovr_metrics[ovr_metrics['support'] > 0]

    weight = ovr_metrics.loc[:, 'support'] / ovr_metrics.loc[:, 'support'].sum()
    ovr_metrics['weight'] = weight

    # weighted = ovr_metrics.drop(columns=['support', 'weight'])
    weighted = ovr_metrics.copy()
    weighted.iloc[:] = weighted.values * weight.values[:, None]
    weighted_ave = weighted.sum(axis=0)
    weighted_ave['support'] = ovr_metrics['support'].sum()
    weighted_ave['weight'] = ovr_metrics['weight'].sum()

    report = {
        'ovr': ovr_metrics,
        'ave': weighted_ave,
    }

    if verbose or log:
        if log is None:
            log = print
        ovr_metrics = report['ovr']
        weighted_ave = report['ave']
        log('ovr_metrics')
        log(pd.concat([ovr_metrics, weighted_ave.to_frame('__accum__').T]))
    return report
