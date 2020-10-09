import numpy as np
import ubelt as ub
import warnings


def draw_perclass_roc(cx_to_info, classes=None, prefix='', fnum=1,
                      fp_axis='count', **kw):
    """
    Args:
        cx_to_info (PerClass_Measures | Dict):

        fp_axis (str): can be count or rate
    """
    import kwplot
    # Sort by descending AP
    cxs = list(cx_to_info.keys())
    priority = np.array([item['auc'] for item in cx_to_info.values()])
    priority[np.isnan(priority)] = -np.inf
    cxs = list(ub.take(cxs, np.argsort(priority)))[::-1]
    xydata = ub.odict()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)
        mAUC = np.nanmean([item['auc'] for item in cx_to_info.values()])

    if fp_axis == 'count':
        xlabel = 'FP-count'
    elif fp_axis == 'rate':
        xlabel = 'FPR'
    else:
        raise KeyError(fp_axis)

    for cx in cxs:
        info = cx_to_info[cx]

        catname = classes[cx] if isinstance(cx, int) else cx

        try:
            auc = info['trunc_auc']
            tpr = info['trunc_tpr']
            fp_count = info['trunc_fp_count']
            fpr = info['trunc_fpr']
        except KeyError:
            auc = info['auc']
            tpr = info['tpr']
            fp_count = info['fp_count']
            fpr = info['fpr']

        nsupport = int(info['nsupport'])
        if 'realpos_total' in info:
            z = info['realpos_total']
            if abs(z - int(z)) < 1e-8:
                label = 'auc={:0.2f}: {} ({:d}/{:d})'.format(auc, catname, int(info['realpos_total']), round(nsupport, 2))
            else:
                label = 'auc={:0.2f}: {} ({:.2f}/{:d})'.format(auc, catname, round(info['realpos_total'], 2), round(nsupport, 2))
        else:
            label = 'auc={:0.2f}: {} ({:d})'.format(auc, catname, round(nsupport, 2))

        if fp_axis == 'count':
            xydata[label] = (fp_count, tpr)
        elif fp_axis == 'rate':
            xydata[label] = (fpr, tpr)

    ax = kwplot.multi_plot(
        xydata=xydata, fnum=fnum,
        ylim=(0, 1), xpad=0.01, ypad=0.01,
        xlabel=xlabel, ylabel='TPR',
        title=prefix + 'perclass mAUC={:.4f}'.format(mAUC),
        legend_loc='lower right',
        color='distinct', linestyle='cycle', marker='cycle', **kw
    )
    return ax


def draw_perclass_prcurve(cx_to_info, classes=None, prefix='', fnum=1, **kw):
    """
    Args:
        cx_to_info (PerClass_Measures | Dict):

    Example:
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> from kwcoco.metrics.drawing import *  # NOQA
        >>> from kwcoco.metrics import DetectionMetrics
        >>> dmet = DetectionMetrics.demo(
        >>>     nimgs=3, nboxes=(0, 10), n_fp=(0, 3), n_fn=(0, 2), classes=3, score_noise=0.1, box_noise=0.1, with_probs=False)
        >>> cfsn_vecs = dmet.confusion_vectors()
        >>> print(cfsn_vecs.data.pandas())
        >>> classes = cfsn_vecs.classes
        >>> cx_to_info = cfsn_vecs.binarize_ovr().measures()['perclass']
        >>> print('cx_to_info = {}'.format(ub.repr2(cx_to_info, nl=1)))
        >>> import kwplot
        >>> kwplot.autompl()
        >>> draw_perclass_prcurve(cx_to_info, classes)
        >>> # xdoctest: +REQUIRES(--show)
        >>> kwplot.show_if_requested()

    Ignore:
        from kwcoco.metrics.drawing import *  # NOQA
        import xdev
        globals().update(xdev.get_func_kwargs(draw_perclass_prcurve))

    """
    import kwplot
    # Sort by descending AP
    cxs = list(cx_to_info.keys())
    priority = np.array([item['ap'] for item in cx_to_info.values()])
    priority[np.isnan(priority)] = -np.inf
    cxs = list(ub.take(cxs, np.argsort(priority)))[::-1]
    aps = []
    xydata = ub.odict()
    for cx in cxs:
        info = cx_to_info[cx]
        catname = classes[cx] if isinstance(cx, int) else cx
        ap = info['ap']
        if 'pr' in info:
            pr = info['pr']
        elif 'ppv' in info:
            pr = (info['ppv'], info['tpr'])
        elif 'prec' in info:
            pr = (info['prec'], info['rec'])
        else:
            raise KeyError('pr, prec, or ppv not in info')

        if np.isfinite(ap):
            aps.append(ap)
            (precision, recall) = pr
        else:
            aps.append(np.nan)
            precision, recall = [0], [0]

        if precision is None and recall is None:
            # I thought AP=nan in this case, but I missed something
            precision, recall = [0], [0]

        nsupport = int(info['nsupport'])
        if 'realpos_total' in info:
            z = info['realpos_total']
            if abs(z - int(z)) < 1e-8:
                label = 'ap={:0.2f}: {} ({:d}/{:d})'.format(ap, catname, int(info['realpos_total']), round(nsupport, 2))
            else:
                label = 'ap={:0.2f}: {} ({:.2f}/{:d})'.format(ap, catname, round(info['realpos_total'], 2), round(nsupport, 2))
        else:
            label = 'ap={:0.2f}: {} ({:d})'.format(ap, catname, round(nsupport, 2))
        xydata[label] = (recall, precision)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)
        mAP = np.nanmean(aps)

    if 0:
        # TODO: deprecate multi_plot for seaborn?
        data_groups = {
            key: {'recall': r, 'precision': p}
            for key, (r, p) in xydata.items()
        }
        print('data_groups = {}'.format(ub.repr2(data_groups, nl=3)))

        longform = []
        for key, subdata in data_groups.items():
            import pandas as pd
            subdata = pd.DataFrame.from_dict(subdata)
            subdata['label'] = key
            longform.append(subdata)
        data = pd.concat(longform)

        import seaborn as sns
        import pandas as pd
        # sns.set()
        fig = kwplot.figure(fnum=fnum)
        ax = fig.gca()
        longform = []
        for key, (r, p) in xydata.items():
            subdata = pd.DataFrame.from_dict({'recall': r, 'precision': p, 'label': key})
            longform.append(subdata)
        data = pd.concat(longform)

        palette = ub.dzip(xydata.keys(), kwplot.distinct_colors(len(xydata)))
        # markers = ub.dzip(xydata.keys(), kwplot.distinct_markers(len(xydata)))

        sns.lineplot(
            data=data, x='recall', y='precision',
            hue='label', style='label', ax=ax,
            # markers=markers,
            estimator=None,
            ci=0,
            hue_order=list(xydata.keys()),
            palette=palette,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    else:
        ax = kwplot.multi_plot(
            xydata=xydata, fnum=fnum,
            xlim=(0, 1), ylim=(0, 1), xpad=0.01, ypad=0.01,
            xlabel='recall', ylabel='precision',
            err_style='bars',
            title=prefix + 'OVR mAP={:.4f}'.format(mAP),
            legend_loc='lower right',
            color='distinct', linestyle='cycle', marker='cycle', **kw
        )
    return ax


def draw_perclass_thresholds(cx_to_info, key='mcc', classes=None, prefix='', fnum=1, **kw):
    """
    Args:
        cx_to_info (PerClass_Measures | Dict):

    Notes:
        Each category is inspected independently of one another, there is no
        notion of confusion.

    Example:
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> from kwcoco.metrics.drawing import *  # NOQA
        >>> from kwcoco.metrics import ConfusionVectors
        >>> cfsn_vecs = ConfusionVectors.demo()
        >>> classes = cfsn_vecs.classes
        >>> ovr_cfsn = cfsn_vecs.binarize_ovr(keyby='name')
        >>> cx_to_info = ovr_cfsn.measures()['perclass']
        >>> import kwplot
        >>> kwplot.autompl()
        >>> key = 'mcc'
        >>> draw_perclass_thresholds(cx_to_info, key, classes)
        >>> # xdoctest: +REQUIRES(--show)
        >>> kwplot.show_if_requested()
    """
    import kwplot
    # Sort by descending "best value"
    cxs = list(cx_to_info.keys())

    try:
        priority = np.array([item['_max_' + key][0] for item in cx_to_info.values()])
        priority[np.isnan(priority)] = -np.inf
        cxs = list(ub.take(cxs, np.argsort(priority)))[::-1]
    except KeyError:
        pass

    xydata = ub.odict()
    for cx in cxs:
        info = cx_to_info[cx]
        catname = classes[cx] if isinstance(cx, int) else cx

        thresholds = info['thresholds']
        measure = info[key]
        try:
            best_label = info['max_{}'.format(key)]
        except KeyError:
            max_idx = measure.argmax()
            best_thresh = thresholds[max_idx]
            best_measure = measure[max_idx]
            best_label = '{}={:0.2f}@{:0.2f}'.format(key, best_measure, best_thresh)

        nsupport = int(info['nsupport'])
        if 'realpos_total' in info:
            z = info['realpos_total']
            if abs(z - int(z)) < 1e-8:
                label = '{}: {} ({:d}/{:d})'.format(best_label, catname, int(info['realpos_total']), round(nsupport, 2))
            else:
                label = '{}: {} ({:.2f}/{:d})'.format(best_label, catname, round(info['realpos_total'], 2), round(nsupport, 2))
        else:
            label = '{}: {} ({:d})'.format(best_label, catname, round(nsupport, 2))
        xydata[label] = (thresholds, measure)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)

    ax = kwplot.multi_plot(
        xydata=xydata, fnum=fnum,
        xlim=(0, 1), ylim=(0, 1), xpad=0.01, ypad=0.01,
        xlabel='threshold', ylabel=key,
        title=prefix + 'OVR {}'.format(key),
        legend_loc='lower right',
        color='distinct', linestyle='cycle', marker='cycle', **kw
    )
    return ax


def draw_roc(info, prefix='', fnum=1, **kw):
    """
    Args:
        info (Measures | Dict)

    NOTE:
        There needs to be enough negative examples for using ROC to make any
        sense!

    Example:
        >>> # xdoctest: +REQUIRES(module:kwplot, module:seaborn)
        >>> from kwcoco.metrics.drawing import *  # NOQA
        >>> from kwcoco.metrics import DetectionMetrics
        >>> dmet = DetectionMetrics.demo(nimgs=30, null_pred=1, classes=3,
        >>>                              nboxes=10, n_fp=10, box_noise=0.3,
        >>>                              with_probs=False)
        >>> dmet.true_detections(0).data
        >>> cfsn_vecs = dmet.confusion_vectors(compat='mutex', prioritize='iou', bias=0)
        >>> print(cfsn_vecs.data._pandas().sort_values('score'))
        >>> classes = cfsn_vecs.classes
        >>> info = ub.peek(cfsn_vecs.binarize_ovr().measures()['perclass'].values())
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> draw_roc(info)
        >>> kwplot.show_if_requested()
    """
    import kwplot
    try:
        fp_count = info['trunc_fp_count']
        fp_rate = info['trunc_fpr']
        tp_rate = info['trunc_tpr']
        auc = info['trunc_auc']
    except KeyError:
        fp_count = info['fp_count']
        fp_rate = info['fpr']
        tp_rate = info['tpr']
        auc = info['auc']
    realpos_total = info['realpos_total']

    title = prefix + 'AUC*: {:.4f}'.format(auc)
    falsepos_total = fp_count[-1]

    if 0:
        # TODO: deprecate multi_plot for seaborn?
        fig = kwplot.figure(fnum=fnum)
        ax = fig.gca()
        import seaborn as sns
        xlabel = 'fpr (count={})'.format(falsepos_total)
        ylabel = 'tpr (count={})'.format(int(realpos_total))
        data = {
            xlabel: list(fp_rate),
            ylabel: list(tp_rate),
        }
        sns.lineplot(data=data, x=xlabel, y=ylabel, markers='', ax=ax)
        ax.set_title(title)
    else:
        ax = kwplot.multi_plot(
            list(fp_rate), list(tp_rate), marker='',
            # xlabel='FA count (false positive count)',
            xlabel='fpr (count={})'.format(falsepos_total),
            ylabel='tpr (count={})'.format(int(realpos_total)),
            title=title,
            ylim=(0, 1), ypad=1e-2,
            xlim=(0, 1), xpad=1e-2,
            fnum=fnum, **kw)

    return ax


def draw_prcurve(info, prefix='', fnum=1, **kw):
    """
    Draws a single pr curve.

    Args:
        info (Measures | Dict)

    Example:
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> from kwcoco.metrics import DetectionMetrics
        >>> dmet = DetectionMetrics.demo(
        >>>     nimgs=10, nboxes=(0, 10), n_fp=(0, 1), classes=3)
        >>> cfsn_vecs = dmet.confusion_vectors()

        >>> classes = cfsn_vecs.classes
        >>> info = cfsn_vecs.binarize_classless().measures()
        >>> import kwplot
        >>> kwplot.autompl()
        >>> draw_prcurve(info)
        >>> # xdoctest: +REQUIRES(--show)
        >>> kwplot.show_if_requested()
    """
    import kwplot
    aps = []
    ap = info['ap']
    if 'pr' in info:
        pr = info['pr']
    elif 'ppv' in info:
        pr = (info['ppv'], info['tpr'])
    elif 'prec' in info:
        pr = (info['prec'], info['rec'])
    else:
        raise KeyError('pr, prec, or ppv not in info')
    if np.isfinite(ap):
        aps.append(ap)
        (precision, recall) = pr
    else:
        precision, recall = [0], [0]
    if precision is None and recall is None:
        # I thought AP=nan in this case, but I missed something
        precision, recall = [0], [0]

    nsupport = int(info['nsupport'])
    if 'realpos_total' in info:
        z = info['realpos_total']
        if abs(z - int(z)) < 1e-8:
            label = 'ap={:0.2f}: ({:d}/{:d})'.format(ap, int(info['realpos_total']), round(nsupport, 2))
        else:
            label = 'ap={:0.2f}: ({:.2f}/{:d})'.format(ap, round(info['realpos_total'], 2), round(nsupport, 2))
    else:
        label = 'ap={:0.2f}: ({:d})'.format(ap, nsupport)

    ax = kwplot.multi_plot(
        xdata=recall, ydata=precision, fnum=fnum, label=label,
        xlim=(0, 1), ylim=(0, 1), xpad=0.01, ypad=0.01,
        xlabel='recall', ylabel='precision',
        title=prefix + 'classless AP={:.4f}'.format(ap),
        legend_loc='lower right',
        color='distinct', linestyle='cycle', marker='cycle', **kw
    )

    # if 0:
    #     # TODO: should show contour lines with F1 scores
    #     x = np.arange(0.0, 1.0, 1e-3)
    #     X, Y = np.meshgrid(x, x)
    #     Z = np.round(2.XY/(X+Y),3)
    #     Z[np.isnan(Z)] = 0
    #     levels =  np.round(np.arange(0.1, 1.0, .1),1)
    #     CS = ax.contour(X, Y, Z,
    #                     levels=levels,
    #                     linewidths=0.75,
    #                     cmap='copper')
    #     location = zip(levels, levels)
    #     ax.clabel(CS, inline=1, fontsize=9, manual=location, fmt='%.1f')
    #     for c in CS.collections:
    #         c.set_linestyle('dashed')

    return ax


def draw_threshold_curves(info, keys=None, prefix='', fnum=1, **kw):
    """
    Args:
        info (Measures | Dict)

    Example:
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/kwcoco'))
        >>> from kwcoco.metrics.drawing import *  # NOQA
        >>> from kwcoco.metrics import DetectionMetrics
        >>> dmet = DetectionMetrics.demo(
        >>>     nimgs=10, nboxes=(0, 10), n_fp=(0, 1), classes=3)
        >>> cfsn_vecs = dmet.confusion_vectors()
        >>> info = cfsn_vecs.binarize_classless().measures()
        >>> keys = None
        >>> import kwplot
        >>> kwplot.autompl()
        >>> draw_threshold_curves(info, keys)
        >>> # xdoctest: +REQUIRES(--show)
        >>> kwplot.show_if_requested()
    """
    import kwplot
    import kwimage
    thresh = info['thresholds']

    if keys is None:
        keys = {'g1', 'f1', 'acc', 'mcc'}

    idx_to_colors = kwimage.Color.distinct(len(keys), space='rgba')
    idx_to_best_pt = {}

    xydata = {}
    colors = {}
    finite_flags = np.isfinite(thresh)

    for idx, key in enumerate(keys):
        color = idx_to_colors[idx]
        measure = info[key][finite_flags]
        max_idx = np.nanargmax(measure)
        offset = (~finite_flags[:max_idx]).sum()
        max_idx += offset
        best_thresh = thresh[max_idx]
        best_measure = measure[max_idx]
        best_label = '{}={:0.2f}@{:0.2f}'.format(key, best_measure, best_thresh)

        nsupport = int(info['nsupport'])
        if 'realpos_total' in info:
            z = info['realpos_total']
            if abs(z - int(z)) < 1e-8:
                label = '{}: ({:d}/{:d})'.format(best_label, int(info['realpos_total']), round(nsupport, 2))
            else:
                label = '{}: ({:.2f}/{:d})'.format(best_label, round(info['realpos_total'], 2), round(nsupport, 2))
        else:
            label = '{}: ({:d})'.format(best_label, nsupport)
        xydata[label] = (thresh, measure)
        colors[label] = color
        idx_to_best_pt[idx] = (best_thresh, best_measure)

    ax = kwplot.multi_plot(
        xydata=xydata, fnum=fnum,
        xlim=(0, 1), ylim=(0, 1), xpad=0.01, ypad=0.01,
        xlabel='threshold', ylabel=key,
        title=prefix + 'threshold curves',
        legend_loc='lower right',
        color=colors,
        linestyle='cycle', marker='cycle', **kw
    )
    for idx, best_pt in idx_to_best_pt.items():
        best_thresh, best_measure = best_pt
        color = idx_to_colors[idx]
        ax.plot(best_thresh, best_measure, '*', color=color)
    return ax

if __name__ == '__main__':
    """

        python ~/code/kwcoco/kwcoco/metrics/drawing.py
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
