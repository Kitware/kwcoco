"""
Faster pure-python versions of sklearn functions that avoid expensive checks
and label rectifications. It is assumed that all labels are consecutive
non-negative integers.
"""
from scipy.sparse import coo_matrix
import numpy as np


def confusion_matrix(y_true, y_pred, n_labels=None, labels=None,
                     sample_weight=None):
    """
    faster version of sklearn confusion matrix that avoids the
    expensive checks and label rectification

    Runs in about 0.7ms

    Returns:
        ndarray: matrix where rows represent real and cols represent pred

    Example:
        >>> y_true = np.array([0, 0, 0, 0, 1, 1, 1, 0,  0, 1])
        >>> y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 1,  1, 1])
        >>> confusion_matrix(y_true, y_pred, 2)
        array([[4, 2],
               [3, 1]])
        >>> confusion_matrix(y_true, y_pred, 2).ravel()
        array([4, 2, 3, 1])

    Benchmarks:
        import ubelt as ub
        y_true = np.random.randint(0, 2, 10000)
        y_pred = np.random.randint(0, 2, 10000)

        n = 1000
        for timer in ub.Timerit(n, bestof=10, label='py-time'):
            sample_weight = [1] * len(y_true)
            confusion_matrix(y_true, y_pred, 2, sample_weight=sample_weight)

        for timer in ub.Timerit(n, bestof=10, label='np-time'):
            sample_weight = np.ones(len(y_true), dtype=int)
            confusion_matrix(y_true, y_pred, 2, sample_weight=sample_weight)
    """
    if sample_weight is None:
        sample_weight = np.ones(len(y_true), dtype=int)
    if n_labels is None:
        n_labels = len(labels)
    CM = coo_matrix((sample_weight, (y_true, y_pred)),
                    shape=(n_labels, n_labels),
                    dtype=np.int64).toarray()
    return CM


def global_accuracy_from_confusion(cfsn):
    # real is rows, pred is columns
    n_ii = np.diag(cfsn)
    # sum over pred = columns = axis1
    t_i = cfsn.sum(axis=1)
    global_acc = n_ii.sum() / t_i.sum()
    return global_acc


def class_accuracy_from_confusion(cfsn):
    # real is rows, pred is columns
    n_ii = np.diag(cfsn)
    # sum over pred = columns = axis1
    t_i = cfsn.sum(axis=1)
    per_class_acc = (n_ii / t_i).mean()
    class_acc = np.nan_to_num(per_class_acc).mean()
    return class_acc


def _binary_clf_curve2(y_true, y_score, pos_label=None, sample_weight=None):
    """
    MODIFIED VERSION OF SCIKIT-LEARN API

    Calculate true and false positives per binary classification threshold.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification

    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function

    pos_label : int or str, default=None
        The label of the positive class

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).

    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).

    thresholds : array, shape = [n_thresholds]
        Decreasing score values.

    Example
    -------
    >>> y_true  = [      1,   1,   1,   1,   1,   1,   0]
    >>> y_score = [ np.nan, 0.2, 0.3, 0.4, 0.5, 0.6, 0.3]
    >>> sample_weight = None
    >>> pos_label = None
    >>> fps, tps, thresholds = _binary_clf_curve2(y_true, y_score)
    """
    import numpy as np
    from sklearn.utils import assert_all_finite
    from sklearn.utils import column_or_1d
    from sklearn.utils import check_consistent_length
    from sklearn.utils.multiclass import type_of_target
    from sklearn.utils.extmath import stable_cumsum
    # Check to make sure y_true is valid
    y_type = type_of_target(y_true)
    if not (y_type == "binary" or
            (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    # assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    # ensure binary classification if pos_label is not specified
    # classes.dtype.kind in ('O', 'U', 'S') is required to avoid
    # triggering a FutureWarning by calling np.array_equal(a, b)
    # when elements in the two arrays are not comparable.
    classes = np.unique(y_true)
    if (pos_label is None and (
            classes.dtype.kind in ('O', 'U', 'S') or
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1])))):
        classes_repr = ", ".join(repr(c) for c in classes)
        raise ValueError("y_true takes value in {{{classes_repr}}} and "
                         "pos_label is not specified: either make y_true "
                         "take value in {{0, 1}} or {{-1, 1}} or "
                         "pass pos_label explicitly.".format(
                             classes_repr=classes_repr))
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # Transform nans into negative infinity
    nan_flags = np.isnan(y_score)
    y_score[nan_flags] = -np.inf

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.

    with np.errstate(invalid="ignore"):
        y_diff = np.diff(y_score)
    # Set difference between -inf to zero
    fix_flags = np.isinf(y_score[:-1]) & np.isnan(y_diff)
    y_diff[fix_flags] = 0

    distinct_value_indices = np.where(y_diff)[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]
