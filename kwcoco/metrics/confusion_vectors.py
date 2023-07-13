"""
Classes that store raw confusion vectors, which can be accumulated into
confusion measures.
"""
import numpy as np
import ubelt as ub
import warnings
from kwcoco.metrics.functional import fast_confusion_matrix
from kwcoco.metrics.sklearn_alts import _binary_clf_curve2
from kwcoco.category_tree import CategoryTree
from kwcoco.metrics.confusion_measures import (
    Measures, PerClass_Measures, populate_info)

try:
    from xdev import profile
except Exception:
    profile = ub.identity


class ConfusionVectors(ub.NiceRepr):
    """
    Stores information used to construct a confusion matrix. This includes
    corresponding vectors of predicted labels, true labels, sample weights,
    etc...

    Attributes:
        data (kwarray.DataFrameArray) : should at least have keys true, pred, weight
        classes (Sequence | CategoryTree): list of category names or category graph
        probs (ndarray | None): probabilities for each class

    Example:
        >>> # xdoctest: IGNORE_WANT
        >>> # xdoctest: +REQUIRES(module:pandas)
        >>> from kwcoco.metrics import DetectionMetrics
        >>> dmet = DetectionMetrics.demo(
        >>>     nimgs=10, nboxes=(0, 10), n_fp=(0, 1), classes=3)
        >>> cfsn_vecs = dmet.confusion_vectors()
        >>> print(cfsn_vecs.data._pandas())
             pred  true   score  weight     iou  txs  pxs  gid
        0       2     2 10.0000  1.0000  1.0000    0    4    0
        1       2     2  7.5025  1.0000  1.0000    1    3    0
        2       1     1  5.0050  1.0000  1.0000    2    2    0
        3       3    -1  2.5075  1.0000 -1.0000   -1    1    0
        4       2    -1  0.0100  1.0000 -1.0000   -1    0    0
        5      -1     2  0.0000  1.0000 -1.0000    3   -1    0
        6      -1     2  0.0000  1.0000 -1.0000    4   -1    0
        7       2     2 10.0000  1.0000  1.0000    0    5    1
        8       2     2  8.0020  1.0000  1.0000    1    4    1
        9       1     1  6.0040  1.0000  1.0000    2    3    1
        ..    ...   ...     ...     ...     ...  ...  ...  ...
        62     -1     2  0.0000  1.0000 -1.0000    7   -1    7
        63     -1     3  0.0000  1.0000 -1.0000    8   -1    7
        64     -1     1  0.0000  1.0000 -1.0000    9   -1    7
        65      1    -1 10.0000  1.0000 -1.0000   -1    0    8
        66      1     1  0.0100  1.0000  1.0000    0    1    8
        67      3    -1 10.0000  1.0000 -1.0000   -1    3    9
        68      2     2  6.6700  1.0000  1.0000    0    2    9
        69      2     2  3.3400  1.0000  1.0000    1    1    9
        70      3    -1  0.0100  1.0000 -1.0000   -1    0    9
        71     -1     2  0.0000  1.0000 -1.0000    2   -1    9

        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:pandas)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> from kwcoco.metrics.confusion_vectors import ConfusionVectors
        >>> cfsn_vecs = ConfusionVectors.demo(
        >>>     nimgs=128, nboxes=(0, 10), n_fp=(0, 3), n_fn=(0, 3), classes=3)
        >>> cx_to_binvecs = cfsn_vecs.binarize_ovr()
        >>> measures = cx_to_binvecs.measures()['perclass']
        >>> print('measures = {!r}'.format(measures))
        measures = <PerClass_Measures({
            'cat_1': <Measures({'ap': 0.227, 'auc': 0.507, 'catname': cat_1, 'max_f1': f1=0.45@0.47, 'nsupport': 788.000})>,
            'cat_2': <Measures({'ap': 0.288, 'auc': 0.572, 'catname': cat_2, 'max_f1': f1=0.51@0.43, 'nsupport': 788.000})>,
            'cat_3': <Measures({'ap': 0.225, 'auc': 0.484, 'catname': cat_3, 'max_f1': f1=0.46@0.40, 'nsupport': 788.000})>,
        }) at 0x7facf77bdfd0>
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> measures.draw(key='pr', fnum=1, pnum=(1, 3, 1))
        >>> measures.draw(key='roc', fnum=1, pnum=(1, 3, 2))
        >>> measures.draw(key='mcc', fnum=1, pnum=(1, 3, 3))
        ...
    """

    def __init__(cfsn_vecs, data, classes, probs=None):
        cfsn_vecs.data = data
        cfsn_vecs.classes = classes
        cfsn_vecs.probs = probs

    def __nice__(cfsn_vecs):
        return cfsn_vecs.data.__nice__()

    def __json__(self):
        """
        Serialize to json

        Example:
            >>> # xdoctest: +REQUIRES(module:pandas)
            >>> from kwcoco.metrics import ConfusionVectors
            >>> self = ConfusionVectors.demo(n_imgs=1, classes=2, n_fp=0, nboxes=1)
            >>> state = self.__json__()
            >>> print('state = {}'.format(ub.urepr(state, nl=2, precision=2, align=1)))
            >>> recon = ConfusionVectors.from_json(state)
        """
        state = {
            'probs': None if self.probs is None else self.probs.tolist(),
            'classes': self.classes.__json__(),
            'data': self.data.pandas().to_dict(orient='list'),
        }
        return state

    @classmethod
    def from_json(cls, state):
        import kwarray
        import kwcoco
        probs = state['probs']
        if probs is not None:
            probs = np.array(probs)
        classes = kwcoco.CategoryTree.from_json(state['classes'])
        data = ub.map_vals(np.array, state['data'])
        data = kwarray.DataFrameArray(data)
        self = cls(data=data, probs=probs, classes=classes)
        return self

    @classmethod
    def demo(cfsn_vecs, **kw):
        """
        Args:
            **kwargs: See :func:`kwcoco.metrics.DetectionMetrics.demo`

        Returns:
            ConfusionVectors

        Example:
            >>> from kwcoco.metrics.confusion_vectors import *  # NOQA
            >>> cfsn_vecs = ConfusionVectors.demo()
            >>> print('cfsn_vecs = {!r}'.format(cfsn_vecs))
            >>> cx_to_binvecs = cfsn_vecs.binarize_ovr()
            >>> print('cx_to_binvecs = {!r}'.format(cx_to_binvecs))
        """
        from kwcoco.metrics import DetectionMetrics
        default = {
            'nimgs': 10,
            'nboxes': (0, 10),
            'n_fp': (0, 1),
            'n_fn': 0,
            'classes': 3,
        }
        demokw = default.copy()
        demokw.update(kw)
        dmet = DetectionMetrics.demo(**demokw)
        # print('dmet = {!r}'.format(dmet))
        cfsn_vecs = dmet.confusion_vectors()
        cfsn_vecs.data._data = ub.dict_isect(cfsn_vecs.data._data, [
            'true', 'pred', 'score', 'weight',
        ])
        return cfsn_vecs

    @classmethod
    def from_arrays(ConfusionVectors, true, pred=None, score=None, weight=None,
                    probs=None, classes=None):
        """
        Construct confusion vector data structure from component arrays

        Example:
            >>> # xdoctest: +REQUIRES(module:pandas)
            >>> import kwarray
            >>> classes = ['person', 'vehicle', 'object']
            >>> rng = kwarray.ensure_rng(0)
            >>> true = (rng.rand(10) * len(classes)).astype(int)
            >>> probs = rng.rand(len(true), len(classes))
            >>> cfsn_vecs = ConfusionVectors.from_arrays(true=true, probs=probs, classes=classes)
            >>> cfsn_vecs.confusion_matrix()
            pred     person  vehicle  object
            real
            person        0        0       0
            vehicle       2        4       1
            object        2        1       0
        """
        import kwarray
        if pred is None:
            if probs is not None:
                if isinstance(classes, CategoryTree):
                    if not classes.is_mutex():
                        raise Exception('Graph categories require explicit pred')
                # We can assume all classes are mutually exclusive here
                pred = probs.argmax(axis=1)
            else:
                raise ValueError('Must specify pred (or probs)')

        data = {
            'true': true,
            'pred': pred,
            'score': score,
            'weight': weight,
        }

        data = {k: v for k, v in data.items() if v is not None}
        cfsn_data = kwarray.DataFrameArray(data)
        cfsn_vecs = ConfusionVectors(cfsn_data, probs=probs, classes=classes)
        return cfsn_vecs

    def confusion_matrix(cfsn_vecs, compress=False):
        """
        Builds a confusion matrix from the confusion vectors.

        Args:
            compress (bool, default=False):
                if True removes rows / columns with no entries

        Returns:
            pd.DataFrame : cm : the labeled confusion matrix
                (Note:  we should write a efficient replacement for
                 this use case. #remove_pandas)

        CommandLine:
            xdoctest -m kwcoco.metrics.confusion_vectors ConfusionVectors.confusion_matrix

        Example:
            >>> # xdoctest: +REQUIRES(module:pandas)
            >>> from kwcoco.metrics import DetectionMetrics
            >>> dmet = DetectionMetrics.demo(
            >>>     nimgs=10, nboxes=(0, 10), n_fp=(0, 1), n_fn=(0, 1),
            >>>     classes=3, cls_noise=.2, newstyle=False)
            >>> cfsn_vecs = dmet.confusion_vectors()
            >>> cm = cfsn_vecs.confusion_matrix()
            ...
            >>> print(cm.to_string(float_format=lambda x: '%.2f' % x))
            pred        background  cat_1  cat_2  cat_3
            real
            background        0.00   1.00   2.00   3.00
            cat_1             3.00  12.00   0.00   0.00
            cat_2             3.00   0.00  14.00   0.00
            cat_3             2.00   0.00   0.00  17.00
        """
        data = cfsn_vecs.data

        y_true = data['true'].copy()
        y_pred = data['pred'].copy()

        # FIXME: hard-coded background class
        if 'background' in cfsn_vecs.classes:
            bg_idx = cfsn_vecs.classes.index('background')
            y_true[y_true < 0] = bg_idx
            y_pred[y_pred < 0] = bg_idx
        else:
            if np.any(y_true < 0):
                raise IndexError('y_true contains invalid indices')
            if np.any(y_pred < 0):
                raise IndexError('y_pred contains invalid indices')

        matrix = fast_confusion_matrix(
            y_true, y_pred, n_labels=len(cfsn_vecs.classes),
            sample_weight=data.get('weight', None)
        )

        import pandas as pd
        cm = pd.DataFrame(matrix, index=list(cfsn_vecs.classes),
                          columns=list(cfsn_vecs.classes))
        if compress:
            iszero = matrix == 0
            unused = (np.all(iszero, axis=0) & np.all(iszero, axis=1))
            cm = cm[~unused].T[~unused].T
        cm.index.name = 'real'
        cm.columns.name = 'pred'
        return cm

    def coarsen(cfsn_vecs, cxs):
        """
        Creates a coarsened set of vectors

        Returns:
            ConfusionVectors
        """
        import kwarray
        assert cfsn_vecs.probs is not None, 'need probs'
        if not isinstance(cfsn_vecs.classes, CategoryTree):
            raise TypeError('classes must be a kwcoco.CategoryTree')

        descendent_map = cfsn_vecs.classes.idx_to_descendants_idxs(include_cfsn_vecs=True)
        valid_descendant_mapping = ub.dict_isect(descendent_map, cxs)
        # mapping from current category indexes to the new coarse ones
        # Anything without an explicit key will be mapped to background

        bg_idx = cfsn_vecs.classes.index('background')
        mapping = {v: k for k, vs in valid_descendant_mapping.items() for v in vs}
        new_true = np.array([mapping.get(x, bg_idx) for x in cfsn_vecs.data['true']])
        new_pred = np.array([mapping.get(x, bg_idx) for x in cfsn_vecs.data['pred']])

        new_score = np.array([p[x] for x, p in zip(new_pred, cfsn_vecs.probs)])

        new_y_df = {
            'true': new_true,
            'pred': new_pred,
            'score': new_score,
            'weight': cfsn_vecs.data['weight'],
            'txs': cfsn_vecs.data['txs'],
            'pxs': cfsn_vecs.data['pxs'],
            'gid': cfsn_vecs.data['gid'],
        }
        new_y_df = kwarray.DataFrameArray(new_y_df)
        coarse_cfsn_vecs = ConfusionVectors(new_y_df, cfsn_vecs.classes, cfsn_vecs.probs)
        return coarse_cfsn_vecs

    def binarize_classless(cfsn_vecs, negative_classes=None):
        """
        Creates a binary representation useful for measuring the performance of
        detectors. It is assumed that scores of "positive" classes should be
        high and "negative" clases should be low.

        Args:
            negative_classes (List[str | int] | None):
                list of negative class names or idxs, by default chooses any
                class with a true class index of -1. These classes should
                ideally have low scores.

        Returns:
            BinaryConfusionVectors

        Note:
            The "classlessness" of this depends on the compat="all" argument
            being used when constructing confusion vectors, otherwise it
            becomes something like a macro-average because the class
            information was used in deciding which true and predicted boxes
            were allowed to match.

        Example:
            >>> from kwcoco.metrics import DetectionMetrics
            >>> dmet = DetectionMetrics.demo(
            >>>     nimgs=10, nboxes=(0, 10), n_fp=(0, 1), n_fn=(0, 1), classes=3)
            >>> cfsn_vecs = dmet.confusion_vectors()
            >>> class_idxs = list(dmet.classes.node_to_idx.values())
            >>> binvecs = cfsn_vecs.binarize_classless()
        """
        import kwarray
        # import warnings
        # warnings.warn('binarize_classless DOES NOT PRODUCE CORRECT RESULTS')

        negative_cidxs = {-1}
        if negative_classes is not None:
            @ub.memoize
            def _lower_classes():
                if cfsn_vecs.classes is None:
                    raise Exception(
                        'classes must be known if negative_classes are strings')
                return [c.lower() for c in cfsn_vecs.classes]
            for c in negative_classes:
                if isinstance(c, str):
                    classes = _lower_classes()
                    try:
                        cidx = classes.index(c)
                    except Exception:
                        continue
                else:
                    cidx = int(c)
                negative_cidxs.add(cidx)

        is_false = kwarray.isect_flags(cfsn_vecs.data['true'], negative_cidxs)

        _data = {
            'is_true': ~is_false,
            'pred_score': cfsn_vecs.data['score'],
        }
        extra = ub.dict_isect(cfsn_vecs.data._data, [
            'txs', 'pxs', 'gid', 'weight'])
        _data.update(extra)
        bin_data = kwarray.DataFrameArray(_data)
        nocls_binvecs = BinaryConfusionVectors(bin_data)
        return nocls_binvecs

    def binarize_ovr(cfsn_vecs,
                     mode=1,
                     keyby='name',
                     ignore_classes={'ignore'},
                     approx=False):
        """
        Transforms cfsn_vecs into one-vs-rest BinaryConfusionVectors for each
        category.

        Args:
            mode (int, default=1): 0 for heirarchy aware or 1 for voc like.
                MODE 0 IS PROBABLY BROKEN

            keyby (int | str) : can be cx or name

            ignore_classes (Set[str]): category names to ignore

            approx (bool, default=0): if True try and approximate missing
                scores otherwise assume they are irrecoverable and use -inf

        Returns:
            OneVsRestConfusionVectors: which behaves like
                Dict[int, BinaryConfusionVectors]: cx_to_binvecs

        Example:
            >>> from kwcoco.metrics.confusion_vectors import *  # NOQA
            >>> cfsn_vecs = ConfusionVectors.demo()
            >>> print('cfsn_vecs = {!r}'.format(cfsn_vecs))
            >>> catname_to_binvecs = cfsn_vecs.binarize_ovr(keyby='name')
            >>> print('catname_to_binvecs = {!r}'.format(catname_to_binvecs))

            cfsn_vecs.data.pandas()
            catname_to_binvecs.cx_to_binvecs['class_1'].data.pandas()

        Note:

            .. code:

                Consider we want to measure how well we can classify beagles.

                Given a multiclass confusion vector, we need to carefully select a
                subset. We ignore any truth that is coarser than our current label.
                We also ignore any background predictions on irrelevant classes

                y_true     | y_pred     | score
                -------------------------------
                dog        | dog          <- ignore coarser truths
                dog        | cat          <- ignore coarser truths
                dog        | beagle       <- ignore coarser truths
                cat        | dog
                cat        | cat
                cat        | background   <- ignore failures to predict unrelated classes
                cat        | maine-coon
                beagle     | beagle
                beagle     | dog
                beagle     | background
                beagle     | cat
                Snoopy     | beagle
                Snoopy     | cat
                maine-coon | background    <- ignore failures to predict unrelated classes
                maine-coon | beagle
                maine-coon | cat

                Anything not marked as ignore is counted. We count anything marked
                as beagle or a finer grained class (e.g.  Snoopy) as a positive
                case. All other cases are negative. The scores come from the
                predicted probability of beagle, which must be remembered outside
                the dataframe.
        """
        import kwarray

        classes = cfsn_vecs.classes
        data = cfsn_vecs.data

        if mode == 0:
            if cfsn_vecs.probs is None:
                raise ValueError('cannot binarize in mode=0 without probs')
            pdist = classes.idx_pairwise_distance()

        cx_to_binvecs = {}
        for cx in range(len(classes)):
            if classes[cx] == 'background' or classes[cx] in ignore_classes:
                continue

            if mode == 0:
                import warnings
                warnings.warn(
                    'THIS CALCLUATION MIGHT BE WRONG. MANY OTHERS '
                    'IN THIS FILE WERE, AND I HAVENT CHECKED THIS ONE YET')

                # Lookup original probability predictions for the class of interest
                new_scores = cfsn_vecs.probs[:, cx]

                # Determine which truth items have compatible classes
                # Note: we ignore any truth-label that is COARSER than the
                # class-of-interest.
                # E.g: how well do we classify Beagle? -> we should ignore any truth
                # label marked as Dog because it may or may not be a Beagle?
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    dist = pdist[cx]
                    coarser_cxs = np.where(dist < 0)[0]
                    finer_eq_cxs = np.where(dist >= 0)[0]

                is_finer_eq = kwarray.isect_flags(data['true'], finer_eq_cxs)
                is_coarser = kwarray.isect_flags(data['true'], coarser_cxs)

                # Construct a binary data frame to pass to sklearn functions.
                bin_data = {
                    'is_true': is_finer_eq.astype(np.uint8),
                    'pred_score': new_scores,
                    'weight': data['weight'] * (np.float32(1.0) - is_coarser),
                    'txs': cfsn_vecs.data['txs'],
                    'pxs': cfsn_vecs.data['pxs'],
                    'gid': cfsn_vecs.data['gid'],
                }
                bin_data = kwarray.DataFrameArray(bin_data)

                # Ignore cases where we failed to predict an irrelevant class
                flags = (data['pred'] == -1) & (bin_data['is_true'] == 0)
                bin_data['weight'][flags] = 0
                # bin_data = bin_data.compress(~flags)
                bin_cfsn = BinaryConfusionVectors(bin_data, cx, classes)

            elif mode == 1:
                # More VOC-like, not heirarchy friendly
                if cfsn_vecs.probs is not None:

                    # TODO: perhaps we shouldn't use these or at least
                    # allow for configuration?

                    # We know the actual score predicted for this category in
                    # this case.
                    is_true = cfsn_vecs.data['true'] == cx
                    pred_score = cfsn_vecs.probs[:, cx]

                    from kwcoco.metrics import assignment
                    if assignment.USE_NEG_INF:
                        pred_score[cfsn_vecs.data['pred'] == -1] = -np.inf
                else:
                    if approx:
                        import warnings
                        warnings.warn(
                            'Binarize ovr is only approximate if not all '
                            'probabilities are known')
                    # If we don't know the probabilities for non-predicted
                    # categories then we have to guess.
                    is_true = cfsn_vecs.data['true'] == cx

                    # do we know the actual predicted score for this category?
                    score_is_unknown = data['pred'] != cx
                    pred_score = data['score'].copy()

                    from kwcoco.metrics import assignment
                    if assignment.USE_NEG_INF:
                        missing_score = -np.inf
                    else:
                        missing_score = 0

                    # These scores were for a different class, so assume
                    # other classes were predicted with a uniform prior
                    if approx == 0:
                        approx_score = np.full(
                            sum(score_is_unknown),
                            fill_value=missing_score)
                    else:
                        approx_score = ((1 - pred_score[score_is_unknown]) /
                                        (len(classes) - 1))

                    # Except in the case where predicted class is -1. In this
                    # case no prediction was actually made (above a threshold)
                    # so the assumed score should be significantly lower, we
                    # conservatively choose zero.
                    unknown_preds = data['pred'][score_is_unknown]
                    approx_score[unknown_preds == -1] = missing_score
                    pred_score[score_is_unknown] = approx_score

                bin_data = {
                    # is_true denotes if the true class of the item is the
                    # category of interest.
                    'is_true': is_true,
                    'pred_score': pred_score,
                }

                extra = ub.dict_isect(data._data, [
                    'txs', 'pxs', 'gid', 'weight'])
                bin_data.update(extra)

                bin_data = kwarray.DataFrameArray(bin_data)
                bin_cfsn = BinaryConfusionVectors(bin_data, cx, classes)
            cx_to_binvecs[cx] = bin_cfsn

        if keyby == 'cx':
            cx_to_binvecs = cx_to_binvecs
        elif keyby == 'name':
            cx_to_binvecs = ub.map_keys(cfsn_vecs.classes, cx_to_binvecs)
        else:
            raise KeyError(keyby)

        ovr_cfns = OneVsRestConfusionVectors(cx_to_binvecs, cfsn_vecs.classes)
        return ovr_cfns

    def classification_report(cfsn_vecs, verbose=0):
        """
        Build a classification report with various metrics.

        Example:
            >>> # xdoctest: +REQUIRES(module:pandas)
            >>> from kwcoco.metrics.confusion_vectors import *  # NOQA
            >>> cfsn_vecs = ConfusionVectors.demo()
            >>> report = cfsn_vecs.classification_report(verbose=1)
        """
        from kwcoco.metrics import clf_report
        y_true = cfsn_vecs.data['true']
        y_pred = cfsn_vecs.data['pred']
        sample_weight = cfsn_vecs.data.get('weight', None)
        target_names = list(cfsn_vecs.classes)
        report = clf_report.classification_report(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
            target_names=target_names,
            verbose=verbose,
        )
        return report


class OneVsRestConfusionVectors(ub.NiceRepr):
    """
    Container for multiple one-vs-rest binary confusion vectors

    Attributes:
        cx_to_binvecs
        classes

    Example:
        >>> from kwcoco.metrics import DetectionMetrics
        >>> dmet = DetectionMetrics.demo(
        >>>     nimgs=10, nboxes=(0, 10), n_fp=(0, 1), classes=3)
        >>> cfsn_vecs = dmet.confusion_vectors()
        >>> self = cfsn_vecs.binarize_ovr(keyby='name')
        >>> print('self = {!r}'.format(self))
    """
    def __init__(self, cx_to_binvecs, classes):
        self.cx_to_binvecs = cx_to_binvecs
        self.classes = classes

    def __nice__(self):
        return ub.urepr(self.cx_to_binvecs, strvals=True, align=':')

    @classmethod
    def demo(cls):
        """
        Args:
            **kwargs: See :func:`kwcoco.metrics.DetectionMetrics.demo`

        Returns:
            ConfusionVectors
        """
        cfsn_vecs = ConfusionVectors.demo(n_fp=(0, 10), n_fn=(0, 10), classes=3)
        self = cfsn_vecs.binarize_ovr(keyby='name')
        return self

    def keys(self):
        return self.cx_to_binvecs.keys()

    def __getitem__(self, cx):
        return self.cx_to_binvecs[cx]

    def measures(self, stabalize_thresh=7, fp_cutoff=None, monotonic_ppv=True,
                 ap_method='pycocotools'):
        """
        Creates binary confusion measures for every one-versus-rest category.

        Args:
            stabalize_thresh (int):
                if fewer than this many data points inserts dummy stabilization
                data so curves can still be drawn. Default to 7.

            fp_cutoff (int | None):
                maximum number of false positives in the truncated roc curves.
                The default ``None`` is equivalent to ``float('inf')``

            monotonic_ppv (bool):
                if True ensures that precision is always increasing as recall
                decreases. This is done in pycocotools scoring, but I'm not
                sure its a good idea. Default to True.

        SeeAlso:
            :func:`BinaryConfusionVectors.measures`

        Example:
            >>> self = OneVsRestConfusionVectors.demo()
            >>> thresh_result = self.measures()['perclass']
        """
        perclass = PerClass_Measures({
            cx: binvecs.measures(
                stabalize_thresh=stabalize_thresh,
                fp_cutoff=fp_cutoff,
                monotonic_ppv=monotonic_ppv,
                ap_method=ap_method,
            )
            for cx, binvecs in self.cx_to_binvecs.items()
        })
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            mAUC = np.nanmean([item['trunc_auc'] for item in perclass.values()])
            mAP = np.nanmean([item['ap'] for item in perclass.values()])
        return {
            'mAP': mAP,
            'mAUC': mAUC,
            'perclass': perclass,
        }

    def ovr_classification_report(self):
        raise NotImplementedError


class BinaryConfusionVectors(ub.NiceRepr):
    """
    Stores information about a binary classification problem.
    This is always with respect to a specific class, which is given
    by `cx` and `classes`.

    The `data` DataFrameArray must contain
        `is_true` - if the row is an instance of class `classes[cx]`
        `pred_score` - the predicted probability of class `classes[cx]`, and
        `weight` - sample weight of the example

    Example:
        >>> from kwcoco.metrics.confusion_vectors import *  # NOQA
        >>> self = BinaryConfusionVectors.demo(n=10)
        >>> print('self = {!r}'.format(self))
        >>> print('measures = {}'.format(ub.urepr(self.measures())))

        >>> self = BinaryConfusionVectors.demo(n=0)
        >>> print('measures = {}'.format(ub.urepr(self.measures())))

        >>> self = BinaryConfusionVectors.demo(n=1)
        >>> print('measures = {}'.format(ub.urepr(self.measures())))

        >>> self = BinaryConfusionVectors.demo(n=2)
        >>> print('measures = {}'.format(ub.urepr(self.measures())))
    """

    def __init__(self, data, cx=None, classes=None):
        import kwarray
        if data is None:
            # Init an empty object
            data = kwarray.DataFrameArray({
                'is_true': np.empty((0,), dtype=bool),
                'pred_score': np.empty((0,), dtype=float),
                'weight': np.empty((0,), dtype=float),
            })
        elif isinstance(data, dict):
            data = kwarray.DataFrameArray(ub.map_vals(np.asarray, data))

        self.data = data
        self.cx = cx
        self.classes = classes

    @classmethod
    def demo(cls, n=10, p_true=0.5, p_error=0.2, p_miss=0.0, rng=None):
        """
        Create random data for tests

        Args:
            n (int): number of rows
            p_true (float): fraction of real positive cases
            p_error (float): probability of making a recoverable mistake
            p_miss (float): probability of making a unrecoverable mistake
            rng (int | RandomState | None): random seed / state

        Returns:
            BinaryConfusionVectors

        Example:
            >>> from kwcoco.metrics.confusion_vectors import *  # NOQA
            >>> cfsn = BinaryConfusionVectors.demo(n=1000, p_error=0.1, p_miss=0.1)
            >>> measures = cfsn.measures()
            >>> print('measures = {}'.format(ub.urepr(measures, nl=1)))
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1, pnum=(1, 2, 1))
            >>> measures.draw('pr')
            >>> kwplot.figure(fnum=1, pnum=(1, 2, 2))
            >>> measures.draw('roc')
        """
        import kwarray
        rng = kwarray.ensure_rng(rng)
        score = rng.rand(n)

        data = kwarray.DataFrameArray({
            'is_true': (score > p_true).astype(np.uint8),
            'pred_score': score,
        })

        flip_flags = rng.rand(n) < p_error
        data['is_true'][flip_flags] = 1 - data['is_true'][flip_flags]

        miss_flags = rng.rand(n) < p_miss
        data['pred_score'][miss_flags] = -np.inf

        classes = ['c1', 'c2', 'c3']
        self = cls(data, cx=1, classes=classes)
        return self

    @property
    def catname(self):
        if self.cx is None:
            return None
        return self.classes[self.cx]

    def __nice__(self):
        return ub.urepr({
            'catname': self.catname,
            'data': self.data.__nice__(),
        }, nl=0, strvals=True, align=':')

    def __len__(self):
        return len(self.data)

    # @ub.memoize_method
    def measures(self, stabalize_thresh=7, fp_cutoff=None, monotonic_ppv=True,
                 ap_method='pycocotools'):
        """
        Get statistics (F1, G1, MCC) versus thresholds

        Args:
            stabalize_thresh (int, default=7):
                if fewer than this many data points inserts dummy stabalization
                data so curves can still be drawn.

            fp_cutoff (int | None):
                maximum number of false positives in the truncated roc curves.
                The default of ``None`` is equivalent to ``float('inf')``

            monotonic_ppv (bool):
                if True ensures that precision is always increasing as recall
                decreases. This is done in pycocotools scoring, but I'm not
                sure its a good idea.

        Example:
            >>> from kwcoco.metrics.confusion_vectors import *  # NOQA
            >>> self = BinaryConfusionVectors.demo(n=0)
            >>> print('measures = {}'.format(ub.urepr(self.measures())))
            >>> self = BinaryConfusionVectors.demo(n=1, p_true=0.5, p_error=0.5)
            >>> print('measures = {}'.format(ub.urepr(self.measures())))
            >>> self = BinaryConfusionVectors.demo(n=3, p_true=0.5, p_error=0.5)
            >>> print('measures = {}'.format(ub.urepr(self.measures())))

            >>> self = BinaryConfusionVectors.demo(n=100, p_true=0.5, p_error=0.5, p_miss=0.3)
            >>> print('measures = {}'.format(ub.urepr(self.measures())))
            >>> print('measures = {}'.format(ub.urepr(ub.odict(self.measures()))))

        References:
            https://en.wikipedia.org/wiki/Confusion_matrix
            https://en.wikipedia.org/wiki/Precision_and_recall
            https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

        Ignore:
            # import matplotlib.cm as cm
            # kwargs = {}
            # cmap = kwargs.get('cmap', mpl.cm.coolwarm)
            # n = len(x)
            # xgrid = np.tile(x[None, :], (n, 1))
            # ygrid = np.tile(y[None, :], (n, 1))
            # zdata = np.tile(z[None, :], (n, 1))
            # ax.contour(xgrid, ygrid, zdata, zdir='x', cmap=cmap)
            # ax.contour(xgrid, ygrid, zdata, zdir='y', cmap=cmap)
            # ax.contour(xgrid, ygrid, zdata, zdir='z', cmap=cmap)

            self.measures().summary_plot()
            import xdev
            globals().update(xdev.get_func_kwargs(BinaryConfusionVectors.measures._func))
        """
        # compute tp, fp, tn, fn at each operating point
        # compute mcc, f1, g1, etc
        info = self._binary_clf_curves(stabalize_thresh=stabalize_thresh,
                                       fp_cutoff=fp_cutoff)
        info['monotonic_ppv'] = monotonic_ppv
        info['ap_method'] = ap_method
        info['cx'] = self.cx
        info['classes'] = self.classes
        measures = Measures(info)
        populate_info(info)
        return measures

    @profile
    def _binary_clf_curves(self, stabalize_thresh=7, fp_cutoff=None):
        """
        Compute TP, FP, TN, and FN counts for this binary confusion vector.

        Code common to ROC, PR, and threshold measures, computes the elements
        of the binary confusion matrix at all relevant operating point
        thresholds.

        Args:
            stabalize_thresh (int): if fewer than this many data points insert
                stabalization data.

            fp_cutoff (int | None): maximum number of false positives

        Example:
            >>> from kwcoco.metrics.confusion_vectors import *  # NOQA
            >>> self = BinaryConfusionVectors.demo(n=1, p_true=0.5, p_error=0.5)
            >>> self._binary_clf_curves()

            >>> self = BinaryConfusionVectors.demo(n=0, p_true=0.5, p_error=0.5)
            >>> self._binary_clf_curves()

            >>> self = BinaryConfusionVectors.demo(n=100, p_true=0.5, p_error=0.5)
            >>> self._binary_clf_curves()

        Ignore:
            import xdev
            globals().update(xdev.get_func_kwargs(BinaryConfusionVectors._binary_clf_curves))
            >>> self = BinaryConfusionVectors.demo(n=10, p_true=0.7, p_error=0.3, p_miss=0.2)
            >>> print('measures = {}'.format(ub.urepr(self._binary_clf_curves())))
            >>> info = self.measures()
            >>> info = ub.dict_isect(info, ['tpr', 'fpr', 'ppv', 'fp_count'])
            >>> print('measures = {}'.format(ub.urepr(ub.odict(i))))
        """
        # try:
        #     from sklearn.metrics._ranking import _binary_clf_curve
        # except ImportError:
        #     from sklearn.metrics.ranking import _binary_clf_curve
        data = self.data
        y_true = data['is_true'].astype(np.uint8)
        y_score = data['pred_score']
        sample_weight = data._data.get('weight', None)

        npad = 0
        if len(self) == 0:
            fps = np.array([np.nan])
            fns = np.array([np.nan])
            tps = np.array([np.nan])
            thresholds = np.array([np.nan])

            realpos_total = 0
            realneg_total = 0
            nsupport = 0
        else:
            if len(self) < stabalize_thresh:
                # add dummy data to stabalize the computation
                if sample_weight is None:
                    sample_weight = np.ones(len(self))
                npad = stabalize_thresh - len(self)
                y_true, y_score, sample_weight = _stabalize_data(
                    y_true, y_score, sample_weight, npad=npad)

            # Get the total weight (typically number of) positive and negative
            # examples of this class
            if sample_weight is None:
                weight = 1
                nsupport = len(y_true) - bool(npad)
            else:
                weight = sample_weight
                nsupport = sample_weight.sum() - bool(npad)

            realpos_total = (y_true * weight).sum()
            realneg_total = ((1 - y_true) * weight).sum()

            fps, tps, thresholds = _binary_clf_curve2(
                y_true, y_score, pos_label=1.0,
                sample_weight=sample_weight)

            # Adjust weighted totals to be robust to floating point errors
            if np.isclose(realneg_total, fps[-1]):
                realneg_total = max(realneg_total, fps[-1])
            if np.isclose(realpos_total, tps[-1]):
                realpos_total = max(realpos_total, tps[-1])

            fps[thresholds == -np.inf] = np.inf

        tns = realneg_total - fps
        fns = realpos_total - tps

        info = {
            'fp_count': fps,
            'tp_count': tps,
            'tn_count': tns,
            'fn_count': fns,
            'thresholds': thresholds,
            'realpos_total': realpos_total,
            'realneg_total': realneg_total,

            'nsupport': nsupport,

            'fp_cutoff': fp_cutoff,
            'stabalize_thresh': stabalize_thresh,
        }

        if self.cx is not None:
            info.update({
                'cx': self.cx,
                'node': self.classes[self.cx],
            })
        return info

    def draw_distribution(self):
        data = self.data
        y_true = data['is_true'].astype(np.uint8)
        y_score = data['pred_score']

        y_true = y_true.astype(bool)

        nbins = 100
        all_freq, xdata = np.histogram(y_score, nbins)
        raw_scores = {
            'true': y_score[y_true],
            'false': y_score[~y_true],
        }
        color = {
            'true': 'dodgerblue',
            'false': 'red'
        }
        ydata = {k: np.histogram(v, bins=xdata)[0]
                 for k, v in raw_scores.items()}
        import kwplot
        return kwplot.multi_plot(xdata=xdata, ydata=ydata, color=color)

    def _3dplot(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> # xdoctest: +REQUIRES(module:pandas)
            >>> from kwcoco.metrics.confusion_vectors import *  # NOQA
            >>> from kwcoco.metrics.detect_metrics import DetectionMetrics
            >>> dmet = DetectionMetrics.demo(
            >>>     n_fp=(0, 1), n_fn=(0, 2), nimgs=256, nboxes=(0, 10),
            >>>     bbox_noise=10,
            >>>     classes=1)
            >>> cfsn_vecs = dmet.confusion_vectors()
            >>> self = bin_cfsn = cfsn_vecs.binarize_classless()
            >>> #dmet.summarize(plot=True)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=3)
            >>> self._3dplot()
        """
        from mpl_toolkits.mplot3d import Axes3D  # NOQA
        import matplotlib as mpl  # NOQA
        import matplotlib.pyplot as plt
        info = self.measures()

        thresh = info['thresholds']
        flags = thresh > -np.inf
        tpr = info['tpr'][flags]
        fpr = info['fpr'][flags]
        ppv = info['ppv'][flags]

        # kwargs = {}
        # cmap = kwargs.get('cmap', mpl.cm.coolwarm)
        # cmap = kwargs.get('cmap', mpl.cm.plasma)
        # cmap = kwargs.get('cmap', mpl.cm.hot)
        # cmap = kwargs.get('cmap', mpl.cm.magma)
        fig = plt.gcf()
        fig.clf()
        ax = fig.add_subplot(projection='3d')

        x = tpr
        y = fpr
        z = ppv

        # mcc_color = cmap(np.maximum(info['mcc'][flags], 0))[:, 0:3]

        ax.plot3D(xs=x, ys=[0] * len(y), zs=z, c='orange')
        ax.plot3D(xs=x, ys=y, zs=[1] * len(z), c='pink')
        ax.plot3D(xs=x, ys=y, zs=z, c='blue')
        # ax.scatter(x, y, z, c=mcc_color)
        # ax.scatter(x, y, [1] * len(z), c=mcc_color)
        # ax.scatter(x, [0] * len(y), z, c=mcc_color)

        ax.set_title('roc + PR')
        ax.set_xlabel('tpr')
        ax.set_ylabel('fpr')
        ax.set_zlabel('ppv')

        # rotate the axes and update
        if 0:
            for angle in range(0, 360):
                ax.view_init(30, angle)
                plt.draw()
                plt.pause(.001)

        # TODO: improve this visualization, can we color the lines better /
        # fill in the meshes with something meaningful?
        # Color the main contour line by MCC,
        # Color the ROC line by PPV
        # color the PR line by FPR

    # def precision_recall(self, stabalize_thresh=7, method='sklearn'):
    #     """
    #     Deprecated, all information lives in measures now
    #     """
    #     warnings.warn('use measures instead', DeprecationWarning)
    #     measures = self.measures(
    #         fp_cutoff=None, stabalize_thresh=stabalize_thresh
    #     )
    #     return measures

    # def roc(self, fp_cutoff=None, stabalize_thresh=7):
    #     """
    #     Deprecated, all information lives in measures now
    #     """
    #     warnings.warn('use measures instead', DeprecationWarning)
    #     info = self.measures(
    #         fp_cutoff=fp_cutoff, stabalize_thresh=stabalize_thresh)
    #     return info


def _stabalize_data(y_true, y_score, sample_weight, npad=7):
    """
    Adds ideally calibrated dummy values to curves with few positive examples.
    This acts somewhat like a Bayesian prior and smooths out the curve.

    Example:
        y_score = np.array([0.5, 0.6])
        y_true = np.array([1, 1])
        sample_weight = np.array([1, 1])
        npad = 7
        _stabalize_data(y_true, y_score, sample_weight, npad=npad)
    """
    finite_scores = y_score[np.isfinite(y_score)]

    if len(finite_scores):
        min_score = finite_scores.min()
        max_score = finite_scores.max()
    else:
        min_score = 0
        max_score = 1

    if max_score <= 1.0 and min_score >= 0.0:
        max_score = 1.0
        min_score = 0.0

    pad_true = np.ones(npad, dtype=np.uint8)
    pad_true[:npad // 2] = 0

    pad_score = np.linspace(min_score, max_score, num=npad,
                            endpoint=True)
    pad_weight = np.exp(np.linspace(2.7, .01, npad))
    pad_weight /= pad_weight.sum()

    y_true = np.hstack([y_true, pad_true])
    y_score = np.hstack([y_score, pad_score])
    sample_weight = np.hstack([sample_weight, pad_weight])
    return y_true, y_score, sample_weight


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/kwcoco/metrics/confusion_vectors.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
