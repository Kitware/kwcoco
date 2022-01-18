"""
Classes that store accumulated confusion measures (usually derived from
confusion vectors).


For each chosen threshold value:
    * thresholds[i] - the i-th threshold value


The primary data we manipulate are arrays of "confusion" counts, i.e.

    * tp_count[i] - true positives at the i-th threshold
    * fp_count[i] - false positives at the i-th threshold
    * fn_count[i] - false negatives at the i-th threshold
    * tn_count[i] - true negatives at the i-th threshold

"""
import kwarray
import numpy as np
import ubelt as ub
import warnings
from kwcoco.metrics.util import DictProxy


class Measures(ub.NiceRepr, DictProxy):
    """
    Holds accumulated confusion counts, and derived measures

    Example:
        >>> from kwcoco.metrics.confusion_vectors import BinaryConfusionVectors  # NOQA
        >>> binvecs = BinaryConfusionVectors.demo(n=100, p_error=0.5)
        >>> self = binvecs.measures()
        >>> print('self = {!r}'.format(self))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> self.draw(doclf=True)
        >>> self.draw(key='pr',  pnum=(1, 2, 1))
        >>> self.draw(key='roc', pnum=(1, 2, 2))
        >>> kwplot.show_if_requested()
    """
    def __init__(self, info):
        self.proxy = info

    @property
    def catname(self):
        return self.get('node', None)

    def __nice__(self):
        info = self.summary()
        return ub.repr2(info, nl=0, precision=3, strvals=True, align=':')

    def reconstruct(self):
        populate_info(info=self)
        return self

    @classmethod
    def from_json(cls, state):
        populate_info(state)
        return cls(state)

    def __json__(self):
        """
        Example:
            >>> from kwcoco.metrics.confusion_vectors import BinaryConfusionVectors  # NOQA
            >>> binvecs = BinaryConfusionVectors.demo(n=10, p_error=0.5)
            >>> self = binvecs.measures()
            >>> info = self.__json__()
            >>> print('info = {}'.format(ub.repr2(info, nl=1)))
            >>> populate_info(info)
            >>> print('info = {}'.format(ub.repr2(info, nl=1)))
            >>> recon = Measures.from_json(info)
        """
        from kwcoco.util.util_json import ensure_json_serializable
        state = {}
        minimal = {
            'fp_count',
            'tp_count',
            'tn_count',
            'fn_count',
            'thresholds',
            'realpos_total',
            'realneg_total',

            'trunc_idx',

            'nsupport',

            'fp_cutoff',
            'stabalize_thresh',
            'cx',
            'node',
            'monotonic_ppv',
        }

        nice_to_have = {
            #  recomputable
            # 'mcc', 'g1', 'f1', 'ppv', 'tpr', 'fpr', 'acc', 'bm', 'mk',
            # 'max_mcc', '_max_mcc', 'max_g1', '_max_g1', 'max_f1',
            # '_max_f1', 'max_acc', '_max_acc',
            # 'trunc_tpr', 'trunc_fpr',

            'trunc_auc', 'auc', 'ap',
            # 'sklish_ap',
            'pycocotools_ap',
            # 'outlier_ap', 'sklearn',
        }
        state = ub.dict_isect(self.proxy, minimal | nice_to_have)
        state = ensure_json_serializable(state)
        return state

    def summary(self):
        return {
            'ap': self.get('ap', None),
            'auc': self.get('auc', None),
            # 'max_mcc': self['max_mcc'],
            'max_f1': self.get('max_f1', None),
            # 'max_g1': self['max_g1'],
            'nsupport': self.get('nsupport', None),
            'realpos_total': self.get('realpos_total', None),
            # 'realneg_total': self['realneg_total'],
            'catname': self.get('node', None),
        }

    def counts(self):
        counts_df = ub.dict_isect(self, [
            'fp_count', 'tp_count', 'tn_count', 'fn_count', 'thresholds'])
        counts_df = kwarray.DataFrameArray(counts_df)
        return counts_df

    def draw(self, key=None, prefix='', **kw):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> # xdoctest: +REQUIRES(module:pandas)
            >>> from kwcoco.metrics.confusion_vectors import ConfusionVectors  # NOQA
            >>> cfsn_vecs = ConfusionVectors.demo()
            >>> ovr_cfsn = cfsn_vecs.binarize_ovr(keyby='name')
            >>> self = ovr_cfsn.measures()['perclass']
            >>> self.draw('mcc', doclf=True, fnum=1)
            >>> self.draw('pr', doclf=1, fnum=2)
            >>> self.draw('roc', doclf=1, fnum=3)
        """
        from kwcoco.metrics import drawing
        if key is None or key == 'thresh':
            return drawing.draw_threshold_curves(self, prefix=prefix, **kw)
        elif key == 'pr':
            return drawing.draw_prcurve(self, prefix=prefix, **kw)
        elif key == 'roc':
            return drawing.draw_roc(self, prefix=prefix, **kw)

    def summary_plot(self, fnum=1, title='', subplots='auto'):
        """
        Example:
            >>> from kwcoco.metrics.confusion_measures import *  # NOQA
            >>> from kwcoco.metrics.confusion_vectors import ConfusionVectors  # NOQA
            >>> cfsn_vecs = ConfusionVectors.demo(n=3, p_error=0.5)
            >>> binvecs = cfsn_vecs.binarize_classless()
            >>> self = binvecs.measures()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> self.summary_plot()
            >>> kwplot.show_if_requested()
        """
        if subplots == 'auto':
            subplots = ['pr', 'roc', 'thresh']
        import kwplot
        kwplot.figure(fnum=fnum, figtitle=title)

        pnum_ = kwplot.PlotNums(nSubplots=len(subplots))
        for key in subplots:
            kwplot.figure(fnum=fnum, pnum=pnum_())
            self.draw(key, fnum=fnum)

            # kwplot.figure(fnum=fnum, pnum=(1, 3, 2))
            # self.draw('roc')
            # kwplot.figure(fnum=fnum, pnum=(1, 3, 3))
            # self.draw('thresh', keys=['mcc', 'f1', 'acc'])

    @classmethod
    def demo(cls, **kwargs):
        """
        Create a demo Measures object for testing / demos

        Args:
            **kwargs: passed to :func:`BinaryConfusionVectors.demo`.
                some valid keys are: n, rng, p_rue, p_error, p_miss.
        """
        from kwcoco.metrics.confusion_vectors import BinaryConfusionVectors
        bin_cfsn = BinaryConfusionVectors.demo(**kwargs)
        measures = bin_cfsn.measures()
        return measures

    @classmethod
    def combine(cls, tocombine, precision=None, growth=None, thresh_bins=None):
        """
        Combine binary confusion metrics

        Args:
            tocombine (List[Measures]):
                a list of measures to combine into one

            precision (int | None):
                If specified rounds thresholds to this precision which can
                prevent a RAM explosion when combining a large number of
                measures. However, this is a lossy operation and will impact
                the underlying scores. NOTE: use ``growth`` instead.

            growth (int | None):
                if specified this limits how much the resulting measures
                are allowed to grow by. If None, growth is unlimited.
                Otherwise, if growth is 'max', the growth is limited to the
                maximum length of an input. We might make this more numerical
                in the future.

            thresh_bins (int):
                Force this many threshold bins.

        Returns:
            Measures

        Example:
            >>> from kwcoco.metrics.confusion_measures import *  # NOQA
            >>> measures1 = Measures.demo(n=15)
            >>> measures2 = measures1
            >>> tocombine = [measures1, measures2]
            >>> new_measures = Measures.combine(tocombine)
            >>> new_measures.reconstruct()
            >>> print('new_measures = {!r}'.format(new_measures))
            >>> print('measures1 = {!r}'.format(measures1))
            >>> print('measures2 = {!r}'.format(measures2))
            >>> print(ub.repr2(measures1.__json__(), nl=1, sort=0))
            >>> print(ub.repr2(measures2.__json__(), nl=1, sort=0))
            >>> print(ub.repr2(new_measures.__json__(), nl=1, sort=0))
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1)
            >>> new_measures.summary_plot()
            >>> measures1.summary_plot()
            >>> measures1.draw('roc')
            >>> measures2.draw('roc')
            >>> new_measures.draw('roc')

        Ignore:
            >>> from kwcoco.metrics.confusion_measures import *  # NOQA
            >>> rng = kwarray.ensure_rng(0)
            >>> tocombine = [
            >>>     Measures.demo(n=rng.randint(40, 50), rng=rng, p_true=0.2, p_error=0.4, p_miss=0.6)
            >>>     for _ in range(80)
            >>> ]
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> for idx, growth in enumerate([None, 'max', 'log', 'root', 'half']):
            >>>     combo = Measures.combine(tocombine, growth=growth).reconstruct()
            >>>     print('growth = {!r}'.format(growth))
            >>>     print('combo = {}'.format(ub.repr2(combo, nl=1)))
            >>>     print('num_thresholds = {}'.format(len(combo['thresholds'])))
            >>>     combo.summary_plot(fnum=idx + 1, title=str(growth))

        Example:
            >>> # Demonstrate issues that can arrise from choosing a precision
            >>> # that is too low  when combining metrics. Breakpoints
            >>> # between different metrics can get muddled, but choosing a
            >>> # precision that is too high can overwhelm memory.
            >>> from kwcoco.metrics.confusion_measures import *  # NOQA
            >>> base = ub.map_vals(np.asarray, {
            >>>     'tp_count':   [ 1,  1,  2,  2,  2,  2,  3],
            >>>     'fp_count':   [ 0,  1,  1,  2,  3,  4,  5],
            >>>     'fn_count':   [ 1,  1,  0,  0,  0,  0,  0],
            >>>     'tn_count':   [ 5,  4,  4,  3,  2,  1,  0],
            >>>     'thresholds': [.0, .0, .0, .0, .0, .0, .0],
            >>> })
            >>> # Make tiny offsets to thresholds
            >>> rng = kwarray.ensure_rng(0)
            >>> n = len(base['thresholds'])
            >>> offsets = [
            >>>     sorted(rng.rand(n) * 10 ** -rng.randint(4, 7))[::-1]
            >>>     for _ in range(20)
            >>> ]
            >>> tocombine = []
            >>> for offset in offsets:
            >>>     base_n = base.copy()
            >>>     base_n['thresholds'] += offset
            >>>     measures_n = Measures(base_n).reconstruct()
            >>>     tocombine.append(measures_n)
            >>> for precision in [6, 5, 2]:
            >>>     combo = Measures.combine(tocombine, precision=precision).reconstruct()
            >>>     print('precision = {!r}'.format(precision))
            >>>     print('combo = {}'.format(ub.repr2(combo, nl=1)))
            >>>     print('num_thresholds = {}'.format(len(combo['thresholds'])))
            >>> for growth in [None, 'max', 'log', 'root', 'half']:
            >>>     combo = Measures.combine(tocombine, growth=growth).reconstruct()
            >>>     print('growth = {!r}'.format(growth))
            >>>     print('combo = {}'.format(ub.repr2(combo, nl=1)))
            >>>     print('num_thresholds = {}'.format(len(combo['thresholds'])))
            >>>     #print(combo.counts().pandas())

        Example:
            >>> # Test case: combining a single measures should leave it unchanged
            >>> from kwcoco.metrics.confusion_measures import *  # NOQA
            >>> measures = Measures.demo(n=40, p_true=0.2, p_error=0.4, p_miss=0.6)
            >>> df1 = measures.counts().pandas().fillna(0)
            >>> print(df1)
            >>> tocombine = [measures]
            >>> combo = Measures.combine(tocombine)
            >>> df2 = combo.counts().pandas().fillna(0)
            >>> print(df2)
            >>> assert np.allclose(df1, df2)

            >>> combo = Measures.combine(tocombine, thresh_bins=2)
            >>> df3 = combo.counts().pandas().fillna(0)
            >>> print(df3)

            >>> # I am NOT sure if this is correct or not
            >>> combo = Measures.combine(tocombine, thresh_bins=20)
            >>> df4 = combo.counts().pandas().fillna(0)
            >>> print(df4)

            assert np.allclose(combo['thresholds'], measures['thresholds'])
            assert np.allclose(combo['fp_count'], measures['fp_count'])
            assert np.allclose(combo['tp_count'], measures['tp_count'])
            assert np.allclose(combo['tp_count'], measures['tp_count'])
        """
        if precision is not None and np.isinf(precision):
            precision = None

        # For each item to combine, we merge all unique considered thresholds
        # into a single array, and then reconstruct the confusion counts with
        # respect to this new threshold array.
        tocombine_thresh = [m['thresholds'] for m in tocombine]

        orig_thresholds = np.unique(np.hstack(tocombine_thresh))
        finite_flags = np.isfinite(orig_thresholds)
        nonfinite_flags = ~finite_flags
        if np.any(nonfinite_flags):
            orig_finite_thresholds = orig_thresholds[finite_flags]
            # If we have any non-finite threshold values we are stuck with them
            nonfinite_vals = orig_thresholds[nonfinite_flags]
        else:
            orig_finite_thresholds = orig_thresholds
            nonfinite_vals = []

        if thresh_bins is not None:
            assert growth is None
            assert precision is None
            threshold_pool = orig_finite_thresholds
            # Subdivide threshold pool until we get enough values
            while thresh_bins > len(threshold_pool):
                # x = [1, 2, 3, 4, 5]
                # x = [1, 2, 3, 4, 5, 6]
                # even_slice = threshold_pool[:(len(threshold_pool) // 2) * 2]
                # odd_slice = threshold_pool[1:(len(threshold_pool) // 2) * 2]
                x = threshold_pool
                even_slice = x[:(len(x) // 2) * 2]
                odd_slice = x[1:len(x) - (len(x) % 2 == 0)]
                mean_vals = np.r_[[0, threshold_pool[0]], even_slice, odd_slice, [1, threshold_pool[-1]]].reshape(-1, 2).mean(axis=1)
                threshold_pool = np.hstack([mean_vals, threshold_pool])
                threshold_pool = np.unique(threshold_pool)

            chosen_idxs = np.linspace(0, len(threshold_pool) - 1, thresh_bins).round().astype(int)
            chosen_idxs = np.unique(chosen_idxs)
            combo_thresh_asc = threshold_pool[chosen_idxs]
        elif growth is not None:
            if precision is not None:
                raise ValueError('dont use precision with growth')
            # Keep a minimum number of threshold and allow
            # the resulting metrics to grow by some factor
            num_per_item = list(map(len, tocombine_thresh))
            if growth == 'max':
                max_num_thresholds = max(num_per_item)
            elif growth == 'min':
                max_num_thresholds = min(num_per_item)
            elif growth == 'log':
                # Each item after the first is only allowed to contribute
                # some of its thresholds defined by the base-e-log.
                modulated_sizes = sorted([np.log(n) for n in num_per_item])
                max_num_thresholds = int(round(sum(modulated_sizes[:-1]) + max(num_per_item)))
            elif growth == 'root':
                # Each item after the first is only allowed to contribute
                # some of its thresholds defined by the square root.
                modulated_sizes = sorted([np.sqrt(n) for n in num_per_item])
                max_num_thresholds = int(round(sum(modulated_sizes[:-1]) + max(num_per_item)))
            elif growth == 'half':
                # Each item after the first is only allowed to contribute
                # some of its thresholds, half of them.
                modulated_sizes = sorted([n / 2 for n in num_per_item])
                max_num_thresholds = int(round(sum(modulated_sizes[:-1]) + max(num_per_item)))
            else:
                raise KeyError(growth)
            chosen_idxs = np.linspace(0, len(orig_finite_thresholds) - 1, max_num_thresholds).round().astype(int)
            chosen_idxs = np.unique(chosen_idxs)
            combo_thresh_asc = orig_finite_thresholds[chosen_idxs]
        elif precision is not None:
            round_thresholds = orig_finite_thresholds.round(precision)
            combo_thresh_asc = np.unique(round_thresholds)
        else:
            combo_thresh_asc = orig_finite_thresholds

        if nonfinite_vals:
            # Force inclusion of non-finite thresholds
            combo_thresh_asc = np.hstack([combo_thresh_asc, nonfinite_vals])
            combo_thresh_asc.sort()

        new_thresh = combo_thresh_asc[::-1]

        # From this point on, the rest of the logic should (if the
        # implementation is correct) work with respect to an arbitrary choice
        # of threshold bins, thus the critical step is choosing what these bins
        # should be in order to minimize lossyness of the combined accumulated
        # measures while also keeping a reasonable memory footprint.
        # Regardless, it might be good to double check this logic.

        summable = {
            'nsupport': 0,
            'realpos_total': 0,
            'realneg_total': 0,
        }

        # Initialize new counts for each entry in the new threshold array
        # XX_accum[idx] represents the number of *new* XX cases at index idx
        fp_accum = np.zeros(len(new_thresh))
        tp_accum = np.zeros(len(new_thresh))
        tn_accum = np.zeros(len(new_thresh))
        fn_accum = np.zeros(len(new_thresh))

        # For each item to combine, find where its counts should go in the new
        # combined confusion array.
        for measures in tocombine:

            # this thresh is descending
            thresholds = measures['thresholds']
            # XX_pos[idx] is the number of *new* XX cases at index idx at
            # thresholds[idx] for *these* measures.

            fp_pos, _, _ = reversable_diff(measures['fp_count'])
            tp_pos, _, _ = reversable_diff(measures['tp_count'])

            tn_pos, tn_p, tn_s = reversable_diff(measures['tn_count'], reverse=1)
            fn_pos, _, _ = reversable_diff(measures['fn_count'], reverse=1)

            # if 0:
            #     arr = measures['tn_count']
            #     diff_arr = tn_pos
            #     prefix = tn_p
            #     suffix = tn_s
            #     offset = tn_offset
            #     recon_arr = np.cumsum(diff_arr[::-1])[::-1] + offset
            #     recon_arr[0:len(prefix)] += prefix
            #     recon_arr[len(recon_arr) - len(suffix):] += suffix

            # # NOTE: if the min is non-zero in each array the diff wont work
            # # reformulate if this case arrises
            # fp_pos = np.diff(np.r_[[0], measures['fp_count']])
            # tp_pos = np.diff(np.r_[[0], measures['tp_count']])
            # tn_pos = np.diff(np.r_[[0], measures['tn_count'][::-1]])[::-1]
            # fn_pos = np.diff(np.r_[[0], measures['fn_count'][::-1]])[::-1]

            # Is this correct? Do we round *this* measure's thresholds?
            # before comparing to the new threshold array?
            # if precision is not None:
            #     thresholds = thresholds.round(precision)

            # Find the locations where the thresholds from "these" measures
            # should be inserted into the combined measures.
            right_idxs = np.searchsorted(combo_thresh_asc, thresholds, 'right')
            left_idxs = len(combo_thresh_asc) - right_idxs
            left_idxs = left_idxs.clip(0, len(combo_thresh_asc) - 1)

            # Accumulate these values from these measures into the appropriate
            # new threshold position. note: np.add.at is unbuffered
            np.add.at(fp_accum, left_idxs, fp_pos)
            np.add.at(tp_accum, left_idxs, tp_pos)
            np.add.at(fn_accum, left_idxs, fn_pos)
            np.add.at(tn_accum, left_idxs, tn_pos)

            for k in summable:
                summable[k] += measures[k]

        new_fp = np.cumsum(fp_accum)  # will increase with index
        new_tp = np.cumsum(tp_accum)  # will increase with index
        new_tn = np.cumsum(tn_accum[::-1])[::-1]  # will decrease with index
        new_fn = np.cumsum(fn_accum[::-1])[::-1]  # will decrease with index

        if -np.inf in nonfinite_vals:
            # Not sure if this is right...
            new_fp[-1] = np.inf
            new_tn[-1] = -np.inf

        new_info = {
            'fp_count': new_fp,
            'tp_count': new_tp,
            'tn_count': new_tn,
            'fn_count': new_fn,
            'thresholds': new_thresh,
        }
        new_info.update(summable)

        other = {
            'fp_cutoff',
            'monotonic_ppv',
            'node',
            'cx',
            'stabalize_thresh',
            'trunc_idx'
        }
        rest = ub.dict_isect(tocombine[0], other)
        new_info.update(rest)
        new_measures = Measures(new_info)
        return new_measures


def reversable_diff(arr, assume_sorted=1, reverse=False):
    """
    Does a reversable array difference operation.

    This will be used to find positions where accumulation happened in
    confusion count array.

    Ignore:
        >>> from kwcoco.metrics.confusion_measures import *  # NOQA
        >>> funcs = {
        >>>     'at_zero': [0, 3, 9, 43, 333],
        >>>     'at_nonzero': [2, 3, 9, 43, 333],
        >>>     'at_negative': [-2, 3, 9, 43, 333],
        >>>     'at_neginf': [-np.inf, 3, 9, 43, 333],
        >>>     'at_dblinf': [-np.inf, -np.inf, 3, 9, 43, 333, np.inf, np.inf],
        >>> }
        >>> for k, arr in funcs.items():
        >>>     arr = np.asarray(arr) + 30
        >>>     diff_arr, prefix, suffix = reversable_diff(arr)
        >>>     recon_arr = np.cumsum(diff_arr)
        >>>     recon_arr[0:len(prefix)] += prefix
        >>>     recon_arr[len(recon_arr) - len(suffix):] += suffix
        >>>     print('k = {!r}'.format(k))
        >>>     print('diff_arr  = {!r}'.format(diff_arr))
        >>>     print('arr       = {!r}'.format(arr))
        >>>     print('recon_arr = {!r}'.format(recon_arr))
        >>> for k, arr in funcs.items():
        >>>     arr = np.asarray(arr)[::-1] + 30
        >>>     diff_arr, prefix, suffix = reversable_diff(arr, reverse=True)
        >>>     recon_arr = np.cumsum(diff_arr[::-1])[::-1]
        >>>     recon_arr[0:len(prefix)] += prefix
        >>>     recon_arr[len(recon_arr) - len(suffix):] += suffix
        >>>     print('k = {!r}'.format(k))
        >>>     print('diff_arr  = {!r}'.format(diff_arr))
        >>>     print('arr       = {!r}'.format(arr))
        >>>     print('recon_arr = {!r}'.format(recon_arr))
    """
    import math
    assert assume_sorted
    if len(arr) == 0:
        raise ValueError('todo: default value for empty')

    if reverse:
        arr = arr[::-1]

    n = len(arr)

    last_idx = n - 1

    first_finite_idx = 0
    for first_finite_idx, v in zip(range(n), arr):
        if math.isfinite(v):
            break

    last_finite_idx = n
    for last_finite_idx, v in zip(range(last_idx, -1, -1), arr[::-1]):
        if math.isfinite(v):
            break

    suffix_len = last_idx - last_finite_idx
    prefix_len = first_finite_idx
    offset = arr[first_finite_idx]  # This is + C
    prefix = arr[:first_finite_idx]
    suffix = arr[last_finite_idx + 1:]
    finite_body = arr[first_finite_idx:last_finite_idx + 1]

    diff_arr = np.r_[[0] * prefix_len, [offset], np.diff(finite_body), [0] * suffix_len]

    # The goal is to be able to recon perfectly
    def invert(diff_arr):
        recon_arr = np.cumsum(diff_arr)
        recon_arr[0:len(prefix)] += prefix
        recon_arr[len(recon_arr) - len(suffix):] += suffix
        return recon_arr

    if reverse:
        diff_arr = diff_arr[::-1]
        prefix, suffix = suffix[::-1], prefix[::-1]
    return diff_arr, prefix, suffix


class PerClass_Measures(ub.NiceRepr, DictProxy):
    """
    """
    def __init__(self, cx_to_info):
        self.proxy = cx_to_info

    def __nice__(self):
        return ub.repr2(self.proxy, nl=2, strvals=True, align=':')

    def summary(self):
        return {k: v.summary() for k, v in self.items()}

    @classmethod
    def from_json(cls, state):
        state = ub.map_vals(Measures.from_json, state)
        self = cls(state)
        return self

    def __json__(self):
        return {k: v.__json__() for k, v in self.items()}

    def draw(self, key='mcc', prefix='', **kw):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> from kwcoco.metrics.confusion_vectors import ConfusionVectors  # NOQA
            >>> cfsn_vecs = ConfusionVectors.demo()
            >>> ovr_cfsn = cfsn_vecs.binarize_ovr(keyby='name')
            >>> self = ovr_cfsn.measures()['perclass']
            >>> self.draw('mcc', doclf=True, fnum=1)
            >>> self.draw('pr', doclf=1, fnum=2)
            >>> self.draw('roc', doclf=1, fnum=3)
        """
        from kwcoco.metrics import drawing
        if key == 'pr':
            return drawing.draw_perclass_prcurve(self, prefix=prefix, **kw)
        elif key == 'roc':
            return drawing.draw_perclass_roc(self, prefix=prefix, **kw)
        else:
            return drawing.draw_perclass_thresholds(
                self, key=key, prefix=prefix, **kw)

    def draw_roc(self, prefix='', **kw):
        from kwcoco.metrics import drawing
        return drawing.draw_perclass_roc(self, prefix=prefix, **kw)

    def draw_pr(self, prefix='', **kw):
        from kwcoco.metrics import drawing
        return drawing.draw_perclass_prcurve(self, prefix=prefix, **kw)

    def summary_plot(self, fnum=1, title='', subplots='auto'):
        """

        CommandLine:
            python ~/code/kwcoco/kwcoco/metrics/confusion_measures.py PerClass_Measures.summary_plot --show

        Example:
            >>> from kwcoco.metrics.confusion_measures import *  # NOQA
            >>> from kwcoco.metrics.detect_metrics import DetectionMetrics
            >>> dmet = DetectionMetrics.demo(
            >>>     n_fp=(0, 1), n_fn=(0, 3), nimgs=32, nboxes=(0, 32),
            >>>     classes=3, rng=0, newstyle=1, box_noise=0.7, cls_noise=0.2, score_noise=0.3, with_probs=False)
            >>> cfsn_vecs = dmet.confusion_vectors()
            >>> ovr_cfsn = cfsn_vecs.binarize_ovr(keyby='name', ignore_classes=['vector', 'raster'])
            >>> self = ovr_cfsn.measures()['perclass']
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> import seaborn as sns
            >>> sns.set()
            >>> self.summary_plot(title='demo summary_plot ovr', subplots=['pr', 'roc'])
            >>> kwplot.show_if_requested()
            >>> self.summary_plot(title='demo summary_plot ovr', subplots=['mcc', 'acc'], fnum=2)
        """
        if subplots == 'auto':
            subplots = ['pr', 'roc', 'mcc', 'f1', 'acc']
        import kwplot
        pnum_ = kwplot.PlotNums(nSubplots=len(subplots))
        kwplot.figure(fnum=fnum, doclf=True, figtitle=title)
        for key in subplots:
            self.draw(key, fnum=fnum, pnum=pnum_())


class MeasureCombiner:
    """
    Helper to iteravely combine binary measures generated by some process

    Example:
        >>> from kwcoco.metrics.confusion_measures import *  # NOQA
        >>> from kwcoco.metrics.confusion_vectors import BinaryConfusionVectors
        >>> rng = kwarray.ensure_rng(0)
        >>> bin_combiner = MeasureCombiner(growth='max')
        >>> for _ in range(80):
        >>>     bin_cfsn_vecs = BinaryConfusionVectors.demo(n=rng.randint(40, 50), rng=rng, p_true=0.2, p_error=0.4, p_miss=0.6)
        >>>     bin_measures = bin_cfsn_vecs.measures()
        >>>     bin_combiner.submit(bin_measures)
        >>> combined = bin_combiner.finalize()
        >>> print('combined = {!r}'.format(combined))

    """
    def __init__(self, precision=None, growth=None, thresh_bins=None):
        self.measures = None
        self.growth = growth
        self.thresh_bins = thresh_bins
        self.precision = precision
        self.queue = []

    @property
    def queue_size(self):
        return len(self.queue)

    def submit(self, other):
        self.queue.append(other)

    def combine(self):
        # Reduce measures over the chunk
        if self.measures is None:
            to_combine = self.queue
        else:
            to_combine = [self.measures] + self.queue

        if len(to_combine) == 0:
            pass
        if len(to_combine) == 1:
            self.measures = to_combine[0]
        else:
            self.measures = Measures.combine(
                to_combine, precision=self.precision, growth=self.growth,
                thresh_bins=self.thresh_bins)
        self.queue = []

    def finalize(self):
        if self.queue:
            self.combine()
        if self.measures is None:
            return False
        else:
            self.measures.reconstruct()
            return self.measures


class OneVersusRestMeasureCombiner:
    """
    Helper to iteravely combine ovr measures generated by some process

    Example:
        >>> from kwcoco.metrics.confusion_measures import *  # NOQA
        >>> from kwcoco.metrics.confusion_vectors import OneVsRestConfusionVectors
        >>> rng = kwarray.ensure_rng(0)
        >>> ovr_combiner = OneVersusRestMeasureCombiner(growth='max')
        >>> for _ in range(80):
        >>>     ovr_cfsn_vecs = OneVsRestConfusionVectors.demo()
        >>>     ovr_measures = ovr_cfsn_vecs.measures()
        >>>     ovr_combiner.submit(ovr_measures)
        >>> combined = ovr_combiner.finalize()
        >>> print('combined = {!r}'.format(combined))
    """
    def __init__(self, precision=None, growth=None, thresh_bins=None):
        self.catname_to_combiner = {}
        self.precision = precision
        self.growth = precision
        self.thresh_bins = thresh_bins
        self.queue_size = 0

    def submit(self, other):
        self.queue_size += 1
        for catname, other_m in other['perclass'].items():
            if catname not in self.catname_to_combiner:
                combiner = MeasureCombiner(
                    precision=self.precision, growth=self.growth,
                    thresh_bins=self.thresh_bins)
                self.catname_to_combiner[catname] = combiner
            self.catname_to_combiner[catname].submit(other_m)

    def _summary(self):
        for catname, combiner in self.catname_to_combiner.items():
            # combiner summary
            # combiner.measures
            if combiner.measures is not None:
                combiner.measures.reconstruct()
            for qx, measure in enumerate(combiner.queue):
                measure.reconstruct()
                print('  * queue[{}] = {}'.format(qx, ub.repr2(measure, nl=1)))

    def combine(self):
        for combiner in self.catname_to_combiner.values():
            combiner.combine()
        self.queue_size = 0

    def finalize(self):
        catname_to_measures = {}
        for catname, combiner in self.catname_to_combiner.items():
            catname_to_measures[catname] = combiner.finalize()
        perclass = PerClass_Measures(catname_to_measures)
        # TODO: consolidate in kwcoco
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            mAUC = np.nanmean([item['trunc_auc'] for item in perclass.values()])
            mAP = np.nanmean([item['ap'] for item in perclass.values()])
        compatible_format = {
            'perclass': perclass,
            'mAUC': mAUC,
            'mAP': mAP,
        }
        return compatible_format


def populate_info(info):
    """
    Given raw accumulated confusion counts, populated secondary measures like
    AP, AUC, F1, MCC, etc..
    """
    info['tp_count'] = tp = np.array(info['tp_count'])
    info['fp_count'] = fp = np.array(info['fp_count'])
    info['tn_count'] = tn = np.array(info['tn_count'])
    info['fn_count'] = fn = np.array(info['fn_count'])
    info['thresholds'] = thresh = np.array(info['thresholds'])

    realpos_total = info.get('realpos_total', None)
    if realpos_total is None:
        realpos_total = info['tp_count'][-1] + info['fn_count'][-1]
        info['realpos_total'] = realpos_total

    realneg_total = info.get('realneg_total', None)
    if realneg_total is None:
        realneg_total = info['tn_count'][-1] + info['fp_count'][-1]
        info['realneg_total'] = realneg_total

    nsupport = info.get('nsupport', None)
    if nsupport is None:
        info['nsupport'] = nsupport = realneg_total + realpos_total

    monotonic_ppv = info.get('monotonic_ppv', True)
    info['monotonic_ppv'] = monotonic_ppv

    finite_flags = np.isfinite(thresh)

    trunc_fp = fp
    # Cutoff the curves at a comparable point
    fp_cutoff = info.get('fp_cutoff', None)
    if fp_cutoff is None:
        fp_cutoff = np.inf
    elif isinstance(fp_cutoff, str):
        if fp_cutoff == 'num_true':
            fp_cutoff = int(np.ceil(realpos_total))
        else:
            raise KeyError(fp_cutoff)

    if np.isfinite(fp_cutoff):
        idxs = np.where(trunc_fp > fp_cutoff)[0]
        if len(idxs) == 0:
            trunc_idx = len(trunc_fp)
        else:
            trunc_idx = idxs[0]
    else:
        trunc_idx = None

    if trunc_idx is None and np.any(np.isinf(fp)):
        idxs = np.where(np.isinf(fp))[0]
        if len(idxs):
            trunc_idx = idxs[0]

    if trunc_idx is None:
        trunc_fp = fp
        trunc_tp = tp
        trunc_thresholds = thresh
    else:
        trunc_fp = fp[:trunc_idx]
        trunc_tp = tp[:trunc_idx]
        trunc_thresholds = thresh[:trunc_idx]

    # if the cuttoff was not reached, horizontally extend the curve
    # This will hurt the scores (aka we may be bias against small
    # scenes), but this will ensure that big scenes are comparable
    from kwcoco.metrics import assignment
    if len(trunc_fp) == 0:
        trunc_fp = np.array([fp_cutoff])
        trunc_tp = np.array([0])

        if assignment.USE_NEG_INF:
            trunc_thresholds = np.array([-np.inf])
        else:
            trunc_thresholds = np.array([0])
        # THIS WILL CAUSE AUC TO RAISE AN ERROR IF IT GETS HIT
    elif trunc_fp[-1] <= fp_cutoff and np.isfinite(fp_cutoff):
        trunc_fp = np.hstack([trunc_fp, [fp_cutoff]])
        trunc_tp = np.hstack([trunc_tp, [trunc_tp[-1]]])
        if assignment.USE_NEG_INF:
            trunc_thresholds = np.hstack([trunc_thresholds, [-np.inf]])
        else:
            trunc_thresholds = np.hstack([trunc_thresholds, [0]])
    info['trunc_idx'] = trunc_idx
    info['trunc_fp_count'] = trunc_fp
    info['trunc_tp_count'] = trunc_tp
    info['trunc_thresholds'] = trunc_thresholds

    with warnings.catch_warnings():
        # It is very possible that we will divide by zero in this func
        warnings.filterwarnings('ignore', message='invalid .* true_divide')
        warnings.filterwarnings('ignore', message='invalid value')

        pred_pos = (tp + fp)  # number of predicted positives
        ppv = tp / pred_pos  # precision
        ppv[np.isnan(ppv)] = 0

        if monotonic_ppv:
            # trick to make precision monotonic (which is probably not correct)
            if 0:
                print('ppv = {!r}'.format(ppv))
            ppv = np.maximum.accumulate(ppv[::-1])[::-1]
            if 0:
                print('ppv = {!r}'.format(ppv))

        # can set tpr_denom denominator to one
        tpr_denom = (tp + fn)  #
        tpr_denom[~(tpr_denom > 0)] = 1
        tpr = tp / tpr_denom  # recall

        debug = 0
        if debug:
            assert ub.allsame(tpr_denom), 'tpr denom should be constant'
            # tpr_denom should be equal to info['realpos_total']
            if np.any(tpr_denom != info['realpos_total']):
                warnings.warn('realpos_total is inconsistent')

        tnr_denom = (tn + fp)
        tnr_denom[tnr_denom == 0] = 1
        tnr = tn / tnr_denom

        pnv_denom = (tn + fn)
        pnv_denom[pnv_denom == 0] = 1
        npv = tn / pnv_denom

        info['ppv'] = ppv
        info['tpr'] = tpr

        # fpr_denom is a proxy for fp + tn as tn is generally unknown in
        # the case where all negatives are specified in the confusion
        # vectors fpr_denom will be exactly (fp + tn)
        # fpr = fp / (fp + tn)
        finite_fp = fp[np.isfinite(fp)]
        fpr_denom = finite_fp[-1] if len(finite_fp) else 0
        if fpr_denom == 0:
            fpr_denom = 1
        fpr = info['fp_count'] / fpr_denom
        info['fpr'] = fpr

        info['bm'] = tpr + tnr - 1  # informedness
        info['mk'] = ppv + npv - 1  # markedness

        info['acc'] = (tp + tn) / (tp + tn + fp + fn)

        # https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
        # mcc_numer = (tp * tn) - (fp * fn)
        # mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        # mcc_denom[np.isnan(mcc_denom) | (mcc_denom == 0)] = 1
        # info['mcc'] = mcc_numer / mcc_denom

        real_pos = fn + tp  # number of real positives
        p_denom = real_pos.copy()
        p_denom[p_denom == 0] = 1
        fnr = fn / p_denom  # miss-rate
        fdr  = 1 - ppv  # false discovery rate
        fmr  = 1 - npv  # false ommision rate (for)

        info['tnr'] = tnr
        info['npv'] = npv
        info['mcc'] = np.sqrt(ppv * tpr * tnr * npv) - np.sqrt(fdr * fnr * fpr * fmr)

        f1_numer = (2 * ppv * tpr)
        f1_denom = (ppv + tpr)
        f1_denom[f1_denom == 0] = 1
        info['f1'] =  f1_numer / f1_denom

        # https://erotemic.wordpress.com/2019/10/23/closed-form-of-the-mcc-when-tn-inf/
        info['g1'] = np.sqrt(ppv * tpr)

        keys = ['mcc', 'g1', 'f1', 'acc']
        finite_thresh = thresh[finite_flags]
        for key in keys:
            measure = info[key][finite_flags]
            try:
                max_idx = np.nanargmax(measure)
            except ValueError:
                best_thresh = np.nan
                best_measure = np.nan
            else:
                best_thresh = float(finite_thresh[max_idx])
                best_measure = float(measure[max_idx])

            best_label = '{}={:0.2f}@{:0.2f}'.format(key, best_measure, best_thresh)

            # if np.isinf(best_thresh) or np.isnan(best_measure):
            #     print('key = {!r}'.format(key))
            #     print('finite_flags = {!r}'.format(finite_flags))
            #     print('measure = {!r}'.format(measure))
            #     print('best_label = {!r}'.format(best_label))
            #     import xdev
            #     xdev.embed()
            info['max_{}'.format(key)] = best_label
            info['_max_{}'.format(key)] = (best_measure, best_thresh)

        import sklearn.metrics  # NOQA
        finite_trunc_fp = info['trunc_fp_count']
        finite_trunc_fp = finite_trunc_fp[np.isfinite(finite_trunc_fp)]
        trunc_fpr_denom = finite_trunc_fp[-1] if len(finite_trunc_fp) else 0
        if trunc_fpr_denom == 0:
            trunc_fpr_denom = 1
        info['trunc_tpr'] = info['trunc_tp_count'] / info['realpos_total']
        info['trunc_fpr'] = info['trunc_fp_count'] / trunc_fpr_denom
        try:
            info['trunc_auc'] = sklearn.metrics.auc(info['trunc_fpr'], info['trunc_tpr'])
        except ValueError:
            # At least 2 points are needed to compute area under curve, but x.shape = 1
            info['trunc_auc'] = np.nan
            if len(info['trunc_fpr']) == 1 and len(info['trunc_tpr']) == 1:
                if info['trunc_fpr'][0] == 0 and info['trunc_tpr'][0] == 1:
                    # Hard code AUC in the perfect detection case
                    info['trunc_auc'] = 1.0

        info['auc'] = info['trunc_auc']
        """
        Note:
            Apparently, consistent scoring is really hard to get right.

            For detection problems scoring via
            confusion_vectors+sklearn produces noticably different
            results than the VOC method. There are a few reasons for
            this.  The VOC method stops counting true positives after
            all assigned predicted boxes have been counted. It simply
            remembers the amount of original true positives to
            normalize the true positive reate. On the other hand,
            confusion vectors maintains a list of these unassigned true
            boxes and gives them a predicted index of -1 and a score of
            zero. This means that this function sees them as having a
            y_true of 1 and a y_score of 0, which allows the
            scikit-learn fp and tp counts to effectively get up to
            100% recall when the threshold is zero. The VOC method
            simply ignores these and handles them implicitly. The
            problem is that if you remove these from the scikit-learn
            inputs, it wont see the correct number of positives and it
            will incorrectly normalize the recall.  In summary:

                VOC:
                    * remembers realpos_total
                    * doesn't count unassigned truths as TP when the
                    threshold is zero.

                CV+SKL:
                    * counts unassigned truths as TP with score=0.
                    * NEW: now counts unassigned truth as TP with score=-np.inf
                    * Always ensure tpr=1, ppv=0 and ppv=1, tpr=0 cases
                    exist.
        """

        # ---
        # sklearn definition
        # AP = sum((R[n] - R[n - 1]) * P[n] for n in range(len(thresholds)))
        # stop when full recall attained
        SKLISH_AP = 1
        if SKLISH_AP:
            last_ind = tpr.searchsorted(tpr[-1])
            rec  = np.r_[0, tpr[:last_ind + 1]]
            prec = np.r_[1, ppv[:last_ind + 1]]
            # scores = np.r_[0, thresh[:last_ind + 1]]

            # Precisions are weighted by the change in recall
            diff_items = np.diff(rec)
            prec_items = prec[1:]

            # basline way
            info['sklish_ap'] = float(np.sum(diff_items * prec_items))

        PYCOCOTOOLS_AP = True
        if PYCOCOTOOLS_AP:
            # similar to pycocotools style "AP"
            R = 101

            feasible_idxs = np.where(thresh > -np.inf)[0]
            if len(feasible_idxs) == 0:
                info['pycocotools_ap'] = np.nan
            else:
                feasible_tpr = tpr[feasible_idxs[-1]]
                last_ind = tpr.searchsorted(feasible_tpr)
                rc  = tpr[:last_ind + 1]
                pr = ppv[:last_ind + 1]

                recThrs = np.linspace(0, 1.0, R)
                inds = np.searchsorted(rc, recThrs, side='left')
                q  = np.zeros((R,))
                # ss = np.zeros((R,))

                try:
                    for ri, pi in enumerate(inds):
                        q[ri] = pr[pi]
                        # ss[ri] = scores[pi]
                except Exception:
                    pass

                pycocotools_ap = q.mean()
                info['pycocotools_ap'] = float(pycocotools_ap)

        OUTLIER_AP = 1
        if OUTLIER_AP:
            # Remove extreme outliers from ap calculation
            # only do this on the first or last 2 items.
            # Hueristically chosen.
            flags = diff_items > 0.1
            idxs = np.where(flags)[0]
            max_idx = len(flags) - 1
            idx_thresh = 2
            try:
                idx_dist = np.minimum(idxs, max_idx - idxs)
                outlier_idxs = idxs[idx_dist < idx_thresh]
                outlier_flags = kwarray.boolmask(outlier_idxs, len(diff_items))
                inlier_flags = ~outlier_flags

                score = prec_items.copy()
                score[outlier_flags] = score[inlier_flags].min()
                score[outlier_flags] = 0

                ap = outlier_ap = np.sum(score * diff_items)

                info['outlier_ap'] = float(outlier_ap)
            except Exception:
                pass

        INF_THRESH_AP = 1
        if INF_THRESH_AP:
            # This may become the de-facto scikit-learn implementation in the
            # future.
            from scipy import integrate
            # MODIFIED SKLEARN AVERAGE PRECISION FOR -INF THRESH
            #
            # Better way of marked unassigned truth as never-recallable.
            # We need to prevent these unassigned truths from incorrectly
            # bringing up our true positive rate at low thresholds.
            #
            # Simply bump last_ind to ensure it is
            feasible_idxs = np.where(thresh > -np.inf)[0]
            if len(feasible_idxs) == 0:
                info['sklearn_ap'] = np.nan
            else:
                feasible_tpr = tpr[feasible_idxs[-1]]
                last_ind = tpr.searchsorted(feasible_tpr)
                rec  = np.r_[0, tpr[:last_ind + 1]]
                prec = np.r_[1, ppv[:last_ind + 1]]

                diff_items = np.diff(rec)
                prec_items = prec[1:]

                # Not sure which is beset here, we no longer have
                # assumption of max-tpr = 1
                # ap = float(np.sum(diff_items * prec_items))
                info['sklearn_ap'] = integrate.trapz(y=prec, x=rec)

        info['ap_method'] = info.get('ap_method', 'pycocotools')
        if info['ap_method'] == 'pycocotools':
            ap = info['pycocotools_ap']
        elif info['ap_method'] == 'outlier':
            ap = info['outlier_ap']
        elif info['ap_method'] == 'sklearn':
            ap = info['sklearn_ap']
        elif info['ap_method'] == 'sklish':
            ap = info['sklish_ap']
        else:
            raise KeyError(info['ap_method'])

        # print('ap = {!r}'.format(ap))
        # print('ap = {!r}'.format(ap))
        # ap = np.sum(np.diff(rec) * prec[1:])
        info['ap'] = float(ap)
