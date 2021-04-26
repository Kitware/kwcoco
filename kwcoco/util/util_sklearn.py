"""
Extensions to sklearn constructs
"""
import warnings
import numpy as np
from sklearn.utils.validation import check_array
from sklearn.model_selection._split import (_BaseKFold,)


class StratifiedGroupKFold(_BaseKFold):
    """Stratified K-Folds cross-validator with Grouping

    Provides train/test indices to split data in train/test sets.

    This cross-validation object is a variation of GroupKFold that returns
    stratified folds. The folds are made by preserving the percentage of
    samples for each class.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.
    """

    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        if not shuffle:
            random_state = None
        super(StratifiedGroupKFold, self).__init__(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _make_test_folds(self, X, y=None, groups=None):
        """
        Args:
            X (ndarray): data
            y (ndarray): labels
            groups (ndarray): groupids for items. Items with the same groupid
                must be placed in the same group.

        Returns:
            list: test_folds

        Example:
            >>> import kwarray
            >>> rng = kwarray.ensure_rng(0)
            >>> groups = [1, 1, 3, 4, 2, 2, 7, 8, 8]
            >>> y      = [1, 1, 1, 1, 2, 2, 2, 3, 3]
            >>> X = np.empty((len(y), 0))
            >>> self = StratifiedGroupKFold(random_state=rng, shuffle=True)
            >>> skf_list = list(self.split(X=X, y=y, groups=groups))
            ...
            >>> import ubelt as ub
            >>> print(ub.repr2(skf_list, nl=1, with_dtype=False))
            [
                (np.array([2, 3, 4, 5, 6]), np.array([0, 1, 7, 8])),
                (np.array([0, 1, 2, 7, 8]), np.array([3, 4, 5, 6])),
                (np.array([0, 1, 3, 4, 5, 6, 7, 8]), np.array([2])),
            ]
        """
        import kwarray
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value')
            n_splits = self.n_splits
            y = np.asarray(y)
            n_samples = y.shape[0]

            unique_y, y_inversed = np.unique(y, return_inverse=True)
            n_classes = max(unique_y) + 1
            unique_groups, group_idxs = kwarray.group_indices(groups)
            grouped_y = kwarray.apply_grouping(y, group_idxs)
            grouped_y_counts = np.array([
                np.bincount(y_, minlength=n_classes) for y_ in grouped_y])

            target_freq = grouped_y_counts.sum(axis=0)
            target_freq = target_freq.astype(np.float)
            target_ratio = target_freq / float(target_freq.sum())

            # Greedilly choose the split assignment that minimizes the local
            # * squared differences in target from actual frequencies
            # * and best equalizes the number of items per fold
            # Distribute groups with most members first
            split_freq = np.zeros((n_splits, n_classes))
            # split_ratios = split_freq / split_freq.sum(axis=1)
            split_ratios = np.ones(split_freq.shape) / split_freq.shape[1]
            split_diffs = ((split_freq - target_ratio) ** 2).sum(axis=1)
            sortx = np.argsort(grouped_y_counts.sum(axis=1))[::-1]
            grouped_splitx = []

            # import ubelt as ub
            # print(ub.repr2(grouped_y_counts, nl=-1))
            # print('target_ratio = {!r}'.format(target_ratio))

            for count, group_idx in enumerate(sortx):
                # print('---------\n')
                group_freq = grouped_y_counts[group_idx]
                cand_freq = (split_freq + group_freq)
                cand_freq = cand_freq.astype(np.float)
                cand_ratio = cand_freq / cand_freq.sum(axis=1)[:, None]
                cand_diffs = ((cand_ratio - target_ratio) ** 2).sum(axis=1)
                # Compute loss
                losses = []
                # others = np.nan_to_num(split_diffs)
                other_diffs = np.array([
                    sum(split_diffs[x + 1:]) + sum(split_diffs[:x])
                    for x in range(n_splits)
                ])
                # penalize unbalanced splits
                ratio_loss = other_diffs + cand_diffs
                # penalize heavy splits
                freq_loss = split_freq.sum(axis=1)
                freq_loss = freq_loss.astype(np.float)
                freq_loss = freq_loss / freq_loss.sum()
                losses = ratio_loss + freq_loss
                #-------
                splitx = np.argmin(losses)
                # print('losses = %r, splitx=%r' % (losses, splitx))
                split_freq[splitx] = cand_freq[splitx]
                split_ratios[splitx] = cand_ratio[splitx]
                split_diffs[splitx] = cand_diffs[splitx]
                grouped_splitx.append(splitx)

            test_folds = np.empty(n_samples, dtype=int)
            for group_idx, splitx in zip(sortx, grouped_splitx):
                idxs = group_idxs[group_idx]
                test_folds[idxs] = splitx

        return test_folds

    def _iter_test_masks(self, X, y=None, groups=None):
        test_folds = self._make_test_folds(X, y, groups)
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.
        """
        y = check_array(y, ensure_2d=False, dtype=None)
        return super(StratifiedGroupKFold, self).split(X, y, groups)

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/kwcoco/util/util_sklearn.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
