from _typeshed import Incomplete
from sklearn.model_selection._split import _BaseKFold

ARGSORT_HAS_STABLE_KIND: Incomplete


class StratifiedGroupKFold(_BaseKFold):

    def __init__(self,
                 n_splits: int = ...,
                 shuffle: bool = ...,
                 random_state: Incomplete | None = ...) -> None:
        ...

    def split(self, X, y, groups: Incomplete | None = ...):
        ...
