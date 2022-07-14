import pandas as pd
from typing import Union
from typing import List
from typing import Set
from numpy.random import RandomState
import ubelt as ub
from _typeshed import Incomplete


class ConfusionVectors(ub.NiceRepr):

    def __init__(cfsn_vecs,
                 data,
                 classes,
                 probs: Incomplete | None = ...) -> None:
        ...

    def __nice__(cfsn_vecs):
        ...

    def __json__(self):
        ...

    @classmethod
    def from_json(cls, state):
        ...

    @classmethod
    def demo(cfsn_vecs, **kw) -> ConfusionVectors:
        ...

    @classmethod
    def from_arrays(ConfusionVectors,
                    true,
                    pred: Incomplete | None = ...,
                    score: Incomplete | None = ...,
                    weight: Incomplete | None = ...,
                    probs: Incomplete | None = ...,
                    classes: Incomplete | None = ...):
        ...

    def confusion_matrix(cfsn_vecs, compress: bool = False) -> pd.DataFrame:
        ...

    def coarsen(cfsn_vecs, cxs) -> ConfusionVectors:
        ...

    def binarize_classless(
        cfsn_vecs,
        negative_classes: List[Union[str,
                                     int]] = None) -> BinaryConfusionVectors:
        ...

    def binarize_ovr(cfsn_vecs,
                     mode: int = 1,
                     keyby: Union[int, str] = 'name',
                     ignore_classes: Set[str] = ...,
                     approx: bool = False) -> OneVsRestConfusionVectors:
        ...

    def classification_report(cfsn_vecs, verbose: int = ...):
        ...


class OneVsRestConfusionVectors(ub.NiceRepr):
    cx_to_binvecs: Incomplete
    classes: Incomplete

    def __init__(self, cx_to_binvecs, classes) -> None:
        ...

    def __nice__(self):
        ...

    @classmethod
    def demo(cls) -> ConfusionVectors:
        ...

    def keys(self):
        ...

    def __getitem__(self, cx):
        ...

    def measures(self,
                 stabalize_thresh: int = 7,
                 fp_cutoff: int = None,
                 monotonic_ppv: bool = True,
                 ap_method: str = ...):
        ...

    def ovr_classification_report(self) -> None:
        ...


class BinaryConfusionVectors(ub.NiceRepr):
    data: Incomplete
    cx: Incomplete
    classes: Incomplete

    def __init__(self,
                 data,
                 cx: Incomplete | None = ...,
                 classes: Incomplete | None = ...) -> None:
        ...

    @classmethod
    def demo(cls,
             n: int = 10,
             p_true: float = 0.5,
             p_error: float = 0.2,
             p_miss: float = 0.0,
             rng: Union[int, RandomState] = None) -> BinaryConfusionVectors:
        ...

    @property
    def catname(self):
        ...

    def __nice__(self):
        ...

    def __len__(self):
        ...

    def measures(self,
                 stabalize_thresh: int = 7,
                 fp_cutoff: int = None,
                 monotonic_ppv: bool = True,
                 ap_method: str = ...):
        ...

    def draw_distribution(self):
        ...
