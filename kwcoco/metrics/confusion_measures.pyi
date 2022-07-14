from typing import List
from typing import Union
import kwcoco
import kwcoco.metrics.confusion_measures
import ubelt as ub
from _typeshed import Incomplete
from kwcoco.metrics.util import DictProxy


class Measures(ub.NiceRepr, DictProxy):
    proxy: Incomplete

    def __init__(self, info) -> None:
        ...

    @property
    def catname(self):
        ...

    def __nice__(self):
        ...

    def reconstruct(self):
        ...

    @classmethod
    def from_json(cls, state):
        ...

    def __json__(self):
        ...

    def summary(self):
        ...

    def maximized_thresholds(self):
        ...

    def counts(self):
        ...

    def draw(self, key: Incomplete | None = ..., prefix: str = ..., **kw):
        ...

    def summary_plot(self,
                     fnum: int = ...,
                     title: str = ...,
                     subplots: str = ...) -> None:
        ...

    @classmethod
    def demo(cls, **kwargs):
        ...

    @classmethod
    def combine(
            cls,
            tocombine: List[Measures],
            precision: Union[int, None] = None,
            growth: Union[int, None] = None,
            thresh_bins: int = None
    ) -> kwcoco.metrics.confusion_measures.Measures:
        ...


def reversable_diff(arr, assume_sorted: int = ..., reverse: bool = ...):
    ...


class PerClass_Measures(ub.NiceRepr, DictProxy):
    proxy: Incomplete

    def __init__(self, cx_to_info) -> None:
        ...

    def __nice__(self):
        ...

    def summary(self):
        ...

    @classmethod
    def from_json(cls, state):
        ...

    def __json__(self):
        ...

    def draw(self, key: str = ..., prefix: str = ..., **kw):
        ...

    def draw_roc(self, prefix: str = ..., **kw):
        ...

    def draw_pr(self, prefix: str = ..., **kw):
        ...

    def summary_plot(self,
                     fnum: int = ...,
                     title: str = ...,
                     subplots: str = ...) -> None:
        ...


class MeasureCombiner:
    measures: Incomplete
    growth: Incomplete
    thresh_bins: Incomplete
    precision: Incomplete
    queue: Incomplete

    def __init__(self,
                 precision: Incomplete | None = ...,
                 growth: Incomplete | None = ...,
                 thresh_bins: Incomplete | None = ...) -> None:
        ...

    @property
    def queue_size(self):
        ...

    def submit(self, other) -> None:
        ...

    def combine(self) -> None:
        ...

    def finalize(self):
        ...


class OneVersusRestMeasureCombiner:
    catname_to_combiner: Incomplete
    precision: Incomplete
    growth: Incomplete
    thresh_bins: Incomplete
    queue_size: int

    def __init__(self,
                 precision: Incomplete | None = ...,
                 growth: Incomplete | None = ...,
                 thresh_bins: Incomplete | None = ...) -> None:
        ...

    def submit(self, other) -> None:
        ...

    def combine(self) -> None:
        ...

    def finalize(self):
        ...


def populate_info(info) -> None:
    ...
