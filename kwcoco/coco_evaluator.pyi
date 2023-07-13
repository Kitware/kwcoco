import scriptconfig as scfg
import ubelt as ub
from _typeshed import Incomplete
from collections.abc import Generator
from kwcoco.util.dict_like import DictProxy
from typing import Any


class CocoEvalConfig(scfg.DataConfig):
    __default__: Incomplete

    def __post_init__(self) -> None:
        ...


class CocoEvaluator:

    def __init__(coco_eval, config) -> None:
        ...

    def log(coco_eval, msg, level: str = ...) -> None:
        ...

    def evaluate(coco_eval) -> CocoResults:
        ...


def dmet_area_weights(dmet,
                      orig_weights,
                      cfsn_vecs,
                      area_ranges,
                      coco_eval,
                      use_area_attr: bool = ...) -> Generator[Any, None, None]:
    ...


class CocoResults(ub.NiceRepr, DictProxy):

    def __init__(results, resdata: Incomplete | None = ...) -> None:
        ...

    def dump_figures(results,
                     out_dpath,
                     expt_title: Incomplete | None = ...,
                     figsize: str = ...,
                     tight: bool = ...) -> None:
        ...

    def __json__(results):
        ...

    @classmethod
    def from_json(cls, state):
        ...

    def dump(result, file, indent: str = ...):
        ...


class CocoSingleResult(ub.NiceRepr):

    def __init__(result,
                 nocls_measures,
                 ovr_measures,
                 cfsn_vecs,
                 meta: Incomplete | None = ...) -> None:
        ...

    def __nice__(result):
        ...

    @classmethod
    def from_json(cls, state):
        ...

    def __json__(result):
        ...

    def dump(result, file, indent: str = ...):
        ...

    def dump_figures(result,
                     out_dpath,
                     expt_title: Incomplete | None = ...,
                     figsize: str = ...,
                     tight: bool = ...,
                     verbose: int = ...) -> None:
        ...
