from typing import Dict
import kwcoco
import kwcoco.metrics.confusion_measures
from typing import List
from _typeshed import Incomplete


def draw_perclass_roc(
        cx_to_info: kwcoco.metrics.confusion_measures.PerClass_Measures | Dict,
        classes: Incomplete | None = ...,
        prefix: str = ...,
        fnum: int = ...,
        fp_axis: str = 'count',
        **kw):
    ...


def demo_format_options() -> None:
    ...


def concice_si_display(val,
                       eps: float = 1e-08,
                       precision: int = 2,
                       si_thresh: int = 4):
    ...


def draw_perclass_prcurve(
        cx_to_info: kwcoco.metrics.confusion_measures.PerClass_Measures | Dict,
        classes: Incomplete | None = ...,
        prefix: str = ...,
        fnum: int = ...,
        **kw):
    ...


def draw_perclass_thresholds(
        cx_to_info: kwcoco.metrics.confusion_measures.PerClass_Measures | Dict,
        key: str = ...,
        classes: Incomplete | None = ...,
        prefix: str = ...,
        fnum: int = ...,
        **kw):
    ...


def draw_roc(info, prefix: str = ..., fnum: int = ..., **kw):
    ...


def draw_prcurve(info, prefix: str = ..., fnum: int = ..., **kw):
    ...


def draw_threshold_curves(info,
                          keys: None | List[str] = None,
                          prefix: str = ...,
                          fnum: int = ...,
                          **kw):
    ...


def determenistic_colors(keys, preset_colors):
    ...
