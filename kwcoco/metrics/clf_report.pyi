from numpy import ndarray
from typing import Any
from nptyping import Int
from typing import Dict
from typing import List
from _typeshed import Incomplete
from typing import Any

ASCII_ONLY: Incomplete


def classification_report(y_true,
                          y_pred,
                          target_names: Incomplete | None = ...,
                          sample_weight: Incomplete | None = ...,
                          verbose: bool = ...,
                          remove_unsupported: bool = ...,
                          log: Incomplete | None = ...,
                          ascii_only: bool = ...):
    ...


def ovr_classification_report(mc_y_true: ndarray[Any, Int],
                              mc_probs: ndarray,
                              target_names: Dict[int, str] = None,
                              sample_weight: ndarray = None,
                              metrics: List[str] = None,
                              verbose: int = ...,
                              remove_unsupported: bool = ...,
                              log: Incomplete | None = ...):
    ...
