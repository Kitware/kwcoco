from numpy import ndarray
from typing import List
from typing import Callable
from typing import Dict
from _typeshed import Incomplete

ASCII_ONLY: Incomplete


def classification_report(y_true: ndarray,
                          y_pred: ndarray,
                          target_names: List | None = None,
                          sample_weight: ndarray | None = None,
                          verbose: int = False,
                          remove_unsupported: bool = False,
                          log: Callable | None = None,
                          ascii_only: bool = False):
    ...


def ovr_classification_report(mc_y_true: ndarray,
                              mc_probs: ndarray,
                              target_names: Dict[int, str] | None = None,
                              sample_weight: ndarray | None = None,
                              metrics: List[str] | None = None,
                              verbose: int = ...,
                              remove_unsupported: bool = ...,
                              log: Incomplete | None = ...):
    ...
