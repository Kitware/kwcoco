from numpy import ndarray
from typing import Union
from typing import List
from typing import Callable
from typing import Dict
from _typeshed import Incomplete

ASCII_ONLY: Incomplete


def classification_report(y_true: ndarray,
                          y_pred: ndarray,
                          target_names: Union[List, None] = None,
                          sample_weight: Union[ndarray, None] = None,
                          verbose: int = False,
                          remove_unsupported: bool = False,
                          log: Union[Callable, None] = None,
                          ascii_only: bool = False):
    ...


def ovr_classification_report(mc_y_true: ndarray,
                              mc_probs: ndarray,
                              target_names: Union[Dict[int, str], None] = None,
                              sample_weight: Union[ndarray, None] = None,
                              metrics: Union[List[str], None] = None,
                              verbose: int = ...,
                              remove_unsupported: bool = ...,
                              log: Incomplete | None = ...):
    ...
