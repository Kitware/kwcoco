from numpy import ndarray
from typing import Any
from nptyping import Int
from typing import Any


def fast_confusion_matrix(y_true: ndarray[Any, Int],
                          y_pred: ndarray[Any, Int],
                          n_labels: int,
                          sample_weight: ndarray = None) -> ndarray:
    ...
