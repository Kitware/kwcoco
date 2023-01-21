from numpy import ndarray
from typing import Union


def fast_confusion_matrix(
        y_true: ndarray,
        y_pred: ndarray,
        n_labels: int,
        sample_weight: Union[ndarray, None] = None) -> ndarray:
    ...
