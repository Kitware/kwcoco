from numpy import ndarray


def fast_confusion_matrix(y_true: ndarray,
                          y_pred: ndarray,
                          n_labels: int,
                          sample_weight: ndarray | None = None) -> ndarray:
    ...
