from numpy import ndarray
from _typeshed import Incomplete


def confusion_matrix(y_true,
                     y_pred,
                     n_labels: Incomplete | None = ...,
                     labels: Incomplete | None = ...,
                     sample_weight: Incomplete | None = ...) -> ndarray:
    ...


def global_accuracy_from_confusion(cfsn):
    ...


def class_accuracy_from_confusion(cfsn):
    ...
