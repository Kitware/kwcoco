import ubelt as ub
from _typeshed import Incomplete


class VOC_Metrics(ub.NiceRepr):
    recs: Incomplete
    cx_to_lines: Incomplete
    classes: Incomplete

    def __init__(self, classes: Incomplete | None = ...) -> None:
        ...

    def __nice__(self):
        ...

    def add_truth(self, true_dets, gid) -> None:
        ...

    def add_predictions(self, pred_dets, gid) -> None:
        ...

    def score(self,
              iou_thresh: float = ...,
              bias: int = ...,
              method: str = ...):
        ...
