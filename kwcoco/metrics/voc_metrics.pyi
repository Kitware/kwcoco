from typing import Dict
from typing import List
import kwcoco
import ubelt as ub
from _typeshed import Incomplete


class VOC_Metrics(ub.NiceRepr):
    recs: Dict[int, List[dict]]
    cx_to_lines: Dict[int, List]
    classes: None | List[str] | kwcoco.CategoryTree

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
