import kwcoco
import kwimage
from typing import Union
from typing import List
import ubelt as ub
from typing import Dict
import ubelt as ub
from _typeshed import Incomplete


class DetectionMetrics(ub.NiceRepr):

    def __init__(dmet, classes: Incomplete | None = ...) -> None:
        ...

    def clear(dmet) -> None:
        ...

    def __nice__(dmet):
        ...

    @classmethod
    def from_coco(DetectionMetrics,
                  true_coco: kwcoco.CocoDataset,
                  pred_coco: kwcoco.CocoDataset,
                  gids: Incomplete | None = ...,
                  verbose: int = ...):
        ...

    def add_predictions(dmet,
                        pred_dets: kwimage.Detections,
                        imgname: str = None,
                        gid: Union[int, None] = None) -> None:
        ...

    def add_truth(dmet,
                  true_dets: kwimage.Detections,
                  imgname: str = None,
                  gid: Union[int, None] = None) -> None:
        ...

    def true_detections(dmet, gid):
        ...

    def pred_detections(dmet, gid):
        ...

    def confusion_vectors(
        dmet,
        iou_thresh: Union[float, List[float]] = 0.5,
        bias: float = 0,
        gids: List[int] = None,
        compat: str = 'mutex',
        prioritize: str = 'iou',
        ignore_classes: Union[set, str] = 'ignore',
        background_class: str = ...,
        verbose: Union[int, str] = 'auto',
        workers: int = 0,
        track_probs: str = 'try',
        max_dets: Incomplete | None = ...
    ) -> kwcoco.metrics.confusion_vectors.ConfusionVectors | Dict[
            float, kwcoco.metrics.confusion_vectors.ConfusionVectors]:
        ...

    def score_kwant(dmet, iou_thresh: float = ...):
        ...

    def score_kwcoco(dmet,
                     iou_thresh: float = ...,
                     bias: int = ...,
                     gids: Incomplete | None = ...,
                     compat: str = ...,
                     prioritize: str = ...):
        ...

    def score_voc(dmet,
                  iou_thresh: float = ...,
                  bias: int = ...,
                  method: str = ...,
                  gids: Incomplete | None = ...,
                  ignore_classes: str = ...):
        ...

    def score_pycocotools(dmet,
                          with_evaler: bool = ...,
                          with_confusion: bool = ...,
                          verbose: int = ...,
                          iou_thresholds: Incomplete | None = ...) -> Dict:
        ...

    score_coco: Incomplete

    @classmethod
    def demo(cls, **kwargs):
        ...

    def summarize(dmet,
                  out_dpath: Incomplete | None = ...,
                  plot: bool = ...,
                  title: str = ...,
                  with_bin: str = ...,
                  with_ovr: str = ...):
        ...


def pycocotools_confusion_vectors(dmet,
                                  evaler,
                                  iou_thresh: float = ...,
                                  verbose: int = ...):
    ...


def eval_detections_cli(**kw) -> None:
    ...


def pct_summarize2(self):
    ...
