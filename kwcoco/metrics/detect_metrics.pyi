from typing import Dict
import kwimage
import kwcoco
from typing import List
from ubelt.util_const import NoParamType
import pathlib
import ubelt as ub
from _typeshed import Incomplete

from kwcoco.metrics.confusion_vectors import ConfusionVectors

__docstubs__: str


class DetectionMetrics(ub.NiceRepr):
    gid_to_true_dets: Dict[int, kwimage.Detections]
    gid_to_pred_dets: Dict[int, kwimage.Detections]

    def __init__(dmet, classes: Incomplete | None = ...) -> None:
        ...

    def clear(dmet) -> None:
        ...

    def __nice__(dmet):
        ...

    def enrich_confusion_vectors(dmet, cfsn_vecs) -> None:
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
                        imgname: str | None = None,
                        gid: int | None = None) -> None:
        ...

    def add_truth(dmet,
                  true_dets: kwimage.Detections,
                  imgname: str | None = None,
                  gid: int | None = None) -> None:
        ...

    def true_detections(dmet, gid):
        ...

    def pred_detections(dmet, gid):
        ...

    @property
    def classes(dmet):
        ...

    @classes.setter
    def classes(dmet, classes) -> None:
        ...

    def confusion_vectors(
        dmet,
        iou_thresh: float | List[float] = 0.5,
        bias: float = 0,
        gids: List[int] | None = None,
        compat: str = 'mutex',
        prioritize: str = 'iou',
        ignore_classes: set | str = 'ignore',
        background_class: str | NoParamType = ...,
        verbose: int | str = 'auto',
        workers: int = 0,
        track_probs: str = 'try',
        max_dets: Incomplete | None = ...
    ) -> ConfusionVectors | Dict[float, ConfusionVectors]:
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

    score_coco = score_pycocotools

    @classmethod
    def demo(cls, **kwargs):
        ...

    def summarize(dmet,
                  out_dpath: pathlib.Path | None = None,
                  plot: bool = False,
                  title: str = '',
                  with_bin: str | bool = 'auto',
                  with_ovr: str | bool = 'auto'):
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
