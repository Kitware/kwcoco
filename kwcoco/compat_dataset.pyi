from typing import List
from typing import Dict
from os import PathLike
import kwimage
from numpy import ndarray
from _typeshed import Incomplete
from kwcoco.coco_dataset import CocoDataset


class COCO(CocoDataset):

    def __init__(self, annotation_file: Incomplete | None = ..., **kw) -> None:
        ...

    def createIndex(self) -> None:
        ...

    def info(self) -> None:
        ...

    parent: Incomplete

    @property
    def imgToAnns(self):
        ...

    @property
    def catToImgs(self):
        ...

    def getAnnIds(self,
                  imgIds: List[int] = ...,
                  catIds: List[int] = ...,
                  areaRng: List[float] = ...,
                  iscrowd: bool | None = None) -> List[int]:
        ...

    def getCatIds(self,
                  catNms: List[str] = ...,
                  supNms: List[str] = ...,
                  catIds: List[int] = ...) -> List[int]:
        ...

    def getImgIds(self,
                  imgIds: List[int] = ...,
                  catIds: List[int] = ...) -> List[int]:
        ...

    def loadAnns(self, ids: List[int] = ...) -> List[dict]:
        ...

    def loadCats(self, ids: List[int] = ...) -> List[dict]:
        ...

    def loadImgs(self, ids: List[int] = ...) -> List[dict]:
        ...

    def showAnns(self, anns: List[Dict], draw_bbox: bool = ...) -> None:
        ...

    def loadRes(self, resFile: str) -> object:
        ...

    def download(self,
                 tarDir: str | PathLike | None = None,
                 imgIds: list = ...) -> None:
        ...

    def loadNumpyAnnotations(self, data) -> List[Dict]:
        ...

    def annToRLE(self, ann) -> kwimage.Mask:
        ...

    def annToMask(self, ann) -> ndarray:
        ...
