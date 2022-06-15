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
                  imgIds=...,
                  catIds=...,
                  areaRng=...,
                  iscrowd: Incomplete | None = ...):
        ...

    def getCatIds(self, catNms=..., supNms=..., catIds=...):
        ...

    def getImgIds(self, imgIds=..., catIds=...):
        ...

    def loadAnns(self, ids=...):
        ...

    def loadCats(self, ids=...):
        ...

    def loadImgs(self, ids=...):
        ...

    def showAnns(self, anns, draw_bbox: bool = ...) -> None:
        ...

    def loadRes(self, resFile):
        ...

    def download(self, tarDir: Incomplete | None = ..., imgIds=...) -> None:
        ...

    def loadNumpyAnnotations(self, data):
        ...

    def annToRLE(self, ann):
        ...

    def annToMask(self, ann):
        ...
