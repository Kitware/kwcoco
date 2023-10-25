from typing import List
from typing import Iterable
from typing import Dict
from typing import Callable
from typing import Any
import kwimage
import ubelt as ub

from kwcoco.coco_dataset import CocoDataset

ObjT = Dict
__docstubs__: str


class ObjectList1D(ub.NiceRepr):

    def __init__(self, ids: List[int], dset: CocoDataset, key: str) -> None:
        ...

    def __nice__(self) -> str:
        ...

    def __iter__(self):
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int | slice) -> ObjectList1D | int:
        ...

    def unique(self) -> ObjectList1D:
        ...

    @property
    def ids(self) -> List[int]:
        ...

    @property
    def objs(self) -> List[ObjT]:
        ...

    def take(self, idxs) -> ObjectList1D:
        ...

    def compress(self, flags) -> ObjectList1D:
        ...

    def peek(self) -> ObjT:
        ...

    def lookup(self,
               key: str | Iterable,
               default=...,
               keepid: bool = ...) -> Dict[str, ObjT]:
        ...

    def sort_values(self,
                    by: str,
                    reverse: bool = ...,
                    key: Callable | None = None) -> ObjectList1D:
        ...

    def get(self,
            key: str,
            default=...,
            keepid: bool = ...) -> Dict[int, Any] | List[Any]:
        ...

    def set(self, key: str, values: Iterable | Any) -> None:
        ...

    def attribute_frequency(self) -> Dict[str, int]:
        ...


class ObjectGroups(ub.NiceRepr):

    def __init__(self, groups: List[ObjectList1D], dset: CocoDataset) -> None:
        ...

    def __getitem__(self, index):
        ...

    def lookup(self, key, default=...):
        ...

    def __nice__(self) -> str:
        ...


class Categories(ObjectList1D):

    def __init__(self, ids: List[int], dset: CocoDataset) -> None:
        ...

    @property
    def cids(self):
        ...

    @property
    def name(self):
        ...

    @property
    def supercategory(self):
        ...


class Videos(ObjectList1D):

    def __init__(self, ids: List[int], dset: CocoDataset) -> None:
        ...

    @property
    def images(self) -> ImageGroups:
        ...


class Images(ObjectList1D):

    def __init__(self, ids: List[int], dset: CocoDataset) -> None:
        ...

    @property
    def coco_images(self):
        ...

    @property
    def gids(self):
        ...

    @property
    def gname(self):
        ...

    @property
    def gpath(self):
        ...

    @property
    def width(self):
        ...

    @property
    def height(self):
        ...

    @property
    def size(self):
        ...

    @property
    def area(self) -> List[float]:
        ...

    @property
    def n_annots(self) -> List[int]:
        ...

    @property
    def aids(self) -> List[set]:
        ...

    @property
    def annots(self) -> AnnotGroups:
        ...


class Annots(ObjectList1D):

    def __init__(self, ids: List[int], dset: CocoDataset) -> None:
        ...

    @property
    def aids(self):
        ...

    @property
    def images(self) -> Images:
        ...

    @property
    def image_id(self):
        ...

    @property
    def category_id(self):
        ...

    @property
    def gids(self) -> List[int]:
        ...

    @property
    def cids(self) -> List[int]:
        ...

    @property
    def cnames(self) -> List[str]:
        ...

    @cnames.setter
    def cnames(self, cnames) -> List[str]:
        ...

    @property
    def category_names(self) -> List[str]:
        ...

    @category_names.setter
    def category_names(self, names) -> List[str]:
        ...

    @property
    def detections(self) -> kwimage.Detections:
        ...

    @property
    def boxes(self) -> kwimage.Boxes:
        ...

    @boxes.setter
    def boxes(self, boxes) -> kwimage.Boxes:
        ...

    @property
    def xywh(self) -> List[List[int]]:
        ...


class Tracks(ObjectList1D):

    def __init__(self, ids: List[int], dset: CocoDataset) -> None:
        ...

    @property
    def track_ids(self):
        ...

    @property
    def name(self):
        ...

    @property
    def annots(self):
        ...


class AnnotGroups(ObjectGroups):

    @property
    def cids(self) -> List[List[int]]:
        ...

    @property
    def cnames(self) -> List[List[str]]:
        ...


class ImageGroups(ObjectGroups):
    ...
