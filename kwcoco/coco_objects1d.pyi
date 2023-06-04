from typing import List
from typing import Iterable
from typing import Dict
from typing import Any
import kwimage
import ubelt as ub
from typing import Any

from typing import Dict

ObjT = Dict
__docstubs__: str


class ObjectList1D(ub.NiceRepr):

    def __init__(self, ids, dset, key) -> None:
        ...

    def __nice__(self):
        ...

    def __iter__(self):
        ...

    def __len__(self):
        ...

    def __getitem__(self, index):
        ...

    def unique(self) -> ObjectList1D:
        ...

    @property
    def ids(self):
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

    def get(self,
            key: str,
            default=...,
            keepid: bool = ...) -> Dict[str, ObjT]:
        ...

    def set(self, key: str, values: Iterable | Any) -> None:
        ...

    def attribute_frequency(self) -> Dict[str, int]:
        ...


class ObjectGroups(ub.NiceRepr):

    def __init__(self, groups, dset) -> None:
        ...

    def __getitem__(self, index):
        ...

    def lookup(self, key, default=...):
        ...

    def __nice__(self):
        ...


class Categories(ObjectList1D):

    def __init__(self, ids, dset) -> None:
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

    def __init__(self, ids, dset) -> None:
        ...

    @property
    def images(self):
        ...


class Images(ObjectList1D):

    def __init__(self, ids, dset) -> None:
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
    def area(self):
        ...

    @property
    def n_annots(self):
        ...

    @property
    def aids(self):
        ...

    @property
    def annots(self):
        ...


class Annots(ObjectList1D):

    def __init__(self, ids, dset) -> None:
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
    def cnames(self) -> List[int]:
        ...

    @cnames.setter
    def cnames(self, cnames) -> List[int]:
        ...

    @property
    def category_names(self) -> List[int]:
        ...

    @category_names.setter
    def category_names(self, names) -> List[int]:
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


class AnnotGroups(ObjectGroups):

    @property
    def cids(self) -> List[List[int]]:
        ...

    @property
    def cnames(self) -> List[List[str]]:
        ...


class ImageGroups(ObjectGroups):
    ...
