from typing import List
from typing import Union
from typing import Iterable
from typing import Any
from typing import TypeVar
import ubelt as ub

ObjT = TypeVar('ObjT')


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

    def unique(self):
        ...

    @property
    def objs(self) -> List[ObjT]:
        ...

    def take(self, idxs):
        ...

    def compress(self, flags):
        ...

    def peek(self) -> ObjT:
        ...

    def lookup(self,
               key: Union[str, Iterable],
               default=...,
               keepid: bool = ...):
        ...

    def get(self, key: str, default=..., keepid: bool = ...):
        ...

    def set(self, key: str, values: Union[Iterable, Any]) -> None:
        ...

    def attribute_frequency(self):
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
    def images(self):
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
    def cids(self):
        ...

    @property
    def cnames(self):
        ...

    @cnames.setter
    def cnames(self, cnames) -> None:
        ...

    @property
    def detections(self):
        ...

    @property
    def boxes(self):
        ...

    @boxes.setter
    def boxes(self, boxes) -> None:
        ...

    @property
    def xywh(self):
        ...


class AnnotGroups(ObjectGroups):

    @property
    def cids(self):
        ...

    @property
    def cnames(self):
        ...


class ImageGroups(ObjectGroups):
    ...
