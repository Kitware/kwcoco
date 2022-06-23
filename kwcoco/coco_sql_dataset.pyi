import pandas
import ubelt as ub
from _typeshed import Incomplete
from collections.abc import Generator
from kwcoco.abstract_coco_dataset import AbstractCocoDataset
from kwcoco.coco_dataset import MixinCocoAccessors, MixinCocoDraw, MixinCocoObjects, MixinCocoStats
from kwcoco.util.dict_like import DictLike
from typing import Any


class CocoBase:
    ...


class Category(CocoBase):
    __tablename__: str
    id: Incomplete
    name: Incomplete
    alias: Incomplete
    supercategory: Incomplete
    extra: Incomplete


class KeypointCategory(CocoBase):
    __tablename__: str
    id: Incomplete
    name: Incomplete
    alias: Incomplete
    supercategory: Incomplete
    reflection_id: Incomplete
    extra: Incomplete


class Video(CocoBase):
    __tablename__: str
    id: Incomplete
    name: Incomplete
    caption: Incomplete
    width: Incomplete
    height: Incomplete
    extra: Incomplete


class Image(CocoBase):
    __tablename__: str
    id: Incomplete
    name: Incomplete
    file_name: Incomplete
    width: Incomplete
    height: Incomplete
    video_id: Incomplete
    timestamp: Incomplete
    frame_index: Incomplete
    channels: Incomplete
    auxiliary: Incomplete
    extra: Incomplete


class Annotation(CocoBase):
    __tablename__: str
    id: Incomplete
    image_id: Incomplete
    category_id: Incomplete
    track_id: Incomplete
    segmentation: Incomplete
    keypoints: Incomplete
    bbox: Incomplete
    score: Incomplete
    weight: Incomplete
    prob: Incomplete
    iscrowd: Incomplete
    caption: Incomplete
    extra: Incomplete


ALCHEMY_MODE_DEFAULT: int
tblname: Incomplete
cls: Incomplete
classname: Incomplete


def orm_to_dict(obj):
    ...


class SqlListProxy(ub.NiceRepr):

    def __init__(proxy, session, cls) -> None:
        ...

    def __len__(proxy):
        ...

    def __nice__(proxy):
        ...

    def __iter__(proxy):
        ...

    def __getitem__(proxy, index):
        ...

    def __contains__(proxy, item) -> None:
        ...

    def __setitem__(proxy, index, value) -> None:
        ...

    def __delitem__(proxy, index) -> None:
        ...


class SqlDictProxy(DictLike):

    def __init__(proxy,
                 session,
                 cls,
                 keyattr: Incomplete | None = ...,
                 ignore_null: bool = ...) -> None:
        ...

    def __len__(proxy):
        ...

    def __nice__(proxy):
        ...

    def __contains__(proxy, key):
        ...

    def __getitem__(proxy, key):
        ...

    def keys(proxy) -> Generator[Any, None, None]:
        ...

    def values(proxy) -> Generator[Any, None, Any]:
        ...

    def items(proxy) -> Generator[Any, None, None]:
        ...


class SqlIdGroupDictProxy(DictLike):

    def __init__(proxy,
                 session,
                 valattr,
                 keyattr,
                 parent_keyattr,
                 group_order_attr: Incomplete | None = ...) -> None:
        ...

    def __nice__(self):
        ...

    def __len__(proxy):
        ...

    def __getitem__(proxy, key):
        ...

    def __contains__(proxy, key):
        ...

    def keys(proxy) -> Generator[Any, None, None]:
        ...

    def items(proxy) -> Generator[Any, None, Any]:
        ...

    def values(proxy) -> Generator[Any, None, None]:
        ...


class CocoSqlIndex:

    def __init__(index) -> None:
        ...

    def build(index, parent) -> None:
        ...


class CocoSqlDatabase(AbstractCocoDataset, MixinCocoAccessors,
                      MixinCocoObjects, MixinCocoStats, MixinCocoDraw,
                      ub.NiceRepr):
    MEMORY_URI: str
    uri: Incomplete
    img_root: Incomplete
    session: Incomplete
    engine: Incomplete
    index: Incomplete
    tag: Incomplete

    def __init__(self,
                 uri: Incomplete | None = ...,
                 tag: Incomplete | None = ...,
                 img_root: Incomplete | None = ...) -> None:
        ...

    def __nice__(self):
        ...

    @classmethod
    def coerce(self, data):
        ...

    def disconnect(self) -> None:
        ...

    def connect(self, readonly: bool = ...):
        ...

    @property
    def fpath(self):
        ...

    def delete(self) -> None:
        ...

    def populate_from(self, dset, verbose: int = ...) -> None:
        ...

    @property
    def dataset(self):
        ...

    @property
    def anns(self):
        ...

    @property
    def cats(self):
        ...

    @property
    def imgs(self):
        ...

    @property
    def name_to_cat(self):
        ...

    def raw_table(self, table_name: str) -> pandas.DataFrame:
        ...

    def tabular_targets(self):
        ...

    @property
    def bundle_dpath(self):
        ...

    @bundle_dpath.setter
    def bundle_dpath(self, value) -> None:
        ...

    @property
    def data_fpath(self):
        ...

    @data_fpath.setter
    def data_fpath(self, value) -> None:
        ...


def cached_sql_coco_view(dct_db_fpath: Incomplete | None = ...,
                         sql_db_fpath: Incomplete | None = ...,
                         dset: Incomplete | None = ...,
                         force_rewrite: bool = ...):
    ...


def ensure_sql_coco_view(dset,
                         db_fpath: Incomplete | None = ...,
                         force_rewrite: bool = ...):
    ...


def demo(num: int = ...):
    ...


def assert_dsets_allclose(dset1, dset2, tag1: str = ..., tag2: str = ...):
    ...


def devcheck() -> None:
    ...
