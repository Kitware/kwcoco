import sqlalchemy
import pandas
import sqlalchemy
import ubelt as ub
from _typeshed import Incomplete
from collections.abc import Generator
from kwcoco.abstract_coco_dataset import AbstractCocoDataset
from kwcoco.coco_dataset import MixinCocoAccessors, MixinCocoDraw, MixinCocoObjects, MixinCocoStats
from kwcoco.util.dict_like import DictLike
from typing import Any

from sqlalchemy.orm import InstrumentedAttribute

__docstubs__: str


class FallbackCocoBase:
    ...


CocoBase: type
sqlalchemy_version: Incomplete
IS_GE_SQLALCH_2x: Incomplete
SQL_ERROR_TYPES: Incomplete
JSONVariant: Incomplete
CocoBase = FallbackCocoBase


def addapt_numpy_float64(numpy_float64):
    ...


def addapt_numpy_int64(numpy_int64):
    ...


UNSTRUCTURED: str
SCHEMA_VERSION: str


class Category(CocoBase):
    __tablename__: str
    id: Incomplete
    name: Incomplete
    alias: Incomplete
    supercategory: Incomplete


class KeypointCategory(CocoBase):
    __tablename__: str
    id: Incomplete
    name: Incomplete
    alias: Incomplete
    supercategory: Incomplete
    reflection_id: Incomplete


class Video(CocoBase):
    __tablename__: str
    id: Incomplete
    name: Incomplete
    caption: Incomplete
    width: Incomplete
    height: Incomplete


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
    warp_img_to_vid: Incomplete
    auxiliary: Incomplete


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


ALCHEMY_MODE_DEFAULT: int
tblname: Incomplete
cls: Incomplete
classname: Incomplete


def orm_to_dict(obj):
    ...


def dict_restructure(item):
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

    def __len__(proxy) -> int:
        ...

    def __nice__(proxy) -> str:
        ...

    def __contains__(proxy, key) -> bool:
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
                 session: sqlalchemy.orm.session.Session,
                 valattr: InstrumentedAttribute,
                 keyattr: InstrumentedAttribute,
                 parent_keyattr: InstrumentedAttribute | None = None,
                 order_attr: InstrumentedAttribute | None = None,
                 order_id: InstrumentedAttribute | None = None) -> None:
        ...

    def __nice__(self) -> str:
        ...

    def __len__(proxy) -> int:
        ...

    def __getitem__(proxy, key):
        ...

    def __contains__(proxy, key) -> bool:
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
    def coerce(self, data, backend: Incomplete | None = ...):
        ...

    def disconnect(self) -> None:
        ...

    def connect(self, readonly: bool = ..., verbose: int = ...):
        ...

    @property
    def fpath(self):
        ...

    def delete(self, verbose: int = ...) -> None:
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

    def pandas_table(self,
                     table_name: str,
                     strict: bool = ...) -> pandas.DataFrame:
        ...

    def raw_table(self, table_name):
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
                         force_rewrite: bool = ...,
                         backend: Incomplete | None = ...):
    ...


def ensure_sql_coco_view(dset,
                         db_fpath: Incomplete | None = ...,
                         force_rewrite: bool = ...,
                         backend: Incomplete | None = ...):
    ...


def demo(num: int = ..., backend: Incomplete | None = ...):
    ...


def assert_dsets_allclose(dset1, dset2, tag1: str = ..., tag2: str = ...):
    ...


def devcheck() -> None:
    ...
