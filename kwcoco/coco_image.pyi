from typing import List
import ubelt as ub
from os import PathLike
import kwcoco
from numpy import ndarray
import kwimage
from typing import Dict
import ubelt as ub
from _typeshed import Incomplete
from collections.abc import Generator
from kwcoco.util.dict_proxy2 import AliasedDictProxy

from kwcoco.coco_objects1d import Annots

__docstubs__: str
DEFAULT_RESOLUTION_KEYS: Incomplete


class CocoImage(AliasedDictProxy, ub.NiceRepr):
    __alias_to_primary__: Incomplete
    img: Incomplete
    dset: Incomplete

    def __init__(self, img, dset: Incomplete | None = ...) -> None:
        ...

    @classmethod
    def from_gid(cls, dset, gid):
        ...

    @property
    def bundle_dpath(self):
        ...

    @bundle_dpath.setter
    def bundle_dpath(self, value) -> None:
        ...

    @property
    def video(self):
        ...

    @video.setter
    def video(self, value) -> None:
        ...

    def detach(self):
        ...

    @property
    def assets(self):
        ...

    @property
    def datetime(self) -> None:
        ...

    def annots(self) -> Annots:
        ...

    def __nice__(self):
        ...

    def stats(self):
        ...

    def __contains__(self, key):
        ...

    def get(self, key, default=...):
        ...

    def keys(self):
        ...

    def __getitem__(self, key):
        ...

    @property
    def channels(self):
        ...

    @property
    def num_channels(self):
        ...

    @property
    def dsize(self):
        ...

    def primary_image_filepath(self, requires: Incomplete | None = ...):
        ...

    def primary_asset(self, requires: List[str] | None = None) -> None | dict:
        ...

    def iter_image_filepaths(self,
                             with_bundle: bool = True
                             ) -> Generator[ub.Path, None, None]:
        ...

    def iter_asset_objs(self) -> Generator[dict, None, None]:
        ...

    def find_asset_obj(self, channels):
        ...

    def add_annotation(self, **ann) -> int:
        ...

    def add_asset(self,
                  file_name: str | PathLike | None = None,
                  channels: str | kwcoco.FusedChannelSpec | None = None,
                  imdata: ndarray | None = None,
                  warp_aux_to_img: kwimage.Affine | None = None,
                  width: int | None = None,
                  height: int | None = None,
                  imwrite: bool = False) -> None:
        ...

    def imdelay(self,
                channels: kwcoco.FusedChannelSpec | None = None,
                space: str = 'image',
                resolution: None | str | float = None,
                bundle_dpath: Incomplete | None = ...,
                interpolation: str = ...,
                antialias: bool = ...,
                nodata_method: Incomplete | None = ...,
                RESOLUTION_KEY: Incomplete | None = ...):
        ...

    def valid_region(self, space: str = ...) -> None | kwimage.MultiPolygon:
        ...

    def warp_vid_from_img(self) -> kwimage.Affine:
        ...

    def warp_img_from_vid(self) -> kwimage.Affine:
        ...

    def resolution(self,
                   space: str = 'image',
                   channel: str | kwcoco.FusedChannelSpec | None = None,
                   RESOLUTION_KEY: Incomplete | None = ...) -> Dict:
        ...

    add_auxiliary_item: Incomplete
    delay: Incomplete

    def show(self, **kwargs):
        ...

    def draw(self, **kwargs):
        ...


class CocoAsset(AliasedDictProxy, ub.NiceRepr):
    __alias_to_primary__: Incomplete

    def __init__(self, asset, bundle_dpath: Incomplete | None = ...) -> None:
        ...

    def __nice__(self):
        ...

    def image_filepath(self):
        ...


def parse_quantity(expr):
    ...


def coerce_resolution(expr):
    ...
