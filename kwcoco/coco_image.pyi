from typing import Union
from typing import List
from os import PathLike
import kwcoco
from numpy import ndarray
import kwimage
import ubelt as ub
from _typeshed import Incomplete
from collections.abc import Generator
from typing import Any

DEFAULT_RESOLUTION_KEYS: Incomplete


class CocoImage(ub.NiceRepr):
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

    def __nice__(self):
        ...

    def stats(self):
        ...

    def __contains__(self, key):
        ...

    def __getitem__(self, key):
        ...

    def keys(self):
        ...

    def get(self, key, default=...):
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

    def primary_asset(self,
                      requires: Union[List[str], None] = None) -> None | dict:
        ...

    def iter_image_filepaths(self,
                             with_bundle: bool = True
                             ) -> Generator[Any, None, None]:
        ...

    def iter_asset_objs(self) -> Generator[dict, None, None]:
        ...

    def find_asset_obj(self, channels):
        ...

    def add_auxiliary_item(self,
                           file_name: Union[str, PathLike, None] = None,
                           channels: Union[str, kwcoco.FusedChannelSpec,
                                           None] = None,
                           imdata: Union[ndarray, None] = None,
                           warp_aux_to_img: Union[kwimage.Affine, None] = None,
                           width: Union[int, None] = None,
                           height: Union[int, None] = None,
                           imwrite: bool = False) -> None:
        ...

    add_asset: Incomplete

    def delay(self,
              channels: kwcoco.FusedChannelSpec = None,
              space: str = 'image',
              resolution: Union[None, str, float] = None,
              bundle_dpath: Incomplete | None = ...,
              interpolation: str = ...,
              antialias: bool = ...,
              nodata_method: Incomplete | None = ...,
              RESOLUTION_KEY: Incomplete | None = ...):
        ...

    def valid_region(self, space: str = ...):
        ...

    def warp_vid_from_img(self):
        ...

    def warp_img_from_vid(self):
        ...

    def resolution(self,
                   space: str = ...,
                   RESOLUTION_KEY: Incomplete | None = ...):
        ...


class CocoAsset:
    __key_aliases__: Incomplete
    __key_resolver__: Incomplete
    obj: Incomplete

    def __init__(self, obj) -> None:
        ...

    def __getitem__(self, key):
        ...

    def keys(self):
        ...

    def get(self, key, default=...):
        ...


def parse_quantity(expr):
    ...


def coerce_resolution(expr):
    ...
