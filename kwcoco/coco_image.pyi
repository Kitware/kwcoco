from typing import List
from typing import Union
import kwcoco
from numpy import ndarray
import kwimage
import ubelt as ub
from _typeshed import Incomplete
from collections.abc import Generator
from typing import Any


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

    def primary_asset(self, requires: List[str] = None):
        ...

    def iter_image_filepaths(self) -> Generator[Any, None, None]:
        ...

    def iter_asset_objs(self) -> Generator[dict, None, None]:
        ...

    def find_asset_obj(self, channels):
        ...

    def add_auxiliary_item(self,
                           file_name: Union[str, None] = None,
                           channels: Union[str,
                                           kwcoco.FusedChannelSpec] = None,
                           imdata: Union[ndarray, None] = None,
                           warp_aux_to_img: kwimage.Affine = None,
                           width: int = None,
                           height: int = None,
                           imwrite: bool = False) -> None:
        ...

    add_asset: Incomplete

    def delay(self,
              channels: Incomplete | None = ...,
              space: str = ...,
              bundle_dpath: Incomplete | None = ...,
              interpolation: str = ...,
              antialias: bool = ...,
              nodata_method: Incomplete | None = ...,
              jagged: bool = ...,
              mode: int = ...):
        ...

    def valid_region(self, space: str = ...):
        ...

    def warp_vid_from_img(self):
        ...

    def warp_img_from_vid(self):
        ...


class CocoAsset:

    def __getitem__(self, key):
        ...

    def keys(self):
        ...

    def get(self, key, default=...):
        ...
