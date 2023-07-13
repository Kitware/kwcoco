from typing import Tuple
from numpy.random import RandomState
from os import PathLike
import kwcoco
from numpy import ndarray
from typing import List

TOYDATA_IMAGE_VERSION: int


def demodata_toy_dset(image_size: Tuple[int, int] = ...,
                      n_imgs: int = 5,
                      verbose: int = 3,
                      rng: int | RandomState | None = 0,
                      newstyle: bool = True,
                      dpath: str | PathLike | None = None,
                      fpath: str | PathLike | None = None,
                      bundle_dpath: str | PathLike | None = None,
                      aux: bool | None = None,
                      use_cache: bool = True,
                      **kwargs) -> kwcoco.CocoDataset:
    ...


def demodata_toy_img(anchors: ndarray | None = None,
                     image_size=...,
                     categories: List[str] | None = None,
                     n_annots: Tuple | int = ...,
                     fg_scale: float = 0.5,
                     bg_scale: float = 0.8,
                     bg_intensity: float = 0.1,
                     fg_intensity: float = 0.9,
                     gray: bool = ...,
                     centerobj: bool | None = None,
                     exact: bool = False,
                     newstyle: bool = True,
                     rng: RandomState | int | None = None,
                     aux: bool | None = None,
                     **kwargs):
    ...
