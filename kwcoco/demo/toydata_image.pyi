from typing import Tuple
from typing import Union
from numpy.random import RandomState
from os import PathLike
import kwcoco
from numpy import ndarray
from typing import List

TOYDATA_IMAGE_VERSION: int


def demodata_toy_dset(image_size: Tuple[int, int] = ...,
                      n_imgs: int = 5,
                      verbose: int = 3,
                      rng: Union[int, RandomState, None] = 0,
                      newstyle: bool = True,
                      dpath: Union[str, PathLike, None] = None,
                      fpath: Union[str, PathLike, None] = None,
                      bundle_dpath: Union[str, PathLike, None] = None,
                      aux: Union[bool, None] = None,
                      use_cache: bool = True,
                      **kwargs) -> kwcoco.CocoDataset:
    ...


def demodata_toy_img(anchors: Union[ndarray, None] = None,
                     image_size=...,
                     categories: Union[List[str], None] = None,
                     n_annots: Union[Tuple, int] = ...,
                     fg_scale: float = 0.5,
                     bg_scale: float = 0.8,
                     bg_intensity: float = 0.1,
                     fg_intensity: float = 0.9,
                     gray: bool = ...,
                     centerobj: Union[bool, None] = None,
                     exact: bool = False,
                     newstyle: bool = True,
                     rng: Union[RandomState, int, None] = None,
                     aux: Union[bool, None] = None,
                     **kwargs):
    ...
