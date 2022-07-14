from typing import Tuple
from typing import Union
from numpy.random import RandomState
import kwcoco
from numpy import ndarray
from typing import List
from _typeshed import Incomplete

TOYDATA_IMAGE_VERSION: int


def demodata_toy_dset(image_size: Tuple[int, int] = ...,
                      n_imgs: int = 5,
                      verbose: int = 3,
                      rng: Union[int, RandomState] = 0,
                      newstyle: bool = True,
                      dpath: str = None,
                      bundle_dpath: str = None,
                      aux: bool = None,
                      use_cache: bool = True,
                      **kwargs) -> kwcoco.CocoDataset:
    ...


def demodata_toy_img(anchors: ndarray = None,
                     image_size=...,
                     categories: List[str] = None,
                     n_annots: Union[Tuple, int] = ...,
                     fg_scale: float = 0.5,
                     bg_scale: float = 0.8,
                     bg_intensity: float = 0.1,
                     fg_intensity: float = 0.9,
                     gray: bool = ...,
                     centerobj: bool = None,
                     exact: bool = False,
                     newstyle: bool = True,
                     rng: RandomState = None,
                     aux: Incomplete | None = ...,
                     **kwargs):
    ...
