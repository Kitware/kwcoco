from typing import Union
from os import PathLike
import kwcoco
from typing import Tuple
from numpy.typing import ArrayLike
from typing import List
from _typeshed import Incomplete
from kwcoco.util.delayed_ops.delayed_nodes import DelayedImage2


class DelayedImageLeaf2(DelayedImage2):

    def optimize(self):
        ...


class DelayedLoad2(DelayedImageLeaf2):
    lazy_ref: Incomplete

    def __init__(self,
                 fpath: Union[str, PathLike],
                 channels: Union[int, str, kwcoco.FusedChannelSpec,
                                 None] = None,
                 dsize: Tuple[int, int] = None,
                 nodata_method: Union[str, None] = None) -> None:
        ...

    @property
    def fpath(self):
        ...

    @classmethod
    def demo(DelayedLoad2,
             key: str = ...,
             dsize: Incomplete | None = ...,
             channels: Incomplete | None = ...):
        ...

    def prepare(self):
        ...

    def finalize(self) -> ArrayLike:
        ...


class DelayedNans2(DelayedImageLeaf2):

    def __init__(self,
                 dsize: Incomplete | None = ...,
                 channels: Incomplete | None = ...) -> None:
        ...

    def finalize(self) -> ArrayLike:
        ...

    def crop(self,
             region_slices: Incomplete | None = ...,
             chan_idxs: List[int] = None) -> DelayedImage2:
        ...

    def warp(self,
             transform,
             dsize: Incomplete | None = ...,
             antialias: bool = ...,
             interpolation: str = ...) -> DelayedImage2:
        ...


class DelayedIdentity2(DelayedImageLeaf2):
    data: Incomplete

    def __init__(self,
                 data,
                 channels: Incomplete | None = ...,
                 dsize: Incomplete | None = ...) -> None:
        ...

    def finalize(self) -> ArrayLike:
        ...
