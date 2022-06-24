from typing import Any
from typing import Tuple
from typing import Union
from typing import List
from numpy import ndarray
from typing import Dict
from _typeshed import Incomplete
from collections.abc import Generator
from kwcoco import channel_spec
from kwcoco.util.delayed_poc.delayed_base import DelayedImage
from typing import Any

profile: Incomplete


class DelayedNans(DelayedImage):
    meta: Incomplete

    def __init__(self,
                 dsize: Incomplete | None = ...,
                 channels: Incomplete | None = ...) -> None:
        ...

    @property
    def shape(self):
        ...

    @property
    def num_bands(self):
        ...

    @property
    def dsize(self):
        ...

    @property
    def channels(self):
        ...

    def children(self) -> Generator[Any, None, None]:
        ...

    def finalize(self, **kwargs):
        ...

    def delayed_crop(self, region_slices):
        ...

    def delayed_warp(self, transform, dsize: Incomplete | None = ...):
        ...


class DelayedLoad(DelayedImage):
    __hack_dont_optimize__: bool
    meta: Incomplete
    cache: Incomplete
    quantization: Incomplete

    def __init__(self,
                 fpath,
                 channels: Incomplete | None = ...,
                 dsize: Incomplete | None = ...,
                 num_bands: Incomplete | None = ...,
                 immediate_crop: Incomplete | None = ...,
                 immediate_chan_idxs: Incomplete | None = ...,
                 immediate_dsize: Incomplete | None = ...,
                 quantization: Incomplete | None = ...) -> None:
        ...

    @classmethod
    def demo(DelayedLoad, key: str = ..., dsize: Incomplete | None = ...):
        ...

    @classmethod
    def coerce(cls, data) -> None:
        ...

    def children(self) -> Generator[Any, None, None]:
        ...

    def nesting(self):
        ...

    def load_shape(self, use_channel_heuristic: bool = ...):
        ...

    @property
    def shape(self):
        ...

    @property
    def num_bands(self):
        ...

    @property
    def dsize(self):
        ...

    @property
    def channels(self):
        ...

    @property
    def fpath(self):
        ...

    def finalize(self, **kwargs):
        ...

    def delayed_crop(self, region_slices: Tuple[slice, slice]) -> DelayedLoad:
        ...

    def take_channels(
        self, channels: Union[List[int], slice, channel_spec.FusedChannelSpec]
    ) -> DelayedLoad:
        ...


def dequantize(quant_data: ndarray, quantization: Dict[str, Any]) -> ndarray:
    ...


class DelayedIdentity(DelayedImage):
    __hack_dont_optimize__: bool
    sub_data: Incomplete
    meta: Incomplete
    cache: Incomplete
    dsize: Incomplete
    quantization: Incomplete
    num_bands: Incomplete
    shape: Incomplete
    channels: Incomplete

    def __init__(self,
                 sub_data,
                 dsize: Incomplete | None = ...,
                 channels: Incomplete | None = ...,
                 quantization: Incomplete | None = ...) -> None:
        ...

    @classmethod
    def demo(cls,
             key: str = ...,
             chan: Incomplete | None = ...,
             dsize: Incomplete | None = ...):
        ...

    def children(self) -> Generator[Any, None, None]:
        ...

    def finalize(self, **kwargs):
        ...

    def take_channels(self, channels) -> DelayedIdentity:
        ...
