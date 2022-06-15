from typing import Tuple
from typing import Union
from typing import List
from typing import Any
import kwimage
import kwimage
import ubelt as ub
from _typeshed import Incomplete
from collections.abc import Generator
from kwcoco import channel_spec
from typing import Any

profile: Incomplete


class DelayedVisionOperation(ub.NiceRepr):

    def __nice__(self):
        ...

    def finalize(self, **kwargs) -> None:
        ...

    def children(self) -> Generator[Any, None, None]:
        ...

    def __json__(self):
        ...

    def nesting(self):
        ...

    def warp(self, *args, **kwargs):
        ...

    def crop(self, *args, **kwargs):
        ...


class DelayedVideoOperation(DelayedVisionOperation):
    ...


class DelayedImageOperation(DelayedVisionOperation):

    def delayed_crop(self, region_slices: Tuple[slice, slice]) -> DelayedCrop:
        ...

    def delayed_warp(self,
                     transform,
                     dsize: Incomplete | None = ...) -> DelayedWarp:
        ...

    def take_channels(self, channels) -> None:
        ...


class DelayedIdentity(DelayedImageOperation):
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

    def take_channels(self, channels):
        ...


def dequantize(quant_data, quantization):
    ...


class DelayedNans(DelayedImageOperation):
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


class DelayedLoad(DelayedImageOperation):
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

    def children(self) -> None:
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


class DelayedFrameConcat(DelayedVideoOperation):
    frames: Incomplete
    dsize: Incomplete
    num_bands: Incomplete
    num_frames: Incomplete
    meta: Incomplete

    def __init__(self, frames, dsize: Incomplete | None = ...) -> None:
        ...

    def children(self) -> None:
        ...

    @property
    def channels(self):
        ...

    @property
    def shape(self):
        ...

    def finalize(self, **kwargs):
        ...

    def delayed_crop(self, region_slices):
        ...

    def delayed_warp(self,
                     transform,
                     dsize: Incomplete | None = ...) -> DelayedWarp:
        ...


class DelayedChannelConcat(DelayedImageOperation):
    components: Incomplete
    dsize: Incomplete
    num_bands: Incomplete
    meta: Incomplete

    def __init__(self, components, dsize: Incomplete | None = ...) -> None:
        ...

    def children(self) -> None:
        ...

    @classmethod
    def random(cls, num_parts: int = ..., rng: Incomplete | None = ...):
        ...

    @property
    def channels(self):
        ...

    @property
    def shape(self):
        ...

    def finalize(self, **kwargs):
        ...

    def delayed_warp(self,
                     transform,
                     dsize: Incomplete | None = ...) -> DelayedWarp:
        ...

    comp: Incomplete
    start: Incomplete
    stop: Incomplete
    codes: Incomplete

    def take_channels(
        self, channels: Union[List[int], slice, channel_spec.FusedChannelSpec]
    ) -> DelayedVisionOperation:
        ...


class DelayedWarp(DelayedImageOperation):
    bounds: Incomplete
    sub_data: Incomplete
    meta: Incomplete

    def __init__(self,
                 sub_data,
                 transform: Incomplete | None = ...,
                 dsize: Incomplete | None = ...) -> None:
        ...

    @classmethod
    def random(cls, nesting=..., rng: Incomplete | None = ...):
        ...

    @property
    def channels(self):
        ...

    def children(self) -> Generator[Any, None, None]:
        ...

    @property
    def dsize(self):
        ...

    @property
    def num_bands(self):
        ...

    @property
    def shape(self):
        ...

    def finalize(self,
                 transform: kwimage.Transform = None,
                 dsize: Tuple[int, int] = None,
                 interpolation: str = ...,
                 **kwargs):
        ...

    def take_channels(self, channels):
        ...


class DelayedCrop(DelayedImageOperation):
    __hack_dont_optimize__: bool
    sub_data: Incomplete
    sub_slices: Incomplete
    num_bands: Incomplete
    shape: Incomplete
    meta: Incomplete

    def __init__(self, sub_data, sub_slices) -> None:
        ...

    @property
    def channels(self):
        ...

    def children(self) -> Generator[Any, None, None]:
        ...

    def finalize(self, **kwargs):
        ...
