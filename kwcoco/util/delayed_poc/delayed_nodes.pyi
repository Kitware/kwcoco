from typing import Any
from typing import Union
from typing import List
from typing import Tuple
import kwimage
import kwimage
import ubelt as ub
from _typeshed import Incomplete
from collections.abc import Generator
from kwcoco import channel_spec
from kwcoco.util.delayed_poc.delayed_base import DelayedImage, DelayedVideo, DelayedVisionMixin as DelayedVisionMixin
from typing import Any

profile: Incomplete


class DelayedFrameStack(DelayedVideo):
    frames: Incomplete
    dsize: Incomplete
    num_bands: Incomplete
    num_frames: Incomplete
    meta: Incomplete

    def __init__(self, frames, dsize: Incomplete | None = ...) -> None:
        ...

    def children(self) -> Generator[Any, None, None]:
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


class JaggedArray(ub.NiceRepr):
    parts: Incomplete
    axis: Incomplete

    def __init__(self, parts, axis) -> None:
        ...

    def __nice__(self):
        ...

    @property
    def shape(self):
        ...


class DelayedChannelStack(DelayedImage):
    components: Incomplete
    jagged: Incomplete
    dsize: Incomplete
    num_bands: Incomplete
    meta: Incomplete

    def __init__(self,
                 components,
                 dsize: Incomplete | None = ...,
                 jagged: bool = ...) -> None:
        ...

    def children(self) -> Generator[Any, None, None]:
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
    ) -> DelayedVisionMixin:
        ...


class DelayedWarp(DelayedImage):
    bounds: Incomplete
    sub_data: Incomplete
    meta: Incomplete

    def __init__(self,
                 sub_data,
                 transform: Incomplete | None = ...,
                 dsize: Incomplete | None = ...) -> None:
        ...

    @classmethod
    def random(cls,
               dsize: Union[Tuple[int, int], None] = None,
               raw_width: Union[int, Tuple[int, int]] = ...,
               raw_height: Union[int, Tuple[int, int]] = ...,
               channels: Union[int, Tuple[int, int]] = ...,
               nesting: Tuple[int, int] = ...,
               rng: Incomplete | None = ...) -> DelayedWarp:
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


class DelayedCrop(DelayedImage):
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
