from typing import Union
from typing import List
from typing import Tuple
import kwimage
from numpy import ndarray
from typing import Dict
import kwimage
import ubelt as ub
from _typeshed import Incomplete
from kwcoco import channel_spec
from kwcoco.util.delayed_ops.delayed_base import DelayedNaryOperation2, DelayedUnaryOperation2


class DelayedStack2(DelayedNaryOperation2):

    def __init__(self, parts, axis) -> None:
        ...

    def __nice__(self):
        ...

    @property
    def shape(self):
        ...


class DelayedConcat2(DelayedNaryOperation2):

    def __init__(self, parts, axis) -> None:
        ...

    def __nice__(self):
        ...

    @property
    def shape(self):
        ...


class DelayedFrameStack2(DelayedStack2):

    def __init__(self, parts) -> None:
        ...


class JaggedArray2(ub.NiceRepr):
    parts: Incomplete
    axis: Incomplete

    def __init__(self, parts, axis) -> None:
        ...

    def __nice__(self):
        ...

    @property
    def shape(self):
        ...


class DelayedChannelConcat2(DelayedConcat2):
    dsize: Incomplete
    num_channels: Incomplete

    def __init__(self,
                 parts,
                 dsize: Incomplete | None = ...,
                 jagged: bool = ...) -> None:
        ...

    def __nice__(self):
        ...

    @property
    def channels(self):
        ...

    @property
    def shape(self):
        ...

    def finalize(self):
        ...

    def optimize(self):
        ...

    comp: Incomplete
    start: Incomplete
    stop: Incomplete
    codes: Incomplete

    def take_channels(
        self, channels: Union[List[int], slice, channel_spec.FusedChannelSpec]
    ) -> DelayedArray2:
        ...

    def crop(self,
             space_slice: Tuple[slice, slice] = None,
             chan_idxs: List[int] = None):
        ...

    def warp(self,
             transform,
             dsize: str = ...,
             antialias: bool = ...,
             interpolation: str = ...):
        ...

    def dequantize(self, quantization):
        ...

    def get_overview(self, overview):
        ...

    def as_xarray(self):
        ...


class DelayedArray2(DelayedUnaryOperation2):

    def __init__(self, subdata: Incomplete | None = ...) -> None:
        ...

    def __nice__(self):
        ...

    @property
    def shape(self):
        ...


class DelayedImage2(DelayedArray2):

    def __init__(self,
                 subdata: Incomplete | None = ...,
                 dsize: Incomplete | None = ...,
                 channels: Incomplete | None = ...) -> None:
        ...

    def __nice__(self):
        ...

    @property
    def shape(self):
        ...

    @property
    def num_channels(self):
        ...

    @property
    def dsize(self):
        ...

    @property
    def channels(self):
        ...

    @channels.setter
    def channels(self, channels) -> None:
        ...

    @property
    def num_overviews(self):
        ...

    def __getitem__(self, sl):
        ...

    def take_channels(
        self, channels: Union[List[int], slice, channel_spec.FusedChannelSpec]
    ) -> DelayedCrop2:
        ...

    def crop(self,
             space_slice: Tuple[slice, slice] = None,
             chan_idxs: List[int] = None):
        ...

    def warp(self,
             transform: Union[ndarray, dict, kwimage.Affine],
             dsize: Union[Tuple[int, int], str] = 'auto',
             antialias: bool = True,
             interpolation: str = 'linear'):
        ...

    def dequantize(self, quantization: Dict):
        ...

    def get_overview(self, overview: int):
        ...

    def as_xarray(self):
        ...


class DelayedAsXarray2(DelayedImage2):

    def finalize(self):
        ...

    def optimize(self):
        ...


class DelayedWarp2(DelayedImage2):

    def __init__(self,
                 subdata: DelayedArray2,
                 transform: Union[ndarray, dict, kwimage.Affine],
                 dsize: Union[Tuple[int, int], str] = 'auto',
                 antialias: bool = True,
                 interpolation: str = 'linear') -> None:
        ...

    def finalize(self):
        ...

    def optimize(self):
        ...


class DelayedDequantize2(DelayedImage2):

    def __init__(self, subdata: DelayedArray2, quantization: Dict) -> None:
        ...

    def finalize(self):
        ...

    def optimize(self):
        ...


class DelayedCrop2(DelayedImage2):
    channels: Incomplete

    def __init__(self,
                 subdata: DelayedArray2,
                 space_slice: Tuple[slice, slice] = None,
                 chan_idxs: Union[List[int], None] = None) -> None:
        ...

    def finalize(self):
        ...

    def optimize(self):
        ...


class DelayedOverview2(DelayedImage2):

    def __init__(self, subdata: DelayedArray2, overview: int):
        ...

    @property
    def num_overviews(self):
        ...

    def finalize(self):
        ...

    def optimize(self):
        ...
