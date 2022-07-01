from typing import Union
from typing import Tuple
import kwcoco
from numpy.typing import ArrayLike
from typing import List
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
    def shape(self) -> None | Tuple[int | None, ...]:
        ...


class DelayedConcat2(DelayedNaryOperation2):

    def __init__(self, parts, axis) -> None:
        ...

    def __nice__(self):
        ...

    @property
    def shape(self) -> None | Tuple[int | None, ...]:
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
    def channels(self) -> None | kwcoco.FusedChannelSpec:
        ...

    @property
    def shape(self) -> Tuple[int | None, int | None, int | None]:
        ...

    def finalize(self) -> ArrayLike:
        ...

    def optimize(self) -> DelayedImage2:
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
             chan_idxs: List[int] = None) -> DelayedArray2:
        ...

    def warp(self,
             transform,
             dsize: str = ...,
             antialias: bool = ...,
             interpolation: str = ...) -> DelayedArray2:
        ...

    def dequantize(self, quantization) -> DelayedArray2:
        ...

    def get_overview(self, overview) -> DelayedArray2:
        ...

    def as_xarray(self) -> DelayedAsXarray2:
        ...


class DelayedArray2(DelayedUnaryOperation2):

    def __init__(self, subdata: Incomplete | None = ...) -> None:
        ...

    def __nice__(self):
        ...

    @property
    def shape(self) -> None | Tuple[int | None, ...]:
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
    def shape(self) -> None | Tuple[int | None, int | None, int | None]:
        ...

    @property
    def num_channels(self) -> None | int:
        ...

    @property
    def dsize(self) -> None | Tuple[int | None, int | None]:
        ...

    @property
    def channels(self) -> None | kwcoco.FusedChannelSpec:
        ...

    @channels.setter
    def channels(self, channels) -> None | kwcoco.FusedChannelSpec:
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
             chan_idxs: List[int] = None) -> DelayedImage2:
        ...

    def warp(self,
             transform: Union[ndarray, dict, kwimage.Affine],
             dsize: Union[Tuple[int, int], str] = 'auto',
             antialias: bool = True,
             interpolation: str = 'linear') -> DelayedImage2:
        ...

    def dequantize(self, quantization: Dict) -> DelayedDequantize2:
        ...

    def get_overview(self, overview: int) -> DelayedOverview2:
        ...

    def as_xarray(self) -> DelayedAsXarray2:
        ...


class DelayedAsXarray2(DelayedImage2):

    def finalize(self) -> ArrayLike:
        ...

    def optimize(self) -> DelayedImage2:
        ...


class DelayedWarp2(DelayedImage2):

    def __init__(self,
                 subdata: DelayedArray2,
                 transform: Union[ndarray, dict, kwimage.Affine],
                 dsize: Union[Tuple[int, int], str] = 'auto',
                 antialias: bool = True,
                 interpolation: str = 'linear') -> None:
        ...

    def finalize(self) -> ArrayLike:
        ...

    def optimize(self) -> DelayedImage2:
        ...


class DelayedDequantize2(DelayedImage2):

    def __init__(self, subdata: DelayedArray2, quantization: Dict) -> None:
        ...

    def finalize(self) -> ArrayLike:
        ...

    def optimize(self) -> DelayedImage2:
        ...


class DelayedCrop2(DelayedImage2):
    channels: Incomplete

    def __init__(self,
                 subdata: DelayedArray2,
                 space_slice: Tuple[slice, slice] = None,
                 chan_idxs: Union[List[int], None] = None) -> None:
        ...

    def finalize(self) -> ArrayLike:
        ...

    def optimize(self) -> DelayedImage2:
        ...


class DelayedOverview2(DelayedImage2):

    def __init__(self, subdata: DelayedArray2, overview: int):
        ...

    @property
    def num_overviews(self):
        ...

    def finalize(self) -> ArrayLike:
        ...

    def optimize(self) -> DelayedImage2:
        ...
