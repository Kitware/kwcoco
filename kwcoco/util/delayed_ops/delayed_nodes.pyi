from typing import List
from typing import Union
from typing import Tuple
from numpy.typing import ArrayLike
import kwcoco
import kwimage
from numpy import ndarray
from typing import Dict
from typing import Any
import kwimage
import ubelt as ub
from _typeshed import Incomplete
from kwcoco import channel_spec
from kwcoco.util.delayed_ops.delayed_base import DelayedNaryOperation2, DelayedUnaryOperation2
from typing import Any


class DelayedStack2(DelayedNaryOperation2):

    def __init__(self, parts: List[DelayedArray2], axis: int) -> None:
        ...

    def __nice__(self) -> str:
        ...

    @property
    def shape(self) -> None | Tuple[int | None, ...]:
        ...


class DelayedConcat2(DelayedNaryOperation2):

    def __init__(self, parts: List[DelayedArray2], axis: int) -> None:
        ...

    def __nice__(self):
        ...

    @property
    def shape(self) -> None | Tuple[int | None, ...]:
        ...


class DelayedFrameStack2(DelayedStack2):

    def __init__(self, parts: List[DelayedArray2]) -> None:
        ...


class JaggedArray2(ub.NiceRepr):
    parts: Incomplete
    axis: Incomplete

    def __init__(self, parts: List[ArrayLike],
                 axis: List[DelayedArray2]) -> None:
        ...

    def __nice__(self) -> str:
        ...

    @property
    def shape(self) -> List[None | Tuple[int | None, ...]]:
        ...


class DelayedChannelConcat2(DelayedConcat2):
    dsize: Incomplete
    num_channels: Incomplete

    def __init__(self,
                 parts: List[DelayedArray2],
                 dsize: Union[Tuple[int, int], None] = None,
                 jagged: bool = ...) -> None:
        ...

    def __nice__(self) -> str:
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
             transform: Union[ndarray, dict, kwimage.Affine],
             dsize: Union[Tuple[int, int], str] = 'auto',
             antialias: bool = True,
             interpolation: str = 'linear') -> DelayedArray2:
        ...

    def dequantize(self, quantization: Dict[str, Any]) -> DelayedArray2:
        ...

    def get_overview(self, overview: int) -> DelayedArray2:
        ...

    def as_xarray(self) -> DelayedAsXarray2:
        ...


class DelayedArray2(DelayedUnaryOperation2):

    def __init__(self, subdata: DelayedArray2 = None) -> None:
        ...

    def __nice__(self) -> str:
        ...

    @property
    def shape(self) -> None | Tuple[int | None, ...]:
        ...


class DelayedImage2(DelayedArray2):

    def __init__(
            self,
            subdata: DelayedArray2 = None,
            dsize: Union[None, Tuple[Union[int, None], Union[int,
                                                             None]]] = None,
            channels: Union[None, int,
                            kwcoco.FusedChannelSpec] = None) -> None:
        ...

    def __nice__(self) -> str:
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
    def num_overviews(self) -> int:
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

    def dequantize(self, quantization: Dict[str, Any]) -> DelayedDequantize2:
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
    def num_overviews(self) -> int:
        ...

    def finalize(self) -> ArrayLike:
        ...

    def optimize(self) -> DelayedImage2:
        ...
