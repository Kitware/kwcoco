from typing import Tuple
import ubelt as ub
from _typeshed import Incomplete

profile: Incomplete


class DelayedVisionOperation(ub.NiceRepr):

    def __nice__(self):
        ...

    def finalize(self, **kwargs) -> None:
        ...

    def children(self) -> None:
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
                     dsize: Incomplete | None = ...) -> DelayedCrop:
        ...

    def take_channels(self, channels) -> None:
        ...
