from typing import Tuple
from typing import Any
from typing import Union
import kwcoco
import kwcoco.util.delayed_poc.delayed_nodes
import ubelt as ub
from _typeshed import Incomplete
from collections.abc import Generator
from typing import Any

profile: Incomplete


class DelayedVisionMixin(ub.NiceRepr):

    def __nice__(self):
        ...

    def finalize(self, **kwargs) -> None:
        ...

    def children(self) -> Generator[DelayedVisionMixin, None, None]:
        ...

    def __json__(self):
        ...

    def nesting(self):
        ...

    def warp(self, *args, **kwargs):
        ...

    def crop(self, *args, **kwargs):
        ...


class DelayedVideo(DelayedVisionMixin):
    ...


class DelayedImage(DelayedVisionMixin):

    def delayed_crop(
            self, region_slices: Tuple[slice, slice]) -> DelayedImage:
        ...

    def delayed_warp(
        self,
        transform: Any,
        dsize: Union[Tuple[int, int], None] = None
    ) -> kwcoco.util.delayed_poc.delayed_nodes.DelayedWarp:
        ...

    def take_channels(self, channels: Any) -> DelayedVisionMixin:
        ...
