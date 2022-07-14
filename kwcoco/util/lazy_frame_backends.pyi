import ubelt as ub
from _typeshed import Incomplete
from collections import OrderedDict

profile: Incomplete


class CacheDict(OrderedDict):
    cache_len: Incomplete

    def __init__(self, *args, cache_len: int = 10, **kwargs) -> None:
        ...

    def __setitem__(self, key, value) -> None:
        ...

    def __getitem__(self, key):
        ...


GLOBAL_GDAL_CACHE: Incomplete


class LazySpectralFrameFile(ub.NiceRepr):
    fpath: Incomplete

    def __init__(self, fpath) -> None:
        ...

    @classmethod
    def available(self):
        ...

    @property
    def ndim(self):
        ...

    @property
    def shape(self):
        ...

    @property
    def dtype(self):
        ...

    def __nice__(self):
        ...

    def __getitem__(self, index):
        ...


class LazyRasterIOFrameFile(ub.NiceRepr):
    fpath: Incomplete

    def __init__(self, fpath) -> None:
        ...

    @classmethod
    def available(self):
        ...

    @property
    def ndim(self):
        ...

    @property
    def shape(self):
        ...

    @property
    def dtype(self):
        ...

    def __nice__(self):
        ...

    def __getitem__(self, index):
        ...


class LazyGDalFrameFile(ub.NiceRepr):
    fpath: Incomplete
    nodata_method: Incomplete
    overview: Incomplete

    def __init__(self,
                 fpath,
                 nodata_method: Incomplete | None = ...,
                 overview: Incomplete | None = ...) -> None:
        ...

    @classmethod
    def available(self):
        ...

    def get_overview(self, overview):
        ...

    @classmethod
    def demo(cls, key: str = ..., dsize: Incomplete | None = ...):
        ...

    @property
    def ndim(self):
        ...

    def num_overviews(self):
        ...

    load_overview: Incomplete
    post_overview: Incomplete
    num_channels: Incomplete
    width: Incomplete
    height: Incomplete

    def shape(self):
        ...

    @property
    def dtype(self):
        ...

    def __nice__(self):
        ...

    def __getitem__(self, index):
        ...

    def __array__(self):
        ...
