from typing import Dict
from typing import List
from _typeshed import Incomplete


class CategoryPatterns:

    @classmethod
    def coerce(CategoryPatterns, data: Incomplete | None = ..., **kwargs):
        ...

    rng: Incomplete
    fg_scale: Incomplete
    fg_intensity: Incomplete
    categories: Incomplete
    cname_to_kp: Incomplete
    obj_catnames: Incomplete
    kp_catnames: Incomplete
    keypoint_categories: Incomplete
    cname_to_cid: Incomplete
    cname_to_cx: Incomplete

    def __init__(self,
                 categories: List[Dict] = None,
                 fg_scale: float = ...,
                 fg_intensity: float = ...,
                 rng: Incomplete | None = ...):
        ...

    def __len__(self):
        ...

    def __getitem__(self, index):
        ...

    def __iter__(self):
        ...

    def index(self, name):
        ...

    def get(self, index, default=...):
        ...

    def random_category(self,
                        chip,
                        xy_offset: Incomplete | None = ...,
                        dims: Incomplete | None = ...,
                        newstyle: bool = ...,
                        size: Incomplete | None = ...):
        ...

    def render_category(self,
                        cname,
                        chip,
                        xy_offset: Incomplete | None = ...,
                        dims: Incomplete | None = ...,
                        newstyle: bool = ...,
                        size: Incomplete | None = ...):
        ...


def star(a, dtype=...):
    ...


class Rasters:

    @staticmethod
    def superstar():
        ...

    @staticmethod
    def eff():
        ...
