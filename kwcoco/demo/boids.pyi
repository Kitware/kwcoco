from numpy import ndarray
import ubelt as ub
from _typeshed import Incomplete


class Boids(ub.NiceRepr):
    rng: Incomplete
    num: Incomplete
    dims: Incomplete
    config: Incomplete
    pos: Incomplete
    vel: Incomplete
    acc: Incomplete

    def __init__(self,
                 num,
                 dims: int = ...,
                 rng: Incomplete | None = ...,
                 **kwargs) -> None:
        ...

    def __nice__(self):
        ...

    def initialize(self):
        ...

    rx_to_neighb_cxs: Incomplete
    speeds: Incomplete
    dirs: Incomplete

    def update_neighbors(self) -> None:
        ...

    sep_steering: Incomplete
    com_steering: Incomplete
    align_steering: Incomplete
    rand_steering: Incomplete
    avoid_steering: Incomplete

    def compute_forces(self):
        ...

    def boundary_conditions(self) -> None:
        ...

    def step(self):
        ...

    def paths(self, num_steps):
        ...


def clamp_mag(vec, mag, axis: Incomplete | None = ...):
    ...


def triu_condense_multi_index(multi_index, dims, symetric: bool = ...):
    ...


def closest_point_on_line_segment(pts: ndarray, e1: ndarray,
                                  e2: ndarray) -> ndarray:
    ...
