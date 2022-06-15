import lark
import ubelt as ub
from _typeshed import Incomplete

SENSOR_CHAN_GRAMMAR: Incomplete
CHANNEL_ONLY_GRAMMAR: Incomplete


class Fused(ub.NiceRepr):
    chan: Incomplete

    def __init__(self, chan) -> None:
        ...


class SensorChan(ub.NiceRepr):
    sensor: Incomplete
    chan: Incomplete

    def __init__(self, sensor, chan) -> None:
        ...


class NormalizeTransformer(lark.Transformer):

    def chan_id(self, items):
        ...

    def chan_single(self, items):
        ...

    def chan_getitem(self, items):
        ...

    def chan_getslice_0b(self, items):
        ...

    def chan_getslice_ab(self, items):
        ...

    def chan_code(self, items):
        ...

    def sensor_seq(self, items):
        ...

    def fused_seq(self, items):
        ...

    def fused(self, items):
        ...

    def channel_rhs(self, items):
        ...

    def sensor_lhs(self, items):
        ...

    def nosensor_chan(self, items):
        ...

    def sensor_chan(self, items):
        ...

    def stream_item(self, items):
        ...

    def stream(self, items):
        ...


class ConciseFused(ub.NiceRepr):
    chan: Incomplete
    concise: Incomplete

    def __init__(self, chan) -> None:
        ...


class ConciseTransformer(lark.Transformer):

    def chan_id(self, items):
        ...

    def chan_single(self, items):
        ...

    def chan_getitem(self, items):
        ...

    def chan_getslice_0b(self, items):
        ...

    def chan_getslice_ab(self, items):
        ...

    def chan_code(self, items):
        ...

    def sensor_seq(self, items):
        ...

    def fused_seq(self, items):
        ...

    def fused(self, items):
        ...

    def channel_rhs(self, items):
        ...

    def sensor_lhs(self, items):
        ...

    def nosensor_chan(self, items):
        ...

    def sensor_chan(self, items):
        ...

    def stream_item(self, items):
        ...

    def stream(self, items):
        ...


class ConcisePartsTransformer(ConciseTransformer):

    def stream(self, items):
        ...


def concise_sensor_chan(spec):
    ...


def sensorchan_parts(spec):
    ...
