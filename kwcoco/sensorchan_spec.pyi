import lark
from _typeshed import Incomplete

SENSOR_CHAN_GRAMMAR: Incomplete


class Fused:
    chan: Incomplete

    def __init__(self, chan) -> None:
        ...


class SensorChan:
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


class ConciseFused:
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
