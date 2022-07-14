import ubelt as ub
from _typeshed import Incomplete
from lark import Transformer

cache: Incomplete


SENSOR_CHAN_GRAMMAR: Incomplete


class SensorChanSpec(ub.NiceRepr):
    spec: Incomplete

    def __init__(self, spec: str) -> None:
        ...

    def __nice__(self):
        ...

    @classmethod
    def coerce(cls, data) -> SensorChanSpec:
        ...

    def normalize(self):
        ...

    def concise(self):
        ...

    def streams(self):
        ...


class SensorChanNode:
    sensor: Incomplete
    chan: Incomplete

    def __init__(self, sensor, chan) -> None:
        ...

    @property
    def spec(self):
        ...


class FusedChanNode:
    data: Incomplete

    def __init__(self, chan) -> None:
        ...

    @property
    def spec(self):
        ...

    def concise(self):
        ...


class SensorChanTransformer(Transformer):
    consise_channels: Incomplete
    concise_sensors: Incomplete

    def __init__(self,
                 concise_channels: int = ...,
                 concise_sensors: int = ...) -> None:
        ...

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


def normalize_sensor_chan(spec):
    ...


def concise_sensor_chan(spec):
    ...


def sensorchan_concise_parts(spec):
    ...


def sensorchan_normalized_parts(spec):
    ...
