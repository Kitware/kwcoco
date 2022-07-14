from typing import Union
from typing import List
from typing import Dict
from torch import Tensor
import abc
import ubelt as ub
from _typeshed import Incomplete
from collections.abc import Generator
from typing import Any


class BaseChannelSpec(ub.NiceRepr):

    @property
    @abc.abstractmethod
    def spec(self) -> str:
        ...

    @classmethod
    @abc.abstractmethod
    def coerce(
            cls, data: Union[str, int, list, dict,
                             BaseChannelSpec]) -> BaseChannelSpec:
        ...

    @abc.abstractmethod
    def streams(self) -> List[FusedChannelSpec]:
        ...

    @abc.abstractmethod
    def normalize(self) -> BaseChannelSpec:
        ...

    @abc.abstractmethod
    def intersection(self, other):
        ...

    @abc.abstractmethod
    def union(self, other):
        ...

    @abc.abstractmethod
    def difference(self):
        ...

    @abc.abstractmethod
    def issubset(self, other):
        ...

    @abc.abstractmethod
    def issuperset(self, other):
        ...

    def __sub__(self, other):
        ...

    def __nice__(self):
        ...

    def __json__(self):
        ...

    def __and__(self, other):
        ...

    def __or__(self, other):
        ...

    def path_sanitize(self, maxlen: int = None) -> str:
        ...


class FusedChannelSpec(BaseChannelSpec):
    parsed: Incomplete

    def __init__(self, parsed, _is_normalized: bool = ...) -> None:
        ...

    def __len__(self):
        ...

    def __getitem__(self, index):
        ...

    @classmethod
    def concat(cls, items):
        ...

    def spec(self):
        ...

    def unique(self):
        ...

    @classmethod
    def parse(cls, spec):
        ...

    def __eq__(self, other):
        ...

    @classmethod
    def coerce(cls, data):
        ...

    def concise(self) -> FusedChannelSpec:
        ...

    def normalize(self) -> FusedChannelSpec:
        ...

    def numel(self):
        ...

    def sizes(self) -> List[int]:
        ...

    def __contains__(self, key):
        ...

    def code_list(self):
        ...

    def as_list(self):
        ...

    def as_oset(self):
        ...

    def as_set(self):
        ...

    to_set: Incomplete
    to_oset: Incomplete
    to_list: Incomplete

    def as_path(self):
        ...

    def __set__(self):
        ...

    def difference(self, other):
        ...

    def intersection(self, other):
        ...

    def union(self, other):
        ...

    def issubset(self, other):
        ...

    def issuperset(self, other):
        ...

    def component_indices(self, axis: int = ...):
        ...

    def streams(self):
        ...

    def fuse(self):
        ...


class ChannelSpec(BaseChannelSpec):

    def __init__(self, spec, parsed: Incomplete | None = ...) -> None:
        ...

    @property
    def spec(self):
        ...

    def __contains__(self, key):
        ...

    @property
    def info(self):
        ...

    @classmethod
    def coerce(cls, data) -> ChannelSpec:
        ...

    def parse(self):
        ...

    def concise(self):
        ...

    def normalize(self) -> ChannelSpec:
        ...

    def keys(self) -> Generator[Any, None, None]:
        ...

    def values(self):
        ...

    def items(self):
        ...

    def fuse(self) -> FusedChannelSpec:
        ...

    def streams(self):
        ...

    def code_list(self):
        ...

    def as_path(self):
        ...

    def difference(self, other):
        ...

    def intersection(self, other):
        ...

    def union(self, other):
        ...

    def issubset(self, other) -> None:
        ...

    def issuperset(self, other) -> None:
        ...

    def numel(self):
        ...

    def sizes(self):
        ...

    def unique(self, normalize: bool = ...):
        ...

    def encode(self,
               item: Dict[str, Tensor],
               axis: int = 0,
               mode: int = ...) -> Dict[str, Tensor]:
        ...

    def decode(self, inputs: Dict[str, Tensor], axis: int = 1):
        ...

    def component_indices(self, axis: int = ...):
        ...


def subsequence_index(oset1, oset2) -> None | slice:
    ...


def oset_insert(self, index, obj) -> None:
    ...


def oset_delitem(self, index) -> None:
    ...
