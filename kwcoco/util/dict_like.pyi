import ubelt as ub
from typing import Any
from _typeshed import Incomplete
from collections.abc import Generator


class DictLike(ub.NiceRepr):

    def getitem(self, key) -> None:
        ...

    def setitem(self, key, value) -> None:
        ...

    def delitem(self, key) -> None:
        ...

    def keys(self) -> Generator[Any, None, None]:
        ...

    def __len__(self):
        ...

    def __iter__(self):
        ...

    def __contains__(self, key):
        ...

    def __delitem__(self, key):
        ...

    def __getitem__(self, key):
        ...

    def __setitem__(self, key, value):
        ...

    def items(self):
        ...

    def values(self):
        ...

    def copy(self):
        ...

    def to_dict(self):
        ...

    asdict: Incomplete

    def update(self, other) -> None:
        ...

    def get(self, key, default: Incomplete | None = ...):
        ...
