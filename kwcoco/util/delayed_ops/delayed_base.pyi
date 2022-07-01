from typing import Any
import ubelt as ub
from _typeshed import Incomplete
from collections.abc import Generator
from typing import Any


class DelayedOperation2(ub.NiceRepr):
    meta: Incomplete

    def __init__(self) -> None:
        ...

    def __nice__(self):
        ...

    def nesting(self):
        ...

    def as_graph(self):
        ...

    def write_network_text(self, with_labels: bool = ...) -> None:
        ...

    @property
    def shape(self) -> None:
        ...

    def children(self) -> Generator[Any, None, None]:
        ...

    def finalize(self) -> None:
        ...

    def optimize(self) -> None:
        ...


class DelayedNaryOperation2(DelayedOperation2):
    parts: Incomplete

    def __init__(self, parts) -> None:
        ...

    def children(self) -> Generator[Any, None, None]:
        ...


class DelayedUnaryOperation2(DelayedOperation2):
    subdata: Incomplete

    def __init__(self, subdata) -> None:
        ...

    def children(self) -> Generator[Any, None, None]:
        ...
