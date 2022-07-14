from typing import Dict
import networkx
from typing import Tuple
from typing import Any
from numpy.typing import ArrayLike
import ubelt as ub
from _typeshed import Incomplete
from collections.abc import Generator
from typing import Any


class DelayedOperation2(ub.NiceRepr):
    meta: Incomplete

    def __init__(self) -> None:
        ...

    def __nice__(self) -> str:
        ...

    def nesting(self) -> Dict[str, dict]:
        ...

    def as_graph(self) -> networkx.DiGraph:
        ...

    def write_network_text(self, with_labels: bool = ...) -> None:
        ...

    @property
    def shape(self) -> None | Tuple[int | None, ...]:
        ...

    def children(self) -> Generator[Any, None, None]:
        ...

    def finalize(self) -> ArrayLike:
        ...

    def optimize(self) -> DelayedOperation2:
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
