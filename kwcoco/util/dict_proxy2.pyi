from typing import Any
from typing import Tuple
from _typeshed import Incomplete
from collections.abc import Generator


class DictInterface:

    def keys(self) -> Generator[str, None, None]:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self):
        ...

    def __contains__(self, key: Any) -> bool:
        ...

    def __delitem__(self, key: Any) -> None:
        ...

    def __getitem__(self, key: Any) -> Any:
        ...

    def __setitem__(self, key: Any, value: Any) -> None:
        ...

    def items(self) -> Generator[Tuple[Any, Any], None, Any]:
        ...

    def values(self) -> Generator[Any, None, Any]:
        ...

    def update(self, other) -> None:
        ...

    def get(self, key: Any, default: Any | None = None) -> Any:
        ...


class DictProxy2(DictInterface):

    def __getitem__(self, key):
        ...

    def __setitem__(self, key, value) -> None:
        ...

    def keys(self):
        ...

    def __json__(self):
        ...


class _AliasMetaclass(type):

    @staticmethod
    def __new__(mcls, name, bases, namespace, *args, **kwargs):
        ...


class AliasedDictProxy(DictProxy2, metaclass=_AliasMetaclass):
    __alias_to_primary__: Incomplete

    def __getitem__(self, key):
        ...

    def __setitem__(self, key, value) -> None:
        ...

    def keys(self):
        ...

    def __json__(self):
        ...

    def __contains__(self, key):
        ...
