from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Generic, TypeVar, overload

import ubelt as ub
from scriptconfig.dict_like import DictLike as DictLike_  # FIXME: do we vendor or not?

# from sqlalchemy.dialects.postgresql import Any


K = TypeVar('K')
V = TypeVar('V')
T = TypeVar('T')


class DictLike(ub.NiceRepr, Generic[K, V]):
    """
    An inherited class must specify the ``getitem``, ``setitem``, and
      ``keys`` methods.

    A class is dictionary like if it has:

    ``__iter__``, ``__len__``, ``__contains__``, ``__getitem__``, ``items``,
    ``keys``, ``values``, ``get``,

    and if it should be writable it should have:
    ``__delitem__``, ``__setitem__``, ``update``,

    And perhaps: ``copy``,


    ``__iter__``, ``__len__``, ``__contains__``, ``__getitem__``, ``items``,
    ``keys``, ``values``, ``get``,

    and if it should be writable it should have:
    ``__delitem__``, ``__setitem__``, ``update``,

    And perhaps: ``copy``,
    """

    def getitem(self, key: K) -> V:
        """
        Args:
            key (Any): a key

        Returns:
            Any: a value
        """
        raise NotImplementedError('abstract getitem function')

    def setitem(self, key: K, value: V) -> None:
        """
        Args:
            key (Any):
            value (Any):
        """
        raise NotImplementedError('abstract setitem function')

    def delitem(self, key: K) -> None:
        """
        Args:
            key (Any):
        """
        raise NotImplementedError('abstract delitem function')

    def keys(self) -> Iterable[K]:
        """
        Yields:
            Any: a key
        """
        raise NotImplementedError('abstract keys function')

    # def __repr__(self):
    #     return repr(self.asdict())

    # def __str__(self):
    #     return str(self.asdict())

    def __len__(self) -> int:
        """
        Returns:
            int:
        """
        return len(list(self.keys()))

    def __iter__(self) -> Iterator[K]:
        return iter(self.keys())

    def __contains__(self, key: object) -> bool:
        """
        Args:
            key (Any):

        Returns:
            bool:
        """
        return key in self.keys()

    def __delitem__(self, key: K) -> None:
        """
        Args:
            key (Any):
        """
        return self.delitem(key)

    def __getitem__(self, key: K) -> V:
        """
        Args:
            key (Any):

        Returns:
            Any:
        """
        return self.getitem(key)

    def __setitem__(self, key: K, value: V) -> None:
        """
        Args:
            key (Any):
            value (Any):
        """
        return self.setitem(key, value)

    def items(self) -> Iterator[tuple[K, V]]:
        """
        Yields:
            Tuple[Any, Any]: a key value pair
        """
        yield from ((key, self[key]) for key in self.keys())

    def values(self) -> Iterator[V]:
        """
        Yields:
            Any: a value
        """
        for key in self.keys():
            yield self[key]
        # yield from (self[key] for key in self.keys())

    def copy(self) -> dict[K, V]:
        """
        Returns:
            Dict:
        """
        return dict(self.items())

    def to_dict(self) -> dict[K, V]:
        """
        Returns:
            Dict:
        """
        # pandas like API
        return dict(self.items())

    # TODO: deprecate and remove
    asdict = to_dict

    def update(self, other: Mapping[K, V]) -> None:
        for k, v in other.items():
            self[k] = v

    @overload
    def get(self, key: K) -> V | None: ...

    @overload
    def get(self, key: K, default: T) -> V | T: ...

    def get(self, key: K, default: Any = None) -> Any:
        """
        Args:
            key (Any):
            default (Any):

        Returns:
            Any:
        """
        try:
            return self[key]
        except KeyError:
            return default


class DictProxy(DictLike_):
    """
    Allows an object to proxy the behavior of a dict attribute
    """

    proxy: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.proxy[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.proxy[key] = value

    def keys(self) -> Iterable[str]:
        return self.proxy.keys()

    def __json__(self) -> dict[str, Any]:
        return ub.odict(self.proxy)
