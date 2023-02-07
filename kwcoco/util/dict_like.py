import ubelt as ub
from scriptconfig.dict_like import DictLike as DictLike_  # FIXME: do we vendor or not?


class DictLike(ub.NiceRepr):
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

    def getitem(self, key):
        """
        Args:
            key (Any): a key

        Returns:
            Any: a value
        """
        raise NotImplementedError('abstract getitem function')

    def setitem(self, key, value):
        """
        Args:
            key (Any):
            value (Any):
        """
        raise NotImplementedError('abstract setitem function')

    def delitem(self, key):
        """
        Args:
            key (Any):
        """
        raise NotImplementedError('abstract delitem function')

    def keys(self):
        """
        Yields:
            Any: a key
        """
        raise NotImplementedError('abstract keys function')

    # def __repr__(self):
    #     return repr(self.asdict())

    # def __str__(self):
    #     return str(self.asdict())

    def __len__(self):
        """
        Returns:
            int:
        """
        return len(list(self.keys()))

    def __iter__(self):
        return self.keys()

    def __contains__(self, key):
        """
        Args:
            key (Any):

        Returns:
            bool:
        """
        return key in self.keys()

    def __delitem__(self, key):
        """
        Args:
            key (Any):
        """
        return self.delitem(key)

    def __getitem__(self, key):
        """
        Args:
            key (Any):

        Returns:
            Any:
        """
        return self.getitem(key)

    def __setitem__(self, key, value):
        """
        Args:
            key (Any):
            value (Any):
        """
        return self.setitem(key, value)

    def items(self):
        """
        Yields:
            Tuple[Any, Any]: a key value pair
        """
        yield from ((key, self[key]) for key in self.keys())

    def values(self):
        """
        Yields:
            Any: a value
        """
        for key in self.keys():
            yield self[key]
        # yield from (self[key] for key in self.keys())

    def copy(self):
        """
        Returns:
            Dict:
        """
        return dict(self.items())

    def to_dict(self):
        """
        Returns:
            Dict:
        """
        # pandas like API
        return dict(self.items())

    # TODO: deprecate and remove
    asdict = to_dict

    def update(self, other):
        for k, v in other.items():
            self[k] = v

    def get(self, key, default=None):
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
    def __getitem__(self, key):
        return self.proxy[key]

    def __setitem__(self, key, value):
        self.proxy[key] = value

    def keys(self):
        return self.proxy.keys()

    def __json__(self):
        return ub.odict(self.proxy)
