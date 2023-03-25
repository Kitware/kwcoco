
# Another variant of DictLike Circa 2023
class DictInterface:
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


    Example:
        from scriptconfig.dict_like import DictLike
        class DuckDict(DictLike):
            def __init__(self, _data=None):
                if _data is None:
                    _data = {}
                self._data = _data

            def getitem(self, key):
                return self._data[key]

            def keys(self):
                return self._data.keys()

        self = DuckDict({1: 2, 3: 4})
        print(f'self._data={self._data}')
        cast = dict(self)
        print(f'cast={cast}')
        print(f'self={self}')

    """

    def keys(self):
        """
        Yields:
            str:
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
        return iter(self.keys())

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
        raise NotImplementedError('abstract delitem function')

    def __getitem__(self, key):
        """
        Args:
            key (Any):

        Returns:
            Any:
        """
        raise NotImplementedError('abstract getitem function')

    def __setitem__(self, key, value):
        """
        Args:
            key (Any):
            value (Any):
        """
        raise NotImplementedError('abstract setitem function')

    def items(self):
        """
        Yields:
            Tuple[Any, Any]: a key value pair
        """
        return ((key, self[key]) for key in self.keys())

    def values(self):
        """
        Yields:
            Any: a value
        """
        return (self[key] for key in self.keys())

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


class DictProxy2(DictInterface):
    """
    Allows an object to proxy the behavior of a _proxy dict attribute
    """
    def __getitem__(self, key):
        return self._proxy[key]

    def __setitem__(self, key, value):
        self._proxy[key] = value

    def keys(self):
        return self._proxy.keys()

    def __json__(self):
        return self._proxy


class _AliasMetaclass(type):
    """
    Populates the __alias_to_aliases__ field at class definition time to reduce
    the overhead of instance creation.
    """
    @staticmethod
    def __new__(mcls, name, bases, namespace, *args, **kwargs):
        secondary_to_primary = namespace.get('__alias_to_primary__')
        primary_to_aliases = {}
        for secondary, primary in secondary_to_primary.items():
            if primary not in primary_to_aliases:
                primary_to_aliases[primary] = [primary]
            primary_to_aliases[primary].append(secondary)

        # Build a mapping from all primary and secondary keys to the level set
        # of equivalent aliases
        alias_to_aliases = {}
        for primary, aliases in primary_to_aliases.items():
            alias_to_aliases[primary] = aliases
            for alias in aliases:
                alias_to_aliases[alias] = aliases
        namespace['__alias_to_aliases__'] = alias_to_aliases
        cls = super().__new__(mcls, name, bases, namespace, *args, **kwargs)
        return cls


class AliasedDictProxy(DictProxy2, metaclass=_AliasMetaclass):
    """
    Can have a class attribute called ``__alias_to_primary__ `` which
    is a Dict[str, str] mapping alias-keys to primary-keys.

    Need to handle cases:

        * image dictionary contains no primary / aliased keys
            * primary keys used

        * image dictionary only has aliased keys
            * aliased keys are updated

        * image dictionary only has primary keys
            * primary keys are updated

        * image dictionary only both primary and aliased keys
            * both keys are updated

    Example:
        >>> from kwcoco.util.dict_proxy2 import *  # NOQA
        >>> class MyAliasedObject(AliasedDictProxy):
        >>>     __alias_to_primary__ = {
        >>>         'foo_alias1': 'foo_primary',
        >>>         'foo_alias2': 'foo_primary',
        >>>         'bar_alias1': 'bar_primary',
        >>>     }
        >>>     def __init__(self, obj):
        >>>         self._proxy = obj
        >>>     def __repr__(self):
        >>>         return repr(self._proxy)
        >>>     def __str__(self):
        >>>         return str(self._proxy)
        >>> # Test starting from empty
        >>> obj = MyAliasedObject({})
        >>> obj['regular_key'] = 'val0'
        >>> assert 'foo_primary' not in obj
        >>> assert 'foo_alias1' not in obj
        >>> assert 'foo_alias2' not in obj
        >>> obj['foo_primary'] = 'val1'
        >>> assert 'foo_primary' in obj
        >>> assert 'foo_alias1' in obj
        >>> assert 'foo_alias2' in obj
        >>> obj['foo_alias1'] = 'val2'
        >>> obj['foo_alias2'] = 'val3'
        >>> obj['bar_alias1'] = 'val4'
        >>> obj['bar_primary'] = 'val5'
        >>> assert obj._proxy == {
        >>>     'regular_key': 'val0',
        >>>     'foo_primary': 'val3',
        >>>     'bar_primary': 'val5'}
        >>> # Test starting with primary keys
        >>> obj = MyAliasedObject({
        >>>     'foo_primary': 123,
        >>>     'bar_primary': 123,
        >>> })
        >>> assert 'foo_alias1' in obj
        >>> assert 'bar_alias1' in obj
        >>> obj['bar_alias1'] = 456
        >>> obj['foo_primary'] = 789
        >>> assert obj._proxy == {
        >>>     'foo_primary': 789,
        >>>     'bar_primary': 456}
        >>> # Test that if aliases keys are existant we dont add primary keys
        >>> obj = MyAliasedObject({
        >>>     'foo_alias1': 123,
        >>> })
        >>> assert 'foo_alias1' in obj
        >>> assert 'foo_primary' in obj
        >>> obj['foo_alias1'] = 456
        >>> obj['foo_primary'] = 789
        >>> assert obj._proxy == {
        >>>     'foo_alias1': 789,
        >>> }
        >>> # Test that if primary and aliases keys exist, we update both
        >>> obj = MyAliasedObject({
        >>>     'foo_primary': 3,
        >>>     'foo_alias2': 5,
        >>> })
        >>> # We do not attempt to detect conflicts
        >>> assert obj['foo_primary'] == 3
        >>> assert obj['foo_alias1'] == 3
        >>> assert obj['foo_alias2'] == 5
        >>> obj['foo_alias1'] = 23
        >>> assert obj['foo_primary'] == 23
        >>> assert obj['foo_alias1'] == 23
        >>> assert obj['foo_alias2'] == 23
        >>> obj['foo_primary'] = -12
        >>> assert obj['foo_primary'] == -12
        >>> assert obj['foo_alias1'] == -12
        >>> assert obj['foo_alias2'] == -12
        >>> assert obj._proxy == {
        >>>     'foo_primary': -12,
        >>>     'foo_alias2': -12}

    """
    __alias_to_primary__ = {}

    def __getitem__(self, key):
        try:
            # Try to do the quick thing first
            return self._proxy[key]
        except KeyError:
            # If the given key doesn't exist try one of its aliases
            for alias in self.__alias_to_aliases__.get(key, []):
                if alias in self._proxy:
                    return self._proxy[alias]
            raise

    def __setitem__(self, key, value):
        # Setting will update all aliases in the level-set of equivalent keys
        _needs_set = True
        for alias in self.__alias_to_aliases__.get(key, []):
            if alias in self._proxy:
                self._proxy[alias] = value
                _needs_set = False
        if _needs_set:
            # If no aliases were set, we can add a new key, but if the new key
            # is aliases we force it to only set the primary key
            key = self.__alias_to_primary__.get(key, key)
            self._proxy[key] = value

    def keys(self):
        return self._proxy.keys()

    def __json__(self):
        return self._proxy

    def __contains__(self, key):
        if key in self._proxy:
            return True
        return any(
            alias in self._proxy
            for alias in self.__alias_to_aliases__.get(key, [])
        )
