import ubelt as ub


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
        raise NotImplementedError('abstract getitem function')

    def setitem(self, key, value):
        raise NotImplementedError('abstract setitem function')

    def delitem(self, key):
        raise NotImplementedError('abstract delitem function')

    def keys(self):
        raise NotImplementedError('abstract keys function')

    # def __repr__(self):
    #     return repr(self.asdict())

    # def __str__(self):
    #     return str(self.asdict())

    def __len__(self):
        return len(list(self.keys()))

    def __iter__(self):
        return self.keys()

    def __contains__(self, key):
        return key in self.keys()

    def __delitem__(self, key):
        return self.delitem(key)

    def __getitem__(self, key):
        return self.getitem(key)

    def __setitem__(self, key, value):
        return self.setitem(key, value)

    def items(self):
        return ((key, self[key]) for key in self.keys())

    def values(self):
        return (self[key] for key in self.keys())

    def copy(self):
        return dict(self.items())

    def to_dict(self):
        # pandas like API
        return dict(self.items())

    asdict = to_dict

    def update(self, other):
        for k, v in other.items():
            self[k] = v

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
