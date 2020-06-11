import ubelt as ub
from scriptconfig.dict_like import DictLike


class DictProxy(DictLike):
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
