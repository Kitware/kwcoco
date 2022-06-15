from scriptconfig.dict_like import DictLike


class DictProxy(DictLike):

    def __getitem__(self, key):
        ...

    def __setitem__(self, key, value) -> None:
        ...

    def keys(self):
        ...

    def __json__(self):
        ...
