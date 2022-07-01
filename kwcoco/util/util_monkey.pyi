from _typeshed import Incomplete


class SupressPrint:
    mods: Incomplete
    enabled: Incomplete
    oldprints: Incomplete

    def __init__(self, *mods, **kw) -> None:
        ...

    def __enter__(self) -> None:
        ...

    def __exit__(self, a, b, c) -> None:
        ...


class Reloadable(type):

    def __subclasscheck__(cls, sub):
        ...

    def __instancecheck__(cls, inst):
        ...

    @classmethod
    def add_metaclass(metaclass, cls):
        ...

    @classmethod
    def developing(metaclass, cls):
        ...
