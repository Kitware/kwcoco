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
