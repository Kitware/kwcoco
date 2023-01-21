import sortedcontainers
from _typeshed import Incomplete


class _NextId:
    parent: Incomplete
    unused: Incomplete

    def __init__(self, parent) -> None:
        ...

    def get(self, key):
        ...


class _ID_Remapper:
    blocklist: Incomplete
    mapping: Incomplete
    reuse: Incomplete

    def __init__(self, reuse: bool = ...) -> None:
        ...

    def remap(self, old_id):
        ...

    def block_seen(self) -> None:
        ...

    def next_id(self):
        ...


class UniqueNameRemapper:
    suffix_pat: Incomplete

    def __init__(self) -> None:
        ...

    def remap(self, name):
        ...


class SortedSet(sortedcontainers.SortedSet):
    ...


SortedSetQuiet = SortedSet
