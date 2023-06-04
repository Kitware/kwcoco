import scriptconfig as scfg
from _typeshed import Incomplete


class CocoMove(scfg.DataConfig):
    __command__: str
    __alias__: Incomplete
    src: Incomplete
    dst: Incomplete
    absolute: Incomplete
    check: Incomplete

    @classmethod
    def main(CocoMove, cmdline: int = ..., **kwargs) -> None:
        ...


__config__ = CocoMove
