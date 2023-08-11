import scriptconfig as scfg
from _typeshed import Incomplete


class CocoUnionCLI:
    name: str

    class CLIConfig(scfg.DataConfig):
        __command__: str
        src: Incomplete
        dst: Incomplete
        absolute: Incomplete
        remember_parent: Incomplete
        io_workers: Incomplete
        compress: Incomplete
        __epilog__: str

    @classmethod
    def main(cls, cmdline: bool = ..., **kw) -> None:
        ...
