import scriptconfig as scfg
from _typeshed import Incomplete


class CocoSplitCLI:
    __command__: str
    name: str

    class CLIConfig(scfg.Config):
        __default__: Incomplete
        __epilog__: str

    @classmethod
    def main(cls, cmdline: bool = ..., **kw) -> None:
        ...
