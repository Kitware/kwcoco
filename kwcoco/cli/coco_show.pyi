import scriptconfig as scfg
from _typeshed import Incomplete


class CocoShowCLI:
    name: str

    class CLIConfig(scfg.Config):
        __epilog__: str
        __default__: Incomplete

    @classmethod
    def main(cls, cmdline: bool = ..., **kw) -> None:
        ...
