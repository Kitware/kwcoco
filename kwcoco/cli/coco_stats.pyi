import scriptconfig as scfg
from _typeshed import Incomplete


class CocoStatsCLI:
    name: str

    class CLIConfig(scfg.Config):
        __command__: str
        __default__: Incomplete
        epilog: str

    @classmethod
    def main(cls, cmdline: bool = ..., **kw):
        ...
