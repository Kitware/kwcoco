import scriptconfig as scfg
from _typeshed import Incomplete


class CocoSplitCLI:
    name: str

    class CLIConfig(scfg.Config):
        default: Incomplete
        epilog: str

    @classmethod
    def main(cls, cmdline: bool = ..., **kw) -> None:
        ...
