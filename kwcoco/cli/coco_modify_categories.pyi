import scriptconfig as scfg
from _typeshed import Incomplete


class CocoModifyCatsCLI:
    name: str

    class CLIConfig(scfg.Config):
        epilog: str
        __default__: Incomplete

    @classmethod
    def main(cls, cmdline: bool = ..., **kw) -> None:
        ...
