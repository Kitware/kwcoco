import scriptconfig as scfg
from _typeshed import Incomplete


class CocoToyDataCLI:
    name: str

    class CLIConfig(scfg.Config):
        __default__: Incomplete
        epilog: str

    @classmethod
    def main(cls, cmdline: bool = ..., **kw) -> None:
        ...
