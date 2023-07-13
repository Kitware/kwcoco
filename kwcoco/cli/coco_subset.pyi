import scriptconfig as scfg
from _typeshed import Incomplete


class CocoSubsetCLI:
    name: str

    class CLIConfig(scfg.Config):
        __command__: str
        __default__: Incomplete
        epilog: str

    @classmethod
    def main(cls, cmdline: bool = ..., **kw) -> None:
        ...


def query_subset(dset, config):
    ...
