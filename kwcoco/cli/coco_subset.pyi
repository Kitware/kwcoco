import scriptconfig as scfg
from _typeshed import Incomplete


class CocoSubsetCLI:
    name: str

    class CocoSubetConfig(scfg.DataConfig):
        __command__: str
        __default__: Incomplete
        __epilog__: str

    CLIConfig = CocoSubetConfig

    @classmethod
    def main(cls, cmdline: bool = ..., **kw) -> None:
        ...


def query_subset(dset, config):
    ...
