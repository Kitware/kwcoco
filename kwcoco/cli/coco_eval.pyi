import scriptconfig as scfg
from _typeshed import Incomplete


class CocoEvalCLIConfig(scfg.Config):
    __doc__: Incomplete
    default: Incomplete


class CocoEvalCLI:
    name: str
    CLIConfig: Incomplete

    @classmethod
    def main(cls, cmdline: bool = ..., **kw) -> None:
        ...


def main(cmdline: bool = ..., **kw):
    ...
