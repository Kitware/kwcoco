import scriptconfig as scfg
from _typeshed import Incomplete


class CocoGrabCLI:
    name: str

    class CLIConfig(scfg.Config):
        default: Incomplete

    @classmethod
    def main(cls, cmdline: bool = ..., **kw) -> None:
        ...
