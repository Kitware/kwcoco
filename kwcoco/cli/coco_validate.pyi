import scriptconfig as scfg
from _typeshed import Incomplete

__autogen_cli_args__: str


class CocoValidateCLI:
    name: str

    class CLIConfig(scfg.DataConfig):
        __default__: Incomplete
        __epilog__: str

    @classmethod
    def main(cls, cmdline: bool = ..., **kw) -> None:
        ...
