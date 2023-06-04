import scriptconfig as scfg
from _typeshed import Incomplete


class CocoRerootCLI:
    name: str

    class CLIConfig(scfg.DataConfig):
        __epilog__: str
        src: Incomplete
        dst: Incomplete
        new_prefix: Incomplete
        old_prefix: Incomplete
        absolute: Incomplete
        check: Incomplete
        autofix: Incomplete
        compress: Incomplete
        inplace: Incomplete

    @classmethod
    def main(cls, cmdline: bool = ..., **kw) -> None:
        ...


def find_reroot_autofix(dset):
    ...
