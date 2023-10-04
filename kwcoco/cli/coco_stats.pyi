import scriptconfig as scfg
from _typeshed import Incomplete


class CocoStatsCLI:
    name: str

    class CLIConfig(scfg.DataConfig):
        __command__: str
        src: Incomplete
        basic: Incomplete
        extended: Incomplete
        catfreq: Incomplete
        boxes: Incomplete
        image_size: Incomplete
        annot_attrs: Incomplete
        image_attrs: Incomplete
        video_attrs: Incomplete
        io_workers: Incomplete
        embed: Incomplete
        __epilog__: str

    @classmethod
    def main(cls, cmdline: bool = ..., **kw):
        ...


main: Incomplete
