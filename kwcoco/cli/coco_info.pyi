from typing import Type
from types import TracebackType
import scriptconfig as scfg
from _typeshed import Incomplete


class CocoFileHelper:
    fpath: Incomplete
    file: Incomplete
    zfile: Incomplete
    mode: Incomplete

    def __init__(self, fpath, mode: str = ...) -> None:
        ...

    def __enter__(self):
        ...

    def __exit__(self, ex_type: Type[BaseException] | None,
                 ex_value: BaseException | None,
                 ex_traceback: TracebackType | None) -> bool | None:
        ...


class CocoInfoCLI(scfg.DataConfig):
    __command__: str
    src: Incomplete
    show_info: Incomplete
    show_licenses: Incomplete
    show_categories: Incomplete
    show_videos: Incomplete
    show_images: Incomplete
    show_tracks: Incomplete
    show_annotations: Incomplete
    rich: Incomplete
    verbose: Incomplete
    image_name: Incomplete

    @classmethod
    def main(cls, cmdline: int = ..., **kwargs) -> None:
        ...


__cli__ = CocoInfoCLI
main: Incomplete
