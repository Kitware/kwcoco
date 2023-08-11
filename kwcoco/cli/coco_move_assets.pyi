import scriptconfig as scfg
from _typeshed import Incomplete


class CocoMoveAssetsCLI(scfg.DataConfig):
    src: Incomplete
    dst: Incomplete
    io_workers: Incomplete
    coco_fpaths: Incomplete


class CocoMoveAssetManager:
    jobs: Incomplete
    coco_dsets: Incomplete
    impacted_assets: Incomplete
    impacted_dsets: Incomplete

    def __init__(self, coco_dsets) -> None:
        ...

    def submit(self, src, dst) -> None:
        ...

    def find_impacted(self) -> None:
        ...

    def modify_datasets(self) -> None:
        ...

    def move_files(self) -> None:
        ...

    def dump_datasets(self) -> None:
        ...

    def run(self) -> None:
        ...


def main(cmdline: int = ..., **kwargs) -> None:
    ...
