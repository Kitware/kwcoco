from typing import Dict
import kwarray
from _typeshed import Incomplete


class KW18(kwarray.DataFrameArray):
    DEFAULT_COLUMNS: Incomplete

    def __init__(self, data) -> None:
        ...

    @classmethod
    def demo(KW18):
        ...

    @classmethod
    def from_coco(KW18, coco_dset):
        ...

    def to_coco(self,
                image_paths: Dict[int, str] | None = None,
                video_name: str | None = None):
        ...

    @classmethod
    def load(KW18, file):
        ...

    @classmethod
    def loads(KW18, text):
        ...

    def dump(self, file) -> None:
        ...

    def dumps(self):
        ...
