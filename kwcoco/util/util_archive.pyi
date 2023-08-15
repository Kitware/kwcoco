from types import ModuleType
import tarfile
import zipfile
from _typeshed import Incomplete


class Archive:
    fpath: str | None
    mode: str
    file: tarfile.TarFile | zipfile.ZipFile | None
    backend: str | ModuleType | None

    def __init__(
            self,
            fpath: str | None = None,
            mode: str = 'r',
            backend: str | ModuleType | None = None,
            file: tarfile.TarFile | zipfile.ZipFile | None = None) -> None:
        ...

    def __iter__(self):
        ...

    def names(self):
        ...

    def read(self, name: str, mode: str = 'rb'):
        ...

    @classmethod
    def coerce(cls, data):
        ...

    def add(self, fpath, arcname: Incomplete | None = ...) -> None:
        ...

    def close(self):
        ...

    def __enter__(self):
        ...

    def __exit__(self, *args) -> None:
        ...

    def extractall(self,
                   output_dpath: str = ...,
                   verbose: int = ...,
                   overwrite: bool = ...):
        ...


def unarchive_file(archive_fpath,
                   output_dpath: str = ...,
                   verbose: int = ...,
                   overwrite: bool = ...):
    ...
