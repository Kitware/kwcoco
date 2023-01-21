from typing import Union
from types import ModuleType
import tarfile
import zipfile
import tarfile
import zipfile
from _typeshed import Incomplete


class Archive:
    fpath: Union[str, None]
    mode: str
    file: Union[tarfile.TarFile, zipfile.ZipFile, None]
    backend: Union[str, ModuleType, None]

    def __init__(
            self,
            fpath: Union[str, None] = None,
            mode: str = 'r',
            backend: Union[str, ModuleType, None] = None,
            file: Union[tarfile.TarFile, zipfile.ZipFile,
                        None] = None) -> None:
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
