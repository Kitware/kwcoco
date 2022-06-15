from _typeshed import Incomplete


class Archive:
    fpath: Incomplete
    mode: Incomplete
    file: Incomplete
    backend: Incomplete

    def __init__(self,
                 fpath: Incomplete | None = ...,
                 mode: str = ...,
                 backend: Incomplete | None = ...,
                 file: Incomplete | None = ...) -> None:
        ...

    def __iter__(self):
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
