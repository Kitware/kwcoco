from __future__ import annotations

import os
from typing import Any, cast, TypeVar, overload


PathLikeT = TypeVar('PathLikeT', bound=os.PathLike[str])


@overload
def fix_msys_path(path: str) -> str: ...


@overload
def fix_msys_path(path: PathLikeT) -> PathLikeT: ...


def fix_msys_path(path: str | os.PathLike[str]) -> str | os.PathLike[str]:
    r"""
    Windows is so special. When using msys bash if you pass a path on the CLI
    it resolves /c to C:/, but if you have you path as part of a config string,
    it doesnt know how to do that, and at that point Python doesn't handle the
    msys style /c paths. This is a hack detects and fixes this in this
    location.

    Example:
        >>> print(fix_msys_path('/c/Users/foobar'))
        C:/Users/foobar
        >>> print(fix_msys_path(r'\c\Users\foobar'))
        C:/Users\foobar
        >>> print(fix_msys_path(r'\d\Users\foobar'))
        D:/Users\foobar
        >>> print(fix_msys_path(r'\z'))
        Z:/
        >>> import pathlib
        >>> assert fix_msys_path(pathlib.Path(r'\z')) == pathlib.Path('Z:/')
    """
    import re

    input_path = path
    path_text = os.fspath(path)
    new_text = path_text

    drive_pat = '(?P<drive>[A-Za-z])'
    slash_pat = r'[/\\]'
    pat1 = re.compile(f'^{slash_pat}{drive_pat}$')
    pat2 = re.compile(f'^{slash_pat}{drive_pat}{slash_pat}.*$')
    match = pat1.match(path_text) or pat2.match(path_text)
    if match is not None:
        drive_name = match.groupdict()['drive'].upper()
        new_text = f'{drive_name}:/{path_text[3:]}'

    if isinstance(input_path, str):
        return new_text
    if new_text == path_text:
        return input_path

    # os.PathLike is a protocol and does not promise a constructor, but this
    # function historically preserves concrete path classes by reconstructing
    # them from a string. Keep that runtime contract explicit at this boundary.
    path_type = cast(Any, type(input_path))
    return path_type(new_text)


def is_windows_path(path: str) -> bool:
    r"""
    Example:
        >>> assert is_windows_path('C:')
        >>> assert is_windows_path('C:/')
        >>> assert is_windows_path('C:\\')
        >>> assert is_windows_path('C:/foo')
        >>> assert is_windows_path('C:\\foo')
        >>> assert not is_windows_path('/foo')
    """
    import re

    drive_pat = '(?P<drive>[A-Za-z])'
    slash_pat = r'[/\\]'
    pat1 = re.compile(f'^{drive_pat}:$')
    pat2 = re.compile(f'^{drive_pat}:{slash_pat}.*$')
    match = pat1.match(path) or pat2.match(path)
    return bool(match)
