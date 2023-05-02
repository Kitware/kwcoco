def fix_msys_path(path):
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
    import os
    was_pathlike = isinstance(path, os.PathLike)
    input_path = path
    new_path = path
    if was_pathlike:
        path = os.fspath(path)
    if isinstance(path, str):
        import re
        drive_pat = '(?P<drive>[A-Za-z])'
        slash_pat = r'[/\\]'
        pat1 = re.compile(f'^{slash_pat}{drive_pat}$')
        pat2 = re.compile(f'^{slash_pat}{drive_pat}{slash_pat}.*$')
        match = pat1.match(path) or pat2.match(path)
        if match is not None:
            drive_name = match.groupdict()['drive'].upper()
            new_path = f'{drive_name}:/{path[3:]}'
    if was_pathlike:
        new_path = input_path.__class__(new_path)
    return new_path


def is_windows_path(path):
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
