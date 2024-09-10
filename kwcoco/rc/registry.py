"""
A registry of resource files bundled with the kwcoco package
"""
from importlib import resources as importlib_resources
import ubelt as ub


def requirement_path(fname):
    """

    CommandLine:
        xdoctest -m kwcoco.rc.registry requirement_path

    Example:
        >>> from kwcoco.rc.registry import requirement_path
        >>> fname = 'runtime.txt'
        >>> requirement_path(fname)
    """
    with importlib_resources.path('kwcoco.rc.requirements', f'{fname}') as p:
        orig_pth = ub.Path(p)
        return orig_pth
