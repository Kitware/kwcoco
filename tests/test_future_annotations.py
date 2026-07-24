"""Regression tests for active ``__future__.annotations`` imports."""

import __future__
from pathlib import Path


def test_future_annotations_imports_are_active():
    """Do not accidentally place future imports inside module docstrings."""
    package_dpath = Path(__file__).parents[1] / 'kwcoco'
    compiler_flag = __future__.annotations.compiler_flag
    inactive = []

    for fpath in package_dpath.rglob('*.py'):
        source = fpath.read_text()
        if 'from __future__ import annotations' in source:
            code = compile(source, str(fpath), 'exec', dont_inherit=True)
            if not code.co_flags & compiler_flag:
                inactive.append(fpath.relative_to(package_dpath.parent))

    assert not inactive, 'Inactive future imports: {!r}'.format(inactive)
