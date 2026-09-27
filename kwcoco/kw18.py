"""Compatibility shim for :mod:`kwcoco.formats.kw18`."""
from __future__ import annotations

import ubelt as ub

ub.schedule_deprecation(
    'kwcoco',
    'kwcoco.kw18',
    'module',
    'use kwcoco.formats.kw18 instead',
    deprecate='0.8.4',
    error='1.0.0',
    remove='1.1.0',
    stacklevel=2,
)

from kwcoco.formats.kw18 import KW18 as KW18

__all__ = ['KW18']
