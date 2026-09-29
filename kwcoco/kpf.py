"""Compatibility shim for :mod:`kwcoco.formats.kpf`."""
from __future__ import annotations

import ubelt as ub

ub.schedule_deprecation(
    'kwcoco',
    'kwcoco.kpf',
    'module',
    'use kwcoco.formats.kpf instead',
    deprecate='0.8.4',
    error='1.0.0',
    remove='1.1.0',
    stacklevel=2,
)

from kwcoco.formats.kpf import KPFStream as KPFStream
from kwcoco.formats.kpf import coco_to_kpf as coco_to_kpf
from kwcoco.formats.kpf import demo as demo

__all__ = ['KPFStream', 'coco_to_kpf', 'demo']
