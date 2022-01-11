"""
Generates "toydata" for demo and testing purposes.

Note:
    The implementation of `demodata_toy_img` and `demodata_toy_dset` should be
    redone using the tools built for `random_video_dset`, which have more
    extensible implementations.
"""

from .toydata_video import random_single_video_dset, random_video_dset
from .toydata_image import demodata_toy_dset, demodata_toy_img


__all__ = ['demodata_toy_dset', 'random_single_video_dset',
           'random_video_dset', 'demodata_toy_img']
