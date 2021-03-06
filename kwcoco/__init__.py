"""
The Kitware COCO module defines a variant of the Microsoft COCO format,
originally developed for the "collected images in context" object detection
challenge. We are backwards compatible with the original module, but we also
have improved implementations in several places, including segmentations and
keypoints.


The :class:`kwcoco.CocoDataset` class is capable of dynamic addition and removal
of categories, images, and annotations. Has better support for keypoints and
segmentation formats than the original COCO format. Despite being written in
Python, this data structure is reasonably efficient.

"""
__version__ = '0.1.9'

__submodules__ = ['coco_dataset']

from kwcoco import coco_dataset

from kwcoco.category_tree import (CategoryTree,)
from kwcoco.coco_dataset import (CocoDataset,)

__all__ = ['CocoDataset', 'CategoryTree', 'coco_dataset']
