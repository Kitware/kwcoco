"""
mkinit ~/code/kwcoco/kwcoco/__init__.py -w
"""
__version__ = '0.1.5'

__submodules__ = ['coco_dataset']

from kwcoco import coco_dataset

from kwcoco.category_tree import (CategoryTree,)
from kwcoco.coco_dataset import (CocoDataset,)

__all__ = ['CocoDataset', 'CategoryTree', 'coco_dataset']
