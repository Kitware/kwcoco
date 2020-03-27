"""
mkinit ~/code/kwcoco/kwcoco/__init__.py -w
"""
__version__ = '0.0.4'

__submodules__ = ['coco_dataset']

from kwcoco import coco_dataset

from kwcoco.coco_dataset import (CocoDataset,)

__all__ = ['CocoDataset', 'coco_dataset']
