from abc import ABC


class AbstractCocoDataset(ABC):
    """
    This is a common base for all variants of the Coco Dataset

    At the time of writing there is kwcoco.CocoDataset (which is the
    dictionary-based backend), and the kwcoco.coco_sql_dataset.CocoSqlDataset,
    which is experimental.
    """
