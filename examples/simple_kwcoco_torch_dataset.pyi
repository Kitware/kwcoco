from _typeshed import Incomplete

DatasetBase: Incomplete
DatasetBase = object


class KWCocoSimpleTorchDataset(DatasetBase):
    coco_dset: Incomplete
    input_dims: Incomplete
    antialias: Incomplete
    rng: Incomplete
    gids: Incomplete
    classes: Incomplete
    augment: bool

    def __init__(self,
                 coco_dset,
                 input_dims: Incomplete | None = ...,
                 antialias: bool = ...,
                 rng: Incomplete | None = ...) -> None:
        ...

    def __len__(self):
        ...

    def __getitem__(self, index):
        ...
