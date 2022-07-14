import kwcoco


def grab_camvid_train_test_val_splits(coco_dset, mode: str = ...):
    ...


def grab_camvid_sampler() -> kwcoco.CocoSampler:
    ...


def grab_coco_camvid():
    ...


def grab_raw_camvid():
    ...


def rgb_to_cid(r, g, b):
    ...


def cid_to_rgb(cid):
    ...


def convert_camvid_raw_to_coco(camvid_raw_info):
    ...


def main() -> None:
    ...
