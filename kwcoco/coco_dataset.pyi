import kwcoco
from typing import Union
import numpy as np
from numpy import ndarray
from os import PathLike
from numpy.typing import ArrayLike
import networkx
from typing import List
from typing import Tuple
from typing import Callable
import kwcoco.coco_objects1d
from typing import Dict
import kwimage
from typing import Any
from typing import IO
import ubelt as ub
from _typeshed import Incomplete
from kwcoco.abstract_coco_dataset import AbstractCocoDataset
from typing import Any

SPEC_KEYS: Incomplete


class MixinCocoDepricate:
    ...


class MixinCocoAccessors:

    def delayed_load(self,
                     gid: int,
                     channels: kwcoco.FusedChannelSpec = None,
                     space: str = 'image'):
        ...

    def load_image(self,
                   gid_or_img: Union[int, dict],
                   channels: Union[str, None] = None) -> np.ndarray:
        ...

    def get_image_fpath(self,
                        gid_or_img: Union[int, dict],
                        channels: str = None) -> PathLike:
        ...

    def get_auxiliary_fpath(self, gid_or_img: Union[int, dict], channels: str):
        ...

    def load_annot_sample(self,
                          aid_or_ann,
                          image: ArrayLike = None,
                          pad: Incomplete | None = ...):
        ...

    def category_graph(self) -> networkx.DiGraph:
        ...

    def object_categories(self) -> kwcoco.CategoryTree:
        ...

    def keypoint_categories(self) -> kwcoco.CategoryTree:
        ...

    def coco_image(self, gid: int) -> kwcoco.coco_image.CocoImage:
        ...


class MixinCocoExtras:

    @classmethod
    def coerce(cls, key, **kw):
        ...

    tag: Incomplete
    fpath: Incomplete

    @classmethod
    def demo(cls, key: str = 'photos', **kwargs):
        ...

    @classmethod
    def random(cls, rng: Incomplete | None = ...):
        ...

    def missing_images(self,
                       check_aux: bool = False,
                       verbose: int = 0) -> List[Tuple[int, str, int]]:
        ...

    def corrupted_images(self,
                         check_aux: bool = False,
                         verbose: int = 0) -> List[Tuple[int, str, int]]:
        ...

    def rename_categories(self,
                          mapper: Union[dict, Callable],
                          rebuild: bool = ...,
                          merge_policy: str = 'ignore') -> None:
        ...

    bundle_dpath: Incomplete

    def reroot(self,
               new_root: str = None,
               old_prefix: str = None,
               new_prefix: str = None,
               absolute: bool = False,
               check: bool = True,
               safe: bool = True):
        ...

    @property
    def data_root(self):
        ...

    @data_root.setter
    def data_root(self, value) -> None:
        ...

    @property
    def img_root(self):
        ...

    @img_root.setter
    def img_root(self, value) -> None:
        ...

    @property
    def data_fpath(self):
        ...

    @data_fpath.setter
    def data_fpath(self, value) -> None:
        ...


class MixinCocoObjects:

    def annots(self,
               aids: List[int] = None,
               gid: int = None,
               trackid: int = None) -> kwcoco.coco_objects1d.Annots:
        ...

    def images(self,
               gids: List[int] = None,
               vidid: int = None,
               names: List[str] = None) -> kwcoco.coco_objects1d.Images:
        ...

    def categories(self,
                   cids: List[int] = None) -> kwcoco.coco_objects1d.Categories:
        ...

    def videos(self,
               vidids: List[int] = None,
               names: List[str] = None) -> kwcoco.coco_objects1d.Videos:
        ...


class MixinCocoStats:

    @property
    def n_annots(self):
        ...

    @property
    def n_images(self):
        ...

    @property
    def n_cats(self):
        ...

    @property
    def n_videos(self):
        ...

    def keypoint_annotation_frequency(self):
        ...

    def category_annotation_frequency(self):
        ...

    def category_annotation_type_frequency(self):
        ...

    def conform(self, **config) -> None:
        ...

    def validate(self, **config) -> dict:
        ...

    def stats(self, **kwargs) -> dict:
        ...

    def basic_stats(self):
        ...

    def extended_stats(self):
        ...

    def boxsize_stats(
            self,
            anchors: int = None,
            perclass: bool = True,
            gids: List[int] = None,
            aids: List[int] = None,
            verbose: int = 0,
            clusterkw: dict = ...,
            statskw: dict = ...) -> Dict[str, Dict[str, Dict | ndarray]]:
        ...

    def find_representative_images(self,
                                   gids: Union[None, List] = None) -> List:
        ...


class MixinCocoDraw:

    def imread(self, gid):
        ...

    def draw_image(self,
                   gid: int,
                   channels: kwcoco.ChannelSpec = None) -> ndarray:
        ...

    def show_image(self,
                   gid: int = None,
                   aids: list = None,
                   aid: int = None,
                   channels: Incomplete | None = ...,
                   **kwargs):
        ...


class MixinCocoAddRemove:

    def add_video(self, name: str, id: Union[None, int] = None, **kw) -> int:
        ...

    def add_image(self,
                  file_name: Union[str, None] = None,
                  id: Union[None, int] = None,
                  **kw) -> int:
        ...

    def add_auxiliary_item(self,
                           gid: int,
                           file_name: Union[str, None] = None,
                           channels: Union[str,
                                           kwcoco.FusedChannelSpec] = None,
                           **kwargs) -> None:
        ...

    def add_annotation(self,
                       image_id: int,
                       category_id: Union[int, None] = None,
                       bbox: Union[list, kwimage.Boxes] = ...,
                       segmentation: Union[Dict, List, Any] = ...,
                       keypoints: Any = ...,
                       id: Union[None, int] = None,
                       **kw) -> int:
        ...

    def add_category(self,
                     name: str,
                     supercategory: Union[str, None] = None,
                     id: Union[int, None] = None,
                     **kw) -> int:
        ...

    def ensure_image(self,
                     file_name: str,
                     id: Union[None, int] = None,
                     **kw) -> int:
        ...

    def ensure_category(self,
                        name,
                        supercategory: Incomplete | None = ...,
                        id: Incomplete | None = ...,
                        **kw) -> int:
        ...

    def add_annotations(self, anns: List[Dict]) -> None:
        ...

    def add_images(self, imgs: List[Dict]) -> None:
        ...

    def clear_images(self) -> None:
        ...

    def clear_annotations(self) -> None:
        ...

    def remove_annotation(self, aid_or_ann) -> None:
        ...

    def remove_annotations(self,
                           aids_or_anns,
                           verbose: int = ...,
                           safe: bool = True) -> Dict:
        ...

    def remove_categories(self,
                          cat_identifiers: List,
                          keep_annots: bool = False,
                          verbose: int = ...,
                          safe: bool = True) -> Dict:
        ...

    def remove_images(self,
                      gids_or_imgs: List,
                      verbose: int = ...,
                      safe: bool = True) -> Dict:
        ...

    def remove_videos(self,
                      vidids_or_videos: List,
                      verbose: int = ...,
                      safe: bool = True) -> Dict:
        ...

    def remove_annotation_keypoints(self, kp_identifiers: List) -> Dict:
        ...

    def remove_keypoint_categories(self, kp_identifiers: List) -> Dict:
        ...

    def set_annotation_category(self, aid_or_ann: Union[dict, int],
                                cid_or_cat: Union[dict, int]) -> None:
        ...


class CocoIndex:

    def __init__(index) -> None:
        ...

    def __bool__(index):
        ...

    __nonzero__: Incomplete
    index: Incomplete

    @property
    def cid_to_gids(index):
        ...

    def clear(index) -> None:
        ...

    def build(index, parent: kwcoco.CocoDataset) -> None:
        ...


class MixinCocoIndex:

    @property
    def anns(self):
        ...

    @property
    def imgs(self):
        ...

    @property
    def cats(self):
        ...

    @property
    def gid_to_aids(self):
        ...

    @property
    def cid_to_aids(self):
        ...

    @property
    def name_to_cat(self):
        ...


class CocoDataset(AbstractCocoDataset, MixinCocoAddRemove, MixinCocoStats,
                  MixinCocoObjects, MixinCocoDraw, MixinCocoAccessors,
                  MixinCocoExtras, MixinCocoIndex, MixinCocoDepricate,
                  ub.NiceRepr):
    index: Incomplete
    hashid: Incomplete
    hashid_parts: Incomplete
    tag: Incomplete
    dataset: Incomplete
    data_fpath: Incomplete
    bundle_dpath: Incomplete
    cache_dpath: Incomplete
    assets_dpath: Incomplete

    def __init__(self,
                 data: Union[str, dict] = None,
                 tag: str = None,
                 bundle_dpath: Union[str, None] = None,
                 img_root: Union[str, None] = None,
                 fname: Incomplete | None = ...,
                 autobuild: bool = ...) -> None:
        ...

    @property
    def fpath(self):
        ...

    @fpath.setter
    def fpath(self, value) -> None:
        ...

    @classmethod
    def from_data(CocoDataset,
                  data,
                  bundle_dpath: Incomplete | None = ...,
                  img_root: Incomplete | None = ...):
        ...

    @classmethod
    def from_image_paths(CocoDataset,
                         gpaths: List[str],
                         bundle_dpath: Incomplete | None = ...,
                         img_root: Incomplete | None = ...):
        ...

    @classmethod
    def from_coco_paths(CocoDataset,
                        fpaths: List[str],
                        max_workers: int = 0,
                        verbose: int = 1,
                        mode: str = 'thread',
                        union: Union[str, bool] = 'try'):
        ...

    def copy(self):
        ...

    def __nice__(self):
        ...

    def dumps(self, indent: Incomplete | None = ..., newlines: bool = False):
        ...

    def dump(self,
             file: Union[PathLike, IO],
             indent: Incomplete | None = ...,
             newlines: bool = False,
             temp_file: Union[bool, str] = True) -> None:
        ...

    def union(*others,
              disjoint_tracks: bool = True,
              **kwargs) -> kwcoco.CocoDataset:
        ...

    def subset(self,
               gids: List[int],
               copy: bool = False,
               autobuild: bool = True):
        ...

    def view_sql(self, force_rewrite: bool = False, memory: bool = False):
        ...


def demo_coco_data():
    ...
