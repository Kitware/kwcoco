from typing import Tuple
from numpy.random import RandomState
from os import PathLike
from numpy import ndarray
from typing import List
import kwcoco
from typing import Dict
from _typeshed import Incomplete

TOYDATA_VIDEO_VERSION: int


def random_video_dset(num_videos: int = 1,
                      num_frames: int = 2,
                      num_tracks: int = 2,
                      anchors: Incomplete | None = ...,
                      image_size: Tuple[int, int] = ...,
                      verbose: int = 3,
                      render: bool | dict = False,
                      aux: bool | None = None,
                      multispectral: bool = False,
                      multisensor: bool = ...,
                      rng: int | None | RandomState = None,
                      dpath: str | PathLike | None = None,
                      max_speed: float = 0.01,
                      channels: str | None = None,
                      background: str = ...,
                      **kwargs):
    ...


def random_single_video_dset(image_size: Tuple[int, int] = ...,
                             num_frames: int = 5,
                             num_tracks: int = 3,
                             tid_start: int = 1,
                             gid_start: int = 1,
                             video_id: int = 1,
                             anchors: ndarray | None = None,
                             rng: RandomState | None | int = None,
                             render: bool | dict = False,
                             dpath: Incomplete | None = ...,
                             autobuild: bool = True,
                             verbose: int = 3,
                             aux: bool | None | List[str] = None,
                             multispectral: bool = False,
                             max_speed: float = 0.01,
                             channels: str | None | kwcoco.ChannelSpec = None,
                             multisensor: bool = False,
                             **kwargs):
    ...


def render_toy_dataset(dset: kwcoco.CocoDataset,
                       rng: int | None | RandomState,
                       dpath: str | PathLike | None = None,
                       renderkw: dict | None = None,
                       verbose: int = ...):
    ...


def render_toy_image(dset: kwcoco.CocoDataset,
                     gid: int,
                     rng: int | None | RandomState = None,
                     renderkw: dict | None = None) -> Dict:
    ...


def render_foreground(imdata, chan_to_auxinfo, dset, annots, catpats,
                      with_sseg, with_kpts, dims, newstyle, gray, rng):
    ...


def render_background(img,
                      rng,
                      gray,
                      bg_intensity,
                      bg_scale,
                      imgspace_background: Incomplete | None = ...):
    ...


def false_color(twochan):
    ...


def random_multi_object_path(num_objects,
                             num_frames,
                             rng: Incomplete | None = ...,
                             max_speed: float = ...):
    ...


def random_path(num: int,
                degree: int = 1,
                dimension: int = 2,
                rng: RandomState | None | int = None,
                mode: str = 'boid'):
    ...
