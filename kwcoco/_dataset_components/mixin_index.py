from __future__ import annotations
from functools import partial
from collections import defaultdict
from kwcoco import exceptions
from typing import Any
from typing import MutableMapping, Protocol
import numbers
import ubelt as ub
import warnings

from kwcoco._helpers import (
    SortedSet,
    _lut_image_frame_index,
    _lut_annot_frame_index,
)


# TODO: better generic reusable protocol
class _CocoIndexParent(Protocol):
    dataset: MutableMapping[str, Any]


class CocoIndex:
    """
    Fast lookup index for the COCO dataset with dynamic modification

    Attributes:
        imgs (Dict[int, dict]):
            mapping between image ids and the image dictionaries

        anns (Dict[int, dict]):
            mapping between annotation ids and the annotation dictionaries

        cats (Dict[int, dict]):
            mapping between category ids and the category dictionaries

        tracks (Dict[int, dict]):
            mapping between track ids and the track dictionaries

        kpcats (Dict[int, dict]):
            mapping between keypoint category ids and keypoint category
            dictionaries

        gid_to_aids (Dict[int, List[int]]):
            mapping between an image-id and annotation-ids that belong to it

        cid_to_aids (Dict[int, List[int]]):
            mapping between an category-id and annotation-ids that belong to it

        cid_to_gids (Dict[int, List[int]]):
            mapping between an category-id and image-ids that contain
            at least one annotation with this category id.

        trackid_to_aids (Dict[int, List[int]]):
            mapping between a track-id and annotation-ids that belong to it

        vidid_to_gids (Dict[int, List[int]]):
            mapping between an video-id and images-ids that belong to it

        name_to_video (Dict[str, dict]):
            mapping between a video name and the video dictionary.

        name_to_cat (Dict[str, dict]):
            mapping between a category name and the category dictionary.

        name_to_img (Dict[str, dict]):
            mapping between a image name and the image dictionary.

        name_to_track (Dict[str, dict]):
            mapping between a track name and the track dictionary.

        file_name_to_img (Dict[str, dict]):
            mapping between a image file_name and the image dictionary.
    """

    # _set = ub.oset  # many operations are much slower for oset
    _set = set

    def _images_set_sorted_by_frame_index(index, gids: Any = None) -> Any:
        """
        Helper for ensuring that vidid_to_gids returns image ids ordered by
        frame index.
        """
        return SortedSet(gids, key=partial(_lut_image_frame_index, index.imgs))

    # Backwards compat
    _set_sorted_by_frame_index = _images_set_sorted_by_frame_index

    def _annots_set_sorted_by_frame_index(index, aids: Any = None) -> Any:
        """
        Helper for ensuring that vidid_to_gids returns image ids ordered by
        frame index.
        """
        # if aids is None:
        #     return set()
        # return set(aids)
        return SortedSet(
            aids, key=partial(_lut_annot_frame_index, index.imgs, index.anns)
        )

    def __init__(index) -> None:
        index.anns = None
        index.imgs = None
        index.videos = None
        index.cats = None
        index.tracks = None
        index.kpcats = None
        index._id_lookup = None

        index.gid_to_aids = None
        index.cid_to_aids = None
        index.vidid_to_gids = None
        index.trackid_to_aids = None

        index.name_to_video = None
        index.name_to_cat = None
        index.name_to_img = None
        index.name_to_track = None
        index.file_name_to_img = None
        index._CHECKS = True
        # index.kpcid_to_aids = None  # TODO

    def __bool__(index) -> bool:
        return index.anns is not None

    # On-demand lookup tables
    @property
    def cid_to_gids(index) -> Any:
        """
        Example:
            >>> import kwcoco
            >>> self = dset = kwcoco.CocoDataset()
            >>> self.index.cid_to_gids
        """
        from scriptconfig.dict_like import DictLike

        class ProxyCidToGids(DictLike):
            def __init__(self, index):
                self.index = index

            def getitem(self, cid: Any) -> Any:  # type: ignore
                aids = self.index.cid_to_aids[cid]
                gids = {self.index.anns[aid]['image_id'] for aid in aids}
                return gids

            def keys(self) -> Any:
                return self.index.cid_to_aids.keys()

        cid_to_gids = ProxyCidToGids(index=index)
        return cid_to_gids

    def _add_video(index, vidid: int, video: dict) -> None:
        if index.videos is not None:
            name = video['name']
            if index._CHECKS:
                if name in index.name_to_video:  # type: ignore
                    raise exceptions.DuplicateAddError(
                        'video with name={} already exists'.format(name)
                    )
            index.videos[vidid] = video
            if vidid not in index.vidid_to_gids:  # type: ignore
                index.vidid_to_gids[vidid] = index._images_set_sorted_by_frame_index()  # type: ignore
            index.name_to_video[name] = video  # type: ignore

    def _add_image(index, gid: int, img: dict) -> None:
        """
        Example:
            >>> # Test adding image to video that doesnt exist
            >>> import kwcoco
            >>> self = dset = kwcoco.CocoDataset()
            >>> dset.add_image(file_name='frame1', video_id=1, frame_index=0)
            >>> dset.add_image(file_name='frame2', video_id=1, frame_index=0)
            >>> dset._check_integrity()
            >>> print('dset.index.vidid_to_gids = {!r}'.format(dset.index.vidid_to_gids))
            >>> assert len(dset.index.vidid_to_gids) == 1
            >>> dset.add_video(name='foo-vid', id=1)
            >>> assert len(dset.index.vidid_to_gids) == 1
            >>> dset._check_integrity()
        """
        if index.imgs is not None:
            file_name = img.get('file_name', None)
            name = img.get('name', None)
            if index._CHECKS:
                if file_name is None and name is None:
                    raise exceptions.InvalidAddError(
                        'at least one of file_name or name must be specified'
                    )
                if file_name in index.file_name_to_img:  # type: ignore
                    raise exceptions.DuplicateAddError(
                        'image with file_name={} already exists'.format(file_name)
                    )
                if name in index.name_to_img:  # type: ignore
                    raise exceptions.DuplicateAddError(
                        'image with name={} already exists'.format(name)
                    )
            index.imgs[gid] = img
            index.gid_to_aids[gid] = index._set()  # type: ignore

            if file_name is not None:
                index.file_name_to_img[file_name] = img  # type: ignore

            if name is not None:
                index.name_to_img[name] = img  # type: ignore

            if 'video_id' in img:
                if img.get('frame_index', None) is None:
                    raise exceptions.InvalidAddError(
                        'Images with video-ids must have a frame_index'
                    )
                vidid = img['video_id']
                try:
                    index.vidid_to_gids[vidid].add(gid)  # type: ignore
                except KeyError:
                    # Should warning messages contain data-specific info?
                    # msg = ('Adding image-id={} to '
                    #        'non-existing video-id={}').format(gid, vidid)
                    msg = 'Adding image to non-existing video'
                    warnings.warn(msg)
                    index.vidid_to_gids[vidid] = (  # type: ignore
                        index._images_set_sorted_by_frame_index()
                    )
                    index.vidid_to_gids[vidid].add(gid)  # type: ignore

    def _add_images(index, imgs: list[dict]) -> None:
        """
        See ../dev/bench/bench_add_image_check.py

        Note:
            THIS FUNCTION WAS DESIGNED FOR SPEED, AS SUCH IT DOES NOT CHECK IF
            THE IMAGE-IDs or FILE_NAMES ARE DUPLICATED AND WILL BLINDLY ADD
            DATA EVEN IF IT IS BAD. THE SINGLE IMAGE VERSION IS SLOWER BUT
            SAFER.
        """
        if index.imgs is not None:
            gids = [img['id'] for img in imgs]
            new_imgs = dict(zip(gids, imgs))
            index.imgs.update(new_imgs)
            index.file_name_to_img.update(  # type: ignore
                {
                    img['file_name']: img
                    for img in imgs
                    if img.get('file_name', None) is not None
                }
            )
            index.name_to_img.update(  # type: ignore
                {img['name']: img for img in imgs if img.get('name', None) is not None}
            )
            for gid in gids:
                index.gid_to_aids[gid] = index._set()  # type: ignore

            if index.vidid_to_gids:
                vidid_to_gids = ub.group_items(
                    [g['id'] for g in imgs], [g.get('video_id', None) for g in imgs]
                )
                vidid_to_gids.pop(None, None)
                for vidid, gids in vidid_to_gids.items():
                    index.vidid_to_gids[vidid].update(gids)

    def _add_annotation(
        index, aid: int, gid: int, cid: int, tid: int, ann: dict
    ) -> None:
        if index.anns is not None:
            index.anns[aid] = ann
            # Note: it should be ok to have None's here
            index.gid_to_aids[gid].add(aid)  # type: ignore
            index.cid_to_aids[cid].add(aid)  # type: ignore
            # index.trackid_to_aids[tid].add(aid)
            try:
                index.trackid_to_aids[tid].add(aid)  # type: ignore
            except KeyError:
                if tid is not None:
                    msg = 'Adding annotation to non-existing track'
                    warnings.warn(msg)
                # Be careful to not apply sorting to trackless annotations
                if tid is None:
                    index.trackid_to_aids[tid] = set()
                else:
                    index.trackid_to_aids[tid] = (  # type: ignore
                        index._annots_set_sorted_by_frame_index()
                    )
                index.trackid_to_aids[tid].add(aid)  # type: ignore

    def _add_keypoint_categories(index, keypoint_cats: list[dict]) -> None:
        if index.kpcats is None:
            index.kpcats = {}
        ids = [obj['id'] for obj in keypoint_cats]
        new_objs = dict(zip(ids, keypoint_cats))
        index.kpcats.update(new_objs)

    def _add_categories(index, cats: list[dict]) -> None:
        if index.cats is not None:
            ids = [obj['id'] for obj in cats]
            names = [obj['name'] for obj in cats]
            new_objs = dict(zip(ids, cats))
            index.cats.update(new_objs)
            for cid, cat, name in zip(ids, cats, names):
                index.cid_to_aids[cid] = index._set()  # type: ignore
                index.name_to_cat[name] = cat  # type: ignore

    def _add_annotations(index, anns: list[dict]) -> None:
        if index.anns is not None:
            aids = [ann['id'] for ann in anns]
            gids = [ann['image_id'] for ann in anns]
            cids = [ann['category_id'] for ann in anns]
            tids = [ann.get('track_id', None) for ann in anns]
            new_anns = dict(zip(aids, anns))
            index.anns.update(new_anns)
            for gid, cid, tid, aid in zip(gids, cids, tids, aids):
                index.gid_to_aids[gid].add(aid)  # type: ignore
                index.cid_to_aids[cid].add(aid)  # type: ignore
                try:
                    index.trackid_to_aids[tid].add(aid)  # type: ignore
                except KeyError:
                    if tid is None:
                        index.trackid_to_aids[tid] = set()  # type: ignore
                    else:
                        index.trackid_to_aids[tid] = (  # type: ignore
                            index._annots_set_sorted_by_frame_index()
                        )
                    index.trackid_to_aids[tid].add(aid)  # type: ignore

    def _add_category(index, cid: int, name: str, cat: dict) -> None:
        if index.cats is not None:
            index.cats[cid] = cat
            index.cid_to_aids[cid] = index._set()  # type: ignore
            index.name_to_cat[name] = cat  # type: ignore

    def _add_track(index, trackid: int, name: str, track: dict) -> None:
        if index.tracks is not None:
            index.tracks[trackid] = track
            index.trackid_to_aids[trackid] = index._annots_set_sorted_by_frame_index()  # type: ignore
            index.name_to_track[name] = track  # type: ignore

    def _remove_all_annotations(index) -> None:
        # Keep the category and image indexes alive
        if index.anns is not None:
            for _ in index.gid_to_aids.values():  # type: ignore
                _.clear()
            for _ in index.cid_to_aids.values():  # type: ignore
                _.clear()
            # for _ in index.trackid_to_aids.values():
            #     _.clear()
            index.anns.clear()
            index.tracks.clear()  # type: ignore
            index.trackid_to_aids.clear()  # type: ignore
            index.name_to_track.clear()  # type: ignore

    def _remove_all_images(index) -> None:
        # Keep the category indexes alive
        if index.imgs is not None:
            index.imgs.clear()
            index.anns.clear()  # type: ignore
            index.gid_to_aids.clear()  # type: ignore
            index.file_name_to_img.clear()  # type: ignore
            for _ in index.cid_to_aids.values():  # type: ignore
                _.clear()
            for _ in index.vidid_to_gids.values():  # type: ignore
                _.clear()

    def _remove_keypoint_categories(index, remove_kpcid: Any, verbose: Any = 0) -> None:
        if index.kpcats is not None:
            for kpcid in remove_kpcid:
                del index.kpcats[kpcid]
                # kpcat = index.kpcats.pop(kpcid)
            if verbose > 2:
                print('Updated category index')

    def _remove_annotations(index, remove_aids: Any, verbose: Any = 0) -> None:
        if index.anns is not None:
            if verbose > 1:
                print('Updating annotation index')
            # This is faster for simple set cid_to_aids
            for aid in remove_aids:
                ann = index.anns[aid]
                gid = ann['image_id']
                cid = ann.get('category_id', None)
                track_id = ann.get('track_id', None)
                index.trackid_to_aids[track_id].remove(aid)  # type: ignore
                index.cid_to_aids[cid].remove(aid)  # type: ignore
                index.gid_to_aids[gid].remove(aid)  # type: ignore
                index.anns.pop(aid)

    def _remove_tracks(index, remove_trackids: Any, verbose: Any = 0) -> None:
        # dynamically update the track index
        if index.tracks is not None:
            for tid in remove_trackids:
                track = index.tracks.pop(tid)
                del index.trackid_to_aids[tid]  # type: ignore
                del index.name_to_track[track['name']]  # type: ignore
            if verbose > 2:
                print('Updated track index')

    def _remove_categories(index, remove_cids: Any, verbose: Any = 0) -> None:
        # dynamically update the category index
        if index.cats is not None:
            for cid in remove_cids:
                cat = index.cats.pop(cid)
                del index.cid_to_aids[cid]  # type: ignore
                del index.name_to_cat[cat['name']]  # type: ignore
            if verbose > 2:
                print('Updated category index')

    def _remove_images(index, remove_gids: Any, verbose: Any = 0) -> None:
        # dynamically update the image index
        if index.imgs is not None:
            for gid in remove_gids:
                img = index.imgs[gid]
                vidid = img.get('video_id', None)
                if vidid in index.vidid_to_gids:  # type: ignore
                    index.vidid_to_gids[vidid].remove(gid)  # type: ignore
                del index.gid_to_aids[gid]  # type: ignore
                gname = img.get('file_name', None)
                if gname is not None:
                    del index.file_name_to_img[gname]  # type: ignore
                name = img.get('name', None)
                if name is not None:
                    del index.name_to_img[name]  # type: ignore
                del index.imgs[gid]
            if verbose > 2:
                print('Updated image index')

    def _remove_videos(index, remove_vidids: Any, verbose: Any = 0) -> None:
        # dynamically update the video index
        lut = index.videos
        if lut is not None:
            for item_id in remove_vidids:
                item = lut.pop(item_id)
                del index.vidid_to_gids[item_id]  # type: ignore
                if index.name_to_video is not None:
                    del index.name_to_video[item['name']]
            if verbose > 2:
                print('Updated video index')

    def clear(index) -> None:
        index.anns = None
        index.imgs = None
        index.videos = None
        index.cats = None
        index.tracks = None
        index.kpcats = None
        index._id_lookup = None
        index.gid_to_aids = None
        index.vidid_to_gids = None
        index.cid_to_aids = None
        index.name_to_cat = None
        index.file_name_to_img = None
        index.name_to_video = None
        index.trackid_to_aids = None
        index.name_to_track = None
        # index.kpcid_to_aids = None  # TODO?

    def build(index, parent: _CocoIndexParent) -> None:
        """
        Build all id-to-obj reverse indexes from scratch.

        Args:
            parent (kwcoco.CocoDataset): the dataset to index

        Notation:
            aid - Annotation ID
            gid - imaGe ID
            cid - Category ID
            vidid - Video ID
            tid - Track ID

        Example:
            >>> import kwcoco
            >>> parent = kwcoco.CocoDataset.demo('vidshapes1', num_frames=4, rng=1)
            >>> index = parent.index
            >>> index.build(parent)
        """
        # create index
        anns, cats, imgs = {}, {}, {}
        videos = {}
        tracks = {}
        kpcats = {}

        # Build one-to-one index-lookup maps
        for cat in parent.dataset.get('categories', []):
            cid = cat['id']
            if cid in cat:
                warnings.warn(
                    'Categories have the same id in {}:\n{} and\n{}'.format(
                        parent, cats[cid], cat
                    )
                )
            cats[cid] = cat

        for obj in parent.dataset.get('keypoint_categories', []):
            cid = obj['id']
            if cid in obj:
                warnings.warn(
                    'Keypoint objegories have the same id in {}:\n{} and\n{}'.format(
                        parent, kpcats[cid], obj
                    )
                )
            kpcats[cid] = obj

        for video in parent.dataset.get('videos', []):
            vidid = video['id']
            if vidid in videos:
                warnings.warn(
                    'Video has the same id in {}:\n{} and\n{}'.format(
                        parent, videos[vidid], video
                    )
                )
            videos[vidid] = video

        for img in parent.dataset.get('images', []):
            gid = img['id']
            if gid in imgs:
                warnings.warn(
                    'Images have the same id in {}:\n{} and\n{}'.format(
                        parent, imgs[gid], img
                    )
                )
            imgs[gid] = img

        for ann in parent.dataset.get('annotations', []):
            aid = ann['id']
            if aid in anns:
                warnings.warn(
                    'Annotations at index {} and {} '
                    'have the same id in {}:\n{} and\n{}'.format(
                        parent.dataset['annotations'].index(anns[aid]),
                        parent.dataset['annotations'].index(ann),
                        parent,
                        anns[aid],
                        ann,
                    )
                )
            anns[aid] = ann

        for track in parent.dataset.get('tracks', []):
            tid = track['id']
            if tid in track:
                warnings.warn(
                    'Tracks have the same id in {}:\n{} and\n{}'.format(
                        parent, tracks[tid], track
                    )
                )
            tracks[tid] = track

        # Build one-to-many lookup maps
        vidid_to_gids = ub.group_items(
            [g['id'] for g in imgs.values()],
            [g.get('video_id', None) for g in imgs.values()],
        )
        vidid_to_gids.pop(None, None)

        if 0:
            # The following is slightly slower, but it is also many fewer lines
            # Not sure if its correct to replace the else block or not
            aids = [d['id'] for d in anns.values()]
            gid_to_aids = ub.group_items(aids, (d['image_id'] for d in anns.values()))
            cid_to_aids = ub.group_items(
                aids, (d.get('category_id', None) for d in anns.values())
            )
            trackid_to_aids = ub.group_items(
                aids, (d.get('track_id', None) for d in anns.values())
            )
            cid_to_aids.pop(None, None)
            gid_to_aids = ub.map_vals(index._set, gid_to_aids)
            cid_to_aids = ub.map_vals(index._set, cid_to_aids)
            trackid_to_aids = ub.map_vals(index._set, trackid_to_aids)
            vidid_to_gids = ub.map_vals(
                index._images_set_sorted_by_frame_index, vidid_to_gids
            )
        else:
            gid_to_aids = defaultdict(index._set)
            cid_to_aids = defaultdict(index._set)
            trackid_to_aids = defaultdict(index._set)
            for ann in anns.values():
                try:
                    aid = ann['id']
                    gid = ann['image_id']
                except KeyError:
                    raise KeyError('Annotation does not have ids {}'.format(ann))

                if not isinstance(aid, numbers.Integral):
                    raise TypeError('bad aid={} type={}'.format(aid, type(aid)))
                if not isinstance(gid, numbers.Integral):
                    raise TypeError('bad gid={} type={}'.format(gid, type(gid)))

                gid_to_aids[gid].add(aid)
                if gid not in imgs:
                    warnings.warn(
                        'Annotation {} in {} references unknown image_id'.format(
                            ann, parent
                        )
                    )

                try:
                    cid = ann['category_id']
                except KeyError:
                    warnings.warn(
                        'Annotation {} in {} is missing a category_id'.format(
                            ann, parent
                        )
                    )
                else:
                    cid_to_aids[cid].add(aid)

                    if not isinstance(cid, numbers.Integral) and cid is not None:
                        raise TypeError('bad cid={} type={}'.format(cid, type(cid)))

                    if cid not in cats and cid is not None:
                        warnings.warn(
                            'Annotation {} in {} references unknown category_id'.format(
                                ann, parent
                            )
                        )

                trackid = ann.get('track_id', None)
                trackid_to_aids[trackid].add(aid)

        # Fix one-to-zero cases
        for cid in cats.keys():
            if cid not in cid_to_aids:
                cid_to_aids[cid] = index._set()

        for gid in imgs.keys():
            if gid not in gid_to_aids:
                gid_to_aids[gid] = index._set()

        for vidid in videos.keys():
            if vidid not in vidid_to_gids:
                vidid_to_gids[vidid] = index._set()  # type: ignore

        # create class members
        index._id_lookup = {
            'categories': cats,
            'images': imgs,
            'annotations': anns,
            'videos': videos,
            'tracks': tracks,
            'keypoint_categories': kpcats,
        }
        index.anns = anns
        index.imgs = imgs
        index.cats = cats
        index.kpcats = kpcats
        index.videos = videos
        index.tracks = tracks

        # Remove defaultdict like behavior
        gid_to_aids.default_factory = None

        # Actually, its important to have defaultdict like behavior for
        # categories so we can allow for the category_id=None case
        # cid_to_aids.default_factory = None
        # vidid_to_gids.default_factory = None

        # Ensure that the values are cast to the appropriate set type
        # This needs to happen after index.imgs is populated
        vidid_to_gids = ub.map_vals(
            index._images_set_sorted_by_frame_index, vidid_to_gids
        )

        # Be careful to not apply sorting to trackless annotations
        _notrack_aids = trackid_to_aids.pop(None, None)
        trackid_to_aids = ub.map_vals(
            index._annots_set_sorted_by_frame_index, trackid_to_aids
        )
        if _notrack_aids is not None:
            trackid_to_aids[None] = set(_notrack_aids)

        index.gid_to_aids = gid_to_aids
        index.cid_to_aids = cid_to_aids
        index.vidid_to_gids = vidid_to_gids
        index.trackid_to_aids = trackid_to_aids

        index.name_to_cat = {cat['name']: cat for cat in index.cats.values()}
        index.file_name_to_img = {
            img['file_name']: img
            for img in index.imgs.values()
            if img.get('file_name', None) is not None
        }
        index.name_to_img = {
            img['name']: img
            for img in index.imgs.values()
            if img.get('name', None) is not None
        }
        index.name_to_video = {
            video['name']: video
            for video in index.videos.values()
            if video.get('name', None) is not None
        }
        index.name_to_track = {
            track['name']: track
            for track in index.tracks.values()
            if track.get('name', None) is not None
        }


class MixinCocoIndex:
    """
    Give the dataset top level access to index attributes
    """

    @property
    def anns(self) -> dict[int, dict]:
        return self.index.anns  # type: ignore

    @property
    def imgs(self) -> dict[int, dict]:
        return self.index.imgs  # type: ignore

    @property
    def cats(self) -> dict[int, dict]:
        return self.index.cats  # type: ignore

    # NOTE: API Issue, overloads previous method
    # @property
    # def videos(self):
    #     return self.index.videos

    @property
    def gid_to_aids(self) -> dict[int, list[int]]:
        return self.index.gid_to_aids  # type: ignore

    @property
    def cid_to_aids(self) -> dict[int, list[int]]:
        return self.index.cid_to_aids  # type: ignore

    @property
    def name_to_cat(self) -> dict[str, dict]:
        return self.index.name_to_cat  # type: ignore
