from __future__ import annotations

from typing import Any, TYPE_CHECKING
import itertools as it
import os
import ubelt as ub

from kwcoco import exceptions
from kwcoco._helpers import _delitems

if TYPE_CHECKING:
    from delayed_image.channel_spec import FusedChannelSpec
    import kwimage


class MixinCocoAddRemove:
    """
    Mixin functions to dynamically add / remove annotations images and
    categories while maintaining lookup indexes.
    """

    def add_video(self, name: str, id: None | int = None, **kw: Any) -> int:
        """
        Register a new video with the dataset

        Args:
            name (str): Unique name for this video.
            id (None | int): ADVANCED. Force using this image id.
            coco_url (str): COCO-style remote URI for the video resource
            **kw : stores arbitrary key/value pairs in this new video

        Returns:
            int : the video id assigned to the new video

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset()
            >>> print('self.index.videos = {}'.format(ub.urepr(self.index.videos, nl=1)))
            >>> print('self.index.imgs = {}'.format(ub.urepr(self.index.imgs, nl=1)))
            >>> print('self.index.vidid_to_gids = {!r}'.format(self.index.vidid_to_gids))

            >>> vidid1 = self.add_video('foo', id=3)
            >>> vidid2 = self.add_video('bar')
            >>> vidid3 = self.add_video('baz')
            >>> print('self.index.videos = {}'.format(ub.urepr(self.index.videos, nl=1)))
            >>> print('self.index.imgs = {}'.format(ub.urepr(self.index.imgs, nl=1)))
            >>> print('self.index.vidid_to_gids = {!r}'.format(self.index.vidid_to_gids))

            >>> gid1 = self.add_image('foo1.jpg', video_id=vidid1, frame_index=0)
            >>> gid2 = self.add_image('foo2.jpg', video_id=vidid1, frame_index=1)
            >>> gid3 = self.add_image('foo3.jpg', video_id=vidid1, frame_index=2)
            >>> gid4 = self.add_image('bar1.jpg', video_id=vidid2, frame_index=0)
            >>> print('self.index.videos = {}'.format(ub.urepr(self.index.videos, nl=1)))
            >>> print('self.index.imgs = {}'.format(ub.urepr(self.index.imgs, nl=1)))
            >>> print('self.index.vidid_to_gids = {!r}'.format(self.index.vidid_to_gids))

            >>> self.remove_images([gid2])
            >>> print('self.index.vidid_to_gids = {!r}'.format(self.index.vidid_to_gids))
        """
        index = self.index  # type: ignore
        if index.videos is not None and name in index.name_to_video:
            raise exceptions.DuplicateAddError(f'Video name={name} already exists')
        if id is not None and index.videos is not None and id in index.videos:
            raise exceptions.DuplicateAddError(f'Video id={id} already exists')
        if id is None:
            id = self._next_ids.get('videos')  # type: ignore
        video = ub.odict()
        video['id'] = id
        video['name'] = name
        video.update(**kw)
        try:
            self.dataset['videos'].append(video)  # type: ignore
        except KeyError:
            self.dataset['videos'] = []  # type: ignore
            self.dataset['videos'].append(video)  # type: ignore
        self.index._add_video(id, video)  # type: ignore
        # self._invalidate_hashid(['videos'])
        return id

    def add_image(
        self, file_name: str | None = None, id: None | int = None, **kw: Any
    ) -> int:
        """
        Register a new image with the dataset

        Args:
            file_name (str | None): relative or absolute path to image.
                 if not given, then "name" must be specified and we will
                 expect that "auxiliary" assets are eventually added.
            id (None | int): ADVANCED. Force using this image id.
            name (str): a unique key to identify this image
            coco_url (str): standard MS-COCO remote URI for the main image data
            width (int): base width of the image
            height (int): base height of the image
            channels (ChannelSpec): specification of base channels.
                Only relevant if file_name is given.
            auxiliary (List[Dict]): specification of auxiliary assets.
                See :func:`CocoImage.add_asset` for details
            video_id (int): id of parent video, if applicable
            frame_index (int): frame index in parent video
            timestamp (number | str): timestamp of frame index
            warp_img_to_vid (Dict): this transform is used to align
                the image to a video if it belongs to one.
            **kw : stores arbitrary key/value pairs in this new image

        Returns:
            int : the image id assigned to the new image

        SeeAlso:
            :func:`add_image`
            :func:`add_images`
            :func:`ensure_image`

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> import kwimage
            >>> gname = kwimage.grab_test_image_fpath('paraview')
            >>> gid = self.add_image(gname)
            >>> assert self.imgs[gid]['file_name'] == gname
        """
        index = self.index  # type: ignore

        norm_file_name: Any
        try:
            norm_file_name = os.fspath(file_name)  # type: ignore
        except TypeError:
            norm_file_name = file_name

        if index.imgs is not None and norm_file_name in index.file_name_to_img:
            raise exceptions.DuplicateAddError(
                'Image with file_name={} already exists'.format(norm_file_name)
            )

        # Perform all checks that do not depend on the chosen id before
        # advancing the automatic id generator.
        name = kw.get('name', None)
        if index.imgs is not None and index._CHECKS:
            if norm_file_name is None and name is None:
                raise exceptions.InvalidAddError(
                    'at least one of file_name or name must be specified'
                )
            if name in index.name_to_img:  # type: ignore
                raise exceptions.DuplicateAddError(
                    'Image with name={} already exists'.format(name)
                )
        if 'video_id' in kw and kw.get('frame_index', None) is None:
            raise exceptions.InvalidAddError(
                'Images with video-ids must have a frame_index'
            )

        if id is not None and self.imgs and id in self.imgs:  # type: ignore
            raise exceptions.DuplicateAddError('Image id={} already exists'.format(id))
        if id is None:
            id = self._next_ids.get('images')  # type: ignore

        img: dict[str, Any] = {}
        img['id'] = int(id)
        img['file_name'] = norm_file_name
        img.update(**kw)

        # Validation above completes before automatic id selection. Insert
        # into the index before the raw append so both views change together.
        self.index._add_image(id, img)  # type: ignore
        self.dataset['images'].append(img)  # type: ignore
        self._invalidate_hashid()  # type: ignore
        return id

    def add_asset(
        self,
        gid: int,
        file_name: str | None = None,
        channels: str | FusedChannelSpec | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Adds an auxiliary / asset item to the image dictionary.

        Args:
            gid (int):
                The image id to add the auxiliary/asset item to.

            file_name (str | None):
                The name of the file relative to the bundle directory. If
                unspecified, imdata must be given.

            channels (str | FusedChannelSpec):
                The channel code indicating what each of the bands represents.
                These channels should be disjoint wrt to the existing data in
                this image (this is not checked).

            **kwargs:
                See :func:`CocoImage.add_asset` for more details

        Example:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset()
            >>> gid = dset.add_image(name='my_image_name', width=200, height=200)
            >>> dset.add_asset(gid, 'path/fake_B0.tif', channels='B0', width=200,
            >>>                height=200, warp_aux_to_img={'scale': 1.0})
        """
        coco_img = self.coco_image(gid)  # type: ignore
        coco_img.add_asset(file_name=file_name, channels=channels, **kwargs)

    # TODO: deprecated add_auxiliary_item for add_asset
    add_auxiliary_item = add_asset

    def add_annotation(
        self,
        image_id: int,
        category_id: int | None = None,
        bbox: list | kwimage.Boxes = ub.NoParam,
        segmentation: dict | list | Any = ub.NoParam,
        keypoints: Any = ub.NoParam,
        id: None | int = None,
        track_id: int | str | None = None,
        **kw: Any,
    ) -> int:
        """
        Register a new annotation with the dataset

        Args:
            image_id (int): image_id the annotation is added to.

            category_id (int | None): category_id for the new annotation

            bbox (list | kwimage.Boxes): bounding box in xywh format

            segmentation (Dict | List | Any): keypoints in some
                accepted format, see :func:`kwimage.Mask.to_coco` and
                :func:`kwimage.MultiPolygon.to_coco`.
                Extended types: `MaskLike | MultiPolygonLike`.

            keypoints (Any): keypoints in some accepted
                format, see :func:`kwimage.Points.to_coco`.
                Extended types: ``PointsLike``.

            id (None | int): Force using this annotation id. Typically you
                should NOT specify this. A new unused id will be chosen and
                returned.

            track_id (int | str | None):
                Some value used to associate annotations that belong to the
                same "track". In the future we may remove support for strings.

            **kw : stores arbitrary key/value pairs in this new image,
                Common respected key/values include but are not limited to the
                following:
                score : float
                prob : List[float]
                weight (float): a weight, usually used to indicate if a ground
                truth annotation is difficult / important. This generalizes
                standard "is_hard" or "ignore" attributes in other formats.
                caption (str): a text caption for this annotation

        Returns:
            int : the annotation id assigned to the new annotation

        SeeAlso:
            :func:`kwcoco.coco_dataset.MixinCocoAddRemove.add_annotation`
            :func:`kwcoco.coco_dataset.MixinCocoAddRemove.add_annotations`

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> image_id = 1
            >>> cid = 1
            >>> bbox = [10, 10, 20, 20]
            >>> aid = self.add_annotation(image_id, cid, bbox)
            >>> assert self.anns[aid]['bbox'] == bbox

        Example:
            >>> import kwimage
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> new_det = kwimage.Detections.random(1, segmentations=True, keypoints=True)
            >>> # kwimage datastructures have methods to convert to coco recognized formats
            >>> new_ann_data = list(new_det.to_coco(style='new'))[0]
            >>> image_id = 1
            >>> aid = self.add_annotation(image_id, **new_ann_data)
            >>> # Lookup the annotation we just added
            >>> ann = self.index.anns[aid]
            >>> print('ann = {}'.format(ub.urepr(ann, nl=-2)))

        Example:
            >>> # Attempt to add annot without a category or bbox
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> image_id = 1
            >>> aid = self.add_annotation(image_id)
            >>> assert None in self.index.cid_to_aids

        Example:
            >>> # Attempt to add annot using various styles of kwimage structures
            >>> import kwcoco
            >>> import kwimage
            >>> self = kwcoco.CocoDataset.demo()
            >>> image_id = 1
            >>> #--
            >>> kw = {}
            >>> kw['segmentation'] = kwimage.Polygon.random()
            >>> kw['keypoints'] = kwimage.Points.random()
            >>> aid = self.add_annotation(image_id, **kw)
            >>> ann = self.index.anns[aid]
            >>> print('ann = {}'.format(ub.urepr(ann, nl=2)))
            >>> #--
            >>> kw = {}
            >>> kw['segmentation'] = kwimage.Mask.random()
            >>> aid = self.add_annotation(image_id, **kw)
            >>> ann = self.index.anns[aid]
            >>> assert ann.get('segmentation', None) is not None
            >>> print('ann = {}'.format(ub.urepr(ann, nl=2)))
            >>> #--
            >>> kw = {}
            >>> kw['segmentation'] = kwimage.Mask.random().to_array_rle()
            >>> aid = self.add_annotation(image_id, **kw)
            >>> ann = self.index.anns[aid]
            >>> assert ann.get('segmentation', None) is not None
            >>> print('ann = {}'.format(ub.urepr(ann, nl=2)))
            >>> #--
            >>> kw = {}
            >>> kw['segmentation'] = kwimage.Polygon.random().to_coco()
            >>> kw['keypoints'] = kwimage.Points.random().to_coco()
            >>> aid = self.add_annotation(image_id, **kw)
            >>> ann = self.index.anns[aid]
            >>> assert ann.get('segmentation', None) is not None
            >>> assert ann.get('keypoints', None) is not None
            >>> print('ann = {}'.format(ub.urepr(ann, nl=2)))
        """
        try:
            import kwimage  # type: ignore
        except ImportError:
            kwimage = None  # type: ignore

        # Preserve the historical KeyError for an unknown image, but raise it
        # before advancing the automatic id generator or mutating either view.
        if self.index.gid_to_aids is not None:  # type: ignore
            if image_id not in self.index.gid_to_aids:  # type: ignore
                raise KeyError(image_id)

        if id is None:
            id = self._next_ids.get('annotations')  # type: ignore
        elif self.anns and id in self.anns:  # type: ignore
            raise IndexError('Annot id={} already exists'.format(id))

        ann: dict[str, Any] = {}
        ann['id'] = int(id)
        ann['image_id'] = int(image_id)
        ann['category_id'] = None if category_id is None else int(category_id)

        if kwimage is not None and hasattr(bbox, 'to_coco'):
            # to_coco works different for boxes atm, might update in future
            try:
                ann['bbox'] = ub.peek(bbox.to_coco(style='new'))  # type: ignore
            except Exception:
                ann['bbox'] = bbox.to_xywh().data.tolist()  # type: ignore
        elif bbox is not ub.NoParam:
            ann['bbox'] = bbox
        else:
            assert bbox is ub.NoParam

        if kwimage is not None and hasattr(keypoints, 'to_coco'):
            ann['keypoints'] = keypoints.to_coco(style='new')
        elif keypoints is not ub.NoParam:
            ann['keypoints'] = keypoints
        else:
            assert keypoints is ub.NoParam

        if kwimage is not None and hasattr(segmentation, 'to_coco'):
            ann['segmentation'] = segmentation.to_coco(style='new')  # type: ignore
        elif segmentation is not ub.NoParam:
            ann['segmentation'] = segmentation
        else:
            assert segmentation is ub.NoParam

        if track_id is not None:
            ann['track_id'] = track_id

        ann.update(**kw)

        self.index._add_annotation(id, image_id, category_id, track_id, ann)  # type: ignore
        self.dataset['annotations'].append(ann)  # type: ignore
        self._invalidate_hashid(['annotations'])  # type: ignore
        return id

    def add_category(
        self,
        name: str,
        supercategory: str | None = None,
        id: int | None = None,
        **kw: Any,
    ) -> int:
        """
        Register a new category with the dataset

        Args:
            name (str): name of the new category
            supercategory (str | None): parent of this category
            id (int | None): use this category id, if it was not taken
            **kw : stores arbitrary key/value pairs in this new image

        Returns:
            int : the category id assigned to the new category

        SeeAlso:
            :func:`kwcoco.coco_dataset.MixinCocoAddRemove.add_category`
            :func:`kwcoco.coco_dataset.MixinCocoAddRemove.ensure_category`

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> prev_n_cats = self.n_cats
            >>> cid = self.add_category('dog', supercategory='object')
            >>> assert self.cats[cid]['name'] == 'dog'
            >>> assert self.n_cats == prev_n_cats + 1
            >>> import pytest
            >>> with pytest.raises(ValueError):
            >>>     self.add_category('dog', supercategory='object')
        """
        index = self.index  # type: ignore
        if index.cats and name in index.name_to_cat:
            raise exceptions.DuplicateAddError(
                'Category name={!r} already exists'.format(name)
            )

        if id is None:
            id = self._next_ids.get('categories')  # type: ignore
        elif index.cats and id in index.cats:
            raise exceptions.DuplicateAddError(
                'Category id={} already exists'.format(id)
            )

        cat: dict[str, Any] = {}
        cat['id'] = int(id)
        cat['name'] = str(name)
        if supercategory:
            cat['supercategory'] = supercategory
        cat.update(**kw)

        # Add to raw data structure
        self.dataset['categories'].append(cat)  # type: ignore

        # And add to the indexes
        index._add_category(id, name, cat)
        self._invalidate_hashid(['categories'])  # type: ignore
        return id

    def add_track(self, name: str = None, id: int | None = None, **kw: Any) -> int:  # type: ignore
        """
        Register a new track with the dataset

        Args:
            name (str | None): Name of the new track.
                If unspecified one is created based on a uuid.

            id (int | None): use this track id, if it was not taken

            **kw : stores arbitrary key/value pairs in this new image

        Returns:
            int : the track id assigned to the new track

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> prev_n_tracks = self.n_tracks
            >>> track_id = self.add_track('dog')
            >>> assert self.index.tracks[track_id]['name'] == 'dog'
            >>> assert self.n_tracks == prev_n_tracks + 1
            >>> import pytest
            >>> with pytest.raises(ValueError):
            >>>     self.add_track('dog')
        """
        index = self.index  # type: ignore

        if name is None:
            # auto choose name
            import uuid

            name = str(uuid.uuid4())
            while index.tracks and name in index.name_to_track:
                # This will never happen, but my brain is irrationally
                # insisting on writing the check anyway.
                name = str(uuid.uuid4())

        if index.tracks and name in index.name_to_track:
            raise exceptions.DuplicateAddError(
                'Track name={!r} already exists'.format(name)
            )

        if id is None:
            id = self._next_ids.get('tracks')  # type: ignore
        elif index.tracks and id in index.tracks:
            raise exceptions.DuplicateAddError('Track id={} already exists'.format(id))

        track: dict[str, Any] = {}
        track['id'] = int(id)
        track['name'] = str(name)
        track.update(**kw)

        # Add to raw data structure
        try:
            self.dataset['tracks'].append(track)  # type: ignore
        except KeyError:
            self.dataset['tracks'] = [track]  # type: ignore

        # And add to the indexes
        index._add_track(id, name, track)
        self._invalidate_hashid(['tracks'])  # type: ignore
        return id

    def ensure_video(self, name: Any, id: None | int = None, **kw: Any) -> int:
        """
        Register a video if it is new or returns an existing id.

        Like :func:`kwcoco.coco_dataset.MixinCocoAddRemove.add_video`, but
        returns the existing video id if it already exists instead of failing.
        In this case all metadata is ignored.

        Args:
            file_name (str): relative or absolute path to video
            id (None | int): ADVANCED. Force using this video id.
            **kw : stores arbitrary key/value pairs in this new video

        Returns:
            int: the existing or new video id

        SeeAlso:
            :func:`kwcoco.coco_dataset.MixinCocoAddRemove.add_video`
            :func:`kwcoco.coco_dataset.MixinCocoAddRemove.ensure_video`

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> id1 = self.ensure_video('video1')
            >>> id2 = self.ensure_video('video1')
            >>> assert id1 == id2
        """
        try:
            id = self.add_video(name=name, id=id, **kw)
        except exceptions.DuplicateAddError:
            obj = self.index.name_to_video[name]  # type: ignore
            id = obj['id']
        return id

    def ensure_track(self, name: Any, id: None | int = None, **kw: Any) -> int:
        """
        Register a track if it is new or returns an existing id.

        Like :func:`kwcoco.coco_dataset.MixinCocoAddRemove.add_track`, but
        returns the existing track id if it already exists instead of failing.
        In this case all metadata is ignored.

        Args:
            file_name (str): relative or absolute path to track
            id (None | int): ADVANCED. Force using this track id.
            **kw : stores arbitrary key/value pairs in this new track

        Returns:
            int: the existing or new track id

        SeeAlso:
            :func:`kwcoco.coco_dataset.MixinCocoAddRemove.add_track`
            :func:`kwcoco.coco_dataset.MixinCocoAddRemove.ensure_track`

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> track_id1 = self.ensure_track('dog')
            >>> track_id2 = self.ensure_track('dog')
            >>> assert track_id1 == track_id2
        """
        try:
            id = self.add_track(name=name, id=id, **kw)
        except exceptions.DuplicateAddError:
            obj = self.index.name_to_track[name]  # type: ignore
            id = obj['id']
        return id

    def ensure_image(self, file_name: str, id: None | int = None, **kw: Any) -> int:
        """
        Register an image if it is new or returns an existing id.

        Like :func:`kwcoco.coco_dataset.MixinCocoAddRemove.add_image`, but
        returns the existing image id if it already exists instead of failing.
        In this case all metadata is ignored.

        Args:
            file_name (str): relative or absolute path to image
            id (None | int): ADVANCED. Force using this image id.
            **kw : stores arbitrary key/value pairs in this new image

        Returns:
            int: the existing or new image id

        SeeAlso:
            :func:`kwcoco.coco_dataset.MixinCocoAddRemove.add_image`
            :func:`kwcoco.coco_dataset.MixinCocoAddRemove.add_images`
            :func:`kwcoco.coco_dataset.MixinCocoAddRemove.ensure_image`
        """
        try:
            id = self.add_image(file_name=file_name, id=id, **kw)
        except exceptions.DuplicateAddError:
            img = self.index.file_name_to_img[file_name]  # type: ignore
            id = img['id']
        return id

    def ensure_category(
        self,
        name: Any,
        supercategory: Any | None = None,
        id: Any | None = None,
        **kw: Any,
    ) -> int:
        """
        Register a category if it is new or returns an existing id.

        Like :func:`kwcoco.coco_dataset.MixinCocoAddRemove.add_category`, but
        returns the existing category id if it already exists instead of
        failing. In this case all metadata is ignored.

        Returns:
            int: the existing or new category id

        SeeAlso:
            :func:`kwcoco.coco_dataset.MixinCocoAddRemove.add_category`
            :func:`kwcoco.coco_dataset.MixinCocoAddRemove.ensure_category`
        """
        try:
            id = self.add_category(name=name, supercategory=supercategory, id=id, **kw)
        except exceptions.DuplicateAddError:
            cat = self.index.name_to_cat[name]  # type: ignore
            id = cat['id']
        return id

    def add_categories(self, cats: list[dict]) -> Any:
        """
        Faster less-safe multi-item alternative to add_category.

        We assume the items are well formatted in kwcoco compliant
        dictionaries, including the "id" field. No validation checks are made
        when calling this function, but the index is updated, and the hashid is
        invalidated.

        Args:
            cats (List[Dict]): list of category dictionaries

        SeeAlso:
            :func:`add_category`
            :func:`add_categories`

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> cats = [self.cats[cid] for cid in [2, 3, 5, 7]]
            >>> self.remove_categories(cats)
            >>> assert self.n_cats == 4 and self._check_integrity()
            >>> self.add_categories(cats)
            >>> assert self.n_cats == 8 and self._check_integrity()
        """
        self.dataset['categories'].extend(cats)  # type: ignore
        self.index._add_categories(cats)  # type: ignore
        self._invalidate_hashid(['categories'])  # type: ignore

    def add_keypoint_categories(self, keypoint_cats: Any) -> Any:
        """
        Faster less-safe multi-item alternative to add_category.

        We assume the items are well formatted in kwcoco compliant
        dictionaries, including the "id" field. No validation checks are made
        when calling this function, but the index is updated, and the hashid is
        invalidated.

        Args:
            cats (List[Dict]): list of category dictionaries

        SeeAlso:
            :func:`add_category`
            :func:`add_categories`

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> keypoint_cats = [{'id': 1, 'name': 'ear'}, {'id': 2, 'name': 'nose'}]
            >>> self.add_keypoint_categories(keypoint_cats)
            >>> kp_identifiers = keypoint_cats
            >>> self.remove_keypoint_categories(kp_identifiers, clean_anns=False)
        """
        if 'keypoint_categories' not in self.dataset:  # type: ignore
            self.dataset['keypoint_categories'] = []  # type: ignore
        self.dataset['keypoint_categories'].extend(keypoint_cats)  # type: ignore
        self.index._add_keypoint_categories(keypoint_cats)  # type: ignore
        self._invalidate_hashid(['keypoint_categories'])  # type: ignore

    def add_annotations(self, anns: list[dict]) -> None:
        """
        Faster less-safe multi-item alternative to add_annotation.

        We assume the annotations are well formatted in kwcoco compliant
        dictionaries, including the "id" field. No validation checks are
        made when calling this function.

        Args:
            anns (List[Dict]): list of annotation dictionaries

        SeeAlso:
            :func:`add_annotation`
            :func:`add_annotations`

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> anns = [self.anns[aid] for aid in [2, 3, 5, 7]]
            >>> self.remove_annotations(anns)
            >>> assert self.n_annots == 7 and self._check_integrity()
            >>> self.add_annotations(anns)
            >>> assert self.n_annots == 11 and self._check_integrity()
        """
        self.dataset['annotations'].extend(anns)  # type: ignore
        self.index._add_annotations(anns)  # type: ignore
        self._invalidate_hashid(['annotations'])  # type: ignore

    def add_images(self, imgs: list[dict]) -> None:
        """
        Faster less-safe multi-item alternative

        We assume the images are well formatted in kwcoco compliant
        dictionaries, including the "id" field. No validation checks are
        made when calling this function.

        Note:
            THIS FUNCTION WAS DESIGNED FOR SPEED, AS SUCH IT DOES NOT CHECK IF
            THE IMAGE-IDs or FILE_NAMES ARE DUPLICATED AND WILL BLINDLY ADD
            DATA EVEN IF IT IS BAD. THE SINGLE IMAGE VERSION IS SLOWER BUT
            SAFER.

        Args:
            imgs (List[Dict]): list of image dictionaries

        SeeAlso:
            :func:`kwcoco.coco_dataset.MixinCocoAddRemove.add_image`
            :func:`kwcoco.coco_dataset.MixinCocoAddRemove.add_images`
            :func:`kwcoco.coco_dataset.MixinCocoAddRemove.ensure_image`

        Example:
            >>> import kwcoco
            >>> imgs = kwcoco.CocoDataset.demo().dataset['images']
            >>> self = kwcoco.CocoDataset()
            >>> self.add_images(imgs)
            >>> assert self.n_images == 3 and self._check_integrity()
        """
        self.dataset['images'].extend(imgs)  # type: ignore
        self.index._add_images(imgs)  # type: ignore
        self._invalidate_hashid(['images'])  # type: ignore

    def clear_images(self) -> None:
        """
        Removes all images and annotations (but not categories)

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> self.clear_images()
            >>> print(ub.urepr(self.basic_stats(), nobr=1, nl=0, si=1))
            n_anns: 0, n_imgs: 0, n_videos: 0, n_cats: 8, n_tracks: 0
        """
        # self.dataset['images'].clear()
        # self.dataset['annotations'].clear()
        del self.dataset['images'][:]  # type: ignore
        del self.dataset['annotations'][:]  # type: ignore
        self.index._remove_all_images()  # type: ignore
        self._invalidate_hashid(['images', 'annotations'])  # type: ignore

    def clear_annotations(self) -> None:
        """
        Removes all annotations and tracks (but not images and categories)

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> self.clear_annotations()
            >>> print(ub.urepr(self.basic_stats(), nobr=1, nl=0, si=1))
            n_anns: 0, n_imgs: 3, n_videos: 0, n_cats: 8, n_tracks: 0
        """
        # self.dataset['annotations'].clear()
        self.dataset['annotations'].clear()  # type: ignore
        if 'tracks' in self.dataset:  # type: ignore
            self.dataset['tracks'].clear()  # type: ignore
        self.index._remove_all_annotations()  # type: ignore
        self._invalidate_hashid(['annotations', 'tracks'])  # type: ignore

    def remove_annotation(self, aid_or_ann: Any) -> None:
        """
        Remove a single annotation from the dataset

        If you have multiple annotations to remove its more efficient to remove
        them in batch with
        :func:`kwcoco.coco_dataset.MixinCocoAddRemove.remove_annotations`

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> aid = 2
            >>> self.remove_annotation(aid)
            >>> assert len(self.dataset['annotations']) == 10
            >>> self._check_integrity()
        """
        self.remove_annotations([aid_or_ann])

    def remove_annotations(
        self, aids_or_anns: Any, verbose: int = 0, safe: bool = True
    ) -> dict:
        """
        Remove multiple annotations from the dataset.

        Args:
            anns_or_aids (List): list of annotation dicts or ids

            safe (bool): if True, we perform checks to remove
                duplicates and non-existing identifiers. Defaults to True.

        Returns:
            Dict: num_removed: information on the number of items removed

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> prev_n_annots = self.n_annots
            >>> aids_or_anns = [self.anns[2], 3, 4, self.anns[1]]
            >>> self.remove_annotations(aids_or_anns)  # xdoc: +IGNORE_WANT
            {'annotations': 4}
            >>> assert len(self.dataset['annotations']) == prev_n_annots - 4
            >>> self._check_integrity()
        """
        remove_info: dict[str, Any] = {'annotations': None}
        # Do nothing if given no input
        if aids_or_anns:
            # build mapping from aid to index O(n)
            # TODO: it would be nice if this mapping was as part of the index.
            aid_to_index = {
                ann['id']: index
                for index, ann in enumerate(self.dataset['annotations'])  # type: ignore
            }
            remove_aids = list(map(self._resolve_to_id, aids_or_anns))  # type: ignore
            if safe:
                remove_aids = sorted(set(remove_aids))
            remove_info['annotations'] = len(remove_aids)

            # Lookup the indices to remove, sort in descending order
            if verbose > 1:
                print('Removing {} annotations'.format(len(remove_aids)))

            remove_idxs = list(ub.take(aid_to_index, remove_aids))

            self.index._remove_annotations(remove_aids, verbose=verbose)  # type: ignore

            _delitems(self.dataset['annotations'], remove_idxs)  # type: ignore

            self._invalidate_hashid(['annotations'])  # type: ignore
        return remove_info

    def remove_categories(
        self,
        cat_identifiers: list,
        keep_annots: bool = False,
        verbose: int = 0,
        safe: bool = True,
    ) -> dict:
        """
        Remove categories and all annotations in those categories.

        Currently does not change any hierarchy information

        Args:
            cat_identifiers (List): list of category dicts, names, or ids

            keep_annots (bool):
                if True, keeps annotations, but removes category labels.
                Defaults to False.

            safe (bool): if True, we perform checks to remove
                duplicates and non-existing identifiers. Defaults to True.

        Returns:
            Dict: num_removed: information on the number of items removed

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> cat_identifiers = [self.cats[1], 'rocket', 3]
            >>> self.remove_categories(cat_identifiers)
            >>> assert len(self.dataset['categories']) == 5
            >>> self._check_integrity()
        """
        remove_info: dict[str, Any] = {
            'annotations': None, 'categories': None
        }
        if cat_identifiers:
            if verbose > 1:
                print('Removing annots of removed categories')

            if safe:
                remove_cid_set: set[int] = set()
                for identifier in cat_identifiers:
                    try:
                        cid = self._resolve_to_cid(identifier)  # type: ignore
                        remove_cid_set.add(cid)
                    except Exception:
                        pass
                remove_cids = sorted(remove_cid_set)
            else:
                remove_cids = list(map(self._resolve_to_cid, cat_identifiers))  # type: ignore
            # First remove any annotation that belongs to those categories
            if self.index.cid_to_aids:  # type: ignore
                remove_aids = list(
                    it.chain(*[self.index.cid_to_aids[cid] for cid in remove_cids])  # type: ignore
                )
            else:
                remove_aids = [
                    ann['id']
                    for ann in self.dataset['annotations']  # type: ignore
                    if ann['category_id'] in remove_cids
                ]

            if keep_annots:
                # Simply remove category information instead of removing the
                # entire annotation.
                for aid in remove_aids:
                    self.anns[aid].pop('category_id')  # type: ignore
            else:
                rminfo = self.remove_annotations(remove_aids, verbose=verbose)
                remove_info.update(rminfo)

            remove_info['categories'] = len(remove_cids)
            if verbose > 1:
                print('Removing {} category entries'.format(len(remove_cids)))
            id_to_index = {
                cat['id']: index for index, cat in enumerate(self.dataset['categories'])  # type: ignore
            }
            # Lookup the indices to remove, sort in descending order
            remove_idxs = list(ub.take(id_to_index, remove_cids))
            _delitems(self.dataset['categories'], remove_idxs)  # type: ignore

            self.index._remove_categories(remove_cids, verbose=verbose)  # type: ignore
            self._invalidate_hashid(['categories', 'annotations'])  # type: ignore

        return remove_info

    def remove_tracks(
        self,
        track_identifiers: list,
        keep_annots: bool = False,
        verbose: int = 0,
        safe: bool = True,
    ) -> dict:
        """
        Remove tracks and all annotations in those tracks.

        Args:
            track_identifiers (List): list of track dicts, names, or ids

            keep_annots (bool):
                if True, keeps annotations, but removes tracks labels.
                Defaults to False.

            safe (bool): if True, we perform checks to remove
                duplicates and non-existing identifiers. Defaults to True.

        Returns:
            Dict: num_removed: information on the number of items removed

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('vidshapes1')
            >>> for ann in self.dataset['annotations']:
            ...     ann.pop('segmentation')
            ...     ann.pop('keypoints')
            >>> print('self.dataset = {}'.format(ub.urepr(self.dataset, nl=2)))
            >>> track_identifiers = [2]
            >>> assert len(self.dataset['tracks']) == 2
            >>> self.remove_tracks(track_identifiers)
            >>> print('self.dataset = {}'.format(ub.urepr(self.dataset, nl=2)))
            >>> assert len(self.dataset['tracks']) == 1
            >>> self._check_integrity()
        """
        remove_info: dict[str, Any] = {'annotations': None, 'tracks': None}
        if track_identifiers:
            if verbose > 1:
                print('Removing annots of removed tracks')

            if safe:
                remove_trackid_set: set[int] = set()
                for identifier in track_identifiers:
                    try:
                        trackid = self._resolve_to_trackid(identifier)  # type: ignore
                        remove_trackid_set.add(trackid)
                    except Exception:
                        pass
                remove_trackids = sorted(remove_trackid_set)
            else:
                remove_trackids = list(map(self._resolve_to_trackid, track_identifiers))  # type: ignore

            # First remove any annotation that belongs to those tracks
            if self.index.trackid_to_aids is not None:  # type: ignore
                remove_aids = list(
                    it.chain(
                        *[
                            self.index.trackid_to_aids[trackid]  # type: ignore
                            for trackid in remove_trackids
                        ]
                    )
                )
            else:
                remove_aids = [
                    ann['id']
                    for ann in self.dataset['annotations']  # type: ignore
                    if ann.get('track_id', None) in remove_trackids
                ]

            if keep_annots:
                # Simply remove track information instead of removing the
                # entire annotation.
                if remove_aids:
                    trackless_aids = self.index.trackid_to_aids.get(  # type: ignore
                        None
                    )
                    if trackless_aids is None:
                        trackless_aids = set()
                        self.index.trackid_to_aids[None] = trackless_aids  # type: ignore
                    for aid in remove_aids:
                        self.anns[aid].pop('track_id', None)  # type: ignore
                        trackless_aids.add(aid)
            else:
                rminfo = self.remove_annotations(remove_aids, verbose=verbose)
                remove_info.update(rminfo)

            remove_info['tracks'] = len(remove_trackids)
            if verbose > 1:
                print('Removing {} track entries'.format(len(remove_trackids)))
            id_to_index = {
                track['id']: index for index, track in enumerate(self.dataset['tracks'])  # type: ignore
            }
            # Lookup the indices to remove, sort in descending order
            remove_idxs = list(ub.take(id_to_index, remove_trackids))
            _delitems(self.dataset['tracks'], remove_idxs)  # type: ignore

            self.index._remove_tracks(remove_trackids, verbose=verbose)  # type: ignore
            self._invalidate_hashid(['tracks', 'annotations'])  # type: ignore

        return remove_info

    def remove_images(
        self, gids_or_imgs: list, verbose: int = 0, safe: bool = True
    ) -> dict:
        """
        Remove images and any annotations contained by them

        Args:
            gids_or_imgs (List): list of image dicts, names, or ids

            safe (bool): if True, we perform checks to remove
                duplicates and non-existing identifiers.

        Returns:
            Dict: num_removed: information on the number of items removed

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> assert len(self.dataset['images']) == 3
            >>> gids_or_imgs = [self.imgs[2], 'astro.png']
            >>> self.remove_images(gids_or_imgs)  # xdoc: +IGNORE_WANT
            {'annotations': 11, 'images': 2}
            >>> assert len(self.dataset['images']) == 1
            >>> self._check_integrity()
            >>> gids_or_imgs = [3]
            >>> self.remove_images(gids_or_imgs)
            >>> assert len(self.dataset['images']) == 0
            >>> self._check_integrity()
        """
        remove_info: dict[str, Any] = {'annotations': None, 'images': None}
        if gids_or_imgs:
            if verbose > 1:
                print('Removing images')

            remove_gids = list(map(self._resolve_to_gid, gids_or_imgs))  # type: ignore
            if safe:
                remove_gids = sorted(set(remove_gids))
            # First remove any annotation that belongs to those images
            if self.index.gid_to_aids:  # type: ignore
                remove_aids = list(
                    it.chain(*[self.index.gid_to_aids[gid] for gid in remove_gids])  # type: ignore
                )
            else:
                remove_aids = [
                    ann['id']
                    for ann in self.dataset['annotations']  # type: ignore
                    if ann['image_id'] in remove_gids
                ]

            rminfo = self.remove_annotations(remove_aids, verbose=verbose)
            remove_info.update(rminfo)

            remove_info['images'] = len(remove_gids)
            if verbose > 1:
                print('Removing {} image entries'.format(len(remove_gids)))
            id_to_index = {
                img['id']: index for index, img in enumerate(self.dataset['images'])  # type: ignore
            }
            # Lookup the indices to remove, sort in descending order
            remove_idxs = list(ub.take(id_to_index, remove_gids))
            _delitems(self.dataset['images'], remove_idxs)  # type: ignore

            self.index._remove_images(remove_gids, verbose=verbose)  # type: ignore
            self._invalidate_hashid(['images', 'annotations'])  # type: ignore

        return remove_info

    def remove_videos(
        self, vidids_or_videos: list, verbose: int = 0, safe: bool = True
    ) -> dict:
        """
        Remove videos and any images / annotations contained by them

        Args:
            vidids_or_videos (List): list of video dicts, names, or ids

            safe (bool):
                if True, we perform checks to remove duplicates and
                non-existing identifiers.

        Returns:
            Dict: num_removed: information on the number of items removed

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('vidshapes8')
            >>> assert len(self.dataset['videos']) == 8
            >>> vidids_or_videos = [self.dataset['videos'][0]['id']]
            >>> self.remove_videos(vidids_or_videos)  # xdoc: +IGNORE_WANT
            {'annotations': 4, 'images': 2, 'videos': 1}
            >>> assert len(self.dataset['videos']) == 7
            >>> self._check_integrity()
        """
        remove_info: dict[str, Any] = {
            'annotations': None, 'images': None, 'videos': None
        }
        if vidids_or_videos:
            if verbose > 1:
                print('Removing videos')

            remove_vidids = list(map(self._resolve_to_vidid, vidids_or_videos))  # type: ignore
            if safe:
                remove_vidids = sorted(set(remove_vidids))
            # First remove any annotation that belongs to those images
            if self.index.vidid_to_gids:  # type: ignore
                remove_gids = list(
                    it.chain(
                        *[self.index.vidid_to_gids[vidid] for vidid in remove_vidids]  # type: ignore
                    )
                )
            else:
                remove_gids = [
                    ann['id']
                    for ann in self.dataset['videos']  # type: ignore
                    if ann['image_id'] in remove_gids  # type: ignore
                ]

            rminfo = self.remove_images(remove_gids, verbose=verbose, safe=safe)
            remove_info.update(rminfo)

            remove_info['videos'] = len(remove_vidids)
            if verbose > 1:
                print('Removing {} video entries'.format(len(remove_vidids)))
            id_to_index = {
                video['id']: index for index, video in enumerate(self.dataset['videos'])  # type: ignore
            }
            # Lookup the indices to remove, sort in descending order
            remove_idxs = list(ub.take(id_to_index, remove_vidids))
            _delitems(self.dataset['videos'], remove_idxs)  # type: ignore

            self.index._remove_videos(remove_vidids, verbose=verbose)  # type: ignore
            self._invalidate_hashid(['videos', 'images', 'annotations'])  # type: ignore
        return remove_info

    def remove_annotation_keypoints(self, kp_identifiers: list) -> dict:
        """
        Removes all keypoints with a particular category

        Args:
            kp_identifiers (List): list of keypoint category dicts, names, or ids

        Returns:
            Dict: num_removed: information on the number of items removed
        """
        # kpnames = {k['name'] for k in remove_kpcats}
        # TODO: needs optimization
        remove_kpcats = list(map(self._resolve_to_kpcat, kp_identifiers))  # type: ignore
        kpcids = {k['id'] for k in remove_kpcats}
        num_kps_removed = 0
        for ann in self.dataset['annotations']:  # type: ignore
            remove_idxs = [
                kp_idx
                for kp_idx, kp in enumerate(ann['keypoints'])
                if kp['keypoint_category_id'] in kpcids
            ]
            num_kps_removed += len(remove_idxs)
            _delitems(ann['keypoints'], remove_idxs)
        remove_info = {'annotation_keypoints': num_kps_removed}
        return remove_info

    def remove_keypoint_categories(
        self, kp_identifiers: list, clean_anns: bool = True
    ) -> dict:
        """
        Removes all keypoints of a particular category as well as all
        annotation keypoints with those ids.

        Args:
            kp_identifiers (List): list of keypoint category dicts, names, or ids

            clean_anns (bool):
                if True, will try to remove the now invalid keypoints from
                annotations that contain it. May cause breakage.

        Returns:
            Dict: num_removed: information on the number of items removed

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('shapes', rng=0)
            >>> kp_identifiers = ['left_eye', 'mid_tip']
            >>> remove_info = self.remove_keypoint_categories(kp_identifiers)
            >>> print('remove_info = {!r}'.format(remove_info))
            >>> # FIXME: for whatever reason demodata generation is not deterministic when seeded
            >>> # assert remove_info == {'keypoint_categories': 2, 'annotation_keypoints': 16, 'reflection_ids': 1}
            >>> assert self._resolve_to_kpcat('right_eye')['reflection_id'] is None
        """
        remove_info: dict[str, Any] = {
            'keypoint_categories': None, 'annotation_keypoints': None
        }
        remove_kpcats = list(map(self._resolve_to_kpcat, kp_identifiers))  # type: ignore

        # This is expensive, should make it optional.
        if clean_anns:
            _ann_remove_info = self.remove_annotation_keypoints(remove_kpcats)
            remove_info.update(_ann_remove_info)

        remove_kpcids = {k['id'] for k in remove_kpcats}

        for kpcat in remove_kpcats:
            self.dataset['keypoint_categories'].remove(kpcat)  # type: ignore

        # handle reflection ids
        remove_reflect_ids = 0
        for kpcat in self.dataset['keypoint_categories']:  # type: ignore
            if kpcat.get('reflection_id', None) in remove_kpcids:
                kpcat['reflection_id'] = None
                remove_reflect_ids += 1

        remove_info['reflection_ids'] = remove_reflect_ids
        remove_info['keypoint_categories'] = len(remove_kpcats)
        self.index._remove_keypoint_categories(remove_kpcids)  # type: ignore
        return remove_info

    def set_annotation_category(
        self, aid_or_ann: dict | int, cid_or_cat: dict | int
    ) -> None:
        """
        Sets the category of a single annotation

        Args:
            aid_or_ann (dict | int): annotation dict or id

            cid_or_cat (dict | int): category dict or id

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> old_freq = self.category_annotation_frequency()
            >>> aid_or_ann = aid = 2
            >>> cid_or_cat = new_cid = self.ensure_category('kitten')
            >>> self.set_annotation_category(aid, new_cid)
            >>> new_freq = self.category_annotation_frequency()
            >>> print('new_freq = {}'.format(ub.urepr(new_freq, nl=1)))
            >>> print('old_freq = {}'.format(ub.urepr(old_freq, nl=1)))
            >>> assert sum(new_freq.values()) == sum(old_freq.values())
            >>> assert new_freq['kitten'] == 1
        """
        new_cid = self._resolve_to_cid(cid_or_cat)  # type: ignore
        ann = self._resolve_to_ann(aid_or_ann)  # type: ignore
        aid = ann['id']
        if self.index:  # type: ignore
            if 'category_id' in ann:
                old_cid = ann['category_id']
                self.index.cid_to_aids[old_cid].remove(aid)  # type: ignore
        ann['category_id'] = new_cid
        if self.index:  # type: ignore
            self.index.cid_to_aids[new_cid].add(aid)  # type: ignore
        self._invalidate_hashid(['annotations'])  # type: ignore
