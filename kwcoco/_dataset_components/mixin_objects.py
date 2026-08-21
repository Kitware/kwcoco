from __future__ import annotations

from typing import Any
import ubelt as ub

from kwcoco.coco_objects1d import Categories, Videos, Images, Annots, Tracks


class MixinCocoObjects:
    """
    Expose methods to construct object lists / groups.

    This is an alternative vectorized ORM-like interface to the coco dataset
    """

    def annots(
        self,
        annot_ids: list[int] | None = None,
        image_id: int | None = None,
        track_id: int | None = None,
        video_id: int | None = None,
        image_name: str | None = None,
        track_name: str | None = None,
        video_name: str | None = None,
        *,
        trackid: Any | None = None,
        aids: Any | None = None,
        gid: Any | None = None,
    ) -> Annots:
        """
        Return vectorized annotation objects

        Args:
            annot_ids (List[int] | None):
                annotation ids to reference, if unspecified all annotations are
                returned. An alias is "aids", which may be removed in the future.

            image_id (int | None):
                return all annotations that belong to this image id.
                Mutually exclusive with other arguments.
                An alias is "gids", which may be removed in the future.

            track_id (int | None):
                return all annotations that belong to this track.
                mutually exclusive with other arguments.
                An alias is "trackid", which may be removed in the future.

            video_id (int | None):
                return all annotations that belong to this video.
                mutually exclusive with other arguments.

            track_name (str | None): name alternative to id

            image_name (str | None): name alternative to id

            video_name (str | None): name alternative to id

        Returns:
            kwcoco.coco_objects1d.Annots: vectorized annotation object

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> annots = self.annots()
            >>> print(annots)
            <Annots(num=11)>
            >>> sub_annots = annots.take([1, 2, 3])
            >>> print(sub_annots)
            <Annots(num=3)>
            >>> print(ub.urepr(sub_annots.get('bbox', None)))
            [
                [350, 5, 130, 290],
                None,
                None,
            ]
        """
        if image_id is None:
            image_id = gid
        if annot_ids is None:
            annot_ids = aids

        if trackid is not None:
            ub.schedule_deprecation(
                'kwcoco',
                'trackid',
                'argument of CocoDataset.annots',
                migration='Use "track_id" instead.',
                deprecate='0.5.9',
                error='1.0.0',
                remove='1.1.0',
            )
            track_id = trackid

        # name to id resolution

        if track_name is not None:
            track_id = self.index.name_to_track[track_name]['id']  # type: ignore

        if video_name is not None:
            video_id = self.index.name_to_video[video_name]['id']  # type: ignore

        if image_name is not None:
            image_id = self.index.name_to_img[image_name]['id']  # type: ignore

        # id lookups

        if image_id is not None:
            annot_ids = sorted(self.index.gid_to_aids[image_id])  # type: ignore

        if track_id is not None:
            annot_ids = self.index.trackid_to_aids[track_id]  # type: ignore

        if video_id is not None:
            annot_ids = sorted(
                ub.flatten(
                    [
                        self.index.gid_to_aids[gid]  # type: ignore
                        for gid in self.index.vidid_to_gids[video_id]  # type: ignore
                    ]
                )
            )

        if annot_ids is None:
            annot_ids = sorted(self.index.anns.keys())  # type: ignore

        return Annots(annot_ids, self)

    def images(
        self,
        image_ids: list[int] | None = None,
        video_id: int | None = None,
        names: list[str] | None = None,
        video_name: str | None = None,
        *,
        gids: Any | None = None,
        vidid: Any | None = None,
    ) -> Images:
        """
        Return vectorized image objects

        Args:
            image_ids (List[int] | None): image ids to reference, if unspecified
                 all images are returned. An alias is `gids`.

            video_id (int | None): returns all images that belong to this video id.
                mutually exclusive with `image_ids` arg.

            video_name (str | None):
                alternative lookup to video-id

            names (List[str] | None):
                lookup images by their names.

        Returns:
            kwcoco.coco_objects1d.Images: vectorized image object

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> images = self.images()
            >>> print(images)
            <Images(num=3)>

            >>> self = kwcoco.CocoDataset.demo('vidshapes2')
            >>> video_id = 1
            >>> images = self.images(video_id=video_id)
            >>> assert all(v == video_id for v in images.lookup('video_id'))
            >>> print(images)
            <Images(num=2)>
        """
        if image_ids is None:
            image_ids = gids

        if vidid is not None:
            ub.schedule_deprecation(
                'kwcoco',
                'vidid',
                'argument of CocoDataset.images',
                migration='Use "video_id" instead.',
                deprecate='0.5.0',
                error='1.0.0',
                remove='1.1.0',
            )
            video_id = vidid

        if video_name is not None:
            video_id = self.index.name_to_video[video_name]['id']  # type: ignore

        if video_id is not None:
            image_ids = self.index.vidid_to_gids[video_id]  # type: ignore

        if names is not None:
            image_ids = [self.index.name_to_img[name]['id'] for name in names]  # type: ignore

        if image_ids is None:
            image_ids = sorted(self.index.imgs.keys())  # type: ignore

        return Images(image_ids, self)

    def categories(
        self, category_ids: list[int] | None = None, *, cids: Any | None = None
    ) -> Categories:
        """
        Return vectorized category objects

        Args:
            category_ids (List[int] | None):
                category ids to reference, if unspecified all categories are
                returned. The `cids` argument is an alias.

        Returns:
            kwcoco.coco_objects1d.Categories: vectorized category object

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> categories = self.categories()
            >>> print(categories)
            <Categories(num=8)>
        """
        if category_ids is None:
            category_ids = cids
            if cids is not None:
                ub.schedule_deprecation(
                    'kwcoco',
                    'cids',
                    'argument of CocoDataset.categories',
                    migration='Use "category_ids" instead.',
                    deprecate='0.7.3',
                    error='1.0.0',
                    remove='1.1.0',
                )
        if category_ids is None:
            # Question: why are we sorting here? Maybe this was handling some
            # Python 3.6 case? Do we still need to?  Would it break UX?
            category_ids = sorted(self.index.cats.keys())  # type: ignore
        return Categories(category_ids, self)

    def videos(
        self,
        video_ids: list[int] | None = None,
        names: list[str] | None = None,
        *,
        vidids: Any | None = None,
    ) -> Videos:
        """
        Return vectorized video objects

        Args:
            video_ids (List[int] | None):
                video ids to reference, if unspecified all videos are returned.
                The `vidids` argument is a deprecated alias.  Mutually
                exclusive with other args.

            names (List[str] | None):
                lookup videos by their name.
                Mutually exclusive with other args.

        Returns:
            kwcoco.coco_objects1d.Videos: vectorized video object

        TODO:
            - [ ] This conflicts with what should be the property that
                should redirect to ``index.videos``, we should resolve this
                somehow. E.g. all other main members of the index (anns, imgs,
                cats) have a toplevel dataset property, we don't have one for
                videos because the name we would pick conflicts with this.

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('vidshapes2')
            >>> videos = self.videos()
            >>> print(videos)
            >>> videos.lookup('name')
            >>> videos.lookup('id')
            >>> print('videos.objs = {}'.format(ub.urepr(videos.objs[0:2], nl=1)))
        """
        if video_ids is None:
            if vidids is not None:
                ub.schedule_deprecation(
                    'kwcoco',
                    name='vidids',
                    type='parameter',
                    deprecate='0.8.8',
                    error='1.0.0',
                    remove='1.1.0',
                    migration=ub.paragraph(
                        """
                        Use video_ids instead.
                        """
                    ),
                )
            video_ids = vidids
        if video_ids is None:
            video_ids = sorted(self.index.videos.keys())  # type: ignore

        if names is not None:
            video_ids = [self.index.name_to_video[name]['id'] for name in names]  # type: ignore
        return Videos(video_ids, self)

    def tracks(
        self,
        track_ids: list[int | str] | None = None,
        names: list[str] | None = None,
        video_id: int | None = None,
        video_name: str | None = None,
    ) -> Tracks:
        """
        Return vectorized track objects

        Args:
            track_ids (List[int | str] | None):
                track ids to reference, if unspecified all tracks are returned.

            names (List[str] | None):
                lookup tracks by their name.
                Mutually exclusive with other args.

            video_id (int | None):
                if specified, return tracks in this video id.
                Note: this query is currently inefficient

            video_name (str | None):
                alternative lookup to video_id

        Returns:
            kwcoco.coco_objects1d.Tracks: vectorized video object

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('vidshapes2')
            >>> tracks = self.tracks()
            >>> print(tracks)
            >>> tracks.lookup('name')
            >>> tracks.lookup('id')
            >>> print('tracks.objs = {}'.format(ub.urepr(tracks.objs[0:2], nl=1)))

        Example:
            >>> # Test case for video id lookup when not all annotations are in tracks
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('vidshapes2')
            >>> for image_id in self.images():
            >>>     self.add_annotation(image_id=image_id, bbox=[1, 2, 3, 4])
            >>> tracks = self.tracks(video_id=1)
            >>> assert len(tracks) == 2
        """
        if track_ids is None:
            track_ids = sorted(self.index.tracks.keys())  # type: ignore

        if names is not None:
            track_ids = [self.index.name_to_track[name]['id'] for name in names]  # type: ignore

        if video_name is not None:
            # TODO handle more mutex checks
            assert video_id is None, 'video_name is mutex with video_id'
            video_id = self.index.name_to_video[video_name]['id']  # type: ignore

        if video_id is not None:
            # Note: this is an inefficient lookup
            annot_ids = sorted(
                ub.flatten(
                    [
                        self.index.gid_to_aids[gid]  # type: ignore
                        for gid in self.index.vidid_to_gids[video_id]  # type: ignore
                    ]
                )
            )
            annot_track_ids = self.annots(annot_ids).lookup('track_id', None)
            track_ids = [tid for tid in ub.unique(annot_track_ids) if tid is not None]
        return Tracks(track_ids, self)
