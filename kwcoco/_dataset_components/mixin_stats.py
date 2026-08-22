from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping
from typing import Any, TYPE_CHECKING
import ubelt as ub

if TYPE_CHECKING:
    from numpy import ndarray


class MixinCocoStats:
    """
    Methods for getting stats about the dataset
    """

    @property
    def n_annots(self) -> int:
        """The number of annotations in the dataset"""
        return len(self.dataset.get('annotations', []))  # type: ignore

    @property
    def n_images(self) -> int:
        """The number of images in the dataset"""
        return len(self.dataset.get('images', []))  # type: ignore

    @property
    def n_cats(self) -> int:
        """The number of categories in the dataset"""
        return len(self.dataset.get('categories', []))  # type: ignore

    @property
    def n_tracks(self) -> int:
        """The number of tracks in the dataset"""
        return len(self.dataset.get('tracks', []))  # type: ignore

    @property
    def n_videos(self) -> int:
        """The number of videos in the dataset"""
        return len(self.dataset.get('videos', []))  # type: ignore

    def category_annotation_frequency(self) -> dict:
        """
        Reports the number of annotations of each category

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> hist = self.category_annotation_frequency()
            >>> print(ub.urepr(hist))
            {
                'astroturf': 0,
                'human': 0,
                'astronaut': 1,
                'astronomer': 1,
                'helmet': 1,
                'rocket': 1,
                'mouth': 2,
                'star': 5,
            }
        """
        catname_to_nannots = ub.map_keys(
            lambda x: None if x is None else self.cats[x]['name'],  # type: ignore
            ub.map_vals(len, self.index.cid_to_aids),  # type: ignore
        )
        catname_to_nannots = ub.odict(
            sorted(catname_to_nannots.items(), key=lambda kv: (kv[1], kv[0]))
        )
        return catname_to_nannots

    def stats(self, **kwargs: Any) -> dict:
        """
        Compute summary statistics to describe the dataset at a high level

        This function corresponds to :mod:`kwcoco.cli.coco_stats`.

        KWargs:
            basic(bool): return basic stats', default=True
            extended(bool): return extended stats', default=True
            catfreq(bool): return category frequency stats', default=True
            boxes(bool): return bounding box stats', default=False

            annot_attrs(bool): return annotation attribute information', default=True
            image_attrs(bool): return image attribute information', default=True

        SeeAlso:
            :func:`kwcoco.coco_dataset.MixinCocoStats.stats`
            :func:`kwcoco.coco_dataset.MixinCocoStats.basic_stats`
            :func:`kwcoco.coco_dataset.MixinCocoStats.extended_stats`

        Returns:
            dict: info
        """
        config = kwargs
        dset = self
        info = {}
        if config.get('basic', True):
            info['basic'] = dset.basic_stats()

        if config.get('extended', True):
            info['extended'] = dset.extended_stats()

        if config.get('catfreq', True):
            info['catfreq'] = dset.category_annotation_frequency()

        def varied(
            obj_lut: Mapping[Any, Mapping[str, Any]]
        ) -> dict[str, dict[str, int]]:
            attrs = ub.ddict(lambda: 0)
            unique = ub.ddict(set)
            for obj in obj_lut.values():
                for key, value in obj.items():
                    if value:
                        attrs[key] += 1
                        try:
                            unique[key].add(value)
                        except TypeError:
                            unique[key].add(ub.hash_data(value, hasher='sha1'))

            varied_attrs = {
                key: {'num_unique': len(unique[key]), 'num_total': num}
                for key, num in attrs.items()
            }
            return varied_attrs

        if config.get('image_attrs', True):
            image_attrs = varied(dset.index.imgs)  # type: ignore
            info['image_attrs'] = image_attrs

        if config.get('annot_attrs', True):
            annot_attrs = varied(dset.index.anns)  # type: ignore
            info['annot_attrs'] = annot_attrs

        if config.get('video_attrs', True):
            annot_attrs = varied(dset.index.videos)  # type: ignore
            info['video_attrs'] = annot_attrs

        if config.get('boxes', False):
            info['boxes'] = dset.boxsize_stats()
        return info

    def basic_stats(self) -> dict:
        """
        Reports number of images, annotations, and categories.

        SeeAlso:
            :func:`kwcoco.coco_dataset.MixinCocoStats.stats`
            :func:`kwcoco.coco_dataset.MixinCocoStats.basic_stats`
            :func:`kwcoco.coco_dataset.MixinCocoStats.extended_stats`

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> print(ub.urepr(self.basic_stats()))
            {
                'n_anns': 11,
                'n_imgs': 3,
                'n_videos': 0,
                'n_cats': 8,
                'n_tracks': 0,
            }

            >>> from kwcoco.demo.toydata_video import random_video_dset
            >>> dset = random_video_dset(render=True, num_frames=2, num_tracks=10, rng=0)
            >>> print(ub.urepr(dset.basic_stats()))
            {
                'n_anns': 20,
                'n_imgs': 2,
                'n_videos': 1,
                'n_cats': 3,
                'n_tracks': 10,
            }
        """

        n_tracks = self.n_tracks

        if len(self.index.trackid_to_aids) > n_tracks + 3:  # type: ignore
            n_tracks = len(self.index.trackid_to_aids)  # type: ignore
            if None in self.index.trackid_to_aids:  # type: ignore
                n_tracks -= 1

        info = ub.odict(
            [
                ('n_anns', self.n_annots),
                ('n_imgs', self.n_images),
                ('n_videos', self.n_videos),
                ('n_cats', self.n_cats),
                ('n_tracks', n_tracks),
            ]
        )
        return info

    def extended_stats(self) -> dict:
        """
        Reports number of images, annotations, and categories.

        SeeAlso:
            :func:`kwcoco.coco_dataset.MixinCocoStats.stats`
            :func:`kwcoco.coco_dataset.MixinCocoStats.basic_stats`
            :func:`kwcoco.coco_dataset.MixinCocoStats.extended_stats`

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> print(ub.urepr(self.extended_stats()))
        """
        import kwarray

        def mapping_stats(
            xid_to_yids: Mapping[Any, Collection[Any]]
        ) -> dict[str, Any]:
            n_yids = [len(yids) for yids in xid_to_yids.values()]
            return kwarray.stats_dict(n_yids, n_extreme=True)

        if hasattr(self, '_all_rows_column_lookup'):
            # Hack for SQL speed, still slow though
            aid_to_cid = dict(
                self._all_rows_column_lookup('annotations', ['id', 'category_id'])  # type: ignore
            )
            gid_to_cidfreq = {}
            for gid, aids in self.index.gid_to_aids.items():  # type: ignore
                gid_to_cidfreq[gid] = ub.dict_hist(ub.take(aid_to_cid, aids))
            # for gid, aids in self.index.gid_to_aids.items():
            #     cids = self._column_lookup(
            #         'annotations', 'category_id', rowids=aids)
            #     gid_to_cidfreq[gid] = ub.dict_hist(cids)
        else:
            # this is slow for sql.
            gid_to_cidfreq = ub.map_vals(
                lambda aids: ub.dict_hist(
                    [self.anns[aid]['category_id'] for aid in aids]  # type: ignore
                ),
                self.index.gid_to_aids,  # type: ignore
            )
        gid_to_cids = {
            gid: list(cidfreq.keys()) for gid, cidfreq in gid_to_cidfreq.items()
        }
        trackid_to_aids = ub.udict(self.index.trackid_to_aids) - {None}  # type: ignore

        img_has_anns = [
            int(len(aids) > 0) for gid, aids in self.index.gid_to_aids.items()  # type: ignore
        ]
        imgs_with_annots = kwarray.stats_dict(
            img_has_anns, n_extreme=True, quantile=False
        )

        return ub.odict(
            [
                ('annots_per_img', mapping_stats(self.index.gid_to_aids)),  # type: ignore
                ('imgs_per_cat', mapping_stats(self.index.cid_to_gids)),  # type: ignore
                ('cats_per_img', mapping_stats(gid_to_cids)),
                ('annots_per_cat', mapping_stats(self.index.cid_to_aids)),  # type: ignore
                ('imgs_per_video', mapping_stats(self.index.vidid_to_gids)),  # type: ignore
                ('annots_per_track', mapping_stats(trackid_to_aids)),
                ('imgs_with_annots', imgs_with_annots),
            ]
        )

    def boxsize_stats(
        self,
        anchors: int | None = None,
        perclass: bool = True,
        gids: list[int] | None = None,
        aids: list[int] | None = None,
        verbose: int = 0,
        clusterkw: dict = {},
        statskw: dict = {},
    ) -> dict[str, dict[str, dict | ndarray]]:
        """
        Compute statistics about bounding box sizes.

        Also computes anchor boxes using kmeans if ``anchors`` is specified.

        Args:
            anchors (int | None): if specified also computes box anchors
                via KMeans clustering

            perclass (bool): if True also computes stats for each category

            gids (List[int] | None):
                if specified only compute stats for these image ids.
                Defaults to None.

            aids (List[int] | None):
                if specified only compute stats for these annotation ids.
                Defaults to None.

            verbose (int): verbosity level

            clusterkw (dict):
                kwargs for :class:`sklearn.cluster.KMeans` used
                if computing anchors.

            statskw (dict): kwargs for :func:`kwarray.stats_dict`

        Returns:
            Dict[str, Dict[str, Dict | ndarray]]:
                Stats are returned in width-height format.

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('shapes32')
            >>> infos = self.boxsize_stats(anchors=4, perclass=False)
            >>> print(ub.urepr(infos, nl=-1, precision=2))

            >>> infos = self.boxsize_stats(gids=[1], statskw=dict(median=True))
            >>> print(ub.urepr(infos, nl=-1, precision=2))
        """
        import kwarray
        import numpy as np

        cname_to_box_sizes = defaultdict(list)

        if bool(gids) and bool(aids):
            raise ValueError('specifying gids and aids is mutually exclusive')

        if gids is not None:
            aids = ub.flatten(ub.take(self.index.gid_to_aids, gids))  # type: ignore
        if aids is not None:
            anns = ub.take(self.anns, aids)  # type: ignore
        else:
            anns = self.dataset.get('annotations', [])  # type: ignore

        for ann in anns:
            if 'bbox' in ann:
                cname = self.cats[ann['category_id']]['name']  # type: ignore
                cname_to_box_sizes[cname].append(ann['bbox'][2:4])
        cname_to_box_sizes_arr = ub.map_vals(np.array, cname_to_box_sizes)

        def _boxes_info(box_sizes):
            box_info = {'stats': kwarray.stats_dict(box_sizes, axis=0, **statskw)}
            if anchors:
                from sklearn import cluster

                defaultkw = {
                    'n_clusters': anchors,
                    'n_init': 20,
                    'max_iter': 10000,
                    'tol': 1e-6,
                    'algorithm': 'elkan',
                    'verbose': verbose,
                }
                kmkw = ub.dict_union(defaultkw, clusterkw)
                algo = cluster.KMeans(**kmkw)
                algo.fit(box_sizes)
                anchor_sizes = algo.cluster_centers_
                box_info['anchors'] = anchor_sizes
            return box_info

        infos = {}

        if perclass:
            cid_to_info = {}
            for cname, box_sizes in cname_to_box_sizes_arr.items():
                if verbose:
                    print('compute {} bbox stats'.format(cname))
                cid_to_info[cname] = _boxes_info(box_sizes)
            infos['perclass'] = cid_to_info

        if verbose:
            print('compute all bbox stats')
        if cname_to_box_sizes_arr:
            all_sizes = np.vstack(list(cname_to_box_sizes_arr.values()))
            all_info = _boxes_info(all_sizes)
        else:
            if anchors:
                raise ValueError('Cannot compute anchor boxes without bounding boxes')
            all_info = {'stats': {}}
        infos['all'] = all_info
        return infos  # type: ignore

    def find_representative_images(self, gids: None | list = None) -> list:
        r"""
        Find images that have a wide array of categories.

        Attempt to find the fewest images that cover all categories using
        images that contain both a large and small number of annotations.

        Args:
            gids (None | List): Subset of image ids to consider when finding
                representative images. Uses all images if unspecified.

        Returns:
            List: list of image ids determined to be representative

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> gids = self.find_representative_images()
            >>> print('gids = {!r}'.format(gids))
            >>> gids = self.find_representative_images([3])
            >>> print('gids = {!r}'.format(gids))

            >>> self = kwcoco.CocoDataset.demo('shapes8')
            >>> gids = self.find_representative_images()
            >>> print('gids = {!r}'.format(gids))
            >>> valid = {7, 1}
            >>> gids = self.find_representative_images(valid)
            >>> assert valid.issuperset(gids)
            >>> print('gids = {!r}'.format(gids))
        """
        import kwarray

        if gids is None:
            gids = sorted(self.imgs.keys())  # type: ignore
            gid_to_aids = self.index.gid_to_aids  # type: ignore
        else:
            gid_to_aids = ub.dict_subset(self.index.gid_to_aids, gids)  # type: ignore

        all_cids = set(self.index.cid_to_aids.keys())  # type: ignore

        # Select representative images to draw such that each category
        # appears at least once.
        gid_to_cidfreq = ub.map_vals(
            lambda aids: ub.dict_hist([self.anns[aid]['category_id'] for aid in aids]),  # type: ignore
            gid_to_aids,
        )

        gid_to_nannots = ub.map_vals(len, gid_to_aids)

        gid_to_cids = {
            gid: list(cidfreq.keys()) for gid, cidfreq in gid_to_cidfreq.items()
        }

        for gid, nannots in gid_to_nannots.items():
            if nannots == 0:
                # Add a dummy category to note images without any annotations
                gid_to_cids[gid].append(-1)
                all_cids.add(-1)

        # Solve setcover with different weight schemes to get a better
        # representative sample.

        candidate_sets = gid_to_cids.copy()

        selected = {}

        large_image_weights = gid_to_nannots
        small_image_weights = ub.map_vals(lambda x: 1 / (x + 1), gid_to_nannots)

        cover1 = kwarray.setcover(candidate_sets, items=all_cids)
        selected.update(cover1)
        candidate_sets = ub.dict_diff(candidate_sets, cover1)

        cover2 = kwarray.setcover(
            candidate_sets, items=all_cids, set_weights=large_image_weights  # type: ignore
        )
        selected.update(cover2)
        candidate_sets = ub.dict_diff(candidate_sets, cover2)

        cover3 = kwarray.setcover(
            candidate_sets, items=all_cids, set_weights=small_image_weights
        )
        selected.update(cover3)
        candidate_sets = ub.dict_diff(candidate_sets, cover3)

        selected_gids = sorted(selected.keys())
        return selected_gids
