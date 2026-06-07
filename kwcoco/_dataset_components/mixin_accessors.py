from __future__ import annotations

from typing import Any, TYPE_CHECKING
import numbers
import ubelt as ub

if TYPE_CHECKING:
    from os import PathLike
    import numpy as np
    import networkx as nx
    from numpy.typing import ArrayLike
    from delayed_image.channel_spec import FusedChannelSpec
    from kwcoco.category_tree import CategoryTree
    from kwcoco.coco_image import CocoImage


class MixinCocoAccessors:
    """
    TODO: better name
    """

    def delayed_load(
        self,
        gid: int,
        channels: FusedChannelSpec | None = None,
        space: str = 'image',
    ) -> Any:
        """
        Experimental method

        Args:
            gid (int): image id to load

            channels (FusedChannelSpec): specific channels to load.
                if unspecified, all channels are loaded.

            space (str):
                can either be "image" for loading in image space, or
                "video" for loading in video space.

        TODO:
            - [X] Currently can only take all or none of the channels from each
                base-image / auxiliary dict. For instance if the main image is
                r|g|b you can't just select g|b at the moment.

            - [X] The order of the channels in the delayed load should
                match the requested channel order.

            - [X] TODO: add nans to bands that don't exist or throw an error

        Example:
            >>> import kwcoco
            >>> gid = 1
            >>> #
            >>> self = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> delayed = self.delayed_load(gid)
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize()))
            >>> #
            >>> self = kwcoco.CocoDataset.demo('shapes8')
            >>> delayed = self.delayed_load(gid)
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize()))

            >>> crop = delayed.crop((slice(0, 3), slice(0, 3)))
            >>> crop.finalize()

            >>> # TODO: should only select the "red" channel
            >>> self = kwcoco.CocoDataset.demo('shapes8')
            >>> delayed = self.delayed_load(gid, channels='r')

            >>> import kwcoco
            >>> gid = 1
            >>> #
            >>> self = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> delayed = self.delayed_load(gid, channels='B1|B2', space='image')
            >>> print('delayed = {!r}'.format(delayed))
            >>> delayed = self.delayed_load(gid, channels='B1|B2|B11', space='image')
            >>> print('delayed = {!r}'.format(delayed))
            >>> delayed = self.delayed_load(gid, channels='B8|B1', space='video')
            >>> print('delayed = {!r}'.format(delayed))

            >>> delayed = self.delayed_load(gid, channels='B8|foo|bar|B1', space='video')
            >>> print('delayed = {!r}'.format(delayed))
        """
        ub.schedule_deprecation(
            'kwcoco',
            'delayed_load',
            'method',
            migration='Use ".coco_image(gid).imdelay(...)" instead.',
            deprecate='0.7.3',
            error='1.0.0',
            remove='1.1.0',
        )
        coco_img = self.coco_image(gid)
        delayed = coco_img.imdelay(channels=channels, space=space)
        return delayed

    def load_image(
        self, gid_or_img: int | dict, channels: str | None = None
    ) -> np.ndarray:
        """
        Reads an image from disk and

        Args:
            gid_or_img (int | dict): image id or image dict
            channels (str | None): if specified, load data from auxiliary
                channels instead

        Returns:
            np.ndarray : the image

        Note:
            Prefer to use the CocoImage methods instead
        """
        try:
            import kwimage

            gpath = self.get_image_fpath(gid_or_img, channels=channels)
            np_img = kwimage.imread(gpath)
        except Exception:
            img = self._resolve_to_img(gid_or_img)  # type: ignore
            np_img = self.delayed_load(img['id'], channels=channels).finalize()  # type: ignore
        return np_img

    def get_image_fpath(
        self, gid_or_img: int | dict, channels: str | None = None
    ) -> PathLike:
        """
        Returns the full path to the image

        Args:
            gid_or_img (int | dict): image id or image dict
            channels (str | None): if specified, return a path to data
                containing auxiliary channels instead

        Note:
            Prefer to use the CocoImage methods instead

        Returns:
            PathLike: full path to the image
        """
        if channels is not None:
            gpath = self.get_auxiliary_fpath(gid_or_img, channels)
        else:
            img = self._resolve_to_img(gid_or_img)
            gpath = ub.Path(self.bundle_dpath) / img['file_name']  # type: ignore
        return gpath

    def _get_img_auxiliary(self, gid_or_img: Any, channels: Any) -> Any:
        """returns the auxiliary dictionary for a specific channel"""
        img = self._resolve_to_img(gid_or_img)
        found = None
        if 'auxiliary' in img:
            auxlist = img['auxiliary']
        elif 'assets' in img:
            auxlist = img['assets']
        else:
            raise KeyError('no auxiliary data')
        for aux in auxlist:
            if aux['channels'] == channels:
                found = aux
                break
        if found is None:
            raise Exception(
                'Image does not have auxiliary channels={}'.format(channels)
            )
        return found

    def get_auxiliary_fpath(self, gid_or_img: int | dict, channels: str) -> ub.Path:
        """
        Returns the full path to auxiliary data for an image

        Args:
            gid_or_img (int | dict): an image or its id
            channels (str): the auxiliary channel to load (e.g. disparity)

        Note:
            Prefer to use the CocoImage methods instead

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('shapes8', aux=True)
            >>> self.get_auxiliary_fpath(1, 'disparity')
        """
        aux = self._get_img_auxiliary(gid_or_img, channels)
        fpath = ub.Path(self.bundle_dpath) / aux['file_name']  # type: ignore
        return fpath

    def load_annot_sample(
        self, aid_or_ann: Any, image: ArrayLike | None = None, pad: Any | None = None
    ) -> dict:
        """
        Reads the chip of an annotation. Note this is much less efficient than
        using a sampler, but it doesn't require disk cache.

        Maybe deprecate?

        Args:
            aid_or_int (int | dict): annot id or dict
            image (ArrayLike | None): preloaded image
                (note: this process is inefficient unless image is specified)

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> sample = self.load_annot_sample(2, pad=100)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(sample['im'])
            >>> kwplot.show_if_requested()
        """
        import kwarray
        import numpy as np

        ann = self._resolve_to_ann(aid_or_ann)
        if image is None:
            image = self.load_image(ann['image_id'])

        x, y, w, h = ann['bbox']
        in_slice = (
            slice(int(y), int(np.ceil(y + h))),
            slice(int(x), int(np.ceil(x + w))),
        )
        data_sliced, transform = kwarray.padded_slice(
            image, in_slice, pad=pad, return_info=True
        )

        sample = {
            'im': data_sliced,
            'transform': transform,
        }
        return sample

    def _resolve_to_id(self, id_or_dict: Any) -> int:
        """
        Ensures output is an id
        """
        if isinstance(id_or_dict, numbers.Integral):
            resolved_id = id_or_dict
        else:
            resolved_id = id_or_dict['id']
        return resolved_id

    def _resolve_to_cid(self, id_or_name_or_dict: Any) -> int:
        """
        Ensures output is an category id

        Note:
            this does not resolve aliases (yet), for that see _alias_to_cat

        TODO:
            we could maintain an alias index to make this fast
        """
        if isinstance(id_or_name_or_dict, numbers.Integral):
            resolved_id = id_or_name_or_dict
        elif isinstance(id_or_name_or_dict, str):
            resolved_id = self.index.name_to_cat[id_or_name_or_dict]['id']  # type: ignore
        else:
            resolved_id = id_or_name_or_dict['id']
        return resolved_id

    def _resolve_to_gid(self, id_or_name_or_dict: Any) -> int:
        """
        Ensures output is an category id
        """
        if isinstance(id_or_name_or_dict, numbers.Integral):
            resolved_id = id_or_name_or_dict
        elif isinstance(id_or_name_or_dict, str):
            resolved_id = self.index.file_name_to_img[id_or_name_or_dict]['id']  # type: ignore
        else:
            resolved_id = id_or_name_or_dict['id']
        return resolved_id

    def _resolve_to_vidid(self, id_or_name_or_dict: Any) -> int:
        """
        Ensures output is an video id
        """
        if isinstance(id_or_name_or_dict, numbers.Integral):
            resolved_id = id_or_name_or_dict
        elif isinstance(id_or_name_or_dict, str):
            resolved_id = self.index.name_to_video[id_or_name_or_dict]['id']  # type: ignore
        else:
            resolved_id = id_or_name_or_dict['id']
        return resolved_id

    def _resolve_to_trackid(self, id_or_name_or_dict: Any) -> int:
        if isinstance(id_or_name_or_dict, numbers.Integral):
            resolved_id = id_or_name_or_dict
        elif isinstance(id_or_name_or_dict, str):
            resolved_id = self.index.name_to_track[id_or_name_or_dict]['id']  # type: ignore
        else:
            resolved_id = id_or_name_or_dict['id']
        return resolved_id

    def _resolve_to_ann(self, aid_or_ann: Any) -> dict:
        """
        Ensures output is an annotation dictionary
        """
        if isinstance(aid_or_ann, numbers.Integral):
            resolved_ann = None
            if self.anns is not None:  # type: ignore
                resolved_ann = self.anns[aid_or_ann]  # type: ignore
            else:
                for ann in self.dataset['annotations']:  # type: ignore
                    if ann['id'] == aid_or_ann:
                        resolved_ann = ann
                        break
                if not resolved_ann:
                    raise IndexError('aid {} not in dataset'.format(aid_or_ann))
        else:
            resolved_ann = aid_or_ann
        return resolved_ann

    def _resolve_to_img(self, gid_or_img: Any) -> dict:
        """
        Ensures output is an image dictionary
        """
        if isinstance(gid_or_img, numbers.Integral):
            resolved_img = None
            if self.imgs is not None:  # type: ignore
                resolved_img = self.imgs[gid_or_img]  # type: ignore
            else:
                for img in self.dataset['images']:  # type: ignore
                    if img['id'] == gid_or_img:
                        resolved_img = img
                        break
                if not resolved_img:
                    raise IndexError('gid {} not in dataset'.format(gid_or_img))
        else:
            resolved_img = gid_or_img
        return resolved_img

    def _resolve_to_kpcat(self, kp_identifier: int | str | dict) -> dict:
        """
        Lookup a keypoint-category dict via its name or id

        Args:
            kp_identifier (int | str | dict): either the keypoint category
                name, alias, or its keypoint_category_id.

        Returns:
            Dict: keypoint category dictionary

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('shapes')
            >>> kpcat1 = self._resolve_to_kpcat(1)
            >>> kpcat2 = self._resolve_to_kpcat('left_eye')
            >>> assert kpcat1 is kpcat2
            >>> import pytest
            >>> with pytest.raises(KeyError):
            >>>     self._resolve_to_cat('human')
        """
        if 'keypoint_categories' not in self.dataset:  # type: ignore
            raise NotImplementedError('Must have newstyle keypoints to use')

        # TODO: add keypoint categories to the index and optimize
        if isinstance(kp_identifier, numbers.Integral):
            kpcat = None
            for _kpcat in self.dataset['keypoint_categories']:  # type: ignore
                if _kpcat['id'] == kp_identifier:
                    kpcat = _kpcat
            if kpcat is None:
                raise KeyError('unable to find keypoint category')
        elif isinstance(kp_identifier, str):
            kpcat = None
            for _kpcat in self.dataset['keypoint_categories']:  # type: ignore
                if _kpcat['name'] == kp_identifier:
                    kpcat = _kpcat
            if kpcat is None:
                for _kpcat in self.dataset['keypoint_categories']:  # type: ignore
                    alias = _kpcat.get('alias', {})
                    alias = alias if ub.iterable(alias) else {alias}
                    if kp_identifier in alias:
                        kpcat = _kpcat
            if kpcat is None:
                raise KeyError('unable to find keypoint category')
        elif isinstance(kp_identifier, dict):
            kpcat = kp_identifier
        else:
            raise TypeError(type(kp_identifier))
        return kpcat

    def _resolve_to_cat(self, cat_identifier: int | str | dict) -> dict:
        """
        Lookup a coco-category dict via its name, alias, or id.

        Args:
            cat_identifier (int | str | dict): either the category name,
                alias, or its category_id.

        Raises:
            KeyError: if the category doesn't exist.

        Note:
            If the index is not built, the method will work but may be slow.

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> cat = self._resolve_to_cat('human')
            >>> import pytest
            >>> assert self._resolve_to_cat(cat['id']) is cat
            >>> assert self._resolve_to_cat(cat) is cat
            >>> with pytest.raises(KeyError):
            >>>     self._resolve_to_cat(32)
            >>> self.index.clear()
            >>> assert self._resolve_to_cat(cat['id']) is cat
            >>> with pytest.raises(KeyError):
            >>>     self._resolve_to_cat(32)
        """
        if isinstance(cat_identifier, numbers.Integral):
            if self.cats:  # type: ignore
                cat = self.cats[cat_identifier]  # type: ignore
            else:
                # If the index is not built
                found = None
                for cat in self.dataset['categories']:  # type: ignore
                    if cat['id'] == cat_identifier:
                        found = cat
                        break
                if found is None:
                    raise KeyError(
                        'Cannot find a category with id={}'.format(cat_identifier)
                    )
        elif isinstance(cat_identifier, str):
            cat = self._alias_to_cat(cat_identifier)
        elif isinstance(cat_identifier, dict):
            cat = cat_identifier
        else:
            raise TypeError(type(cat_identifier))
        return cat

    def _alias_to_cat(self, alias_catname: str) -> dict:
        """
        Lookup a coco-category via its name or an "alias" name.
        In production code, use :func:`_resolve_to_cat` instead.

        Args:
            alias_catname (str): category name or alias

        Returns:
            dict: coco category dictionary

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> cat = self._alias_to_cat('human')
            >>> import pytest
            >>> with pytest.raises(KeyError):
            >>>     self._alias_to_cat('person')
            >>> cat['alias'] = ['person']
            >>> self._alias_to_cat('person')
            >>> cat['alias'] = 'person'
            >>> self._alias_to_cat('person')
            >>> assert self._alias_to_cat(None) is None
        """
        if alias_catname is None:
            return None
        if self.name_to_cat and alias_catname in self.name_to_cat:  # type: ignore
            fixed_catname = alias_catname
            fixed_cat = self.name_to_cat[fixed_catname]  # type: ignore
        else:
            # Try to find an alias
            fixed_catname = None
            fixed_cat = None
            for cat in self.dataset['categories']:  # type: ignore
                alias_list = cat.get('alias', [])
                if isinstance(alias_list, str):
                    alias_list = [alias_list]
                assert isinstance(alias_list, list)
                alias_list = alias_list + [cat['name']]
                for alias in alias_list:
                    if alias_catname.lower() == alias.lower():
                        fixed_cat = cat
                        fixed_catname = cat['name']
                        break
                if fixed_cat is not None:
                    break

            if fixed_cat is None:
                raise KeyError('Unknown category: {}'.format(alias_catname))

        return fixed_cat

    def category_graph(self) -> nx.DiGraph:
        """
        Construct a networkx category hierarchy

        Returns:
            networkx.DiGraph:
                graph: a directed graph where category names are the nodes,
                supercategories define edges, and items in each category dict
                (e.g. category id) are added as node properties.

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> graph = self.category_graph()
            >>> assert 'astronaut' in graph.nodes()
            >>> assert 'keypoints' in graph.nodes['human']

        Ignore:
            import graphid
            graphid.util.show_nx(graph)
        """
        # TODO: should supercategories that don't exist as nodes be added here?
        import networkx as nx

        graph = nx.DiGraph()
        for cat in self.dataset['categories']:  # type: ignore
            graph.add_node(cat['name'], **cat)
            if cat.get('supercategory', None) is not None:
                u = cat['supercategory']
                v = cat['name']
                if u != v:
                    graph.add_edge(u, v)
        return graph

    def object_categories(self) -> CategoryTree:
        """
        Construct a consistent CategoryTree representation of object classes

        Returns:
            kwcoco.CategoryTree: category data structure

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> classes = self.object_categories()
            >>> print('classes = {}'.format(classes))
        """
        from kwcoco.category_tree import CategoryTree

        graph = self.category_graph()
        classes = CategoryTree(graph)
        return classes

    def keypoint_categories(self) -> CategoryTree:
        """
        Construct a consistent CategoryTree representation of keypoint classes

        Returns:
            kwcoco.CategoryTree: category data structure

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> classes = self.keypoint_categories()
            >>> print('classes = {}'.format(classes))
        """
        from kwcoco.category_tree import CategoryTree

        try:
            if self.index.kpcats is not None:  # type: ignore
                kpcats = self.index.kpcats.values()  # type: ignore
            else:
                kpcats = iter(self.dataset['keypoint_categories'])  # type: ignore
            kpcats = list(kpcats)
            if len(kpcats) == 0:
                raise KeyError  # hack for old behavior
        except KeyError:
            catnames = self._keypoint_category_names()
            classes = CategoryTree.coerce(catnames)
        else:
            import networkx as nx

            graph = nx.DiGraph()
            for cat in kpcats:
                graph.add_node(cat['name'], **cat)
                if cat.get('supercategory', None) is not None:
                    graph.add_edge(cat['supercategory'], cat['name'])
            classes = CategoryTree(graph)
        return classes

    def _keypoint_category_names(self) -> list[str]:
        """
        Construct keypoint categories names.

        Uses new-style if possible, otherwise this falls back on old-style.

        Returns:
            List[str]:
                names - list of keypoint category names

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> names = self._keypoint_category_names()
            >>> print(names)
        """
        kpcats = self.dataset.get('keypoint_categories', None)  # type: ignore
        if kpcats is not None:
            return [c['name'] for c in self.dataset['keypoint_categories']]  # type: ignore
        else:
            names = []
            cats = sorted(self.dataset['categories'], key=lambda c: c['id'])  # type: ignore
            for cat in cats:
                if 'keypoints' in cat:
                    names.extend(cat['keypoints'])
            return names

    def _lookup_kpnames(self, cid: int) -> list[str]:
        """Get the keypoint categories for a certain class"""
        kpnames = None
        orig_cat = self.cats[cid]  # type: ignore
        while kpnames is None:
            # Extract keypoint names for each annotation
            cat = self.cats[cid]  # type: ignore
            parent = cat.get('supercategory', None)
            if 'keypoints' in cat:
                kpnames = cat['keypoints']
            elif parent is not None:
                cid = self.name_to_cat[cat['supercategory']]['id']  # type: ignore
            else:
                raise KeyError(
                    'could not find keypoint names for cid={}, cat={}, orig_cat={}'.format(
                        cid, cat, orig_cat
                    )
                )
        return kpnames

    def _coco_image(self, gid: int) -> CocoImage:
        ub.schedule_deprecation(
            'kwcoco',
            '_coco_image',
            'method',
            migration='Use "coco_image" instead.',
            deprecate='0.5.9',
            error='1.0.0',
            remove='1.1.0',
        )
        return self.coco_image(gid)

    def coco_image(self, gid: int) -> CocoImage:
        """
        Args:
            gid (int): image id

        Returns:
            CocoImage
        """
        # Experimental
        from kwcoco.coco_image import CocoImage

        img = self.index.imgs[gid]  # type: ignore
        image = CocoImage(img, dset=self)
        return image
