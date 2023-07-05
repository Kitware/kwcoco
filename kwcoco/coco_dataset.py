"""
An implementation and extension of the original MS-COCO API [CocoFormat]_.

Extends the format to also include line annotations.

The following describes psuedo-code for the high level spec (some of which may
not be have full support in the Python API). A formal json-schema is defined in
:mod:`kwcoco.coco_schema`.


Note:
    The main object in this file is :class:`CocoDataset`, which is composed of
    several mixin classes. See the class and method documentation for more
    details.

An informal description of the spec given in: `coco_schema_informal.rst <coco_schema_informal.rst>`_.

For a formal description of the spec see the  `coco_schema.json <coco_schema.json>`_.

TODO:
    - [ ] Use ijson (modified to support NaN) to lazilly load pieces of the
        dataset in the background or on demand. This will give us faster access
        to categories / images, whereas we will always have to wait for
        annotations etc...

    - [X] Should img_root be changed to bundle_dpath?

    - [ ] Read video data, return numpy arrays (requires API for images)

    - [ ] Spec for video URI, and convert to frames @ framerate function.

    - [x] Document channel spec

    - [x] Document sensor-channel spec

    - [X] Add remove videos method

    - [ ] Efficiency: Make video annotations more efficient by only tracking
          keyframes, provide an API to obtain a dense or interpolated
          annotation on an intermediate frame.

    - [ ] Efficiency: Allow each section of the kwcoco file to be written as a
          separate json file. Perhaps allow genric pointer support? Might get
          messy.

    - [ ] Reroot needs to be redesigned very carefully.

    - [ ] Allow parts of the kwcoco file to be references to other json files.

    - [ ] Add top-level track table (in progress)

References:
    .. [CocoFormat] http://cocodataset.org/#format-data
    .. [PyCocoToolsMask] https://github.com/nightrome/cocostuffapi/blob/master/PythonAPI/pycocotools/mask.py
    .. [CocoTutorial] https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#coco-dataset-format
"""
import copy
import sys
import itertools as it
import numbers
import os
import ubelt as ub
import warnings

from packaging.version import parse as Version
from collections import OrderedDict, defaultdict
from os.path import (dirname, basename, join, exists, isdir, relpath)
from functools import partial

# Vectorized ORM-Like containers
from kwcoco.coco_objects1d import Categories, Videos, Images, Annots
from kwcoco.abstract_coco_dataset import AbstractCocoDataset
from kwcoco import exceptions

from kwcoco._helpers import (
    SortedSet, UniqueNameRemapper, _ID_Remapper, _NextId,
    _delitems, _lut_image_frame_index, _lut_annot_frame_index,
    _load_and_postprocess, _image_corruption_check
)

import json as pjson
from types import ModuleType
# The ujson library is faster than Python's json, but the API has some
# limitations and requires a minimum version. Currently we only use it to read,
# we have to wait for https://github.com/ultrajson/ultrajson/pull/518 to land
# before we use it to write.
try:
    import ujson
except ImportError:
    ujson = None

KWCOCO_USE_UJSON = bool(os.environ.get('KWCOCO_USE_UJSON'))

if ujson is not None and Version(ujson.__version__) >= Version('5.2.0') and KWCOCO_USE_UJSON:
    json_r: ModuleType = ujson
    json_w: ModuleType = pjson
else:
    json_r: ModuleType = pjson
    json_w: ModuleType = pjson


if sys.version_info <= (3, 6):
    _dict = OrderedDict
else:
    # TODO: Ensure that switching to dict in 3.7+ doesn't change anything
    # _dict = dict
    _dict = OrderedDict


# These are the keys that are / should be supported by the API
SPEC_KEYS = [
    'info',
    'licenses',
    'categories',
    'keypoint_categories',  # support only partially implemented
    'videos',
    'images',
    'annotations',
    'tracks',
]


class MixinCocoDepricate(object):
    """
    These functions are marked for deprication and will be removed
    """

    def keypoint_annotation_frequency(self):
        """
        DEPRECATED

        Example:
            >>> import kwcoco
            >>> import ubelt as ub
            >>> self = kwcoco.CocoDataset.demo('shapes', rng=0)
            >>> hist = self.keypoint_annotation_frequency()
            >>> hist = ub.odict(sorted(hist.items()))
            >>> # FIXME: for whatever reason demodata generation is not determenistic when seeded
            >>> print(ub.urepr(hist))  # xdoc: +IGNORE_WANT
            {
                'bot_tip': 6,
                'left_eye': 14,
                'mid_tip': 6,
                'right_eye': 14,
                'top_tip': 6,
            }
        """
        ub.schedule_deprecation(
            'kwcoco', name='keypoint_annotation_frequency', type='method',
            deprecate='0.3.4', error='1.0.0', remove='1.1.0',
            migration=(
                'Implement this functionality explicitly. '
                'It is too niche for a the core API.'
                'Or propose a better way on '
                'https://gitlab.kitware.com/computer-vision/kwcoco/-/issues '
            )
        )
        ann_kpcids = [kp['keypoint_category_id']
                      for ann in self.dataset['annotations']
                      for kp in ann.get('keypoints', [])]
        kpcid_to_name = {kpcat['id']: kpcat['name']
                         for kpcat in self.dataset['keypoint_categories']}
        kpcid_to_num = ub.dict_hist(ann_kpcids,
                                    labels=list(kpcid_to_name.keys()))
        kpname_to_num = ub.map_keys(kpcid_to_name, kpcid_to_num)
        return kpname_to_num

    def category_annotation_type_frequency(self):
        """
        DEPRECATED

        Reports the number of annotations of each type for each category

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> hist = self.category_annotation_frequency()
            >>> print(ub.urepr(hist))
        """
        catname_to_nannot_types = {}
        ub.schedule_deprecation(
            'kwcoco', name='category_annotation_type_frequency', type='method',
            deprecate='0.3.4', error='1.0.0', remove='1.1.0',
            migration=(
                'Implement this functionality explicitly. '
                'It is too niche for a the core API.'
                'Or propose a better way on '
                'https://gitlab.kitware.com/computer-vision/kwcoco/-/issues '
            )
        )

        def _annot_type(ann):
            """
            Returns what type of annotation ``ann`` is.
            """
            return tuple(sorted(set(ann) & {'bbox', 'line', 'keypoints'}))

        for cid, aids in self.index.cid_to_aids.items():
            name = self.cats[cid]['name']
            hist = ub.dict_hist(map(_annot_type, ub.take(self.anns, aids)))
            catname_to_nannot_types[name] = ub.map_keys(
                lambda k: k[0] if len(k) == 1 else k, hist)
        return catname_to_nannot_types

    def imread(self, gid):
        """
        DEPRECATED: use load_image or delayed_image

        Loads a particular image
        """
        ub.schedule_deprecation(
            'kwcoco', name='imread', type='method',
            deprecate='0.3.4', error='1.0.0', remove='1.1.0',
            migration=(
                'use `self.coco_image(gid).imdelay().finalize()`.'
            )
        )
        return self.load_image(gid)


class MixinCocoAccessors(object):
    """
    TODO: better name
    """

    def delayed_load(self, gid, channels=None, space='image'):
        """
        Experimental method

        Args:
            gid (int): image id to load

            channels (kwcoco.FusedChannelSpec): specific channels to load.
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
        coco_img = self.coco_image(gid)
        delayed = coco_img.imdelay(channels=channels, space=space)
        return delayed

    def load_image(self, gid_or_img, channels=None):
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
            img = self._resolve_to_img(gid_or_img)
            np_img = self.delayed_load(img['id'], channels=channels).finalize()
        return np_img

        return np_img

    def get_image_fpath(self, gid_or_img, channels=None):
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
            gpath = ub.Path(self.bundle_dpath) / img['file_name']
        return gpath

    def _get_img_auxiliary(self, gid_or_img, channels):
        """ returns the auxiliary dictionary for a specific channel """
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
                'Image does not have auxiliary channels={}'.format(channels))
        return found

    def get_auxiliary_fpath(self, gid_or_img, channels):
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
        fpath = ub.Path(self.bundle_dpath) / aux['file_name']
        return fpath

    def load_annot_sample(self, aid_or_ann, image=None, pad=None):
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
        data_sliced, transform = kwarray.padded_slice(image, in_slice, pad=pad,
                                                      return_info=True)

        sample = {
            'im': data_sliced,
            'transform': transform,
        }
        return sample

    def _resolve_to_id(self, id_or_dict):
        """
        Ensures output is an id
        """
        if isinstance(id_or_dict, numbers.Integral):
            resolved_id = id_or_dict
        else:
            resolved_id = id_or_dict['id']
        return resolved_id

    def _resolve_to_cid(self, id_or_name_or_dict):
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
            resolved_id = self.index.name_to_cat[id_or_name_or_dict]['id']
        else:
            resolved_id = id_or_name_or_dict['id']
        return resolved_id

    def _resolve_to_gid(self, id_or_name_or_dict):
        """
        Ensures output is an category id
        """
        if isinstance(id_or_name_or_dict, numbers.Integral):
            resolved_id = id_or_name_or_dict
        elif isinstance(id_or_name_or_dict, str):
            resolved_id = self.index.file_name_to_img[id_or_name_or_dict]['id']
        else:
            resolved_id = id_or_name_or_dict['id']
        return resolved_id

    def _resolve_to_vidid(self, id_or_name_or_dict):
        """
        Ensures output is an video id
        """
        if isinstance(id_or_name_or_dict, numbers.Integral):
            resolved_id = id_or_name_or_dict
        elif isinstance(id_or_name_or_dict, str):
            resolved_id = self.index.name_to_video[id_or_name_or_dict]['id']
        else:
            resolved_id = id_or_name_or_dict['id']
        return resolved_id

    def _resolve_to_ann(self, aid_or_ann):
        """
        Ensures output is an annotation dictionary
        """
        if isinstance(aid_or_ann, numbers.Integral):
            resolved_ann = None
            if self.anns is not None:
                resolved_ann = self.anns[aid_or_ann]
            else:
                for ann in self.dataset['annotations']:
                    if ann['id'] == aid_or_ann:
                        resolved_ann = ann
                        break
                if not resolved_ann:
                    raise IndexError(
                        'aid {} not in dataset'.format(aid_or_ann))
        else:
            resolved_ann = aid_or_ann
        return resolved_ann

    def _resolve_to_img(self, gid_or_img):
        """
        Ensures output is an image dictionary
        """
        if isinstance(gid_or_img, numbers.Integral):
            resolved_img = None
            if self.imgs is not None:
                resolved_img = self.imgs[gid_or_img]
            else:
                for img in self.dataset['images']:
                    if img['id'] == gid_or_img:
                        resolved_img = img
                        break
                if not resolved_img:
                    raise IndexError(
                        'gid {} not in dataset'.format(gid_or_img))
        else:
            resolved_img = gid_or_img
        return resolved_img

    def _resolve_to_kpcat(self, kp_identifier):
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
        if 'keypoint_categories' not in self.dataset:
            raise NotImplementedError('Must have newstyle keypoints to use')

        # TODO: add keypoint categories to the index and optimize
        if isinstance(kp_identifier, numbers.Integral):
            kpcat = None
            for _kpcat in self.dataset['keypoint_categories']:
                if _kpcat['id'] == kp_identifier:
                    kpcat = _kpcat
            if kpcat is None:
                raise KeyError('unable to find keypoint category')
        elif isinstance(kp_identifier, str):
            kpcat = None
            for _kpcat in self.dataset['keypoint_categories']:
                if _kpcat['name'] == kp_identifier:
                    kpcat = _kpcat
            if kpcat is None:
                for _kpcat in self.dataset['keypoint_categories']:
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

    def _resolve_to_cat(self, cat_identifier):
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
            if self.cats:
                cat = self.cats[cat_identifier]
            else:
                # If the index is not built
                found = None
                for cat in self.dataset['categories']:
                    if cat['id'] == cat_identifier:
                        found = cat
                        break
                if found is None:
                    raise KeyError(
                        'Cannot find a category with id={}'.format(cat_identifier))
        elif isinstance(cat_identifier, str):
            cat = self._alias_to_cat(cat_identifier)
        elif isinstance(cat_identifier, dict):
            cat = cat_identifier
        else:
            raise TypeError(type(cat_identifier))
        return cat

    def _alias_to_cat(self, alias_catname):
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
        if self.name_to_cat and alias_catname in self.name_to_cat:
            fixed_catname = alias_catname
            fixed_cat = self.name_to_cat[fixed_catname]
        else:
            # Try to find an alias
            fixed_catname = None
            fixed_cat = None
            for cat in self.dataset['categories']:
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

    def category_graph(self):
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
        for cat in self.dataset['categories']:
            graph.add_node(cat['name'], **cat)
            if cat.get('supercategory', None) is not None:
                u = cat['supercategory']
                v = cat['name']
                if u != v:
                    graph.add_edge(u, v)
        return graph

    def object_categories(self):
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

    def keypoint_categories(self):
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
            if self.index.kpcats is not None:
                kpcats = self.index.kpcats.values()
            else:
                kpcats = iter(self.dataset['keypoint_categories'])
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

    def _keypoint_category_names(self):
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
        kpcats = self.dataset.get('keypoint_categories', None)
        if kpcats is not None:
            return [c['name'] for c in self.dataset['keypoint_categories']]
        else:
            names = []
            cats = sorted(self.dataset['categories'], key=lambda c: c['id'])
            for cat in cats:
                if 'keypoints' in cat:
                    names.extend(cat['keypoints'])
            return names

    def _lookup_kpnames(self, cid):
        """ Get the keypoint categories for a certain class """
        kpnames = None
        orig_cat = self.cats[cid]
        while kpnames is None:
            # Extract keypoint names for each annotation
            cat = self.cats[cid]
            parent = cat.get('supercategory', None)
            if 'keypoints' in cat:
                kpnames = cat['keypoints']
            elif parent is not None:
                cid = self.name_to_cat[cat['supercategory']]['id']
            else:
                raise KeyError('could not find keypoint names for cid={}, cat={}, orig_cat={}'.format(cid, cat, orig_cat))
        return kpnames

    def _coco_image(self, gid):
        ub.schedule_deprecation(
            'kwcoco', '_coco_image', 'method',
            migration='Use "coco_image" instead.',
            deprecate='0.5.9', error='1.0.0', remove='1.1.0',
        )
        return self.coco_image(gid)

    def coco_image(self, gid):
        """
        Args:
            gid (int): image id

        Returns:
            kwcoco.coco_image.CocoImage
        """
        # Experimental
        from kwcoco.coco_image import CocoImage
        img = self.index.imgs[gid]
        image = CocoImage(img, dset=self)
        return image


class MixinCocoExtras(object):
    """
    Misc functions for coco
    """

    @classmethod
    def coerce(cls, key, sqlview=False, **kw):
        """
        Attempt to transform the input into the intended CocoDataset.

        Args:
            key : this can either be an instance of a CocoDataset, a
               string URI pointing to an on-disk dataset, or a special
               key for creating demodata.

            sqlview (bool | str):
                If truthy, will return the dataset as a cached sql view, which
                can be quicker to load and use in some instances. Can be given
                as a string, which sets the backend that is used: either sqlite
                or postgresql.  Defaults to False.

            **kw: passed to whatever constructor is chosen (if any)

        Returns:
            AbstractCocoDataset | kwcoco.CocoDataset | kwcoco.CocoSqlDatabase

        Example:
            >>> # test coerce for various input methods
            >>> import kwcoco
            >>> from kwcoco.coco_sql_dataset import assert_dsets_allclose
            >>> dct_dset = kwcoco.CocoDataset.coerce('special:shapes8')
            >>> copy1 = kwcoco.CocoDataset.coerce(dct_dset)
            >>> copy2 = kwcoco.CocoDataset.coerce(dct_dset.fpath)
            >>> assert assert_dsets_allclose(dct_dset, copy1)
            >>> assert assert_dsets_allclose(dct_dset, copy2)
            >>> # xdoctest: +REQUIRES(module:sqlalchemy)
            >>> sql_dset = dct_dset.view_sql()
            >>> copy3 = kwcoco.CocoDataset.coerce(sql_dset)
            >>> copy4 = kwcoco.CocoDataset.coerce(sql_dset.fpath)
            >>> assert assert_dsets_allclose(dct_dset, sql_dset)
            >>> assert assert_dsets_allclose(dct_dset, copy3)
            >>> assert assert_dsets_allclose(dct_dset, copy4)
        """
        import kwcoco
        if isinstance(key, cls):
            self = key
        if isinstance(key, os.PathLike):
            key = str(key)
        if isinstance(key, str):
            import uritools
            dset_fpath = ub.expandpath(key)
            # Parse the the "file" URI scheme
            # https://tools.ietf.org/html/rfc8089
            result = uritools.urisplit(dset_fpath)
            if result.scheme == 'sqlite' or result.path.endswith('.sqlite'):
                from kwcoco.coco_sql_dataset import CocoSqlDatabase
                self = CocoSqlDatabase(dset_fpath).connect()
            elif result.path.endswith('.json') or '.json' in result.path:
                if sqlview:
                    from kwcoco.coco_sql_dataset import CocoSqlDatabase
                    kw['backend'] = sqlview
                    self = CocoSqlDatabase.coerce(dset_fpath, **kw)
                else:
                    self = kwcoco.CocoDataset(dset_fpath, **kw)
            elif result.scheme == 'special':
                self = cls.demo(key=key, **kw)
            else:
                # This case can be env-dependant in the unlikely case where you
                # have a file with the same name as a demo key. But hey, you
                # are using the coerce function, be more explicit if you want
                # predictable behavior.
                if exists(dset_fpath):
                    self = kwcoco.CocoDataset(dset_fpath, **kw)
                else:
                    self = cls.demo(key=key, **kw)
        elif type(key).__name__ == 'CocoSqlDatabase':
            self = key
        elif type(key).__name__ == 'CocoDataset':
            self = key
        elif type(key).__name__ == 'CocoSampler':
            self = key.dset
        else:
            raise TypeError(type(key))
        return self

    @classmethod
    def demo(cls, key='photos', **kwargs):
        """
        Create a toy coco dataset for testing and demo puposes

        Args:
            key (str):
                Either 'photos' (default), 'shapes', or 'vidshapes'. There are
                also special sufixes that can control behavior.

                Basic options that define which flavor of demodata to generate
                are: `photos`, `shapes`, and `vidshapes`. A numeric suffix e.g.
                `vidshapes8` can be specified to indicate the size of the
                generated demo dataset.  There are other special suffixes that
                are available.  See the code in this function for explicit
                details on what is allowed.

                TODO: better documentation for these demo datasets.

                As a quick summary: the vidshapes key is the most robust and
                mature demodata set, and here are several useful variants of
                the vidshapes key.

                (1) vidshapes8 - the 8 suffix is the number of videos in this case.
                (2) vidshapes8-multispectral - generate 8 multispectral videos.
                (3) vidshapes8-msi - msi is an alias for multispectral.
                (4) vidshapes8-frames5 - generate 8 videos with 5 frames each.
                (5) vidshapes2-tracks5 - generate 2 videos with 5 tracks each.
                (6) vidshapes2-speed0.1-frames7 - generate 2 videos with 7
                frames where the objects move with with a speed of 0.1.

            **kwargs : if key is shapes, these arguments are passed to toydata
                generation. The Kwargs section of this docstring documents a
                subset of the available options. For full details, see
                :func:`demodata_toy_dset` and :func:`random_video_dset`.

        Kwargs:
            image_size (Tuple[int, int]): width / height size of the images

            dpath (str | PathLike):
                path to the directory where any generated demo bundles will be
                written to.  Defaults to using kwcoco cache dir.

            aux (bool): if True generates dummy auxiliary channels

            rng (int | RandomState | None):
                random number generator or seed

            verbose (int): verbosity mode. Defaults to 3.

        Example:
            >>> # Basic demodata keys
            >>> print(CocoDataset.demo('photos', verbose=1))
            >>> print(CocoDataset.demo('shapes', verbose=1))
            >>> print(CocoDataset.demo('vidshapes', verbose=1))
            >>> # Varaints of demodata keys
            >>> print(CocoDataset.demo('shapes8', verbose=0))
            >>> print(CocoDataset.demo('shapes8-msi', verbose=0))
            >>> print(CocoDataset.demo('shapes8-frames1-speed0.2-msi', verbose=0))

        Example:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes5', num_frames=5,
            >>>                                verbose=0, rng=None)
            >>> dset = kwcoco.CocoDataset.demo('vidshapes5', num_frames=5,
            >>>                                num_tracks=4, verbose=0, rng=44)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> pnums = kwplot.PlotNums(nSubplots=len(dset.index.imgs))
            >>> fnum = 1
            >>> for gx, gid in enumerate(dset.index.imgs.keys()):
            >>>     canvas = dset.draw_image(gid=gid)
            >>>     kwplot.imshow(canvas, pnum=pnums[gx], fnum=fnum)
            >>>     #dset.show_image(gid=gid, pnum=pnums[gx])
            >>> kwplot.show_if_requested()

        Example:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes5-aux', num_frames=1,
            >>>                                verbose=0, rng=None)

        Example:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes1-multispectral', num_frames=5,
            >>>                                verbose=0, rng=None)
            >>> # This is the first use-case of image names
            >>> assert len(dset.index.file_name_to_img) == 0, (
            >>>     'the multispectral demo case has no "base" image')
            >>> assert len(dset.index.name_to_img) == len(dset.index.imgs) == 5
            >>> dset.remove_images([1])
            >>> assert len(dset.index.name_to_img) == len(dset.index.imgs) == 4
            >>> dset.remove_videos([1])
            >>> assert len(dset.index.name_to_img) == len(dset.index.imgs) == 0
        """
        import parse
        kwargs.pop('autobuild', None)

        if key.startswith('special:'):
            key = key.split(':')[1]

        if key.startswith('shapes'):
            from kwcoco.demo import toydata_image
            res = parse.parse('shapes{num_imgs:d}', key)
            if res:
                kwargs['n_imgs'] = int(res.named['num_imgs'])
            if 'rng' not in kwargs and 'n_imgs' in kwargs:
                kwargs['rng'] = kwargs['n_imgs']
            self = toydata_image.demodata_toy_dset(**kwargs)
            self.tag = key
        elif key.startswith('vidshapes'):
            from kwcoco.demo import toydata_video
            verbose = kwargs.get('verbose', 1)
            res = parse.parse('vidshapes{num_videos:d}', key)
            if res is None:
                res = parse.parse('vidshapes{num_videos:d}-{suffix}', key)
            if res is None:
                res = parse.parse('vidshapes-{suffix}', key)
            """
            The rule is that the suffix will be split by the '-' character
            and any registered pattern or alias will impact the kwargs
            for random_video_dset
            """
            suff_parts = []
            if verbose > 3:
                print('res = {!r}'.format(res))
            if res:
                if 'num_videos' in res.named:
                    kwargs['num_videos'] = int(res.named['num_videos'])
                if 'suffix' in res.named:
                    suff_parts = [p for p in res.named['suffix'].split('-') if p]
            if verbose > 3:
                print('suff_parts = {!r}'.format(suff_parts))

            # The allowed suffix patterns and aliases are defined here
            vidkw = {
                'render': True,
                'num_videos': 1,
                'num_frames': 2,
                'num_tracks': 2,
                'anchors': None,
                'image_size': (600, 600),
                'aux': None,
                'multispectral': None,
                'multisensor': False,
                'max_speed': 0.01,
                'verbose': verbose,
            }
            vidkw_aliases = {
                'num_frames': {'frames'},
                'num_tracks': {'tracks'},
                'num_videos': {'videos'},
                'max_speed': {'speed'},
                'image_size': {'gsize'},
                'multispectral': {'msi'},
            }
            alias_to_key = {k: v for v, ks in vidkw_aliases.items() for k in ks}
            import re
            # These are the variables the vidshapes generator accepts
            for part in suff_parts:
                match = re.search(r'[\d]', part)
                if match is None:
                    value = True
                    key = part
                else:
                    key = part[:match.span()[0]]
                    value = part[match.span()[0]:]
                key = alias_to_key.get(key, key)
                if key == 'image_size':
                    value = int(value)
                if key == 'num_frames':
                    value = int(value)
                if key == 'num_tracks':
                    value = int(value)
                if key == 'num_videos':
                    value = int(value)
                if key == 'max_speed':
                    value = float(value)
                if key == 'multispectral':
                    value = bool(value)
                if key == 'multisensor':
                    value = bool(value)
                if key == 'render':
                    value = bool(value)

                # if key.startswith('rand'):
                #     pass
                if key in {'randgsize', 'randsize', 'image_sizerandom'}:
                    key = 'image_size'
                    value = 'random'

                if key == 'amazon':
                    # Hack to put in amazon background
                    key = 'render'
                    if isinstance(vidkw[key], dict):
                        value = vidkw[key]
                    else:
                        value = {}
                    value['background'] = 'amazon'

                vidkw[key] = value

            vidkw.update(kwargs)

            if isinstance(vidkw['image_size'], int):
                vidkw['image_size'] = (vidkw['image_size'], vidkw['image_size'])

            use_cache = vidkw.pop('use_cache', True)

            if 'rng' not in vidkw:
                # Make rng determined by other params by default
                vidkw['rng'] = int(ub.hash_data(sorted(vidkw.items()))[0:8], 16)

            depends_items = vidkw.copy()
            depends_items.pop('verbose', None)
            depends = ub.hash_data(sorted(depends_items.items()), base='abc')[0:14]

            if verbose > 0:
                print('vidkw = {}'.format(ub.urepr(vidkw, nl=1)))
            if verbose > 3:
                print('depends = {!r}'.format(depends))

            tag = key + '_' + depends
            dpath = vidkw.pop('dpath', None)
            fpath = vidkw.pop('fpath', None)
            bundle_dpath = vidkw.get('bundle_dpath', None)
            if fpath is not None:
                bundle_dpath = ub.Path(fpath).parent
            elif dpath is not None:
                bundle_dpath = dpath

            if bundle_dpath is None:
                # Even if the cache is off, we still will need this because it
                # will write rendered data to disk. Perhaps we can make this
                # optional in the future.
                dpath = ub.Path.appdir('kwcoco', 'demo_vidshapes').ensuredir()
                bundle_dpath = vidkw['bundle_dpath'] = join(dpath, tag)

            cache_dpath = join(bundle_dpath, '_cache')
            if fpath is None:
                fpath = join(bundle_dpath, 'data.kwcoco.json')

            stamp = ub.CacheStamp(
                'vidshape_stamp_v{:03d}'.format(toydata_video.TOYDATA_VIDEO_VERSION),
                dpath=cache_dpath,
                depends=depends, enabled=use_cache,
                product=[fpath], verbose=verbose,
                meta=vidkw
            )
            if verbose > 3:
                print('stamp = {!r}'.format(stamp))
            if stamp.expired():
                vidkw['dpath'] = bundle_dpath
                vidkw.pop('bundle_dpath', None)
                self = toydata_video.random_video_dset(**vidkw)
                if verbose > 3:
                    print('self.fpath = {!r}'.format(self.fpath))
                    print('self.bundle_dpath = {!r}'.format(self.bundle_dpath))

                self.fpath = fpath
                if fpath is not None:
                    ub.Path(fpath).parent.ensuredir()
                    self.dump(fpath, newlines=True)
                    stamp.renew()
            else:
                self = cls(data=fpath, bundle_dpath=bundle_dpath)

        elif key == 'photos':
            dataset = demo_coco_data()
            self = cls(dataset, tag=key)
        else:
            raise KeyError(key)
        return self

    def _tree(self):
        """
        developer helper

        Ignore:
            import kwcoco
            self = kwcoco.CocoDataset.demo('photos')
            print('self = {!r}'.format(self))
            self._tree()

            self = kwcoco.CocoDataset.demo('shapes1')
            print(self.imgs[1])
            print('self = {!r}'.format(self))
            self._tree()

            self = kwcoco.CocoDataset.demo('vidshapes2')
            print('self = {!r}'.format(self))
            print(self.imgs[1])
            self._tree()
        """
        print('self.bundle_dpath = {!r}'.format(self.bundle_dpath))
        print('self.fpath = {!r}'.format(self.fpath))
        print('self.tag = {!r}'.format(self.tag))
        if not self.bundle_dpath:
            raise Exception('We dont have a bundle')
        _ = ub.cmd('tree {}'.format(self.bundle_dpath), verbose=2)  # NOQA

    @classmethod
    def random(cls, rng=None):
        """
        Creates a random CocoDataset according to distribution parameters

        TODO:
            - [ ] parametarize
        """
        from kwcoco.demo.toydata_video import random_single_video_dset
        dset = random_single_video_dset(num_frames=5, num_tracks=3,
                                        tid_start=1, rng=rng)
        return dset

    def _build_hashid(self, hash_pixels=False, verbose=0):
        """
        Construct a hash that uniquely identifies the state of this dataset.

        Args:
            hash_pixels (bool):
                If False the image data is not included in the hash, which can
                speed up computation, but is not 100% robust. Defaults to
                False.

            verbose (int): verbosity level

        Returns:
            str: the hashid

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> self._build_hashid(hash_pixels=True, verbose=3)
            ...
            >>> # Shorten hashes for readability
            >>> import ubelt as ub
            >>> walker = ub.IndexableWalker(self.hashid_parts)
            >>> for path, val in walker:
            >>>     if isinstance(val, str):
            >>>         walker[path] = val[0:8]
            >>> # Note: this may change in different versions of kwcoco
            >>> print('self.hashid_parts = ' + ub.urepr(self.hashid_parts))
            >>> print('self.hashid = {!r}'.format(self.hashid[0:8]))
            self.hashid_parts = {
                'annotations': {
                    'json': 'c1d1b9c3',
                    'num': 11,
                },
                'images': {
                    'pixels': '88e37cc3',
                    'json': '9b8e8be3',
                    'num': 3,
                },
                'categories': {
                    'json': '82d22e00',
                    'num': 8,
                },
            }
            self.hashid = 'bf69bf15'

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> self._build_hashid(hash_pixels=True, verbose=3)
            >>> self.hashid_parts
            >>> # Test that when we modify the dataset only the relevant
            >>> # hashid parts are recomputed.
            >>> orig = self.hashid_parts['categories']['json']
            >>> self.add_category('foobar')
            >>> assert 'categories' not in self.hashid_parts
            >>> self.hashid_parts
            >>> self.hashid_parts['images']['json'] = 'should not change'
            >>> self._build_hashid(hash_pixels=True, verbose=3)
            >>> assert self.hashid_parts['categories']['json']
            >>> assert self.hashid_parts['categories']['json'] != orig
            >>> assert self.hashid_parts['images']['json'] == 'should not change'
        """
        # Construct nested container that we will populate with hashable
        # info corresponding to each type of data that we track.
        hashid_parts = self.hashid_parts
        if hashid_parts is None:
            hashid_parts = OrderedDict()

        # TODO: hashid for videos? Maybe?

        # Ensure hashid_parts has the proper root structure
        # TODO: rely on subet of SPEC_KEYS
        parts = ['annotations', 'images', 'categories']
        for part in parts:
            if not hashid_parts.get(part, None):
                hashid_parts[part] = OrderedDict()

        rebuild_parts = []
        reuse_parts = []

        gids = None
        if hash_pixels:
            if not hashid_parts['images'].get('pixels', None):
                gids = sorted(self.imgs.keys())
                gpaths = [join(self.bundle_dpath, gname)
                          for gname in self.images(gids).lookup('file_name')]
                gpath_sha512s = [
                    ub.hash_file(gpath, hasher='sha512')
                    for gpath in ub.ProgIter(gpaths, desc='hashing images',
                                             verbose=verbose)
                ]
                hashid_parts['images']['pixels'] = ub.hash_data(gpath_sha512s)
                rebuild_parts.append('images.pixels')
            else:
                reuse_parts.append('images.pixels')

        # Hash individual components
        with ub.Timer(label='hash coco parts', verbose=verbose > 1):
            # Dumping annots to json takes the longest amount of time
            # However, its faster than hashing the data directly
            def _ditems(d):
                # return sorted(d.items())
                return list(d.items()) if isinstance(d, OrderedDict) else sorted(d.items())

            if not hashid_parts['annotations'].get('json', None):
                aids = sorted(self.anns.keys())
                _anns_ordered = (self.anns[aid] for aid in aids)
                anns_ordered = [_ditems(ann) for ann in _anns_ordered]
                try:
                    anns_text = json_w.dumps(anns_ordered)
                except TypeError:
                    if __debug__:
                        for ann in anns_ordered:
                            try:
                                json_w.dumps(ann)
                            except TypeError:
                                print('FAILED TO ENCODE ann = {!r}'.format(ann))
                                break
                    raise

                hashid_parts['annotations']['json'] = ub.hash_data(
                    anns_text, hasher='sha512')
                hashid_parts['annotations']['num'] = len(aids)
                rebuild_parts.append('annotations.json')
            else:
                reuse_parts.append('annotations.json')

            if not hashid_parts['images'].get('json', None):
                if gids is None:
                    gids = sorted(self.imgs.keys())
                imgs_text = json_w.dumps(
                    [_ditems(self.imgs[gid]) for gid in gids])
                hashid_parts['images']['json'] = ub.hash_data(
                    imgs_text, hasher='sha512')
                hashid_parts['images']['num'] = len(gids)
                rebuild_parts.append('images.json')
            else:
                reuse_parts.append('images.json')

            if not hashid_parts['categories'].get('json', None):
                cids = sorted(self.cats.keys())
                cats_text = json_w.dumps(
                    [_ditems(self.cats[cid]) for cid in cids])
                hashid_parts['categories']['json'] = ub.hash_data(
                    cats_text, hasher='sha512')
                hashid_parts['categories']['num'] = len(cids)
                rebuild_parts.append('categories.json')
            else:
                reuse_parts.append('categories.json')

        if verbose > 1:
            if reuse_parts:
                print('Reused hashid_parts: {}'.format(reuse_parts))
                print('Rebuilt hashid_parts: {}'.format(rebuild_parts))

        hashid = ub.hash_data(hashid_parts)
        self.hashid = hashid
        self.hashid_parts = hashid_parts
        return hashid

    def _invalidate_hashid(self, parts=None):
        """
        Called whenever the coco dataset is modified. It is possible to specify
        which parts were modified so unmodified parts can be reused next time
        the hash is constructed.

        TODO:
            - [ ] Rename to _notify_modification --- or something like that
        """
        self._state['was_modified'] = True
        self.hashid = None
        if parts is not None and self.hashid_parts is not None:
            for part in parts:
                self.hashid_parts.pop(part, None)
        else:
            self.hashid_parts = None

    def _cached_hashid(self):
        """
        Under Construction.

        The idea is to cache the hashid when we are sure that the dataset was
        loaded from a file and has not been modified.  We can record the
        modification time of the file (because we know it hasn't changed in
        memory), and use that as a key to the cache. If the modification time
        on the file is different than the one recorded in the cache, we know
        the cache could be invalid, so we recompute the hashid.
        """
        cache_miss = True
        enable_cache = (
            self._state['was_loaded'] and
            not self._state['was_modified']
        )
        if enable_cache:
            coco_fpath = ub.Path(self.fpath)
            enable_cache = coco_fpath.exists()

        if enable_cache:
            cache_dpath = (coco_fpath.parent / '_cache')
            cache_fname = coco_fpath.name + '.hashid.cache'
            hashid_sidecar_fpath = cache_dpath / cache_fname
            # Generate current lookup key
            fpath_stat = coco_fpath.stat()
            status_key = {
                'st_size': fpath_stat.st_size,
                'st_mtime': fpath_stat.st_mtime
            }
            if hashid_sidecar_fpath.exists():
                cached_data = json_r.loads(hashid_sidecar_fpath.read_text())
                if cached_data['status_key'] == status_key:
                    self.hashid = cached_data['hashid']
                    self.hashid_parts = cached_data['hashid_parts']
                    cache_miss = False

        if cache_miss:
            self._build_hashid()
            if enable_cache:
                hashid_cache_data = {
                    'hashid': self.hashid,
                    'hashid_parts': self.hashid_parts,
                    'status_key': status_key,
                }
                try:
                    hashid_sidecar_fpath.parent.ensuredir()
                    hashid_sidecar_fpath.write_text(json_w.dumps(hashid_cache_data))
                except PermissionError as ex:
                    warnings.warn(f'Cannot write a cached hashid: {repr(ex)}')
        return self.hashid

    @classmethod
    def _cached_hashid_for(cls, fpath):
        """
        Lookup the cached hashid for a kwcoco json file if it exists.
        """
        coco_fpath = ub.Path(fpath)
        enable_cache = coco_fpath.exists()

        if enable_cache:
            cache_dpath = (coco_fpath.parent / '_cache')
            cache_fname = coco_fpath.name + '.hashid.cache'
            hashid_sidecar_fpath = cache_dpath / cache_fname
            # Generate current lookup key
            fpath_stat = coco_fpath.stat()
            status_key = {
                'st_size': fpath_stat.st_size,
                'st_mtime': fpath_stat.st_mtime
            }
            if hashid_sidecar_fpath.exists():
                cached_data = json_r.loads(hashid_sidecar_fpath.read_text())
                if cached_data['status_key'] == status_key:
                    cached_data['hashid']
                    cached_data['hashid_parts']
                    cache_miss = False

        if cache_miss:
            raise Exception("cache miss")
            # TODO: fixme?
            # self._build_hashid()
            # if enable_cache:
            #     hashid_cache_data = {
            #         'hashid': self.hashid,
            #         'hashid_parts': self.hashid_parts,
            #         'status_key': status_key,
            #     }
            #     hashid_sidecar_fpath.parent.ensuredir()
            #     hashid_sidecar_fpath.write_text(json_w.dumps(hashid_cache_data))

    def _dataset_id(self):
        """
        A human interpretable name that can be used to uniquely identify the
        dataset.

        Note:
            This function is currently subject to change.

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> print(self._dataset_id())
            >>> self = kwcoco.CocoDataset.demo('vidshapes8')
            >>> print(self._dataset_id())
            >>> self = kwcoco.CocoDataset()
            >>> print(self._dataset_id())
        """
        hashid = self._cached_hashid()
        if self.fpath is None:
            raise Exception('Dataset doesnt have a fpath')
        coco_fpath = ub.Path(self.fpath)
        dset_id = '_'.join([coco_fpath.parent.stem, coco_fpath.stem, hashid[0:8]])
        return dset_id

    def _ensure_imgsize(self, workers=0, verbose=1, fail=False):
        """
        Populate the imgsize field if it does not exist.

        Args:
            workers (int): number of workers for parallel
                processing.

            verbose (int): verbosity level

            fail (bool): if True, raises an exception if
               anything size fails to load.

        Returns:
            List[dict]: a list of "bad" image dictionaries where the size could
                not be determined. Typically these are corrupted images and
                should be removed.

        Example:
            >>> # Normal case
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> bad_imgs = self._ensure_imgsize()
            >>> assert len(bad_imgs) == 0
            >>> assert self.imgs[1]['width'] == 512
            >>> assert self.imgs[2]['width'] == 328
            >>> assert self.imgs[3]['width'] == 256

            >>> # Fail cases
            >>> self = kwcoco.CocoDataset()
            >>> self.add_image('does-not-exist.jpg')
            >>> bad_imgs = self._ensure_imgsize()
            >>> assert len(bad_imgs) == 1
            >>> import pytest
            >>> with pytest.raises(Exception):
            >>>     self._ensure_imgsize(fail=True)
        """
        bad_images = []
        if any('width' not in img or 'height' not in img
               for img in self.dataset['images']):
            import kwimage
            if self.tag:
                desc = 'populate imgsize for ' + self.tag
            else:
                desc = 'populate imgsize for untagged coco dataset'

            pool = ub.JobPool('thread', max_workers=workers)
            bundle_dpath = self.bundle_dpath
            for img in ub.ProgIter(self.dataset['images'], verbose=verbose,
                                   desc='submit image size jobs'):
                auxiliary = img.get('auxiliary', img.get('assets', []))
                for obj in [img] + auxiliary:
                    fname = obj['file_name']
                    if fname is not None:
                        gpath = join(bundle_dpath, fname)
                        if 'width' not in obj or 'height' not in obj:
                            job = pool.submit(kwimage.load_image_shape, gpath)
                            job.obj = obj

            for job in pool.as_completed(desc=desc, progkw={'verbose': verbose}):
                try:
                    h, w = job.result()[0:2]
                except Exception:
                    if fail:
                        raise
                    bad_images.append(job.obj)
                else:
                    job.obj['width'] = w
                    job.obj['height'] = h
        return bad_images

    def _ensure_image_data(self, gids=None, verbose=1):
        """
        Download data from "url" fields if specified.

        Args:
            gids (List): subset of images to download
        """
        def _gen_missing_imgs():
            for img in self.dataset['images']:
                gpath = join(self.bundle_dpath, img['file_name'])
                if not exists(gpath):
                    yield img

        def _has_download_permission(_HAS_PREMISSION=[False]):
            if not _HAS_PREMISSION[0] or ub.argflag(('-y', '--yes')):
                try:
                    from rich.prompt import Confirm
                    ans = Confirm.ask('is it ok to download?')
                    if ans:
                        _HAS_PREMISSION[0] = True
                except ImportError:
                    ans = input('is it ok to download? (enter y for yes)')
                    if ans in ['yes', 'y']:
                        _HAS_PREMISSION[0] = True
            return _HAS_PREMISSION[0]

        if gids is None:
            gids = _gen_missing_imgs()

        for img in ub.ProgIter(gids, desc='ensure image data'):
            if 'url' in img:
                if _has_download_permission():
                    gpath = join(self.bundle_dpath, img['file_name'])
                    ub.ensuredir(dirname(gpath))
                    ub.grabdata(img['url'], gpath)
                else:
                    raise Exception('no permission, abort')
            else:
                raise Exception('missing image, but no url')

    def missing_images(self, check_aux=True, verbose=0):
        """
        Check for images that don't exist

        Args:
            check_aux (bool):
                if specified also checks auxiliary images

            verbose (int): verbosity level

        Returns:
            List[Tuple[int, str, int]]: bad indexes and paths and ids
        """
        bad_paths = []
        img_enum = enumerate(self.dataset['images'])
        for index, img in ub.ProgIter(img_enum,
                                      total=len(self.dataset['images']),
                                      desc='check for missing images',
                                      verbose=verbose):
            gid = img.get('id', None)
            fname = img.get('file_name', None)
            if fname is not None:
                gpath = join(self.bundle_dpath, fname)
                if not exists(gpath):
                    bad_paths.append((index, gpath, gid))

            if check_aux:
                for aux in img.get('auxiliary', img.get('assets', [])):
                    gpath = join(self.bundle_dpath, aux['file_name'])
                    if not exists(gpath):
                        bad_paths.append((index, gpath, gid))
        return bad_paths

    def corrupted_images(self, check_aux=True, verbose=0, workers=0):
        """
        Check for images that don't exist or can't be opened

        Args:
            check_aux (bool):
                if specified also checks auxiliary images
            verbose (int): verbosity level
            workers (int): number of background workers

        Returns:
            List[Tuple[int, str, int]]: bad indexes and paths and ids
        """
        bad_paths = []

        jobs = ub.JobPool(mode='process', max_workers=workers)

        img_enum = enumerate(self.dataset['images'])
        for index, img in ub.ProgIter(img_enum,
                                      total=len(self.dataset['images']),
                                      desc='submit corruption checks',
                                      verbose=verbose):
            gid = img.get('id', None)
            fname = img.get('file_name', None)
            if fname is not None:
                gpath = join(self.bundle_dpath, fname)
                job = jobs.submit(_image_corruption_check, gpath)
                job.input_info = (index, gpath, gid)

            if check_aux:
                for aux in img.get('auxiliary', img.get('assets', [])):
                    gpath = join(self.bundle_dpath, aux['file_name'])
                    job = jobs.submit(_image_corruption_check, gpath)
                    job.input_info = (index, gpath, gid)

        for job in jobs.as_completed(desc='check corrupted images',
                                     progkw={'verbose': verbose}):
            info = job.result()
            if info['failed']:
                bad_paths.append(job.input_info)

        return bad_paths

    def rename_categories(self, mapper, rebuild=True, merge_policy='ignore'):
        """
        Rename categories with a potentially coarser categorization.

        Args:
            mapper (dict | Callable): maps old names to new names.
                If multiple names are mapped to the same category, those
                categories will be merged.

            merge_policy (str):
                How to handle multiple categories that map to the same name.
                Can be update or ignore.

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> self.rename_categories({'astronomer': 'person',
            >>>                         'astronaut': 'person',
            >>>                         'mouth': 'person',
            >>>                         'helmet': 'hat'})
            >>> assert 'hat' in self.name_to_cat
            >>> assert 'helmet' not in self.name_to_cat
            >>> # Test merge case
            >>> self = kwcoco.CocoDataset.demo()
            >>> mapper = {
            >>>     'helmet': 'rocket',
            >>>     'astronomer': 'rocket',
            >>>     'human': 'rocket',
            >>>     'mouth': 'helmet',
            >>>     'star': 'gas'
            >>> }
            >>> self.rename_categories(mapper)
        """
        old_cats = self.dataset['categories']

        # Ignore identity mappings
        mapper = {k: v for k, v in mapper.items() if k != v}

        # Perform checks to determine what bookkeeping needs to be done
        orig_cnames = {cat['name'] for cat in old_cats}
        src_cnames = set(mapper.keys())
        dst_cnames = set(mapper.values())

        bad_cnames = src_cnames - orig_cnames
        if bad_cnames:
            raise ValueError(
                'The following categories to not exist: {}'.format(bad_cnames))

        has_orig_merges = dst_cnames.intersection(orig_cnames)
        has_src_merges = dst_cnames.intersection(src_cnames)
        has_dup_dst = set(ub.find_duplicates(mapper.values()).keys())

        has_merges = has_orig_merges or has_src_merges or has_dup_dst

        if not has_merges:
            # In the simple case we are just changing the labels, so
            # nothing special needs to happen.
            for key, value in mapper.items():
                for cat in self.dataset['categories']:
                    if cat['name'] == key:
                        cat['name'] = value
        else:
            # Remember the original categories
            orig_cats = {cat['name']: cat for cat in old_cats}

            # Remember all annotations of the original categories
            src_cids = [self.index.name_to_cat[c]['id']
                        for c in src_cnames]
            src_cname_to_aids = {c: self.index.cid_to_aids[cid]
                                 for c, cid in zip(src_cnames, src_cids)}

            # Track which srcs each dst is constructed from
            dst_to_srcs = ub.invert_dict(mapper, unique_vals=False)

            # Mark unreferenced cats for removal
            rm_cnames = src_cnames - dst_cnames

            # Mark unseen cats for addition
            add_cnames = dst_cnames - orig_cnames

            # Mark renamed cats for update
            update_cnames = dst_cnames - add_cnames

            # Populate new category information
            new_cats = {}
            for dst in sorted(dst_cnames):
                # Combine information between existing categories that are
                # being collapsed into a single category.
                srcs = dst_to_srcs[dst]
                new_cat = {}
                if len(srcs) <= 1:
                    # in the case of 1 source, then there is no merger
                    for src in srcs:
                        new_cat.update(orig_cats[src])
                elif merge_policy == 'update':
                    # When there are multiple sources then union all
                    # original source attributes.
                    # Note: the order of srcs is arbitrary so we sort
                    # to make the update order consistent
                    for src in sorted(srcs):
                        new_cat.update(orig_cats[src])
                elif merge_policy == 'ignore':
                    # When there are multiple sources then ignore all
                    # original source attributes.
                    pass
                else:
                    # There may be better merge policies that should be
                    # implemented
                    raise KeyError('Unknown merge_policy={}'.format(
                        merge_policy))

                new_cat['name'] = dst
                new_cat.pop('id', None)
                new_cats[dst] = new_cat

            # Apply category deltas

            # Remove category names that were renamed
            self.remove_categories(rm_cnames, keep_annots=True)

            # Update any existing categories that were merged
            for cname in sorted(update_cnames):
                if merge_policy == 'ignore':
                    new_cats[cname] = dict(orig_cats[cname])
                elif merge_policy == 'update':
                    new_cat = new_cats[cname]
                    orig_cat = orig_cats[cname]
                    # Only update name and non-existing information
                    # TODO: check for conflicts?
                    new_info = ub.dict_diff(new_cat, orig_cat)
                    orig_cat.update(new_info)
                else:
                    raise KeyError('Unknown merge_policy={}'.format(
                        merge_policy))

            # Add new category names that were created
            for cname in sorted(add_cnames):
                self.add_category(**new_cats[cname])

            # Apply the annotation deltas
            for src, aids in src_cname_to_aids.items():
                cat = self.name_to_cat[mapper[src]]
                cid = cat['id']
                for aid in aids:
                    self.anns[aid]['category_id'] = cid

            # force rebuild if any annotations we changed
            if src_cname_to_aids:
                rebuild = True

        if rebuild:
            self._build_index()
        else:
            self.index.clear()
        self._invalidate_hashid()

    def _ensure_json_serializable(self):
        # inplace convert any ndarrays to lists
        from kwcoco.util.util_json import ensure_json_serializable
        _ = ensure_json_serializable(self.dataset, verbose=1)

    def _aspycoco(self):
        """
        Converts to the official pycocotools.coco.COCO object

        TODO:
            - [ ] Maybe expose as a public API?
        """
        from pycocotools import coco
        pycoco = coco.COCO()
        pycoco.dataset = self.dataset
        pycoco.createIndex()
        return pycoco

    def reroot(self, new_root=None,
               old_prefix=None,
               new_prefix=None,
               absolute=False,
               check=True,
               safe=True,
               verbose=1):
        """
        Modify the prefix of the image/data paths onto a new image/data root.

        Args:
            new_root (str | PathLike | None):
                New image root. If unspecified the current
                ``self.bundle_dpath`` is used. If old_prefix and new_prefix are
                unspecified, they will attempt to be determined based on the
                current root (which assumes the file paths exist at that root)
                and this new root.  Defaults to None.

            old_prefix (str | None):
                If specified, removes this prefix from file names.
                This also prevents any inferences that might be made via
                "new_root". Defaults to None.

            new_prefix (str | None):
                If specified, adds this prefix to the file names.
                This also prevents any inferences that might be made via
                "new_root". Defaults to None.

            absolute (bool):
                if True, file names are stored as absolute paths, otherwise
                they are relative to the new image root. Defaults to False.

            check (bool):
                if True, checks that the images all exist.
                Defaults to True.

            safe (bool):
                if True, does not overwrite values until all checks pass.
                Defaults to True.

            verbose (int):
                verbosity level, default=0.

        CommandLine:
            xdoctest -m kwcoco.coco_dataset MixinCocoExtras.reroot

        TODO:
            - [ ] Incorporate maximum ordered subtree embedding?

        Example:
            >>> # xdoctest: +REQUIRES(module:rich)
            >>> import kwcoco
            >>> import ubelt as ub
            >>> import rich
            >>> def report(dset):
            >>>     gid = 1
            >>>     abs_fpath = ub.Path(dset.get_image_fpath(gid))
            >>>     rel_fpath = dset.index.imgs[gid]['file_name']
            >>>     color = 'green' if abs_fpath.exists() else 'red'
            >>>     print(ub.color_text(f'abs_fpath = {abs_fpath!r}', color))
            >>>     print(f'rel_fpath = {rel_fpath!r}')
            >>> dset = self = kwcoco.CocoDataset.demo()
            >>> # Change base relative directory
            >>> bundle_dpath = ub.expandpath('~')
            >>> rich.print('ORIG self.imgs = {}'.format(ub.urepr(self.imgs, nl=1)))
            >>> rich.print('ORIG dset.bundle_dpath = {!r}'.format(dset.bundle_dpath))
            >>> rich.print('NEW(1) bundle_dpath       = {!r}'.format(bundle_dpath))
            >>> # Test relative reroot
            >>> rich.print('[blue] --- 1. RELATIVE REROOT ---')
            >>> self.reroot(bundle_dpath, verbose=3)
            >>> report(self)
            >>> rich.print('NEW(1) self.imgs = {}'.format(ub.urepr(self.imgs, nl=1)))
            >>> if not ub.WIN32:
            >>>     assert self.imgs[1]['file_name'].startswith('.cache')
            >>> # Test absolute reroot
            >>> rich.print('[blue] --- 2. ABSOLUTE REROOT ---')
            >>> self.reroot(absolute=True, verbose=3)
            >>> rich.print('NEW(2) self.imgs = {}'.format(ub.urepr(self.imgs, nl=1)))
            >>> assert self.imgs[1]['file_name'].startswith(bundle_dpath)

            >>> # Switch back to relative paths
            >>> rich.print('[blue] --- 3. ABS->REL REROOT ---')
            >>> self.reroot()
            >>> rich.print('NEW(3) self.imgs = {}'.format(ub.urepr(self.imgs, nl=1)))
            >>> if not ub.WIN32:
            >>>     assert self.imgs[1]['file_name'].startswith('.cache')

        Example:
            >>> # demo with auxiliary data
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('shapes8', aux=True)
            >>> bundle_dpath = ub.expandpath('~')
            >>> print(self.imgs[1]['file_name'])
            >>> print(self.imgs[1]['auxiliary'][0]['file_name'])
            >>> self.reroot(new_root=bundle_dpath)
            >>> print(self.imgs[1]['file_name'])
            >>> print(self.imgs[1]['auxiliary'][0]['file_name'])
            >>> if not ub.WIN32:
            >>>     assert self.imgs[1]['file_name'].startswith('.cache')
            >>>     assert self.imgs[1]['auxiliary'][0]['file_name'].startswith('.cache')

        Ignore:
            See ~/code/kwcoco/dev/devcheck/devcheck_reroot.py
        """
        cur_bundle_dpath = self.bundle_dpath
        if cur_bundle_dpath == '':
            cur_bundle_dpath = '.'
        new_bundle_dpath = self.bundle_dpath if new_root is None else new_root
        new_bundle_dpath = os.fspath(new_bundle_dpath)

        # Find the the prefix that to make new relative paths work.
        rel_prefix = os.path.relpath(cur_bundle_dpath, new_bundle_dpath)

        if absolute:
            new_bundle_dpath = os.fspath(ub.Path(new_bundle_dpath).absolute())

        if verbose > 1:
            print('kwcoco.reroot')
            print(' * new_bundle_dpath = {}'.format(ub.urepr(new_bundle_dpath, nl=1)))
            print(' * cur_bundle_dpath = {}'.format(ub.urepr(cur_bundle_dpath, nl=1)))
            print(f' * old_prefix={old_prefix}')
            print(f' * new_prefix={new_prefix}')
            print(f' * absolute={absolute}')
            print(f' * safe={safe}')
            print(f' * check={check}')
            if verbose > 2:
                # First assets
                if len(self.dataset['images']):
                    from kwcoco.coco_image import CocoImage
                    first_image = CocoImage(self.dataset['images'][0], dset=self)
                    first_image_filepaths = list(first_image.iter_image_filepaths(with_bundle=False))
                    print(' * first_image_filepaths = {}'.format(ub.urepr(first_image_filepaths, nl=1)))
                else:
                    print('No images to reroot')

        def _normalize_prefix(path):
            # removes leading ./ on paths
            path = os.fspath(path)
            while path.startswith('./'):
                path = path[2:]
            return path

        def _reroot_path(file_name):
            """ Reroot a single file """

            # TODO: can this logic be cleaned up, its difficult to follow and
            # describe. The gist is we want reroot to modify the file names of
            # the images based on the new root, unless we are explicitly given
            # information about new and old prefixes. Can we do better than
            # this?
            _old_fname = file_name
            file_name = _normalize_prefix(file_name)
            cur_gpath = join(cur_bundle_dpath, file_name)

            if old_prefix is not None:
                if file_name.startswith(old_prefix):
                    file_name = relpath(file_name, old_prefix)
                elif cur_gpath.startswith(old_prefix):
                    file_name = relpath(cur_gpath, old_prefix)

            # If we don't specify new or old prefixes, but new_root was
            # specified then modify relative file names to be correct with
            # respect to this new root (assuming the previous root was also
            # valid)
            DO_OLD_CHECK = 0
            if DO_OLD_CHECK:
                if old_prefix is None and new_prefix is None:
                    # This is not a good check, fails if we want to
                    # do a relative reroot outside of the original dataset
                    if new_bundle_dpath is not None and cur_gpath.startswith(new_bundle_dpath):
                        file_name = relpath(cur_gpath, new_bundle_dpath)

            if new_prefix is not None:
                file_name = join(new_prefix, file_name)

            if absolute:
                new_file_name = join(new_bundle_dpath, file_name)
            else:
                new_file_name = join(rel_prefix, file_name)
                if os.path.isabs(new_file_name):
                    # Handle absolute -> relative case
                    new_file_name = relpath(new_file_name, new_bundle_dpath)

            if check:
                new_gpath = join(new_bundle_dpath, new_file_name)
                if not exists(new_gpath):
                    print('ERROR')
                    print(' * file_name (old) = {!r}'.format(_old_fname))
                    print(' * file_name (new) = {!r}'.format(file_name))
                    print(' * cur_gpath = {!r}'.format(cur_gpath))
                    print(' * new_gpath = {!r}'.format(new_gpath))
                    print(' * new_file_name = {!r}'.format(new_file_name))
                    print(' * cur_bundle_dpath = {!r}'.format(cur_bundle_dpath))
                    print(' * new_bundle_dpath = {!r}'.format(new_bundle_dpath))
                    raise Exception(
                        'Image does not exist: {!r}'.format(new_gpath))
            return new_file_name

        # from kwcoco.util import util_reroot

        num_images = len(self.imgs)
        enable_prog = verbose > 0

        if safe:
            # First compute all new values in memory but don't overwrite
            gid_to_new = {}
            prog = ub.ProgIter(self.imgs.items(), total=num_images, enabled=enable_prog, desc='prepare reroot')
            for gid, img in prog:
                gname = img.get('file_name', None)
                try:
                    new = {}
                    if gname is not None:
                        new['file_name'] = _reroot_path(gname)
                    if 'auxiliary' in img:
                        new['auxiliary'] = aux_fname = []
                        for aux in img.get('auxiliary', []):
                            aux_fname.append(_reroot_path(aux['file_name']))
                    if 'assets' in img:
                        new['assets'] = aux_fname = []
                        for aux in img.get('assets', []):
                            aux_fname.append(_reroot_path(aux['file_name']))
                    gid_to_new[gid] = new
                except Exception:
                    # img_repr = ub.urepr(img)
                    asset_fpaths = list(self.coco_image(img['id']).iter_image_filepaths())
                    raise Exception('Failed to reroot gid={} with fpaths={}'.format(img['id'], asset_fpaths))

            # Overwrite old values
            prog = ub.ProgIter(gid_to_new.items(), total=num_images, enabled=enable_prog, desc='finalize reroot')
            for gid, new in prog:
                img = self.imgs[gid]
                img['file_name'] = new.get('file_name', None)
                if 'auxiliary' in new:
                    for aux_fname, aux in zip(new['auxiliary'], img['auxiliary']):
                        aux['file_name'] = aux_fname
                if 'assets' in new:
                    for aux_fname, aux in zip(new['assets'], img['assets']):
                        aux['file_name'] = aux_fname
        else:
            prog = ub.ProgIter(self.imgs.values(), total=num_images, enabled=enable_prog, desc='rerooting')
            for img in prog:
                try:
                    gname = img.get('file_name', None)
                    if gname is not None:
                        img['file_name'] = _reroot_path(gname)
                    for aux in img.get('auxiliary', []):
                        aux['file_name'] = _reroot_path(aux['file_name'])
                    for aux in img.get('assets', []):
                        aux['file_name'] = _reroot_path(aux['file_name'])
                except Exception:
                    # img_repr = ub.urepr(img)
                    # raise Exception('Failed to reroot img={}'.format(ub.urepr(img)))
                    asset_fpaths = list(self.coco_image(img['id']).iter_image_filepaths())
                    raise Exception('Failed to reroot gid={} with fpaths={}'.format(img['id'], asset_fpaths))

        if self.index:
            # Only need to recompute the self.index.file_name_to_img
            # We dont have to invalidate everything
            # FIXME: the index should have some method for doing this
            # (ideally lazilly)
            self.index.file_name_to_img = {
                img['file_name']: img for img in self.index.imgs.values()
                if img.get('file_name', None) is not None
            }

        self.bundle_dpath = new_bundle_dpath
        return self

    @property
    def data_root(self):
        """ In the future we will deprecate data_root for bundle_dpath """
        return self.bundle_dpath

    @data_root.setter
    def data_root(self, value):
        self.bundle_dpath = value if value is None else os.fspath(value)

    @property
    def img_root(self):
        """ In the future we will deprecate img_root for bundle_dpath """
        return self.bundle_dpath

    @img_root.setter
    def img_root(self, value):
        self.bundle_dpath = value

    @property
    def data_fpath(self):
        """ data_fpath is an alias of fpath """
        return self.fpath

    @data_fpath.setter
    def data_fpath(self, value):
        self.fpath = value


class MixinCocoObjects(object):
    """
    Expose methods to construct object lists / groups.

    This is an alternative vectorized ORM-like interface to the coco dataset
    """

    def annots(self, annot_ids=None, image_id=None, track_id=None, trackid=None, aids=None, gid=None):
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
                'kwcoco', 'trackid', 'argument of CocoDataset.annots',
                migration='Use "track_id" instead.',
                deprecate='0.5.9', error='1.0.0', remove='1.1.0',
            )
            track_id = trackid

        if image_id is not None:
            annot_ids = sorted(self.index.gid_to_aids[image_id])

        if track_id is not None:
            annot_ids = self.index.trackid_to_aids[track_id]

        if annot_ids is None:
            annot_ids = sorted(self.index.anns.keys())

        return Annots(annot_ids, self)

    def images(self, image_ids=None, video_id=None, names=None, gids=None, vidid=None):
        """
        Return vectorized image objects

        Args:
            image_ids (List[int] | None): image ids to reference, if unspecified
                 all images are returned. An alias is `gids`.

            video_id (int | None): returns all images that belong to this video id.
                mutually exclusive with `image_ids` arg.

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
                'kwcoco', 'vidid', 'argument of CocoDataset.images',
                migration='Use "video_id" instead.',
                deprecate='0.5.0', error='1.0.0', remove='1.1.0',
            )
            video_id = vidid

        if video_id is not None:
            image_ids = self.index.vidid_to_gids[video_id]

        if names is not None:
            image_ids = [self.index.name_to_img[name]['id'] for name in names]

        if image_ids is None:
            image_ids = sorted(self.index.imgs.keys())

        return Images(image_ids, self)

    def categories(self, category_ids=None, cids=None):
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
        if category_ids is None:
            category_ids = sorted(self.index.cats.keys())
        return Categories(category_ids, self)

    def videos(self, video_ids=None, names=None, vidids=None):
        """
        Return vectorized video objects

        Args:
            video_ids (List[int] | None):
                video ids to reference, if unspecified all videos are returned.
                The `vidids` argument is an alias.  Mutually exclusive with
                other args.

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
            video_ids = vidids
        if video_ids is None:
            video_ids = sorted(self.index.videos.keys())

        if names is not None:
            video_ids = [self.index.name_to_video[name]['id'] for name in names]
        return Videos(video_ids, self)


class MixinCocoStats(object):
    """
    Methods for getting stats about the dataset
    """

    @property
    def n_annots(self):
        """ The number of annotations in the dataset """
        return len(self.dataset.get('annotations', []))

    @property
    def n_images(self):
        """ The number of images in the dataset """
        return len(self.dataset.get('images', []))

    @property
    def n_cats(self):
        """ The number of categories in the dataset """
        return len(self.dataset.get('categories', []))

    @property
    def n_videos(self):
        """ The number of videos in the dataset """
        return len(self.dataset.get('videos', []))

    def category_annotation_frequency(self):
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
            lambda x: None if x is None else self.cats[x]['name'],
            ub.map_vals(len, self.index.cid_to_aids))
        catname_to_nannots = ub.odict(sorted(catname_to_nannots.items(),
                                             key=lambda kv: (kv[1], kv[0])))
        return catname_to_nannots

    def conform(self, **config):
        """
        Make the COCO file conform a stricter spec, infers attibutes where
        possible.

        Corresponds to the ``kwcoco conform`` CLI tool.

        KWArgs:
            **config :

                pycocotools_info (default=True): returns info required by pycocotools

                ensure_imgsize (default=True): ensure image size is populated

                mmlab (default=False): if True tries to convert data
                to be compatible with open-mmlab tooling.

                legacy (default=False): if True tries to convert data
                structures to items compatible with the original
                pycocotools spec

                workers (int): number of parallel jobs for IO tasks

        Example:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('shapes8')
            >>> dset.index.imgs[1].pop('width')
            >>> dset.conform(legacy=True)
            >>> assert 'width' in dset.index.imgs[1]
            >>> assert 'area' in dset.index.anns[1]
        """

        if config.get('ensure_imgsize', True):
            self._ensure_imgsize(workers=config.get('workers', 8))

            for vidid, gids in self.index.vidid_to_gids.items():
                if len(gids):
                    # Each image in a video should be the same size (unless we
                    # implement some transform from image to video space)
                    # populate each video with the consistent width/height
                    video = self.index.videos[vidid]
                    vid_frames = self.images(gids)
                    width_cand = vid_frames.lookup('width')
                    height_cand = vid_frames.lookup('height')
                    if ub.allsame(height_cand) and ub.allsame(width_cand):
                        video['width'] = width_cand[0]
                        video['height'] = height_cand[0]

        if config.get('pycocotools_info', True):
            for ann in ub.ProgIter(self.dataset['annotations'], desc='update anns'):
                if 'iscrowd' not in ann:
                    ann['iscrowd'] = False

                if 'ignore' not in ann:
                    ann['ignore'] = ann.get('weight', 1.0) < .5

                if 'area' not in ann:
                    # Use segmentation if available
                    if 'segmentation' in ann:
                        try:
                            import kwimage
                            poly = kwimage.MultiPolygon.from_coco(ann['segmentation'])
                            ann['area'] = float(poly.to_shapely().area)
                        except Exception:
                            import warnings
                            warnings.warn(ub.paragraph(
                                '''
                                Unable to coerce segmentation to a polygon.
                                This may be indicative of a bug in
                                `kwimage.MultiPolygon.coerce` or a misformatted
                                segmentation
                                '''))
                    else:
                        try:
                            x, y, w, h = ann['bbox']
                        except KeyError:
                            warnings.warn(ub.paragraph(
                                '''
                                Unable to add "area" key because an annotation
                                is missing or has a malformed "bbox" entry
                                '''))
                        else:
                            ann['area'] = w * h

        if config.get('mmlab', False):
            # Add support for open-mmlab coco dataset ingestion
            if self.dataset.get('videos', None):
                for images in self.videos().images:
                    # mmdet wants the frame_id property
                    for frame_index, img in enumerate(images.objs):
                        img['frame_id'] = frame_index
                # mmdet wants the instance_id property
                for ann in enumerate(self.dataset['annotations']):
                    if 'track_id' in ann:
                        ann['instance_id'] = ann['track_id']

        if config.get('legacy', False):
            try:
                kpcats = self.keypoint_categories()
            except Exception:
                kpcats = None
            for ann in ub.ProgIter(self.dataset['annotations'], desc='update orig coco anns'):
                # Use segmentation if available
                if 'segmentation' in ann:
                    # TODO: any original style coco dict is ok, we dont
                    # always need it to be a poly if it is RLE
                    import kwimage
                    poly = kwimage.MultiPolygon.from_coco(ann['segmentation'])
                    # Hack, looks like kwimage does not wrap the original
                    # coco polygon with a list, but pycocotools needs that
                    ann['segmentation'] = poly.to_coco(style='orig')
                if 'keypoints' in ann:
                    import kwimage
                    # TODO: these have to be in some defined order for
                    # each category, currently it is arbitrary
                    pts = kwimage.Points.from_coco(ann['keypoints'], classes=kpcats)
                    ann['keypoints'] = pts.to_coco(style='orig')

    def validate(self, **config):
        """
        Performs checks on this coco dataset.

        Corresponds to the ``kwcoco validate`` CLI tool.

        Args:
            **config :
                schema (default=True): if True, validate the json-schema

                unique (default=True): if True, validate unique secondary keys

                missing (default=True): if True, validate registered files exist

                corrupted (default=False): if True, validate data in registered files

                channels (default=True):
                if True, validate that channels in auxiliary/asset items
                are all unique.

                require_relative (default=False):
                if True, causes validation to fail if paths are
                non-portable, i.e.  all paths must be relative to the
                bundle directory. if>0, paths must be relative to bundle
                root.  if>1, paths must be inside bundle root.

                img_attrs (default='warn'):
                if truthy, check that image attributes contain width and
                height entries. If 'warn', then warn if they do not exist.
                If 'error', then fail.

                verbose (default=1): verbosity flag

                workers (int): number of workers for parallel checks. defaults to 0

                fastfail (default=False): if True raise errors immediately

        Returns:
            dict: result containing keys -
                status (bool): False if any errors occurred
                errors (List[str]): list of all error messages
                missing (List): List of any missing images
                corrupted (List): List of any corrupted images

        SeeAlso:
            :func:`_check_integrity` - performs internal checks

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> import pytest
            >>> with pytest.warns(UserWarning):
            >>>     result = self.validate()
            >>> assert not result['errors']
            >>> assert result['warnings']
        """
        dset = self

        result = {
            'errors': [],
            'warnings': [],
        }
        verbose = config.get('verbose', 1)
        fastfail = config.get('fastfail', 0)

        def _error(msg):
            if verbose:
                print(msg)
            if fastfail:
                raise Exception(msg)
            result['errors'].append(msg)

        def _warn(msg):
            if verbose:
                print(msg)
            warnings.warn(msg, UserWarning)
            result['warnings'].append(msg)

        if config.get('schema', True):
            import jsonschema
            from kwcoco.coco_schema import COCO_SCHEMA
            if verbose:
                print('Validate json-schema')
            try:
                COCO_SCHEMA.validate(dset.dataset)
            except jsonschema.exceptions.ValidationError as ex:
                err = ex
                print(f'err.absolute_path={err.absolute_path}')
                msg = 'Failed to validate schema: {}'.format(str(err))
                _error(msg)

        def _check_unique(dset, table_key, col_key, required=True):
            if verbose:
                print('Check table {!r} has unique {!r}'.format(
                    table_key, col_key))
            items = dset.dataset.get(table_key, [])
            seen = set()
            num_unset = 0
            for obj in items:
                value = obj.get(col_key, None)
                if value is None:
                    num_unset += 1
                else:
                    if value in seen:
                        msg = 'Duplicate {} {} = {!r}'.format(
                            table_key, col_key, value)
                        _error(msg)
                    else:
                        seen.add(value)
            if num_unset > 0:
                msg = ub.paragraph(
                    '''
                    Table {!r} is missing {} / {} values for column {!r}
                    '''
                ).format(table_key, num_unset, len(items), col_key)
                if required:
                    _error(msg)
                else:
                    _warn(msg)

        def _check_attrs(dset, table_key, col_key, required=True):
            if verbose:
                print('Check table {!r} entries have attr {!r}'.format(
                    table_key, col_key))
            items = dset.dataset.get(table_key, [])
            num_unset = 0
            for obj in items:
                value = obj.get(col_key, None)
                if value is None:
                    num_unset += 1
            if num_unset > 0:
                msg = ub.paragraph(
                    '''
                    Table {!r} is missing {} / {} values for column {!r}
                    '''
                ).format(table_key, num_unset, len(items), col_key)
                if required:
                    _error(msg)
                else:
                    _warn(msg)

        def _check_subtable_attrs(dset, table_key, subtable_keys, col_key, required=True):
            if verbose:
                print('Check subtable {!r} / {!r} entries have attr {!r}'.format(
                    table_key, subtable_keys, col_key))
            items = dset.dataset.get(table_key, [])
            num_unset = 0
            num_subobjs = 0
            for obj in items:
                # hack for asset/auxiliary
                _ = ub.dict_isect(obj, subtable_keys)
                sub_objs = ub.peek(_.values()) if _ else None
                if sub_objs:
                    for sub_obj in sub_objs:
                        num_subobjs += 1
                        value = sub_obj.get(col_key, None)
                        if value is None:
                            num_unset += 1
            if num_unset > 0:
                msg = ub.paragraph(
                    '''
                    Subtable {!r}/{!r} is missing {} / {} values for column {!r}
                    '''
                ).format(table_key, subtable_keys, num_unset, num_subobjs, col_key)
                if required:
                    _error(msg)
                else:
                    _warn(msg)

        if config.get('unique', True):

            _check_unique(dset, table_key='categories', col_key='name')
            _check_unique(dset, table_key='videos', col_key='name')
            _check_unique(dset, table_key='images', col_key='name',
                          required=False)

            _check_unique(dset, table_key='images', col_key='id')
            _check_unique(dset, table_key='videos', col_key='id')
            _check_unique(dset, table_key='annotations', col_key='id')
            _check_unique(dset, table_key='categories', col_key='id')

        if config.get('img_attrs', False):
            required = config.get('img_attrs', False) == 'error'
            # Refine this?
            _check_attrs(dset, table_key='images', col_key='width', required=required)
            _check_attrs(dset, table_key='images', col_key='height', required=required)
            _check_subtable_attrs(dset, table_key='images', subtable_keys=['asset', 'auxiliary'], col_key='width', required=required)
            _check_subtable_attrs(dset, table_key='images', subtable_keys=['asset', 'auxiliary'], col_key='height', required=required)

        if config.get('annot_attrs', True):
            required = config.get('annot_attrs', False) == 'error'
            _check_attrs(dset, table_key='annotations', col_key='bbox', required=False)

        if config.get('channels', True):
            for img in self.dataset.get('images', []):
                seen = set()
                assets = img.get('auxiliary', []) + img.get('assets', [])

                for obj in assets:
                    channels = obj.get('channels', None)
                    if channels is None:
                        gid = img['id']
                        _error('Asset in gid={} is missing channels, obj={}'.format(gid, obj))
                    else:
                        from kwcoco import FusedChannelSpec
                        channels = FusedChannelSpec.coerce(channels)
                        for chan in channels.as_list():
                            if chan in seen:
                                gid = img['id']
                                _error('The chan={} is specified more than once in gid={}'.format(chan, gid))
                            seen.add(chan)
        if config.get('missing', True):
            missing = dset.missing_images(check_aux=True, verbose=verbose)
            if missing:
                msg = ub.paragraph(
                    f'''
                    There are {len(missing)} missing images.
                    The first one is {missing[0][1]!r}.
                    ''')
                _error(msg)
                result['missing'] = missing

        if config.get('corrupted', False):
            corrupted = dset.corrupted_images(check_aux=True, verbose=verbose,
                                              workers=config.get('workers', 0))
            if corrupted:
                msg = ub.paragraph(
                    f'''
                    There are {len(corrupted)} corrupted images.
                    The first one is {corrupted[0][1]!r}.
                    ''')
                _error(msg)
                result['corrupted'] = corrupted

        if config.get('require_relative', False):
            rr = config.get('require_relative', False)
            for gid in dset.imgs.keys():
                coco_img = dset.coco_image(gid)
                for obj in coco_img.iter_asset_objs():
                    fname = ub.Path(obj['file_name'])
                    if rr > 0:
                        if fname.is_absolute():
                            _error('non relative fname = {!r}'.format(fname))
                    if rr > 1:
                        if '..' in fname.parts:
                            _error('non internal fname = {!r}'.format(fname))

        return result

    def stats(self, **kwargs):
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

        def varied(obj_lut):
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
            image_attrs = varied(dset.index.imgs)
            info['image_attrs'] = image_attrs

        if config.get('annot_attrs', True):
            annot_attrs = varied(dset.index.anns)
            info['annot_attrs'] = annot_attrs

        if config.get('video_attrs', True):
            annot_attrs = varied(dset.index.videos)
            info['video_attrs'] = annot_attrs

        if config.get('boxes', False):
            info['boxes'] = dset.boxsize_stats()
        return info

    def basic_stats(self):
        """
        Reports number of images, annotations, and categories.

        SeeAlso:
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
            }

            >>> from kwcoco.demo.toydata_video import random_video_dset
            >>> dset = random_video_dset(render=True, num_frames=2, num_tracks=10, rng=0)
            >>> print(ub.urepr(dset.basic_stats()))
            {
                'n_anns': 20,
                'n_imgs': 2,
                'n_videos': 1,
                'n_cats': 3,
            }
        """
        return ub.odict([
            ('n_anns', self.n_annots),
            ('n_imgs', self.n_images),
            ('n_videos', self.n_videos),
            ('n_cats', self.n_cats),
        ])

    def extended_stats(self):
        """
        Reports number of images, annotations, and categories.

        SeeAlso:
            :func:`kwcoco.coco_dataset.MixinCocoStats.basic_stats`
            :func:`kwcoco.coco_dataset.MixinCocoStats.extended_stats`

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> print(ub.urepr(self.extended_stats()))
        """
        def mapping_stats(xid_to_yids):
            import kwarray
            n_yids = list(ub.map_vals(len, xid_to_yids).values())
            return kwarray.stats_dict(n_yids, n_extreme=True)

        if hasattr(self, '_all_rows_column_lookup'):
            # Hack for SQL speed, still slow though
            aid_to_cid = dict(self._all_rows_column_lookup(
                'annotations', ['id', 'category_id']))
            gid_to_cidfreq = {}
            for gid, aids in self.index.gid_to_aids.items():
                gid_to_cidfreq[gid] = ub.dict_hist(
                    ub.take(aid_to_cid, aids))
            # for gid, aids in self.index.gid_to_aids.items():
            #     cids = self._column_lookup(
            #         'annotations', 'category_id', rowids=aids)
            #     gid_to_cidfreq[gid] = ub.dict_hist(cids)
        else:
            # this is slow for sql.
            gid_to_cidfreq = ub.map_vals(
                lambda aids: ub.dict_hist([
                    self.anns[aid]['category_id'] for aid in aids]),
                self.index.gid_to_aids)
        gid_to_cids = {
            gid: list(cidfreq.keys())
            for gid, cidfreq in gid_to_cidfreq.items()
        }
        return ub.odict([
            ('annots_per_img', mapping_stats(self.index.gid_to_aids)),
            ('imgs_per_cat', mapping_stats(self.index.cid_to_gids)),
            ('cats_per_img', mapping_stats(gid_to_cids)),
            ('annots_per_cat', mapping_stats(self.index.cid_to_aids)),
            ('imgs_per_video', mapping_stats(self.index.vidid_to_gids)),
        ])

    def boxsize_stats(self, anchors=None, perclass=True, gids=None, aids=None,
                      verbose=0, clusterkw={}, statskw={}):
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
            aids = ub.flatten(ub.take(self.index.gid_to_aids, gids))
        if aids is not None:
            anns = ub.take(self.anns, aids)
        else:
            anns = self.dataset.get('annotations', [])

        for ann in anns:
            if 'bbox' in ann:
                cname = self.cats[ann['category_id']]['name']
                cname_to_box_sizes[cname].append(ann['bbox'][2:4])
        cname_to_box_sizes = ub.map_vals(np.array, cname_to_box_sizes)

        def _boxes_info(box_sizes):
            box_info = {
                'stats': kwarray.stats_dict(box_sizes, axis=0, **statskw)
            }
            if anchors:
                from sklearn import cluster
                defaultkw = {
                    'n_clusters': anchors,
                    'n_init': 20,
                    'max_iter': 10000,
                    'tol': 1e-6,
                    'algorithm': 'elkan',
                    'verbose': verbose
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
            for cname, box_sizes in cname_to_box_sizes.items():
                if verbose:
                    print('compute {} bbox stats'.format(cname))
                cid_to_info[cname] = _boxes_info(box_sizes)
            infos['perclass'] = cid_to_info

        if verbose:
            print('compute all bbox stats')
        all_sizes = np.vstack(list(cname_to_box_sizes.values()))
        all_info = _boxes_info(all_sizes)
        infos['all'] = all_info
        return infos

    def find_representative_images(self, gids=None):
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
            gids = sorted(self.imgs.keys())
            gid_to_aids = self.index.gid_to_aids
        else:
            gid_to_aids = ub.dict_subset(self.index.gid_to_aids, gids)

        all_cids = set(self.index.cid_to_aids.keys())

        # Select representative images to draw such that each category
        # appears at least once.
        gid_to_cidfreq = ub.map_vals(
            lambda aids: ub.dict_hist([self.anns[aid]['category_id']
                                       for aid in aids]),
            gid_to_aids)

        gid_to_nannots = ub.map_vals(len, gid_to_aids)

        gid_to_cids = {
            gid: list(cidfreq.keys())
            for gid, cidfreq in gid_to_cidfreq.items()
        }

        for gid, nannots in gid_to_nannots.items():
            if nannots == 0:
                # Add a dummy category to note images without any annotations
                gid_to_cids[gid].append(-1)
                all_cids.add(-1)

        all_cids = list(all_cids)

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
                candidate_sets,
                items=all_cids,
                set_weights=large_image_weights)
        selected.update(cover2)
        candidate_sets = ub.dict_diff(candidate_sets, cover2)

        cover3 = kwarray.setcover(
                candidate_sets,
                items=all_cids,
                set_weights=small_image_weights)
        selected.update(cover3)
        candidate_sets = ub.dict_diff(candidate_sets, cover3)

        selected_gids = sorted(selected.keys())
        return selected_gids


class MixinCocoDraw(object):
    """
    Matplotlib / display functionality
    """

    def draw_image(self, gid, channels=None):
        """
        Use kwimage to draw all annotations on an image and return the pixels
        as a numpy array.

        Args:
            gid (int): image id to draw
            channels (kwcoco.ChannelSpec): the channel to draw on

        Returns:
            ndarray: canvas

        SeeAlso
            :func:`kwcoco.coco_dataset.MixinCocoDraw.draw_image`
            :func:`kwcoco.coco_dataset.MixinCocoDraw.show_image`

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('shapes8')
            >>> self.draw_image(1)
            >>> # Now you can dump the annotated image to disk / whatever
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
        """
        import kwimage
        # Load the raw image pixels
        coco_img = self.coco_image(gid)
        delayed = coco_img.imdelay(space='image', channels=channels)

        if delayed.channels is not None and delayed.channels.numel() > 3:
            import warnings
            warnings.warn('Requested drawing more than 3 channels. '
                          'Only picking first 3')
            first_channels = coco_img.imdelay(space='image').channels.fuse()[0:3]
            delayed = delayed.take_channels(first_channels)

        canvas = delayed.finalize()
        canvas = _normalize_intensity_if_needed(canvas)

        # Get annotation IDs from this image
        aids = self.index.gid_to_aids[gid]
        # Grab relevant annotation dictionaries
        anns = [self.anns[aid] for aid in aids]
        # Transform them into a kwimage.Detections datastructure

        dset = self
        try:
            classes = dset.object_categories()
        except Exception:
            classes = list(dset.name_to_cat.keys())
        try:
            kp_classes = dset.keypoint_categories()
        except Exception:
            # hack
            anns = [ann.copy() for ann in anns]
            for ann in anns:
                ann.pop('keypoints', None)
            kp_classes = None
        cats = dset.dataset['categories']
        # dets = kwimage.Detections.from_coco_annots(anns, dset=self)
        dets = kwimage.Detections.from_coco_annots(
            anns, classes=classes, cats=cats, kp_classes=kp_classes)

        canvas = dets.draw_on(canvas)
        return canvas

    def show_image(self, gid=None, aids=None, aid=None, channels=None, setlim=None, **kwargs):
        """
        Use matplotlib to show an image with annotations overlaid

        Args:
            gid (int | None):
                image id to show
            aids (list | None):
                aids to highlight within the image
            aid (int | None):
                a specific aid to focus on. If gid is not give, look up gid
                based on this aid.
            setlim (None | str):
                if 'image' sets the limit to the image extent
            **kwargs:
                show_annots, show_aid, show_catname, show_kpname,
                show_segmentation, title, show_gid, show_filename,
                show_boxes,

        SeeAlso
            :func:`kwcoco.coco_dataset.MixinCocoDraw.draw_image`
            :func:`kwcoco.coco_dataset.MixinCocoDraw.show_image`

        Ignore:
            # Programmatically collect the kwargs for docs generation
            import xinspect
            import kwcoco
            kwargs = xinspect.get_kwargs(kwcoco.CocoDataset.show_image)
            print(ub.urepr(list(kwargs.keys()), nl=1, si=1))

        Example:
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-msi')
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> # xdoctest: -REQUIRES(--show)
            >>> dset.show_image(gid=1, channels='B8')
            >>> # xdoctest: +REQUIRES(--show)
            >>> kwplot.show_if_requested()
        """
        import matplotlib as mpl
        from matplotlib import pyplot as plt
        import kwimage
        import kwplot
        import numpy as np

        figkw = {k: kwargs[k] for k in ['fnum', 'pnum', 'doclf', 'docla']
                 if k in kwargs}
        if figkw:
            kwplot.figure(**figkw)

        if gid is None:
            primary_ann = self.anns[aid]
            gid = primary_ann['image_id']

        show_all = kwargs.get('show_all', True)
        show_annots = kwargs.get('show_annots', True)
        show_labels = kwargs.get('show_labels', True)

        highlight_aids = set()
        if aid is not None:
            highlight_aids.add(aid)
        if aids is not None:
            highlight_aids.update(aids)

        coco_img = self.coco_image(gid)
        img = coco_img.img
        aids = self.index.gid_to_aids.get(img['id'], [])

        # Collect annotation overlays
        colored_segments = defaultdict(list)
        keypoints = []
        rects = []
        texts = []

        sseg_masks = []
        sseg_polys = []

        if show_annots:
            for aid in aids:
                ann = self.anns[aid]

                if 'keypoints' in ann:
                    cid = ann['category_id']
                    if ann['keypoints'] is not None and len(ann['keypoints']) > 0:
                        # TODO: rely on kwimage.Points to parse multiple format info?
                        kpts_data = ann['keypoints']
                        if isinstance(ub.peek(kpts_data), dict):
                            xys = np.array([p['xy'] for p in kpts_data])
                            isvisible = np.array([p.get('visible', True) for p in kpts_data])
                            kpnames = None
                            isvisible = np.array([p.get('visible', True) for p in kpts_data])
                        else:
                            try:
                                kpnames = self._lookup_kpnames(cid)
                            except KeyError:
                                kpnames = None
                            kpts = np.array(ann['keypoints']).reshape(-1, 3)
                            isvisible = kpts.T[2] > 0
                            xys = kpts.T[0:2].T[isvisible]
                    else:
                        kpnames = None
                        xys = None
                else:
                    kpnames = None
                    xys = None

                # Note standard coco bbox is [x,y,width,height]
                if 'bbox' in ann:
                    x1, y1 = ann['bbox'][0:2]
                elif 'line' in ann:
                    x1, y1 = ann['line'][0:2]
                elif 'keypoints' in ann:
                    x1, y1 = xys.min(axis=0)
                else:
                    raise Exception('no bbox, line, or keypoint position')

                cid = ann.get('category_id', None)
                if cid is not None:
                    cat = self.cats[cid]
                    catname = cat['name']
                else:
                    cat = None
                    catname = ann.get('category_name', 'None')
                textkw = {
                    'horizontalalignment': 'left',
                    'verticalalignment': 'top',
                    'backgroundcolor': (0, 0, 0, .3),
                    'color': 'white',
                    'fontproperties': mpl.font_manager.FontProperties(
                        size=6, family='monospace'),
                }
                annot_text_parts = []
                if show_labels:
                    if kwargs.get('show_aid', show_all):
                        annot_text_parts.append('aid={}'.format(aid))
                    if kwargs.get('show_catname', show_all):
                        annot_text_parts.append(catname)
                    if annot_text_parts:
                        annot_text = ' '.join(annot_text_parts)
                        texts.append((x1, y1, annot_text, textkw))

                color = 'orange' if aid in highlight_aids else 'blue'
                if 'obox' in ann:
                    # Oriented bounding box
                    segs = np.array(ann['obox']).reshape(-1, 3)[:, 0:2]
                    for pt1, pt2 in ub.iter_window(segs, wrap=True):
                        colored_segments[color].append([pt1, pt2])
                elif 'bbox' in ann:
                    [x, y, w, h] = ann['bbox']
                    rect = mpl.patches.Rectangle((x, y), w, h, facecolor='none',
                                                 edgecolor=color)
                    rects.append(rect)
                if 'line' in ann:
                    x1, y1, x2, y2 = ann['line']
                    pt1, pt2 = (x1, y1), (x2, y2)
                    colored_segments[color].append([pt1, pt2])
                if 'keypoints' in ann:
                    if xys is not None and len(xys):
                        keypoints.append(xys)
                        if show_labels and kwargs.get('show_kpname', True):
                            if kpnames is not None:
                                for (kp_x, kp_y), kpname in zip(xys, kpnames):
                                    texts.append((kp_x, kp_y, kpname, textkw))

                if ann.get('segmentation', None) is not None and kwargs.get('show_segmentation', True):
                    sseg = ann['segmentation']
                    # Respect the 'color' attribute of categories
                    if cat is not None:
                        catcolor = cat.get('color', None)
                    else:
                        catcolor = None

                    if catcolor is not None:
                        catcolor = kwplot.Color(catcolor).as01()
                    # TODO: Unify masks and polygons into a kwimage
                    # segmentation class
                    sseg = kwimage.Segmentation.coerce(sseg).data
                    if isinstance(sseg, kwimage.Mask):
                        m = sseg.to_c_mask()
                        sseg_masks.append((m.data, catcolor))
                    else:
                        # TODO: interior
                        multipoly = sseg.to_multi_polygon()
                        for poly in multipoly.data:
                            poly_xys = poly.data['exterior'].data
                            polykw = {}
                            if catcolor is not None:
                                polykw['color'] = catcolor
                            poly = mpl.patches.Polygon(poly_xys, **polykw)
                            try:
                                # hack
                                poly.area = sseg.to_shapely().area
                            except Exception:
                                pass
                            sseg_polys.append(poly)

        # Show image
        avail_channels = coco_img.channels

        if channels is None and avail_channels is not None:
            avail_spec = avail_channels.fuse().normalize()
            num_chans = avail_spec.numel()

            if num_chans == 0:
                raise Exception('no channels!')
            elif num_chans <= 2:
                print('Auto choosing 1 channel')
                channels = avail_spec.parsed[0]
            elif num_chans > 3:
                print('Auto choosing 3 channel')
                sensible_defaults = [
                    'red|green|blue',
                    'r|g|b',
                ]
                chosen = None
                for sensible in sensible_defaults:
                    cand = avail_spec & sensible
                    if cand.numel() == 3:
                        chosen = cand
                        break
                if chosen is None:
                    chosen = avail_spec[0:3]
                channels = chosen

        print('loading image')
        delayed = coco_img.imdelay(space='image', channels=channels)
        np_img = delayed.finalize()
        print('loaded image')

        np_img = kwimage.atleast_3channels(np_img)

        np_img01 = None
        if np_img.max() > 255:
            np_img01 = _normalize_intensity_if_needed(np_img)

        fig = plt.gcf()
        ax = fig.gca()
        ax.cla()

        if sseg_masks:
            if np_img01 is None:
                np_img01 = kwimage.ensure_float01(np_img)
            layers = []
            layers.append(kwimage.ensure_alpha_channel(np_img01))
            distinct_colors = kwplot.Color.distinct(len(sseg_masks))

            for (mask, _catcolor), col in zip(sseg_masks, distinct_colors):
                if _catcolor is not None:
                    col = kwimage.ensure_float01(np.array(_catcolor)).tolist()

                col = np.array(col + [1])[None, None, :]
                alpha_mask = col * mask[:, :, None]
                alpha_mask[..., 3] = mask * 0.5
                layers.append(alpha_mask)

            masked_img = kwimage.overlay_alpha_layers(layers[::-1])
            ax.imshow(masked_img)
        else:
            if np_img01 is not None:
                ax.imshow(np_img01)
            else:
                ax.imshow(np_img)

        title = kwargs.get('title', None)
        if title is None:
            title_parts = []
            if kwargs.get('show_gid', True):
                title_parts.append('gid={}'.format(gid))
            if kwargs.get('show_filename', True):
                title_parts.append(str(img['file_name']))
            title = ' '.join(title_parts)
        if title:
            ax.set_title(title)

        if sseg_polys:
            # print('sseg_polys = {!r}'.format(sseg_polys))
            # hack: show smaller polygons first.
            if len(sseg_polys):
                areas = np.array([getattr(p, 'area', np.inf) for p in sseg_polys])
                sortx = np.argsort(areas)[::-1]
                sseg_polys = list(ub.take(sseg_polys, sortx))

            poly_col = mpl.collections.PatchCollection(
                sseg_polys, 2, alpha=0.4)
            ax.add_collection(poly_col)

        # Show all annotations inside it
        if kwargs.get('show_boxes', True):
            for (x1, y1, catname, textkw) in texts:
                ax.text(x1, y1, catname, **textkw)

            for color, segments in colored_segments.items():
                line_col = mpl.collections.LineCollection(segments, linewidths=2, color=color)
                ax.add_collection(line_col)

            rect_col = mpl.collections.PatchCollection(rects, match_original=True)
            ax.add_collection(rect_col)
            if keypoints:
                xs, ys = np.vstack(keypoints).T
                ax.plot(xs, ys, 'bo')

        if setlim:
            if not isinstance(setlim, str):
                setlim = 'image'

            if setlim == 'image':
                ax.set_xlim(0, img['width'])
                ax.set_ylim(img['height'], 0)
            else:
                raise NotImplementedError(setlim)

        return ax


def _normalize_intensity_if_needed(canvas):
    # We really need a function that can adapt to dynamic ranges
    # in a robust way that ensures at least something is visible.
    # Normalize intensity is pretty good, but has common edge cases.
    import kwimage
    max_value = canvas.max()
    min_value = canvas.min()
    if canvas.dtype.kind in ['u', 'i'] and max_value > 255 or min_value < 0:
        canvas = kwimage.normalize_intensity(canvas)
    elif max_value > 0 or min_value < 0:
        canvas = kwimage.normalize_intensity(canvas)
    return canvas


class MixinCocoAddRemove(object):
    """
    Mixin functions to dynamically add / remove annotations images and
    categories while maintaining lookup indexes.
    """

    def add_video(self, name, id=None, **kw):
        """
        Register a new video with the dataset

        Args:
            name (str): Unique name for this video.
            id (None | int): ADVANCED. Force using this image id.
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
        if id is None:
            id = self._next_ids.get('videos')
        video = ub.odict()
        video['id'] = id
        video['name'] = name
        video.update(**kw)
        self.dataset['videos'].append(video)
        self.index._add_video(id, video)
        # self._invalidate_hashid(['videos'])
        return id

    def add_image(self, file_name=None, id=None, **kw):
        """
        Register a new image with the dataset

        Args:
            file_name (str | None): relative or absolute path to image.
                 if not given, then "name" must be specified and we will
                 expect that "auxiliary" assets are eventually added.
            id (None | int): ADVANCED. Force using this image id.
            name (str): a unique key to identify this image
            width (int): base width of the image
            height (int): base height of the image
            channels (ChannelSpec): specification of base channels.
                Only relevant if file_name is given.
            auxiliary (List[Dict]): specification of auxiliary assets.
                See CocoImage.add_auxiliary_item for details
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
        if id is None:
            id = self._next_ids.get('images')
        elif self.imgs and id in self.imgs:
            raise exceptions.DuplicateAddError('Image id={} already exists'.format(id))

        img = _dict()
        img['id'] = int(id)
        try:
            img['file_name'] = os.fspath(file_name)
        except TypeError:
            img['file_name'] = file_name
        img.update(**kw)
        self.index._add_image(id, img)
        self.dataset['images'].append(img)
        self._invalidate_hashid()
        return id

    def add_auxiliary_item(self, gid, file_name=None, channels=None, **kwargs):
        """
        Adds an auxiliary / asset item to the image dictionary.

        Args:
            gid (int):
                The image id to add the auxiliary/asset item to.

            file_name (str | None):
                The name of the file relative to the bundle directory. If
                unspecified, imdata must be given.

            channels (str | kwcoco.FusedChannelSpec):
                The channel code indicating what each of the bands represents.
                These channels should be disjoint wrt to the existing data in
                this image (this is not checked).

            **kwargs:
                See :func:`CocoImage.add_auxiliary_item` for more details

        Example:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset()
            >>> gid = dset.add_image(name='my_image_name', width=200, height=200)
            >>> dset.add_auxiliary_item(gid, 'path/fake_B0.tif', channels='B0',
            >>>                         width=200, height=200,
            >>>                         warp_aux_to_img={'scale': 1.0})
        """
        coco_img = self.coco_image(gid)
        coco_img.add_auxiliary_item(file_name=file_name, channels=channels,
                                    **kwargs)

    def add_annotation(self, image_id, category_id=None, bbox=ub.NoParam,
                       segmentation=ub.NoParam, keypoints=ub.NoParam, id=None,
                       **kw):
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
                format, see :func:`kwimage.Keypoints.to_coco`.
                Extended types: `KeypointsLike`.

            id (None | int): Force using this annotation id. Typically you
                should NOT specify this. A new unused id will be chosen and
                returned.

            **kw : stores arbitrary key/value pairs in this new image,
                Common respected key/values include but are not limited to the
                following:
                track_id (int | str): some value used to associate annotations
                that belong to the same "track".
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
            import kwimage
        except ImportError:
            kwimage = None

        if id is None:
            id = self._next_ids.get('annotations')
        elif self.anns and id in self.anns:
            raise IndexError('Annot id={} already exists'.format(id))

        ann = _dict()
        ann['id'] = int(id)
        ann['image_id'] = int(image_id)
        ann['category_id'] = None if category_id is None else int(category_id)

        if kwimage is not None and hasattr(bbox, 'to_coco'):
            # to_coco works different for boxes atm, might update in future
            try:
                ann['bbox'] = ub.peek(bbox.to_coco(style='new'))
            except Exception:
                ann['bbox'] = bbox.to_xywh().data.tolist()
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
            ann['segmentation'] = segmentation.to_coco(style='new')
        elif segmentation is not ub.NoParam:
            ann['segmentation'] = segmentation
        else:
            assert segmentation is ub.NoParam

        ann.update(**kw)
        track_id = ann.get('track_id', None)

        self.dataset['annotations'].append(ann)
        self.index._add_annotation(id, image_id, category_id, track_id, ann)
        self._invalidate_hashid(['annotations'])
        return id

    def add_category(self, name, supercategory=None, id=None, **kw):
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
        index = self.index
        if index.cats and name in index.name_to_cat:
            raise exceptions.DuplicateAddError('Category name={!r} already exists'.format(name))

        if id is None:
            id = self._next_ids.get('categories')
        elif index.cats and id in index.cats:
            raise exceptions.DuplicateAddError('Category id={} already exists'.format(id))

        cat = _dict()
        cat['id'] = int(id)
        cat['name'] = str(name)
        if supercategory:
            cat['supercategory'] = supercategory
        cat.update(**kw)

        # Add to raw data structure
        self.dataset['categories'].append(cat)

        # And add to the indexes
        index._add_category(id, name, cat)
        self._invalidate_hashid(['categories'])
        return id

    def ensure_image(self, file_name, id=None, **kw):
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
            img = self.index.file_name_to_img[file_name]
            id = img['id']
        return id

    def ensure_category(self, name, supercategory=None, id=None, **kw):
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
            id = self.add_category(name=name, supercategory=supercategory,
                                   id=id, **kw)
        except exceptions.DuplicateAddError:
            cat = self.index.name_to_cat[name]
            id = cat['id']
        return id

    def add_annotations(self, anns):
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
            >>> assert self.n_annots == 7 and self._check_index()
            >>> self.add_annotations(anns)
            >>> assert self.n_annots == 11 and self._check_index()
        """
        self.dataset['annotations'].extend(anns)
        self.index._add_annotations(anns)
        self._invalidate_hashid(['annotations'])

    def add_images(self, imgs):
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
            >>> assert self.n_images == 3 and self._check_index()
        """
        self.dataset['images'].extend(imgs)
        self.index._add_images(imgs)
        self._invalidate_hashid(['images'])

    def clear_images(self):
        """
        Removes all images and annotations (but not categories)

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> self.clear_images()
            >>> print(ub.urepr(self.basic_stats(), nobr=1, nl=0, si=1))
            n_anns: 0, n_imgs: 0, n_videos: 0, n_cats: 8
        """
        # self.dataset['images'].clear()
        # self.dataset['annotations'].clear()
        del self.dataset['images'][:]
        del self.dataset['annotations'][:]
        self.index._remove_all_images()
        self._invalidate_hashid(['images', 'annotations'])

    def clear_annotations(self):
        """
        Removes all annotations (but not images and categories)

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> self.clear_annotations()
            >>> print(ub.urepr(self.basic_stats(), nobr=1, nl=0, si=1))
            n_anns: 0, n_imgs: 3, n_videos: 0, n_cats: 8
        """
        # self.dataset['annotations'].clear()
        del self.dataset['annotations'][:]
        self.index._remove_all_annotations()
        self._invalidate_hashid(['annotations'])

    def remove_annotation(self, aid_or_ann):
        """
        Remove a single annotation from the dataset

        If you have multiple annotations to remove its more efficient to remove
        them in batch with
        :func:`kwcoco.coco_dataset.MixinCocoAddRemove.remove_annotations`

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> aids_or_anns = [self.anns[2], 3, 4, self.anns[1]]
            >>> self.remove_annotations(aids_or_anns)
            >>> assert len(self.dataset['annotations']) == 7
            >>> self._check_index()
        """
        # Do the simple thing, its O(n) anyway,
        remove_ann = self._resolve_to_ann(aid_or_ann)
        self.dataset['annotations'].remove(remove_ann)
        self.index.clear()
        self._invalidate_hashid(['annotations'])

    def remove_annotations(self, aids_or_anns, verbose=0, safe=True):
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
            >>> self._check_index()
        """
        remove_info = {'annotations': None}
        # Do nothing if given no input
        if aids_or_anns:

            # build mapping from aid to index O(n)
            # TODO: it would be nice if this mapping was as part of the index.
            aid_to_index = {
                ann['id']: index
                for index, ann in enumerate(self.dataset['annotations'])
            }
            remove_aids = list(map(self._resolve_to_id, aids_or_anns))
            if safe:
                remove_aids = sorted(set(remove_aids))
            remove_info['annotations'] = len(remove_aids)

            # Lookup the indices to remove, sort in descending order
            if verbose > 1:
                print('Removing {} annotations'.format(len(remove_aids)))

            remove_idxs = list(ub.take(aid_to_index, remove_aids))

            self.index._remove_annotations(remove_aids, verbose=verbose)

            _delitems(self.dataset['annotations'], remove_idxs)

            self._invalidate_hashid(['annotations'])
        return remove_info

    def remove_categories(self, cat_identifiers, keep_annots=False, verbose=0,
                          safe=True):
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
            >>> self._check_index()
        """
        remove_info = {'annotations': None, 'categories': None}
        if cat_identifiers:
            if verbose > 1:
                print('Removing annots of removed categories')

            if safe:
                remove_cids = set()
                for identifier in cat_identifiers:
                    try:
                        cid = self._resolve_to_cid(identifier)
                        remove_cids.add(cid)
                    except Exception:
                        pass
                remove_cids = sorted(remove_cids)
            else:
                remove_cids = list(map(self._resolve_to_cid, cat_identifiers))
            # First remove any annotation that belongs to those categories
            if self.index.cid_to_aids:
                remove_aids = list(it.chain(*[self.index.cid_to_aids[cid]
                                              for cid in remove_cids]))
            else:
                remove_aids = [ann['id'] for ann in self.dataset['annotations']
                               if ann['category_id'] in remove_cids]

            if keep_annots:
                # Simply remove category information instead of removing the
                # entire annotation.
                for aid in remove_aids:
                    self.anns[aid].pop('category_id')
            else:
                rminfo = self.remove_annotations(remove_aids, verbose=verbose)
                remove_info.update(rminfo)

            remove_info['categories'] = len(remove_cids)
            if verbose > 1:
                print('Removing {} category entries'.format(len(remove_cids)))
            id_to_index = {
                cat['id']: index
                for index, cat in enumerate(self.dataset['categories'])
            }
            # Lookup the indices to remove, sort in descending order
            remove_idxs = list(ub.take(id_to_index, remove_cids))
            _delitems(self.dataset['categories'], remove_idxs)

            self.index._remove_categories(remove_cids, verbose=verbose)
            self._invalidate_hashid(['categories', 'annotations'])

        return remove_info

    def remove_images(self, gids_or_imgs, verbose=0, safe=True):
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
            >>> self._check_index()
            >>> gids_or_imgs = [3]
            >>> self.remove_images(gids_or_imgs)
            >>> assert len(self.dataset['images']) == 0
            >>> self._check_index()
        """
        remove_info = {'annotations': None, 'images': None}
        if gids_or_imgs:
            if verbose > 1:
                print('Removing images')

            remove_gids = list(map(self._resolve_to_gid, gids_or_imgs))
            if safe:
                remove_gids = sorted(set(remove_gids))
            # First remove any annotation that belongs to those images
            if self.index.gid_to_aids:
                remove_aids = list(it.chain(*[self.index.gid_to_aids[gid]
                                              for gid in remove_gids]))
            else:
                remove_aids = [ann['id'] for ann in self.dataset['annotations']
                               if ann['image_id'] in remove_gids]

            rminfo = self.remove_annotations(remove_aids, verbose=verbose)
            remove_info.update(rminfo)

            remove_info['images'] = len(remove_gids)
            if verbose > 1:
                print('Removing {} image entries'.format(len(remove_gids)))
            id_to_index = {
                img['id']: index
                for index, img in enumerate(self.dataset['images'])
            }
            # Lookup the indices to remove, sort in descending order
            remove_idxs = list(ub.take(id_to_index, remove_gids))
            _delitems(self.dataset['images'], remove_idxs)

            self.index._remove_images(remove_gids, verbose=verbose)
            self._invalidate_hashid(['images', 'annotations'])

        return remove_info

    def remove_videos(self, vidids_or_videos, verbose=0, safe=True):
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
            >>> self._check_index()
        """
        remove_info = {'annotations': None, 'images': None, 'videos': None}
        if vidids_or_videos:
            if verbose > 1:
                print('Removing videos')

            remove_vidids = list(map(self._resolve_to_vidid, vidids_or_videos))
            if safe:
                remove_vidids = sorted(set(remove_vidids))
            # First remove any annotation that belongs to those images
            if self.index.vidid_to_gids:
                remove_gids = list(it.chain(*[self.index.vidid_to_gids[vidid]
                                              for vidid in remove_vidids]))
            else:
                remove_gids = [ann['id'] for ann in self.dataset['videos']
                               if ann['image_id'] in remove_gids]

            rminfo = self.remove_images(remove_gids, verbose=verbose,
                                        safe=safe)
            remove_info.update(rminfo)

            remove_info['videos'] = len(remove_vidids)
            if verbose > 1:
                print('Removing {} video entries'.format(len(remove_vidids)))
            id_to_index = {
                video['id']: index
                for index, video in enumerate(self.dataset['videos'])
            }
            # Lookup the indices to remove, sort in descending order
            remove_idxs = list(ub.take(id_to_index, remove_vidids))
            _delitems(self.dataset['videos'], remove_idxs)

            self.index._remove_videos(remove_vidids, verbose=verbose)
            self._invalidate_hashid(['videos', 'images', 'annotations'])
        return remove_info

    def remove_annotation_keypoints(self, kp_identifiers):
        """
        Removes all keypoints with a particular category

        Args:
            kp_identifiers (List): list of keypoint category dicts, names, or ids

        Returns:
            Dict: num_removed: information on the number of items removed
        """
        # kpnames = {k['name'] for k in remove_kpcats}
        # TODO: needs optimization
        remove_kpcats = list(map(self._resolve_to_kpcat, kp_identifiers))
        kpcids = {k['id'] for k in remove_kpcats}
        num_kps_removed = 0
        for ann in self.dataset['annotations']:
            remove_idxs = [
                kp_idx for kp_idx, kp in enumerate(ann['keypoints'])
                if kp['keypoint_category_id'] in kpcids
            ]
            num_kps_removed += len(remove_idxs)
            _delitems(ann['keypoints'], remove_idxs)
        remove_info = {'annotation_keypoints': num_kps_removed}
        return remove_info

    def remove_keypoint_categories(self, kp_identifiers):
        """
        Removes all keypoints of a particular category as well as all
        annotation keypoints with those ids.

        Args:
            kp_identifiers (List): list of keypoint category dicts, names, or ids

        Returns:
            Dict: num_removed: information on the number of items removed

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('shapes', rng=0)
            >>> kp_identifiers = ['left_eye', 'mid_tip']
            >>> remove_info = self.remove_keypoint_categories(kp_identifiers)
            >>> print('remove_info = {!r}'.format(remove_info))
            >>> # FIXME: for whatever reason demodata generation is not determenistic when seeded
            >>> # assert remove_info == {'keypoint_categories': 2, 'annotation_keypoints': 16, 'reflection_ids': 1}
            >>> assert self._resolve_to_kpcat('right_eye')['reflection_id'] is None
        """
        remove_info = {
            'keypoint_categories': None,
            'annotation_keypoints': None
        }
        remove_kpcats = list(map(self._resolve_to_kpcat, kp_identifiers))

        _ann_remove_info = self.remove_annotation_keypoints(remove_kpcats)
        remove_info.update(_ann_remove_info)

        remove_kpcids = {k['id'] for k in remove_kpcats}

        for kpcat in remove_kpcats:
            self.dataset['keypoint_categories'].remove(kpcat)

        # handle reflection ids
        remove_reflect_ids = 0
        for kpcat in self.dataset['keypoint_categories']:
            if kpcat.get('reflection_id', None) in remove_kpcids:
                kpcat['reflection_id'] = None
                remove_reflect_ids += 1

        remove_info['reflection_ids'] = remove_reflect_ids
        remove_info['keypoint_categories'] = len(remove_kpcats)
        return remove_info

    def set_annotation_category(self, aid_or_ann, cid_or_cat):
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
        new_cid = self._resolve_to_cid(cid_or_cat)
        ann = self._resolve_to_ann(aid_or_ann)
        aid = ann['id']
        if self.index:
            if 'category_id' in ann:
                old_cid = ann['category_id']
                self.index.cid_to_aids[old_cid].remove(aid)
        ann['category_id'] = new_cid
        if self.index:
            self.index.cid_to_aids[new_cid].add(aid)
        self._invalidate_hashid(['annotations'])


class CocoIndex(object):
    """
    Fast lookup index for the COCO dataset with dynamic modification

    Attributes:
        imgs (Dict[int, dict]):
            mapping between image ids and the image dictionaries

        anns (Dict[int, dict]):
            mapping between annotation ids and the annotation dictionaries

        cats (Dict[int, dict]):
            mapping between category ids and the category dictionaries

        kpcats (Dict[int, dict]):
            mapping between keypoint category ids and keypoint category
            dictionaries

        gid_to_aids (Dict[int, List[int]]):
            mapping between an image-id and annotation-ids that belong to it

        cid_to_aids (Dict[int, List[int]]):
            mapping between an category-id and annotation-ids that belong to it

        cid_to_gids (Dict[int, List[int]]):
            mapping between an category-id and image-ids that contain
            at least one annotation with this cateogry id.

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

        file_name_to_img (Dict[str, dict]):
            mapping between a image file_name and the image dictionary.
    """

    # _set = ub.oset  # many operations are much slower for oset
    _set = set

    def _images_set_sorted_by_frame_index(index, gids=None):
        """
        Helper for ensuring that vidid_to_gids returns image ids ordered by
        frame index.
        """
        return SortedSet(gids, key=partial(_lut_image_frame_index, index.imgs))

    # Backwards compat
    _set_sorted_by_frame_index = _images_set_sorted_by_frame_index

    def _annots_set_sorted_by_frame_index(index, aids=None):
        """
        Helper for ensuring that vidid_to_gids returns image ids ordered by
        frame index.
        """
        # if aids is None:
        #     return set()
        # return set(aids)
        return SortedSet(aids, key=partial(_lut_annot_frame_index, index.imgs, index.anns))

    def __init__(index):
        index.anns = None
        index.imgs = None
        index.videos = None
        index.cats = None
        index.kpcats = None
        index._id_lookup = None

        index.gid_to_aids = None
        index.cid_to_aids = None
        index.vidid_to_gids = None
        index.trackid_to_aids = None

        index.name_to_video = None
        index.name_to_cat = None
        index.name_to_img = None
        index.file_name_to_img = None
        index._CHECKS = True
        # index.kpcid_to_aids = None  # TODO

    def __bool__(index):
        return index.anns is not None

    # On-demand lookup tables
    @property
    def cid_to_gids(index):
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
            def getitem(self, cid):
                aids = self.index.cid_to_aids[cid]
                gids = {self.index.anns[aid]['image_id'] for aid in aids}
                return gids
            def keys(self):
                return self.index.cid_to_aids.keys()
        cid_to_gids = ProxyCidToGids(index=index)
        return cid_to_gids

    def _add_video(index, vidid, video):
        if index.videos is not None:
            name = video['name']
            if index._CHECKS:
                if name in index.name_to_video:
                    raise exceptions.DuplicateAddError(
                        'video with name={} already exists'.format(name))
            index.videos[vidid] = video
            if vidid not in index.vidid_to_gids:
                index.vidid_to_gids[vidid] = index._images_set_sorted_by_frame_index()
            index.name_to_video[name] = video

    def _add_image(index, gid, img):
        """
        Example:
            >>> # Test adding image to video that doesnt exist
            >>> import kwcoco
            >>> self = dset = kwcoco.CocoDataset()
            >>> dset.add_image(file_name='frame1', video_id=1, frame_index=0)
            >>> dset.add_image(file_name='frame2', video_id=1, frame_index=0)
            >>> dset._check_pointers()
            >>> dset._check_index()
            >>> print('dset.index.vidid_to_gids = {!r}'.format(dset.index.vidid_to_gids))
            >>> assert len(dset.index.vidid_to_gids) == 1
            >>> dset.add_video(name='foo-vid', id=1)
            >>> assert len(dset.index.vidid_to_gids) == 1
            >>> dset._check_pointers()
            >>> dset._check_index()
        """
        if index.imgs is not None:
            file_name = img.get('file_name', None)
            name = img.get('name', None)
            if index._CHECKS:
                if file_name is None and name is None:
                    raise exceptions.InvalidAddError(
                        'at least one of file_name or name must be specified')
                if file_name in index.file_name_to_img:
                    raise exceptions.DuplicateAddError(
                        'image with file_name={} already exists'.format(
                            file_name))
                if name in index.name_to_img:
                    raise exceptions.DuplicateAddError(
                        'image with name={} already exists'.format(name))
            index.imgs[gid] = img
            index.gid_to_aids[gid] = index._set()

            if file_name is not None:
                index.file_name_to_img[file_name] = img

            if name is not None:
                index.name_to_img[name] = img

            if 'video_id' in img:
                if img.get('frame_index', None) is None:
                    raise exceptions.InvalidAddError(
                        'Images with video-ids must have a frame_index')
                vidid = img['video_id']
                try:
                    index.vidid_to_gids[vidid].add(gid)
                except KeyError:
                    # Should warning messages contain data-specific info?
                    # msg = ('Adding image-id={} to '
                    #        'non-existing video-id={}').format(gid, vidid)
                    msg = 'Adding image to non-existing video'
                    warnings.warn(msg)
                    index.vidid_to_gids[vidid] = index._images_set_sorted_by_frame_index()
                    index.vidid_to_gids[vidid].add(gid)

    def _add_images(index, imgs):
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
            index.file_name_to_img.update(
                {img['file_name']: img for img in imgs
                 if img.get('file_name', None) is not None})
            index.name_to_img.update(
                {img['name']: img for img in imgs
                 if img.get('name', None) is not None})
            for gid in gids:
                index.gid_to_aids[gid] = index._set()

            if index.vidid_to_gids:
                vidid_to_gids = ub.group_items(
                    [g['id'] for g in imgs],
                    [g.get('video_id', None) for g in imgs]
                )
                vidid_to_gids.pop(None, None)
                for vidid, gids in vidid_to_gids.items():
                    index.vidid_to_gids[vidid].update(gids)

    def _add_annotation(index, aid, gid, cid, tid, ann):
        if index.anns is not None:
            index.anns[aid] = ann
            # Note: it should be ok to have None's here
            index.gid_to_aids[gid].add(aid)
            index.cid_to_aids[cid].add(aid)
            # index.trackid_to_aids[tid].add(aid)
            try:
                index.trackid_to_aids[tid].add(aid)
            except KeyError:
                # Be careful to not apply sorting to trackless annotations
                if tid is None:
                    index.trackid_to_aids[tid] = set()
                else:
                    index.trackid_to_aids[tid] = index._annots_set_sorted_by_frame_index()
                index.trackid_to_aids[tid].add(aid)

    def _add_annotations(index, anns):
        if index.anns is not None:
            aids = [ann['id'] for ann in anns]
            gids = [ann['image_id'] for ann in anns]
            cids = [ann['category_id'] for ann in anns]
            tids = [ann.get('track_id', None) for ann in anns]
            new_anns = dict(zip(aids, anns))
            index.anns.update(new_anns)
            for gid, cid, tid, aid in zip(gids, cids, tids, aids):
                index.gid_to_aids[gid].add(aid)
                index.cid_to_aids[cid].add(aid)
                try:
                    index.trackid_to_aids[tid].add(aid)
                except KeyError:
                    if tid is None:
                        index.trackid_to_aids[tid] = set()
                    else:
                        index.trackid_to_aids[tid] = index._annots_set_sorted_by_frame_index()
                    index.trackid_to_aids[tid].add(aid)

    def _add_category(index, cid, name, cat):
        if index.cats is not None:
            index.cats[cid] = cat
            index.cid_to_aids[cid] = index._set()
            index.name_to_cat[name] = cat

    def _remove_all_annotations(index):
        # Keep the category and image indexes alive
        if index.anns is not None:
            for _ in index.gid_to_aids.values():
                _.clear()
            for _ in index.cid_to_aids.values():
                _.clear()
            for _ in index.trackid_to_aids.values():
                _.clear()
            index.anns.clear()

    def _remove_all_images(index):
        # Keep the category indexes alive
        if index.imgs is not None:
            index.imgs.clear()
            index.anns.clear()
            index.gid_to_aids.clear()
            index.file_name_to_img.clear()
            for _ in index.cid_to_aids.values():
                _.clear()
            for _ in index.vidid_to_gids.values():
                _.clear()

    def _remove_annotations(index, remove_aids, verbose=0):
        if index.anns is not None:
            if verbose > 1:
                print('Updating annotation index')
            # This is faster for simple set cid_to_aids
            for aid in remove_aids:
                ann = index.anns[aid]
                gid = ann['image_id']
                cid = ann.get('category_id', None)
                track_id = ann.get('track_id', None)
                index.trackid_to_aids[track_id].remove(aid)
                index.cid_to_aids[cid].remove(aid)
                index.gid_to_aids[gid].remove(aid)
                index.anns.pop(aid)

    def _remove_categories(index, remove_cids, verbose=0):
        # dynamically update the category index
        if index.cats is not None:
            for cid in remove_cids:
                cat = index.cats.pop(cid)
                del index.cid_to_aids[cid]
                del index.name_to_cat[cat['name']]
            if verbose > 2:
                print('Updated category index')

    def _remove_images(index, remove_gids, verbose=0):
        # dynamically update the image index
        if index.imgs is not None:
            for gid in remove_gids:
                img = index.imgs[gid]
                vidid = img.get('video_id', None)
                if vidid in index.vidid_to_gids:
                    index.vidid_to_gids[vidid].remove(gid)
                del index.gid_to_aids[gid]
                gname = img.get('file_name', None)
                if gname is not None:
                    del index.file_name_to_img[gname]
                name = img.get('name', None)
                if name is not None:
                    del index.name_to_img[name]
                del index.imgs[gid]
            if verbose > 2:
                print('Updated image index')

    def _remove_videos(index, remove_vidids, verbose=0):
        # dynamically update the video index
        lut = index.videos
        if lut is not None:
            for item_id in remove_vidids:
                item = lut.pop(item_id)
                del index.vidid_to_gids[item_id]
                if index.name_to_video is not None:
                    del index.name_to_video[item['name']]
            if verbose > 2:
                print('Updated video index')

    def clear(index):
        index.anns = None
        index.imgs = None
        index.videos = None
        index.cats = None
        index.kpcats = None
        index._id_lookup = None
        index.gid_to_aids = None
        index.vidid_to_gids = None
        index.cid_to_aids = None
        index.name_to_cat = None
        index.file_name_to_img = None
        index.name_to_video = None
        index.trackid_to_aids = None
        # index.kpcid_to_aids = None  # TODO

    def build(index, parent):
        """
        Build all id-to-obj reverse indexes from scratch.

        Args:
            parent (kwcoco.CocoDataset): the dataset to index

        Notation:
            aid - Annotation ID
            gid - imaGe ID
            cid - Category ID
            vidid - Video ID

        Example:
            >>> import kwcoco
            >>> parent = kwcoco.CocoDataset.demo('vidshapes1', num_frames=4, rng=1)
            >>> index = parent.index
            >>> index.build(parent)
        """
        # create index
        anns, cats, imgs = {}, {}, {}
        videos = {}

        # Build one-to-one index-lookup maps
        for cat in parent.dataset.get('categories', []):
            cid = cat['id']
            if cid in cat:
                warnings.warn(
                    'Categories have the same id in {}:\n{} and\n{}'.format(
                        parent, cats[cid], cat))
            cats[cid] = cat

        for video in parent.dataset.get('videos', []):
            vidid = video['id']
            if vidid in videos:
                warnings.warn(
                    'Video has the same id in {}:\n{} and\n{}'.format(
                        parent, videos[vidid], video))
            videos[vidid] = video

        for img in parent.dataset.get('images', []):
            gid = img['id']
            if gid in imgs:
                warnings.warn(
                    'Images have the same id in {}:\n{} and\n{}'.format(
                        parent, imgs[gid], img))
            imgs[gid] = img

        for ann in parent.dataset.get('annotations', []):
            aid = ann['id']
            if aid in anns:
                warnings.warn(
                    'Annotations at index {} and {} '
                    'have the same id in {}:\n{} and\n{}'.format(
                        parent.dataset['annotations'].index(anns[aid]),
                        parent.dataset['annotations'].index(ann),
                        parent, anns[aid], ann))
            anns[aid] = ann

        # Build one-to-many lookup maps
        vidid_to_gids = ub.group_items(
            [g['id'] for g in imgs.values()],
            [g.get('video_id', None) for g in imgs.values()]
        )
        vidid_to_gids.pop(None, None)

        if 0:
            # The following is slightly slower, but it is also many fewer lines
            # Not sure if its correct to replace the else block or not
            aids = [d['id'] for d in anns.values()]
            gid_to_aids = ub.group_items(aids, (d['image_id'] for d in anns.values()))
            cid_to_aids = ub.group_items(aids, (d.get('category_id', None) for d in anns.values()))
            trackid_to_aids = ub.group_items(aids, (d.get('track_id', None) for d in anns.values()))
            cid_to_aids.pop(None, None)
            gid_to_aids = ub.map_vals(index._set, gid_to_aids)
            cid_to_aids = ub.map_vals(index._set, cid_to_aids)
            trackid_to_aids = ub.map_vals(index._set, trackid_to_aids)
            vidid_to_gids = ub.map_vals(index._images_set_sorted_by_frame_index, vidid_to_gids)
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
                    warnings.warn('Annotation {} in {} references '
                                  'unknown image_id'.format(ann, parent))

                try:
                    cid = ann['category_id']
                except KeyError:
                    warnings.warn('Annotation {} in {} is missing '
                                  'a category_id'.format(ann, parent))
                else:
                    cid_to_aids[cid].add(aid)

                    if not isinstance(cid, numbers.Integral) and cid is not None:
                        raise TypeError('bad cid={} type={}'.format(cid, type(cid)))

                    if cid not in cats and cid is not None:
                        warnings.warn('Annotation {} in {} references '
                                      'unknown category_id'.format(ann, parent))

                tid = ann.get('track_id', None)
                trackid_to_aids[tid].add(aid)

        # Fix one-to-zero cases
        for cid in cats.keys():
            if cid not in cid_to_aids:
                cid_to_aids[cid] = index._set()

        for gid in imgs.keys():
            if gid not in gid_to_aids:
                gid_to_aids[gid] = index._set()

        for vidid in videos.keys():
            if vidid not in vidid_to_gids:
                vidid_to_gids[vidid] = index._set()

        # create class members
        index._id_lookup = {
            'categories': cats,
            'images': imgs,
            'annotations': anns,
            'videos': videos,
        }
        index.anns = anns
        index.imgs = imgs
        index.cats = cats
        index.kpcats = None  # TODO
        index.videos = videos

        # Remove defaultdict like behavior
        gid_to_aids.default_factory = None

        # Actually, its important to have defaultdict like behavior for
        # categories so we can allow for the category_id=None case
        # cid_to_aids.default_factory = None
        # vidid_to_gids.default_factory = None

        # Ensure that the values are cast to the appropriate set type
        # This needs to happen after index.imgs is populated
        vidid_to_gids = ub.map_vals(index._images_set_sorted_by_frame_index, vidid_to_gids)

        # Be careful to not apply sorting to trackless annotations
        _notrack_aids = trackid_to_aids.pop(None, None)
        trackid_to_aids = ub.map_vals(index._annots_set_sorted_by_frame_index, trackid_to_aids)
        if _notrack_aids is not None:
            trackid_to_aids[None] = set(_notrack_aids)

        index.gid_to_aids = gid_to_aids
        index.cid_to_aids = cid_to_aids
        index.vidid_to_gids = vidid_to_gids
        index.trackid_to_aids = trackid_to_aids

        index.name_to_cat = {cat['name']: cat for cat in index.cats.values()}
        index.file_name_to_img = {
            img['file_name']: img for img in index.imgs.values()
            if img.get('file_name', None) is not None
        }
        index.name_to_img = {
            img['name']: img for img in index.imgs.values()
            if img.get('name', None) is not None
        }
        index.name_to_video = {
            video['name']: video for video in index.videos.values()
            if video.get('name', None) is not None
        }


class MixinCocoIndex(object):
    """
    Give the dataset top level access to index attributes
    """
    @property
    def anns(self):
        return self.index.anns

    @property
    def imgs(self):
        return self.index.imgs

    @property
    def cats(self):
        return self.index.cats

    # NOTE: API Issue, overloads previous method
    # @property
    # def videos(self):
    #     return self.index.videos

    @property
    def gid_to_aids(self):
        return self.index.gid_to_aids

    @property
    def cid_to_aids(self):
        return self.index.cid_to_aids

    @property
    def name_to_cat(self):
        return self.index.name_to_cat


class CocoDataset(AbstractCocoDataset, MixinCocoAddRemove, MixinCocoStats,
                  MixinCocoObjects, MixinCocoDraw,
                  MixinCocoAccessors, MixinCocoExtras, MixinCocoIndex,
                  MixinCocoDepricate, ub.NiceRepr):
    """
    The main coco dataset class with a json dataset backend.

    Attributes:
        dataset (Dict): raw json data structure. This is the base dictionary
            that contains {'annotations': List, 'images': List,
            'categories': List}

        index (CocoIndex): an efficient lookup index into the coco data
            structure. The index defines its own attributes like
            ``anns``, ``cats``, ``imgs``, ``gid_to_aids``,
            ``file_name_to_img``, etc. See :class:`CocoIndex` for more details
            on which attributes are available.

        fpath (PathLike | None):
            if known, this stores the filepath the dataset was loaded from

        tag (str | None):
            A tag indicating the name of the dataset.

        bundle_dpath (PathLike | None) :
            If known, this is the root path that all image file names are
            relative to. This can also be manually overwritten by the user.

        hashid (str | None) :
            If computed, this will be a hash uniquely identifing the dataset.
            To ensure this is computed see
            :func:`kwcoco.coco_dataset.MixinCocoExtras._build_hashid`.

    References:
        http://cocodataset.org/#format
        http://cocodataset.org/#download

    CommandLine:
        python -m kwcoco.coco_dataset CocoDataset --show

    Example:
        >>> from kwcoco.coco_dataset import demo_coco_data
        >>> import kwcoco
        >>> import ubelt as ub
        >>> # Returns a coco json structure
        >>> dataset = demo_coco_data()
        >>> # Pass the coco json structure to the API
        >>> self = kwcoco.CocoDataset(dataset, tag='demo')
        >>> # Now you can access the data using the index and helper methods
        >>> #
        >>> # Start by looking up an image by it's COCO id.
        >>> image_id = 1
        >>> img = self.index.imgs[image_id]
        >>> print(ub.urepr(img, nl=1, sort=1))
        {
            'file_name': 'astro.png',
            'id': 1,
            'url': 'https://i.imgur.com/KXhKM72.png',
        }
        >>> #
        >>> # Use the (gid_to_aids) index to lookup annotations in the iamge
        >>> annotation_id = sorted(self.index.gid_to_aids[image_id])[0]
        >>> ann = self.index.anns[annotation_id]
        >>> print(ub.urepr((ub.udict(ann) - {'segmentation'}).sorted_keys(), nl=1))
        {
            'bbox': [10, 10, 360, 490],
            'category_id': 1,
            'id': 1,
            'image_id': 1,
            'keypoints': [247, 101, 2, 202, 100, 2],
        }
        >>> #
        >>> # Use annotation category id to look up that information
        >>> category_id = ann['category_id']
        >>> cat = self.index.cats[category_id]
        >>> print('cat = {}'.format(ub.urepr(cat, nl=1, sort=1)))
        cat = {
            'id': 1,
            'name': 'astronaut',
            'supercategory': 'human',
        }
        >>> #
        >>> # Now play with some helper functions, like extended statistics
        >>> extended_stats = self.extended_stats()
        >>> # xdoctest: +IGNORE_WANT
        >>> print('extended_stats = {}'.format(ub.urepr(extended_stats, nl=1, precision=2, sort=1)))
        extended_stats = {
            'annots_per_img': {'mean': 3.67, 'std': 3.86, 'min': 0.00, 'max': 9.00, 'nMin': 1, 'nMax': 1, 'shape': (3,)},
            'imgs_per_cat': {'mean': 0.88, 'std': 0.60, 'min': 0.00, 'max': 2.00, 'nMin': 2, 'nMax': 1, 'shape': (8,)},
            'cats_per_img': {'mean': 2.33, 'std': 2.05, 'min': 0.00, 'max': 5.00, 'nMin': 1, 'nMax': 1, 'shape': (3,)},
            'annots_per_cat': {'mean': 1.38, 'std': 1.49, 'min': 0.00, 'max': 5.00, 'nMin': 2, 'nMax': 1, 'shape': (8,)},
            'imgs_per_video': {'empty_list': True},
        }
        >>> # You can "draw" a raster of the annotated image with cv2
        >>> canvas = self.draw_image(2)
        >>> # Or if you have matplotlib you can "show" the image with mpl objects
        >>> # xdoctest: +REQUIRES(--show)
        >>> from matplotlib import pyplot as plt
        >>> fig = plt.figure()
        >>> ax1 = fig.add_subplot(1, 2, 1)
        >>> self.show_image(gid=2)
        >>> ax2 = fig.add_subplot(1, 2, 2)
        >>> ax2.imshow(canvas)
        >>> ax1.set_title('show with matplotlib')
        >>> ax2.set_title('draw with cv2')
        >>> plt.show()
    """

    def __init__(self, data=None, tag=None, bundle_dpath=None, img_root=None,
                 fname=None, autobuild=True):
        """
        Args:

            data (str | PathLike | dict | None):
                Either a filepath to a coco json file, or a dictionary
                containing the actual coco json structure. For a more generally
                coercable constructor see func:`CocoDataset.coerce`.

            tag (str | None) :
                Name of the dataset for display purposes, and does not
                influence behavior of the underlying data structure, although
                it may be used via convinience methods. We attempt to
                autopopulate this via information in ``data`` if available.
                If unspecfied and ``data`` is a filepath this becomes the
                basename.

            bundle_dpath (str | None):
                the root of the dataset that images / external data will be
                assumed to be relative to. If unspecfied, we attempt to
                determine it using information in ``data``. If ``data`` is a
                filepath, we use the dirname of that path. If ``data`` is a
                dictionary, we look for the "img_root" key. If unspecfied and
                we fail to introspect then, we fallback to the current working
                directory.

            img_root (str | None):
                deprecated alias for bundle_dpath
        """
        self._fpath = None

        # Info about what was the origin of this object and if anything
        # happened to it over its lifetime.
        self._state = {
            'was_loaded': False,
            'was_saved': False,
            'was_modified': 0,
        }

        if img_root is not None:
            bundle_dpath = img_root
        if data is None:
            # TODO: rely on subset of SPEC keys
            data = {
                'categories': [],
                'videos': [],
                'images': [],
                'annotations': [],
                'tracks': [],
                'licenses': [],
                'info': [],
            }

        fpath = None
        inferred_date_type = None
        if isinstance(data, dict):
            # Assumption: If data is a dict and are not explicitly given
            # bundle_dpath, then we assume it is relative to the cwd.
            assumed_root = '.'
            inferred_date_type = 'json-dict'
        elif isinstance(data, (str, os.PathLike)):
            path = os.fspath(data)
            if isdir(path):
                # data was a pointer to hopefully a kwcoco bundle
                if bundle_dpath is None:
                    bundle_dpath = path
                else:
                    if bundle_dpath != path:
                        raise Exception('ambiguous')
                inferred_date_type = 'bundle-path'
            else:
                # data was a pointer to hopefully a kwcoco filepath
                # TODO: do some validation of that here
                fpath = data
                if bundle_dpath is None:
                    bundle_dpath = dirname(fpath)
                inferred_date_type = 'file-path'
        else:
            raise TypeError(
                'data must be a dict or path to json file, '
                'but got: {!r}'.format(type(data)))

        if fpath is None and bundle_dpath is not None and inferred_date_type == 'bundle-path':
            # This should probably be reserved for a coercion method
            # If we are givena bundle path, assume a standard name convention
            if fname is None:
                import glob
                candidates = [
                    'data',
                    'data.json',
                    'data.kwcoco.json',
                    'data.kwcoco.zip',  # Allow zipfiles
                    'data.kwcoco.json.zip',
                    '*.kwcoco.json',
                    '*.mscoco.json',
                ]
                # Check for standard bundle manifest names
                manifest_candidate_iter = iter(ub.oset(ub.flatten([
                    glob.glob(join(bundle_dpath, p))
                    for p in candidates])))
                try:
                    fpath = ub.peek(manifest_candidate_iter)
                except StopIteration:
                    fpath = join(bundle_dpath, 'data.kwcoco.json')
                    # raise Exception('No manifest in Dataset Bundle')
                else:
                    remain = list(manifest_candidate_iter)
                    if len(remain) > 0:
                        raise Exception('Ambiguous Dataset Bundle {}: {}'.format(fpath, remain))
            else:
                fpath = join(bundle_dpath, fname)
            key = basename(bundle_dpath)

        if fpath is not None:
            fname = basename(fpath)
            if fname == 'data.kwcoco.json':
                if bundle_dpath is not None:
                    bundle_dpath = dirname(fpath)
                key = basename(bundle_dpath)
            else:
                key = fname

        if bundle_dpath is not None:
            assumed_root = bundle_dpath

        if isinstance(data, (str, os.PathLike)):
            if not exists(fpath):
                raise Exception(ub.paragraph(
                    '''
                    Specified fpath={} does not exist. If you are trying
                    to create a new dataset fist create a CocoDataset without
                    any arguments, and then set the fpath attribute.
                    We may loosen this requirement in the future.
                    ''').format(fpath))

            self._state['was_loaded'] = True

            # Test to see if the kwcoco file is compressed
            import zipfile
            if zipfile.is_zipfile(fpath):
                with open(fpath, 'rb') as file:
                    with zipfile.ZipFile(file, 'r') as zfile:
                        members = zfile.namelist()
                        if len(members) != 1:
                            raise Exception(
                                'Currently only zipfiles with exactly 1 '
                                'kwcoco member are supported')
                        text = zfile.read(members[0]).decode('utf8')
                        data = json_r.loads(text)
            else:
                with open(fpath, 'r') as file:
                    data = json_r.load(file)

            # If data is a path it gives us the absolute location of the root
            if tag is None:
                tag = key

        # Backwards compat hack, allow the coco file to specify the
        # bundle_dpath
        # TODO: deprecate img_root
        if isinstance(data, dict) and 'img_root' in data:
            ub.schedule_deprecation(
                'kwcoco', name='img_root', type='dataset member',
                deprecate='0.6.3', error='1.0.0', remove='1.1.0',
                migration=ub.paragraph(
                    '''
                    Ensure the location of the saved kwcoco file encodes the
                    bundle dpath or ensure bundle_dpath is correctly set in the
                    in-memory CocoDataset object.
                    ''')
            )
            # allow image root to be specified in the dataset
            # we refer to this as a json data "body root".
            body_root = data.get('img_root', '')
            if body_root is None:
                body_root = ''
            elif isinstance(body_root, str):
                _tmp = ub.expandpath(body_root)
                if exists(_tmp):
                    body_root = _tmp
            else:
                if isinstance(body_root, list) and body_root == []:
                    body_root = ''
                else:
                    raise TypeError('body_root = {!r}'.format(body_root))
            try:
                bundle_dpath = join(assumed_root, body_root)
            except Exception:
                print('body_root = {!r}'.format(body_root))
                print('assumed_root = {!r}'.format(assumed_root))
                raise

        if bundle_dpath is None:
            bundle_dpath = assumed_root

        bundle_dpath = ub.expandpath(bundle_dpath)

        if fpath is None:
            fpath = join(bundle_dpath, 'data.kwcoco.json')

        self.index = CocoIndex()

        self.hashid = None
        self.hashid_parts = None

        self.tag = tag
        self.dataset = data

        self.data_fpath = fpath
        self.bundle_dpath = bundle_dpath

        self.cache_dpath = None
        self.assets_dpath = None

        # Keep track of an unused id we may use
        self._next_ids = _NextId(self)

        self._infer_dirs()

        if autobuild:
            self._build_index()

    @property
    def fpath(self):
        """ In the future we will deprecate img_root for bundle_dpath """
        return self._fpath

    @fpath.setter
    def fpath(self, value):
        # Cant use update fpath because of reroot checks.
        # self._update_fpath(value)
        self._fpath = value
        self._infer_dirs()

    def _update_fpath(self, new_fpath):
        # New method for more robustly updating the file path and bundle
        # directory, still a WIP. Only works when the current dataset is
        # already valid.
        if new_fpath is None:
            # Bundle directory is clobbered, so we should make everything
            # absolute
            self.reroot(absolute=True)
        else:
            old_fpath = self.fpath
            if old_fpath is not None:
                old_fpath_ = ub.Path(old_fpath)
                new_fpath_ = ub.Path(new_fpath)

                same_bundle = (
                    (old_fpath_.parent == new_fpath_.parent) or
                    (old_fpath_.resolve() == new_fpath_.resolve())
                )
                if not same_bundle:
                    # The bundle directory has changed, so we need to reroot
                    new_root = new_fpath_.parent
                    self.reroot(new_root)

            self._fpath = new_fpath
            self._infer_dirs()

    def _infer_dirs(self):
        """
        Ignore:
            self = dset
        """
        data_fpath = self.fpath
        if data_fpath is not None:
            bundle_dpath = dirname(data_fpath)
            assets_dpath = join(bundle_dpath, '_assets')
            cache_dpath = join(bundle_dpath, '_cache')
            # OPINION: Do we want conditions?
            # data_fname = basename(data_fpath)
            # bundle_conditions = {
            #     # 'name': data_fname == 'data.kwcoco.json',
            #     # 'ext': data_fname.endswith('.kwcoco.json'),
            #     # 'has_assets': exists(assets_dpath),
            # }
            # is_bundle = all(bundle_conditions.values())
            self.bundle_dpath = bundle_dpath
            self.assets_dpath = assets_dpath
            self.cache_dpath = cache_dpath

    @classmethod
    def from_data(CocoDataset, data, bundle_dpath=None, img_root=None):
        """
        Constructor from a json dictionary
        """
        coco_dset = CocoDataset(data, bundle_dpath=bundle_dpath,
                                img_root=img_root)
        return coco_dset

    @classmethod
    def from_image_paths(CocoDataset, gpaths, bundle_dpath=None,
                         img_root=None):
        """
        Constructor from a list of images paths.

        This is a convinience method.

        Args:
            gpaths (List[str]): list of image paths

        Example:
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.from_image_paths(['a.png', 'b.png'])
            >>> assert coco_dset.n_images == 2
        """
        coco_dset = CocoDataset(bundle_dpath=bundle_dpath, img_root=img_root)
        for gpath in gpaths:
            coco_dset.add_image(gpath)
        return coco_dset

    @classmethod
    def coerce_multiple(cls, datas, workers=0, mode='process', verbose=1,
                        postprocess=None, ordered=True, **kwargs):
        """
        Coerce multiple CocoDataset objects in parallel.

        Args:
            datas (List): list of kwcoco coercables to load

            workers (int | str): number of worker threads / processes.
                Can also accept coerceable workers.

            mode (str): thread, process, or serial. Defaults to process.

            verbose (int): verbosity level

            postprocess (Callable | None):
                A function taking one arg (the loaded dataset) to run on the
                loaded kwcoco dataset in background workers. This can be more
                efficient when postprocessing is independent per kwcoco file.

            ordered (bool):
                if True yields datasets in the same order as given. Otherwise
                results are yielded as they become available. Defaults to True.

            **kwargs:
                arguments passed to the constructor

        Yields:
            CocoDataset

        SeeAlso:
            * load_multiple - like this function but is a strict file-path-only loader

        CommandLine:
            xdoctest -m kwcoco.coco_dataset CocoDataset.coerce_multiple

        Example:
            >>> import kwcoco
            >>> dset1 = kwcoco.CocoDataset.demo('shapes1')
            >>> dset2 = kwcoco.CocoDataset.demo('shapes2')
            >>> dset3 = kwcoco.CocoDataset.demo('vidshapes8')
            >>> dsets = [dset1, dset2, dset3]
            >>> input_fpaths = [d.fpath for d in dsets]
            >>> results = list(kwcoco.CocoDataset.coerce_multiple(input_fpaths, ordered=True))
            >>> result_fpaths = [r.fpath for r in results]
            >>> assert result_fpaths == input_fpaths
            >>> # Test unordered
            >>> results1 = list(kwcoco.CocoDataset.coerce_multiple(input_fpaths, ordered=False))
            >>> result_fpaths = [r.fpath for r in results]
            >>> assert set(result_fpaths) == set(input_fpaths)
            >>> #
            >>> # Coerce from existing datasets
            >>> results2 = list(kwcoco.CocoDataset.coerce_multiple(dsets, ordered=True, workers=0))
            >>> assert results2[0] is dsets[0]
        """
        import kwcoco
        from kwcoco.util.util_parallel import coerce_num_workers
        _loader = kwcoco.CocoDataset.coerce
        workers = coerce_num_workers(workers)
        workers = min(workers, len(datas))
        # Reuse coerce_multiple logic but overload the loader function.
        yield from cls._load_multiple(_loader, datas, workers=workers,
                                      mode=mode, verbose=verbose,
                                      postprocess=postprocess, ordered=ordered,
                                      **kwargs)

    @classmethod
    def load_multiple(cls, fpaths, workers=0, mode='process', verbose=1,
                      postprocess=None, ordered=True, **kwargs):
        """
        Load multiple CocoDataset objects in parallel.

        Args:
            fpaths (List[str | PathLike]):
                list of paths to multiple coco files to be loaded

            workers (int): number of worker threads / processes

            mode (str): thread, process, or serial. Defaults to process.

            verbose (int): verbosity level

            postprocess (Callable | None):
                A function taking one arg (the loaded dataset) to run on the
                loaded kwcoco dataset in background workers and returns the
                modified dataset. This can be more efficient when
                postprocessing is independent per kwcoco file.

            ordered (bool):
                if True yields datasets in the same order as given. Otherwise
                results are yielded as they become available. Defaults to True.

            **kwargs:
                arguments passed to the constructor

        Yields:
            CocoDataset

        SeeAlso:
            * coerce_multiple - like this function but accepts general
                coercable inputs.
        """
        import kwcoco
        _loader = kwcoco.CocoDataset
        # Reuse coerce_multiple logic but overload the loader function.
        yield from cls._load_multiple(_loader, fpaths, workers=workers,
                                      mode=mode, verbose=verbose,
                                      postprocess=postprocess, ordered=ordered,
                                      **kwargs)

    @classmethod
    def _load_multiple(cls, _loader, inputs, workers=0, mode='process',
                       verbose=1, postprocess=None, ordered=True, **kwargs):
        """
        Shared logic for multiprocessing loaders.

        SeeAlso:
            * coerce_multiple
            * load_multiple
        """
        _submit_prog = ub.ProgIter(inputs, desc='submit load kwcoco jobs',
                                   enabled=workers > 0, verbose=verbose)
        executor = ub.Executor(mode=mode, max_workers=workers)
        with executor:
            jobs = []
            for job_idx, data in enumerate(_submit_prog):
                job = executor.submit(
                    _load_and_postprocess,
                    data=data,
                    loader=_loader,
                    postprocess=postprocess, **kwargs)
                job.job_idx = job_idx
                jobs.append(job)

            if ordered:
                _jobiter = jobs
            else:
                from concurrent.futures import as_completed
                _jobiter = as_completed(jobs)

            _collect_prog = ub.ProgIter(_jobiter, total=len(jobs),
                                        desc='loading kwcoco files',
                                        verbose=verbose)
            for job in _collect_prog:
                # Clear the reference to this job
                jobs[job.job_idx] = None
                yield job.result()

    @classmethod
    def from_coco_paths(CocoDataset, fpaths, max_workers=0, verbose=1,
                        mode='thread', union='try'):
        """
        Constructor from multiple coco file paths.

        Loads multiple coco datasets and unions the result

        Note:
            if the union operation fails, the list of individually loaded files
            is returned instead.

        Args:
            fpaths (List[str]): list of paths to multiple coco files to be
                loaded and unioned.

            max_workers (int): number of worker threads / processes

            verbose (int): verbosity level

            mode (str): thread, process, or serial

            union (str | bool): If True, unions the result
                datasets after loading. If False, just returns the result list.
                If 'try', then try to preform the union, but return the result
                list if it fails. Default='try'

        Note:
            This may be deprecated. Use load_multiple or coerce_multiple and
            then manually perform the union.
        """
        results = CocoDataset.load_multiple(
            fpaths, workers=max_workers, verbose=verbose, mode=mode,
            ordered=False, autobuild=False)

        results = list(results)

        if union:
            try:
                if verbose:
                    # TODO: it would be nice if we had a way to combine results
                    # on the fly, so we can work while the remaining io jobs
                    # are loading
                    print('combining results')
                coco_dset = CocoDataset.union(*results)
            except Exception as ex:
                if union == 'try':
                    warnings.warn(
                        'Failed to union coco results: {!r}'.format(ex))
                    return results
                else:
                    raise
            else:
                return coco_dset
        else:
            return results

    def copy(self):
        """
        Deep copies this object

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> new = self.copy()
            >>> assert new.imgs[1] is new.dataset['images'][0]
            >>> assert new.imgs[1] == self.dataset['images'][0]
            >>> assert new.imgs[1] is not self.dataset['images'][0]
        """
        new = copy.copy(self)
        new.index = CocoIndex()
        new.hashid_parts = copy.deepcopy(self.hashid_parts)
        new.dataset = copy.deepcopy(self.dataset)
        new._next_ids = _NextId(new)
        new._build_index()
        return new

    def __nice__(self):
        parts = []
        parts.append('tag={}'.format(self.tag))
        if self.dataset is not None:
            info = ub.urepr(self.basic_stats(), kvsep='=', si=1, nobr=1, nl=0)
            parts.append(info)
        return ', '.join(parts)

    def dumps(self, indent=None, newlines=False):
        """
        Writes the dataset out to the json format

        Args:
            newlines (bool) :
                if True, each annotation, image, category gets its own line

            indent (int | str | None): indentation for the json file. See
                :func:`json.dump` for details.

            newlines (bool):
                if True, each annotation, image, category gets its own line.

        Note:
            Using newlines=True is similar to:
                print(ub.urepr(dset.dataset, nl=2, trailsep=False))
                However, the above may not output valid json if it contains
                ndarrays.

        Example:
            >>> import kwcoco
            >>> import json
            >>> self = kwcoco.CocoDataset.demo()
            >>> text = self.dumps(newlines=True)
            >>> print(text)
            >>> self2 = kwcoco.CocoDataset(json.loads(text), tag='demo2')
            >>> assert self2.dataset == self.dataset
            >>> assert self2.dataset is not self.dataset

            >>> text = self.dumps(newlines=True)
            >>> print(text)
            >>> self2 = kwcoco.CocoDataset(json.loads(text), tag='demo2')
            >>> assert self2.dataset == self.dataset
            >>> assert self2.dataset is not self.dataset

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.coerce('vidshapes1-msi-multisensor', verbose=3)
            >>> self.remove_annotations(self.annots())
            >>> text = self.dumps(newlines=0, indent='  ')
            >>> print(text)
            >>> text = self.dumps(newlines=True, indent='  ')
            >>> print(text)
        """
        from kwcoco.util import util_special_json
        # Instead of using json to dump the whole thing make the text a bit
        # more pretty.
        try:
            if newlines:
                text = util_special_json._special_kwcoco_pretty_dumps_orig(self.dataset)
            else:
                # TODO: do main key sorting here as well
                text = util_special_json._json_dumps(self.dataset, indent=indent)
        except Exception as ex:
            print('Failed to dump ex = {!r}'.format(ex))
            self._check_json_serializable()
            raise

        return text

    def _compress_dump_to_fileptr(self, file, arcname=None, indent=None, newlines=False):
        """
        Experimental method to save compressed kwcoco files, may be folded into
        dump in the future.
        """
        import zipfile
        from kwcoco.util import util_archive
        compression = util_archive._coerce_zipfile_compression('auto')
        zipkw = {
            'compression': compression,
        }
        if sys.version_info[0:2] >= (3, 7):
            zipkw['compresslevel'] = None
        if arcname is None:
            if not self.fpath:
                arcname = '_data.kwcoco.json'
            else:
                # Use the current name of the file to compress
                arcname = basename(self.fpath)
                if arcname.endswith('.zip'):
                    arcname = arcname[:-4]
                    if not arcname.endswith('.json'):
                        arcname = arcname + '.json'
        with zipfile.ZipFile(file, 'w', **zipkw) as zfile:
            text = self.dumps(indent=indent, newlines=newlines)
            zfile.writestr(arcname, text.encode('utf8'))

    def _dump(self, file, indent, newlines, compress):
        """
        Case where we are dumping to an open file pointer.
        We assume this means the dataset has been written to disk.
        """
        if compress:
            self._compress_dump_to_fileptr(
                file, indent=indent, newlines=newlines)
        else:
            if newlines:
                file.write(self.dumps(indent=indent, newlines=newlines))
            else:
                try:
                    json_w.dump(self.dataset, file, indent=indent,
                                ensure_ascii=False)
                except Exception as ex:
                    print('Failed to dump ex = {!r}'.format(ex))
                    self._check_json_serializable()
                    raise
        self._state['was_saved'] = True

    def dump(self, file=None, indent=None, newlines=False, temp_file='auto',
             compress='auto'):
        """
        Writes the dataset out to the json format

        Args:
            file (PathLike | IO | None):
                Where to write the data. Can either be a path to a file or an
                open file pointer / stream. If unspecified, it will be written
                to the current ``fpath`` property.

            indent (int | str | None): indentation for the json file. See
                :func:`json.dump` for details.

            newlines (bool):
                if True, each annotation, image, category gets its own line.

            temp_file (bool | str):
                Argument to :func:`safer.open`.  Ignored if ``file`` is not a
                PathLike object. Defaults to 'auto', which is False on Windows
                and True everywhere else.

            compress (bool | str):
                if True, dumps the kwcoco file as a compressed zipfile.
                In this case a literal IO file object must be opened in binary
                write mode. If auto, then it will default to False unless
                it can introspect the file name and the name ends with .zip

        Example:
            >>> import kwcoco
            >>> import ubelt as ub
            >>> dpath = ub.Path.appdir('kwcoco/demo/dump').ensuredir()
            >>> dset = kwcoco.CocoDataset.demo()
            >>> dset.fpath = dpath / 'my_coco_file.json'
            >>> # Calling dump writes to the current fpath attribute.
            >>> dset.dump()
            >>> assert dset.dataset == kwcoco.CocoDataset(dset.fpath).dataset
            >>> assert dset.dumps() == dset.fpath.read_text()
            >>> #
            >>> # Using compress=True can save a lot of space and it
            >>> # is transparent when reading files via CocoDataset
            >>> dset.dump(compress=True)
            >>> assert dset.dataset == kwcoco.CocoDataset(dset.fpath).dataset
            >>> assert dset.dumps() != dset.fpath.read_text(errors='replace')

        Example:
            >>> import kwcoco
            >>> import ubelt as ub
            >>> # Compression auto-defaults based on the file name.
            >>> dpath = ub.Path.appdir('kwcoco/demo/dump').ensuredir()
            >>> dset = kwcoco.CocoDataset.demo()
            >>> fpath1 = dset.fpath = dpath / 'my_coco_file.zip'
            >>> dset.dump()
            >>> fpath2 = dset.fpath = dpath / 'my_coco_file.json'
            >>> dset.dump()
            >>> assert fpath1.read_bytes()[0:8] != fpath2.read_bytes()[0:8]
        """
        from kwcoco.util.util_json import coerce_indent
        indent = coerce_indent(indent)

        if file is None:
            file = self.fpath

        try:
            fpath = os.fspath(file)
        except TypeError:
            input_was_pathlike = False
        else:
            input_was_pathlike = True

        if compress == 'auto':
            compress = False
            if not input_was_pathlike:
                fpath = getattr(file, 'name', None)
            if fpath is not None:
                if os.fspath(fpath).endswith('.zip'):
                    compress = True

        mode = 'wb' if compress else 'w'

        if input_was_pathlike:
            import safer
            if temp_file == 'auto':
                temp_file = not ub.WIN32

            with safer.open(fpath, mode, temp_file=temp_file) as fp:
                self._dump(
                    fp, indent=indent, newlines=newlines, compress=compress)
        else:
            # We are likely dumping to a real file.
            self._dump(
                file, indent=indent, newlines=newlines, compress=compress)

    def _check_json_serializable(self, verbose=1):
        """
        Debug which part of a coco dataset might not be json serializable
        """
        from kwcoco.util.util_json import find_json_unserializable
        bad_parts_gen = find_json_unserializable(self.dataset)
        bad_parts = []
        for part in bad_parts_gen:
            if verbose == 3:
                print('part = {!r}'.format(part))
            elif verbose and len(bad_parts) == 0:
                # print out the first one we find
                print('Found at least one bad part = {!r}'.format(part))
            bad_parts.append(part)

        if verbose:
            # if bad_parts:
            #     print(ub.urepr(bad_parts))
            summary = 'There are {} total errors'.format(len(bad_parts))
            print('summary = {}'.format(summary))
        return bad_parts

    def _check_integrity(self):
        """ perform all checks """
        self._check_index()
        self._check_pointers()
        self._check_json_serializable()
        self.validate()
        # assert len(self.missing_images()) == 0

    def _check_index(self):
        """
        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> self._check_index()
            >>> # Force a failure
            >>> self.index.anns.pop(1)
            >>> self.index.anns.pop(2)
            >>> import pytest
            >>> with pytest.raises(AssertionError):
            >>>     self._check_index()
        """
        # We can verify our index invariants by copying the raw dataset and
        # checking if the newly constructed index is the same as this index.
        new_dataset = copy.deepcopy(self.dataset)
        new = self.__class__(new_dataset, autobuild=False)
        new._build_index()
        checks = {}
        checks['anns'] = self.index.anns == new.index.anns
        checks['imgs'] = self.index.imgs == new.index.imgs
        checks['cats'] = self.index.cats == new.index.cats
        checks['gid_to_aids'] = self.index.gid_to_aids == new.index.gid_to_aids
        checks['cid_to_aids'] = self.index.cid_to_aids == new.index.cid_to_aids
        checks['name_to_cat'] = self.index.name_to_cat == new.index.name_to_cat
        checks['name_to_img'] = self.index.name_to_img == new.index.name_to_img
        checks['file_name_to_img'] = self.index.file_name_to_img == new.index.file_name_to_img

        one_to_many1 = self.index.trackid_to_aids
        one_to_many2 = new.index.trackid_to_aids

        missing2 = ub.dict_diff(one_to_many1, one_to_many2)
        missing1 = ub.dict_diff(one_to_many2, one_to_many1)
        common1 = ub.dict_isect(one_to_many1, one_to_many2)
        common2 = ub.dict_isect(one_to_many2, one_to_many1)
        checks['trackid_to_aids'] = all([
            all(len(v) == 0 for v in missing1.values()),
            all(len(v) == 0 for v in missing2.values()),
            common1 == common2])
        checks['vidid_to_gids'] = self.index.vidid_to_gids == new.index.vidid_to_gids

        failed_checks = {k: v for k, v in checks.items() if not v}
        if any(failed_checks):
            raise AssertionError(
                'Failed index checks: {}'.format(list(failed_checks)))
        return True

    def _check_pointers(self, verbose=1):
        """
        Check that all category and image ids referenced by annotations exist
        """
        if not self.index:
            raise Exception('Build index before running pointer check')
        errors = []
        annots = self.dataset['annotations']
        iter_ = ub.ProgIter(annots, desc='check annots', enabled=verbose)
        for ann in iter_:
            aid = ann['id']
            cid = ann['category_id']
            gid = ann['image_id']

            if cid not in self.index.cats:
                if cid is not None:
                    errors.append('aid={} references bad cid={}'.format(aid, cid))
            else:
                if self.index.cats[cid]['id'] != cid:
                    errors.append('cid={} has a bad index'.format(cid))

            if gid not in self.index.imgs:
                errors.append('aid={} references bad gid={}'.format(aid, gid))
            else:
                if self.index.imgs[gid]['id'] != gid:
                    errors.append('gid={} has a bad index'.format(gid))

        iter_ = ub.ProgIter(self.dataset['images'], desc='check images', enabled=verbose)
        for img in iter_:
            gid = img['id']
            vidid = img.get('video_id', None)
            if vidid is not None:
                if vidid not in self.index.videos:
                    pass
                    # Dont make this an error because a video dictionary is not
                    # strictly necessary for images to be linked via videos.
                    # We could make this a warning.
                    # errors.append('gid={} references bad video_id={}'.format(gid, vidid))
                elif self.index.videos[vidid]['id'] != vidid:
                    errors.append('video_id={} has a bad index'.format(vidid))

        if errors:
            raise Exception('\n'.join(errors))
        elif verbose:
            print('Pointers are consistent')
        return True

    def _build_index(self):
        self.index.build(self)

    def union(*others, disjoint_tracks=True, remember_parent=False, **kwargs):
        """
        Merges multiple :class:`CocoDataset` items into one. Names and
        associations are retained, but ids may be different.

        Args:
            *others : a series of CocoDatasets that we will merge.
                Note, if called as an instance method, the "self" instance
                will be the first item in the "others" list. But if called
                like a classmethod, "others" will be empty by default.

            disjoint_tracks (bool):
                if True, we will assume track-ids are disjoint and if two
                datasets share the same track-id, we will disambiguate them.
                Otherwise they will be copied over as-is. Defaults to True.

            remember_parent (bool):
                if True, videos and images will save information about their
                parent in the "union_parent" field.

            **kwargs : constructor options for the new merged CocoDataset

        Returns:
            kwcoco.CocoDataset: a new merged coco dataset

        CommandLine:
            xdoctest -m kwcoco.coco_dataset CocoDataset.union

        Example:
            >>> import kwcoco
            >>> # Test union works with different keypoint categories
            >>> dset1 = kwcoco.CocoDataset.demo('shapes1')
            >>> dset2 = kwcoco.CocoDataset.demo('shapes2')
            >>> dset1.remove_keypoint_categories(['bot_tip', 'mid_tip', 'right_eye'])
            >>> dset2.remove_keypoint_categories(['top_tip', 'left_eye'])
            >>> dset_12a = kwcoco.CocoDataset.union(dset1, dset2)
            >>> dset_12b = dset1.union(dset2)
            >>> dset_21 = dset2.union(dset1)
            >>> def add_hist(h1, h2):
            >>>     return {k: h1.get(k, 0) + h2.get(k, 0) for k in set(h1) | set(h2)}
            >>> kpfreq1 = dset1.keypoint_annotation_frequency()
            >>> kpfreq2 = dset2.keypoint_annotation_frequency()
            >>> kpfreq_want = add_hist(kpfreq1, kpfreq2)
            >>> kpfreq_got1 = dset_12a.keypoint_annotation_frequency()
            >>> kpfreq_got2 = dset_12b.keypoint_annotation_frequency()
            >>> assert kpfreq_want == kpfreq_got1
            >>> assert kpfreq_want == kpfreq_got2

            >>> # Test disjoint gid datasets
            >>> dset1 = kwcoco.CocoDataset.demo('shapes3')
            >>> for new_gid, img in enumerate(dset1.dataset['images'], start=10):
            >>>     for aid in dset1.gid_to_aids[img['id']]:
            >>>         dset1.anns[aid]['image_id'] = new_gid
            >>>     img['id'] = new_gid
            >>> dset1.index.clear()
            >>> dset1._build_index()
            >>> # ------
            >>> dset2 = kwcoco.CocoDataset.demo('shapes2')
            >>> for new_gid, img in enumerate(dset2.dataset['images'], start=100):
            >>>     for aid in dset2.gid_to_aids[img['id']]:
            >>>         dset2.anns[aid]['image_id'] = new_gid
            >>>     img['id'] = new_gid
            >>> dset1.index.clear()
            >>> dset2._build_index()
            >>> others = [dset1, dset2]
            >>> merged = kwcoco.CocoDataset.union(*others)
            >>> print('merged = {!r}'.format(merged))
            >>> print('merged.imgs = {}'.format(ub.urepr(merged.imgs, nl=1)))
            >>> assert set(merged.imgs) & set([10, 11, 12, 100, 101]) == set(merged.imgs)

            >>> # Test data is not preserved
            >>> dset2 = kwcoco.CocoDataset.demo('shapes2')
            >>> dset1 = kwcoco.CocoDataset.demo('shapes3')
            >>> others = (dset1, dset2)
            >>> cls = self = kwcoco.CocoDataset
            >>> merged = cls.union(*others)
            >>> print('merged = {!r}'.format(merged))
            >>> print('merged.imgs = {}'.format(ub.urepr(merged.imgs, nl=1)))
            >>> assert set(merged.imgs) & set([1, 2, 3, 4, 5]) == set(merged.imgs)

            >>> # Test track-ids are mapped correctly
            >>> dset1 = kwcoco.CocoDataset.demo('vidshapes1')
            >>> dset2 = kwcoco.CocoDataset.demo('vidshapes2')
            >>> dset3 = kwcoco.CocoDataset.demo('vidshapes3')
            >>> others = (dset1, dset2, dset3)
            >>> for dset in others:
            >>>     [a.pop('segmentation', None) for a in dset.index.anns.values()]
            >>>     [a.pop('keypoints', None) for a in dset.index.anns.values()]
            >>> cls = self = kwcoco.CocoDataset
            >>> merged = cls.union(*others, disjoint_tracks=1)
            >>> print('dset1.anns = {}'.format(ub.urepr(dset1.anns, nl=1)))
            >>> print('dset2.anns = {}'.format(ub.urepr(dset2.anns, nl=1)))
            >>> print('dset3.anns = {}'.format(ub.urepr(dset3.anns, nl=1)))
            >>> print('merged.anns = {}'.format(ub.urepr(merged.anns, nl=1)))

        Example:
            >>> import kwcoco
            >>> # Test empty union
            >>> empty_union = kwcoco.CocoDataset.union()
            >>> assert len(empty_union.index.imgs) == 0

        TODO:
            - [ ] are supercategories broken?
            - [ ] reuse image ids where possible
            - [ ] reuse annotation / category ids where possible
            - [X] handle case where no inputs are given
            - [x] disambiguate track-ids
            - [x] disambiguate video-ids
        """
        from os.path import normpath

        # Dev Note:
        # See ~/misc/tests/python/test_multiarg_classmethod.py
        # for tests for how to correctly implement this method such that it can
        # behave as a method or a classmethod
        if len(others) > 0:
            cls = type(others[0])
        else:
            cls = CocoDataset

        # TODO: add an option such that the union will fail if the names
        # are not already disjoint. Alternatively, it could be the case
        # that a union between images with the same name really does
        # mean that they are the same image.
        unique_img_names = UniqueNameRemapper()
        unique_video_names = UniqueNameRemapper()

        def update_ifnotin(d1, d2):
            """ copies keys from d2 that doent exist in d1 into d1 """
            for k, v in d2.items():
                if k not in d1:
                    d1[k] = v
            return d1

        def _has_duplicates(items):
            seen = set()
            for item in items:
                if item in seen:
                    return True
                seen.add(item)
            return False

        def _coco_union(relative_dsets, common_root):
            """ union of dictionary based data structure """
            # TODO: rely on subset of SPEC keys
            merged = _dict([
                ('licenses', []),
                ('info', []),
                ('categories', []),
                ('videos', []),
                ('images', []),
                ('annotations', []),
            ])

            # TODO: need to handle keypoint_categories
            merged_cat_name_to_id = {}
            merged_kp_name_to_id = {}

            # Check if the image-ids are unique and can be preserved
            _all_imgs = (img for _, _, d in relative_dsets for img in d['images'])
            _all_gids = (img['id'] for img in _all_imgs)
            preserve_gids = not _has_duplicates(_all_gids)

            # Check if the video-ids are unique and can be preserved
            _all_videos = (video for _, _, d in relative_dsets
                           for video in (d.get('videos', None) or []))
            _all_vidids = (video['id'] for video in _all_videos)
            preserve_vidids = not _has_duplicates(_all_vidids)

            # If disjoint_tracks is True keep track of track-ids we've seen in
            # so far in previous datasets and ensure we dont reuse them.
            # TODO: do this Remapper class with other ids?
            track_id_map = _ID_Remapper(reuse=False)

            for subdir, old_fpath, old_dset in relative_dsets:
                # Create temporary indexes to map from old to new
                cat_id_map = {None: None}
                img_id_map = {}
                video_id_map = {}
                kpcat_id_map = {}

                # Add the licenses / info into the merged dataset
                # Licenses / info are unused in our datas, so this might not be
                # correct
                merged['licenses'].extend(old_dset.get('licenses', []))
                merged['info'].extend(old_dset.get('info', []))

                # Add the categories into the merged dataset
                for old_cat in old_dset['categories']:
                    new_id = merged_cat_name_to_id.get(old_cat['name'], None)
                    # The same category might exist in different datasets.
                    if new_id is None:
                        # Only add if it does not yet exist
                        new_id = len(merged_cat_name_to_id) + 1
                        merged_cat_name_to_id[old_cat['name']] = new_id
                        new_cat = _dict([
                            ('id', new_id),
                            ('name', old_cat['name']),
                            # ('supercategory', old_cat['supercategory']),
                        ])
                        update_ifnotin(new_cat, old_cat)
                        merged['categories'].append(new_cat)
                    cat_id_map[old_cat['id']] = new_id

                # Add the keypoint categories into the merged dataset
                if 'keypoint_categories' in old_dset:
                    if 'keypoint_categories' not in merged:
                        merged['keypoint_categories'] = []
                    old_id_to_name = {k['id']: k['name']
                                      for k in old_dset['keypoint_categories']}
                    postproc_kpcats = []
                    for old_kpcat in old_dset['keypoint_categories']:
                        new_id = merged_kp_name_to_id.get(old_kpcat['name'], None)
                        # The same kpcategory might exist in different datasets.
                        if new_id is None:
                            # Only add if it does not yet exist
                            new_id = len(merged_kp_name_to_id) + 1
                            merged_kp_name_to_id[old_kpcat['name']] = new_id
                            new_kpcat = _dict([
                                ('id', new_id),
                                ('name', old_kpcat['name']),
                            ])
                            update_ifnotin(new_kpcat, old_kpcat)

                            old_reflect_id = new_kpcat.get('reflection_id', None)
                            if old_reflect_id is not None:
                                # Temporarilly overwrite reflectid with name
                                reflect_name = old_id_to_name.get(old_reflect_id, None)
                                new_kpcat['reflection_id'] = reflect_name
                                postproc_kpcats.append(new_kpcat)

                            merged['keypoint_categories'].append(new_kpcat)
                        kpcat_id_map[old_kpcat['id']] = new_id

                    # Fix reflection ids
                    for kpcat in postproc_kpcats:
                        reflect_name = kpcat['reflection_id']
                        new_reflect_id = merged_kp_name_to_id.get(reflect_name, None)
                        kpcat['reflection_id'] = new_reflect_id

                # Add the videos into the merged dataset
                for old_video in old_dset.get('videos', []):
                    if preserve_vidids:
                        new_id = old_video['id']
                    else:
                        new_id = len(merged['videos']) + 1
                    new_vidname = unique_video_names.remap(old_video['name'])
                    new_video = _dict([
                        ('id', new_id),
                        ('name', new_vidname),
                    ])
                    # copy over other metadata
                    update_ifnotin(new_video, old_video)
                    video_id_map[old_video['id']] = new_video['id']
                    if remember_parent:
                        new_video['union_parent'] = old_fpath
                    merged['videos'].append(new_video)

                # Add the images into the merged dataset
                for old_img in old_dset['images']:
                    if preserve_gids:
                        new_id = old_img['id']
                    else:
                        new_id = len(merged['images']) + 1
                    old_gname = old_img['file_name']
                    new_gname = None if old_gname is None else (
                        join(subdir, old_gname)
                    )
                    new_img = _dict([
                        ('id', new_id),
                        ('file_name', new_gname),
                    ])
                    old_name = old_img.get('name', None)
                    if old_name is not None:
                        new_name = unique_img_names.remap(old_name)
                        new_img['name'] = new_name
                    if 'auxiliary' in old_img:
                        new_auxiliary = []
                        for old_aux in old_img['auxiliary']:
                            new_aux = old_aux.copy()
                            new_aux['file_name'] = join(subdir, old_aux['file_name'])
                            new_auxiliary.append(new_aux)
                        new_img['auxiliary'] = new_auxiliary
                    if 'assets' in old_img:
                        new_auxiliary = []
                        for old_aux in old_img['assets']:
                            new_aux = old_aux.copy()
                            new_aux['file_name'] = join(subdir, old_aux['file_name'])
                            new_auxiliary.append(new_aux)
                        new_img['assets'] = new_auxiliary

                    video_img_id = video_id_map.get(old_img.get('video_id'), None)
                    if video_img_id is not None:
                        new_img['video_id'] = video_img_id
                    # copy over other metadata
                    update_ifnotin(new_img, old_img)
                    img_id_map[old_img['id']] = new_img['id']
                    if remember_parent:
                        new_img['union_parent'] = old_fpath
                    merged['images'].append(new_img)

                # Add the annotations into the merged dataset
                for old_annot in old_dset['annotations']:
                    old_cat_id = old_annot['category_id']
                    old_img_id = old_annot['image_id']
                    new_cat_id = cat_id_map.get(old_cat_id, ub.NoParam)
                    new_img_id = img_id_map.get(old_img_id, None)

                    if new_cat_id is ub.NoParam:
                        # NOTE: category_id is allowed to be None
                        warnings.warn('annot {} in {} has bad category-id {}'.format(
                            old_annot, subdir, old_cat_id))
                    if new_img_id is None:
                        warnings.warn('annot {} in {} has bad image-id {}'.format(
                            old_annot, subdir, old_img_id))
                    new_annot = _dict([
                        ('id', len(merged['annotations']) + 1),
                        ('image_id', new_img_id),
                        ('category_id', new_cat_id),
                    ])
                    if disjoint_tracks:
                        old_track_id = old_annot.get('track_id', None)
                        if old_track_id is not None:
                            new_track_id = track_id_map.remap(old_track_id)
                            new_annot['track_id'] = new_track_id
                    update_ifnotin(new_annot, old_annot)

                    if kpcat_id_map:
                        # Need to copy keypoint dict to not clobber original
                        # dset
                        if 'keypoints' in new_annot:
                            old_keypoints = new_annot['keypoints']
                            new_keypoints = copy.deepcopy(old_keypoints)
                            for kp in new_keypoints:
                                kp['keypoint_category_id'] = kpcat_id_map.get(
                                    kp['keypoint_category_id'], None)
                            new_annot['keypoints'] = new_keypoints
                    merged['annotations'].append(new_annot)

                # Mark that we are not allowed to use the same track-ids again
                track_id_map.block_seen()
            return merged

        # New behavior is simplified and I believe it is correct
        def longest_common_prefix(items, sep='/'):
            """
            Example:
                >>> items = [
                >>>     '/foo/bar/always/the/same/set1/img1.png',
                >>>     '/foo/bar/always/the/same/set1/img2.png',
                >>>     '/foo/bar/always/the/same/set2/img1.png',
                >>>     '/foo/bar/always/the/same/set2/img2.png',
                >>>     '/foo/baz/file1.txt',
                >>> ]
                >>> sep = '/'
                >>> longest_common_prefix(items, sep=sep)
                >>> longest_common_prefix(items[:-1], sep=sep)
            """
            # I would use a trie, but I don't know if pygtrie can do this efficiently
            # (not that this is efficient)
            freq = defaultdict(lambda: 0)
            for item in items:
                path = tuple(item.split(sep))
                for i in range(len(path)):
                    prefix = path[:i + 1]
                    freq[prefix] += 1
            # Find the longest common prefix
            if len(freq) == 0:
                longest_prefix = ''
            else:
                value, freq = max(freq.items(), key=lambda kv: (kv[1], len(kv[0])))
                longest_prefix = sep.join(value)
            return longest_prefix

        dset_roots = [dset.bundle_dpath for dset in others]
        dset_roots = [normpath(r) if r is not None else None
                      for r in dset_roots]
        items = [join('.', p) for p in dset_roots]
        common_root = longest_common_prefix(items, sep=os.path.sep)

        relative_dsets = [
            (relpath(normpath(d.bundle_dpath), common_root),
             str(d.fpath),
             d.dataset) for d in others]

        merged = _coco_union(relative_dsets, common_root)

        kwargs['bundle_dpath'] = common_root
        new_dset = cls(merged, **kwargs)
        return new_dset

    def subset(self, gids, copy=False, autobuild=True):
        """
        Return a subset of the larger coco dataset by specifying which images
        to port. All annotations in those images will be taken.

        Args:
            gids (List[int]):
                image-ids to copy into a new dataset

            copy (bool):
                if True, makes a deep copy of all nested attributes, otherwise
                makes a shallow copy.  Defaults to True.

            autobuild (bool):
                if True will automatically build the fast lookup index.
                Defaults to True.

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> gids = [1, 3]
            >>> sub_dset = self.subset(gids)
            >>> assert len(self.index.gid_to_aids) == 3
            >>> assert len(sub_dset.gid_to_aids) == 2

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('vidshapes2')
            >>> gids = [1, 2]
            >>> sub_dset = self.subset(gids, copy=True)
            >>> assert len(sub_dset.index.videos) == 1
            >>> assert len(self.index.videos) == 2

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> sub1 = self.subset([1])
            >>> sub2 = self.subset([2])
            >>> sub3 = self.subset([3])
            >>> others = [sub1, sub2, sub3]
            >>> rejoined = kwcoco.CocoDataset.union(*others)
            >>> assert len(sub1.anns) == 9
            >>> assert len(sub2.anns) == 2
            >>> assert len(sub3.anns) == 0
            >>> assert rejoined.basic_stats() == self.basic_stats()
        """
        new_dataset = _dict([(k, []) for k in self.dataset])
        new_dataset['categories'] = self.dataset['categories']
        new_dataset['info'] = self.dataset.get('info', [])
        new_dataset['licenses'] = self.dataset.get('licenses', [])

        chosen_gids = sorted(set(gids))

        chosen_imgs = list(ub.take(self.imgs, chosen_gids))
        new_dataset['images'] = chosen_imgs

        if 'keypoint_categories' in self.dataset:
            new_dataset['keypoint_categories'] = self.dataset['keypoint_categories']

        if 'videos' in self.dataset:
            # TODO: Take only videos with image support?
            vidids = sorted(set(img.get('video_id', None)
                                for img in chosen_imgs) - {None})
            chosen_vids = list(ub.take(self.index.videos, vidids))
            new_dataset['videos'] = chosen_vids

        sub_aids = sorted([aid for gid in chosen_gids
                           for aid in self.index.gid_to_aids.get(gid, [])])
        new_dataset['annotations'] = list(ub.take(self.index.anns, sub_aids))
        new_dataset['img_root'] = self.dataset.get('img_root', None)

        if copy:
            from copy import deepcopy
            new_dataset = deepcopy(new_dataset)

        sub_dset = CocoDataset(new_dataset, bundle_dpath=self.bundle_dpath,
                               autobuild=autobuild)
        return sub_dset

    def view_sql(self, force_rewrite=False, memory=False, backend='sqlite',
                 sql_db_fpath=None):
        """
        Create a cached SQL interface to this dataset suitable for large scale
        multiprocessing use cases.

        Args:
            force_rewrite (bool):
                if True, forces an update to any existing cache file on disk

            memory (bool):
                if True, the database is constructed in memory.

            backend (str): sqlite or postgresql

            sql_db_fpath (str | PathLike | None): overrides the database uri

        Note:
            This view cache is experimental and currently depends on the
            timestamp of the file pointed to by ``self.fpath``. In other words
            dont use this on in-memory datasets.

        CommandLine:
            KWCOCO_WITH_POSTGRESQL=1 xdoctest -m /home/joncrall/code/kwcoco/kwcoco/coco_dataset.py CocoDataset.view_sql

        Example:
            >>> # xdoctest: +REQUIRES(module:sqlalchemy)
            >>> # xdoctest: +REQUIRES(env:KWCOCO_WITH_POSTGRESQL)
            >>> # xdoctest: +REQUIRES(module:psycopg2)
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes32')
            >>> postgres_dset = dset.view_sql(backend='postgresql', force_rewrite=True)
            >>> sqlite_dset = dset.view_sql(backend='sqlite', force_rewrite=True)
            >>> list(dset.anns.keys())
            >>> list(postgres_dset.anns.keys())
            >>> list(sqlite_dset.anns.keys())

        Ignore:
            import timerit
            ti = timerit.Timerit(100, bestof=10, verbose=2)
            for timer in ti.reset('dct_dset'):
                dset.annots().detections
            for timer in ti.reset('postgresql'):
                postgres_dset.annots().detections
            for timer in ti.reset('sqlite'):
                sqlite_dset.annots().detections

            ub.udict(sql_dset.annots().objs[0]) - {'segmentation'}
            ub.udict(dct_dset.annots().objs[0]) - {'segmentation'}
        """
        from kwcoco.coco_sql_dataset import cached_sql_coco_view
        if sql_db_fpath is None:
            if memory:
                sql_db_fpath = ':memory:'
        sql_dset = cached_sql_coco_view(dset=self, sql_db_fpath=sql_db_fpath,
                                        force_rewrite=force_rewrite,
                                        backend=backend)
        return sql_dset


def demo_coco_data():
    """
    Simple data for testing.

    This contains several non-standard fields, which help ensure robustness of
    functions tested with this data. For more compliant demodata see the
    ``kwcoco.demodata`` submodule.

    Example:
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwcoco
        >>> from kwcoco.coco_dataset import demo_coco_data
        >>> dataset = demo_coco_data()
        >>> self = kwcoco.CocoDataset(dataset, tag='demo')
        >>> import kwplot
        >>> kwplot.autompl()
        >>> self.show_image(gid=1)
        >>> kwplot.show_if_requested()
    """
    import kwimage
    from kwimage.im_demodata import _TEST_IMAGES
    from os.path import commonprefix

    # FIXME: be robust to broken urls
    test_imgs_keys = ['astro', 'carl', 'stars']
    urls = {k: _TEST_IMAGES[k]['url'] for k in test_imgs_keys}
    gpaths = {k: kwimage.grab_test_image_fpath(k) for k in test_imgs_keys}
    img_root = commonprefix(list(gpaths.values()))

    gpath1, gpath2, gpath3 = ub.take(gpaths, test_imgs_keys)
    url1, url2, url3 = ub.take(urls, test_imgs_keys)

    # Make file names relative for consistent testing purpose
    gname1 = relpath(gpath1, img_root)
    gname2 = relpath(gpath2, img_root)
    gname3 = relpath(gpath3, img_root)

    dataset = {
        'img_root': img_root,

        'categories': [
            {
                'id': 1, 'name': 'astronaut',
                'supercategory': 'human',
            },
            {'id': 2, 'name': 'rocket', 'supercategory': 'object'},
            {'id': 3, 'name': 'helmet', 'supercategory': 'object'},
            {
                'id': 4, 'name': 'mouth',
                'supercategory': 'human',
                'keypoints': [
                    'mouth-right-corner', 'mouth-right-bot',
                    'mouth-left-bot', 'mouth-left-corner',
                ],
                'skeleton': [[0, 1]],
            },
            {
                'id': 5, 'name': 'star',
                'supercategory': 'object',
                'keypoints': ['star-center'],
                'skeleton': [],
            },
            {'id': 6, 'name': 'astronomer', 'supercategory': 'human'},
            {'id': 7, 'name': 'astroturf', 'supercategory': 'object'},
            {
                'id': 8, 'name': 'human',
                'keypoints': ['left-eye', 'right-eye'],
                'skeleton': [[0, 1]],
            },
        ],
        'images': [
            {'id': 1, 'file_name': gname1, 'url': url1},
            {'id': 2, 'file_name': gname2, 'url': url2},
            {'id': 3, 'file_name': gname3, 'url': url3},
        ],
        'annotations': [
            {'id': 1, 'image_id': 1, 'category_id': 1,
             'bbox': [10, 10, 360, 490],
             'keypoints': [247, 101, 2, 202, 100, 2],
             'segmentation': [[
                 40, 509, 26, 486, 20, 419, 28, 334, 51, 266, 85, 229, 102,
                 216, 118, 197, 125, 176, 148, 151, 179, 147, 182, 134, 174,
                 128, 166, 115, 156, 94, 155, 64, 162, 48, 193, 34, 197, 26,
                 210, 21, 231, 14, 265, 24, 295, 49, 300, 90, 297, 111, 280,
                 126, 277, 132, 266, 137, 264, 152, 255, 164, 256, 174, 283,
                 195, 301, 220, 305, 234, 338, 262, 350, 286, 360, 326, 363,
                 351, 324, 369, 292, 404, 280, 448, 276, 496, 280, 511]],
             },
            {'id': 2, 'image_id': 1, 'category_id': 2,
             'bbox': [350, 5, 130, 290]},
            {'id': 3, 'image_id': 1, 'category_id': 3,
             'line': [326, 369, 500, 500]},
            {'id': 4, 'image_id': 1, 'category_id': 4,
             'keypoints': [
                 202, 139, 2, 215, 150, 2, 229, 150, 2, 244, 142, 2,
             ]},
            {'id': 5, 'image_id': 1, 'category_id': 5,
             'keypoints': [37, 65, 1]},
            {'id': 6, 'image_id': 1, 'category_id': 5,
             'keypoints': [37, 16, 1]},
            {'id': 7, 'image_id': 1, 'category_id': 5,
             'keypoints': [3, 9, 1]},
            {'id': 8, 'image_id': 1, 'category_id': 5,
             'keypoints': [2, 111, 1]},
            {'id': 9, 'image_id': 1, 'category_id': 5,
             'keypoints': [2, 60, 1]},

            {'id': 10, 'image_id': 2, 'category_id': 6,
             'bbox': [37, 6, 230, 240]},
            {'id': 11, 'image_id': 2, 'category_id': 4,
             'bbox': [156, 130, 45, 18]}
        ],
        'licenses': [],
        'info': [],
    }
    return dataset


if __name__ == '__main__':
    r"""
    CommandLine:
        xdoctest kwcoco.coco_dataset all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
