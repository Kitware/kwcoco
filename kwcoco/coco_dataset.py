# -*- coding: utf-8 -*-
"""
An implementation and extension of the original MS-COCO API [1]_.

Extends the format to also include line annotations.

Dataset Spec:
    dataset = {
        # these are object level categories
        'categories': [
            {
                'id': <int:category_id>,
                'name': <str:>,
                'supercategory': str  # optional

                # Note: this is the original way to specify keypoint
                # categories, but our implementation supports a more general
                # alternative schema
                "keypoints": [kpname_1, ..., kpname_K], # length <k> array of keypoint names
                "skeleton": [(kx_a1, kx_b1), ..., (kx_aE, kx_bE)], # list of edge pairs (of keypoint indices), defining connectivity of keypoints.
            },
            ...
        ],
        'images': [
            {
                'id': int, 'file_name': str
            },
            ...
        ],
        'annotations': [
            {
                'id': int,
                'image_id': int,
                'category_id': int,
                'bbox': [tl_x, tl_y, w, h],  # optional (xywh format)
                "score" : float,
                "caption": str,  # an optional text caption for this annotation
                "iscrowd" : <0 or 1>,  # denotes if the annotation covers a single object (0) or multiple objects (1)
                "keypoints" : [x1,y1,v1,...,xk,yk,vk], # or new dict-based format
                'segmentation': <RunLengthEncoding | Polygon>,  # formats are defined bellow
            },
            ...
        ],
        'licenses': [],
        'info': [],
    }

    Polygon:
        A flattned list of xy coordinates.
        [x1, y1, x2, y2, ..., xn, yn]

        or a list of flattned list of xy coordinates if the CCs are disjoint
        [[x1, y1, x2, y2, ..., xn, yn], [x1, y1, ..., xm, ym],]

        Note: the original coco spec does not allow for holes in polygons.

        (PENDING) We also allow a non-standard dictionary encoding of polygons
            {'exterior': [(x1, y1)...],
             'interiors': [[(x1, y1), ...], ...]}

    RunLengthEncoding:
        The RLE can be in a special bytes encoding or in a binary array
        encoding. We reuse the original C functions are in [2]_ in
        `kwimage.structs.Mask` to provide a convinient way to abstract this
        rather esoteric bytes encoding.

        For pure python implementations see kwimage:
            Converting from an image to RLE can be done via kwimage.run_length_encoding
            Converting from RLE back to an image can be done via:
                kwimage.decode_run_length

            For compatibility with the COCO specs ensure the binary flags
            for these functions are set to true.

    Keypoints:
        (PENDING)
        Annotation keypoints may also be specified in this non-standard (but
        ultimately more general) way:

        'annotations': [
            {
                'keypoints': [
                    {
                        'xy': <x1, y1>,
                        'visible': <0 or 1 or 2>,
                        'keypoint_category_id': <kp_cid>,
                        'keypoint_category': <kp_name, optional>,  # this can be specified instead of an id
                    }, ...
                ]
            }, ...
        ],
        'keypoint_categories': [{
            'name': <str>,
            'id': <int>,  # an id for this keypoint category
            'supercategory': <kp_name>  # name of coarser parent keypoint class (for hierarchical keypoints)
            'reflection_id': <kp_cid>  # specify only if the keypoint id would be swapped with another keypoint type
        },...
        ]

        In this scheme the "keypoints" property of each annotation (which used
        to be a list of floats) is now specified as a list of dictionaries that
        specify each keypoints location, id, and visibility explicitly. This
        allows for things like non-unique keypoints, partial keypoint
        annotations. This also removes the ordering requirement, which makes it
        simpler to keep track of each keypoints class type.

        We also have a new top-level dictionary to specify all the possible
        keypoint categories.

    Auxillary Channels:
        For multimodal or multispectral images it is possible to specify
        auxillary channels in an image dictionary as follows:

        {
            'id': int, 'file_name': str
            'channels': <spec>,  # a spec code that indicates the layout of these channels.
            'auxillary': [  # information about auxillary channels
                {
                    'file_name':
                    'channels': <spec>
                }, ... # can have many auxillary channels with unique specs
            ]
        }


Notes:
    The main object in this file is `class`:CocoDataset, which is composed of
    several mixin classes. See the class and method documentation for more
    details.

References:
    .. [1] http://cocodataset.org/#format-data
    .. [2] https://github.com/nightrome/cocostuffapi/blob/master/PythonAPI/pycocotools/mask.py

"""
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import dirname
import warnings
from os.path import splitext
from os.path import basename
from os.path import join
from collections import OrderedDict
import json
import numpy as np
import ubelt as ub
import six
import itertools as it
from six.moves import cStringIO as StringIO
import copy

__all__ = [
    'CocoDataset',
]

_dict = OrderedDict


INT_TYPES = (int, np.integer)


def _annot_type(ann):
    """
    Returns what type of annotation `ann` is.
    """
    return tuple(sorted(set(ann) & {'bbox', 'line', 'keypoints'}))


class ObjectList1D(ub.NiceRepr):
    """
    Lightweight reference to a set of annotations that allows for convenient
    property access.

    Similar to ibeis._ibeis_object.ObjectList1D

    Types:
        ObjT = Ann | Img | Cat  # can be one of these types
        ObjectList1D gives us access to a List[ObjT]

    Example:
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo()
        >>> # Both annots and images are object lists
        >>> self = dset.annots()
        >>> self = dset.images()
        >>> # can call with a list of ids or not, for everything
        >>> self = dset.annots([1, 2, 11])
        >>> self = dset.images([1, 2, 3])
        >>> self.lookup('id')
        >>> self.lookup(['id'])
    """

    def __init__(self, ids, dset, key):
        self._key = key
        self._ids = ids
        self._dset = dset

    def __nice__(self):
        return 'num={!r}'.format(len(self))

    def __iter__(self):
        return iter(self._ids)

    def __len__(self):
        return len(self._ids)

    @property
    def _id_to_obj(self):
        return self._dset.index._id_lookup[self._key]

    def take(self, idxs):
        """
        Take a subset by index

        Example:
            >>> self = CocoDataset.demo().annots()
            >>> assert len(self.take([0, 2, 3])) == 3
        """
        subids = list(ub.take(self._ids, idxs))
        newself = self.__class__(subids, self._dset)
        return newself

    def compress(self, flags):
        """
        Take a subset by flags

        Example:
            >>> self = CocoDataset.demo().images()
            >>> assert len(self.compress([True, False, True])) == 2
        """
        subids = list(ub.compress(self._ids, flags))
        newself = self.__class__(subids, self._dset)
        return newself

    def peek(self):
        return ub.peek(self._id_to_obj.values())

    def lookup(self, key, default=ub.NoParam, keepid=False):
        """
        Lookup a list of object attributes

        Args:
            key (str | Iterable): name of the property you want to lookup
                can also be a list of names, in which case we return a dict

            default : if specified, uses this value if it doesn't exist
                in an ObjT.

            keepid: if True, return a mapping from ids to the property

        Returns:
            List[ObjT]: a list of whatever type the object is
            Dict[str, ObjT]

        Example:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo()
            >>> self = dset.annots()
            >>> self.lookup('id')
            >>> key = ['id']
            >>> default = None
            >>> self.lookup(key=['id', 'image_id'])
            >>> self.lookup(key=['id', 'image_id'])
            >>> self.lookup(key='foo', default=None, keepid=True)
            >>> self.lookup(key=['foo'], default=None, keepid=True)
            >>> self.lookup(key=['id', 'image_id'], keepid=True)
        """
        # Note: while the old _lookup code was slightly faster than this, the
        # difference is extremely negligable (179us vs 178us).
        if ub.iterable(key):
            return {k: self.lookup(k, default, keepid) for k in key}
        else:
            _lut = self._id_to_obj
            if keepid:
                if default is ub.NoParam:
                    attr_list = {_id: _lut[_id][key] for _id in self._ids}
                else:
                    attr_list = {_id: _lut[_id].get(key, default) for _id in self._ids}
            else:
                if default is ub.NoParam:
                    attr_list = [_lut[_id][key] for _id in self._ids]
                else:
                    attr_list = [_lut[_id].get(key, default) for _id in self._ids]
            return attr_list

    def get(self, key, default=ub.NoParam):
        """ alias for lookup """
        assert not ub.iterable(key)
        return self.lookup(key, default=default)

    def set(self, key, values):
        """
        Assign a value to each annotation

        Args:
            key (str): the annotation property to modify
            values (Iterable | scalar): an iterable of values to set for each
                annot in the dataset. If the item is not iterable, it is
                assigned to all objects.

        Example:
            >>> dset = CocoDataset.demo()
            >>> self = dset.annots()
            >>> self.set('my-key1', 'my-scalar-value')
            >>> self.set('my-key2', np.random.rand(len(self)))
            >>> print('dset.imgs = {}'.format(ub.repr2(dset.imgs, nl=1)))
            >>> self.get('my-key2')
        """
        if not ub.iterable(values):
            values = [values] * len(self)
        elif not isinstance(values, list):
            values = list(values)
        assert len(self) == len(values)
        self._set(key, values)

    def _set(self, key, values):
        """ faster less safe version of set """
        objs = ub.take(self._id_to_obj, self._ids)
        for obj, value in zip(objs, values):
            obj[key] = value

    def _lookup(self, key, default=ub.NoParam):
        """
        Benchmark:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('shapes256')
            >>> self = annots = dset.annots()

            >>> import timerit
            >>> ti = timerit.Timerit(100, bestof=10, verbose=2)

            for timer in ti.reset('lookup'):
                with timer:
                    self.lookup('image_id')

            for timer in ti.reset('_lookup'):
                with timer:
                    self._lookup('image_id')

            for timer in ti.reset('image_id'):
                with timer:
                    self.image_id

            for timer in ti.reset('raw1'):
                with timer:
                    key = 'image_id'
                    [self._dset.anns[_id][key] for _id in self._ids]

            for timer in ti.reset('raw2'):
                with timer:
                    anns = self._dset.anns
                    key = 'image_id'
                    [anns[_id][key] for _id in self._ids]

            for timer in ti.reset('lut-gen'):
                with timer:
                    _lut = self._obj_lut
                    objs = (_lut[_id] for _id in self._ids)
                    [obj[key] for obj in objs]

            for timer in ti.reset('lut-gen-single'):
                with timer:
                    _lut = self._obj_lut
                    [_lut[_id][key] for _id in self._ids]
        """
        return self.lookup(key, default=default)


class ObjectGroups(ub.NiceRepr):
    """
    An object for holding a groups of `ObjectList1D` objects
    """
    def __init__(self, groups, dset):
        self._groups = groups

    def _lookup(self, key):
        return self._lookup(key)

    def lookup(self, key, default=ub.NoParam):
        return [group.lookup(key, default) for group in self._groups]

    def __nice__(self):
        # import timerit
        # mu = timerit.core._trychar('μ', 'm')
        # sigma = timerit.core._trychar('σ', 's')
        mu = 'm'
        sigma = 's'
        len_list = list(map(len, self._groups))
        num = len(self._groups)
        mean = np.mean(len_list)
        std = np.std(len_list)
        nice = 'n={!r}, {}={:.1f}, {}={:.1f}'.format(
            num, mu, mean, sigma, std)
        return nice


class Images(ObjectList1D):
    """
    """

    def __init__(self, ids, dset):
        super(Images, self).__init__(ids, dset, 'images')

    @property
    def gids(self):
        return self._ids

    @property
    def gname(self):
        return self.lookup('file_name')

    @property
    def gpath(self):
        root = self._dset.img_root
        return [join(root, gname) for gname in self.gname]

    @property
    def width(self):
        return self.lookup('width')

    @property
    def height(self):
        return self.lookup('height')

    @property
    def size(self):
        """
        Example:
            >>> from kwcoco.coco_dataset import *
            >>> self = CocoDataset.demo().images()
            >>> self._dset._ensure_imgsize()
            >>> print(self.size)
            [(512, 512), (300, 250), (256, 256)]
        """
        return list(zip(self.lookup('width'), self.lookup('height')))

    @property
    def area(self):
        """
        Example:
            >>> from kwcoco.coco_dataset import *
            >>> self = CocoDataset.demo().images()
            >>> self._dset._ensure_imgsize()
            >>> print(self.area)
            [262144, 75000, 65536]
        """
        return [w * h for w, h in zip(self.lookup('width'), self.lookup('height'))]

    @property
    def n_annots(self):
        """
        Example:
            >>> self = CocoDataset.demo().images()
            >>> print(ub.repr2(self.n_annots, nl=0))
            [9, 2, 0]
        """
        return list(map(len, ub.take(self._dset.gid_to_aids, self._ids)))

    @property
    def aids(self):
        """
        Example:
            >>> self = CocoDataset.demo().images()
            >>> print(ub.repr2(list(map(list, self.aids)), nl=0))
            [[1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11], []]
        """
        return list(ub.take(self._dset.gid_to_aids, self._ids))

    @property
    def annots(self):
        """
        Example:
            >>> self = CocoDataset.demo().images()
            >>> print(self.annots)
            <AnnotGroups(n=3, m=3.7, s=3.9)>
        """
        return AnnotGroups([self._dset.annots(aids) for aids in self.aids],
                           self._dset)


class Annots(ObjectList1D):
    """
    """

    def __init__(self, ids, dset):
        super(Annots, self).__init__(ids, dset, 'annotations')

    @property
    def aids(self):
        """ The annotation ids of this column of annotations """
        return self._ids

    @property
    def images(self):
        """
        Get the column of images

        Returns:
            Images
        """
        return self._dset.images(self.gids)

    @property
    def image_id(self):
        return self.lookup('image_id')

    @property
    def category_id(self):
        return self.lookup('category_id')

    @property
    def gids(self):
        """
        Get the column of image-ids

        Returns:
            List[int]: list of image ids
        """
        return self.lookup('image_id')

    @property
    def cids(self):
        """
        Get the column of category-ids

        Returns:
            List[int]
        """
        return self.lookup('category_id')

    @property
    def cnames(self):
        """
        Get the column of category names

        Returns:
            List[int]
        """
        return [cat['name'] for cat in ub.take(self._dset.cats, self.cids)]

    @cnames.setter
    def cnames(self, cnames):
        """
        Args:
            cnames (List[str]):

        Example:
            >>> from kwcoco.coco_dataset import *  # NOQA
            >>> self = CocoDataset.demo().annots([1, 2, 11])
            >>> print('self.cnames = {!r}'.format(self.cnames))
            >>> print('self.cids = {!r}'.format(self.cids))
            >>> cnames = ['boo', 'bar', 'rocket']
            >>> list(map(self._dset.ensure_category, set(cnames)))
            >>> self.cnames = cnames
            >>> print('self.cnames = {!r}'.format(self.cnames))
            >>> print('self.cids = {!r}'.format(self.cids))
        """
        cats = map(self._dset._alias_to_cat, cnames)
        cids = (cat['id'] for cat in cats)
        self.set('category_id', cids)

    @property
    def detections(self):
        """
        Get the kwimage-style detection objects

        Returns:
            kwimage.Detections

        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> from kwcoco.coco_dataset import *  # NOQA
            >>> self = CocoDataset.demo('shapes32').annots([1, 2, 11])
            >>> dets = self.detections
            >>> print('dets.data = {!r}'.format(dets.data))
            >>> print('dets.meta = {!r}'.format(dets.meta))
        """
        import kwimage
        anns = [self._id_to_obj[aid] for aid in self.aids]
        dets = kwimage.Detections.from_coco_annots(anns, dset=self._dset)
        # dets.data['aids'] = np.array(self.aids)
        return dets

    @property
    def boxes(self):
        """
        Get the column of kwimage-style bounding boxes

        Example:
            >>> self = CocoDataset.demo().annots([1, 2, 11])
            >>> print(self.boxes)
            <Boxes(xywh,
                array([[ 10,  10, 360, 490],
                       [350,   5, 130, 290],
                       [124,  96,  45,  18]]))>
        """
        import kwimage
        xywh = self.lookup('bbox')
        boxes = kwimage.Boxes(xywh, 'xywh')
        return boxes

    @boxes.setter
    def boxes(self, boxes):
        """
        Args:
            boxes (kwimage.Boxes):

        Example:
            >>> import kwimage
            >>> from kwcoco.coco_dataset import *  # NOQA
            >>> self = CocoDataset.demo().annots([1, 2, 11])
            >>> print('self.boxes = {!r}'.format(self.boxes))
            >>> boxes = kwimage.Boxes.random(3).scale(512).astype(np.int)
            >>> self.boxes = boxes
            >>> print('self.boxes = {!r}'.format(self.boxes))
        """
        anns = ub.take(self._dset.anns, self.aids)
        xywh = boxes.to_xywh().data.tolist()
        for ann, xywh in zip(anns, xywh):
            ann['bbox'] = xywh

    @property
    def xywh(self):
        """
        Returns raw boxes

        Example:
            >>> self = CocoDataset.demo().annots([1, 2, 11])
            >>> print(self.xywh)
        """
        xywh = self.lookup('bbox')
        return xywh


class AnnotGroups(ObjectGroups):
    @property
    def cids(self):
        return self.lookup('category_id')


class ImageGroups(ObjectGroups):
    pass


class MixinCocoDepricate(object):
    """
    These functions are marked for deprication and may be removed at any time
    """

    def lookup_imgs(self, filename=None):
        """
        Linear search for an images with specific attributes

        # DEPRICATE

        Ignore:
            filename = '201503.20150525.101841191.573975.png'
            list(self.lookup_imgs(filename))
            gid = 64940
            img = self.imgs[gid]
            img['file_name'] = filename
        """
        import warnings
        warnings.warn('DEPRECATED: this method name may be recycled and '
                      'do something different in a later version',
                      DeprecationWarning)
        for img in self.imgs.values():
            if filename is not None:
                fpath = img['file_name']
                fname = basename(fpath)
                fname_noext = splitext(fname)[0]
                if filename in [fpath, fname, fname_noext]:
                    print('img = {!r}'.format(img))
                    yield img

    def lookup_anns(self, has=None):
        """
        Linear search for an annotations with specific attributes

        # DEPRICATE

        Ignore:
            list(self.lookup_anns(has='radius'))
            gid = 112888
            img = self.imgs[gid]
            img['file_name'] = filename
        """
        import warnings
        warnings.warn('DEPRECATED: this method name may be recycled and '
                      'do something different in a later version',
                      DeprecationWarning)
        for ann in self.anns.values():
            if has is not None:
                if hasattr(ann, has):
                    print('ann = {!r}'.format(ann))
                    yield ann

    def _mark_annotated_images(self):
        """
        Mark any image that explicitly has annotations.

        # DEPRICATE
        """
        import warnings
        warnings.warn('DEPRECATED: this method should not be used', DeprecationWarning)
        for gid, img in self.imgs.items():
            aids = self.gid_to_aids.get(gid, [])
            # If there is at least one annotation, always mark as has_annots
            if len(aids) > 0:
                assert img.get('has_annots', ub.NoParam) in [ub.NoParam, True], (
                    'image with annots was explictly labeled as non-True!')
                img['has_annots'] = True
            else:
                # Otherwise set has_annots to null if it has not been
                # explicitly labeled
                if 'has_annots' not in img:
                    img['has_annots'] = None

    def _find_bad_annotations(self):
        import warnings
        warnings.warn('DEPRECATED: this method should not be used', DeprecationWarning)
        to_remove = []
        for ann in self.dataset['annotations']:
            if ann['image_id'] is None or ann['category_id'] is None:
                to_remove.append(ann)
            else:
                if ann['image_id'] not in self.imgs:
                    to_remove.append(ann)
                if ann['category_id'] not in self.cats:
                    to_remove.append(ann)
        return to_remove

    def _remove_keypoint_annotations(self, rebuild=True):
        """
        Remove annotations with keypoints only

        Example:
            >>> self = CocoDataset.demo()
            >>> self._remove_keypoint_annotations()
        """
        import warnings
        warnings.warn('DEPRECATED: this method should not be used', DeprecationWarning)
        to_remove = []
        for ann in self.dataset['annotations']:
            roi_shape = ann.get('roi_shape', None)
            if roi_shape is None:
                if 'keypoints' in ann and ann.get('bbox', None) is None:
                    to_remove.append(ann)
            elif roi_shape == 'keypoints':
                to_remove.append(ann)
        print('Removing {} keypoint annotations'.format(len(to_remove)))
        self.remove_annotations(to_remove)
        if rebuild:
            self._build_index()

    def _remove_bad_annotations(self, rebuild=True):
        import warnings
        warnings.warn('DEPRECATED: this method should not be used', DeprecationWarning)
        to_remove = []
        for ann in self.dataset['annotations']:
            if ann['image_id'] is None or ann['category_id'] is None:
                to_remove.append(ann)
        print('Removing {} bad annotations'.format(len(to_remove)))
        self.remove_annotations(to_remove)
        if rebuild:
            self._build_index()

    def _remove_radius_annotations(self, rebuild=False):
        import warnings
        warnings.warn('DEPRECATED: this method should not be used', DeprecationWarning)
        to_remove = []
        for ann in self.dataset['annotations']:
            if 'radius' in ann:
                to_remove.append(ann)
        print('Removing {} radius annotations'.format(len(to_remove)))
        self.remove_annotations(to_remove)
        if rebuild:
            self._build_index()

    def _remove_empty_images(self):
        import warnings
        warnings.warn('DEPRECATED: this method should not be used', DeprecationWarning)
        to_remove = []
        for gid in self.imgs.keys():
            aids = self.gid_to_aids.get(gid, [])
            if not aids:
                to_remove.append(self.imgs[gid])
        print('Removing {} empty images'.format(len(to_remove)))
        for img in to_remove:
            self.dataset['images'].remove(img)
        self._build_index()


class MixinCocoExtras(object):
    """
    Misc functions for coco
    """

    def load_image(self, gid_or_img):
        """
        Reads an image from disk and

        Args:
            gid_or_img (int or dict): image id or image dict

        Returns:
            np.ndarray : the image
        """
        import kwimage
        gpath = self.get_image_fpath(gid_or_img)
        np_img = kwimage.imread(gpath)
        return np_img

    def load_image_fpath(self, gid_or_img):
        import warnings
        warnings.warn(
            'get_image_fpath is deprecated use get_image_fpath instead',
            DeprecationWarning)
        return self.get_image_fpath(gid_or_img)

    def get_image_fpath(self, gid_or_img):
        """
        Returns the full path to the image

        Args:
            gid_or_img (int or dict): image id or image dict

        Returns:
            PathLike: full path to the image
        """
        img = self._resolve_to_img(gid_or_img)
        gpath = join(self.img_root, img['file_name'])
        return gpath

    def _get_img_auxillary(self, gid_or_img, channels):
        """ returns the auxillary dictionary for a specific channel """
        img = self._resolve_to_img(gid_or_img)
        found = None
        for aux in img['auxillary']:
            if aux['channels'] == channels:
                found = aux
                break
        if found is None:
            raise Exception('Image does not have auxillary channels={}'.format(channels))
        return found

    def get_auxillary_fpath(self, gid_or_img, channels):
        """
        Returns the full path to auxillary data for an image

        Args:
            gid_or_img (int | dict): an image or its id
            channels (str): the auxillary channel to load (e.g. disparity)

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('shapes8', aux=True)
            >>> self.get_auxillary_fpath(1, 'disparity')
        """
        aux = self._get_img_auxillary(gid_or_img, channels)
        fpath = join(self.img_root, aux['file_name'])
        return fpath

    def load_annot_sample(self, aid_or_ann, image=None, pad=None):
        """
        Reads the chip of an annotation. Note this is much less efficient than
        using a sampler, but it doesn't require disk cache.

        Args:
            aid_or_int (int or dict): annot id or dict
            image (ArrayLike, default=None): preloaded image
                (note: this process is inefficient unless image is specified)

        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> sample = self.load_annot_sample(2, pad=100)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(sample['im'])
            >>> kwplot.show_if_requested()
        """
        from ndsampler.coco_sampler import padded_slice
        ann = self._resolve_to_ann(aid_or_ann)
        if image is None:
            image = self.load_image(ann['image_id'])

        x, y, w, h = ann['bbox']
        in_slice = (
            slice(int(y), int(np.ceil(y + h))),
            slice(int(x), int(np.ceil(x + w))),
        )
        data_sliced, transform = padded_slice(image, in_slice, pad_slice=pad)

        sample = {
            'im': data_sliced,
            'transform': transform,
        }
        return sample

    @classmethod
    def coerce(cls, key):
        from os.path import exists
        if key.startswith('special:'):
            self = cls.demo(key=key.split(':')[1])
        elif exists(key):
            self = cls(key)
        else:
            self = cls.demo(key=key)
        return self

    @classmethod
    def demo(cls, key='photos', **kw):
        """
        Create a toy coco dataset for testing and demo puposes

        Args:
            key (str): either photos or shapes
            **kw : if key is shapes, these arguments are passed to toydata
                generation

        Example:
            >>> print(CocoDataset.demo('photos'))
            >>> print(CocoDataset.demo('shapes', verbose=0))
            >>> print(CocoDataset.demo('shapes256', verbose=0))
            >>> print(CocoDataset.demo('shapes-8', verbose=0))
        """
        if key.startswith('shapes'):
            from kwcoco import toydata
            import parse
            res = parse.parse('{prefix}{num_imgs:d}', key)
            if res:
                kw['n_imgs'] = int(res.named['num_imgs'])
            if 'rng' not in kw and 'n_imgs' in kw:
                kw['rng'] = kw['n_imgs']
            dataset = toydata.demodata_toy_dset(**kw)
            self = cls(dataset, tag=key)
        elif key == 'photos':
            dataset = demo_coco_data()
            self = cls(dataset, tag=key)
        else:
            raise KeyError(key)
        return self

    def _build_hashid(self, hash_pixels=False, verbose=0):
        """
        Construct a hash that uniquely identifies the state of this dataset.

        Args:
            hash_pixels (bool, default=False): If False the image data is not
                included in the hash, which can speed up computation, but is
                not 100% robust.
            verbose (int): verbosity level

        Example:
            >>> self = CocoDataset.demo()
            >>> self._build_hashid(hash_pixels=True, verbose=3)
            ...
            >>> print('self.hashid_parts = ' + ub.repr2(self.hashid_parts))
            >>> print('self.hashid = {!r}'.format(self.hashid))
            self.hashid_parts = {
                'annotations': {
                    'json': 'e573f49da7b76e27d0...',
                    'num': 11,
                },
                'images': {
                    'pixels': '67d741fefc8...',
                    'json': '6a446126490aa...',
                    'num': 3,
                },
                'categories': {
                    'json': '82d22e0079...',
                    'num': 8,
                },
            }
            self.hashid = '4769119614e921...

        Doctest:
            >>> self = CocoDataset.demo()
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

        # Ensure hashid_parts has the proper root structure
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
                gpaths = [join(self.img_root, gname)
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
                    anns_text = json.dumps(anns_ordered)
                except TypeError:
                    if __debug__:
                        for ann in anns_ordered:
                            try:
                                json.dumps(ann)
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
                imgs_text = json.dumps(
                    [_ditems(self.imgs[gid]) for gid in gids])
                hashid_parts['images']['json'] = ub.hash_data(
                    imgs_text, hasher='sha512')
                hashid_parts['images']['num'] = len(gids)
                rebuild_parts.append('images.json')
            else:
                reuse_parts.append('images.json')

            if not hashid_parts['categories'].get('json', None):
                cids = sorted(self.cats.keys())
                cats_text = json.dumps(
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
        """
        self.hashid = None
        if parts is not None and self.hashid_parts is not None:
            for part in parts:
                self.hashid_parts.pop(part, None)
        else:
            self.hashid_parts = None

    def _ensure_imgsize(self, workers=0, verbose=1, fail=False):
        """
        Populate the imgsize field if it does not exist.

        Args:
            workers (int, default=0): number of workers for parallel
                processing.

            verbose (int, default=1): verbosity level

            fail (bool, default=False): if True, raises an exception if
               anything size fails to load.

        Returns:
            List[dict]: a list of "bad" image dictionaries where the size could
                not be determined. Typically these are corrupted images and
                should be removed.

        Example:
            >>> # Normal case
            >>> self = CocoDataset.demo()
            >>> bad_imgs = self._ensure_imgsize()
            >>> assert len(bad_imgs) == 0
            >>> assert self.imgs[1]['width'] == 512
            >>> assert self.imgs[2]['width'] == 300
            >>> assert self.imgs[3]['width'] == 256

            >>> # Fail cases
            >>> self = CocoDataset()
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
            from kwcoco.util import util_futures

            if self.tag:
                desc = 'populate imgsize for ' + self.tag
            else:
                desc = 'populate imgsize for untagged coco dataset'

            pool = util_futures.JobPool('thread', max_workers=workers)
            for img in ub.ProgIter(self.dataset['images'], verbose=verbose,
                                   desc='submit image size jobs'):
                gpath = join(self.img_root, img['file_name'])
                if 'width' not in img or 'height' not in img:
                    job = pool.submit(kwimage.load_image_shape, gpath)
                    job.img = img

            for job in ub.ProgIter(pool.as_completed(), total=len(pool),
                                   verbose=verbose, desc=desc):
                try:
                    h, w = job.result()[0:2]
                except Exception:
                    if fail:
                        raise
                    bad_images.append(job.img)
                else:
                    job.img['width'] = w
                    job.img['height'] = h
        return bad_images

    def _resolve_to_id(self, id_or_dict):
        """
        Ensures output is an id
        """
        if isinstance(id_or_dict, INT_TYPES):
            resolved_id = id_or_dict
        else:
            resolved_id = id_or_dict['id']
        return resolved_id

    def _resolve_to_cid(self, id_or_name_or_dict):
        """
        Ensures output is an category id

        Note: this does not resolve aliases (yet), for that see _alias_to_cat
        Todo: we could maintain an alias index to make this fast
        """
        if isinstance(id_or_name_or_dict, INT_TYPES):
            resolved_id = id_or_name_or_dict
        elif isinstance(id_or_name_or_dict, six.string_types):
            resolved_id = self.index.name_to_cat[id_or_name_or_dict]['id']
        else:
            resolved_id = id_or_name_or_dict['id']
        return resolved_id

    def _resolve_to_gid(self, id_or_name_or_dict):
        """
        Ensures output is an category id
        """
        if isinstance(id_or_name_or_dict, INT_TYPES):
            resolved_id = id_or_name_or_dict
        elif isinstance(id_or_name_or_dict, six.string_types):
            resolved_id = self.index.file_name_to_img[id_or_name_or_dict]['id']
        else:
            resolved_id = id_or_name_or_dict['id']
        return resolved_id

    def _resolve_to_ann(self, aid_or_ann):
        """
        Ensures output is an annotation dictionary
        """
        if isinstance(aid_or_ann, INT_TYPES):
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
        if isinstance(gid_or_img, INT_TYPES):
            resolved_img = None
            if self.imgs is not None:
                resolved_img = self.imgs[gid_or_img]
            else:
                for img in self.dataset['imgotations']:
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
            >>> self = CocoDataset.demo('shapes')
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
        if isinstance(kp_identifier, INT_TYPES):
            kpcat = None
            for _kpcat in self.dataset['keypoint_categories']:
                if _kpcat['id'] == kp_identifier:
                    kpcat = _kpcat
            if kpcat is None:
                raise KeyError('unable to find keypoint category')
        elif isinstance(kp_identifier, six.string_types):
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

        Notes:
            If the index is not built, the method will work but may be slow.

        Example:
            >>> self = CocoDataset.demo()
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
        if isinstance(cat_identifier, INT_TYPES):
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
        elif isinstance(cat_identifier, six.string_types):
            cat = self._alias_to_cat(cat_identifier)
        elif isinstance(cat_identifier, dict):
            cat = cat_identifier
        else:
            raise TypeError(type(cat_identifier))
        return cat

    def _alias_to_cat(self, alias_catname):
        """
        Lookup a coco-category via its name or an "alias" name.
        In production code, use `_resolve_to_cat` instead.

        Args:
            alias_catname (str): category name or alias

        Returns:
            dict: coco category dictionary

        Example:
            >>> self = CocoDataset.demo()
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
                if isinstance(alias_list, six.string_types):
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
            network.DiGraph: graph: a directed graph where category names are
                the nodes, supercategories define edges, and items in each
                category dict (e.g. category id) are added as node properties.

        Example:
            >>> self = CocoDataset.demo()
            >>> graph = self.category_graph()
            >>> assert 'astronaut' in graph.nodes()
            >>> assert 'keypoints' in graph.nodes['human']

            import graphid
            graphid.util.show_nx(graph)
        """
        # TODO: should supercategories that don't exist as nodes be added here?
        import networkx as nx
        graph = nx.DiGraph()
        for cat in self.dataset['categories']:
            graph.add_node(cat['name'], **cat)
            if 'supercategory' in cat:
                graph.add_edge(cat['supercategory'], cat['name'])
        return graph

    def object_categories(self):
        """
        Construct a consistent ndsampler representation of object classes

        Returns:
            nsampler.CategoryTree:

        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> self = CocoDataset.demo()
            >>> classes = self.object_categories()
            >>> print('classes = {}'.format(classes))
        """
        import ndsampler
        graph = self.category_graph()
        classes = ndsampler.CategoryTree(graph)
        return classes

    def keypoint_categories(self):
        """
        Construct a consistent ndsampler representation of keypoint classes

        Returns:
            nsampler.CategoryTree:

        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> self = CocoDataset.demo()
            >>> classes = self.keypoint_categories()
            >>> print('classes = {}'.format(classes))
        """
        import ndsampler
        if 'keypoint_categories' in self.dataset:
            import networkx as nx
            graph = nx.DiGraph()
            for cat in self.dataset['keypoint_categories']:
                graph.add_node(cat['name'], **cat)
                if 'supercategory' in cat:
                    graph.add_edge(cat['supercategory'], cat['name'])
            classes = ndsampler.CategoryTree(graph)
        else:
            catnames = self._keypoint_category_names()
            classes = ndsampler.CategoryTree.coerce(catnames)
        return classes

    def _keypoint_category_names(self):
        """
        Construct keypoint categories names.

        Uses new-style if possible, otherwise this falls back on old-style.

        Returns:
            List[str]: names: list of keypoint category names

        Example:
            >>> self = CocoDataset.demo()
            >>> names = self._keypoint_category_names()
            >>> print(names)
        """
        if 'keypoint_categories' in self.dataset:
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

    def _ensure_image_data(self, verbose=1):
        import os
        def _gen_missing_imgs():
            for img in self.dataset['images']:
                gpath = join(self.img_root, img['file_name'])
                if not os.path.exists(gpath):
                    yield img

        def _has_download_permission(_HAS_PREMISSION=[False]):
            if not _HAS_PREMISSION[0] or ub.argflag(('-y', '--yes')):
                ans = input('is it ok to download? (enter y for yes)')
                if ans in ['yes', 'y']:
                    _HAS_PREMISSION[0] = True
            return _HAS_PREMISSION[0]

        for img in ub.ProgIter(_gen_missing_imgs(), desc='ensure image data'):
            if 'url' in img:
                if _has_download_permission():
                    gpath = join(self.img_root, img['file_name'])
                    ub.ensuredir(os.path.dirname(gpath))
                    ub.grabdata(img['url'], gpath)
                else:
                    raise Exception('no permission, abort')
            else:
                raise Exception('missing image, but no url')

    def missing_images(self, verbose=0):
        import os
        bad_paths = []
        for index in ub.ProgIter(range(len(self.dataset['images'])),
                                 verbose=verbose):
            img = self.dataset['images'][index]
            gpath = join(self.img_root, img['file_name'])
            if not os.path.exists(gpath):
                bad_paths.append((index, gpath))
        return bad_paths
        # if bad_paths:
        #     print('bad paths:')
        #     print(ub.repr2(bad_paths, nl=1))
        # raise AssertionError('missing images')

    def rename_categories(self, mapper, strict=False, preserve=False,
                          rebuild=True, simple=True, merge_policy='ignore'):
        """
        Create a coarser categorization

        Note: this function has been unstable in the past, and has not yet been
        properly stabalized. Either avoid or use with care.
        Ensuring `simple=True` should result in newer saner behavior that will
        likely be backwards compatible.

        TODO:
            - [X] Simple case where we relabel names with no conflicts
            - [ ] Case where annotation labels need to change to be coarser
                    - dev note: see internal libraries for work on this
            - [ ] Other cases

        Args:
            mapper (dict or Function): maps old names to new names.

            strict (bool): DEPRICATED IGNORE.
                if True, fails if mapper doesnt map all classes

            preserve (bool): DEPRICATED IGNORE.
                if True, preserve old categories as supercatgories. Broken.

            simple (bool, default=True): defaults to the new way of doing this.
                The old way is depricated.

            merge_policy (str):
                How to handle multiple categories that map to the same name.
                Can be update or ignore.

        Example:
            >>> self = CocoDataset.demo()
            >>> self.rename_categories({'astronomer': 'person',
            >>>                         'astronaut': 'person',
            >>>                         'mouth': 'person',
            >>>                         'helmet': 'hat'}, preserve=0)
            >>> assert 'hat' in self.name_to_cat
            >>> assert 'helmet' not in self.name_to_cat
            >>> # Test merge case
            >>> self = CocoDataset.demo()
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

        if simple:
            # orig_mapper = mapper

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
                        # Note: this update order is arbitrary and may be funky
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
                self.remove_categories(rm_cnames, keep_annots=True)

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

                for cname in add_cnames:
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

        else:
            new_cats = []
            old_cats = self.dataset['categories']
            new_name_to_cat = {}
            old_to_new_id = {}

            if not callable(mapper):
                mapper = mapper.__getitem__

            for old_cat in old_cats:
                try:
                    new_name = mapper(old_cat['name'])
                except KeyError:
                    if strict:
                        raise
                    new_name = old_cat['name']

                old_cat['supercategory'] = new_name

                if new_name in new_name_to_cat:
                    # Multiple old categories are mapped to this new one
                    new_cat = new_name_to_cat[new_name]
                else:
                    if old_cat['name'] == new_name:
                        # new name is an existing category
                        new_cat = old_cat.copy()
                        new_cat['id'] = len(new_cats) + 1
                    else:
                        # new name is a entirely new category
                        new_cat = _dict([
                            ('id', len(new_cats) + 1),
                            ('name', new_name),
                        ])
                    new_name_to_cat[new_name] = new_cat
                    new_cats.append(new_cat)

                old_to_new_id[old_cat['id']] = new_cat['id']

            if preserve:
                raise NotImplementedError
                # for old_cat in old_cats:
                #     # Ensure all old cats are preserved
                #     if old_cat['name'] not in new_name_to_cat:
                #         new_cat = old_cat.copy()
                #         new_cat['id'] = len(new_cats) + 1
                #         new_name_to_cat[new_name] = new_cat
                #         new_cats.append(new_cat)
                #         old_to_new_id[old_cat['id']] = new_cat['id']

            # self.dataset['fine_categories'] = old_cats
            self.dataset['categories'] = new_cats

            # Fix annotations of modified categories
            # (todo: if the index is built, we can use that to only modify
            #  a potentially smaller subset of annotations)
            for ann in self.dataset['annotations']:
                old_id = ann['category_id']
                new_id = old_to_new_id[old_id]

                if old_id != new_id:
                    ann['category_id'] = new_id
        if rebuild:
            self._build_index()
        else:
            self.index.clear()
        self._invalidate_hashid()

    def _ensure_json_serializable(self):
        # inplace convert any ndarrays to lists

        def _walk_json(data, prefix=[]):
            items = None
            if isinstance(data, list):
                items = enumerate(data)
            elif isinstance(data, dict):
                items = data.items()
            else:
                raise TypeError(type(data))

            root = prefix
            level = {}
            for key, value in items:
                level[key] = value

            # yield a dict so the user can choose to not walk down a path
            yield root, level

            for key, value in level.items():
                if isinstance(value, (dict, list)):
                    path = prefix + [key]
                    for _ in _walk_json(value, prefix=path):
                        yield _

        to_convert = []
        for root, level in ub.ProgIter(_walk_json(self.dataset), desc='walk json'):
            for key, value in level.items():
                if isinstance(value, np.ndarray):
                    to_convert.append((root, key))

        for root, key in to_convert:
            d = self.dataset
            for k in root:
                d = d[k]
            d[key] = d[key].tolist()

    def _aspycoco(self):
        # Converts to the official pycocotools.coco.COCO object
        from pycocotools import coco
        pycoco = coco.COCO()
        pycoco.dataset = self.dataset
        pycoco.createIndex()
        return pycoco

    def rebase(self, img_root=None, absolute=False, check=True):
        """
        Rebase image paths onto a new image root.

        Args:
            img_root (str, default=None):
                New image root. If unspecified the current root is used.

            absolute (bool, default=False):
                if True, file names are stored as absolute paths, otherwise
                they are relative to the image root.

            check (bool, default=True):
                if True, checks that the images all exist

        CommandLine:
            xdoctest -m /home/joncrall/code/kwcoco/kwcoco/coco_dataset.py MixinCocoExtras.rebase

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()

            >>> # Change base relative directory
            >>> img_root = ub.expandpath('~')
            >>> self.rebase(img_root)
            >>> assert self.imgs[1]['file_name'].startswith('.cache')

            >>> # Use absolute paths
            >>> self.rebase(absolute=True)
            >>> assert self.imgs[1]['file_name'].startswith(img_root)

            >>> # Switch back to relative paths
            >>> self.rebase()
            >>> assert self.imgs[1]['file_name'].startswith('.cache')

        Example:
            >>> # demo with auxillary data
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('shapes8', aux=True)
            >>> img_root = ub.expandpath('~')
            >>> self.rebase(img_root)
            >>> assert self.imgs[1]['file_name'].startswith('.cache')
            >>> assert self.imgs[1]['auxillary'][0]['file_name'].startswith('.cache')
        """
        from os.path import exists, relpath

        old_img_root = self.img_root
        new_img_root = img_root
        if new_img_root is None:
            new_img_root = old_img_root

        for img in self.imgs.values():
            abs_file_path = join(old_img_root, img['file_name'])
            if absolute:
                img['file_name'] = abs_file_path
            else:
                img['file_name'] = relpath(abs_file_path, new_img_root)

            if check:
                abs_gpath = join(new_img_root, img['file_name'])
                if not exists(abs_gpath):
                    raise Exception(
                        'Image does not exist: {!r}'.format(abs_gpath))

            for aux in img.get('auxillary', []):
                abs_file_path = join(old_img_root, aux['file_name'])
                if absolute:
                    aux['file_name'] = abs_file_path
                else:
                    aux['file_name'] = relpath(abs_file_path, new_img_root)

        self.img_root = new_img_root
        return self


class MixinCocoAttrs(object):
    """
    Expose methods to construct object lists / groups
    """

    def annots(self, aids=None, gid=None):
        """
        Return boxes for annotations

        Example:
            >>> self = CocoDataset.demo()
            >>> annots = self.annots()
            >>> print(annots)
            <Annots(num=11)>
        """
        if aids is None and gid is not None:
            aids = sorted(self.gid_to_aids[gid])
        if aids is None:
            aids = sorted(self.anns.keys())
        return Annots(aids, self)

    def images(self, gids=None):
        """
        Return boxes for annotations

        Example:
            >>> self = CocoDataset.demo()
            >>> images = self.images()
            >>> print(images)
            <Images(num=3)>
        """
        if gids is None:
            gids = sorted(self.imgs.keys())
        return Images(gids, self)


class MixinCocoStats(object):
    """
    Methods for getting stats about the dataset
    """

    @property
    def n_annots(self):
        return len(self.dataset['annotations'])

    @property
    def n_images(self):
        return len(self.dataset['images'])

    @property
    def n_cats(self):
        return len(self.dataset['categories'])

    def keypoint_annotation_frequency(self):
        """
        Example:
            >>> from kwcoco.coco_dataset import *
            >>> self = CocoDataset.demo('shapes', rng=0)
            >>> hist = self.keypoint_annotation_frequency()
            >>> hist = ub.odict(sorted(hist.items()))
            >>> # FIXME: for whatever reason demodata generation is not determenistic when seeded
            >>> print(ub.repr2(hist))  # xdoc: +IGNORE_WANT
            {
                'bot_tip': 6,
                'left_eye': 14,
                'mid_tip': 6,
                'right_eye': 14,
                'top_tip': 6,
            }
        """
        ann_kpcids = [kp['keypoint_category_id']
                      for ann in self.dataset['annotations']
                      for kp in ann.get('keypoints', [])]
        kpcid_to_name = {kpcat['id']: kpcat['name']
                         for kpcat in self.dataset['keypoint_categories']}
        kpcid_to_num = ub.dict_hist(ann_kpcids,
                                    labels=list(kpcid_to_name.keys()))
        kpname_to_num = ub.map_keys(kpcid_to_name, kpcid_to_num)
        return kpname_to_num

    def category_annotation_frequency(self):
        """
        Reports the number of annotations of each category

        Example:
            >>> from kwcoco.coco_dataset import *
            >>> self = CocoDataset.demo()
            >>> hist = self.category_annotation_frequency()
            >>> print(ub.repr2(hist))
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
            ub.map_vals(len, self.cid_to_aids))
        catname_to_nannots = ub.odict(sorted(catname_to_nannots.items(),
                                             key=lambda kv: (kv[1], kv[0])))
        return catname_to_nannots

    def category_annotation_type_frequency(self):
        """
        Reports the number of annotations of each type for each category

        Example:
            >>> self = CocoDataset.demo()
            >>> hist = self.category_annotation_frequency()
            >>> print(ub.repr2(hist))
        """
        catname_to_nannot_types = {}
        for cid, aids in self.cid_to_aids.items():
            name = self.cats[cid]['name']
            hist = ub.dict_hist(map(_annot_type, ub.take(self.anns, aids)))
            catname_to_nannot_types[name] = ub.map_keys(
                lambda k: k[0] if len(k) == 1 else k, hist)
        return catname_to_nannot_types

    def basic_stats(self):
        """
        Reports number of images, annotations, and categories.

        Example:
            >>> self = CocoDataset.demo()
            >>> print(ub.repr2(self.basic_stats()))
            {
                'n_anns': 11,
                'n_imgs': 3,
                'n_cats': 8,
            }
        """
        return ub.odict([
            ('n_anns', self.n_annots),
            ('n_imgs', self.n_images),
            ('n_cats', self.n_cats),
        ])

    def extended_stats(self):
        """
        Reports number of images, annotations, and categories.

        Example:
            >>> self = CocoDataset.demo()
            >>> print(ub.repr2(self.extended_stats()))
        """
        def mapping_stats(xid_to_yids):
            import kwarray
            n_yids = list(ub.map_vals(len, xid_to_yids).values())
            return kwarray.stats_dict(n_yids, n_extreme=True)
        return ub.odict([
            ('annots_per_img', mapping_stats(self.gid_to_aids)),
            # ('cats_per_img', mapping_stats(self.cid_to_gids)),
            ('annots_per_cat', mapping_stats(self.cid_to_aids)),
        ])

    def boxsize_stats(self, anchors=None, perclass=True, gids=None, aids=None,
                      verbose=0, clusterkw={}, statskw={}):
        """
        Compute statistics about bounding box sizes.

        Also computes anchor boxes using kmeans if ``anchors`` is specified.

        Args:
            anchors (int): if specified also computes box anchors
            perclass (bool): if True also computes stats for each category
            gids (List[int], default=None):
                if specified only compute stats for these image ids.
            aids (List[int], default=None):
                if specified only compute stats for these annotation ids.
            verbose (int): verbosity level
            clusterkw (dict): kwargs for :class:`sklearn.cluster.KMeans` used
                if computing anchors.
            statskw (dict): kwargs for :func:`kwarray.stats_dict`

        Returns:
            Dict[str, Dict[str, Dict | ndarray]

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('shapes32')
            >>> infos = self.boxsize_stats(anchors=4, perclass=False)
            >>> print(ub.repr2(infos, nl=-1, precision=2))

            >>> infos = self.boxsize_stats(gids=[1], statskw=dict(median=True))
            >>> print(ub.repr2(infos, nl=-1, precision=2))
        """
        import kwarray
        cname_to_box_sizes = ub.ddict(list)

        if bool(gids) and bool(aids):
            raise ValueError('specifying gids and aids is mutually exclusive')

        if gids is not None:
            aids = ub.flatten(ub.take(self.index.gid_to_aids, gids))
        if aids is not None:
            anns = ub.take(self.anns, aids)
        else:
            anns = self.dataset['annotations']

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


class _NextId(object):
    """ Helper class to tracks unused ids for new items """

    def __init__(self, parent):
        self.parent = parent
        self.unused = {
            'cid': None,
            'gid': None,
            'aid': None,
        }

    def set(self, key):
        # Determines what the next safe id can be
        key2 = {'aid': 'annotations', 'gid': 'images',
                'cid': 'categories'}[key]
        item_list = self.parent.dataset[key2]
        max_id = max(item['id'] for item in item_list) if item_list else 0
        next_id = max(max_id + 1, len(item_list))
        self.unused[key] = next_id
        # for i in it.count(len(self.cats) + 1):
        #     if i not in self.cats:
        #         return i

    def get(self, key):
        """ Get the next safe item id """
        if self.unused[key] is None:
            self.set(key)
        new_id = self.unused[key]
        self.unused[key] += 1
        return new_id


class MixinCocoDraw(object):
    """
    Matplotlib / display functionality
    """

    def imread(self, gid):
        """
        Loads a particular image
        """
        pass

    def show_image(self, gid=None, aids=None, aid=None, **kwargs):
        """
        Use matplotlib to show an image with annotations overlaid

        Args:
            gid (int): image to show
            aids (list): aids to highlight within the image
            aid (int): a specific aid to focus on. If gid is not give,
                look up gid based on this aid.
            **kwargs: show_all, show_aid, show_catname, show_kpname,

        Ignore:
            # Programatically collect the kwargs for docs generation
            import xinspect
            kwargs = xinspect.get_kwargs(kwcoco.CocoDataset.show_image)
            print(ub.repr2(list(kwargs.keys()), nl=1, si=1))

        """
        import matplotlib as mpl
        from matplotlib import pyplot as plt
        # from PIL import Image
        import kwimage
        import kwplot

        figkw = {k: kwargs[k] for k in ['fnum', 'pnum', 'doclf', 'docla']
                 if k in kwargs}
        if figkw:
            kwplot.figure(**figkw)

        if gid is None:
            primary_ann = self.anns[aid]
            gid = primary_ann['image_id']

        show_all = kwargs.get('show_all', False)

        highlight_aids = set()
        if aid is not None:
            highlight_aids.add(aid)
        if aids is not None:
            highlight_aids.update(aids)

        img = self.imgs[gid]
        aids = self.gid_to_aids.get(img['id'], [])

        # Collect annotation overlays
        colored_segments = ub.ddict(list)
        keypoints = []
        rects = []
        texts = []

        sseg_masks = []
        sseg_polys = []

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
                        # kpnames = []
                        # for p in kpts_data:
                        #     if 'keypoint_category_id' in p:
                        #         pass
                        #     pass
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
            if kwargs.get('show_aid', show_all):
                annot_text_parts.append('aid={}'.format(aid))
            if kwargs.get('show_catname', True):
                annot_text_parts.append(catname)
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
                    if kwargs.get('show_kpname', show_all):
                        if kpnames is not None:
                            for (kp_x, kp_y), kpname in zip(xys, kpnames):
                                texts.append((kp_x, kp_y, kpname, textkw))

            if 'segmentation' in ann and kwargs.get('show_segmentation', True):
                sseg = ann['segmentation']
                # Respect the 'color' attribute of categories
                if cat is not None:
                    catcolor = cat.get('color', None)
                else:
                    catcolor = None

                HAVE_KWIMAGE = True
                if HAVE_KWIMAGE:
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
                else:
                    # print('sseg = {!r}'.format(sseg))
                    if isinstance(sseg, dict):
                        # Handle COCO-RLE-segmentations; convert to raw binary masks
                        sseg = dict(sseg)
                        if 'shape' not in sseg and 'size' in sseg:
                            # NOTE: size here is actually h/w unlike almost
                            # everywhere else
                            sseg['shape'] = sseg['size']
                        if isinstance(sseg['counts'], (six.binary_type, six.text_type)):
                            mask = kwimage.Mask(sseg, 'bytes_rle').to_c_mask().data
                        else:
                            mask = kwimage.Mask(sseg, 'array_rle').to_c_mask().data
                        sseg_masks.append((mask, catcolor))
                    elif isinstance(sseg, list):
                        # Handle COCO-polygon-segmentation
                        # If the segmentation is a list of polygons
                        if not (len(sseg) and isinstance(sseg[0], list)):
                            sseg = [sseg]
                        for flat in sseg:
                            poly_xys = np.array(flat).reshape(-1, 2)
                            polykw = {}
                            if catcolor is not None:
                                polykw['color'] = catcolor

                            poly = mpl.patches.Polygon(poly_xys, **polykw)
                            sseg_polys.append(poly)
                    else:
                        raise TypeError(type(sseg))

        # Show image
        np_img = self.load_image(img)

        np_img = kwimage.atleast_3channels(np_img)

        np_img01 = None
        if np_img.dtype.kind in {'i', 'u'}:
            if np_img.max() > 255:
                np_img01 = np_img / np_img.max()

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

            with ub.Timer('overlay'):
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
                title_parts.append(img['file_name'])
            title = ' '.join(title_parts)
        if title:
            ax.set_title(title)

        if sseg_polys:
            # print('sseg_polys = {!r}'.format(sseg_polys))
            if True:
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
                line_col = mpl.collections.LineCollection(segments, 2, color=color)
                ax.add_collection(line_col)

            rect_col = mpl.collections.PatchCollection(rects, match_original=True)
            ax.add_collection(rect_col)
            if keypoints:
                xs, ys = np.vstack(keypoints).T
                ax.plot(xs, ys, 'bo')

        return ax


class MixinCocoAddRemove(object):
    """
    Mixin functions to dynamically add / remove annotations images and
    categories while maintaining lookup indexes.
    """

    def add_image(self, file_name, id=None, **kw):
        """
        Add an image to the dataset (dynamically updates the index)

        Args:
            file_name (str): image name
            id (None or int): ADVANCED. Force using this image id.

        Example:
            >>> self = CocoDataset.demo()
            >>> import kwimage
            >>> gname = kwimage.grab_test_image_fpath('paraview')
            >>> gid = self.add_image(gname)
            >>> assert self.imgs[gid]['file_name'] == gname
        """
        if id is None:
            id = self._next_ids.get('gid')
        elif self.imgs and id in self.imgs:
            raise IndexError('Image id={} already exists'.format(id))

        img = _dict()
        img['id'] = int(id)
        img['file_name'] = str(file_name)
        img.update(**kw)
        self.index._add_image(id, img)
        self.dataset['images'].append(img)
        self._invalidate_hashid()
        return id

    def add_annotation(self, image_id, category_id=None, bbox=None, id=None, **kw):
        """
        Add an annotation to the dataset (dynamically updates the index)

        Args:
            image_id (int): image_id to add to
            category_id (int): category_id to add to
            bbox (list or kwimage.Boxes): bounding box in xywh format
            id (None or int): ADVANCED. Force using this annotation id.

        Example:
            >>> self = CocoDataset.demo()
            >>> image_id = 1
            >>> cid = 1
            >>> bbox = [10, 10, 20, 20]
            >>> aid = self.add_annotation(image_id, cid, bbox)
            >>> assert self.anns[aid]['bbox'] == bbox
        """
        if id is None:
            id = self._next_ids.get('aid')
        elif self.anns and id in self.anns:
            raise IndexError('Annot id={} already exists'.format(id))

        ann = _dict()
        ann['id'] = int(id)
        ann['image_id'] = int(image_id)
        ann['category_id'] = None if category_id is None else int(category_id)
        if bbox is not None:
            try:
                import kwimage
                if isinstance(bbox, kwimage.Boxes):
                    bbox = bbox.to_xywh().data.tolist()
            except ImportError:
                pass
            ann['bbox'] = bbox
        # assert not set(kw).intersection(set(ann))
        ann.update(**kw)
        self.dataset['annotations'].append(ann)
        self.index._add_annotation(id, image_id, category_id, ann)
        self._invalidate_hashid(['annotations'])
        return id

    def add_category(self, name, supercategory=None, id=None, **kw):
        """
        Adds a category

        Args:
            name (str): name of the new category
            supercategory (str, optional): parent of this category
            id (int, optional): use this category id, if it was not taken

        Example:
            >>> self = CocoDataset.demo()
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
            raise ValueError('Category name={!r} already exists'.format(name))

        if id is None:
            id = self._next_ids.get('cid')
        elif index.cats and id in index.cats:
            raise IndexError('Category id={} already exists'.format(id))

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
        Like add_image, but returns the existing image id if it already
        exists instead of failing. In this case all metadata is ignored.

        Returns:
            int: the existing or new image id
        """
        try:
            id = self.add_image(file_name=file_name, id=id, **kw)
        except ValueError:
            img = self.index.file_name_to_img[file_name]
            id = img['id']
        return id

    def ensure_category(self, name, supercategory=None, id=None, **kw):
        """
        Like add_category, but returns the existing category id if it already
        exists instead of failing. In this case all metadata is ignored.

        Returns:
            int: the existing or new category id
        """
        try:
            id = self.add_category(name=name, supercategory=supercategory,
                                   id=id, **kw)
        except ValueError:
            cat = self.index.name_to_cat[name]
            id = cat['id']
        return id

    def add_annotations(self, anns):
        """
        Faster less-safe multi-item alternative

        Args:
            anns (List[Dict]): list of annotation dictionaries

        Example:
            >>> self = CocoDataset.demo()
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

        Note:
            THIS FUNCTION WAS DESIGNED FOR SPEED, AS SUCH IT DOES NOT CHECK IF
            THE IMAGE-IDs or FILE_NAMES ARE DUPLICATED AND WILL BLINDLY ADD
            DATA EVEN IF IT IS BAD. THE SINGLE IMAGE VERSION IS SLOWER BUT
            SAFER.

        Args:
            imgs (List[Dict]): list of image dictionaries

        Example:
            >>> imgs = CocoDataset.demo().dataset['images']
            >>> self = CocoDataset()
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
            >>> self = CocoDataset.demo()
            >>> self.clear_images()
            >>> print(ub.repr2(self.basic_stats(), nobr=1, nl=0, si=1))
            n_anns: 0, n_imgs: 0, n_cats: 8
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
            >>> self = CocoDataset.demo()
            >>> self.clear_annotations()
            >>> print(ub.repr2(self.basic_stats(), nobr=1, nl=0, si=1))
            n_anns: 0, n_imgs: 3, n_cats: 8
        """
        # self.dataset['annotations'].clear()
        del self.dataset['annotations'][:]
        self.index._remove_all_annotations()
        self._invalidate_hashid(['annotations'])

    remove_all_images = clear_images
    remove_all_annotations = clear_annotations

    def remove_annotation(self, aid_or_ann):
        """
        Remove a single annotation from the dataset

        If you have multiple annotations to remove its more efficient to remove
        them in batch with `self.remove_annotations`

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

            safe (bool, default=True): if True, we perform checks to remove
                duplicates and non-existing identifiers.

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
            delitems(self.dataset['annotations'], remove_idxs)

            self.index._remove_annotations(remove_aids, verbose=verbose)
            self._invalidate_hashid(['annotations'])
        return remove_info

    def remove_categories(self, cat_identifiers, keep_annots=False, verbose=0,
                          safe=True):
        """
        Remove categories and all annotations in those categories.
        Currently does not change any hierarchy information

        Args:
            cat_identifiers (List): list of category dicts, names, or ids

            keep_annots (bool, default=False):
                if True, keeps annotations, but removes category labels.

            safe (bool, default=True): if True, we perform checks to remove
                duplicates and non-existing identifiers.

        Returns:
            Dict: num_removed: information on the number of items removed

        Example:
            >>> self = CocoDataset.demo()
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
            if self.cid_to_aids:
                remove_aids = list(it.chain(*[self.cid_to_aids[cid]
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
            cid_to_index = {
                cat['id']: index
                for index, cat in enumerate(self.dataset['categories'])
            }
            # Lookup the indices to remove, sort in descending order
            remove_idxs = list(ub.take(cid_to_index, remove_cids))
            delitems(self.dataset['categories'], remove_idxs)

            self.index._remove_categories(remove_cids, verbose=verbose)
            self._invalidate_hashid(['categories', 'annotations'])

        return remove_info

    def remove_images(self, gids_or_imgs, verbose=0, safe=True):
        """
        Args:
            gids_or_imgs (List): list of image dicts, names, or ids

            safe (bool, default=True): if True, we perform checks to remove
                duplicates and non-existing identifiers.

        Returns:
            Dict: num_removed: information on the number of items removed

        Example:
            >>> from kwcoco.coco_dataset import *
            >>> self = CocoDataset.demo()
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
                print('Removing annots of removed images')

            remove_gids = list(map(self._resolve_to_gid, gids_or_imgs))
            if safe:
                remove_gids = sorted(set(remove_gids))
            # First remove any annotation that belongs to those images
            if self.gid_to_aids:
                remove_aids = list(it.chain(*[self.gid_to_aids[gid]
                                              for gid in remove_gids]))
            else:
                remove_aids = [ann['id'] for ann in self.dataset['annotations']
                               if ann['image_id'] in remove_gids]

            rminfo = self.remove_annotations(remove_aids, verbose=verbose)
            remove_info.update(rminfo)

            remove_info['images'] = len(remove_gids)
            if verbose > 1:
                print('Removing {} image entries'.format(len(remove_gids)))
            gid_to_index = {
                img['id']: index
                for index, img in enumerate(self.dataset['images'])
            }
            # Lookup the indices to remove, sort in descending order
            remove_idxs = list(ub.take(gid_to_index, remove_gids))
            delitems(self.dataset['images'], remove_idxs)

            self.index._remove_images(remove_gids, verbose=verbose)
            self._invalidate_hashid(['images', 'annotations'])

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
            delitems(ann['keypoints'], remove_idxs)
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
            >>> self = CocoDataset.demo('shapes', rng=0)
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
    """

    # _set = ub.oset  # many operations are much slower for oset
    _set = set

    def __init__(self):
        self.anns = None
        self.imgs = None
        self.cats = None
        self._id_lookup = None
        self.gid_to_aids = None
        self.cid_to_aids = None
        self.name_to_cat = None
        self.file_name_to_img = None
        self._CHECKS = True

    def __bool__(self):
        return self.anns is not None

    __nonzero__ = __bool__  # python 2 support

    def _add_image(self, gid, img):
        if self.imgs is not None:
            file_name = img['file_name']
            if self._CHECKS:
                if file_name in self.file_name_to_img:
                    raise ValueError(
                        'image with file_name={} already exists'.format(
                            file_name))
            self.imgs[gid] = img
            self.gid_to_aids[gid] = self._set()
            self.file_name_to_img[file_name] = img

    def _add_images(self, imgs):
        """
        Note:
            THIS FUNCTION WAS DESIGNED FOR SPEED, AS SUCH IT DOES NOT CHECK IF
            THE IMAGE-IDs or FILE_NAMES ARE DUPLICATED AND WILL BLINDLY ADD
            DATA EVEN IF IT IS BAD. THE SINGLE IMAGE VERSION IS SLOWER BUT
            SAFER.

        Ignore:
            # If we did do checks, what would be the fastest way?

            import kwcoco
            x = kwcoco.CocoDataset()
            for i in range(1000):
                x.add_image(file_name=str(i))

            y = kwcoco.CocoDataset()
            for i in range(1000, 2000):
                y.add_image(file_name=str(i))

            imgs = list(y.imgs.values())
            new_file_name_to_img = {img['file_name']: img for img in imgs}

            import ubelt as ub
            ti = ub.Timerit(100, bestof=10, verbose=2)

            for timer in ti.reset('set intersection'):
                with timer:
                    # WINNER
                    bool(set(x.index.file_name_to_img) & set(new_file_name_to_img))

            for timer in ti.reset('dict contains'):
                with timer:
                    any(f in x.index.file_name_to_img
                        for f in new_file_name_to_img.keys())
        """
        if self.imgs is not None:
            gids = [img['id'] for img in imgs]
            new_imgs = dict(zip(gids, imgs))
            self.imgs.update(new_imgs)
            self.file_name_to_img.update(
                {img['file_name']: img for img in imgs})
            for gid in gids:
                self.gid_to_aids[gid] = self._set()

    def _add_annotation(self, aid, gid, cid, ann):
        if self.anns is not None:
            self.anns[aid] = ann
            self.gid_to_aids[gid].add(aid)
            self.cid_to_aids[cid].add(aid)

    def _add_annotations(self, anns):
        if self.anns is not None:
            aids = [ann['id'] for ann in anns]
            gids = [ann['image_id'] for ann in anns]
            cids = [ann['category_id'] for ann in anns]
            new_anns = dict(zip(aids, anns))
            self.anns.update(new_anns)
            for gid, cid, aid in zip(gids, cids, aids):
                self.gid_to_aids[gid].add(aid)
                self.cid_to_aids[cid].add(aid)

    def _add_category(self, cid, name, cat):
        if self.cats is not None:
            self.cats[cid] = cat
            self.cid_to_aids[cid] = self._set()
            self.name_to_cat[name] = cat

    def _remove_all_annotations(self):
        # Keep the category and image indexes alive
        if self.anns is not None:
            self.anns.clear()
            for _ in self.gid_to_aids.values():
                _.clear()
            for _ in self.cid_to_aids.values():
                _.clear()

    def _remove_all_images(self):
        # Keep the category indexes alive
        if self.imgs is not None:
            self.imgs.clear()
            self.anns.clear()
            self.gid_to_aids.clear()
            self.file_name_to_img.clear()
            for _ in self.cid_to_aids.values():
                _.clear()

    def _remove_annotations(self, remove_aids, verbose=0):
        if self.anns is not None:
            if verbose > 1:
                print('Updating annotation index')
            # This is faster for simple set cid_to_aids
            for aid in remove_aids:
                ann = self.anns.pop(aid)
                gid = ann['image_id']
                cid = ann['category_id']
                self.cid_to_aids[cid].remove(aid)
                self.gid_to_aids[gid].remove(aid)

    def _remove_categories(self, remove_cids, verbose=0):
        # dynamically update the category index
        if self.cats is not None:
            for cid in remove_cids:
                cat = self.cats.pop(cid)
                del self.cid_to_aids[cid]
                del self.name_to_cat[cat['name']]
            if verbose > 2:
                print('Updated category index')

    def _remove_images(self, remove_gids, verbose=0):
        # dynamically update the image index
        if self.imgs is not None:
            for gid in remove_gids:
                img = self.imgs.pop(gid)
                del self.gid_to_aids[gid]
                del self.file_name_to_img[img['file_name']]
            if verbose > 2:
                print('Updated image index')

    def clear(self):
        self.anns = None
        self.imgs = None
        self.cats = None
        self._id_lookup = None
        self.gid_to_aids = None
        self.cid_to_aids = None
        self.name_to_cat = None
        self.file_name_to_img = None

    def build(self, parent):
        """
        build reverse indexes

        Notation:
            aid - Annotation ID
            gid - imaGe ID
            cid - Category ID
        """
        # create index
        anns, cats, imgs = {}, {}, {}
        gid_to_aids = ub.ddict(self._set)
        cid_to_aids = ub.ddict(self._set)

        # Build one-to-one self-lookup maps
        for cat in parent.dataset.get('categories', []):
            cid = cat['id']
            if cid in cat:
                warnings.warn(
                    'Categories have the same id in {}:\n{} and\n{}'.format(
                        parent, cats[cid], cat))
            cats[cid] = cat

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
        for ann in anns.values():
            try:
                aid = ann['id']
                gid = ann['image_id']
            except KeyError:
                raise KeyError('Annotation does not have ids {}'.format(ann))

            if not isinstance(aid, INT_TYPES):
                raise TypeError('bad aid={} type={}'.format(aid, type(aid)))
            if not isinstance(gid, INT_TYPES):
                raise TypeError('bad gid={} type={}'.format(gid, type(gid)))

            gid_to_aids[gid].add(aid)
            if gid not in imgs:
                warnings.warn('Annotation {} in {} references '
                              'unknown image_id'.format(ann, parent))

            ALLOW_EMPTY_CATEGORIES = True

            try:
                cid = ann['category_id']
            except KeyError:
                if ALLOW_EMPTY_CATEGORIES:
                    warnings.warn('Annotation {} in {} is missing '
                                  'a category_id'.format(ann, parent))
                else:
                    raise KeyError(
                        'Annotation does not have category id {}'.format(ann))
            else:
                cid_to_aids[cid].add(aid)

                if not isinstance(cid, INT_TYPES) and cid is not None:
                    raise TypeError('bad cid={} type={}'.format(cid, type(cid)))

                if cid not in cats and cid is not None:
                    warnings.warn('Annotation {} in {} references '
                                  'unknown category_id'.format(ann, parent))

        # Fix one-to-zero cases
        for cid in cats.keys():
            if cid not in cid_to_aids:
                cid_to_aids[cid] = self._set()

        for gid in imgs.keys():
            if gid not in gid_to_aids:
                gid_to_aids[gid] = self._set()

        # create class members
        self._id_lookup = {
            'categories': cats,
            'images': imgs,
            'annotations': anns,
        }
        self.anns = anns
        self.imgs = imgs
        self.cats = cats

        self.gid_to_aids = gid_to_aids
        self.cid_to_aids = cid_to_aids
        self.name_to_cat = {cat['name']: cat for cat in self.cats.values()}
        self.file_name_to_img = {
            img['file_name']: img for img in self.imgs.values()}


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

    @property
    def gid_to_aids(self):
        return self.index.gid_to_aids

    @property
    def cid_to_aids(self):
        return self.index.cid_to_aids

    @property
    def name_to_cat(self):
        return self.index.name_to_cat


class CocoDataset(ub.NiceRepr, MixinCocoAddRemove, MixinCocoStats,
                  MixinCocoAttrs, MixinCocoDraw, MixinCocoExtras,
                  MixinCocoIndex, MixinCocoDepricate):
    """
    Notes:
        A keypoint annotation
            {
                "image_id" : int,
                "category_id" : int,
                "keypoints" : [x1,y1,v1,...,xk,yk,vk],
                "score" : float,
            }
            Note that `v[i]` is a visibility flag, where v=0: not labeled,
                v=1: labeled but not visible, and v=2: labeled and visible.

        A bounding box annotation
            {
                "image_id" : int,
                "category_id" : int,
                "bbox" : [x,y,width,height],
                "score" : float,
            }

        We also define a non-standard "line" annotation (which
            our fixup scripts will interpret as the diameter of a circle to
            convert into a bounding box)

        A line* annotation (note this is a non-standard field)
            {
                "image_id" : int,
                "category_id" : int,
                "line" : [x1,y1,x2,y2],
                "score" : float,
            }

        Lastly, note that our datasets will sometimes specify multiple bbox,
        line, and/or, keypoints fields. In this case we may also specify a
        field roi_shape, which denotes which field is the "main" annotation
        type.

    Attributes:
        dataset (Dict): raw json data structure. This is the base dictionary
            that contains {'annotations': List, 'images': List,
            'categories': List}

        index (CocoIndex): an efficient lookup index into the coco data
            structure. The index defines its own attributes like
            `anns`, `cats`, `imgs`, etc. See :class:`CocoIndex` for more
            details on which attributes are available.

        fpath (PathLike | None):
            if known, this stores the filepath the dataset was loaded from

        tag (str):
            A tag indicating the name of the dataset.

        img_root (PathLike | None) :
            If known, this is the root path that all image file names are
            relative to. This can also be manually overwritten by the user.

        hashid (str | None) :
            If computed, this will be a hash uniquely identifing the dataset.
            To ensure this is computed see  :func:`_build_hashid`.

    References:
        http://cocodataset.org/#format
        http://cocodataset.org/#download

    CommandLine:
        python -m kwcoco.coco_dataset CocoDataset --show

    Example:
        >>> dataset = demo_coco_data()
        >>> self = CocoDataset(dataset, tag='demo')
        >>> # xdoctest: +REQUIRES(--show)
        >>> self.show_image(gid=2)
        >>> from matplotlib import pyplot as plt
        >>> plt.show()
    """

    def __init__(self, data=None, tag=None, img_root=None, autobuild=True):
        if data is None:
            data = {
                'categories': [],
                'images': [],
                'annotations': [],
                'licenses': [],
                'info': [],
            }

        fpath = None

        if isinstance(data, six.string_types):
            fpath = data
            key = basename(fpath)
            data = json.load(open(fpath, 'r'))

            # If data is a path it gives us the absolute location of the root
            root = dirname(fpath)
            if tag is None:
                tag = key
        else:
            # If data is a dict, we dont know where the root is, so assume its
            # relative to the cwd.
            root = '.'
            if not isinstance(data, dict):
                raise TypeError('data must be a dict or path to json file')

        if img_root is None:
            if 'img_root' in data:
                # allow image root to be specified in the dataset
                _root = data['img_root']
                if _root is None:
                    _root = ''
                elif isinstance(_root, six.string_types):
                    import os
                    _tmp = ub.expandpath(_root)
                    if os.path.exists(_tmp):
                        _root = _tmp
                else:
                    if isinstance(_root, list) and _root == []:
                        _root = ''
                    else:
                        raise TypeError('_root = {!r}'.format(_root))
                try:
                    img_root = join(root, _root)
                except Exception:
                    print('_root = {!r}'.format(_root))
                    print('root = {!r}'.format(root))
                    raise
            else:
                img_root = root

        self.index = CocoIndex()

        self.hashid = None
        self.hashid_parts = None
        self.fpath = fpath

        self.tag = tag
        self.dataset = data
        self.img_root = ub.expandpath(img_root)

        # Keep track of an unused id we may use
        self._next_ids = _NextId(self)

        if autobuild:
            self._build_index()

    @classmethod
    def from_image_paths(cls, gpaths):
        """
        Create a coco dataset from a list of images paths

        Example:
            >>> coco_dset = CocoDataset.from_image_paths(['a.png', 'b.png'])
            >>> assert coco_dset.n_images == 2
        """
        coco_dset = cls()
        for gpath in gpaths:
            coco_dset.add_image(gpath)
        return coco_dset

    def copy(self):
        """
        Deep copies this object

        Example:
            >>> from kwcoco.coco_dataset import *
            >>> self = CocoDataset.demo()
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
            info = ub.repr2(self.basic_stats(), kvsep='=', si=1, nobr=1, nl=0)
            parts.append(info)
        return ', '.join(parts)

    def dumps(self, indent=None, newlines=False):
        """
        Writes the dataset out to the json format

        Args:
            newlines (bool) :
                if True, each annotation, image, category gets its own line

        Notes:
            Using newlines=True is similar to:
                print(ub.repr2(dset.dataset, nl=2, trailsep=False))
                However, the above may not output valid json if it contains
                ndarrays.

        Example:
            >>> from kwcoco.coco_dataset import *
            >>> import json
            >>> self = CocoDataset.demo()
            >>> text = self.dumps(newlines=True)
            >>> print(text)
            >>> self2 = CocoDataset(json.loads(text), tag='demo2')
            >>> assert self2.dataset == self.dataset
            >>> assert self2.dataset is not self.dataset

            >>> text = self.dumps(newlines=True)
            >>> print(text)
            >>> self2 = CocoDataset(json.loads(text), tag='demo2')
            >>> assert self2.dataset == self.dataset
            >>> assert self2.dataset is not self.dataset

        Ignore:
            for k in self2.dataset:
                if self.dataset[k] == self2.dataset[k]:
                    print('YES: k = {!r}'.format(k))
                else:
                    print('NO: k = {!r}'.format(k))
            self2.dataset['categories']
            self.dataset['categories']

        """
        def _json_dumps(data, indent=None):
            fp = StringIO()
            json.dump(data, fp, indent=indent, ensure_ascii=False)
            fp.seek(0)
            text = fp.read()
            return text

        # Instead of using json to dump the whole thing make the text a bit
        # more pretty.
        if newlines:
            if indent is None:
                indent = ''
            if isinstance(indent, int):
                indent = ' ' * indent
            dict_lines = []
            main_keys = ['info', 'licenses', 'categories',
                         'keypoint_categories', 'images', 'annotations']
            other_keys = sorted(set(self.dataset.keys()) - set(main_keys))
            for key in main_keys:
                if key not in self.dataset:
                    continue
                # We know each main entry is a list, so make it such that
                # Each entry gets its own line
                value = self.dataset[key]
                value_lines = [_json_dumps(v) for v in value]
                if value_lines:
                    value_body = (',\n' + indent).join(value_lines)
                    value_repr = '[\n' + indent + value_body + '\n]'
                else:
                    value_repr = '[]'
                item_repr = '{}: {}'.format(_json_dumps(key), value_repr)
                dict_lines.append(item_repr)

            for key in other_keys:
                # Dont assume anything about other data
                value = self.dataset.get(key, [])
                value_repr = _json_dumps(value)
                item_repr = '{}: {}'.format(_json_dumps(key), value_repr)
                dict_lines.append(item_repr)
            text = '{\n' + ',\n'.join(dict_lines) + '\n}'
        else:
            text = _json_dumps(self.dataset, indent=indent)

        return text

    def dump(self, file, indent=None, newlines=False):
        """
        Writes the dataset out to the json format

        Args:
            file (PathLike | FileLike):
                Where to write the data.  Can either be a path to a file or an
                open file pointer / stream.

            newlines (bool) : if True, each annotation, image, category gets
                its own line.

        Example:
            >>> import tempfile
            >>> from kwcoco.coco_dataset import *
            >>> self = CocoDataset.demo()
            >>> file = tempfile.NamedTemporaryFile('w')
            >>> self.dump(file)
            >>> file.seek(0)
            >>> text = open(file.name, 'r').read()
            >>> print(text)
            >>> file.seek(0)
            >>> dataset = json.load(open(file.name, 'r'))
            >>> self2 = CocoDataset(dataset, tag='demo2')
            >>> assert self2.dataset == self.dataset
            >>> assert self2.dataset is not self.dataset

            >>> file = tempfile.NamedTemporaryFile('w')
            >>> self.dump(file, newlines=True)
            >>> file.seek(0)
            >>> text = open(file.name, 'r').read()
            >>> print(text)
            >>> file.seek(0)
            >>> dataset = json.load(open(file.name, 'r'))
            >>> self2 = CocoDataset(dataset, tag='demo2')
            >>> assert self2.dataset == self.dataset
            >>> assert self2.dataset is not self.dataset
        """
        if isinstance(file, six.string_types):
            with open(file, 'w') as fp:
                self.dump(fp, indent=indent, newlines=newlines)
        else:
            if newlines:
                file.write(self.dumps(indent=indent, newlines=newlines))
            else:
                json.dump(self.dataset, file, indent=indent, ensure_ascii=False)

    def _check_integrity(self):
        """ perform all checks """
        self._check_index()
        self._check_pointers()
        assert len(self.missing_images()) == 0

    def _check_index(self):
        # We can verify our index invariants by copying the raw dataset and
        # checking if the newly constructed index is the same as this index.
        new = copy.copy(self)
        new.dataset = copy.deepcopy(self.dataset)
        new._build_index()
        assert self.index.anns == new.index.anns
        assert self.index.imgs == new.index.imgs
        assert self.index.cats == new.index.cats
        assert self.index.gid_to_aids == new.index.gid_to_aids
        assert self.index.cid_to_aids == new.index.cid_to_aids
        assert self.index.name_to_cat == new.index.name_to_cat
        assert self.index.file_name_to_img == new.index.file_name_to_img
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

            if cid not in self.cats:
                if cid is not None:
                    errors.append('aid={} references bad cid={}'.format(aid, cid))
            else:
                if self.cats[cid]['id'] != cid:
                    errors.append('cid={} has a bad index'.format(cid))

            if gid not in self.imgs:
                errors.append('aid={} references bad gid={}'.format(aid, gid))
            else:
                if self.imgs[gid]['id'] != gid:
                    errors.append('gid={} has a bad index'.format(gid))
        if errors:
            raise Exception('\n'.join(errors))
        elif verbose:
            print('Pointers are consistent')
        return True

    def _build_index(self):
        self.index.build(self)

    def _clear_index(self):
        self.index.clear()

    def union(self, *others, **kwargs):
        """
        Merges multiple `CocoDataset` items into one. Names and associations
        are retained, but ids may be different.

        Args:
            self : note that `union` can be called as an instance method or a class method.
                If it is a class method, then this is the class type, otherwise the instance
                will also be unioned with `others`.
            *others : a series of CocoDatasets that we will merge
            **kwargs : constructor options for the new merged CocoDataset

        Returns:
            CocoDataset: a new merged coco dataset

        Example:
            >>> # Test union works with different keypoint categories
            >>> dset1 = CocoDataset.demo('shapes1')
            >>> dset2 = CocoDataset.demo('shapes2')
            >>> dset1.remove_keypoint_categories(['bot_tip', 'mid_tip', 'right_eye'])
            >>> dset2.remove_keypoint_categories(['top_tip', 'left_eye'])
            >>> dset_12a = CocoDataset.union(dset1, dset2)
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
            >>> import kwcoco
            >>> dset1 = kwcoco.CocoDataset.demo('shapes3')
            >>> for new_gid, img in enumerate(dset1.dataset['images'], start=10):
            >>>     for aid in dset1.gid_to_aids[img['id']]:
            >>>         dset1.anns[aid]['image_id'] = new_gid
            >>>     img['id'] = new_gid
            >>> dset1._clear_index()
            >>> dset1._build_index()
            >>> # ------
            >>> dset2 = kwcoco.CocoDataset.demo('shapes2')
            >>> for new_gid, img in enumerate(dset2.dataset['images'], start=100):
            >>>     for aid in dset2.gid_to_aids[img['id']]:
            >>>         dset2.anns[aid]['image_id'] = new_gid
            >>>     img['id'] = new_gid
            >>> dset2._clear_index()
            >>> dset2._build_index()
            >>> others = [dset1, dset2]
            >>> merged = kwcoco.CocoDataset.union(*others)
            >>> print('merged = {!r}'.format(merged))
            >>> print('merged.imgs = {}'.format(ub.repr2(merged.imgs, nl=1)))
            >>> assert set(merged.imgs) & set([10, 11, 12, 100, 101]) == set(merged.imgs)

            >>> # Test data is not preserved
            >>> dset2 = kwcoco.CocoDataset.demo('shapes2')
            >>> dset1 = kwcoco.CocoDataset.demo('shapes3')
            >>> others = (dset1, dset2)
            >>> cls = self = kwcoco.CocoDataset
            >>> merged = cls.union(*others)
            >>> print('merged = {!r}'.format(merged))
            >>> print('merged.imgs = {}'.format(ub.repr2(merged.imgs, nl=1)))
            >>> assert set(merged.imgs) & set([1, 2, 3, 4, 5]) == set(merged.imgs)

        TODO:
            - [ ] are supercategories broken?
            - [ ] reuse image ids where possible
            - [ ] reuse annotation / category ids where possible
        """
        if self.__class__ is type:
            # Method called as classmethod
            cls = self
        else:
            # Method called as instancemethod
            cls = self.__class__
            others = (self,) + others

        def _coco_union(relative_dsets, common_root):
            """ union of dictionary based data structure """
            merged = _dict([
                ('categories', []),
                ('licenses', []),
                ('info', []),
                ('images', []),
                ('annotations', []),
            ])

            # TODO: need to handle keypoint_categories

            merged_cat_name_to_id = {}
            merged_kp_name_to_id = {}

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

            _all_imgs = (img for _, d in relative_dsets for img in d['images'])
            _all_gids = (img['id'] for img in _all_imgs)
            preserve_gids = not _has_duplicates(_all_gids)

            for subdir, old_dset in relative_dsets:
                # Create temporary indexes to map from old to new
                cat_id_map = {None: None}
                img_id_map = {}
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

                # Add the images into the merged dataset
                for old_img in old_dset['images']:
                    if preserve_gids:
                        new_id = old_img['id']
                    else:
                        new_id = len(merged['images']) + 1
                    new_img = _dict([
                        ('id', new_id),
                        ('file_name', join(subdir, old_img['file_name'])),
                    ])
                    # copy over other metadata
                    update_ifnotin(new_img, old_img)
                    img_id_map[old_img['id']] = new_img['id']
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
                        # raise Exception
                    if new_img_id is None:
                        warnings.warn('annot {} in {} has bad image-id {}'.format(
                            old_annot, subdir, old_img_id))
                        # sanity check:
                        # if any(img['id'] == old_img_id for img in old_dset['images']):
                        #     raise Exception('Image id {} does not exist in {}'.format(old_img_id, subdir))
                    new_annot = _dict([
                        ('id', len(merged['annotations']) + 1),
                        ('image_id', new_img_id),
                        ('category_id', new_cat_id),
                    ])
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
            return merged

        # handle soft data roots
        from os.path import normpath
        soft_dset_roots = [dset.img_root for dset in others]
        soft_dset_roots = [normpath(r) if r is not None else None for r in soft_dset_roots]
        if ub.allsame(soft_dset_roots):
            soft_img_root = ub.peek(soft_dset_roots)
        else:
            soft_img_root = None

        # Handle hard coded data roots
        from os.path import normpath
        hard_dset_roots = [dset.dataset.get('img_root', None) for dset in others]
        hard_dset_roots = [normpath(r) if r is not None else None for r in hard_dset_roots]
        if ub.allsame(hard_dset_roots):
            common_root = ub.peek(hard_dset_roots)
            relative_dsets = [('', d.dataset) for d in others]
        else:
            common_root = None
            relative_dsets = [(d.img_root, d.dataset) for d in others]

        merged = _coco_union(relative_dsets, common_root)

        if common_root is not None:
            merged['img_root'] = common_root

        new_dset = cls(merged, **kwargs)

        if common_root is None and soft_img_root is not None:
            new_dset.img_root = soft_img_root
        return new_dset

    def subset(self, gids, copy=False):
        """
        Return a subset of the larger coco dataset by specifying which images
        to port. All annotations in those images will be taken.

        Args:
            gids (List[int]): image-ids to copy into a new dataset
            copy (bool, default=False): if True, makes a deep copy of
                all nested attributes, otherwise makes a shallow copy.

        Example:
            >>> self = CocoDataset.demo()
            >>> gids = [1, 3]
            >>> sub_dset = self.subset(gids)
            >>> assert len(self.gid_to_aids) == 3
            >>> assert len(sub_dset.gid_to_aids) == 2

        Example:
            >>> self = CocoDataset.demo()
            >>> sub1 = self.subset([1])
            >>> sub2 = self.subset([2])
            >>> sub3 = self.subset([3])
            >>> others = [sub1, sub2, sub3]
            >>> rejoined = CocoDataset.union(*others)
            >>> assert len(sub1.anns) == 9
            >>> assert len(sub2.anns) == 2
            >>> assert len(sub3.anns) == 0
            >>> assert rejoined.basic_stats() == self.basic_stats()
        """
        new_dataset = _dict([(k, []) for k in self.dataset])
        new_dataset['categories'] = self.dataset['categories']
        new_dataset['info'] = self.dataset.get('info', [])
        new_dataset['licenses'] = self.dataset.get('licenses', [])

        if 'keypoint_categories' in self.dataset:
            new_dataset['keypoint_categories'] = self.dataset['keypoint_categories']

        gids = sorted(set(gids))
        sub_aids = sorted([aid for gid in gids
                           for aid in self.gid_to_aids.get(gid, [])])
        new_dataset['annotations'] = list(ub.take(self.anns, sub_aids))
        new_dataset['images'] = list(ub.take(self.imgs, gids))
        new_dataset['img_root'] = self.dataset.get('img_root', None)

        if copy:
            from copy import deepcopy
            new_dataset = deepcopy(new_dataset)

        sub_dset = CocoDataset(new_dataset, img_root=self.img_root)
        return sub_dset


def delitems(items, remove_idxs, thresh=750):
    """
    Args:
        items (List): list which will be modified
        remove_idxs (List[int]): integers to remove (MUST BE UNIQUE)
    """
    if len(remove_idxs) > thresh:
        # Its typically faster to just make a new list when there are
        # lots and lots of items to remove.
        keep_idxs = sorted(set(range(len(items))) - set(remove_idxs))
        newlist = [items[idx] for idx in keep_idxs]
        items[:] = newlist
    else:
        # However, when there are a few hundred items to remove, del is faster.
        for idx in sorted(remove_idxs, reverse=True):
            del items[idx]


def demo_coco_data():
    """
    Simple data for testing

    Ignore:
        # code for getting a segmentation polygon
        kwimage.grab_test_image_fpath('astro')
        labelme /home/joncrall/.cache/kwimage/demodata/astro.png
        cat /home/joncrall/.cache/kwimage/demodata/astro.json

    Example:
        >>> # xdoctest: +REQUIRES(--show)
        >>> from kwcoco.coco_dataset import demo_coco_data, CocoDataset
        >>> dataset = demo_coco_data()
        >>> self = CocoDataset(dataset, tag='demo')
        >>> import kwplot
        >>> kwplot.autompl()
        >>> self.show_image(gid=1)
        >>> kwplot.show_if_requested()
    """
    import kwimage
    from kwimage.im_demodata import _TEST_IMAGES
    from os.path import commonprefix, relpath

    test_imgs_keys = ['astro', 'carl', 'stars']
    urls = {k: _TEST_IMAGES[k]['url'] for k in test_imgs_keys}
    gpaths = {k: kwimage.grab_test_image_fpath(k) for k in test_imgs_keys}
    img_root = commonprefix(list(gpaths.values()))

    gpath1, gpath2, gpath3 = ub.take(gpaths, test_imgs_keys)
    url1, url2, url3 = ub.take(urls, test_imgs_keys)
    # gpath2 = kwimage.grab_test_image_fpath('carl')
    # gpath3 = kwimage.grab_test_image_fpath('stars')
    # gpath1 = ub.grabdata('https://i.imgur.com/KXhKM72.png')
    # gpath2 = ub.grabdata('https://i.imgur.com/flTHWFD.png')
    # gpath3 = ub.grabdata('https://i.imgur.com/kCi7C1r.png')

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
                    'mouth-right-corner',
                    'mouth-right-bot',
                    'mouth-left-bot',
                    'mouth-left-corner',
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
            # {'id': 1, 'file_name': gname1},
            # {'id': 2, 'file_name': gname2},
            # {'id': 3, 'file_name': gname3},
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
                 202, 139, 2,
                 215, 150, 2,
                 229, 150, 2,
                 244, 142, 2,
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
             'bbox': [124, 96, 45, 18]}
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
