"""
Vectorized ORM-like objects used in conjunction with coco_dataset.

This powers the ``.images()``, ``.videos()``, and ``.annotation()`` methods of
:class:`kwcoco.CocoDataset`.

See:
    :func:`kwcoco.coco_dataset.MixinCocoObjects.categories`
    :func:`kwcoco.coco_dataset.MixinCocoObjects.videos`
    :func:`kwcoco.coco_dataset.MixinCocoObjects.images`
    :func:`kwcoco.coco_dataset.MixinCocoObjects.annots`

"""
from os.path import join
import numpy as np
import ubelt as ub


__docstubs__ = """
from typing import Dict
ObjT = Dict
"""


class ObjectList1D(ub.NiceRepr):
    """
    Vectorized access to lists of dictionary objects

    Lightweight reference to a set of object (e.g. annotations, images) that
    allows for convenient property access.

    Args:
        ids (List[int]): list of ids
        dset (CocoDataset): parent dataset
        key (str): main object name (e.g. 'images', 'annotations')

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

    def __getitem__(self, index):
        if isinstance(index, slice):
            subids = self._ids[index]
            newself = self.__class__(subids, self._dset)
            return newself
        else:
            return self._ids[index]

    def unique(self):
        """
        Removes any duplicates entries in this object

        Returns:
            ObjectList1D
        """
        subids = list(ub.unique(self._ids))
        newself = self.__class__(subids, self._dset)
        return newself

    @property
    def ids(self):
        return self._ids

    @property
    def objs(self):
        """
        Get the underlying object dictionary for each object.

        Returns:
            List[ObjT]: all object dictionaries
        """
        return list(ub.take(self._id_to_obj, self._ids))

    def take(self, idxs):
        """
        Take a subset by index

        Returns:
            ObjectList1D

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo().annots()
            >>> assert len(self.take([0, 2, 3])) == 3
        """
        subids = list(ub.take(self._ids, idxs))
        newself = self.__class__(subids, self._dset)
        return newself

    def compress(self, flags):
        """
        Take a subset by flags

        Returns:
            ObjectList1D

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo().images()
            >>> assert len(self.compress([True, False, True])) == 2
        """
        subids = list(ub.compress(self._ids, flags))
        newself = self.__class__(subids, self._dset)
        return newself

    def peek(self):
        """
        Return the first object dictionary

        Returns:
            ObjT: object dictionary

        Example:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo()
            >>> self = dset.images()
            >>> assert self.peek()['id'] == 1
            >>> # Check that subsets return correct items
            >>> sub0 = self.compress([i % 2 == 0 for i in range(len(self))])
            >>> sub1 = self.compress([i % 2 == 1 for i in range(len(self))])
            >>> assert sub0.peek()['id'] == 1
            >>> assert sub1.peek()['id'] == 2
        """
        id_ = self._ids[0]
        obj = self._id_to_obj[id_]
        return obj

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
            return self.get(key, default=default, keepid=keepid)

    def get(self, key, default=ub.NoParam, keepid=False):
        """
        Lookup a list of object attributes

        Args:
            key (str): name of the property you want to lookup

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
            >>> self.get('id')
            >>> self.get(key='foo', default=None, keepid=True)

        Example:
            >>> # xdoctest: +REQUIRES(module:sqlalchemy)
            >>> import kwcoco
            >>> dct_dset = kwcoco.CocoDataset.demo('vidshapes8', rng=303232)
            >>> dct_dset.anns[3]['blorgo'] = 3
            >>> dct_dset.annots().lookup('blorgo', default=None)
            >>> for a in dct_dset.anns.values():
            ...     a['wizard'] = '10!'
            >>> dset = dct_dset.view_sql(force_rewrite=1)
            >>> assert dset.anns[3]['blorgo'] == 3
            >>> assert dset.anns[3]['wizard'] == '10!'
            >>> assert 'blorgo' not in dset.anns[2]
            >>> dset.annots().lookup('blorgo', default=None)
            >>> dset.annots().lookup('wizard', default=None)
            >>> import pytest
            >>> with pytest.raises(KeyError):
            >>>     dset.annots().lookup('blorgo')
            >>> dset.annots().lookup('wizard')
            >>> #self = dset.annots()
        """
        if hasattr(self._dset, '_column_lookup') and default is ub.NoParam:
            # Special case for SQL speed. This only works on schema columns.
            try:
                # TODO: check if the column is in the stuctured schema
                return self._dset._column_lookup(
                    tablename=self._key, key=key, rowids=self._ids)
            except KeyError:
                # We can read only the unstructured bit, which is the best we
                # can do in this case.
                _lutv = self._dset._column_lookup(
                    tablename=self._key, key='_unstructured', rowids=self._ids)
                _lut = dict(zip(self._ids, _lutv))
            # TODO: optimize the case where default is given
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

    def _iter_get(self, key, default=ub.NoParam):
        """
        Iterator version of get, not in stable API yet.
        """
        # TODO: sql variant
        _lut = self._id_to_obj
        if default is ub.NoParam:
            attr_iter = (_lut[_id][key] for _id in self._ids)
        else:
            attr_iter = (_lut[_id].get(key, default) for _id in self._ids)
        return attr_iter

    def set(self, key, values):
        """
        Assign a value to each annotation

        Args:
            key (str): the annotation property to modify
            values (Iterable | Any): an iterable of values to set for each
                annot in the dataset. If the item is not iterable, it is
                assigned to all objects.

        Example:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo()
            >>> self = dset.annots()
            >>> self.set('my-key1', 'my-scalar-value')
            >>> self.set('my-key2', np.random.rand(len(self)))
            >>> print('dset.imgs = {}'.format(ub.urepr(dset.imgs, nl=1)))
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
        Example:
            >>> # xdoctest: +REQUIRES(--benchmark)
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('shapes256')
            >>> self = annots = dset.annots()
            >>> #
            >>> import timerit
            >>> ti = timerit.Timerit(100, bestof=10, verbose=2)
            >>> #
            >>> for timer in ti.reset('lookup'):
            >>>     with timer:
            >>>         self.lookup('image_id')
            >>> #
            >>> for timer in ti.reset('_lookup'):
            >>>     with timer:
            >>>         self._lookup('image_id')
            >>> #
            >>> for timer in ti.reset('image_id'):
            >>>     with timer:
            >>>         self.image_id
            >>> #
            >>> for timer in ti.reset('raw1'):
            >>>     with timer:
            >>>         key = 'image_id'
            >>>         [self._dset.anns[_id][key] for _id in self._ids]
            >>> #
            >>> for timer in ti.reset('raw2'):
            >>>     with timer:
            >>>         anns = self._dset.anns
            >>>         key = 'image_id'
            >>>         [anns[_id][key] for _id in self._ids]
            >>> #
            >>> for timer in ti.reset('lut-gen'):
            >>>     with timer:
            >>>         _lut = self._obj_lut
            >>>         objs = (_lut[_id] for _id in self._ids)
            >>>         [obj[key] for obj in objs]
            >>> #
            >>> for timer in ti.reset('lut-gen-single'):
            >>>     with timer:
            >>>         _lut = self._obj_lut
            >>>         [_lut[_id][key] for _id in self._ids]
        """
        return self.lookup(key, default=default)

    def attribute_frequency(self):
        """
        Compute the number of times each key is used in a dictionary

        Returns:
            Dict[str, int]

        Example:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo()
            >>> self = dset.annots()
            >>> attrs = self.attribute_frequency()
            >>> print('attrs = {}'.format(ub.urepr(attrs, nl=1)))
        """
        attrs = ub.ddict(lambda: 0)
        for obj in self._id_to_obj.values():
            for key, value in obj.items():
                attrs[key] += 1
        return attrs


class ObjectGroups(ub.NiceRepr):
    """
    An object for holding a groups of :class:`ObjectList1D` objects
    """
    def __init__(self, groups, dset):
        self._groups = groups

    def _lookup(self, key):
        return self._lookup(key)  # broken?

    def __getitem__(self, index):
        if isinstance(index, slice):
            subgroups = self._groups[index]
            newself = self.__class__(subgroups, self._dset)
            return newself
        else:
            return self._groups[index]

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


class Categories(ObjectList1D):
    """
    Vectorized access to category attributes

    SeeAlso:
        :func:`kwcoco.coco_dataset.MixinCocoObjects.categories`

    Example:
        >>> from kwcoco.coco_objects1d import Categories  # NOQA
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo()
        >>> ids = list(dset.cats.keys())
        >>> self = Categories(ids, dset)
        >>> print('self.name = {!r}'.format(self.name))
        >>> print('self.supercategory = {!r}'.format(self.supercategory))
    """
    def __init__(self, ids, dset):
        super().__init__(ids, dset, 'categories')

    @property
    def cids(self):
        return self.lookup('id')

    @property
    def name(self):
        return self.lookup('name')

    @property
    def supercategory(self):
        return self.lookup('supercategory', None)


class Videos(ObjectList1D):
    """
    Vectorized access to video attributes

    SeeAlso:
        :func:`kwcoco.coco_dataset.MixinCocoObjects.videos`

    Example:
        >>> from kwcoco.coco_objects1d import Videos  # NOQA
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('vidshapes5')
        >>> ids = list(dset.index.videos.keys())
        >>> self = Videos(ids, dset)
        >>> print('self = {!r}'.format(self))
        self = <Videos(num=5) at ...>
    """
    def __init__(self, ids, dset):
        super().__init__(ids, dset, 'videos')

    @property
    def images(self):
        """
        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('vidshapes8').videos()
            >>> print(self.images)
            <ImageGroups(n=8, m=2.0, s=0.0)>
        """
        return ImageGroups(
            [self._dset.images(video_id=vidid) for vidid in self._ids],
            self._dset)


class Images(ObjectList1D):
    """
    Vectorized access to image attributes

    Example:
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('photos')
        >>> images = dset.images()
        >>> print('images = {}'.format(images))
        images = <Images(num=3)...>
        >>> print('images.gname = {}'.format(images.gname))
        images.gname = ['astro.png', 'carl.jpg', 'stars.png']

    SeeAlso:
        :func:`kwcoco.coco_dataset.MixinCocoObjects.images`
    """

    def __init__(self, ids, dset):
        super().__init__(ids, dset, 'images')

    @property
    def coco_images(self):
        return [self._dset.coco_image(gid) for gid in self]

    @property
    def gids(self):
        return self._ids

    @property
    def gname(self):
        return self.lookup('file_name')

    @property
    def gpath(self):
        root = self._dset.bundle_dpath
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
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo().images()
            >>> self._dset._ensure_imgsize()
            ...
            >>> print(self.size)
            [(512, 512), (328, 448), (256, 256)]
        """
        return list(zip(self.lookup('width'), self.lookup('height')))

    @property
    def area(self):
        """
        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo().images()
            >>> self._dset._ensure_imgsize()
            ...
            >>> print(self.area)
            [262144, 146944, 65536]
        """
        return [w * h for w, h in zip(self.lookup('width'), self.lookup('height'))]

    @property
    def n_annots(self):
        """
        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo().images()
            >>> print(ub.urepr(self.n_annots, nl=0))
            [9, 2, 0]
        """
        return list(map(len, ub.take(self._dset.index.gid_to_aids, self._ids)))

    @property
    def aids(self):
        """
        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo().images()
            >>> print(ub.urepr(list(map(list, self.aids)), nl=0))
            [[1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11], []]
        """
        return list(ub.take(self._dset.index.gid_to_aids, self._ids))

    @property
    def annots(self):
        """
        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo().images()
            >>> print(self.annots)
            <AnnotGroups(n=3, m=3.7, s=3.9)>
        """
        return AnnotGroups([self._dset.annots(aids) for aids in self.aids],
                           self._dset)


class Annots(ObjectList1D):
    """
    Vectorized access to annotation attributes

    SeeAlso:
        :func:`kwcoco.coco_dataset.MixinCocoObjects.annots`

    Example:
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('photos')
        >>> annots = dset.annots()
        >>> print('annots = {}'.format(annots))
        annots = <Annots(num=11)>
        >>> image_ids = annots.lookup('image_id')
        >>> print('image_ids = {}'.format(image_ids))
        image_ids = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]
    """

    def __init__(self, ids, dset):
        super().__init__(ids, dset, 'annotations')

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
        # TODO: deprecate cnames and use category_names instead
        return [cat['name'] for cat in ub.take(self._dset.cats, self.cids)]

    @cnames.setter
    def cnames(self, cnames):
        """
        Args:
            cnames (List[str]):

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo().annots([1, 2, 11])
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
    def category_names(self):
        """
        Get the column of category names

        Returns:
            List[int]
        """
        return self.cnames

    @category_names.setter
    def category_names(self, names):
        """
        Get the column of category names

        Returns:
            List[int]
        """
        self.cnames = names

    @property
    def detections(self):
        """
        Get the kwimage-style detection objects

        Returns:
            kwimage.Detections

        Example:
            >>> # xdoctest: +REQUIRES(module:kwimage)
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('shapes32').annots([1, 2, 11])
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

        Returns:
            kwimage.Boxes

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo().annots([1, 2, 11])
            >>> print(self.boxes)
            <Boxes(xywh,
                array([[ 10,  10, 360, 490],
                       [350,   5, 130, 290],
                       [156, 130,  45,  18]]))>
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
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo().annots([1, 2, 11])
            >>> print('self.boxes = {!r}'.format(self.boxes))
            >>> boxes = kwimage.Boxes.random(3).scale(512).astype(int)
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

        DEPRECATED.

        Returns:
            List[List[int]]: raw boxes in xywh format

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo().annots([1, 2, 11])
            >>> print(self.xywh)
        """
        ub.schedule_deprecation(
            'kwcoco', name='Annots.xywh', type='property',
            deprecate='0.4.0', error='1.0.0', remove='1.1.0',
            migration=(
                'use `Annots.lookup("bbox")`.'
            )
        )

        xywh = self.lookup('bbox')
        return xywh


class AnnotGroups(ObjectGroups):
    """
    Annotation groups are vectorized lists of lists.

    Each item represents a set of annotations that corresopnds with something
    (i.e.  belongs to a particular image).

    Example:
        >>> from kwcoco.coco_objects1d import ImageGroups
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('photos')
        >>> images = dset.images()
        >>> # Requesting the "annots" property from a Images object
        >>> # will return an AnnotGroups object
        >>> group: AnnotGroups = images.annots
        >>> # Printing the group gives info on the mean/std of the number
        >>> # of items per group.
        >>> print(group)
        <AnnotGroups(n=3, m=3.7, s=3.9)...>
        >>> # Groups are fairly restrictive, they dont provide property level
        >>> # access in many cases, but the lookup method is available
        >>> print(group.lookup('id'))
        [[1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11], []]
        >>> print(group.lookup('image_id'))
        [[1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2], []]
        >>> print(group.lookup('category_id'))
        [[1, 2, 3, 4, 5, 5, 5, 5, 5], [6, 4], []]
    """
    @property
    def cids(self):
        """
        Get the grouped category ids for annotations in this group

        Returns:
            List[List[int]]:

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('photos').images().annots
            >>> print('self.cids = {}'.format(ub.urepr(self.cids, nl=0)))
            self.cids = [[1, 2, 3, 4, 5, 5, 5, 5, 5], [6, 4], []]
        """
        return self.lookup('category_id')

    @property
    def cnames(self):
        """
        Get the grouped category names for annotations in this group

        Returns:
            List[List[str]]:

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('photos').images().annots
            >>> print('self.cnames = {}'.format(ub.urepr(self.cnames, nl=0)))
            self.cnames = [['astronaut', 'rocket', 'helmet', 'mouth', 'star', 'star', 'star', 'star', 'star'], ['astronomer', 'mouth'], []]
        """
        return [getattr(group, 'cnames') for group in self._groups]


class ImageGroups(ObjectGroups):
    """
    Image groups are vectorized lists of other Image objects.

    Each item represents a set of images that corresopnds with something (i.e.
    belongs to a particular video).

    Example:
        >>> from kwcoco.coco_objects1d import ImageGroups
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8')
        >>> videos = dset.videos()
        >>> # Requesting the "images" property from a Videos object
        >>> # will return an ImageGroups object
        >>> group: ImageGroups = videos.images
        >>> # Printing the group gives info on the mean/std of the number
        >>> # of items per group.
        >>> print(group)
        <ImageGroups(n=8, m=2.0, s=0.0)...>
        >>> # Groups are fairly restrictive, they dont provide property level
        >>> # access in many cases, but the lookup method is available
        >>> print(group.lookup('id'))
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        >>> print(group.lookup('video_id'))
        [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8]]
        >>> print(group.lookup('frame_index'))
        [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    """
    ...
