"""
Vectorized ORM-like objects used in conjunction with coco_dataset
"""
from os.path import join
import numpy as np
import ubelt as ub


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
        return self._ids[index]

    @property
    def objs(self):
        """
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
        """
        # if hasattr(self._dset, '_column_lookup'):
        #     # Hack for SQL speed
        #     # Agg, this doesn't work because some columns need json decoding
        #     return self._dset._column_lookup(
        #         tablename=self._key, key=key, rowids=self._ids)
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

    def set(self, key, values):
        """
        Assign a value to each annotation

        Args:
            key (str): the annotation property to modify
            values (Iterable | scalar): an iterable of values to set for each
                annot in the dataset. If the item is not iterable, it is
                assigned to all objects.

        Example:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo()
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
            >>> print('attrs = {}'.format(ub.repr2(attrs, nl=1)))
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
            [self._dset.images(vidid=vidid) for vidid in self._ids],
            self._dset)


class Images(ObjectList1D):
    """
    Vectorized access to image attributes

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
            >>> print(self.size)
            [(512, 512), (300, 250), (256, 256)]
        """
        return list(zip(self.lookup('width'), self.lookup('height')))

    @property
    def area(self):
        """
        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo().images()
            >>> self._dset._ensure_imgsize()
            >>> print(self.area)
            [262144, 75000, 65536]
        """
        return [w * h for w, h in zip(self.lookup('width'), self.lookup('height'))]

    @property
    def n_annots(self):
        """
        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo().images()
            >>> print(ub.repr2(self.n_annots, nl=0))
            [9, 2, 0]
        """
        return list(map(len, ub.take(self._dset.gid_to_aids, self._ids)))

    @property
    def aids(self):
        """
        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo().images()
            >>> print(ub.repr2(list(map(list, self.aids)), nl=0))
            [[1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11], []]
        """
        return list(ub.take(self._dset.gid_to_aids, self._ids))

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

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo().annots([1, 2, 11])
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

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo().annots([1, 2, 11])
            >>> print(self.xywh)
        """
        xywh = self.lookup('bbox')
        return xywh


class AnnotGroups(ObjectGroups):
    @property
    def cids(self):
        return self.lookup('category_id')

    @property
    def cnames(self):
        return [getattr(group, 'cname') for group in self._groups]


class ImageGroups(ObjectGroups):
    pass
