import ubelt as ub


class CocoImage(ub.NiceRepr):
    """
    An object-oriented representation of a coco image.

    It provides helper methods that are specific to a single image.

    This operates directly on a single coco image dictionary, but it can
    optionally be connected to a parent dataset, which allows it to use
    CocoDataset methods to query about relationships and resolve pointers.

    This is different than the Images class in coco_object1d, which is just a
    vectorized interface to multiple objects.

    Example:
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('shapes8')
        >>> self = dset._coco_image(1)


        >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        >>> self = dset._coco_image(1)
    """

    def __init__(self, img, dset=None):
        self.img = img
        self.dset = dset

    def __nice__(self):
        return 'wh={dsize}'.format(dsize=self.dsize)

    def __getitem__(self, key):
        return self.img[key]

    def get(self, key, default=ub.NoParam):
        """
        Duck type some of the dict interface
        """
        if default is ub.NoParam:
            return self.img.get(key)
        else:
            return self.img.get(key, default)

    @property
    def dsize(self):
        width = self.img.get('width', None)
        height = self.img.get('height', None)
        return width, height

    def _iter_asset_objs(self):
        """
        Iterate through base + auxiliary dicts that have file paths
        """
        img = self.img
        has_base_image = img.get('file_name', None) is not None
        if has_base_image:
            obj = ub.dict_diff(img, {'auxiliary'})
            yield obj
        for obj in img.get('auxiliary', []):
            yield obj

    # @property
    # def shape(self):
    #     width = self.img.get('width', None)
    #     height = self.img.get('height', None)
    #     return width, height
