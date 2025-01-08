"""
Helpers for labelme files
"""
import ubelt as ub
import os


def labelme_to_coco_structure(labelme_data, special_options=True):
    """
    Helper to convert labelme data into dictionaries suitable for adding to a
    CocoDataset.

    Args:
        labelme_data (dict): data read from a labelme json file.

    Example:
        >>> from kwcoco.formats.labelme import *  # NOQA
        >>> labelme_data = {
        >>>     'flags': {},
        >>>     'imageData': None,
        >>>     'imageHeight': 4032,
        >>>     'imagePath': 'filename.jpg',
        >>>     'imageWidth': 3024,
        >>>     'shapes': [
        >>>         {
        >>>             'description': '',
        >>>             'flags': {},
        >>>             'group_id': None,
        >>>             'label': 'category1',
        >>>             'points': [[1527.0, 2319.5], [1512.0, 2317.5], [1503.5, 2295.0], [1568.5, 2243.0], [1561.5, 2278.0], [1548.5, 2307.0], [1541.0, 2315.5]],
        >>>             'shape_type': 'polygon',
        >>>         },
        >>>         {
        >>>             'description': '',
        >>>             'flags': {},
        >>>             'group_id': None,
        >>>             'label': 'category1',
        >>>             'points': [[1346.0, 2285.5], [1318.0, 2282.5], [1370.5, 2241.0], [1360.5, 2258.0], [1357.5, 2272.0], [1354.5, 2278.0]],
        >>>             'shape_type': 'polygon',
        >>>         },
        >>>         {
        >>>             'description': 'image level description',
        >>>             'flags': {},
        >>>             'group_id': None,
        >>>             'label': '__metadata__',
        >>>             'points': [[1346.0, 2285.5]],
        >>>             'shape_type': 'point',
        >>>         },
        >>>     ],
        >>>     'version': '5.3.1',
        >>> }
        >>> img, anns = labelme_to_coco_structure(labelme_data)
        >>> print(f'img = {ub.urepr(img, nl=1)}')
        >>> print(f'anns = {ub.urepr(anns, nl=2)}')
    """
    import kwimage
    import numpy as np
    img = {
        'file_name': labelme_data['imagePath'],
        'width': labelme_data['imageWidth'],
        'height': labelme_data['imageHeight'],
    }
    anns = []
    for shape in labelme_data['shapes']:
        points = shape['points']

        if 0 and shape['group_id'] is not None:
            print(f'unhandled shape groupid = {ub.urepr(shape, nl=1)}')
            # raise NotImplementedError(f'groupid: {shape}')

        desc = shape.get('description', None)
        if desc is not None and not desc.strip():
            desc = None
        # else:
        #     tags = desc.split(';')
        #     # raise NotImplementedError(f'desc: {shape}')

        shape_type = shape['shape_type']
        flags = shape['flags']
        if flags:
            raise NotImplementedError('flags')

        category_name = shape['label']
        if shape_type == 'polygon':
            poly = kwimage.Polygon.coerce(np.array(points))
            ann = {
                'category_name': category_name,
                'bbox': poly.box().quantize().to_coco(),
                'segmentation': poly.to_coco(style='new'),
            }
            if desc:
                ann['description'] = desc
            anns.append(ann)
        elif shape_type == 'point':
            if special_options:
                # Handle points in a special case.
                if category_name == '__metadata__':
                    if desc is not None:
                        if img.get('description'):
                            raise AssertionError('Multiple metadata points in an image')
                        img['description'] = desc
                else:
                    raise NotImplementedError(shape_type)
            else:
                raise NotImplementedError(shape_type)
        else:
            raise NotImplementedError(shape_type)

    return img, anns


class LabelMeFile(ub.NiceRepr):
    """
    Helper class to manage and create a LabelMe JSON file.

    SeeAlso:
        ~/code/labelme/labelme/label_file.py
        ~/code/labelme/labelme/shape.py

    Example:
        >>> # xdoctest: +REQUIRES(module:kwutil)
        >>> from kwcoco.formats.labelme import LabelMeFile
        >>> self = LabelMeFile.demo()
        >>> print(self.dumps())
    """
    # Keys for top level data
    __datakeys__ = [
        "version",  # LabelMe version
        "imageData",  # Can be populated with base64 image data
        "imagePath",
        "shapes",  # polygonal annotations
        "flags",  # image level flags
        "imageHeight",
        "imageWidth",
    ]
    # Keys for data['shapes']

    # TODO: special class for shapes?
    __shapekeys__ = [
        "label",
        "points",
        "group_id",
        "shape_type",
        "flags",
        "description",
        "mask",
    ]

    def __init__(self, data, fpath=None):
        """
        Initialize the LabelMe file structure.

        See :func:`LabelMeFile.new` to create an empty file to populate

        Args:
            data (Dict):
                the labelme dictionary

            fpath (str | PathLike | None):
                The parent of this path determines where relative paths are
                resolved from. This is also where the data will be written on a
                dump.
        """
        self.data = data
        self.fpath = fpath

    def __nice__(self):
        parts = []
        parts.append(str(self.fpath))
        # parts.append(str(self.data.get('imagePath')))
        nshapes = len(self.data.get('shapes', []))
        parts.append(f'nshapes={nshapes}')
        return ', '.join(parts)

    @classmethod
    def demo(cls):
        """
        Create an instance of this class for demos and tests

        Returns:
            LabeMeFile
        """
        labelme_data = {
            'version': '5.3.1',
            'flags': {},
            'imagePath': 'filename.jpg',
            'imageHeight': 4032,
            'imageWidth': 3024,
            'imageData': None,
            'shapes': [
                {
                    'description': '',
                    'flags': {},
                    'group_id': None,
                    'label': 'category1',
                    'points': [[1527.0, 2319.5], [1512.0, 2317.5], [1503.5, 2295.0], [1568.5, 2243.0], [1561.5, 2278.0], [1548.5, 2307.0], [1541.0, 2315.5]],
                    'shape_type': 'polygon',
                },
                {
                    'description': '',
                    'flags': {},
                    'group_id': None,
                    'label': 'category1',
                    'points': [[1346.0, 2285.5], [1318.0, 2282.5], [1370.5, 2241.0], [1360.5, 2258.0], [1357.5, 2272.0], [1354.5, 2278.0]],
                    'shape_type': 'polygon',
                },
                {
                    'description': 'image level description',
                    'flags': {},
                    'group_id': None,
                    'label': '__metadata__',
                    'points': [[1346.0, 2285.5]],
                    'shape_type': 'point',
                },
            ],
        }
        fpath = 'filename.json'
        self = cls(labelme_data, fpath)
        return self

    @classmethod
    def empty(cls, image_path=None, image_height=None, image_width=None, fpath=None):
        """
        Create a new empty file for a specific image.

        Returns:
            LabeMeFile

        Example:
            >>> from kwcoco.formats.labelme import *  # NOQA
            >>> self = LabelMeFile.empty('foo.png')
            >>> print(f'self={self}')
        """
        if image_path is not None:
            image_path = os.fspath(image_path)

        if fpath is None and image_path is not None:
            fpath = ub.Path(image_path).augment(ext='.json')

        data = {
            "version": "5.0.1",  # LabelMe version
            "flags": {},  # Additional flags
            "imagePath": image_path,
            "imageHeight": image_height,
            "imageWidth": image_width,
            "imageData": None,  # Can be populated with base64 image data
            "shapes": [],  # List of shapes
        }
        self = cls(data, fpath)
        return self

    @classmethod
    def load(cls, file):
        """
        Load a file from a path.

        Returns:
            LabeMeFile
        """
        import kwutil
        try:
            fpath = os.fspath(file)
        except TypeError:
            input_was_pathlike = False
        else:
            input_was_pathlike = True
        if input_was_pathlike:
            data = kwutil.Json.load(file)
            self = cls(data, fpath)
        else:
            data = kwutil.Json.load(file)
            self = cls(data)
        return self

    @classmethod
    def multiple_from_coco(cls, coco_dset):
        for image_id in coco_dset.images():
            self = cls.from_coco(coco_dset, image_id)
            yield self

    def reroot(self, absolute=True):
        assert not absolute
        old_fpath = self.data['imagePath']
        parent_dpath = ub.Path(self.fpath).parent
        abs_fpath = parent_dpath / old_fpath
        rel_fpath = ub.Path(abs_fpath).relative_to(parent_dpath)
        self.data['imagePath'] = os.fspath(rel_fpath)

    @classmethod
    def from_coco(cls, coco_dset, image_id=None):
        """
        Convert an image in a CocoDataset into a LabeMeFile.

        Args:
            coco_dset (CocoDataset): dataset to convert to labelme

            image_id (int | None):
                The image in the cocodataset to convert to a labelme file.  if
                unspecified, the dataset must have one image in it, otherwise
                    we raise an error.

        Returns:
            LabeMeFile

        Example:
            >>> import kwcoco
            >>> from kwcoco.formats.labelme import LabelMeFile
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes8')
            >>> image_id = sorted(coco_dset.images())[0]
            >>> self = LabelMeFile.from_coco(coco_dset, image_id)
            >>> coco_recon = self.to_coco()
            >>> recon = LabelMeFile.from_coco(coco_recon)
            >>> assert self.data == recon.data
        """
        if image_id is None:
            images = coco_dset.images()
            if len(images) == 0:
                raise ValueError('CocoDataset has no images!')
            elif len(images) > 1:
                raise ValueError(ub.paragraph(
                    '''
                    CocoDataset has more than one image, Choose which image to
                    convert by specifing image_id.
                    '''))
            image_id = list(images)[0]

        img = coco_dset.index.imgs[image_id]

        self = cls.empty(image_path=img['file_name'])
        self.data['imageHeight'] = img.get('height', None)
        self.data['imageWidth'] = img.get('width', None)

        curr_groupid = 1

        # Collect annotations for this image
        for ann in coco_dset.annots(image_id=image_id).objs:
            category_id = ann['category_id']
            cat = coco_dset.index.cats[category_id]
            catname = cat.get('name', 'unknown')
            segmentation = ann.get('segmentation', None)
            bbox = ann.get('bbox', [])
            extra = ub.udict(ann) - {'category_id', 'segmentation', 'bbox', 'image_id'}

            if segmentation:
                import kwimage
                sseg = kwimage.Segmentation.coerce(segmentation)
                mpoly = sseg.to_multi_polygon()
                group_id = None
                if len(mpoly) > 1:
                    group_id = curr_groupid
                    curr_groupid += 1

                for poly in mpoly.data:
                    # TODO: error checks for multipolygons / holes
                    # and other stuff labelme cant handle.
                    # add fallback options.
                    points = poly.exterior.data.tolist()
                    self.add_polygon(catname, points, group_id=group_id, extra=extra)

            elif bbox:
                # Add bounding box as a rectangle
                self.add_rectangle(catname, bbox, extra=extra)

            else:
                raise NotImplementedError(f'Unable to convert {ann} to a labelme object')

        return self

    def add_to_coco(self, coco_dset):
        """
        Add the information in this labelme file to an existing coco file.

        Args:
            coco_dset (CocoDataset): the dataset to add to

        SeeAlso:
            LabelMeFile.to_coco
        """
        img, anns = labelme_to_coco_structure(self.data)
        image_id = coco_dset.add_image(**img)

        for ann in anns:
            catname = ann.pop('category_name')
            cid = coco_dset.ensure_category(catname)
            ann['image_id'] = image_id
            ann['category_id'] = cid
            coco_dset.add_annotation(**ann)

    def to_coco(self):
        """
        Convert this labelme file into a standalone coco dataset.

        Returns:
            CocoDataset

        SeeAlso:
            LabelMeFile.add_to_coco

        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.formats.labelme import LabelMeFile
            >>> self = LabelMeFile.demo()
            >>> coco_dset = self.to_coco()
            >>> print(f'coco_dset.dataset = {ub.urepr(coco_dset.dataset, nl=2)}')
            >>> recon = LabelMeFile.from_coco(coco_dset)
            >>> # FIXME: recon is not perfect
            >>> print(f'self={self}')
            >>> print(f'recon={recon}')
            >>> print(self.dumps())
            >>> print(recon.dumps())
        """
        import kwcoco
        coco_dset = kwcoco.CocoDataset.empty()
        self.add_to_coco(coco_dset)
        return coco_dset

    def add_polygon(self, label, points, group_id=None, flags=None, **kwargs):
        """
        Add a polygon shape.

        Args:
            label (str): Category namae / label for the shape.
            points (list of list of float): List of (x, y) points defining the polygon.
            group_id (int, optional): Group ID for the shape.
            flags (dict, optional): Additional flags for the shape.
        """
        shape = {
            "label": label,
            "points": points,
            "group_id": group_id,
            "shape_type": "polygon",
            "flags": flags or {},
            **kwargs
        }
        self.data["shapes"].append(shape)

    def add_rectangle(self, label, bbox, group_id=None, flags=None, **kwargs):
        """
        Add a rectangle shape.

        Args:
            label (str): Label for the shape.
            bbox (list of float): Bounding box [x, y, width, height].
            group_id (int, optional): Group ID for the shape.
            flags (dict, optional): Additional flags for the shape.
        """
        x, y, w, h = bbox
        shape = {
            "label": label,
            "points": [[x, y], [x + w, y + h]],
            "group_id": group_id,
            "shape_type": "rectangle",
            "flags": flags or {},
            **kwargs,
        }
        self.data["shapes"].append(shape)

    def dump(self, file=None):
        """
        Save the LabelMe JSON data to a file.

        Args:
            output_path (str): Path where the JSON file will be saved.
        """
        import kwutil
        if file is None:
            file = self.fpath
        try:
            fpath = os.fspath(file)
        except TypeError:
            input_was_pathlike = False
        else:
            input_was_pathlike = True
        if input_was_pathlike:
            with open(fpath, 'w') as file:
                self.dump(file)
        else:
            kwutil.Json.dump(self.data, file)

    def dumps(self):
        """
        Save the LabelMe JSON data to a file.

        Returns:
            str
        """
        import kwutil
        return kwutil.Json.dumps(self.data)
