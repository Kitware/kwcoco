"""
Helpers for labelme files
"""
import ubelt as ub


def labelme_to_coco_structure(labelme_data, special_options=True):
    """
    Helper to convert labelme data into dictionaries suitable for adding to a
    CocoDataset.

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

        if shape['group_id'] is not None:
            print(f'unhandled shape = {ub.urepr(shape, nl=1)}')
            raise NotImplementedError(f'groupid: {shape}')

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
