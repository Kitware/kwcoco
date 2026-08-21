"""
Tests for small components of the schema
"""


def test_info_schema_accepts_mscoco_dict_and_kwcoco_list():
    import kwcoco

    base = kwcoco.CocoDataset().dataset
    for info in [
        {'description': 'mscoco metadata', 'custom_field': 1},
        [{'type': 'process', 'properties': {'name': 'demo'}}],
    ]:
        data = dict(base)
        data['info'] = info
        kwcoco.coco_schema.COCO_SCHEMA.validate(data)


def test_polygon_schema():
    from kwcoco import coco_schema
    import jsonschema
    import kwimage
    import ubelt as ub
    import kwarray

    rng = kwarray.ensure_rng(0)

    coco_schema.KWCOCO_POLYGON
    coco_schema.MSCOCO_POLYGON
    basis = {
        # 'n': [3, 4, 6],
        'n': [3, 6],
        # 'n_holes': [0, 1, 2],
        'n_holes': [0, 1],
        # 'convex': [True, False],
        'type': ['float32', 'int32'],
        'scale': [1, 512],
    }
    polygon_cases = {}
    for kw in ub.named_product(basis):
        key = ub.urepr(kw, compact=True)
        polykw = ub.compatible(kw, kwimage.Polygon.random)
        poly01 = kwimage.Polygon.random(**polykw, rng=rng)
        if kw['scale'] == 1 and kw['type'].startswith('int'):
            continue
        poly = poly01.scale(kw['scale'])
        polygon_cases[key] = poly
        poly.exterior.astype(kw['type'], inplace=True)
        for hole in poly.interiors:
            hole.astype(kw['type'], inplace=True)

    for key, poly in polygon_cases.items():
        kwcoco_poly = poly.to_coco(style='new')
        jsonschema.validate(kwcoco_poly, schema=coco_schema.KWCOCO_POLYGON)
        jsonschema.validate(kwcoco_poly, schema=coco_schema.POLYGON)
        if not poly.interiors:
            coco_poly = poly.to_coco(style='orig')
            jsonschema.validate(coco_poly, schema=coco_schema.MSCOCO_POLYGON)
            jsonschema.validate(coco_poly, schema=coco_schema.POLYGON)


def test_jsonschema_element_default_containers_are_not_shared():
    from kwcoco.util.jsonschema_elements import ARRAY, OBJECT

    arr1 = ARRAY()
    arr2 = ARRAY()
    arr1['items']['sentinel'] = True
    assert arr2['items'] == {}

    obj1 = OBJECT()
    obj2 = OBJECT()
    obj1['properties']['sentinel'] = True
    assert obj2['properties'] == {}


def test_coco_url_schema_fields():
    """Image/video coco_url values must be absolute URI-like strings."""
    import jsonschema
    import pytest

    from kwcoco import coco_schema

    valid_coco_urls = [
        'https://example.com/image.png',
        's3://bucket/path/to/image.tif',
        'ipfs://bafybeigdyrzt/example.png',
    ]
    invalid_coco_urls = [
        '',
        'example.com/image.png',
        'relative/path/image.png',
        'https://example.com/has whitespace.png',
    ]

    for value in valid_coco_urls:
        coco_schema.IMAGE.validate(
            {'id': 1, 'file_name': 'image.png', 'coco_url': value}
        )
        coco_schema.VIDEO.validate(
            {'id': 1, 'name': 'video', 'coco_url': value}
        )

    for value in invalid_coco_urls + [None]:
        with pytest.raises(jsonschema.ValidationError):
            coco_schema.IMAGE.validate(
                {'id': 1, 'file_name': 'image.png', 'coco_url': value}
            )
        with pytest.raises(jsonschema.ValidationError):
            coco_schema.VIDEO.validate(
                {'id': 1, 'name': 'video', 'coco_url': value}
            )
