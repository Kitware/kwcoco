"""
Tests for small components of the schema
"""


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
        key = ub.urepr(kw, compact=1)
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
