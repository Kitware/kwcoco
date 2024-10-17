
def test_column_based_keypoints():
    import pytest
    pytest.skip('keypoint support needs an overhaul here and in kwimage')

    import kwcoco
    keypoints = {
        'x': [0, 1, 2, 3],
        'y': [0, 1, 2, 3],
        'keypoint_category_id': [0, 0, 0, 0],
    }
    dset = kwcoco.CocoDataset()
    image_id = dset.add_image(file_name='dummy.png')
    dset.add_annotation(image_id=image_id, bbox=[0, 0, 100, 100], keypoints=keypoints)
    dset.annots().detections.data

    import kwimage
    import kwcoco

    classes = kwcoco.CategoryTree.coerce(['head', 'tail', 'ears', 'nose'])
    points = kwimage.Points.from_coco({
        'x': [0, 1, 2, 3],
        'y': [0, 1, 2, 3],
        'keypoint_category_id': [0, 1, 3, 2],
    }, classes=classes)
    point_variant = {}
    point_variant['kwimage'] = points
    # point_variant['orig'] = points.to_coco('orig')
    point_variant['new'] = points.to_coco('new')
    point_variant['new-v2'] = points.to_coco('new-v2')

    errors = []
    for variant, keypoints in point_variant.items():
        try:
            dset = kwcoco.CocoDataset()
            catid = dset.ensure_category('object')
            dset.dataset['keypoint_categories'] = list(classes.to_coco())
            dset._build_index()
            image_id = dset.add_image(file_name='dummy.png')
            dset.add_annotation(image_id=image_id, bbox=[0, 0, 100, 100],
                                keypoints=keypoints, category_id=catid)
            dset.annots().objs
            dset.annots().detections.data
        except Exception as ex:
            errors.append({
                'variant': variant,
                'message': repr(ex),
            })
            raise
    import ubelt as ub
    print(f'errors = {ub.urepr(errors, nl=1)}')
