
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


def test_keypoint_formats():
    """
    Test different combinations of what data might be / not be specified
    when adding keypoints to a kwcoco file.
    """
    import kwimage
    import kwcoco
    import ubelt as ub

    keypoint_classes = kwcoco.CategoryTree.coerce(['head', 'tail', 'ears', 'nose'])
    object_classes_with_keypoints = kwcoco.CategoryTree.from_coco([{
        'name': 'object',
        'keypoints': list(keypoint_classes),
        'skeleton': [[0, 3], [0, 2], [0, 1]],
    }])
    object_classes_without_keypoints = kwcoco.CategoryTree.from_coco([{
        'name': 'object'
    }])
    points = kwimage.Points.from_coco({
        'x': [4, 30, 770, 5148],
        'y': [0, 1, 5, 2],
        'keypoint_category_id': [2, 1, 3, 0],
    }, classes=keypoint_classes)

    # kwimage_points_variants = {
    #     'with_keypoint_catid': points,
    # }

    object_class_variants = {
        'with_keypoints': object_classes_with_keypoints,
        'without_keypoints': object_classes_without_keypoints,
    }

    keypoint_class_variants = {
        'keypoint_classes': keypoint_classes,
        'null': None,
    }

    keypoints_variants = {}
    keypoints_variants['kwimage-with-class-meta'] = points
    keypoints_variants['orig'] = points.to_coco('orig')
    keypoints_variants['new-name'] = points.to_coco('new-name')
    keypoints_variants['new-id'] = points.to_coco('new-id')
    keypoints_variants['new-v2'] = points.to_coco('new-v2')

    basis = {
        'object_class_variant': object_class_variants,
        'keypoint_class_variant': keypoint_class_variants,
        'keypoints_variant': keypoints_variants,
    }
    grid_items = list(ub.named_product(basis))
    DEBUG = 1
    errors = []
    results = []
    for variants in grid_items:
        row = {
            **variants
        }
        object_classes = object_class_variants[variants['object_class_variant']]
        keypoint_classes = keypoint_class_variants[variants['keypoint_class_variant']]
        keypoints = keypoints_variants[variants['keypoints_variant']]
        if DEBUG:
            print('--- Start ---')
            print(f'row = {ub.urepr(row, nl=1)}')
        failpoint = 'early'
        try:
            dset = kwcoco.CocoDataset()
            if object_classes is not None:
                for cat in object_classes.to_coco():
                    catid = dset.ensure_category(**cat)
            if keypoint_classes is not None:
                dset.dataset['keypoint_categories'] = list(keypoint_classes.to_coco())
                dset._build_index()
            image_id = dset.add_image(file_name='dummy.png')
            failpoint = 'CocoDataset.add_annotation'
            dset.add_annotation(image_id=image_id, bbox=[0, 0, 100, 100],
                                keypoints=keypoints, category_id=catid)
            # Look at how the input was translated to coco
            # added_as = dset.annots().objs[0]['keypoints']
            failpoint = 'annots.detections'
            dets = dset.annots().detections

            failpoint = 'recon'
            recon_v2 = dets.data['keypoints'][0].to_coco('new-v2')
        except Exception as ex:
            import traceback
            row['status'] = 'error'
            row['failpoint'] = failpoint
            row['error'] = repr(ex)
            row['traceback'] = traceback.format_exc()
            print(row['traceback'])
            print(row['error'])
            errors.append(row)
        else:
            row['status'] = 'success'
            # row['added_as'] = added_as
            row['recon_v2'] = recon_v2
        results.append(row)
        if DEBUG:
            print('--- End ---')

    allowfail_variants = {
        ('without_keypoints', 'null', 'kwimage-with-class-meta'),
        ('without_keypoints', 'null', 'new-name'),
        ('without_keypoints', 'null', 'new-id'),
        ('without_keypoints', 'null', 'new-v2'),
    }
    failed_varaints = {tuple((ub.udict(e) & basis).values()) for e in errors}
    unexpected_failures = failed_varaints - allowfail_variants
    if unexpected_failures:
        raise AssertionError('Had unexpected failures of keypoint handling')

    # Peek at the status for each variant, we likely need to add more checks to
    # ensure everything is working properly
    print(f'errors = {ub.urepr(errors, nl=1, sv=1)}')
    if 1:
        import pandas as pd
        import rich
        df = pd.DataFrame(results)
        df = df.drop(['traceback'], axis=1)
        rich.print(df.to_string())
