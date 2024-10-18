
def test_column_based_keypoints_without_categories():
    import kwcoco
    keypoints = {
        'x': [0, 1, 2, 3],
        'y': [0, 1, 2, 3],
    }
    dset = kwcoco.CocoDataset()
    image_id = dset.add_image(file_name='dummy.png')
    dset.add_annotation(image_id=image_id, bbox=[0, 0, 100, 100], keypoints=keypoints)
    dets = dset.annots().detections
    dets.data


def test_column_based_keypoints_with_categories():
    import kwcoco
    keypoints = {
        'x': [0, 1, 2, 3],
        'y': [0, 1, 2, 3],
        'keypoint_category_id': [2, 1, 3, 0],
    }
    dset = kwcoco.CocoDataset()
    keypoint_classes = kwcoco.CategoryTree.coerce(['head', 'tail', 'ears', 'nose'])
    dset.add_keypoint_categories(list(keypoint_classes.to_coco()))
    image_id = dset.add_image(file_name='dummy.png')
    dset.add_annotation(image_id=image_id, bbox=[0, 0, 100, 100], keypoints=keypoints)
    dets = dset.annots().detections
    recon = dets.data['keypoints'][0].to_coco('new-v2')
    assert keypoints['keypoint_category_id'] == recon['keypoint_category_id']


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
    allowfail_variants = {
        ('without_keypoints', 'null', 'kwimage-with-class-meta'),
        ('without_keypoints', 'null', 'new-name'),
        ('without_keypoints', 'null', 'new-id'),
        ('without_keypoints', 'null', 'new-v2'),
    }
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
            ann = dset.annots().objs[0]
            coco_kpts_v2 = ann['keypoints']
            dets = dset.annots().detections
            failpoint = 'recon'
            points_v2 = dets.data['keypoints'][0]
            recon_v2 = points_v2.to_coco('new-v2')
            if DEBUG:
                print(f'coco_kpts_v2 = {ub.urepr(coco_kpts_v2, nl=1)}')
                print(f'points_v2 = {ub.urepr(points_v2, nl=1)}')
                print(f'recon_v2 = {ub.urepr(recon_v2, nl=1)}')
        except Exception as ex:
            import traceback

            if tuple(variants.values()) in allowfail_variants:
                row['status'] = 'allowed_failure'
            else:
                row['status'] = 'error'
            row['failpoint'] = failpoint
            row['error'] = repr(ex)
            row['traceback'] = traceback.format_exc()
            if DEBUG:
                print(row['traceback'])
                print(row['error'])
            if row['status'] == 'error':
                errors.append(row)
        else:
            row['status'] = 'success'
            # row['added_as'] = added_as
            row['recon_v2'] = recon_v2
        results.append(row)
        if DEBUG:
            print('--- End ---')

    # Peek at the status for each variant, we likely need to add more checks to
    # ensure everything is working properly
    print(f'errors = {ub.urepr(errors, nl=1, sv=1)}')
    failed_varaints = {tuple((ub.udict(e) & basis).values()) for e in errors}
    unexpected_failures = failed_varaints - allowfail_variants
    if 1:
        import pandas as pd
        import rich
        df = pd.DataFrame(results)
        try:
            df = df.drop(['traceback'], axis=1)
        except KeyError:
            ...  # use case for safe drop
        rich.print(df.to_string())
    if unexpected_failures:
        print(f'unexpected_failures={unexpected_failures}')
        raise AssertionError('Had unexpected failures of keypoint handling')
