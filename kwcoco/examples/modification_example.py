
def dataset_modification_example_via_copy():
    """
    Say you are given a dataset as input and you need to add your own
    annotation "predictions" to it. You could copy the existing dataset,
    remove all the annotations, and then add your new annotations.
    """

    import kwcoco
    dset = kwcoco.CocoDataset.demo()

    # Do a deep copy of the dataset
    out_dset = dset.copy()

    # Remove all annotations
    out_dset.remove_annotations(list(dset.index.anns.keys()))

    # Add your custom annotations (make sure they are in IMAGE pixel coords)
    import kwimage
    poly = kwimage.Polygon.random().scale((10, 20)).translate((0, 2))
    gid = 1
    my_new_ann = {
        'image_id': gid,
        'bbox': [0, 2, 10, 20],
        'score': 0.8,
        'category_id': dset.index.name_to_cat['astronaut']['id'],
        'segmentation': poly.to_coco(),
    }

    out_dset.add_annotation(**my_new_ann)


def dataset_modification_example_via_construction():
    """
    Alternatively you can make a new dataset and copy over categories / images
    as needed
    """
    import kwcoco
    import kwimage

    dset = kwcoco.CocoDataset.demo()

    new_dset = kwcoco.CocoDataset()

    for cat in dset.index.cats.values():
        new_dset.add_cateogry(**cat)

    for img in dset.index.imgs.values():
        new_dset.add_image(**img)

    poly = kwimage.Polygon.random().scale((10, 20)).translate((0, 2))
    gid = 1
    my_new_ann = {
        'image_id': gid,
        'bbox': [0, 2, 10, 20],
        'score': 0.8,
        'category_id': dset.index.name_to_cat['astronaut']['id'],
        'segmentation': poly.to_coco(),
    }

    new_dset.add_annotation(**my_new_ann)
