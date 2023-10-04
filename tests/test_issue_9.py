def test_annot_groups_use_lists():
    """
    References:
        https://gitlab.kitware.com/computer-vision/kwcoco/-/issues/9#note_1425119
    """
    import kwcoco
    dset = kwcoco.CocoDataset.demo()
    images = dset.images()
    annot_groups = images.annots
    assert isinstance(annot_groups[0]._ids, list)
    assert isinstance(dset.annots(image_id=1)._ids, list)
