def test_resolution_with_channels():
    """
    We had a bug where the shape of the returned delayed image would not change
    if you requested a non-existing channel at a particular resolution. This
    tests if that bug is fixed.
    """
    import kwcoco
    import ubelt as ub
    dset = kwcoco.CocoDataset.demo('vidshapes8-msi-multisensor')
    coco_img = dset.coco_image(1)
    coco_img.img['resolution'] = '1 meter'
    # Test with a channel that does / does not exist
    chan = coco_img.channels.fuse().to_list()[0]
    shapes1 = []
    shapes2 = []
    shapes3 = []
    for resolution in [None, 1.1, 2.2]:
        d1 = coco_img.delay(chan, space='video', resolution=resolution)
        d2 = coco_img.delay('fds', space='video', resolution=resolution)
        d3 = coco_img.delay(chan + '|fds', space='video', resolution=resolution)
        shapes1.append(d1.dsize)
        shapes2.append(d2.dsize)
        shapes3.append(d3.dsize)
    assert not ub.allsame(shapes1)
    assert shapes1 == shapes2
    assert shapes1 == shapes3
