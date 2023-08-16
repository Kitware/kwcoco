import ubelt as ub


def test_resolution_with_channels():
    """
    We had a bug where the shape of the returned delayed image would not change
    if you requested a non-existing channel at a particular resolution. This
    tests if that bug is fixed.
    """
    import kwcoco
    dset = kwcoco.CocoDataset.demo('vidshapes8-msi-multisensor')
    coco_img = dset.coco_image(1)
    coco_img.img['resolution'] = '1 meter'
    # Test with a channel that does / does not exist
    chan = coco_img.channels.fuse().to_list()[0]
    shapes1 = []
    shapes2 = []
    shapes3 = []
    for resolution in [None, 1.1, 2.2]:
        d1 = coco_img.imdelay(chan, space='video', resolution=resolution)
        d2 = coco_img.imdelay('fds', space='video', resolution=resolution)
        d3 = coco_img.imdelay(chan + '|fds', space='video', resolution=resolution)
        shapes1.append(d1.dsize)
        shapes2.append(d2.dsize)
        shapes3.append(d3.dsize)
    assert not ub.allsame(shapes1)
    assert shapes1 == shapes2
    assert shapes1 == shapes3


def test_coco_image_add_asset():
    import kwcoco
    dset = kwcoco.CocoDataset.demo('vidshapes8-msi-multisensor')
    coco_img = dset.images().coco_images[0]

    orig_num_assets = len(coco_img.assets)

    primary_asset = coco_img.primary_asset()
    new_asset1 = primary_asset.copy()
    # Remove the asset id, so a new one is created (which will be important
    # once assets move to their own table). But is not currently necessary in
    # version 0.6.4
    new_asset1.pop('id', None)
    # Use a modified version of the primary asset to make a new asset
    # dictionary.
    new_asset1['channels'] = 'rando-brando1'

    coco_img.add_asset(**new_asset1)
    assert len(coco_img.assets) == (orig_num_assets + 1)

    # Test with a non-schema property
    new_asset2 = primary_asset.copy()
    new_asset2.pop('id', None)
    new_asset2['channels'] = 'rando-brando2'
    new_asset2['image_id'] = coco_img['id']
    new_asset2['non_schema_property'] = 'foobar'
    coco_img.add_asset(**new_asset2)
    assert len(coco_img.assets) == (orig_num_assets + 2)

    # Test with an image-id that disagrees
    new_asset3 = primary_asset.copy()
    new_asset3.pop('id', None)
    new_asset3['channels'] = 'rando-brando3'
    new_asset3['image_id'] = coco_img['id'] + 1000

    import pytest
    with pytest.raises(AssertionError):
        coco_img.add_asset(**new_asset3)

    assert len(coco_img.assets) == (orig_num_assets + 2)


def test_imdelay_with_interpolation():
    import ubelt as ub
    dpath = ub.Path.appdir('kwcoco/tests/imdelay-with-interp').ensuredir()

    import kwimage
    import numpy as np
    imdata = (kwimage.checkerboard() * 255).astype(np.uint8)
    fpath = dpath / 'checkers.png'
    kwimage.imwrite(fpath, imdata)

    import kwcoco
    dset = kwcoco.CocoDataset()
    gid = dset.add_image(file_name=fpath, channels='gray', resolution='10 GSD')
    dset._ensure_imgsize()

    coco_img = dset.coco_image(gid)

    delayed = coco_img.imdelay(interpolation='nearest', antialias=False)
    data = delayed.finalize()
    assert np.all(data == imdata)

    # Test to make sure imdelay respect interplation
    # Subsequent changes are still subject to the user overwriting these defaults.
    scaled = coco_img.imdelay(interpolation='nearest', antialias=False, resolution='33GSD')
    data_scaled = scaled.finalize()
    scaled.write_network_text(fields=True)
    scaled = scaled.optimize()
    scaled.write_network_text(fields=True)
    assert set(imdata.ravel()) == set(data_scaled.ravel())

    assert np.all(data == imdata)

    # TODO: move this test to delayed image and
    # make it such that interpolation defaults to ones that are already used.
    # We shouldn't need to do it like this, but alas...
    scaled = delayed.scale(0.13)
    scaled.write_network_text(fields=True)
    scaled = scaled.optimize()
    scaled.write_network_text(fields=True)
    scaled._set_nested_params(interpolation='nearest', antialias=False)
    scaled.write_network_text(fields=True)

    data_scaled = scaled.finalize()
    data_scaled2 = scaled.finalize(interpolation='nearest', antialias=False)

    assert set(imdata.ravel()) == set(data_scaled2.ravel())
    assert set(imdata.ravel()) == set(data_scaled.ravel())
