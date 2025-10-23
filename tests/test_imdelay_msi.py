def test_imdelay_msi():
    """
    Test issue caused by delayed image lazy warping is fixed.
    """
    import kwcoco
    dset = kwcoco.CocoDataset.demo('vidshapes5-frames5-randgsize-speed0.2-msi-multisensor')
    for coco_img in dset.images().coco_images:
        final = coco_img.imdelay().finalize()
        h, w = final.shape[0:2]
        assert coco_img.img['width'] == w
        assert coco_img.img['height'] == h
