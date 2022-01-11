def demo_load_msi_data():
    import kwcoco
    import kwimage
    coco_dset = kwcoco.CocoDataset.demo('vidshapes8-msi')
    # Get the "CocoImage" for a gid (imaGe id)
    gid = 3
    coco_img = coco_dset.coco_image(gid)
    # Request specific channels via the pipe-separated kwcoco channel spec
    channels = 'B1|B8a|B8|B11|B10'
    # The delayed image is just a reference, it does not load data
    # until you finish describing all the operations you want to do
    delayed = coco_img.delay(channels=channels)
    delayed = delayed.crop((slice(30, 130), slice(20, 200)))
    delayed = delayed.warp(kwimage.Affine.scale(2), dsize='auto')
    # Calling finalize returns the warped and aligned sub-patch of data
    imdata = delayed.finalize()
    print('imdata.shape = {!r}'.format(imdata.shape))
    # imdata.shape = (200, 360, 5)

    #
    # Note: It is also possible to get similar behavior with ndsampler
    # which has the benefit that it also returns any underlying annotations
    # in image-space or video-space (depending on what the user requests)
    import ndsampler
    import numpy as np
    import ubelt as ub
    sampler = ndsampler.CocoSampler(coco_dset)
    tr = {
        'gid': gid,
        'channels': kwcoco.FusedChannelSpec.coerce(channels),
        'space_slice': (slice(-10, 512), slice(20, 512)),
    }
    sample = sampler.load_sample(tr)
    imdata = sample['im']
    annots = sample['annots']['frame_dets'][0]

    # hack to workaround minor bug in kwimage 0.8.0
    annots.data['class_idxs'] = np.array(list(ub.take(annots.classes.id_to_idx, annots.data['cids'])))

    # Do PCA to make a false-color image
    import sklearn
    import sklearn.decomposition
    pca = sklearn.decomposition.PCA(n_components=3)
    projected = pca.fit_transform(imdata.reshape(-1, 5)).reshape(imdata.shape[0:2] + (3,))
    false_color = kwimage.normalize_intensity(projected)

    canvas = false_color.copy()

    # Draw annotations on the canvas
    canvas = annots.draw_on(canvas)

    import kwplot
    kwplot.autompl()
    kwplot.imshow(canvas)
