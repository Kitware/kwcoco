
def draw_true_and_pred_boxes(true_fpath, pred_fpath, gid, viz_fpath):
    """
    How do you generally visualize gt and predicted bounding boxes together?

    Example:
        >>> import kwcoco
        >>> import ubelt as ub
        >>> from os.path import join
        >>> from kwcoco.demo.perterb import perterb_coco
        >>> # Create a working directory
        >>> dpath = ub.Path.appdir('kwcoco/examples/draw_true_and_pred_boxes').ensuredir()
        >>> # Lets setup some dummy true data
        >>> true_dset = kwcoco.CocoDataset.demo('shapes2')
        >>> true_dset.fpath = join(dpath, 'true_dset.kwcoco.json')
        >>> true_dset.dump(true_dset.fpath, newlines=True)
        >>> # Lets setup some dummy predicted data
        >>> pred_dset = perterb_coco(true_dset, box_noise=100, rng=421)
        >>> pred_dset.fpath = join(dpath, 'pred_dset.kwcoco.json')
        >>> pred_dset.dump(pred_dset.fpath, newlines=True)
        >>> #
        >>> # We now have our true and predicted data, lets visualize
        >>> true_fpath = true_dset.fpath
        >>> pred_fpath = pred_dset.fpath
        >>> print('dpath = {!r}'.format(dpath))
        >>> print('true_fpath = {!r}'.format(true_fpath))
        >>> print('pred_fpath = {!r}'.format(pred_fpath))
        >>> # Lets choose an image id to visualize and a path to write to
        >>> gid = 1
        >>> viz_fpath = join(dpath, 'viz_{}.jpg'.format(gid))
        >>> # The answer to the question is in the logic of this function
        >>> draw_true_and_pred_boxes(true_fpath, pred_fpath, gid, viz_fpath)
    """
    import kwimage
    import kwcoco
    true_dset = kwcoco.CocoDataset(true_fpath)
    pred_dset = kwcoco.CocoDataset(pred_fpath)

    if __debug__:
        # I hope your image ids are aligned between datasets
        true_img = true_dset.imgs[gid]
        pred_img = pred_dset.imgs[gid]
        assert true_img['file_name'] == pred_img['file_name']

    # Get the true/pred annotation dictionaries from the chosen image
    true_aids = true_dset.index.gid_to_aids[gid]
    pred_aids = pred_dset.index.gid_to_aids[gid]
    true_anns = [true_dset.index.anns[aid] for aid in true_aids]
    pred_anns = [pred_dset.index.anns[aid] for aid in pred_aids]

    # Create Detections from the coco annotation dictionaries
    true_dets = kwimage.Detections.from_coco_annots(true_anns, dset=true_dset)
    pred_dets = kwimage.Detections.from_coco_annots(pred_anns, dset=pred_dset)

    print('true_dets.boxes = {!r}'.format(true_dets.boxes))
    print('pred_dets.boxes = {!r}'.format(pred_dets.boxes))

    # Load the image
    fpath = true_dset.get_image_fpath(gid)
    canvas = kwimage.imread(fpath)

    # Use kwimage.Detections draw_on method to modify the image
    canvas = true_dets.draw_on(canvas, color='green')
    canvas = pred_dets.draw_on(canvas, color='blue')

    kwimage.imwrite(viz_fpath, canvas)
