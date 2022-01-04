def demo_vectorized_interface():
    """
    This demonstrates how to use the kwcoco vectorized interface for images /
    categories / annotations.
    """
    import kwcoco
    import ubelt as ub
    # Dummy data for the demo
    coco_dset = kwcoco.CocoDataset.demo('vidshapes3', num_frames=5)
    # Reroot to make file-paths more readable in this demo as relative paths
    coco_dset.reroot(absolute=False)

    ###
    # Images
    ###

    # The :func:`images` method, returns information about multiple images in
    # a dataset. By default, all images are returned. But parameters can
    # be specified to query for particular images.
    images = coco_dset.images()
    print('images = {!r}'.format(images))
    """
    <Images(num=16) at 0x7f1f720407c0>
    """
    # Vectorized objects are just lightweight pointers to the underlying
    # dataset. The only data they contain are object-ids and pointers
    # to the coco dataset. Iterating / indexing into them treats them as
    # integer object ids.
    print('list(images) = {!r}'.format(list(images)))
    """
    list(images) = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    """

    # Data in the underlying image dictionaries can be accessed via lookup
    # or via using the special ".objs" property

    # Query the frame index, video_id, and image-id for each image
    video_ids = images.lookup('id')
    image_ids = images.lookup('video_id')
    frame_idxs = images.lookup('frame_index')
    print('video_ids = {!r}'.format(video_ids))
    print('image_ids = {!r}'.format(image_ids))
    print('frame_idxs = {!r}'.format(frame_idxs))
    """
    video_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    image_ids = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    frame_idxs = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    """

    images.objs
    print('images.objs = {}'.format(ub.repr2(images.objs, nl=1)))
    """
    images.objs = [
        {'channels': 'r|g|b', 'file_name': '_assets/images/img_00001.png', 'frame_index': 0, 'height': 600, 'id': 1, 'video_id': 1, 'warp_img_to_vid': {'type': 'affine'}, 'width': 600},
        {'channels': 'r|g|b', 'file_name': '_assets/images/img_00002.png', 'frame_index': 1, 'height': 600, 'id': 2, 'video_id': 1, 'warp_img_to_vid': {'type': 'affine'}, 'width': 600},
        {'channels': 'r|g|b', 'file_name': '_assets/images/img_00003.png', 'frame_index': 2, 'height': 600, 'id': 3, 'video_id': 1, 'warp_img_to_vid': {'type': 'affine'}, 'width': 600},
        {'channels': 'r|g|b', 'file_name': '_assets/images/img_00004.png', 'frame_index': 3, 'height': 600, 'id': 4, 'video_id': 1, 'warp_img_to_vid': {'type': 'affine'}, 'width': 600},
        {'channels': 'r|g|b', 'file_name': '_assets/images/img_00005.png', 'frame_index': 4, 'height': 600, 'id': 5, 'video_id': 1, 'warp_img_to_vid': {'type': 'affine'}, 'width': 600},
        {'channels': 'r|g|b', 'file_name': '_assets/images/img_00006.png', 'frame_index': 0, 'height': 600, 'id': 6, 'video_id': 2, 'warp_img_to_vid': {'type': 'affine'}, 'width': 600},
        {'channels': 'r|g|b', 'file_name': '_assets/images/img_00007.png', 'frame_index': 1, 'height': 600, 'id': 7, 'video_id': 2, 'warp_img_to_vid': {'type': 'affine'}, 'width': 600},
        {'channels': 'r|g|b', 'file_name': '_assets/images/img_00008.png', 'frame_index': 2, 'height': 600, 'id': 8, 'video_id': 2, 'warp_img_to_vid': {'type': 'affine'}, 'width': 600},
        {'channels': 'r|g|b', 'file_name': '_assets/images/img_00009.png', 'frame_index': 3, 'height': 600, 'id': 9, 'video_id': 2, 'warp_img_to_vid': {'type': 'affine'}, 'width': 600},
        {'channels': 'r|g|b', 'file_name': '_assets/images/img_00010.png', 'frame_index': 4, 'height': 600, 'id': 10, 'video_id': 2, 'warp_img_to_vid': {'type': 'affine'}, 'width': 600},
        {'channels': 'r|g|b', 'file_name': '_assets/images/img_00011.png', 'frame_index': 0, 'height': 600, 'id': 11, 'video_id': 3, 'warp_img_to_vid': {'type': 'affine'}, 'width': 600},
        {'channels': 'r|g|b', 'file_name': '_assets/images/img_00012.png', 'frame_index': 1, 'height': 600, 'id': 12, 'video_id': 3, 'warp_img_to_vid': {'type': 'affine'}, 'width': 600},
        {'channels': 'r|g|b', 'file_name': '_assets/images/img_00013.png', 'frame_index': 2, 'height': 600, 'id': 13, 'video_id': 3, 'warp_img_to_vid': {'type': 'affine'}, 'width': 600},
        {'channels': 'r|g|b', 'file_name': '_assets/images/img_00014.png', 'frame_index': 3, 'height': 600, 'id': 14, 'video_id': 3, 'warp_img_to_vid': {'type': 'affine'}, 'width': 600},
        {'channels': 'r|g|b', 'file_name': '_assets/images/img_00015.png', 'frame_index': 4, 'height': 600, 'id': 15, 'video_id': 3, 'warp_img_to_vid': {'type': 'affine'}, 'width': 600},
    ]
    """

    ###
    # Annotations
    ###

    # Annotations work similarly, find all annotations for a specific image
    gid = image_ids[-1]
    annots = coco_dset.annots(gid=gid)
    print('annots = {!r}'.format(annots))
    """
    <Annots(num=2) at 0x7f202f3c3e20>
    """

    # The ".objs" property returns a list of the raw coco dictionaries
    # (this works on all vectorized objects)
    anns = annots.objs
    # Lets remove the segmentations for readability
    for ann in anns:
        ann.pop('segmentation', None)
        ann.pop('keypoints', None)
    print('annots.objs = {}'.format(ub.repr2(annots.objs, nl=1)))
    """
    annots.objs = [
        {'bbox': [-15.0, 192.4, 138.3, 50.9], 'category_id': 1, 'id': 3, 'image_id': 3, 'track_id': 0},
        {'bbox': [440.7, -62.2, 65.1, 260.3], 'category_id': 1, 'id': 8, 'image_id': 3, 'track_id': 1},
    ]
    """

    # Annotations have a special method that returns kwimage detection objects
    dets = annots.detections
    print('dets.data = {}'.format(ub.repr2(dets.data, nl=1)))
    """
    dets.data = {
        'boxes': <Boxes(xywh,
                     array([[-15. , 192.4, 138.3,  50.9],
                            [440.7, -62.2,  65.1, 260.3]], dtype=float32))>,
        'class_idxs': np.array([0, 0], dtype=np.int64),
        'keypoints': <PointsList(n=2) at 0x7f1f72040d90>,
        'segmentations': <PolygonList(n=2) at 0x7f1f72927310>,
    }
    """

    ###
    # Images (Advanced)
    ###

    # First query for images (e.g. within a video)
    images = coco_dset.images(vidid=1)
    print('images = {!r}'.format(images))
    print('images.objs = {}'.format(ub.repr2(images.objs, nl=1)))
    """
    images = <Images(num=5) at 0x7f1f729bc490>
    images.objs = [
        {'channels': 'r|g|b', 'file_name': '_assets/images/img_00001.png', 'frame_index': 0, 'height': 600, 'id': 1, 'video_id': 1, 'warp_img_to_vid': {'type': 'affine'}, 'width': 600},
        {'channels': 'r|g|b', 'file_name': '_assets/images/img_00002.png', 'frame_index': 1, 'height': 600, 'id': 2, 'video_id': 1, 'warp_img_to_vid': {'type': 'affine'}, 'width': 600},
        {'channels': 'r|g|b', 'file_name': '_assets/images/img_00003.png', 'frame_index': 2, 'height': 600, 'id': 3, 'video_id': 1, 'warp_img_to_vid': {'type': 'affine'}, 'width': 600},
        {'channels': 'r|g|b', 'file_name': '_assets/images/img_00004.png', 'frame_index': 3, 'height': 600, 'id': 4, 'video_id': 1, 'warp_img_to_vid': {'type': 'affine'}, 'width': 600},
        {'channels': 'r|g|b', 'file_name': '_assets/images/img_00005.png', 'frame_index': 4, 'height': 600, 'id': 5, 'video_id': 1, 'warp_img_to_vid': {'type': 'affine'}, 'width': 600},
    ]
    """

    # with an Images you can also access Annotations objects directly
    # Queries against a group are naturally grouped
    annot_groups = images.annots
    aids_per_image = annot_groups.lookup('id')
    print('annot_groups = {!r}'.format(annot_groups))
    print('aids_per_image = {!r}'.format(aids_per_image))
    """
    annot_groups = <AnnotGroups(n=5, m=2.0, s=0.0) at 0x7f1f7206ce80>
    aids_per_image = [[1, 6], [2, 7], [8, 3], [9, 4], [10, 5]]
    """
