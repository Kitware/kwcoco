"""
These are answers to the questions: How do I?
"""


def get_images_with_videoid():
    """
    Q: How would you recommend querying a kwcoco file to get all of the images
    associated with a video id?
    """
    import kwcoco
    import ubelt as ub
    dset = kwcoco.CocoDataset.demo('vidshapes8-msi')
    video_id = 1

    # With Object1d API
    images = dset.images(vidid=video_id)
    print('images = {!r}'.format(images))
    print('images.objs = {}'.format(ub.repr2(images.objs, nl=1)))

    # With the index
    image_ids = dset.index.vidid_to_gids[video_id]
    imgs = [dset.index.imgs[gid] for gid in image_ids]
    print('image_ids = {!r}'.format(image_ids))
    print('imgs = {}'.format(ub.repr2(imgs, nl=1)))
