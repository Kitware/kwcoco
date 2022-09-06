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


def get_all_channels_in_dataset():
    """
    Q.  After I load a kwcoco.json into a kwcoco_dset, is there a nice way to
    query what channels are available for the input imagery? It looks like I
    can iterate over .imgs and build my own set, but maybe theres a built in
    way

    A. The better way is to use the CocoImage API.
    """
    import kwcoco
    dset = kwcoco.CocoDataset.demo('vidshapes8-msi-multisensor')

    all_channels = []
    for gid in dset.images():
        # Build the CocoImage (a lightweight wrapper around the image
        # dictionary) and then access the "channels" attribute.
        coco_img = dset.coco_image(gid)
        channels = coco_img.channels
        print(f'channels={channels}')
        all_channels.append(channels)

    # You can build a histogram if you want:
    import ubelt as ub
    hist = ub.ddict(lambda: 0)
    for channels in all_channels:
        hist[channels.spec] += 1

    print('hist = {}'.format(ub.repr2(hist, nl=1)))


def whats_the_difference_between_Images_and_CocoImage():
    """
    Q. What is the difference between `kwcoco.Images` and `kwcoco.CocoImage`.

    It's a little weird because it grew organically, but the "vectorized API"
    calls like `.images`, `.annots`, `.videos` are methods for handling
    multiple dictionaries at once. E.g. `dset.images().lookup('width')` returns
    a list of the width attribute for each dictionary that particular `Images`
    object is indexing (which by default is all of them, although you can
    filter).

    In contrast the `kwcoco.CocoImage` object is for working with exactly one
    image. The important thing to note is if you have a CocoImage `coco_img =
    dset.coco_image(1)` The `coco_img.img` attribute is exactly the underlying
    dictionary. So you are never too far away from it.

    Similarly for the `Images` objects: `dset.images().objs`  returns a list of
    all of the image dictionaries in that set.
    """
    ...
