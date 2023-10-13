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
    print('images.objs = {}'.format(ub.urepr(images.objs, nl=1)))

    # With the index
    image_ids = dset.index.vidid_to_gids[video_id]
    imgs = [dset.index.imgs[gid] for gid in image_ids]
    print('image_ids = {!r}'.format(image_ids))
    print('imgs = {}'.format(ub.urepr(imgs, nl=1)))


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

    print('hist = {}'.format(ub.urepr(hist, nl=1)))


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


def what_order_are_images_returned_in():
    """
    Q. What order are images returned in? What about when using ndsampler.

    A. When requesting images from a video they are ordered by their timestamp
    / frame_index.  For ndsampler, the order is always the requested order.

    """
    import kwcoco
    import ndsampler
    dset = kwcoco.CocoDataset.demo('vidshapes1', num_frames=10)
    images = dset.images()

    videos = dset.videos()

    # Images always ordered by timestamp/frame-index here
    images = dset.images(video_id=videos[0])

    # But you can always lookup the attribute and check yourself
    frame_idxs = images.lookup('frame_index')
    assert sorted(frame_idxs) == frame_idxs

    gid1, gid2, gid3 = images[0:3]

    sampler = ndsampler.CocoSampler(dset)

    target = {'gids': [gid1, gid2, gid3]}
    sample = sampler.load_sample(target)
    # Data is in the requested order
    assert sample['target']['gids'] == [gid1, gid2, gid3]

    target = {'gids': [gid2, gid3, gid1]}
    sample = sampler.load_sample(target)
    # Data is in the requested order
    assert sample['target']['gids'] == [gid2, gid3, gid1]


def remap_category_ids_demo():
    """
    Q: Hey! I'm using kwcoco union to combine two kwcoco files. When I run it,
    the category ids get shifted by 1.

    A: Currently not with a single operation. To keep the implementation simple
    I opted to always reassign category ids, and by convention with the
    original mscoco spec I start ids at 1. However, a bit of post processing
    can fix this.

    It's also worth noting that the id's in each object are meant to be internal
    and not used by an external program expecting persistence. So in code like
    kwcoco eval, when I have to handle categories in two coco files, I use the
    category names to build a mapping from category ids betwen the files.


    A bit of custom code like this should help fix the issue.  As long as you
    change ids in the core annotation and category dictionaries that should be
    enough. Just be sure to rebuild the index after.
    """

    import kwcoco
    dset = kwcoco.CocoDataset.demo()

    existing_cids = dset.categories().lookup('id')
    cid_mapping = {cid: cid + 100 for cid in existing_cids}

    for cat in dset.dataset['categories']:
        old_cid = cat['id']
        new_cid = cid_mapping[old_cid]
        cat['id'] = new_cid

    for ann in dset.dataset['annotations']:
        old_cid = ann['category_id']
        new_cid = cid_mapping[old_cid]
        ann['category_id'] = new_cid

    dset._build_index()


def filter_images_by_attribute():
    """
    Question:
        I have a coco dataset example video frame image:


                In [149]: coco.imgs[10]
                Out[149]:
                {'id': 10,
                 'file_name': 'images/frame_00001.png',
                 'video_id': 1,
                 'frame_index': 3,
                 'gt': 0,
                 'pred': 0,
                 'conf': [0.455, 0.0126]}

        There are multiple videos in the coco dataset. I can filter coco.imgs for indexes with the video_id "1" like this:

            coco.index.vidid_to_gids[1]

        Now, how do I filter coco.imgs for indexes with gt=0, using .index?

    Ansswer:
        Unfortunately it's going to be a linear operation (As the library
        evolves it's sql support this might change), but typically there aren't
        so many images where that is an issue. Just access the image
        dictionaries directly..

        Here are several ways to do it:
    """
    import kwcoco
    import kwarray
    dset = kwcoco.CocoDataset.demo('vidshapes2', num_frames=10)

    # Add random attributes for the demo:
    rng = kwarray.ensure_rng(0)
    for img in dset.dataset['images']:
        img['gt'] = rng.randint(0, 2)

    # Option 1: Direct access
    list_of_image_ids: list[int] = dset.index.vidid_to_gids[1]
    filtered_image_ids: list[int] = []
    for gid in list_of_image_ids:
        img = dset.index.imgs[gid]
        if img.get('gt', None) == 0:
            filtered_image_ids.append(gid)

    # Option 2: Using the vectorized API (which is really doing pretty much the
    # same thing)
    list_of_image_ids: list[int] = dset.index.vidid_to_gids[1]
    images = dset.images(list_of_image_ids)
    flags : list[bool] = [x == 0 for x in images.lookup('gt', default=None)]
    filtered_images = images.compress(flags)

    filtered_image_ids: list[int] = list(filtered_images)
    filtered_image_dicts: list[dict] = filtered_images.objs

    # Option 3: Using the vectorized API at the object level
    list_of_image_ids: list[int] = dset.index.vidid_to_gids[1]
    images = dset.images(list_of_image_ids)

    filtered_image_dicts: list[dict] =  [img for img in images.objs if img.get('gt') == 0]
    filtered_image_ids: list[int] = [img['id'] for img in filtered_image_dicts]

    # Option 4: A 1 liner
    filtered_image_ids: list[int] = [img['id'] for img in dset.videos(video_ids=[1]).images[0].objs if img['gt'] == 0]
