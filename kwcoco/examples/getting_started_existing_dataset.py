
def getting_started_existing_dataset():
    """
    If you want to start using the Python API. Just open IPython and try:
    """

    import kwcoco
    dset = kwcoco.CocoDataset('<DATASET_NAME>/data.kwcoco.json')

    """
    You can access category, image, and annotation infomation using the index:
    """
    cat = dset.index.cats[1]
    print('The First Category = {!r}'.format(cat))

    img = dset.index.imgs[1]
    print('The First Image = {!r}'.format(img))

    ann = dset.index.anns[1]
    print('The First Annotation = {!r}'.format(ann))


def the_core_dataset_backend():

    import kwcoco
    dset = kwcoco.CocoDataset.demo('shapes2')

    # Make data slightly tider for display
    for ann in dset.dataset['annotations']:
        ann.pop('segmentation', None)
        ann.pop('keypoints', None)
        ann.pop('area', None)

    for cat in dset.dataset['categories']:
        cat.pop('keypoints', None)

    for img in dset.dataset['images']:
        img.pop('channels', None)

    from os.path import dirname
    import ubelt as ub
    import kwarray
    dset.reroot(dirname(dset.fpath), absolute=False)
    dset.remove_annotations(kwarray.shuffle(list(dset.anns.keys()))[10:])

    print('dset.dataset = {}'.format(ub.repr2(dset.dataset, nl=2)))


def demo_vectorize_interface():

    """

    >>> import kwcoco
    >>> dset = kwcoco.CocoDataset.demo('vidshapes2')
    >>> #
    >>> aids = [1, 2, 3, 4]
    >>> annots = dset.annots(aids)
    ...
    >>> print('annots = {!r}'.format(annots))
    annots = <Annots(num=4) at ...>

    >>> annots.lookup('bbox')
    [[346.5, 335.2, 33.2, 99.4],
     [344.5, 327.7, 48.8, 111.1],
     [548.0, 154.4, 57.2, 62.1],
     [548.7, 151.0, 59.4, 80.5]]

    >>> gids = annots.lookup('image_id')
    >>> print('gids = {!r}'.format(gids))
    gids = [1, 2, 1, 2]

    >>> images = dset.images(gids)
    >>> list(zip(images.lookup('width'), images.lookup('height')))
    [(600, 600), (600, 600), (600, 600), (600, 600)]


    """
