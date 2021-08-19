KWCoco "Vectorized" Interface
=============================

Note, this is just a quick "paste dump" from conversations I've had about the
kwcoco vectorized interface. Reorganizing this as proper docs is left as a
TODO.

The vectorized methods are the "fancy" vectorized way of looking up information
in kwcoco, as opposed to accessing the dictionaries 


The Mixin Coco Objects:
https://kwcoco.readthedocs.io/en/release/#cocodataset-methods-via-mixincocoobjects
/
https://kwcoco.readthedocs.io/en/release/autoapi/kwcoco/coco_dataset/index.html#kwcoco.coco_dataset.MixinCocoObjects
is where annots comes from. It also has an images, categories, and videos
method.


CocoDataset methods (via MixinCocoObjects)
******************************************

* :func:`kwcoco.CocoDataset.annots<kwcoco.coco_dataset.MixinCocoObjects.annots>` - Return vectorized annotation objects
* :func:`kwcoco.CocoDataset.categories<kwcoco.coco_dataset.MixinCocoObjects.categories>` - Return vectorized category objects
* :func:`kwcoco.CocoDataset.images<kwcoco.coco_dataset.MixinCocoObjects.images>` - Return vectorized image objects
* :func:`kwcoco.CocoDataset.videos<kwcoco.coco_dataset.MixinCocoObjects.videos>` - Return vectorized video objects


The logic for all of these is defined in the coco_objects1d.py file and they
all inherit from ObjectList1D:
https://kwcoco.readthedocs.io/en/release/autoapi/kwcoco/coco_objects1d/index.html
They are a concice way of accessing similar attributes of multiple objects at
once, but they all go through the dictionary interface under the hood.


.. code:: python

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


The "Annots"  object has a "detections" convinience method


.. code:: python

    import kwimage
    import kwcoco

    # Demo data
    dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
    transform = kwimage.Affine.random()

    annots_objs: kwcoco.coco_objects1d.Annots = dset.annots(gid=1)
    dets: kwimage.Detections = annots_objs.detections

    # Inspect dets.data to see what it holds (note it does not have everything from the ann objects themselves)
    print(dets.data)

    # You can see the raw ann dictionary objects as such
    print(annots_objs.objs)

    # The useful thing about kwimage structures is they all have a warp method
    new_dets = dets.warp(transform)

    # The new annot from detections will not transfer all of the properties, but the relevant geometries 
    # will all be warped. 
    new_anns = list(new_dets.to_coco(style='new'))

    for old_ann, new_ann in zip(annots_objs.objs, new_anns):
        # Transfer relevant data
        new_ann['track_id'] = old_ann['track_id']
        new_ann['image_id'] = 2  # need to set this

