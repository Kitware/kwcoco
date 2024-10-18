KWCOCO Views / Spaces
=====================

NOTE:

We are working on improving terminology and data structures when discussing
this topic. Not all documentation may be up to date. There are several things
to be aware of.

* Moving forward we may replace the word "Space" with "View"

* The old term is "Auxiliary", which is being replaced with "Asset"

* "Assets" are currently (as of 0.7.5) stored as part of the Image table, but we will likely move them to their own Asset table in the future.

Recall that kwcoco has a list of image dictionaries.
Each image may contain a "video_id" that references a video sequence it is part of.
Each image can contain multiple auxiliary-images (i.e. assets).

In the common case, images do not contain any auxiliary / asset files. Assets
(or auxiliary files) are mainly used in multispectral or multimodal datasets.
If an image has auxiliary files, these will be stored as a list of auxiliary
dictionaries in the image dictionary.


For example a multispectral video with may look like this:

.. code::

    "videos": [
        {
            "name": "TheVideoName",
            "width": 300,
            "height": 400
        },
     ]

    "images": [
        {
            "name": "TheImageName",
            "width": 600,
            "height": 800,
            "video_id": 1,
            "frame_index": 0,
            "warp_img_to_vid": {"scale": 0.5},
            "assets": [
                {
                    "file_name": "band1.tif",
                    "warp_aux_to_img": {"scale": 2.0},
                    "width": 300,
                    "height": 400
                    "channels": "band1",
                    "num_bands": 1,
                },
                {
                    "file_name": "band2.tif",
                    "warp_aux_to_img": {"scale": 1.0},
                    "channels": "band2",
                    "num_bands": 1,
                },
            ],
        },
    ]

A rgb-video with no assets with may look like this:

.. code::

    "videos": [
        {
            "name": "TheVideoName",
            "width": 300,
            "height": 400
        },
     ]

    "images": [
        {
            "name": "TheImageName",
            "width": 600,
            "height": 800,
            "video_id": 1,
            "frame_index": 0,
            "warp_img_to_vid": {"scale": 0.5},
            "file_name": "images/the_image.jpg"
            "channels": "r|g|b",
            "num_bands": 3,
        },
    ]


There are 3 coordinate spaces that we are concerned about in kwcoco.

* Asset Space - all the images are in their native on-disk resolutions.
* Image Space - coordinates that belong to the base image or the coordinate that all "auxiliary" images are aligned to.
* Video Spce - the coordinates that all images in a video sequence are aligned to.

The following visualizes these key spaces:

.. image:: https://i.imgur.com/QuiSJwR.png


NOTE: we use the terms "auxiliary" and "asset" interchangeably. The code was
originally written using "auxiliary", but we later switched to "asset".
Terminology will vary based on when something was written, eventually we will
deprecate all instances of "auxiliary" and only refer to "assets".

Each "asset" / "auxiliary" dictionary, if they exist, contains a
``warp_aux_to_img`` transform that aligns the auxiliary image into the common
"image-space".

Each "image" dictionary in a video sequence contains a ``warp_img_to_vid``
transform that aligns anything in image space into video space such that all
frames in the video are aligned.

If these transforms are unspecified they are assumed to be the identity
transform (i.e. no change), so they are only needed if your on-disk data is not
already aligned (which can be desirable to save disk-space and reduce
resampling artifacts).

The transform dictionaries can be in any format coercible by
:class:`kwimage.Affine<kwimage.transforms.Affine>`.

The :class:`kwcoco.CocoDataset<kwcoco.coco_dataset.CocoDataset>` also exposes
the
:func:`kwcoco.CocoDataset.delayed_load<kwcoco.coco_dataset.MixinCocoAccessors.delayed_load>`
method, which can be used to access image information in image or video space.


.. code:: python

    import kwcoco
    import ubelt as ub
    gid = 1
    self = kwcoco.CocoDataset.demo('vidshapes8-multispectral')

    # Show the structure of the image and auxiliary dictionaries
    print(ub.urepr(self.index.imgs[1], nl=-1, sort=0))

    # The delayed object is a pointer to the image files that contains
    # appropriate transformation. Additional transformations can be
    # specified. These are all fused together to reduce resampling
    # artifacts.
    img_delayed = self.delayed_load(gid, space='image')
    # Execute all transforms
    img_final = img_delayed.finalize()

    #
    vid_delayed = self.delayed_load(gid, space='image')
    # Execute all transforms
    vid_final = vid_delayed.finalize()

Currently the ``warp_img_to_vid`` transform in the demo
vidshapes8-multispectral dataset is the identity, but if it was different, then
"vid_final" and "img_final" would be returned in different coordinate systems.
(TODO: update demo data with an option such that the video and image space are
different)
