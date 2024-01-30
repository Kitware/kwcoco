Getting Started With KW-COCO
============================

This document is a work in progress, and does need to be updated and
refactored.

FAQ
---

Q: What is ``kwcoco``? A: An extension of the MS-COCO data format for
storing a “manifest” of categories, images, and annotations.

Q: Why yet another data format? A: MS-COCO did not have support for
video and multimodal imagery. These are important problems in computer
vision and it seems reasonable (although challenging) that there could
be a data format that could be used as an interchange for almost all
vision problems.

Q: Why extend MS-COCO and not create something else? A: To draw on the
existing adoption of the MS-COCO format.

Q: What’s so great about MS-COCO? A: It has an intuitive data structure
that’s simple to interface with.

Q: Why not pycocotools? A: That module doesn’t allow you to edit the
dataset programmatically, and requires C backend. This module allows
dynamic modification addition and removal of images / categories /
annotations / videos, in addition to other places where it goes beyond
the functionality of the pycocotools module. We have a much more
configurable / expressive way of computing and recording object
detection metrics. If we are using an mscoco-compliant database (which
can be verified / coerced from the ``kwcoco conform`` CLI tool), then we
do call pycocotools for functionality not directly implemented here.

Q: Would you ever extend kwcoco to go beyond computer vision? A: Maybe,
it would be something new though, and only use kwcoco as an inspiration.
If extending past computer vision I would want to go back and rename /
reorganize the spec.

Examples
--------

These python files have a few example uses cases of kwcoco

-  `draw_gt_and_predicted_boxes <https://github.com/Kitware/kwcoco/blob/master/kwcoco/examples/draw_gt_and_predicted_boxes.py>`__
-  `modification_example <https://github.com/Kitware/kwcoco/blob/master/kwcoco/examples/modification_example.py>`__
-  `simple_kwcoco_torch_dataset <https://github.com/Kitware/kwcoco/blob/master/kwcoco/examples/simple_kwcoco_torch_dataset.py>`__
-  `getting_started_existing_dataset <https://github.com/Kitware/kwcoco/blob/master/kwcoco/examples/getting_started_existing_dataset.py>`__

Design Goals
------------

-  Always be a strict superset of the original MS-COCO format

-  Extend the scope of MS-COCO to broader computer-vision domains.

-  Have a fast pure-Python API to perform lower level tasks. (Allow
   optional C backends for features that need speed boosts)

-  Have an easy-to-use command line interface to perform higher level
   tasks.

Use cases
---------

KWCoco has been designed to work with these tasks in these image
modalities.

Tasks
~~~~~

-  Captioning

-  Classification

-  Segmentation

-  Keypoint Detection / Pose Estimation

-  Object Detection

Modalities
~~~~~~~~~~

-  Single Image

-  Video

-  Multispectral Imagery

-  Images with auxiliary information (2.5d, flow, disparity, stereo)

-  Combinations of the above.

KWCOCO Spec
-----------

A high level description of the kwcoco spec is given in :py:mod:`kwcoco.coco_dataset`.

A formal json-schema is defined in :py:mod:`kwcoco.coco_schema` and is shown
here:

.. .. jsonschema:: kwcoco.coco_schema.COCO_SCHEMA

.. .. TODO: Fix the width on this
.. jsonschema:: coco_schema.json



The Python API
--------------

Creating a dataset
~~~~~~~~~~~~~~~~~~

The Python API can be used to load an existing dataset or initialize an
empty dataset. In both cases the dataset can be modified by
adding/removing/editing categories, videos, images, and annotations.

You can load an existing dataset as such:

.. code:: python

   import kwcoco
   dset = kwcoco.CocoDataset('path/to/data.kwcoco.json')

You can initialize an empty dataset as such:

.. code:: python

   import kwcoco
   dset = kwcoco.CocoDataset()

In both cases you can add and remove data items. When you add an item,
it returns the internal integer primary id used to refer to that item.

.. code:: python

   cid = dset.add_category(name='cat')

   gid = dset.add_image(file_name='/path/to/limecat.jpg')

   aid = dset.add_annotation(image_id=gid, category_id=cid, bbox=[0, 0, 100, 100])

The ``CocoDataset`` class has an instance variable ``dset.dataset``
which is the loaded JSON data structure. This dataset can be interacted
with directly.

.. code:: python

   # Loop over all categories, images, and annotations

   for img in dset.dataset['categories']:
       print(img)

   for img in dset.dataset['images']:
       print(img)

   for img in dset.dataset['annotations']:
       print(img)

This the above example, this will result in:

::

   OrderedDict([('id', 1), ('name', 'cat')])
   OrderedDict([('id', 1), ('file_name', '/path/to/limecat.jpg')])
   OrderedDict([('id', 1), ('image_id', 1), ('category_id', 1), ('bbox', [0, 0, 100, 100])])

In the above example, you can display the underlying ``dataset``
structure as such

.. code:: python

   print(dset.dumps(indent='    ', newlines=True))

This results in

::

   {
   "info": [],
   "licenses": [],
   "categories": [
       {"id": 1, "name": "cat"}
   ],
   "videos": [],
   "images": [
       {"id": 1, "file_name": "/path/to/limecat.jpg"}
   ],
   "annotations": [
       {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 100, 100]}
   ]
   }

In addition to accessing ``dset.dataset`` directly, the ``CocoDataset``
object maintains an ``index`` which allows the user to quickly lookup
objects by primary or secondary keys. A list of available indexes are:

.. code:: python

   dset.index.anns    # a mapping from annotation-ids to annotation dictionaries
   dset.index.imgs    # a mapping from image-ids to image dictionaries
   dset.index.videos  # a mapping from video-ids to video dictionaries
   dset.index.cats    # a mapping from category-ids to category dictionaries

   dset.index.gid_to_aids    # a mapping from an image id to annotation ids contained in the image
   dset.index.cid_to_aids    # a mapping from an annotation id to annotation ids with that category
   dset.index.vidid_to_gids  # a mapping from an video id to image ids contained in the video

   dset.index.name_to_video  # a mapping from a video name to the video dictionary
   dset.index.name_to_cat    # a mapping from a category name to the category dictionary
   dset.index.name_to_img    # a mapping from an image name to the image dictionary
   dset.index.file_name_to_img  # a mapping from an image file name to the image dictionary

These indexes are dynamically updated when items are added or removed.

Using kwcoco to write a torch dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to write a torch dataset with kwcoco is to combine it
with
`ndsampler <https://gitlab.kitware.com/computer-vision/ndsampler>`__

Examples of kwcoco + ndsampler being to write torch datasets to train
deep networks can be found in
`netharn's <https://gitlab.kitware.com/computer-vision/netharn>`__
examples for:
`detection <https://gitlab.kitware.com/computer-vision/netharn/-/blob/master/netharn/examples/object_detection.py>`__, 
`classification <https://gitlab.kitware.com/computer-vision/netharn/-/blob/master/netharn/examples/classification.py>`__, and
`segmentation <https://gitlab.kitware.com/computer-vision/netharn/-/blob/master/netharn/examples/segmentation.py>`__

(Note: netharn is deprecated in favor of pytorch-lightning, but the dataset examples still hold)

Technical Debt
--------------

Based on design decisions made in the original MS-COCO and KW-COCO,
there are a few weird things

-  The “bbox” field gives no indication it should be xywh format.

-  We can’t use “vid” as a variable name for “video-id” because “vid” is
   also an abbreviation for “video”. Hence, while category, image, and
   annotation all have a nice 1-letter prefix to their id in the
   standard variable names I use (i.e. cid, gid, aid). I have to use
   vidid to refer to “video-ids”.

-  I’m not in love with the way “keypoint_categories” are handled.

-  Are “images” always “images”? Are “videos” always “videos”?

-  Would we benefit from using JSON-LD?

-  The “prob” field needs to be better defined

-  The name “video” might be confusing. Its just a temporally ordered
   group of images.

Code Examples
-------------

See the README and the doctests.
