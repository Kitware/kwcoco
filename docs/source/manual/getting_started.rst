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

    {'id': 1, 'name': 'cat'}
    {'id': 1, 'file_name': '/path/to/limecat.jpg'}
    {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 100, 100]}


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

Due to design decisions inherited from the original MS-COCO specification and early iterations of KW-COCO, a few legacy quirks and inconsistencies remain:

- Ambiguous bbox format: The bbox field does not explicitly indicate that it uses the [x, y, width, height] (xywh) format, which can lead to confusion without referencing the documentation.

- Naming conflict with vid: The abbreviation vid is ambiguous - it can mean either video or video-id. To avoid confusion in code, we use vidid to refer to video IDs. This breaks the otherwise clean 1-letter prefix pattern used for other identifiers (e.g., aid, gid, cid for annotations, images, categories). We are thus moving away from this in favor of more verbose but explicit identifiers, but the old ones still exist.

- Keypoint category representation: The current design for keypoint_categories is awkward and may benefit from a clearer structure or better integration with existing category metadata.

- Terminology ambiguity: The terms images and videos are overloaded. For example, a video is simply a temporally ordered group of images, but this abstraction may not be immediately obvious.

- Potential use of JSON-LD: It's unclear whether adopting JSON-LD would improve interoperability or clarity. This remains an open question worth exploring.

- Poorly defined prob field: The meaning and semantics of the prob (probability) field are underspecified. Clarifying its purpose and standardizing its use would improve consistency across datasets.

- Confusing use of video: As mentioned, the term video may imply an actual video file, when in practice, it refers to an ordered sequence of image frames. A clearer term might reduce confusion.




Code Examples
-------------

See the README and the doctests.
