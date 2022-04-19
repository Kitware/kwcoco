KWCOCO - The Kitware COCO Module
================================

.. # TODO Get CI services running on gitlab 

|GitlabCIPipeline| |GitlabCICoverage| |Appveyor| |Pypi| |Downloads| |ReadTheDocs|

+------------------+------------------------------------------------------+
| Read the docs    | https://kwcoco.readthedocs.io                        |
+------------------+------------------------------------------------------+
| Gitlab (main)    | https://gitlab.kitware.com/computer-vision/kwcoco    |
+------------------+------------------------------------------------------+
| Github (mirror)  | https://github.com/Kitware/kwcoco                    |
+------------------+------------------------------------------------------+
| Pypi             | https://pypi.org/project/kwcoco                      |
+------------------+------------------------------------------------------+

The main webpage for this project is: https://gitlab.kitware.com/computer-vision/kwcoco

The Kitware COCO module defines a variant of the Microsoft COCO format,
originally developed for the "collected images in context" object detection
challenge. We are backwards compatible with the original module, but we also
have improved implementations in several places, including segmentations,
keypoints, annotation tracks, multi-spectral images, and videos (which
represents a generic sequence of images).

A KWCOCO file is a "manifest" that serves as a single reference that points to
all images, categories, and annotations in a computer vision dataset. Thus,
when applying an algorithm to a dataset, it is sufficient to have the algorithm
take one dataset parameter: the path to the KWCOCO file.  Generally a KWCOCO
file will live in a "bundle" directory along with the data that it references,
and paths in the KWCOCO file will be relative to the location of the KWCOCO
file itself.

The main data structure in this model is largely based on the implementation in
https://github.com/cocodataset/cocoapi It uses the same efficient core indexing
data structures, but in our implementation the indexing can be optionally
turned off, functions are silent by default (with the exception of long running
processes, which optionally show progress by default). We support helper
functions that add and remove images, categories, and annotations. 

We have reimplemented the object detection scoring code in the `kwcoco.metrics`
submodule.  

The original `pycocoutils` API is exposed via the `kwcoco.compat_dataset.COCO`
class for drop-in replacement with existing tools that use `pycocoutils`. 

There is some support for `kw18` files in the `kwcoco.kw18` module.

Installation
------------

The `kwcoco <https://pypi.org/project/kwcoco/>`_.  package can be installed via pip:

.. code-block:: bash

    pip install kwcoco


The KWCOCO CLI
--------------

After installing KWCOCO, you will also have the ``kwcoco`` command line tool. 
This uses a ``scriptconfig`` / ``argparse`` CLI interface. Running ``kwcoco
--help`` should provide a good starting point.

.. code-block:: 

    usage: kwcoco [-h] [--version] {stats,union,split,show,toydata,eval,conform,modify_categories,reroot,validate,subset,grab} ...

    The Kitware COCO CLI

    positional arguments:
      {stats,union,split,show,toydata,eval,conform,modify_categories,reroot,validate,subset,grab}
                            specify a command to run
        stats               Compute summary statistics about a COCO dataset
        union               Combine multiple COCO datasets into a single merged dataset.
        split               Split a single COCO dataset into two sub-datasets.
        show                Visualize a COCO image using matplotlib or opencv, optionally writing
        toydata             Create COCO toydata
        eval                Evaluate detection metrics using a predicted and truth coco file.
        conform             Make the COCO file conform to the spec.
        modify_categories   Rename or remove categories
        reroot              Reroot image paths onto a new image root.
        validate            Validate that a coco file conforms to the json schema, that assets
        subset              Take a subset of this dataset and write it to a new file
        grab                Grab standard datasets.

    optional arguments:
      -h, --help            show this help message and exit
      --version             show version number and exit (default: False)


This should help you inspect (via stats and show), combine (via union), and
make training splits (via split) using the command line. Also ships with
toydata, which generates a COCO file you can use for testing.


Toy Data
--------

Don't have a dataset with you, but you still want to test out your algorithms?
Try the KWCOCO shapes demo dataset, and generate an arbitrarily large dataset.

The toydata submodule renders simple objects on a noisy background ---
optionally with auxiliary channels --- and provides bounding boxes,
segmentations, and keypoint annotations. The following example illustrates a
generated toy image with and without overlaid annotations. 


..  ..image:: https://i.imgur.com/2K17R2U.png

.. image:: https://i.imgur.com/Vk0zUH1.png
   :height: 100px
   :align: left



The CocoDataset object
----------------------

The ``kwcoco.CocoDataset`` class is capable of dynamic addition and removal of
categories, images, and annotations. Has better support for keypoints and
segmentation formats than the original COCO format. Despite being written in
Python, this data structure is reasonably efficient.


.. code-block:: python

        >>> import kwcoco
        >>> import json
        >>> # Create demo data
        >>> demo = CocoDataset.demo()
        >>> # could also use demo.dump / demo.dumps, but this is more explicit
        >>> text = json.dumps(demo.dataset)
        >>> with open('demo.json', 'w') as file:
        >>>    file.write(text)

        >>> # Read from disk
        >>> self = CocoDataset('demo.json')

        >>> # Add data
        >>> cid = self.add_category('Cat')
        >>> gid = self.add_image('new-img.jpg')
        >>> aid = self.add_annotation(image_id=gid, category_id=cid, bbox=[0, 0, 100, 100])

        >>> # Remove data
        >>> self.remove_annotations([aid])
        >>> self.remove_images([gid])  
        >>> self.remove_categories([cid])

        >>> # Look at data
        >>> print(ub.repr2(self.basic_stats(), nl=1))
        >>> print(ub.repr2(self.extended_stats(), nl=2))
        >>> print(ub.repr2(self.boxsize_stats(), nl=3))
        >>> print(ub.repr2(self.category_annotation_frequency()))
        

        >>> # Inspect data
        >>> import kwplot
        >>> kwplot.autompl()
        >>> self.show_image(gid=1)

        >>> # Access single-item data via imgs, cats, anns
        >>> cid = 1
        >>> self.cats[cid]
        {'id': 1, 'name': 'astronaut', 'supercategory': 'human'}

        >>> gid = 1
        >>> self.imgs[gid]
        {'id': 1, 'file_name': 'astro.png', 'url': 'https://i.imgur.com/KXhKM72.png'}

        >>> aid = 3
        >>> self.anns[aid]
        {'id': 3, 'image_id': 1, 'category_id': 3, 'line': [326, 369, 500, 500]}

        # Access multi-item data via the annots and images helper objects
        >>> aids = self.index.gid_to_aids[2]
        >>> annots = self.annots(aids)

        >>> print('annots = {}'.format(ub.repr2(annots, nl=1, sv=1)))
        annots = <Annots(num=2)>

        >>> annots.lookup('category_id')
        [6, 4]

        >>> annots.lookup('bbox')
        [[37, 6, 230, 240], [124, 96, 45, 18]]

        >>> # built in conversions to efficient kwimage array DataStructures
        >>> print(ub.repr2(annots.detections.data))
        {
            'boxes': <Boxes(xywh,
                         array([[ 37.,   6., 230., 240.],
                                [124.,  96.,  45.,  18.]], dtype=float32))>,
            'class_idxs': np.array([5, 3], dtype=np.int64),
            'keypoints': <PointsList(n=2) at 0x7f07eda33220>,
            'segmentations': <PolygonList(n=2) at 0x7f086365aa60>,
        }
        
        >>> gids = list(self.imgs.keys())
        >>> images = self.images(gids)
        >>> print('images = {}'.format(ub.repr2(images, nl=1, sv=1)))
        images = <Images(num=3)>

        >>> images.lookup('file_name')
        ['astro.png', 'carl.png', 'stars.png']

        >>> print('images.annots = {}'.format(images.annots))
        images.annots = <AnnotGroups(n=3, m=3.7, s=3.9)>

        >>> print('images.annots.cids = {!r}'.format(images.annots.cids))
        images.annots.cids = [[1, 2, 3, 4, 5, 5, 5, 5, 5], [6, 4], []]


The JSON Spec
-------------

A COCO file is a json file that follows a particular spec. It is used for
storing computer vision datasets: namely images, categories, and annotations.
Images have an id and a file name, which holds a relative or absolute path to
the image data. Images can also have auxiliary files (e.g. for depth masks,
infrared, or motion). A category has an id, a name, and an optional
supercategory.  Annotations always have an id, an image-id, and a bounding box.
Usually they also contain a category-id. Sometimes they contain keypoints,
segmentations. The dataset can also store videos, in which case images should
have video_id field, and annotations should have a track_id field.

An implementation and extension of the original MS-COCO API [1]_.

Dataset Spec:

An informal description of the spec is written here:

.. code-block:: 

    # All object categories are defined here.
    category = {
        'id': int,
        'name': str,  # unique name of the category
        'supercategory': str,   # parent category name
    }

    # Videos are used to manage collections or sequences of images.
    # Frames do not necesarilly have to be aligned or uniform time steps
    video = {
        'id': int,
        'name': str,  # a unique name for this video.

        'width': int  # the base width of this video (all associated images must have this width)
        'height': int  # the base height of this video (all associated images must have this height)

        # In the future this may be extended to allow pointing to video files
    }

    # Specifies how to find sensor data of a particular scene at a particular
    # time. This is usually paths to rgb images, but auxiliary information
    # can be used to specify multiple bands / etc...

    # NOTE: in the future we will transition from calling these auxiliary items
    # to calling these asset items. As such the key will change from
    # "auxiliary" to "asset". The API will be updated to maintain backwards
    # compatibility while this transition occurs.

    image = {
        'id': int,

        'name': str,  # an encouraged but optional unique name
        'file_name': str,  # relative path to the "base" image data (optional if auxiliary items are specified)

        'width': int,   # pixel width of "base" image
        'height': int,  # pixel height of "base" image

        'channels': <ChannelSpec>,   # a string encoding of the channels in the main image (optional if auxiliary items are specified)

        'auxiliary': [  # information about any auxiliary channels / bands
            {
                'file_name': str,     # relative path to associated file
                'channels': <ChannelSpec>,   # a string encoding
                'width':     <int>    # pixel width of image asset
                'height':    <int>    # pixel height of image asset
                'warp_aux_to_img': <TransformSpec>,  # tranform from "base" image space to auxiliary/asset space. (identity if unspecified)
                'quantization': <QuantizationSpec>,  # indicates that the underlying data was quantized
            }, ...
        ]

        'video_id': str  # if this image is a frame in a video sequence, this id is shared by all frames in that sequence.
        'timestamp': str | int  # a iso-string timestamp or an integer in flicks.
        'frame_index': int  # ordinal frame index which can be used if timestamp is unknown.
        'warp_img_to_vid': <TransformSpec>  # a transform image space to video space (identity if unspecified), can be used for sensor alignment or video stabilization
    }

    TransformSpec:
        The spec can be anything coercable to a kwimage.Affine object.
        This can be an explicit affine transform matrix like:
            {'type': 'affine': 'matrix': <a-3x3 matrix>},

        But it can also be a concise dict containing one or more of these keys
            {
                'scale': <float|Tuple[float, float]>,
                'offset': <float|Tuple[float, float]>,
                'skew': <float>,
                'theta': <float>,  # radians counter-clock-wise
            }

    ChannelSpec:
        This is a string that describes the channel composition of an image.
        For the purposes of kwcoco, separate different channel names with a
        pipe ('|'). If the spec is not specified, methods may fall back on
        grayscale or rgb processing. There are special string. For instance
        'rgb' will expand into 'r|g|b'. In other applications you can "late
        fuse" inputs by separating them with a "," and "early fuse" by
        separating with a "|". Early fusion returns a solid array/tensor, late
        fusion returns separated arrays/tensors.

    QuantizationSpec:
        This is a dictionary of the form:
            {
                'orig_min': <float>, # min original intensity
                'orig_max': <float>, # min original intensity
                'quant_min': <int>, # min quantized intensity
                'quant_max': <int>, # max quantized intensity
                'nodata': <int|None>,  # integer value to interpret as nan
            }

    # Ground truth is specified as annotations, each belongs to a spatial
    # region in an image. This must reference a subregion of the image in pixel
    # coordinates. Additional non-schma properties can be specified to track
    # location in other coordinate systems. Annotations can be linked over time
    # by specifying track-ids.
    annotation = {
        'id': int,
        'image_id': int,
        'category_id': int,

        'track_id': <int | str | uuid>  # indicates association between annotations across images

        'bbox': [tl_x, tl_y, w, h],  # xywh format)
        'score' : float,
        'prob' : List[float],
        'weight' : float,

        'caption': str,  # a text caption for this annotation
        'keypoints' : <Keypoints | List[int] > # an accepted keypoint format
        'segmentation': <RunLengthEncoding | Polygon | MaskPath | WKT >,  # an accepted segmentation format
    }

    # A dataset bundles a manifest of all aformentioned data into one structure.
    dataset = {
        'categories': [category, ...],
        'videos': [video, ...]
        'images': [image, ...]
        'annotations': [annotation, ...]
        'licenses': [],
        'info': [],
    }

    Polygon:
        A flattned list of xy coordinates.
        [x1, y1, x2, y2, ..., xn, yn]

        or a list of flattned list of xy coordinates if the CCs are disjoint
        [[x1, y1, x2, y2, ..., xn, yn], [x1, y1, ..., xm, ym],]

        Note: the original coco spec does not allow for holes in polygons.

        We also allow a non-standard dictionary encoding of polygons
            {'exterior': [(x1, y1)...],
             'interiors': [[(x1, y1), ...], ...]}

        TODO: Support WTK

    RunLengthEncoding:
        The RLE can be in a special bytes encoding or in a binary array
        encoding. We reuse the original C functions are in [2]_ in
        ``kwimage.structs.Mask`` to provide a convinient way to abstract this
        rather esoteric bytes encoding.

        For pure python implementations see kwimage:
            Converting from an image to RLE can be done via kwimage.run_length_encoding
            Converting from RLE back to an image can be done via:
                kwimage.decode_run_length

            For compatibility with the COCO specs ensure the binary flags
            for these functions are set to true.

    Keypoints:
        Annotation keypoints may also be specified in this non-standard (but
        ultimately more general) way:

        'annotations': [
            {
                'keypoints': [
                    {
                        'xy': <x1, y1>,
                        'visible': <0 or 1 or 2>,
                        'keypoint_category_id': <kp_cid>,
                        'keypoint_category': <kp_name, optional>,  # this can be specified instead of an id
                    }, ...
                ]
            }, ...
        ],
        'keypoint_categories': [{
            'name': <str>,
            'id': <int>,  # an id for this keypoint category
            'supercategory': <kp_name>  # name of coarser parent keypoint class (for hierarchical keypoints)
            'reflection_id': <kp_cid>  # specify only if the keypoint id would be swapped with another keypoint type
        },...
        ]

        In this scheme the "keypoints" property of each annotation (which used
        to be a list of floats) is now specified as a list of dictionaries that
        specify each keypoints location, id, and visibility explicitly. This
        allows for things like non-unique keypoints, partial keypoint
        annotations. This also removes the ordering requirement, which makes it
        simpler to keep track of each keypoints class type.

        We also have a new top-level dictionary to specify all the possible
        keypoint categories.

        TODO: Support WTK

    Auxiliary Channels / Image Assets:
        For multimodal or multispectral images it is possible to specify
        auxiliary channels in an image dictionary as follows:

        {
            'id': int,
            'file_name': str,    # path to the "base" image (may be None)
            'name': str,         # a unique name for the image (must be given if file_name is None)
            'channels': <ChannelSpec>,  # a spec code that indicates the layout of the "base" image channels.
            'auxiliary': [  # information about auxiliary channels
                {
                    'file_name': str,
                    'channels': <ChannelSpec>
                }, ... # can have many auxiliary channels with unique specs
            ]
        }

        Note that specifing a filename / channels for the base image is not
        necessary, and mainly useful for augmenting an existing single-image
        dataset with multimodal information. Typically if an image consists of
        more than one file, all file information should be stored in the
        "auxiliary" or "assets" list.

        NEW DOCS:
            In an MSI use case you should think of the "auxiliary" list as a
            list of single file assets that are composed to make the entire
            image. Your assets might include sensed bands, computed features,
            or quality information. For instance a list of auxiliary items may
            look like this:

            image = {
                "name": "my_msi_image",
                "width": 400,
                "height": 400,

                "video_id": 2,
                "timestamp": "2020-01-1",
                "frame_index": 5,
                "warp_img_to_vid": {"type": "affine", "scale", 1.4},

                "auxiliary": [
                   {"channels": "red|green|blue": "file_name": "rgb.tif", "warp_aux_to_img": {"scale": 1.0}, "height": 400, "width": 400, ...},
                   ...
                   {"channels": "cloudmask": "file_name": "cloudmask.tif", "warp_aux_to_img": {"scale": 4.0}, "height": 100, "width": 100, ...},
                   {"channels": "nir": "file_name": "nir.tif", "warp_aux_to_img": {"scale": 2.0}, "height": 200, "width": 200, ...},
                   {"channels": "swir": "file_name": "swir.tif", "warp_aux_to_img": {"scale": 2.0}, "height": 200, "width": 200, ...},
                   {"channels": "model1_predictions:0.6": "file_name": "model1_preds.tif", "warp_aux_to_img": {"scale": 8.0}, "height": 50, "width": 50, ...},
                   {"channels": "model2_predictions:0.3": "file_name": "model2_preds.tif", "warp_aux_to_img": {"scale": 8.0}, "height": 50, "width": 50, ...},
                ]
            }

            Note that there is no file_name or channels parameter in the image
            object itself. This pattern indicates that image is composed of
            multiple assets. One could indicate that an asset is primary by
            giving its information to the parent image, but for better STAC
            compatibility, all assets for MSI images should simply be listed
            as "auxiliary" items.


    Video Sequences:
        For video sequences, we add the following video level index:

        'videos': [
            { 'id': <int>, 'name': <video_name:str> },
        ]

        Note that the videos might be given as encoded mp4/avi/etc.. files (in
        which case the name should correspond to a path) or as a series of
        frames in which case the images should be used to index the extracted
        frames and information in them.

        Then image dictionaries are augmented as follows:

        {
            'video_id': str  # optional, if this image is a frame in a video sequence, this id is shared by all frames in that sequence.
            'timestamp': int  # optional, timestamp (ideally in flicks), used to identify the timestamp of the frame. Only applicable video inputs.
            'frame_index': int  # optional, ordinal frame index which can be used if timestamp is unknown.
        }

        And annotations are augmented as follows:

        {
            'track_id': <int | str | uuid>  # optional, indicates association between annotations across frames
        }


For a formal description of the spec see the  `kwcoco/coco_schema.json <kwcoco/coco_schema.json>`_.

For more information on the "warp" transforms see `warping_and_spaces <docs/source/warping_and_spaces.rst>`_. 


The CocoDatset API Grouped by Functinoality
-------------------------------------------

The following are grouped attribute/method names of a ``kwcoco.CocoDataset``.
See the in-code documentation for further details.

.. code-block:: python

    {
        'classmethod': [
            'coerce',
            'demo',
            'from_coco_paths',
            'from_data',
            'from_image_paths',
            'random',
        ],
        'slots': [
            'index',
            'hashid',
            'hashid_parts',
            'tag',
            'dataset',
            'bundle_dpath',
            'assets_dpath',
            'cache_dpath',
        ],
        'property': [
            'anns',
            'cats',
            'cid_to_aids',
            'data_fpath',
            'data_root',
            'fpath',
            'gid_to_aids',
            'img_root',
            'imgs',
            'n_annots',
            'n_cats',
            'n_images',
            'n_videos',
            'name_to_cat',
        ],
        'method(via MixinCocoAddRemove)': [
            'add_annotation',
            'add_annotations',
            'add_category',
            'add_image',
            'add_images',
            'add_video',
            'clear_annotations',
            'clear_images',
            'ensure_category',
            'ensure_image',
            'remove_annotation',
            'remove_annotation_keypoints',
            'remove_annotations',
            'remove_categories',
            'remove_images',
            'remove_keypoint_categories',
            'remove_videos',
            'set_annotation_category',
        ],
        'method(via MixinCocoObjects)': [
            'annots',
            'categories',
            'images',
            'videos',
        ],
        'method(via MixinCocoStats)': [
            'basic_stats',
            'boxsize_stats',
            'category_annotation_frequency',
            'category_annotation_type_frequency',
            'conform',
            'extended_stats',
            'find_representative_images',
            'keypoint_annotation_frequency',
            'stats',
            'validate',
        ],
        'method(via MixinCocoAccessors)': [
            'category_graph',
            'delayed_load',
            'get_auxiliary_fpath',
            'get_image_fpath',
            'keypoint_categories',
            'load_annot_sample',
            'load_image',
            'object_categories',
        ],
        'method(via CocoDataset)': [
            'copy',
            'dump',
            'dumps',
            'subset',
            'union',
            'view_sql',
        ],
        'method(via MixinCocoExtras)': [
            'corrupted_images',
            'missing_images',
            'rename_categories',
            'reroot',
        ],
        'method(via MixinCocoDraw)': [
            'draw_image',
            'imread',
            'show_image',
        ],
    }


Converting your RGB data to KWCOCO
----------------------------------

Assuming you have programmatic access to your dataset you can easily convert to
a coco file using process similar to the following code:

.. code-block:: python

    # ASSUME INPUTS 
    # my_classes: a list of category names
    # my_annots: a list of annotation objects with bounding boxes, images, and categories
    # my_images: a list of image files.

    my_images = [
        'image1.png',
        'image2.png',
        'image3.png',
    ]

    my_classes = [
        'spam', 'eggs', 'ham', 'jam'
    ]

    my_annots = [
        {'image': 'image1.png', 'box': {'tl_x':  2, 'tl_y':  3, 'br_x':  5, 'br_y':  7}, 'category': 'spam'},
        {'image': 'image1.png', 'box': {'tl_x': 11, 'tl_y': 13, 'br_x': 17, 'br_y': 19}, 'category': 'spam'},
        {'image': 'image3.png', 'box': {'tl_x': 23, 'tl_y': 29, 'br_x': 31, 'br_y': 37}, 'category': 'eggs'},
        {'image': 'image3.png', 'box': {'tl_x': 41, 'tl_y': 43, 'br_x': 47, 'br_y': 53}, 'category': 'spam'},
        {'image': 'image3.png', 'box': {'tl_x': 59, 'tl_y': 61, 'br_x': 67, 'br_y': 71}, 'category': 'jam'},
        {'image': 'image3.png', 'box': {'tl_x': 73, 'tl_y': 79, 'br_x': 83, 'br_y': 89}, 'category': 'spam'},
    ]

    # The above is just an example input, it is left as an exercise for the
    # reader to translate that to your own dataset.

    import kwcoco
    import kwimage

    # A kwcoco.CocoDataset is simply an object that manages an underlying
    # `dataset` json object. It contains methods to dynamically, add, remove,
    # and modify these data structures, efficient lookup tables, and many more
    # conveniences when working and playing with vision datasets.
    my_dset = kwcoco.CocoDataset()

    for catname in my_classes:
        my_dset.add_category(name=catname)

    for image_path in my_images:
        my_dset.add_image(file_name=image_path)

    for annot in my_annots:
        # The index property provides fast lookups into the json data structure
        cat = my_dset.index.name_to_cat[annot['category']]
        img = my_dset.index.file_name_to_img[annot['image']]
        # One quirk of the coco format is you need to be aware that
        # boxes are in <top-left-x, top-left-y, width-w, height-h> format.
        box = annot['box']
        # Use kwimage.Boxes to preform quick, reliable, and readable
        # conversions between common bounding box formats.
        tlbr = [box['tl_x'], box['tl_y'], box['br_x'], box['br_y']]
        xywh = kwimage.Boxes([tlbr], 'tlbr').toformat('xywh').data[0].tolist()
        my_dset.add_annotation(bbox=xywh, image_id=img['id'], category_id=cat['id'])

    # Dump the underlying json `dataset` object to a file
    my_dset.fpath = 'my-converted-dataset.mscoco.json'
    my_dset.dump(my_dset.fpath, newlines=True)

    # Dump the underlying json `dataset` object to a string
    print(my_dset.dumps(newlines=True))


KWCOCO Spaces
-------------

There are 3 spaces that a user of kwcoco may need to be concerned with
depending on their dataset: (1) video space, (2) image space, and (3)
asset/auxiliary space.

Videos can contain multiple images, images can contain multiple asset/auxiliary
items, and kwcoco needs to know about any transformation that relates between
different levels in this heirarchy.

1. Video space - In a sequence of images, each individual image might be at a
   different resolution, or misaligned with other images in the sequence.
   This space is only important when working with images in "video" sequences.

2. Image space - If an image contains multiple auxiliary / asset items, this is
   the space that they are all re sampled to at the "image level". Note all
   annotations on images should always be given in image space by convention.

1. Auxiliary / Asset Space - This is the native space/resolution of the raster
   image data that lives on disk that KWCOCO points to. When an image consists of
   only a single asset. This space is only important when an image contains
   multiple files at different resolutions.


When an item is registered in a space. (i.e. you register a video, image, or
auxiliary/asset item), kwcoco will benefit from knowing (1) the width/height of
the object in it's own space, and any transformation from that object to it's
parent space --- i.e. an auxiliary/asset item needs to know how to be
transformed into image space, and an image needs to know how to be transformed
into video space (if applicable). This warping can be as simple as a scale
factor or as complex as a full homography matrix (and we may generalize beyond
this), and is specified via the `TransformSpec`. When this transform is
unspecified it is assumed to be the identity transform, so for pre-aligned
datasets, the user does not need to worry about the differentiation between
spaces and simply work in "image space".


Converting your Multispectral Multiresolution Data to KWCOCO
------------------------------------------------------------

KWCOCO has the ability to work with multispectral images. More generally, a
KWCOCO image can contain any number of "raster assets". The motivating use case
is multispectral imagery, but this also incorporates more general use cases
where rasters can represent metadata from a depth sensor, or stereo images,
etc.

Put plainly, a KWCOCO image can consist of multiple image files, and each of
those image file can have any number of channels. Furthermore, these image
files do not need to have the same resolution. However, the channels
within a single image currently must be unique.

Because images can be in different resolutions, we need to bring up the topic
of "KWCOCO spaces". For full info on this, see the discussion on "KWCOCO
spaces", but briefly, there are 3 spaces that a user of kwcoco needs to be
concerned with: (1) video space, (2) image space, and (3) asset/auxiliary
space, and KWCOCO will want to know how. 

As a simple example, lets assume you have a dataset containing sequences of RGB
images, corresponding infrared images, depth estimations, and optical flow
estimations. The infrared images are stored in half-resolution of the RGB
images, but the depth and flow data is at the same resolution as the RGB data.
The RGB images have 3 channels the flow images have 2 channels, and depth and
ir have 1 channel.


If our images on disk look like:


.. code-block:: 

    - video1/vid1_frame1_rgb.tif
    - video1/vid1_frame1_ir.tif
    - video1/vid1_frame1_depth.tif
    - video1/vid1_frame1_flow.tif
    - video1/vid1_frame2_rgb.tif
    - video1/vid1_frame2_ir.tif
    - video1/vid1_frame2_depth.tif
    - video1/vid1_frame2_flow.tif
    - video1/vid1_frame3_rgb.tif
    - video1/vid1_frame3_ir.tif
    - video1/vid1_frame3_depth.tif
    - video1/vid1_frame3_flow.tif


We can add them to a custom kwcoco file using the following code.

First, lets's actually make dummy data for those images on disk.

.. code-block:: python

   import numpy as np
   import kwimage
   import ubelt as ub
   num_frames = 3
   num_videos = 1
   width, height = 64, 64

   bundle_dpath = ub.Path('demo_bundle').ensuredir()
   for vidid in range(1, num_videos + 1):
       vid_dpath = (bundle_dpath / f'video{vidid}').ensuredir()
       for frame_num in range(1, num_frames + 1):
           kwimage.imwrite(vid_dpath / f'vid{vidid}_frame{frame_num}_rgb.tif', np.random.rand(height, width, 3))
           kwimage.imwrite(vid_dpath / f'vid{vidid}_frame{frame_num}_ir.tif', np.random.rand(height // 2, width // 2))
           kwimage.imwrite(vid_dpath / f'vid{vidid}_frame{frame_num}_depth.tif', np.random.rand(height, width, 1))
           kwimage.imwrite(vid_dpath / f'vid{vidid}_frame{frame_num}_flow.tif', np.random.rand(height, width, 2))


Now lets create a kwcoco dataset to register them. We use the channel spec to denote what the channels are.

.. code-block:: python

    import ubelt as ub
    import os
    bundle_dpath = ub.Path('demo_bundle')

    import kwcoco
    import kwimage
    dset = kwcoco.CocoDataset()
    dset.fpath = bundle_dpath / 'data.kwcoco.json'

    # We will define a map from our suffix codes in the filename to
    # kwcoco channel specs that indicate the number of channels
    channel_spec_mapping = {
       'rgb': 'red|green|blue',  # rgb is 3 channels
       'flow': 'fx|fy',  # flow is 2 channels
       'ir': 'ir',
       'depth': 'depth',
    }

    for video_dpath in bundle_dpath.glob('video*'):
       # Add a video and give it a name.
       vidid = dset.add_video(name=video_dpath.name)

       # Parse out information that we need from the filenames. 
       # Lots of different ways to do this depending on the use case.
       assets = []
       for fpath in video_dpath.glob('*.tif'):
           _, frame_part, chan_part = fpath.stem.split('_')
           frame_index = int(frame_part[5:])
           assets.append({
               'frame_num': frame_index,
               'channels': channel_spec_mapping[chan_part],
               'fpath': fpath,
           })

       # Group all data from the same frame together.
       frame_to_group = ub.group_items(assets, lambda x: x['frame_num'])
       for frame_index, group in frame_to_group.items():
           # Let us lookup data by channels
           chan_to_item = {item['channels']: item for item in group}
           # Grab the RGB data as it will be our "primary" asset
           rgbdata = chan_to_item['red|green|blue']

           # Use the prefix for the image name
           name = rgbdata['fpath'].stem.split('_rgb')[0]

           height, width = kwimage.load_image_shape(rgbdata['fpath'])[0:2]

           # First add the base image. We will add this image as
           # without a file_name because all of its data will be stored 
           # in its auxiliary list. We will assume all images in the
           # video are aligned, so we set `warp_img_to_vid` to be the
           # identity matrix.
           gid = dset.add_image(
               name=name, width=width, height=height,
               warp_img_to_vid=kwimage.Affine.eye().concise())

           # We could have constructed the auxiliary item dictionaries
           # explicitly and added them in the previous step, but we 
           # will use the CocoImage api to do this instead.
           coco_img = dset.coco_image(gid)

           for item in group:
               fpath = item['fpath']
               height, width = kwimage.load_image_shape(fpath)[0:2]
               file_name = os.fspath(fpath.relative_to(bundle_dpath))
               coco_img.add_auxiliary_item(
                   file_name=file_name, channels=item['channels'], width=width,
                   height=height)

    # We can always double check we did not make errors using kwcoco validate
    dset.validate()


Now we have a multispectral multi-resolution dataset. You can load specific
subsets of channels (in specific subregions is your data is stored in the COG
or a RAW format) using the delayed load interface.

.. code-block:: python


    # Get a coco image.
    gid = 1
    coco_img = dset.coco_image(gid)

    # Tell delayed load what channels we want. We can 
    # also specify which "space" we want to load it in.
    # Note: that when specifying channels from multiple asset items
    # it is not possible to sample in the the auxiliary / asset space 
    # so only image and video are allowed there.
    delayed_img = coco_img.delay('fx|depth|red', space='image')

    # We finalize the data to load it
    imdata = delayed_img.finalize()

    # We can show it if we want, but it's just random data.
    import kwplot
    kwplot.autompl()
    kwplot.imshow(imdata)


Somewhat more interesting is to use the KWCOCO demodata. We can see here that
videos can contain multiple images at different resolutions and each image can
contain different number of channels.

.. code-block:: python

    import kwcoco
    import kwarray
    import kwimage
    dset = kwcoco.CocoDataset.demo('vidshapes8-msi-multisensor')

    gid = 1
    coco_img = dset.coco_image(gid)

    # Randomly select 3 channels to use
    avail_channels = coco_img.channels.fuse().as_list()
    channels = '|'.join(kwarray.shuffle(avail_channels)[0:3])
    print('channels = {!r}'.format(channels))

    delayed_img = coco_img.delay(channels, space='video')

    imdata = delayed_img.finalize()

    # Depending on the sensor intensity might be out of standard ranges,
    # we can use kwimage to robustly normalize for this. This lets
    # us visualize data with false color.
    canvas = kwimage.normalize_intensity(imdata, axis=2)
    canvas = np.ascontiguousarray(canvas)

    # We can draw the annotations on the image, but be cognizant of the spaces.
    # Annotations are always in "image" space, so if we loaded in "video" space
    # then we need to warp to that.
    imgspace_dets = dset.annots(gid=gid).detections
    vidspace_dets = imgspace_dets.warp(coco_img.warp_vid_from_img)

    canvas = vidspace_dets.draw_on(canvas)

    import kwplot
    kwplot.autompl()
    kwplot.imshow(canvas)


The result of the above code is (note the data is random, so it may differ on your machine):

.. image:: https://i.imgur.com/hrFFwII.png
   :height: 100px
   :align: left


Key notes to takeaway:

* KWCOCO can register many assets at different resolutions, register groups depicting the same scene at a particular time into an "image", and then groups of images can be grouped into "videos".

* Annotations are always specified in image space

* Channel code within a single image should never be duplicated.


The KWCOCO Channel Specification
--------------------------------

See the documentation for ``kwcoco/channel_spec.py`` for more details.



.. [1] http://cocodataset.org/#format-data

.. [2] https://github.com/nightrome/cocostuffapi/blob/master/PythonAPI/pycocotools/mask.py
      

.. |Pypi| image:: https://img.shields.io/pypi/v/kwcoco.svg
   :target: https://pypi.python.org/pypi/kwcoco

.. |Downloads| image:: https://img.shields.io/pypi/dm/kwcoco.svg
   :target: https://pypistats.org/packages/kwcoco

.. |ReadTheDocs| image:: https://readthedocs.org/projects/kwcoco/badge/?version=release
    :target: https://kwcoco.readthedocs.io/en/release/

.. # See: https://ci.appveyor.com/project/jon.crall/kwcoco/settings/badges
.. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/py3s2d6tyfjc8lm3/branch/master?svg=true
   :target: https://ci.appveyor.com/project/jon.crall/kwcoco/branch/master

.. |GitlabCIPipeline| image:: https://gitlab.kitware.com/computer-vision/kwcoco/badges/master/pipeline.svg
   :target: https://gitlab.kitware.com/computer-vision/kwcoco/-/jobs

.. |GitlabCICoverage| image:: https://gitlab.kitware.com/computer-vision/kwcoco/badges/master/coverage.svg
    :target: https://gitlab.kitware.com/computer-vision/kwcoco/commits/master
