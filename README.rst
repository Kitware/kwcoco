The Kitware COCO Module
=======================

.. # TODO Get CI services running on gitlab 

|GitlabCIPipeline| |GitlabCICoverage| |Appveyor| |Pypi| |Downloads| |ReadTheDocs|

The Kitware COCO module defines a variant of the Microsoft COCO format,
originally developed for the "collected images in context" object detection
challenge. We are backwards compatible with the original module, but we also
have improved implementations in several places, including segmentations and
keypoints.


The main data structure in this model is largely based on the implementation in
https://github.com/cocodataset/cocoapi It uses the same efficient core indexing
data structures, but in our implementation the indexing can be optionally
turned off, functions are silent by default (with the exception of long running
processes, which optionally show progress by default). We support helper
functions that add and remove images, categories, and annotations. 

We do not reimplement the scoring code fro pycocotools in this module. Instead
that functionality currently lives in netharn:
https://gitlab.kitware.com/computer-vision/netharn/-/blob/master/netharn/metrics/detect_metrics.py
We may move that file either this repo or kwannot in the future. This module is
more focused on efficient access and modification of a COCO dataset.


The kwcoco CLI
--------------

After installing kwcoco, you will also have the ``kwcoco`` command line tool. 
This uses a ``scriptconfig`` / ``argparse`` CLI interface. Running ``kwcoco
--help`` should provide a good starting point.

.. code:: 

    usage: kwcoco [-h] {stats,union,split,show,toydata} ...

    The Kitware COCO CLI

    positional arguments:
      {stats,union,split,show,toydata}
                            specify a command to run
        stats               Compute summary statistics about a COCO dataset
        union               Combine multiple COCO datasets into a single merged dataset.
        split               Split a single COCO dataset into two sub-datasets.
        show                Visualize a COCO image
        toydata             Create COCO toydata

    optional arguments:
      -h, --help            show this help message and exit


This should help you inspect (via stats and show), combine (via union), and
make training splits (via split) using the command line. Also ships with
toydata, which generates a COCO file you can use for testing.


The CocoDataset object
----------------------

The ``kwcoco.CocoDataset`` class is capable of dynamic addition and removal of
categories, images, and annotations. Has better support for keypoints and
segmentation formats than the original COCO format. Despite being written in
Python, this data structure is reasonably efficient.


.. code:: python

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
the image data. Images can also have auxillary files (e.g. for depth masks,
infared, or motion). A category has an id, a name, and an optional
supercategory.  Annotations always have an id, an image-id, and a bounding box.
Usually they also contain a category-id. Sometimes they contain keypoints,
segmentations. 

An implementation and extension of the original MS-COCO API [1]_.

Extends the format to also include line annotations.

Dataset Spec:

.. code:: 

    dataset = {
        # these are object level categories
        'categories': [
            {
                'id': <int:category_id>,
                'name': <str:>,
                'supercategory': str  # optional

                # Note: this is the original way to specify keypoint
                # categories, but our implementation supports a more general
                # alternative schema
                "keypoints": [kpname_1, ..., kpname_K], # length <k> array of keypoint names
                "skeleton": [(kx_a1, kx_b1), ..., (kx_aE, kx_bE)], # list of edge pairs (of keypoint indices), defining connectivity of keypoints.
            },
            ...
        ],
        'images': [
            {
                'id': int, 'file_name': str
            },
            ...
        ],
        'annotations': [
            {
                'id': int,
                'image_id': int,
                'category_id': int,
                'bbox': [tl_x, tl_y, w, h],  # optional (xywh format)
                "score" : float,
                "caption": str,  # an optional text caption for this annotation
                "iscrowd" : <0 or 1>,  # denotes if the annotation covers a single object (0) or multiple objects (1)
                "keypoints" : [x1,y1,v1,...,xk,yk,vk], # or new dict-based format
                'segmentation': <RunLengthEncoding | Polygon>,  # formats are defined bellow
            },
            ...
        ],
        'licenses': [],
        'info': [],
    }

    Polygon:
        A flattned list of xy coordinates.
        [x1, y1, x2, y2, ..., xn, yn]

        or a list of flattned list of xy coordinates if the CCs are disjoint
        [[x1, y1, x2, y2, ..., xn, yn], [x1, y1, ..., xm, ym],]

        Note: the original COCO spec does not allow for holes in polygons.

        (PENDING) We also allow a non-standard dictionary encoding of polygons
            {'exterior': [(x1, y1)...],
             'interiors': [[(x1, y1), ...], ...]}

    RunLengthEncoding:
        The RLE can be in a special bytes encoding or in a binary array
        encoding. We reuse the original C functions are in [2]_ in
        `kwimage.structs.Mask` to provide a convinient way to abstract this
        rather esoteric bytes encoding.

        For pure python implementations see kwimage:
            Converting from an image to RLE can be done via kwimage.run_length_encoding
            Converting from RLE back to an image can be done via:
                kwimage.decode_run_length

            For compatibility with the COCO specs ensure the binary flags
            for these functions are set to true.

    Keypoints:
        (PENDING)
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

    Auxillary Channels:
        For multimodal or multispectral images it is possible to specify
        auxillary channels in an image dictionary as follows:

        {
            'id': int, 'file_name': str
            'channels': <spec>,  # a spec code that indicates the layout of these channels.
            'auxillary': [  # information about auxillary channels
                {
                    'file_name':
                    'channels': <spec>
                }, ... # can have many auxillary channels with unique specs
            ]
        }


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

.. |GitlabCICoverage| image:: https://gitlab.kitware.com/computer-vision/kwcoco/badges/master/coverage.svg?job=coverage
    :target: https://gitlab.kitware.com/computer-vision/kwcoco/commits/master


