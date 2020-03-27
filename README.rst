The Kitware COCO Module
=======================

.. # TODO Get CI services running on gitlab 

|GitlabCIPipeline| |GitlabCICoverage| |Appveyor| |Pypi| |Downloads| |ReadTheDocs|

The Kitware COCO module defines a variant of the Microsoft COCO format,
originally developed for the "collected images in context" object detection
challenge. We are backwards compatible with the original module, but we also
have improved implementations in seval places, including segmentations and
keypoints.

The ``kwcoco.coco_dataset.CocoDataset`` class is capable of dynamic
addition and removal of categories, images, and annotations. Has better support
for keypoints and segmentation formats than the original COCO format. Despite
being written in Python, this data structure is reasonably efficient.

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
        union               Combine multiple coco datasets into a single merged dataset.
        split               Split a single coco dataset into two sub-datasets.
        show                Visualize a COCO image
        toydata             Create coco toydata

    optional arguments:
      -h, --help            show this help message and exit


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

        Note: the original coco spec does not allow for holes in polygons.

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


