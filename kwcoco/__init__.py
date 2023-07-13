"""
The Kitware COCO module defines a variant of the Microsoft COCO format,
originally developed for the "collected images in context" object detection
challenge. We are backwards compatible with the original module, but we also
have improved implementations in several places, including segmentations,
keypoints, annotation tracks, multi-spectral images, and videos (which
represents a generic sequence of images).

A kwcoco file is a "manifest" that serves as a single reference that points to
all images, categories, and annotations in a computer vision dataset. Thus,
when applying an algorithm to a dataset, it is sufficient to have the algorithm
take one dataset parameter: the path to the kwcoco file.  Generally a kwcoco
file will live in a "bundle" directory along with the data that it references,
and paths in the kwcoco file will be relative to the location of the kwcoco
file itself.

The main data structure in this model is largely based on the implementation in
https://github.com/cocodataset/cocoapi It uses the same efficient core indexing
data structures, but in our implementation the indexing can be optionally
turned off, functions are silent by default (with the exception of long running
processes, which optionally show progress by default). We support helper
functions that add and remove images, categories, and annotations.


The :class:`kwcoco.CocoDataset` class is capable of dynamic addition and removal
of categories, images, and annotations. Has better support for keypoints and
segmentation formats than the original COCO format. Despite being written in
Python, this data structure is reasonably efficient.


.. code:: python

        >>> import kwcoco
        >>> import json
        >>> # Create demo data
        >>> demo = kwcoco.CocoDataset.demo()
        >>> # Reroot can switch between absolute / relative-paths
        >>> demo.reroot(absolute=True)
        >>> # could also use demo.dump / demo.dumps, but this is more explicit
        >>> text = json.dumps(demo.dataset)
        >>> with open('demo.json', 'w') as file:
        >>>    file.write(text)

        >>> # Read from disk
        >>> self = kwcoco.CocoDataset('demo.json')

        >>> # Add data
        >>> cid = self.add_category('Cat')
        >>> gid = self.add_image('new-img.jpg')
        >>> aid = self.add_annotation(image_id=gid, category_id=cid, bbox=[0, 0, 100, 100])

        >>> # Remove data
        >>> self.remove_annotations([aid])
        >>> self.remove_images([gid])
        >>> self.remove_categories([cid])

        >>> # Look at data
        >>> import ubelt as ub
        >>> print(ub.urepr(self.basic_stats(), nl=1))
        >>> print(ub.urepr(self.extended_stats(), nl=2))
        >>> print(ub.urepr(self.boxsize_stats(), nl=3))
        >>> print(ub.urepr(self.category_annotation_frequency()))


        >>> # Inspect data
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> self.show_image(gid=1)

        >>> # Access single-item data via imgs, cats, anns
        >>> cid = 1
        >>> self.cats[cid]
        {'id': 1, 'name': 'astronaut', 'supercategory': 'human'}

        >>> gid = 1
        >>> self.imgs[gid]
        {'id': 1, 'file_name': '...astro.png', 'url': 'https://i.imgur.com/KXhKM72.png'}

        >>> aid = 3
        >>> self.anns[aid]
        {'id': 3, 'image_id': 1, 'category_id': 3, 'line': [326, 369, 500, 500]}

        >>> # Access multi-item data via the annots and images helper objects
        >>> aids = self.index.gid_to_aids[2]
        >>> annots = self.annots(aids)

        >>> print('annots = {}'.format(ub.urepr(annots, nl=1, sv=1)))
        annots = <Annots(num=2)>

        >>> annots.lookup('category_id')
        [6, 4]

        >>> annots.lookup('bbox')
        [[37, 6, 230, 240], [124, 96, 45, 18]]

        >>> # built in conversions to efficient kwimage array DataStructures
        >>> print(ub.urepr(annots.detections.data, sv=1))
        {
            'boxes': <Boxes(xywh,
                         array([[ 37.,   6., 230., 240.],
                                [124.,  96.,  45.,  18.]], dtype=float32))>,
            'class_idxs': [5, 3],
            'keypoints': <PointsList(n=2)>,
            'segmentations': <PolygonList(n=2)>,
        }

        >>> gids = list(self.imgs.keys())
        >>> images = self.images(gids)
        >>> print('images = {}'.format(ub.urepr(images, nl=1, sv=1)))
        images = <Images(num=3)>

        >>> images.lookup('file_name')
        ['...astro.png', '...carl.png', '...stars.png']

        >>> print('images.annots = {}'.format(images.annots))
        images.annots = <AnnotGroups(n=3, m=3.7, s=3.9)>

        >>> print('images.annots.cids = {!r}'.format(images.annots.cids))
        images.annots.cids = [[1, 2, 3, 4, 5, 5, 5, 5, 5], [6, 4], []]


CocoDataset API
###############

The following is a logical grouping of the public kwcoco.CocoDataset API attributes and methods. See the in-code documentation for further details.

CocoDataset classmethods (via MixinCocoExtras)
**********************************************

 * :func:`kwcoco.CocoDataset.coerce<kwcoco.coco_dataset.MixinCocoExtras.coerce>` - Attempt to transform the input into the intended CocoDataset.
 * :func:`kwcoco.CocoDataset.demo<kwcoco.coco_dataset.MixinCocoExtras.demo>` - Create a toy coco dataset for testing and demo puposes
 * :func:`kwcoco.CocoDataset.random<kwcoco.coco_dataset.MixinCocoExtras.random>` - Creates a random CocoDataset according to distribution parameters

CocoDataset classmethods (via CocoDataset)
******************************************

 * :func:`kwcoco.CocoDataset.from_coco_paths<kwcoco.coco_dataset.CocoDataset.from_coco_paths>` - Constructor from multiple coco file paths.
 * :func:`kwcoco.CocoDataset.from_data<kwcoco.coco_dataset.CocoDataset.from_data>` - Constructor from a json dictionary
 * :func:`kwcoco.CocoDataset.from_image_paths<kwcoco.coco_dataset.CocoDataset.from_image_paths>` - Constructor from a list of images paths.

CocoDataset slots
*****************

 * :attr:`kwcoco.CocoDataset.index<kwcoco.coco_dataset.CocoDataset.index>` - an efficient lookup index into the coco data structure. The index defines its own attributes like ``anns``, ``cats``, ``imgs``, ``gid_to_aids``, ``file_name_to_img``, etc. See :class:`CocoIndex` for more details on which attributes are available.
 * :attr:`kwcoco.CocoDataset.hashid<kwcoco.coco_dataset.CocoDataset.hashid>` - If computed, this will be a hash uniquely identifing the dataset.  To ensure this is computed see  :func:`kwcoco.coco_dataset.MixinCocoExtras._build_hashid`.
 * :attr:`kwcoco.CocoDataset.hashid_parts<kwcoco.coco_dataset.CocoDataset.hashid_parts>` -
 * :attr:`kwcoco.CocoDataset.tag<kwcoco.coco_dataset.CocoDataset.tag>` - A tag indicating the name of the dataset.
 * :attr:`kwcoco.CocoDataset.dataset<kwcoco.coco_dataset.CocoDataset.dataset>` - raw json data structure. This is the base dictionary that contains {'annotations': List, 'images': List, 'categories': List}
 * :attr:`kwcoco.CocoDataset.bundle_dpath<kwcoco.coco_dataset.CocoDataset.bundle_dpath>` - If known, this is the root path that all image file names are relative to. This can also be manually overwritten by the user.
 * :attr:`kwcoco.CocoDataset.assets_dpath<kwcoco.coco_dataset.CocoDataset.assets_dpath>` -
 * :attr:`kwcoco.CocoDataset.cache_dpath<kwcoco.coco_dataset.CocoDataset.cache_dpath>` -

CocoDataset properties
**********************

 * :attr:`kwcoco.CocoDataset.anns<kwcoco.coco_dataset.CocoDataset.anns>` -
 * :attr:`kwcoco.CocoDataset.cats<kwcoco.coco_dataset.CocoDataset.cats>` -
 * :attr:`kwcoco.CocoDataset.cid_to_aids<kwcoco.coco_dataset.CocoDataset.cid_to_aids>` -
 * :attr:`kwcoco.CocoDataset.data_fpath<kwcoco.coco_dataset.CocoDataset.data_fpath>` -
 * :attr:`kwcoco.CocoDataset.data_root<kwcoco.coco_dataset.CocoDataset.data_root>` -
 * :attr:`kwcoco.CocoDataset.fpath<kwcoco.coco_dataset.CocoDataset.fpath>` - if known, this stores the filepath the dataset was loaded from
 * :attr:`kwcoco.CocoDataset.gid_to_aids<kwcoco.coco_dataset.CocoDataset.gid_to_aids>` -
 * :attr:`kwcoco.CocoDataset.img_root<kwcoco.coco_dataset.CocoDataset.img_root>` -
 * :attr:`kwcoco.CocoDataset.imgs<kwcoco.coco_dataset.CocoDataset.imgs>` -
 * :attr:`kwcoco.CocoDataset.n_annots<kwcoco.coco_dataset.CocoDataset.n_annots>` -
 * :attr:`kwcoco.CocoDataset.n_cats<kwcoco.coco_dataset.CocoDataset.n_cats>` -
 * :attr:`kwcoco.CocoDataset.n_images<kwcoco.coco_dataset.CocoDataset.n_images>` -
 * :attr:`kwcoco.CocoDataset.n_videos<kwcoco.coco_dataset.CocoDataset.n_videos>` -
 * :attr:`kwcoco.CocoDataset.name_to_cat<kwcoco.coco_dataset.CocoDataset.name_to_cat>` -

CocoDataset methods (via MixinCocoAddRemove)
********************************************

 * :func:`kwcoco.CocoDataset.add_annotation<kwcoco.coco_dataset.MixinCocoAddRemove.add_annotation>` - Add an annotation to the dataset (dynamically updates the index)
 * :func:`kwcoco.CocoDataset.add_annotations<kwcoco.coco_dataset.MixinCocoAddRemove.add_annotations>` - Faster less-safe multi-item alternative to add_annotation.
 * :func:`kwcoco.CocoDataset.add_category<kwcoco.coco_dataset.MixinCocoAddRemove.add_category>` - Adds a category
 * :func:`kwcoco.CocoDataset.add_image<kwcoco.coco_dataset.MixinCocoAddRemove.add_image>` - Add an image to the dataset (dynamically updates the index)
 * :func:`kwcoco.CocoDataset.add_images<kwcoco.coco_dataset.MixinCocoAddRemove.add_images>` - Faster less-safe multi-item alternative
 * :func:`kwcoco.CocoDataset.add_video<kwcoco.coco_dataset.MixinCocoAddRemove.add_video>` - Add a video to the dataset (dynamically updates the index)
 * :func:`kwcoco.CocoDataset.clear_annotations<kwcoco.coco_dataset.MixinCocoAddRemove.clear_annotations>` - Removes all annotations (but not images and categories)
 * :func:`kwcoco.CocoDataset.clear_images<kwcoco.coco_dataset.MixinCocoAddRemove.clear_images>` - Removes all images and annotations (but not categories)
 * :func:`kwcoco.CocoDataset.ensure_category<kwcoco.coco_dataset.MixinCocoAddRemove.ensure_category>` - Like :func:`add_category`, but returns the existing category id if it already exists instead of failing. In this case all metadata is ignored.
 * :func:`kwcoco.CocoDataset.ensure_image<kwcoco.coco_dataset.MixinCocoAddRemove.ensure_image>` - Like :func:`add_image`,, but returns the existing image id if it already exists instead of failing. In this case all metadata is ignored.
 * :func:`kwcoco.CocoDataset.remove_annotation<kwcoco.coco_dataset.MixinCocoAddRemove.remove_annotation>` - Remove a single annotation from the dataset
 * :func:`kwcoco.CocoDataset.remove_annotation_keypoints<kwcoco.coco_dataset.MixinCocoAddRemove.remove_annotation_keypoints>` - Removes all keypoints with a particular category
 * :func:`kwcoco.CocoDataset.remove_annotations<kwcoco.coco_dataset.MixinCocoAddRemove.remove_annotations>` - Remove multiple annotations from the dataset.
 * :func:`kwcoco.CocoDataset.remove_categories<kwcoco.coco_dataset.MixinCocoAddRemove.remove_categories>` - Remove categories and all annotations in those categories. Currently does not change any hierarchy information
 * :func:`kwcoco.CocoDataset.remove_images<kwcoco.coco_dataset.MixinCocoAddRemove.remove_images>` - Remove images and any annotations contained by them
 * :func:`kwcoco.CocoDataset.remove_keypoint_categories<kwcoco.coco_dataset.MixinCocoAddRemove.remove_keypoint_categories>` - Removes all keypoints of a particular category as well as all annotation keypoints with those ids.
 * :func:`kwcoco.CocoDataset.remove_videos<kwcoco.coco_dataset.MixinCocoAddRemove.remove_videos>` - Remove videos and any images / annotations contained by them
 * :func:`kwcoco.CocoDataset.set_annotation_category<kwcoco.coco_dataset.MixinCocoAddRemove.set_annotation_category>` - Sets the category of a single annotation

CocoDataset methods (via MixinCocoObjects)
******************************************

 * :func:`kwcoco.CocoDataset.annots<kwcoco.coco_dataset.MixinCocoObjects.annots>` - Return vectorized annotation objects
 * :func:`kwcoco.CocoDataset.categories<kwcoco.coco_dataset.MixinCocoObjects.categories>` - Return vectorized category objects
 * :func:`kwcoco.CocoDataset.images<kwcoco.coco_dataset.MixinCocoObjects.images>` - Return vectorized image objects
 * :func:`kwcoco.CocoDataset.videos<kwcoco.coco_dataset.MixinCocoObjects.videos>` - Return vectorized video objects

CocoDataset methods (via MixinCocoStats)
****************************************

 * :func:`kwcoco.CocoDataset.basic_stats<kwcoco.coco_dataset.MixinCocoStats.basic_stats>` - Reports number of images, annotations, and categories.
 * :func:`kwcoco.CocoDataset.boxsize_stats<kwcoco.coco_dataset.MixinCocoStats.boxsize_stats>` - Compute statistics about bounding box sizes.
 * :func:`kwcoco.CocoDataset.category_annotation_frequency<kwcoco.coco_dataset.MixinCocoStats.category_annotation_frequency>` - Reports the number of annotations of each category
 * :func:`kwcoco.CocoDataset.category_annotation_type_frequency<kwcoco.coco_dataset.MixinCocoStats.category_annotation_type_frequency>` - Reports the number of annotations of each type for each category
 * :func:`kwcoco.CocoDataset.conform<kwcoco.coco_dataset.MixinCocoStats.conform>` - Make the COCO file conform a stricter spec, infers attibutes where possible.
 * :func:`kwcoco.CocoDataset.extended_stats<kwcoco.coco_dataset.MixinCocoStats.extended_stats>` - Reports number of images, annotations, and categories.
 * :func:`kwcoco.CocoDataset.find_representative_images<kwcoco.coco_dataset.MixinCocoStats.find_representative_images>` - Find images that have a wide array of categories. Attempt to find the fewest images that cover all categories using images that contain both a large and small number of annotations.
 * :func:`kwcoco.CocoDataset.keypoint_annotation_frequency<kwcoco.coco_dataset.MixinCocoStats.keypoint_annotation_frequency>` -
 * :func:`kwcoco.CocoDataset.stats<kwcoco.coco_dataset.MixinCocoStats.stats>` - This function corresponds to :mod:`kwcoco.cli.coco_stats`.
 * :func:`kwcoco.CocoDataset.validate<kwcoco.coco_dataset.MixinCocoStats.validate>` - Performs checks on this coco dataset.

CocoDataset methods (via MixinCocoAccessors)
********************************************

 * :func:`kwcoco.CocoDataset.category_graph<kwcoco.coco_dataset.MixinCocoAccessors.category_graph>` - Construct a networkx category hierarchy
 * :func:`kwcoco.CocoDataset.delayed_load<kwcoco.coco_dataset.MixinCocoAccessors.delayed_load>` - Experimental method
 * :func:`kwcoco.CocoDataset.get_auxiliary_fpath<kwcoco.coco_dataset.MixinCocoAccessors.get_auxiliary_fpath>` - Returns the full path to auxiliary data for an image
 * :func:`kwcoco.CocoDataset.get_image_fpath<kwcoco.coco_dataset.MixinCocoAccessors.get_image_fpath>` - Returns the full path to the image
 * :func:`kwcoco.CocoDataset.keypoint_categories<kwcoco.coco_dataset.MixinCocoAccessors.keypoint_categories>` - Construct a consistent CategoryTree representation of keypoint classes
 * :func:`kwcoco.CocoDataset.load_annot_sample<kwcoco.coco_dataset.MixinCocoAccessors.load_annot_sample>` - Reads the chip of an annotation. Note this is much less efficient than using a sampler, but it doesn't require disk cache.
 * :func:`kwcoco.CocoDataset.load_image<kwcoco.coco_dataset.MixinCocoAccessors.load_image>` - Reads an image from disk and
 * :func:`kwcoco.CocoDataset.object_categories<kwcoco.coco_dataset.MixinCocoAccessors.object_categories>` - Construct a consistent CategoryTree representation of object classes

CocoDataset methods (via CocoDataset)
*************************************

 * :func:`kwcoco.CocoDataset.copy<kwcoco.coco_dataset.CocoDataset.copy>` - Deep copies this object
 * :func:`kwcoco.CocoDataset.dump<kwcoco.coco_dataset.CocoDataset.dump>` - Writes the dataset out to the json format
 * :func:`kwcoco.CocoDataset.dumps<kwcoco.coco_dataset.CocoDataset.dumps>` - Writes the dataset out to the json format
 * :func:`kwcoco.CocoDataset.subset<kwcoco.coco_dataset.CocoDataset.subset>` - Return a subset of the larger coco dataset by specifying which images to port. All annotations in those images will be taken.
 * :func:`kwcoco.CocoDataset.union<kwcoco.coco_dataset.CocoDataset.union>` - Merges multiple :class:`CocoDataset` items into one. Names and associations are retained, but ids may be different.
 * :func:`kwcoco.CocoDataset.view_sql<kwcoco.coco_dataset.CocoDataset.view_sql>` - Create a cached SQL interface to this dataset suitable for large scale multiprocessing use cases.

CocoDataset methods (via MixinCocoExtras)
*****************************************

 * :func:`kwcoco.CocoDataset.corrupted_images<kwcoco.coco_dataset.MixinCocoExtras.corrupted_images>` - Check for images that don't exist or can't be opened
 * :func:`kwcoco.CocoDataset.missing_images<kwcoco.coco_dataset.MixinCocoExtras.missing_images>` - Check for images that don't exist
 * :func:`kwcoco.CocoDataset.rename_categories<kwcoco.coco_dataset.MixinCocoExtras.rename_categories>` - Rename categories with a potentially coarser categorization.
 * :func:`kwcoco.CocoDataset.reroot<kwcoco.coco_dataset.MixinCocoExtras.reroot>` - Rebase image/data paths onto a new image/data root.

CocoDataset methods (via MixinCocoDraw)
***************************************

 * :func:`kwcoco.CocoDataset.draw_image<kwcoco.coco_dataset.MixinCocoDraw.draw_image>` - Use kwimage to draw all annotations on an image and return the pixels as a numpy array.
 * :func:`kwcoco.CocoDataset.imread<kwcoco.coco_dataset.MixinCocoDraw.imread>` - Loads a particular image
 * :func:`kwcoco.CocoDataset.show_image<kwcoco.coco_dataset.MixinCocoDraw.show_image>` - Use matplotlib to show an image with annotations overlaid

"""

__dev__ = """

Some of the above docs were generated via:
    ~/code/kwcoco/dev/coco_dataset_api_introspect.py

The logic of this init is generated via:
    mkinit -m kwcoco --diff

    mkinit kwcoco --lazy

Testing:

    EAGER_IMPORT=1 python -c "import kwcoco"
    python -c "import kwcoco; print(kwcoco.CocoSqlDatabase)"

"""

__version__ = '0.6.3'


__submodules__ = {
    'abstract_coco_dataset': ['AbstractCocoDataset'],
    'coco_dataset': ['CocoDataset'],
    'coco_image': ['CocoImage'],
    'category_tree': ['CategoryTree'],
    'channel_spec': ['ChannelSpec', 'FusedChannelSpec'],
    'exceptions': [],
    'coco_sql_dataset': ['CocoSqlDatabase'],
    'sensorchan_spec': ['SensorChanSpec'],
}

# __lazy_submodules__ = {
#     # TODO: always lazy submodules
#     'coco_sql_dataset': ['CocoSqlDatabase'],
#     'sensorchan_spec': ['SensorChanSpec'],
# }

import sys
if sys.version_info[0:2] < (3, 7):
    # 3.6 does not have lazy imports
    from kwcoco.sensorchan_spec import SensorChanSpec
    from kwcoco.coco_sql_dataset import CocoSqlDatabase

    from kwcoco import abstract_coco_dataset
    from kwcoco import category_tree
    from kwcoco import channel_spec
    from kwcoco import coco_dataset
    from kwcoco import coco_image
    from kwcoco import exceptions

    from kwcoco.abstract_coco_dataset import (AbstractCocoDataset,)
    from kwcoco.coco_dataset import (CocoDataset,)
    from kwcoco.coco_image import (CocoImage,)
    from kwcoco.category_tree import (CategoryTree,)
    from kwcoco.channel_spec import (ChannelSpec, FusedChannelSpec,)

####


def lazy_import(module_name, submodules, submod_attrs):
    import importlib
    import os
    name_to_submod = {
        func: mod for mod, funcs in submod_attrs.items()
        for func in funcs
    }

    def __getattr__(name):
        if name in submodules:
            attr = importlib.import_module(
                '{module_name}.{name}'.format(
                    module_name=module_name, name=name)
            )
        elif name in name_to_submod:
            submodname = name_to_submod[name]
            module = importlib.import_module(
                '{module_name}.{submodname}'.format(
                    module_name=module_name, submodname=submodname)
            )
            attr = getattr(module, name)
        else:
            raise AttributeError(
                'No {module_name} attribute {name}'.format(
                    module_name=module_name, name=name))
        globals()[name] = attr
        return attr

    if os.environ.get('EAGER_IMPORT', ''):
        for name in submodules:
            __getattr__(name)

        for attrs in submod_attrs.values():
            for attr in attrs:
                __getattr__(attr)
    return __getattr__


__getattr__ = lazy_import(
    __name__,
    submodules={
        'abstract_coco_dataset',
        'category_tree',
        'channel_spec',
        'coco_dataset',
        'coco_image',
        'coco_sql_dataset',
        'exceptions',
        'sensorchan_spec',
    },
    submod_attrs={
        'abstract_coco_dataset': [
            'AbstractCocoDataset',
        ],
        'category_tree': [
            'CategoryTree',
        ],
        'channel_spec': [
            'ChannelSpec',
            'FusedChannelSpec',
        ],
        'coco_dataset': [
            'CocoDataset',
        ],
        'coco_image': [
            'CocoImage',
        ],
        'coco_sql_dataset': [
            'CocoSqlDatabase',
        ],
        'sensorchan_spec': [
            'SensorChanSpec',
        ],
    },
)


def __dir__():
    return __all__

__all__ = ['AbstractCocoDataset', 'CategoryTree', 'ChannelSpec', 'CocoDataset',
           'CocoImage', 'CocoSqlDatabase', 'FusedChannelSpec',
           'SensorChanSpec', 'abstract_coco_dataset', 'category_tree',
           'channel_spec', 'coco_dataset', 'coco_image', 'coco_sql_dataset',
           'exceptions', 'sensorchan_spec']
