"""
An implementation and extension of the original MS-COCO API [CocoFormat]_.

The KWCoco format is backwards compatible with MS-COCO. It adds support for
multispectral imagery, videos in the form of images sequences, annotation
tracks, line annotations, and defines alternative (and often more intuitive)
encodings of keypoint and segmentation data structure.

An informal description of the spec is given in:
 `coco_schema_informal.rst <coco_schema_informal.rst>`_.
A formal json-schema is defined in :mod:`kwcoco.coco_schema`.

Note:
    The main object in this file is :class:`CocoDataset`, which is composed of
    several mixin classes. See the class and method documentation for more
    details.

TODO:
    - [ ] Use ijson (modified to support NaN) to lazilly load pieces of the
        dataset in the background or on demand. This will give us faster access
        to categories / images, whereas we will always have to wait for
        annotations etc...

    - [X] Should img_root be changed to bundle_dpath?

    - [ ] Read video data, return numpy arrays (requires API for images)

    - [ ] Spec for video URI, and convert to frames @ framerate function.

    - [x] Document channel spec

    - [x] Document sensor-channel spec

    - [X] Add remove videos method

    - [ ] Efficiency: Make video annotations more efficient by only tracking
          keyframes, provide an API to obtain a dense or interpolated
          annotation on an intermediate frame.

    - [ ] Efficiency: Allow each section of the kwcoco file to be written as a
          separate json file. Perhaps allow generic pointer support? Might get
          messy.

    - [ ] Reroot needs to be redesigned very carefully.

    - [ ] Allow parts of the kwcoco file to be references to other json files.

    - [X] Add top-level track table

    - [ ] Fully transition to integer track ids (in progress)

References:
    .. [CocoFormat] http://cocodataset.org/#format-data
    .. [PyCocoToolsMask] https://github.com/nightrome/cocostuffapi/blob/master/PythonAPI/pycocotools/mask.py
    .. [CocoTutorial] https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#coco-dataset-format
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, IO, cast

import copy
import sys
import os
import ubelt as ub
import warnings

from collections import defaultdict
from os.path import dirname, basename, join, exists, isdir, relpath

from kwcoco.abstract_coco_dataset import AbstractCocoDataset
from kwcoco._helpers import UniqueNameRemapper, _NextId
from kwcoco._helpers import _ID_Remapper
from kwcoco._helpers import _CategoryID_Remapper
from kwcoco._helpers import _coerce_info_list
from kwcoco._dataset_components.mixin_deprecate import MixinCocoDeprecate
from kwcoco._dataset_components.mixin_hash import MixinCocoHashing
from kwcoco._dataset_components.mixin_draw import MixinCocoDraw
from kwcoco._dataset_components.mixin_index import MixinCocoIndex, CocoIndex
from kwcoco._dataset_components.mixin_accessors import MixinCocoAccessors
from kwcoco._dataset_components.mixin_constructors import MixinCocoConstructors
from kwcoco._dataset_components.mixin_extras import MixinCocoExtras
from kwcoco._dataset_components.mixin_objects import MixinCocoObjects
from kwcoco._dataset_components.mixin_stats import MixinCocoStats
from kwcoco._dataset_components.mixin_add_remove import MixinCocoAddRemove
from kwcoco._dataset_components.json_backend import json_r, json_w


if TYPE_CHECKING:
    from os import PathLike

# The ujson library is faster than Python's json, but the API has some
# limitations and requires a minimum version. Currently we only use it to read,
# we have to wait for https://github.com/ultrajson/ultrajson/pull/518 to land
# before we use it to write.


# These are the keys that are / should be supported by the API
SPEC_KEYS = [
    'info',
    'licenses',
    'categories',
    'keypoint_categories',  # support only partially implemented
    'videos',
    'images',
    'tracks',
    'annotations',
]


class CocoDataset(
    AbstractCocoDataset,
    MixinCocoAddRemove,
    MixinCocoStats,
    MixinCocoObjects,
    MixinCocoDraw,
    MixinCocoAccessors,
    MixinCocoConstructors,
    MixinCocoExtras,
    MixinCocoHashing,
    MixinCocoIndex,
    MixinCocoDeprecate,
    ub.NiceRepr,
):
    """
    The main coco dataset class with a json dataset backend.

    Attributes:
        dataset (Dict): raw json data structure. This is the base dictionary
            that contains {'annotations': List, 'images': List,
            'categories': List}

        index (CocoIndex): an efficient lookup index into the coco data
            structure. The index defines its own attributes like
            ``anns``, ``cats``, ``imgs``, ``gid_to_aids``,
            ``file_name_to_img``, etc. See :class:`CocoIndex` for more details
            on which attributes are available.

        fpath (PathLike | None):
            if known, this stores the filepath the dataset was loaded from

        tag (str | None):
            A tag indicating the name of the dataset.

        bundle_dpath (PathLike | None) :
            If known, this is the root path that all image file names are
            relative to. This can also be manually overwritten by the user.

        hashid (str | None) :
            If computed, this will be a hash uniquely identifying the dataset.
            To ensure this is computed see
            :func:`kwcoco.coco_dataset.MixinCocoExtras._build_hashid`.

    References:
        http://cocodataset.org/#format
        http://cocodataset.org/#download

    CommandLine:
        python -m kwcoco.coco_dataset CocoDataset --show

    Example:
        >>> from kwcoco.coco_dataset import demo_coco_data
        >>> import kwcoco
        >>> import ubelt as ub
        >>> # Returns a coco json structure
        >>> dataset = demo_coco_data()
        >>> # Pass the coco json structure to the API
        >>> self = kwcoco.CocoDataset(dataset, tag='demo')
        >>> # Now you can access the data using the index and helper methods
        >>> #
        >>> # Start by looking up an image by it's COCO id.
        >>> image_id = 1
        >>> img = self.index.imgs[image_id]
        >>> print(ub.urepr(img, nl=1, sort=1))
        {
            'file_name': 'astro.png',
            'id': 1,
            'url': 'https://i.imgur.com/KXhKM72.png',
        }
        >>> #
        >>> # Use the (gid_to_aids) index to lookup annotations in the image
        >>> annotation_id = sorted(self.index.gid_to_aids[image_id])[0]
        >>> ann = self.index.anns[annotation_id]
        >>> print(ub.urepr((ub.udict(ann) - {'segmentation'}).sorted_keys(), nl=1))
        {
            'bbox': [10, 10, 360, 490],
            'category_id': 1,
            'id': 1,
            'image_id': 1,
            'keypoints': [247, 101, 2, 202, 100, 2],
        }
        >>> #
        >>> # Use annotation category id to look up that information
        >>> category_id = ann['category_id']
        >>> cat = self.index.cats[category_id]
        >>> print('cat = {}'.format(ub.urepr(cat, nl=1, sort=1)))
        cat = {
            'id': 1,
            'name': 'astronaut',
            'supercategory': 'human',
        }
        >>> #
        >>> # Now play with some helper functions, like extended statistics
        >>> extended_stats = self.extended_stats()
        >>> # xdoctest: +IGNORE_WANT
        >>> print('extended_stats = {}'.format(ub.urepr(extended_stats, nl=1, precision=2, sort=1)))
        extended_stats = {
            'annots_per_img': {'mean': 3.67, 'std': 3.86, 'min': 0.00, 'max': 9.00, 'nMin': 1, 'nMax': 1, 'shape': (3,)},
            'imgs_per_cat': {'mean': 0.88, 'std': 0.60, 'min': 0.00, 'max': 2.00, 'nMin': 2, 'nMax': 1, 'shape': (8,)},
            'cats_per_img': {'mean': 2.33, 'std': 2.05, 'min': 0.00, 'max': 5.00, 'nMin': 1, 'nMax': 1, 'shape': (3,)},
            'annots_per_cat': {'mean': 1.38, 'std': 1.49, 'min': 0.00, 'max': 5.00, 'nMin': 2, 'nMax': 1, 'shape': (8,)},
            'imgs_per_video': {'empty_list': True},
        }
        >>> # You can "draw" a raster of the annotated image with cv2
        >>> canvas = self.draw_image(2)
        >>> # Or if you have matplotlib you can "show" the image with mpl objects
        >>> # xdoctest: +REQUIRES(--show)
        >>> from matplotlib import pyplot as plt
        >>> fig = plt.figure()
        >>> ax1 = fig.add_subplot(1, 2, 1)
        >>> self.show_image(gid=2)
        >>> ax2 = fig.add_subplot(1, 2, 2)
        >>> ax2.imshow(canvas)
        >>> ax1.set_title('show with matplotlib')
        >>> ax2.set_title('draw with cv2')
        >>> plt.show()
    """

    dataset: dict[str, Any]
    index: CocoIndex
    _fpath: str | PathLike | None
    bundle_dpath: str
    assets_dpath: str | None
    cache_dpath: str | None
    hashid: str | None
    hashid_parts: Any
    tag: str | None

    def __init__(
        self,
        data: str | PathLike | dict | None = None,
        tag: str | None = None,
        bundle_dpath: str | None = None,
        img_root: str | None = None,
        fname: Any | None = None,
        autobuild: bool = True,
    ) -> None:
        """
        Args:

            data (str | PathLike | dict | None):
                Either a filepath to a coco json file, or a dictionary
                containing the actual coco json structure. For a more generally
                coercible constructor see func:`CocoDataset.coerce`.

                Note: in the future, we may only accept construction from a
                data dictionary, and use a `.load` classmethod to handle
                reading data from a file. This will simplify (and accelerate)
                the code in the constructor.

            tag (str | None) :
                Name of the dataset for display purposes, and does not
                influence behavior of the underlying data structure, although
                it may be used via convenience methods. We attempt to
                autopopulate this via information in ``data`` if available.
                If unspecfied and ``data`` is a filepath this becomes the
                basename.

            bundle_dpath (str | None):
                the root of the dataset that images / external data will be
                assumed to be relative to. If unspecfied, we attempt to
                determine it using information in ``data``. If ``data`` is a
                filepath, we use the dirname of that path. If ``data`` is a
                dictionary, we look for the "img_root" key. If unspecfied and
                we fail to introspect then, we fallback to the current working
                directory.

            img_root (str | None):
                deprecated alias for bundle_dpath
        """
        self._fpath = None

        # Info about what was the origin of this object and if anything
        # happened to it over its lifetime.
        self._state = {
            'was_loaded': False,
            'was_saved': False,
            'was_modified': 0,
        }

        if img_root is not None:
            bundle_dpath = img_root
        if data is None:
            # TODO: rely on subset of SPEC keys
            data = {
                'categories': [],
                'videos': [],
                'images': [],
                'annotations': [],
                'tracks': [],
                'licenses': [],
                'info': [],
            }

        fpath: str | PathLike | None = None
        inferred_date_type = None
        if isinstance(data, dict):
            # Assumption: If data is a dict and are not explicitly given
            # bundle_dpath, then we assume it is relative to the cwd.
            assumed_root = '.'
            inferred_date_type = 'json-dict'
        elif isinstance(data, (str, os.PathLike)):
            path = os.fspath(data)
            if isdir(path):
                # data was a pointer to hopefully a kwcoco bundle
                if bundle_dpath is None:
                    bundle_dpath = path
                else:
                    if bundle_dpath != path:
                        raise Exception('ambiguous')
                inferred_date_type = 'bundle-path'
            else:
                # data was a pointer to hopefully a kwcoco filepath
                # TODO: do some validation of that here
                fpath = data
                if bundle_dpath is None:
                    bundle_dpath = dirname(fpath)
                inferred_date_type = 'file-path'
        else:
            raise TypeError(
                'data must be a dict or path to json file, but got: {!r}'.format(
                    type(data)
                )
            )

        if (
            fpath is None
            and bundle_dpath is not None
            and inferred_date_type == 'bundle-path'
        ):
            # This should probably be reserved for a coercion method
            # If we are givena bundle path, assume a standard name convention
            if fname is None:
                import glob

                candidates = [
                    'data',
                    'data.json',
                    'data.kwcoco.json',
                    'data.kwcoco.zip',  # Allow zipfiles
                    'data.kwcoco.json.zip',
                    '*.kwcoco.json',
                    '*.mscoco.json',
                ]
                # Check for standard bundle manifest names
                manifest_candidate_iter = iter(
                    ub.oset(
                        ub.flatten(
                            [glob.glob(join(bundle_dpath, p)) for p in candidates]
                        )
                    )
                )
                try:
                    fpath = ub.peek(manifest_candidate_iter)
                except StopIteration:
                    fpath = join(bundle_dpath, 'data.kwcoco.json')
                    # raise Exception('No manifest in Dataset Bundle')
                else:
                    remain = list(manifest_candidate_iter)
                    if len(remain) > 0:
                        raise Exception(
                            'Ambiguous Dataset Bundle {}: {}'.format(fpath, remain)
                        )
            else:
                fpath = join(bundle_dpath, fname)
            key = basename(bundle_dpath)

        if fpath is not None:
            fname = basename(fpath)
            if fname == 'data.kwcoco.json':
                if bundle_dpath is not None:
                    bundle_dpath = dirname(fpath)
                key = basename(bundle_dpath)  # type: ignore
            else:
                key = fname

        if bundle_dpath is not None:
            assumed_root = bundle_dpath

        if isinstance(data, (str, os.PathLike)):
            if not exists(fpath):  # type: ignore
                raise Exception(
                    ub.paragraph(
                        """
                    You cannot construct a KWCocoDataset from a non-existing
                    file: fpath={!r}.
                    If you are trying to create a new dataset fist create a
                    CocoDataset without arguments: `dset = kwcoco.CocoDataset`,
                    and then set the fpath attribute:
                    `dset.fpath = 'your-file-path.kwcoco.zip'`
                    """
                    ).format(fpath)
                )

            self._state['was_loaded'] = True

            # Test to see if the kwcoco file is compressed
            import zipfile

            if zipfile.is_zipfile(fpath):  # type: ignore
                with open(fpath, 'rb') as file:  # type: ignore
                    with zipfile.ZipFile(file, 'r') as zfile:
                        members = zfile.namelist()
                        if len(members) != 1:
                            raise Exception(
                                'Currently only zipfiles with exactly 1 '
                                'kwcoco member are supported'
                            )
                        text = zfile.read(members[0]).decode('utf8')
                        data = json_r.loads(text)
            else:
                with open(fpath, 'r') as file:  # type: ignore
                    data = json_r.load(file)

            # If data is a path it gives us the absolute location of the root
            if tag is None:
                tag = key

        # Backwards compat hack, allow the coco file to specify the
        # bundle_dpath
        # TODO: deprecate img_root
        if isinstance(data, dict) and 'img_root' in data:
            # Historical code paths may have serialized ``img_root`` with a
            # null value. Treat that as unspecified and avoid warning for it.
            if data.get('img_root', None) is not None:
                ub.schedule_deprecation(
                    'kwcoco',
                    name='img_root',
                    type='dataset member',
                    deprecate='0.6.3',
                    error='1.0.0',
                    remove='1.1.0',
                    migration=ub.paragraph(
                        """
                        Ensure the location of the saved kwcoco file encodes the
                        bundle dpath or ensure bundle_dpath is correctly set in the
                        in-memory CocoDataset object.
                        """
                    ),
                )
            # allow image root to be specified in the dataset
            # we refer to this as a json data "body root".
            body_root = data.get('img_root', '')
            if body_root is None:
                body_root = ''
            elif isinstance(body_root, str):
                _tmp = ub.expandpath(body_root)
                if exists(_tmp):
                    body_root = _tmp
            else:
                if isinstance(body_root, list) and body_root == []:
                    body_root = ''
                else:
                    raise TypeError('body_root = {!r}'.format(body_root))
            try:
                bundle_dpath = join(assumed_root, body_root)
            except Exception:
                print('body_root = {!r}'.format(body_root))
                print('assumed_root = {!r}'.format(assumed_root))
                raise

        if bundle_dpath is None:
            bundle_dpath = assumed_root

        bundle_dpath = ub.expandpath(bundle_dpath)

        if fpath is None:
            fpath = join(bundle_dpath, 'data.kwcoco.json')

        self.index = CocoIndex()

        self.hashid = None
        self.hashid_parts = None

        self.tag = tag
        self.dataset = cast('dict[str, Any]', data)

        self.data_fpath = fpath
        self.bundle_dpath = bundle_dpath

        self.cache_dpath = None
        self.assets_dpath = None

        # Keep track of an unused id we may use
        self._next_ids = _NextId(self)

        self._infer_dirs()

        if autobuild:
            self.rebuild_index()

    @property
    def fpath(self) -> None | str | PathLike:
        """In the future we will deprecate img_root for bundle_dpath"""
        return self._fpath

    @fpath.setter
    def fpath(self, value: str | PathLike | None) -> None:
        # Cant use update fpath because of reroot checks.
        # self._update_fpath(value)
        self._fpath = value
        self._infer_dirs()

    def _update_fpath(self, new_fpath: str | PathLike | None) -> None:
        """
        Update the write location for this dataset and reroot if necessary.

        New method for more robustly updating the file path and bundle
        directory, still a WIP. Only works when the current dataset is
        already valid.

        This is used when you are reading a source dataset, and modifying, and
        then plan to write it to a new location. If the new location is in a
        different bundle dpath than the original file, we need to reroot to
        otherwise any relative paths will break.

        We need to come up with a good name for this function.  I don't think
        this should happen when just modifying the ``fpath`` property. But I
        also don't know what a good name that indicates this sets fpath and
        does a conditional reroot would be.
        """
        if new_fpath is None:
            # Bundle directory is clobbered, so we should make everything
            # absolute
            self.reroot(absolute=True)
        else:
            old_fpath = self.fpath
            if old_fpath is not None:
                old_fpath_ = ub.Path(old_fpath)
                new_fpath_ = ub.Path(new_fpath)

                same_bundle = (old_fpath_.parent == new_fpath_.parent) or (
                    old_fpath_.resolve() == new_fpath_.resolve()
                )
                if not same_bundle:
                    # The bundle directory has changed, so we need to reroot
                    new_root = new_fpath_.parent
                    self.reroot(new_root)

            self._fpath = new_fpath
            self._infer_dirs()

    def _infer_dirs(self) -> None:
        """
        Ignore:
            self = dset
        """
        data_fpath = self.fpath
        if data_fpath is not None:
            bundle_dpath = dirname(data_fpath)
            assets_dpath = join(bundle_dpath, '_assets')
            cache_dpath = join(bundle_dpath, '_cache')
            # OPINION: Do we want conditions?
            # data_fname = basename(data_fpath)
            # bundle_conditions = {
            #     # 'name': data_fname == 'data.kwcoco.json',
            #     # 'ext': data_fname.endswith('.kwcoco.json'),
            #     # 'has_assets': exists(assets_dpath),
            # }
            # is_bundle = all(bundle_conditions.values())
            self.bundle_dpath = bundle_dpath
            self.assets_dpath = assets_dpath
            self.cache_dpath = cache_dpath

    def __nice__(self) -> str:
        parts = []
        parts.append('tag={}'.format(self.tag))
        if self.dataset is not None:
            info = ub.urepr(self.basic_stats(), kvsep='=', si=1, nobr=1, nl=0)  # type: ignore
            parts.append(info)
        return ', '.join(parts)

    def copy(self) -> CocoDataset:
        """
        Deep copies this object

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> new = self.copy()
            >>> assert new.imgs[1] is new.dataset['images'][0]
            >>> assert new.imgs[1] == self.dataset['images'][0]
            >>> assert new.imgs[1] is not self.dataset['images'][0]
        """
        new = copy.copy(self)
        new.index = CocoIndex()
        new.hashid_parts = copy.deepcopy(self.hashid_parts)
        new.dataset = copy.deepcopy(self.dataset)
        new._next_ids = _NextId(new)
        new.rebuild_index()
        return new

    def dumps(self, indent: int | str | None = None, newlines: bool = False) -> str:
        """
        Writes the dataset out to the json format

        Args:
            newlines (bool) :
                if True, each annotation, image, category gets its own line

            indent (int | str | None): indentation for the json file. See
                :func:`json.dump` for details.

            newlines (bool):
                if True, each annotation, image, category gets its own line.

        Note:
            Using newlines=True is similar to:
                print(ub.urepr(dset.dataset, nl=2, trailsep=False))
                However, the above may not output valid json if it contains
                ndarrays.

        Example:
            >>> import kwcoco
            >>> import json
            >>> self = kwcoco.CocoDataset.demo()
            >>> text = self.dumps(newlines=True)
            >>> print(text)
            >>> self2 = kwcoco.CocoDataset(json.loads(text), tag='demo2')
            >>> assert self2.dataset == self.dataset
            >>> assert self2.dataset is not self.dataset

            >>> text = self.dumps(newlines=True)
            >>> print(text)
            >>> self2 = kwcoco.CocoDataset(json.loads(text), tag='demo2')
            >>> assert self2.dataset == self.dataset
            >>> assert self2.dataset is not self.dataset

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.coerce('vidshapes1-msi-multisensor', verbose=3)
            >>> self.remove_annotations(self.annots())
            >>> text = self.dumps(newlines=0, indent='  ')
            >>> print(text)
            >>> text = self.dumps(newlines=True, indent='  ')
            >>> print(text)
        """
        from kwcoco.util import util_special_json

        # Instead of using json to dump the whole thing make the text a bit
        # more pretty.
        try:
            if newlines:
                # TODO: make this work faster!
                text = util_special_json._special_kwcoco_pretty_dumps_orig(self.dataset)
            else:
                # TODO: do main key sorting here as well
                text = util_special_json._json_dumps(self.dataset, indent=indent)
        except Exception as ex:
            print('Failed to dump ex = {!r}'.format(ex))
            self._check_json_serializable()
            raise

        return text

    def _compress_dump_to_fileptr(
        self,
        file: Any,
        arcname: None | str = None,
        indent: int | str | None = None,
        newlines: bool = False,
    ) -> Any:
        """
        Experimental method to save compressed kwcoco files, may be folded into
        dump in the future.
        """
        import zipfile
        from kwcoco.util import util_archive

        compression = util_archive._coerce_zipfile_compression('auto')
        zipkw = {
            'compression': compression,
        }
        if sys.version_info[0:2] >= (3, 7):
            zipkw['compresslevel'] = None
        if arcname is None:
            if not self.fpath:
                arcname = '_data.kwcoco.json'
            else:
                # Use the current name of the file to compress
                arcname = basename(self.fpath)
                if arcname.endswith('.zip'):
                    arcname = arcname[:-4]
                    if not arcname.endswith('.json'):
                        arcname = arcname + '.json'
        with zipfile.ZipFile(file, 'w', **zipkw) as zfile:  # type: ignore
            # It would be nice if we could dump directly to a binary
            # encoded format here. Then we could use self.dump(zfile.open())
            text = self.dumps(indent=indent, newlines=newlines)
            zfile.writestr(arcname, text.encode('utf8'))

    def _dump(
        self, file: Any, indent: int | str | None, newlines: bool, compress: bool
    ) -> None:
        """
        Case where we are dumping to an open file pointer.
        We assume this means the dataset has been written to disk.
        """
        if compress:
            self._compress_dump_to_fileptr(file, indent=indent, newlines=newlines)
        else:
            if newlines:
                file.write(self.dumps(indent=indent, newlines=newlines))
            else:
                try:
                    json_w.dump(self.dataset, file, indent=indent, ensure_ascii=False)
                except Exception as ex:
                    print('Failed to dump ex = {!r}'.format(ex))
                    if 'Circular reference detected' in str(ex):
                        raise
                    self._check_json_serializable()
                    raise
        self._state['was_saved'] = True

    def dump(
        self,
        file: PathLike | IO | None = None,
        indent: int | str | None = None,
        newlines: bool = False,
        temp_file: bool | str = 'auto',
        compress: bool | str = 'auto',
        verbose: int = 0,
    ) -> None:
        """
        Writes the dataset out to the json format

        Args:
            file (PathLike | IO | None):
                Where to write the data. Can either be a path to a file or an
                open file pointer / stream. If unspecified, it will be written
                to the current ``fpath`` property.

            indent (int | str | None): indentation for the json file. See
                :func:`json.dump` for details.

            newlines (bool):
                if True, each annotation, image, category gets its own line.

            temp_file (bool | str):
                Argument to :func:`safer.open`.  Ignored if ``file`` is not a
                PathLike object. Defaults to 'auto', which is False on Windows
                and True everywhere else.

            compress (bool | str):
                if True, dumps the kwcoco file as a compressed zipfile.
                In this case a literal IO file object must be opened in binary
                write mode. If auto, then it will default to False unless
                it can introspect the file name and the name ends with .zip

            verbose (int):
                verbosity level

        Example:
            >>> import kwcoco
            >>> import ubelt as ub
            >>> dpath = ub.Path.appdir('kwcoco/demo/dump').ensuredir()
            >>> dset = kwcoco.CocoDataset.demo()
            >>> dset.fpath = dpath / 'my_coco_file.json'
            >>> # Calling dump writes to the current fpath attribute.
            >>> dset.dump()
            >>> assert dset.dataset == kwcoco.CocoDataset(dset.fpath).dataset
            >>> assert dset.dumps() == dset.fpath.read_text()
            >>> #
            >>> # Using compress=True can save a lot of space and it
            >>> # is transparent when reading files via CocoDataset
            >>> dset.dump(compress=True)
            >>> assert dset.dataset == kwcoco.CocoDataset(dset.fpath).dataset
            >>> assert dset.dumps() != dset.fpath.read_text(errors='replace')

        Example:
            >>> import kwcoco
            >>> import ubelt as ub
            >>> # Compression auto-defaults based on the file name.
            >>> dpath = ub.Path.appdir('kwcoco/demo/dump').ensuredir()
            >>> dset = kwcoco.CocoDataset.demo()
            >>> fpath1 = dset.fpath = dpath / 'my_coco_file.zip'
            >>> dset.dump(verbose=1)
            >>> fpath2 = dset.fpath = dpath / 'my_coco_file.json'
            >>> dset.dump(verbose=1)
            >>> assert fpath1.read_bytes()[0:8] != fpath2.read_bytes()[0:8]
        """
        from kwcoco.util.util_json import coerce_indent

        indent = coerce_indent(indent)

        if file is None:
            file = self.fpath  # type: ignore

        try:
            fpath = os.fspath(file)  # type: ignore
        except TypeError:
            input_was_pathlike = False
        else:
            input_was_pathlike = True

        if compress == 'auto':
            compress = False
            if not input_was_pathlike:
                fpath = getattr(file, 'name', None)
            if fpath is not None:
                if os.fspath(fpath).endswith('.zip'):
                    compress = True

        mode = 'wb' if compress else 'w'

        if verbose:
            if input_was_pathlike:
                print(f'Writing to: {fpath}... ', end='')
            else:
                print('Writing to file... ', end='')

        if input_was_pathlike:
            import safer

            if temp_file == 'auto':
                temp_file = not ub.WIN32

            with safer.open(fpath, mode, temp_file=temp_file) as fp:  # type: ignore
                self._dump(fp, indent=indent, newlines=newlines, compress=compress)  # type: ignore
        else:
            # We are likely dumping to a real file.
            self._dump(file, indent=indent, newlines=newlines, compress=compress)  # type: ignore

        if verbose:
            print('done.')

    def _check_json_serializable(self, verbose: Any = 1) -> list:
        """
        Debug which part of a coco dataset might not be json serializable
        """
        from kwcoco.util.util_json import find_json_unserializable

        bad_parts_gen = find_json_unserializable(self.dataset)
        bad_parts: list[Any] = []
        for part in bad_parts_gen:
            if verbose == 3:
                print('part = {!r}'.format(part))
            elif verbose and len(bad_parts) == 0:
                # print out the first one we find
                print('Found at least one bad part = {!r}'.format(part))
            bad_parts.append(part)

        if verbose:
            # if bad_parts:
            #     print(ub.urepr(bad_parts))
            summary = 'There are {} total errors'.format(len(bad_parts))
            print('summary = {}'.format(summary))
        return bad_parts

    def _check_integrity(self) -> bool:
        """perform most checks"""
        self._check_index()
        self._check_pointers()
        self._check_json_serializable()
        self.validate(img_attrs=False, annot_attrs=False, unique=False)
        # assert len(self.missing_images()) == 0
        return True

    def _check_warnables(self) -> None:
        warning_checks = {}
        referenced_track_ids = set()
        for annot in self.dataset['annotations']:
            track_id = annot.get('track_id', None)
            referenced_track_ids.add(track_id)

        table_track_ids = {track['id'] for track in self.dataset.get('tracks', [])}
        # referenced_track_ids - table_track_ids
        # table_track_ids - referenced_track_ids
        # True if there is a track in the table without an annotation.
        warning_checks['unreferenced_track_ids'] = table_track_ids.issubset(
            referenced_track_ids
        )
        # True if there are annotations with a track id not in the track table.
        warning_checks['unregistered_track_ids'] = table_track_ids.issuperset(
            referenced_track_ids
        )

        failed_warnings = {k: v for k, v in warning_checks.items() if not v}
        if any(failed_warnings):
            msg = 'Potential issues: {}'.format(list(failed_warnings))
            warnings.warn(msg)

    def _check_index(self) -> bool:
        """
        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> self._check_index()
            >>> # Force a failure
            >>> self.index.anns.pop(1)
            >>> self.index.anns.pop(2)
            >>> import pytest
            >>> with pytest.raises(AssertionError):
            >>>     self._check_index()
        """
        # We can verify our index invariants by copying the raw dataset and
        # checking if the newly constructed index is the same as this index.
        new_dataset = copy.deepcopy(self.dataset)
        new = self.__class__(new_dataset, autobuild=False)
        new.rebuild_index()
        checks = {}
        checks['anns'] = self.index.anns == new.index.anns
        checks['imgs'] = self.index.imgs == new.index.imgs
        checks['cats'] = self.index.cats == new.index.cats
        checks['tracks'] = self.index.tracks == new.index.tracks
        checks['gid_to_aids'] = self.index.gid_to_aids == new.index.gid_to_aids
        checks['cid_to_aids'] = self.index.cid_to_aids == new.index.cid_to_aids
        checks['name_to_cat'] = self.index.name_to_cat == new.index.name_to_cat
        checks['name_to_img'] = self.index.name_to_img == new.index.name_to_img
        checks['file_name_to_img'] = (
            self.index.file_name_to_img == new.index.file_name_to_img
        )
        checks['name_to_track'] = self.index.name_to_track == new.index.name_to_track

        one_to_many1: dict = self.index.trackid_to_aids  # type: ignore
        one_to_many2: dict = new.index.trackid_to_aids  # type: ignore

        missing2: dict[Any, Any] = ub.dict_diff(one_to_many1, one_to_many2)
        missing1: dict[Any, Any] = ub.dict_diff(one_to_many2, one_to_many1)
        common1: dict[Any, Any] = ub.dict_isect(one_to_many1, one_to_many2)
        common2: dict[Any, Any] = ub.dict_isect(one_to_many2, one_to_many1)
        checks['trackid_to_aids'] = all(
            [
                all(len(v) == 0 for v in missing1.values()),
                all(len(v) == 0 for v in missing2.values()),
                common1 == common2,
            ]
        )
        checks['vidid_to_gids'] = self.index.vidid_to_gids == new.index.vidid_to_gids

        failed_checks = {k: v for k, v in checks.items() if not v}
        if any(failed_checks):
            raise AssertionError('Failed index checks: {}'.format(list(failed_checks)))
        return True

    def _check_pointers(self, verbose: Any = 1) -> bool:
        """
        Check that all category and image ids referenced by annotations exist
        """
        if not self.index:
            raise Exception('Build index before running pointer check')
        errors = []
        warnings = []

        annots = self.dataset['annotations']
        iter_ = ub.ProgIter(annots, desc='check annots', enabled=verbose)
        for ann in iter_:
            aid = ann['id']
            cid = ann['category_id']
            gid = ann['image_id']
            trackid = ann.get('track_id', None)

            if cid not in self.index.cats:  # type: ignore
                if cid is not None:
                    errors.append('aid={} references bad cid={}'.format(aid, cid))
            else:
                if self.index.cats[cid]['id'] != cid:  # type: ignore
                    errors.append('cid={} has a bad index'.format(cid))

            if gid not in self.index.imgs:  # type: ignore
                errors.append('aid={} references bad gid={}'.format(aid, gid))
            else:
                if self.index.imgs[gid]['id'] != gid:  # type: ignore
                    errors.append('gid={} has a bad index'.format(gid))

            if trackid is not None:
                if trackid not in self.index.tracks:  # type: ignore
                    warnings.append(
                        'aid={} references bad track_id={}'.format(aid, trackid)
                    )

        iter_ = ub.ProgIter(
            self.dataset['images'], desc='check images', enabled=verbose
        )
        for img in iter_:
            gid = img['id']
            vidid = img.get('video_id', None)
            if vidid is not None:
                if vidid not in self.index.videos:  # type: ignore
                    # Dont make this an error because a video dictionary is not
                    # strictly necessary for images to be linked via videos.
                    warnings.append(
                        'gid={} references bad video_id={}'.format(gid, vidid)
                    )
                elif self.index.videos[vidid]['id'] != vidid:  # type: ignore
                    errors.append('video_id={} has a bad index'.format(vidid))

        if 'categories' in self.dataset:
            iter_ = ub.ProgIter(
                self.dataset['categories'], desc='check categories', enabled=verbose
            )
            for category in iter_:
                category_id = category['id']
                if self.index.cats[category_id]['id'] != category_id:  # type: ignore
                    errors.append(f'category_id={category_id} has a bad index')

        if 'tracks' in self.dataset:
            iter_ = ub.ProgIter(
                self.dataset['tracks'], desc='check tracks', enabled=verbose
            )
            for track in iter_:
                track_id = track['id']
                if self.index.tracks[track_id]['id'] != track_id:  # type: ignore
                    errors.append(f'track_id={track_id} has a bad index')

        if 'videos' in self.dataset:
            iter_ = ub.ProgIter(
                self.dataset['videos'], desc='check videos', enabled=verbose
            )
            for video in iter_:
                video_id = video['id']
                if self.index.videos[video_id]['id'] != video_id:  # type: ignore
                    errors.append(f'video_id={video_id} has a bad index')

        if errors:
            raise Exception('\n'.join(errors))
        elif verbose:
            print('Pointers are consistent')
        return True

    def _build_index(self) -> None:
        # use rebuild_index instead
        # ub.schedule_deprecation
        self.index.build(self)

    def rebuild_index(self) -> None:
        """
        Build or rebuild the fast lookup index.

        This can be used to ensure a clean fast lookup index if the user
        directly modifies values in `self.dataset`.
        """
        self.index.build(self)

    def union(
        *others: Any,
        disjoint_tracks: bool = True,
        remember_parent: bool = False,
        **kwargs: Any,
    ) -> CocoDataset:
        """
        Combines multiple :class:`CocoDataset` items into one. Names and
        associations are retained, but ids may be different.

        Args:
            *others : a series of CocoDatasets that we will merge.
                Note, if called as an instance method, the "self" instance
                will be the first item in the "others" list. But if called
                like a classmethod, "others" will be empty by default.

            disjoint_tracks (bool):
                if True, we will assume track-names are disjoint and if two
                datasets share the same track-name, we will disambiguate them.
                Otherwise they will be copied over as-is. Defaults to True.
                In most cases you do not want to set this to False.

            remember_parent (bool):
                if True, videos and images will save information about their
                parent in the "union_parent" field.

            **kwargs : constructor options for the new merged CocoDataset

        Note:
            While categories in datasets are merged, all other information is
            not. I.e. this assumes images, videos, and annotations in the
            datsets are all disjoint. If you need to combine datasets you
            currently must copy one dataset, and then add information from
            other datasets while removing ambiguities in a way that makes sense
            in context.

        Returns:
            kwcoco.CocoDataset: a new coco dataset containing original datasets.

        CommandLine:
            xdoctest -m kwcoco.coco_dataset CocoDataset.union

        Example:
            >>> import kwcoco
            >>> # Test union works with different keypoint categories
            >>> dset1 = kwcoco.CocoDataset.demo('shapes1')
            >>> dset2 = kwcoco.CocoDataset.demo('shapes2')
            >>> dset1.remove_keypoint_categories(['bot_tip', 'mid_tip', 'right_eye'], clean_anns=True)
            >>> dset2.remove_keypoint_categories(['top_tip', 'left_eye'], clean_anns=True)
            >>> dset_12a = kwcoco.CocoDataset.union(dset1, dset2)
            >>> dset_12b = dset1.union(dset2)
            >>> dset_21 = dset2.union(dset1)
            >>> def add_hist(h1, h2):
            >>>     return {k: h1.get(k, 0) + h2.get(k, 0) for k in set(h1) | set(h2)}
            >>> kpfreq1 = dset1.keypoint_annotation_frequency()
            >>> kpfreq2 = dset2.keypoint_annotation_frequency()
            >>> kpfreq_want = add_hist(kpfreq1, kpfreq2)
            >>> kpfreq_got1 = dset_12a.keypoint_annotation_frequency()
            >>> kpfreq_got2 = dset_12b.keypoint_annotation_frequency()
            >>> assert kpfreq_want == kpfreq_got1
            >>> assert kpfreq_want == kpfreq_got2

            >>> # Test disjoint gid datasets
            >>> dset1 = kwcoco.CocoDataset.demo('shapes3')
            >>> for new_gid, img in enumerate(dset1.dataset['images'], start=10):
            >>>     for aid in dset1.gid_to_aids[img['id']]:
            >>>         dset1.anns[aid]['image_id'] = new_gid
            >>>     img['id'] = new_gid
            >>> dset1.index.clear()
            >>> dset1.rebuild_index()
            >>> # ------
            >>> dset2 = kwcoco.CocoDataset.demo('shapes2')
            >>> for new_gid, img in enumerate(dset2.dataset['images'], start=100):
            >>>     for aid in dset2.gid_to_aids[img['id']]:
            >>>         dset2.anns[aid]['image_id'] = new_gid
            >>>     img['id'] = new_gid
            >>> dset1.index.clear()
            >>> dset2.rebuild_index()
            >>> others = [dset1, dset2]
            >>> merged = kwcoco.CocoDataset.union(*others)
            >>> print('merged = {!r}'.format(merged))
            >>> print('merged.imgs = {}'.format(ub.urepr(merged.imgs, nl=1)))
            >>> assert set(merged.imgs) & set([10, 11, 12, 100, 101]) == set(merged.imgs)

            >>> # Test data is not preserved
            >>> dset2 = kwcoco.CocoDataset.demo('shapes2')
            >>> dset1 = kwcoco.CocoDataset.demo('shapes3')
            >>> others = (dset1, dset2)
            >>> cls = self = kwcoco.CocoDataset
            >>> merged = cls.union(*others)
            >>> print('merged = {!r}'.format(merged))
            >>> print('merged.imgs = {}'.format(ub.urepr(merged.imgs, nl=1)))
            >>> assert set(merged.imgs) & set([1, 2, 3, 4, 5]) == set(merged.imgs)

            >>> # Test track-ids are mapped correctly
            >>> dset1 = kwcoco.CocoDataset.demo('vidshapes1')
            >>> dset2 = kwcoco.CocoDataset.demo('vidshapes2')
            >>> dset3 = kwcoco.CocoDataset.demo('vidshapes3')
            >>> others = (dset1, dset2, dset3)
            >>> for dset in others:
            >>>     [a.pop('segmentation', None) for a in dset.index.anns.values()]
            >>>     [a.pop('keypoints', None) for a in dset.index.anns.values()]
            >>> cls = self = kwcoco.CocoDataset
            >>> merged = cls.union(*others, disjoint_tracks=1)
            >>> print('dset1.anns = {}'.format(ub.urepr(dset1.anns, nl=1)))
            >>> print('dset2.anns = {}'.format(ub.urepr(dset2.anns, nl=1)))
            >>> print('dset3.anns = {}'.format(ub.urepr(dset3.anns, nl=1)))
            >>> print('merged.anns = {}'.format(ub.urepr(merged.anns, nl=1)))

        Example:
            >>> import kwcoco
            >>> # Test empty union
            >>> empty_union = kwcoco.CocoDataset.union()
            >>> assert len(empty_union.index.imgs) == 0

        TODO:
            - [ ] are supercategories broken?
            - [ ] reuse image ids where possible
            - [ ] reuse annotation ids where possible
            - [x] reuse category ids where possible
            - [X] handle case where no inputs are given
            - [x] disambiguate track-ids
            - [x] disambiguate video-ids
        """
        from os.path import normpath

        # Dev Note:
        # See ~/misc/tests/python/test_multiarg_classmethod.py
        # for tests for how to correctly implement this method such that it can
        # behave as a method or a classmethod
        if len(others) > 0:
            cls = type(others[0])
        else:
            cls = CocoDataset

        # TODO: add an option such that the union will fail if the names
        # are not already disjoint. Alternatively, it could be the case
        # that a union between images with the same name really does
        # mean that they are the same image.
        unique_img_names = UniqueNameRemapper(policy='warn', name_type='image')
        unique_video_names = UniqueNameRemapper(policy='warn', name_type='video')
        unique_track_names = UniqueNameRemapper(policy='warn', name_type='track')

        def update_ifnotin(d1, d2):
            """copies keys from d2 that doent exist in d1 into d1"""
            for k, v in d2.items():
                if k not in d1:
                    d1[k] = v
            return d1

        def _has_duplicates(items):
            seen = set()
            for item in items:
                if item in seen:
                    return True
                seen.add(item)
            return False

        def _coco_union(relative_dsets, common_root):
            """union of dictionary based data structure"""
            # TODO: rely on subset of SPEC keys
            merged = dict(
                [
                    ('licenses', []),
                    ('info', []),
                    ('categories', []),
                    ('videos', []),
                    ('images', []),
                    ('tracks', []),
                    ('annotations', []),
                ]
            )

            # Check if the image-ids are unique and can be preserved
            _all_imgs = (img for _, _, d in relative_dsets for img in d['images'])
            _all_gids = (img['id'] for img in _all_imgs)
            preserve_gids = not _has_duplicates(_all_gids)

            # Check if the video-ids are unique and can be preserved
            _all_videos = (
                video
                for _, _, d in relative_dsets
                for video in (d.get('videos', None) or [])
            )
            _all_vidids = (video['id'] for video in _all_videos)
            preserve_vidids = not _has_duplicates(_all_vidids)

            _all_tracks = (
                track
                for _, _, d in relative_dsets
                for track in (d.get('tracks', None) or [])
            )
            _all_trackids = (track['id'] for track in _all_tracks)
            preserve_trackids = not _has_duplicates(_all_trackids)

            if not disjoint_tracks:
                # Only relevant if we are not changing track names
                merged_track_name_to_id = {}

            # Only useful while we still support the case where a track-ids
            # doesn't have an associated track.
            hacked_track_id_map = _ID_Remapper(reuse=False)

            # New category id remap strategy
            global_catid_remaper = _CategoryID_Remapper()
            global_keypoint_catid_remaper = _CategoryID_Remapper()
            merged['categories'] = global_catid_remaper._categories

            # If disjoint_tracks is True keep track of track-ids we've seen in
            # so far in previous datasets and ensure we dont reuse them.
            # TODO: do this Remapper class with other ids?
            # track_id_map = _ID_Remapper(reuse=False)

            for subdir, old_fpath, old_dset in relative_dsets:
                # Create temporary indexes to map from old to new
                cat_id_map = {None: None}
                img_id_map = {}
                video_id_map = {}
                track_id_map = {}
                kpcat_id_map = {}

                # Add the licenses / info into the merged dataset. MS-COCO
                # represents info as one dict, while KWCOCO also permits a
                # list of metadata records. A union necessarily promotes the
                # single-dict form to the list form.
                merged['licenses'].extend(old_dset.get('licenses', []))
                merged['info'].extend(_coerce_info_list(old_dset.get('info', [])))

                # Add the categories into the merged dataset
                for old_cat in old_dset['categories']:
                    # The same category might exist in different datasets.
                    new_cat = global_catid_remaper.remap(old_cat)
                    cat_id_map[old_cat['id']] = new_cat['id']

                # Add the keypoint categories into the merged dataset
                if 'keypoint_categories' in old_dset:
                    if 'keypoint_categories' not in merged:
                        merged['keypoint_categories'] = (
                            global_keypoint_catid_remaper._categories
                        )
                    old_id_to_name = {
                        k['id']: k['name'] for k in old_dset['keypoint_categories']
                    }
                    postproc_kpcats = []
                    for old_kpcat in old_dset['keypoint_categories']:
                        new_kpcat = global_keypoint_catid_remaper.remap(old_kpcat)
                        kpcat_id_map[old_kpcat['id']] = new_kpcat['id']
                        old_reflect_id = old_kpcat.get('reflection_id', None)
                        if old_reflect_id is not None:
                            # Temporarily overwrite reflectid with name
                            reflect_name = old_id_to_name.get(old_reflect_id, None)
                            new_kpcat['reflection_id'] = reflect_name
                            postproc_kpcats.append(new_kpcat)

                    # Fix reflection ids
                    for kpcat in postproc_kpcats:
                        reflect_name = kpcat['reflection_id']
                        new_relect_cat = global_keypoint_catid_remaper._name_to_cat.get(
                            reflect_name
                        )
                        kpcat['reflection_id'] = new_relect_cat['id']  # type: ignore

                # Add the tracks into the merged dataset
                for old_track in old_dset.get('tracks', []):
                    if preserve_trackids:
                        new_id = old_track['id']
                    else:
                        new_id = len(merged['tracks']) + 1

                    if disjoint_tracks:
                        # Tracks with the same name are considered distinct
                        # and given new non-conflicting names
                        new_trackname = unique_track_names.remap(old_track['name'])
                        new_track = dict(
                            [
                                ('id', new_id),
                                ('name', new_trackname),
                            ]
                        )
                        # copy over other metadata
                        update_ifnotin(new_track, old_track)
                        track_id_map[old_track['id']] = new_track['id']
                        if remember_parent:
                            new_track['union_parent'] = old_fpath
                        merged['tracks'].append(new_track)
                    else:
                        # Tracks with the same name are considered the same
                        # across datasets.
                        if old_track['name'] not in unique_track_names:
                            new_trackname = unique_track_names.remap(old_track['name'])
                            new_track = dict(
                                [
                                    ('id', new_id),
                                    ('name', new_trackname),
                                ]
                            )
                            # copy over other metadata
                            update_ifnotin(new_track, old_track)
                            track_id_map[old_track['id']] = new_track['id']
                            if remember_parent:
                                new_track['union_parent'] = old_fpath
                            merged['tracks'].append(new_track)
                            merged_track_name_to_id[new_trackname] = new_id
                        else:
                            new_id = merged_track_name_to_id[old_track['name']]
                            track_id_map[old_track['id']] = new_id

                # Add the tracks into the merged dataset
                for old_video in old_dset.get('videos', []):
                    if preserve_vidids:
                        new_id = old_video['id']
                    else:
                        new_id = len(merged['videos']) + 1
                    new_vidname = unique_video_names.remap(old_video['name'])
                    new_video = dict(
                        [
                            ('id', new_id),
                            ('name', new_vidname),
                        ]
                    )
                    # copy over other metadata
                    update_ifnotin(new_video, old_video)
                    video_id_map[old_video['id']] = new_video['id']
                    if remember_parent:
                        new_video['union_parent'] = old_fpath
                    merged['videos'].append(new_video)

                # Add the images into the merged dataset
                for old_img in old_dset['images']:
                    if preserve_gids:
                        new_id = old_img['id']
                    else:
                        new_id = len(merged['images']) + 1
                    old_gname = old_img['file_name']
                    new_gname = None if old_gname is None else (join(subdir, old_gname))
                    new_img = dict(
                        [
                            ('id', new_id),
                            ('file_name', new_gname),
                        ]
                    )
                    old_name = old_img.get('name', None)
                    if old_name is not None:
                        new_name = unique_img_names.remap(old_name)
                        new_img['name'] = new_name
                    if 'auxiliary' in old_img:
                        new_auxiliary = []
                        for old_aux in old_img['auxiliary']:
                            new_aux = old_aux.copy()
                            new_aux['file_name'] = join(subdir, old_aux['file_name'])
                            new_auxiliary.append(new_aux)
                        new_img['auxiliary'] = new_auxiliary
                    if 'assets' in old_img:
                        new_auxiliary = []
                        for old_aux in old_img['assets']:
                            new_aux = old_aux.copy()
                            new_aux['file_name'] = join(subdir, old_aux['file_name'])
                            new_auxiliary.append(new_aux)
                        new_img['assets'] = new_auxiliary

                    video_img_id = video_id_map.get(old_img.get('video_id'), None)
                    if video_img_id is not None:
                        new_img['video_id'] = video_img_id
                    # copy over other metadata
                    update_ifnotin(new_img, old_img)
                    img_id_map[old_img['id']] = new_img['id']
                    if remember_parent:
                        new_img['union_parent'] = old_fpath
                    merged['images'].append(new_img)

                # Add the annotations into the merged dataset
                for old_annot in old_dset['annotations']:
                    old_cat_id = old_annot['category_id']
                    old_img_id = old_annot['image_id']
                    new_cat_id = cat_id_map.get(old_cat_id, ub.NoParam)
                    new_img_id = img_id_map.get(old_img_id, None)

                    if new_cat_id is ub.NoParam:
                        # NOTE: category_id is allowed to be None
                        warnings.warn(
                            f'annot {old_annot} in {subdir} has bad category-id {old_cat_id}'
                        )
                    if new_img_id is None:
                        warnings.warn(
                            f'annot {old_annot} in {subdir} has bad image-id {old_img_id}'
                        )
                    new_annot = dict(
                        [
                            ('id', len(merged['annotations']) + 1),
                            ('image_id', new_img_id),
                            ('category_id', new_cat_id),
                        ]
                    )

                    old_track_id = old_annot.get('track_id')
                    if old_track_id is not None:
                        new_track_id = track_id_map.get(old_track_id, None)
                        if new_track_id is None:
                            warnings.warn(
                                ub.paragraph(
                                    f"""
                                annot {old_annot} in {subdir} has track_id
                                {old_track_id} that does not have an associated
                                entry in the tracks table
                                """
                                )
                            )
                            if disjoint_tracks:
                                new_track_id = hacked_track_id_map.remap(old_track_id)
                            else:
                                new_track_id = old_track_id
                        new_annot['track_id'] = new_track_id

                    # if disjoint_tracks:
                    #     old_track_id = old_annot.get('track_id', None)
                    #     if old_track_id is not None:
                    #         new_track_id = track_id_map.remap(old_track_id)
                    #         new_annot['track_id'] = new_track_id

                    update_ifnotin(new_annot, old_annot)

                    if kpcat_id_map:
                        # Need to copy keypoint dict to not clobber original
                        # dset
                        if 'keypoints' in new_annot:
                            old_keypoints = new_annot['keypoints']
                            new_keypoints = copy.deepcopy(old_keypoints)
                            for kp in new_keypoints:  # type: ignore
                                kp['keypoint_category_id'] = kpcat_id_map.get(
                                    kp['keypoint_category_id'], None
                                )
                            new_annot['keypoints'] = new_keypoints
                    merged['annotations'].append(new_annot)

                # Mark that we are not allowed to use the same hacked track-ids
                # again
                if disjoint_tracks:
                    hacked_track_id_map.block_seen()
            return merged

        # New behavior is simplified and I believe it is correct
        def longest_common_prefix(items, sep='/'):
            """
            Example:
                >>> items = [
                >>>     '/foo/bar/always/the/same/set1/img1.png',
                >>>     '/foo/bar/always/the/same/set1/img2.png',
                >>>     '/foo/bar/always/the/same/set2/img1.png',
                >>>     '/foo/bar/always/the/same/set2/img2.png',
                >>>     '/foo/baz/file1.txt',
                >>> ]
                >>> sep = '/'
                >>> longest_common_prefix(items, sep=sep)
                >>> longest_common_prefix(items[:-1], sep=sep)
            """
            # I would use a trie, but I don't know if pygtrie can do this efficiently
            # (not that this is efficient)
            freq = defaultdict(lambda: 0)
            for item in items:
                path = tuple(item.split(sep))
                for i in range(len(path)):
                    prefix = path[: i + 1]
                    freq[prefix] += 1
            # Find the longest common prefix
            if len(freq) == 0:
                longest_prefix = ''
            else:
                value, freq = max(freq.items(), key=lambda kv: (kv[1], len(kv[0])))
                longest_prefix = sep.join(value)
            return longest_prefix

        dset_roots = [dset.bundle_dpath for dset in others]
        dset_roots = [normpath(r) if r is not None else None for r in dset_roots]
        items = [join('.', p) for p in dset_roots]  # type: ignore
        common_root = longest_common_prefix(items, sep=os.path.sep)

        relative_dsets = [
            (relpath(normpath(d.bundle_dpath), common_root), str(d.fpath), d.dataset)
            for d in others
        ]

        merged = _coco_union(relative_dsets, common_root)

        kwargs['bundle_dpath'] = common_root
        new_dset = cls(merged, **kwargs)
        return new_dset

    def subset(
        self,
        gids: list[int] = None,  # type: ignore
        video_ids: list[int] | None = None,
        copy: bool = False,
        autobuild: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Return a subset of the larger coco dataset by specifying which images
        to port. All annotations in those images will be taken.

        Args:
            gids (List[int] | None):
                image-ids to copy into a new dataset.

            video_ids (List[int] | None):
                list of video ids to copy into the new dataset.
                This is a convenience argument that simply selects all image
                ids associated with the given videos.

            copy (bool):
                if True, makes a deep copy of all nested attributes, otherwise
                makes a shallow copy.  Defaults to True.

            autobuild (bool):
                if True will automatically build the fast lookup index.
                Defaults to True.

            **kwargs:
                Only used for new cannonical names:
                    image_id := gids

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> gids = [1, 3]
            >>> sub_dset = self.subset(gids)
            >>> assert len(self.index.gid_to_aids) == 3
            >>> assert len(sub_dset.gid_to_aids) == 2

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('vidshapes2')
            >>> gids = [1, 2]
            >>> sub_dset = self.subset(gids, copy=True)
            >>> assert len(sub_dset.index.videos) == 1
            >>> assert len(self.index.videos) == 2
            >>> assert len(sub_dset.index.tracks) == 2
            >>> assert len(self.index.tracks) == 4

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('vidshapes2')
            >>> sub_dset = self.subset(video_ids=[1], copy=True)
            >>> assert len(sub_dset.index.videos) == 1
            >>> assert len(self.index.videos) == 2

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> sub1 = self.subset([1])
            >>> sub2 = self.subset([2])
            >>> sub3 = self.subset([3])
            >>> others = [sub1, sub2, sub3]
            >>> rejoined = kwcoco.CocoDataset.union(*others)
            >>> assert len(sub1.anns) == 9
            >>> assert len(sub2.anns) == 2
            >>> assert len(sub3.anns) == 0
            >>> assert rejoined.basic_stats() == self.basic_stats()
        """
        new_dataset: dict[str, Any] = dict([(k, []) for k in self.dataset])
        new_dataset['categories'] = self.dataset['categories']
        new_dataset['info'] = self.dataset.get('info', [])
        new_dataset['licenses'] = self.dataset.get('licenses', [])

        from kwcoco.util.util_deprecate import migrate_argnames

        # TODO: schedule a time to switch the explicit args to the cannonical
        # values.
        cannonical = migrate_argnames(
            aliases={
                'image_ids': ['gids'],
            },
            explicit_args=dict(gids=gids),
            kwargs=kwargs,
            warn_non_cannon=False,
        )
        gids = cannonical['image_ids']

        if gids is None:
            gids = []

        if video_ids is not None:
            gids = gids + list(ub.flatten(self.videos(video_ids).images.lookup('id')))

        chosen_gids = sorted(set(gids))

        chosen_imgs = list(ub.take(self.imgs, chosen_gids))
        new_dataset['images'] = chosen_imgs

        if 'keypoint_categories' in self.dataset:
            new_dataset['keypoint_categories'] = self.dataset['keypoint_categories']

        if 'videos' in self.dataset:
            # TODO: Take only videos with image support?
            vidids = sorted(
                {
                    video_id
                    for img in chosen_imgs
                    if (video_id := img.get('video_id', None)) is not None
                }
            )
            chosen_vids = list(ub.take(self.index.videos, vidids))  # type: ignore
            new_dataset['videos'] = chosen_vids

        sub_aids = sorted(
            [aid for gid in chosen_gids for aid in self.index.gid_to_aids.get(gid, [])]  # type: ignore
        )
        new_dataset['annotations'] = list(ub.take(self.index.anns, sub_aids))  # type: ignore

        if 'tracks' in self.dataset:
            sub_track_ids = sorted(
                set(ann.get('track_id', None) for ann in new_dataset['annotations'])
                - {None}
            )
            new_dataset['tracks'] = list(self.tracks(sub_track_ids).objs_iter())

        new_dataset['img_root'] = self.dataset.get('img_root', None)  # type: ignore

        if copy:
            from copy import deepcopy

            new_dataset = deepcopy(new_dataset)

        sub_dset = CocoDataset(
            new_dataset, bundle_dpath=self.bundle_dpath, autobuild=autobuild
        )
        return sub_dset

    def view_sql(
        self,
        force_rewrite: bool = False,
        memory: bool = False,
        backend: str = 'sqlite',
        sql_db_fpath: str | PathLike | None = None,
    ) -> Any:
        """
        Create a cached SQL interface to this dataset suitable for large scale
        multiprocessing use cases.

        Args:
            force_rewrite (bool):
                if True, forces an update to any existing cache file on disk

            memory (bool):
                if True, the database is constructed in memory.

            backend (str): sqlite or postgresql

            sql_db_fpath (str | PathLike | None): overrides the database uri

        Note:
            This view cache is experimental and currently depends on the
            timestamp of the file pointed to by ``self.fpath``. In other words
            dont use this on in-memory datasets.

        CommandLine:
            KWCOCO_WITH_POSTGRESQL=1 xdoctest -m /home/joncrall/code/kwcoco/kwcoco/coco_dataset.py CocoDataset.view_sql

        Example:
            >>> # xdoctest: +REQUIRES(module:sqlalchemy)
            >>> # xdoctest: +REQUIRES(env:KWCOCO_WITH_POSTGRESQL)
            >>> # xdoctest: +REQUIRES(module:psycopg2)
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes32')
            >>> postgres_dset = dset.view_sql(backend='postgresql', force_rewrite=True)
            >>> sqlite_dset = dset.view_sql(backend='sqlite', force_rewrite=True)
            >>> list(dset.anns.keys())
            >>> list(postgres_dset.anns.keys())
            >>> list(sqlite_dset.anns.keys())

        Ignore:
            import timerit
            ti = timerit.Timerit(100, bestof=10, verbose=2)
            for timer in ti.reset('dct_dset'):
                dset.annots().detections
            for timer in ti.reset('postgresql'):
                postgres_dset.annots().detections
            for timer in ti.reset('sqlite'):
                sqlite_dset.annots().detections

            ub.udict(sql_dset.annots().objs[0]) - {'segmentation'}
            ub.udict(dct_dset.annots().objs[0]) - {'segmentation'}
        """
        from kwcoco.coco_sql_dataset import cached_sql_coco_view

        if sql_db_fpath is None:
            if memory:
                sql_db_fpath = ':memory:'
        sql_dset = cached_sql_coco_view(
            dset=self,
            sql_db_fpath=sql_db_fpath,
            force_rewrite=force_rewrite,
            backend=backend,
        )
        return sql_dset


def demo_coco_data() -> Any:
    """
    Simple data for testing.

    This contains several non-standard fields, which help ensure robustness of
    functions tested with this data. For more compliant demodata see the
    ``kwcoco.demodata`` submodule.

    Example:
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwcoco
        >>> from kwcoco.coco_dataset import demo_coco_data
        >>> dataset = demo_coco_data()
        >>> self = kwcoco.CocoDataset(dataset, tag='demo')
        >>> import kwplot
        >>> kwplot.autompl()
        >>> self.show_image(gid=1)
        >>> kwplot.show_if_requested()
    """
    import kwimage
    from kwimage.im_demodata import _TEST_IMAGES
    from os.path import commonprefix

    # FIXME: be robust to broken urls
    test_imgs_keys = ['astro', 'carl', 'stars']
    urls = {k: _TEST_IMAGES[k]['url'] for k in test_imgs_keys}
    gpaths = {k: kwimage.grab_test_image_fpath(k) for k in test_imgs_keys}
    img_root = commonprefix(list(gpaths.values()))

    gpath1: str | PathLike
    gpath2: str | PathLike
    gpath3: str | PathLike
    url1: str
    url2: str
    url3: str

    gpath1, gpath2, gpath3 = ub.take(gpaths, test_imgs_keys)
    url1, url2, url3 = ub.take(urls, test_imgs_keys)

    # Make file names relative for consistent testing purpose
    gname1 = relpath(gpath1, img_root)
    gname2 = relpath(gpath2, img_root)
    gname3 = relpath(gpath3, img_root)

    dataset = {
        # TODO: need to deprecate this somehow?
        # It probably does make sense to allow this to be specified in the
        # dictionary itself if we are constructing a CocoDataset class from it,
        # but we likely don't want to save it (or if we do it must be
        # constantly updated to be correct)
        'img_root': img_root,
        'categories': [
            {
                'id': 1,
                'name': 'astronaut',
                'supercategory': 'human',
            },
            {'id': 2, 'name': 'rocket', 'supercategory': 'object'},
            {'id': 3, 'name': 'helmet', 'supercategory': 'object'},
            {
                'id': 4,
                'name': 'mouth',
                'supercategory': 'human',
                'keypoints': [
                    'mouth-right-corner',
                    'mouth-right-bot',
                    'mouth-left-bot',
                    'mouth-left-corner',
                ],
                'skeleton': [[0, 1]],
            },
            {
                'id': 5,
                'name': 'star',
                'supercategory': 'object',
                'keypoints': ['star-center'],
                'skeleton': [],
            },
            {'id': 6, 'name': 'astronomer', 'supercategory': 'human'},
            {'id': 7, 'name': 'astroturf', 'supercategory': 'object'},
            {
                'id': 8,
                'name': 'human',
                'keypoints': ['left-eye', 'right-eye'],
                'skeleton': [[0, 1]],
            },
        ],
        'images': [
            {'id': 1, 'file_name': gname1, 'url': url1},
            {'id': 2, 'file_name': gname2, 'url': url2},
            {'id': 3, 'file_name': gname3, 'url': url3},
        ],
        'annotations': [
            {
                'id': 1,
                'image_id': 1,
                'category_id': 1,
                'bbox': [10, 10, 360, 490],
                'keypoints': [247, 101, 2, 202, 100, 2],
                'segmentation': [
                    [
                        40,
                        509,
                        26,
                        486,
                        20,
                        419,
                        28,
                        334,
                        51,
                        266,
                        85,
                        229,
                        102,
                        216,
                        118,
                        197,
                        125,
                        176,
                        148,
                        151,
                        179,
                        147,
                        182,
                        134,
                        174,
                        128,
                        166,
                        115,
                        156,
                        94,
                        155,
                        64,
                        162,
                        48,
                        193,
                        34,
                        197,
                        26,
                        210,
                        21,
                        231,
                        14,
                        265,
                        24,
                        295,
                        49,
                        300,
                        90,
                        297,
                        111,
                        280,
                        126,
                        277,
                        132,
                        266,
                        137,
                        264,
                        152,
                        255,
                        164,
                        256,
                        174,
                        283,
                        195,
                        301,
                        220,
                        305,
                        234,
                        338,
                        262,
                        350,
                        286,
                        360,
                        326,
                        363,
                        351,
                        324,
                        369,
                        292,
                        404,
                        280,
                        448,
                        276,
                        496,
                        280,
                        511,
                    ]
                ],
            },
            {'id': 2, 'image_id': 1, 'category_id': 2, 'bbox': [350, 5, 130, 290]},
            {'id': 3, 'image_id': 1, 'category_id': 3, 'line': [326, 369, 500, 500]},
            {
                'id': 4,
                'image_id': 1,
                'category_id': 4,
                'keypoints': [
                    202,
                    139,
                    2,
                    215,
                    150,
                    2,
                    229,
                    150,
                    2,
                    244,
                    142,
                    2,
                ],
            },
            {'id': 5, 'image_id': 1, 'category_id': 5, 'keypoints': [37, 65, 1]},
            {'id': 6, 'image_id': 1, 'category_id': 5, 'keypoints': [37, 16, 1]},
            {'id': 7, 'image_id': 1, 'category_id': 5, 'keypoints': [3, 9, 1]},
            {'id': 8, 'image_id': 1, 'category_id': 5, 'keypoints': [2, 111, 1]},
            {'id': 9, 'image_id': 1, 'category_id': 5, 'keypoints': [2, 60, 1]},
            {'id': 10, 'image_id': 2, 'category_id': 6, 'bbox': [37, 6, 230, 240]},
            {'id': 11, 'image_id': 2, 'category_id': 4, 'bbox': [156, 130, 45, 18]},
        ],
        'licenses': [],
        'info': [],
    }
    return dataset


if __name__ == '__main__':
    r"""
    CommandLine:
        xdoctest kwcoco.coco_dataset all
    """
    import xdoctest

    xdoctest.doctest_module(__file__)
