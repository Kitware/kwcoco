"""
Defines the CocoImage class which is an object oriented way of manipulating
data pointed to by a COCO image dictionary.

Notably this provides the ``.imdelay`` method for delayed image loading ( which
enables things like fast loading of subimage-regions / coarser scales in images
that contain tiles / overviews - e.g. Cloud Optimized Geotiffs or COGs (Medical
image formats may be supported in the future).

TODO:
    This file no longer is only images, it has logic for generic single-class
    objects. It should be refactored into coco_objects0d.py or something.
"""
import ubelt as ub
import os
import numpy as np
from os.path import join
from kwcoco.util.util_deprecate import deprecated_function_alias
from kwcoco.util.dict_proxy2 import AliasedDictProxy


from delayed_image.channel_spec import FusedChannelSpec
from delayed_image.channel_spec import ChannelSpec
from delayed_image import DelayedNans
from delayed_image import DelayedChannelConcat
from delayed_image import DelayedLoad, DelayedIdentity


__docstubs__ = """
from delayed_image.channel_spec import FusedChannelSpec
from kwcoco.coco_objects1d import Annots
"""


DEFAULT_RESOLUTION_KEYS = {
    'resolution',
    'target_gsd',  # only exists as a convenience for other projects. Remove in the future.
}


class _CocoObject(AliasedDictProxy, ub.NiceRepr):
    """
    General coco scalar object
    """
    __alias_to_primary__ = {}

    def __init__(self, obj, dset=None, bundle_dpath=None):
        self._proxy = obj
        self.dset = dset
        self._bundle_dpath = bundle_dpath

    @property
    def bundle_dpath(self):
        if self.dset is not None:
            return self.dset.bundle_dpath
        else:
            return self._bundle_dpath

    @bundle_dpath.setter
    def bundle_dpath(self, value):
        self._bundle_dpath = value

    def detach(self):
        """
        Removes references to the underlying coco dataset, but keeps special
        information such that it wont be needed.
        """
        self._bundle_dpath = self.bundle_dpath
        self.dset = None
        return self


class CocoImage(_CocoObject):
    """
    An object-oriented representation of a coco image.

    It provides helper methods that are specific to a single image.

    This operates directly on a single coco image dictionary, but it can
    optionally be connected to a parent dataset, which allows it to use
    CocoDataset methods to query about relationships and resolve pointers.

    This is different than the Images class in coco_object1d, which is just a
    vectorized interface to multiple objects.

    Example:
        >>> import kwcoco
        >>> dset1 = kwcoco.CocoDataset.demo('shapes8')
        >>> dset2 = kwcoco.CocoDataset.demo('vidshapes8-multispectral')

        >>> self = kwcoco.CocoImage(dset1.imgs[1], dset1)
        >>> print('self = {!r}'.format(self))
        >>> print('self.channels = {}'.format(ub.urepr(self.channels, nl=1)))

        >>> self = kwcoco.CocoImage(dset2.imgs[1], dset2)
        >>> print('self.channels = {}'.format(ub.urepr(self.channels, nl=1)))
        >>> self.primary_asset()
        >>> assert 'auxiliary' in self
    """

    __alias_to_primary__ = {
        # In the future we will switch assets to be primary.
        'assets': 'auxiliary',
    }

    def __init__(self, img, dset=None):
        super().__init__(img, dset=dset)
        self.img = img
        self._video = None

    @classmethod
    def from_gid(cls, dset, gid):
        img = dset.index.imgs[gid]
        self = cls(img, dset=dset)
        return self

    @property
    def video(self):
        """
        Helper to grab the video for this image if it exists
        """
        if self._video is None and self.dset is not None:
            vidid = self.img.get('video_id', None)
            if vidid is None:
                video = None
            else:
                video = self.dset.index.videos[vidid]
        else:
            video = self._video
        return video

    @video.setter
    def video(self, value):
        # TODO: ducktype with an object
        self._video = value

    @property
    def name(self):
        return self['name']

    def detach(self):
        """
        Removes references to the underlying coco dataset, but keeps special
        information such that it wont be needed.
        """
        self._video = self.video
        return super().detach()

    @property
    def assets(self):
        """
        Convenience wrapper around :func: `CocoImage.iter_assets`.
        """
        return list(self.iter_assets())

    @ub.memoize_property
    def datetime(self):
        """
        Try to get datetime information for this image. Not always possible.

        Returns:
            datetime.datetime | None
        """
        from kwutil import util_time
        candidate_keys = [
            'datetime',
            'timestamp',
            'date_captured'
        ]
        for k in candidate_keys:
            v = self.img.get(k)
            if v is not None:
                return util_time.coerce_datetime(v)
        raise KeyError(f'No keys {candidate_keys} to coerce datetime '
                       'found in {list(self.img.keys())}')

    def annots(self):
        """
        Returns:
            Annots: a 1d annotations object referencing annotations in this image
        """
        return self.dset.annots(image_id=self.img['id'])

    def __nice__(self):
        """
        Example:
            >>> from kwcoco.coco_image import *  # NOQA
            >>> import kwcoco
            >>> with ub.CaptureStdout() as cap:
            ...     dset = kwcoco.CocoDataset.demo('shapes8')
            >>> self = CocoImage(dset.dataset['images'][0], dset)
            >>> print('self = {!r}'.format(self))

            >>> dset = kwcoco.CocoDataset.demo()
            >>> self = CocoImage(dset.dataset['images'][0], dset)
            >>> print('self = {!r}'.format(self))
        """
        from kwcoco.util.util_truncate import smart_truncate
        from functools import partial
        stats = ub.udict(self.stats())
        stats = stats.map_values(str)
        stats = stats.map_values(
            partial(smart_truncate, max_length=32, trunc_loc=0.5))
        return ub.urepr(stats, compact=1, nl=0)

    def stats(self):
        """
        """
        key_attrname = [
            ('wh', 'dsize'),
            ('n_chan', 'num_channels'),
            ('channels', 'channels'),
            ('name', 'name'),
        ]
        stats = {}
        for key, attrname in key_attrname:
            try:
                stats[key] = getattr(self, attrname)
            except Exception as ex:
                stats[key] = repr(ex)
        if 'channels' in stats:
            if stats['channels'] is not None:
                stats['channels'] = stats['channels'].spec
        return stats

    def __contains__(self, key):
        if '_unstructured' in self._proxy:
            if AliasedDictProxy.__contains__(self, key):
                return True
            return key in self._proxy['_unstructured']
        else:
            return AliasedDictProxy.__contains__(self, key)

    def get(self, key, default=ub.NoParam):
        try:
            return self[key]
        except KeyError:
            if default is ub.NoParam:
                raise
            else:
                return default

    def keys(self):
        """
        Proxy getter attribute for underlying `self.img` dictionary
        """
        if '_unstructured' in self._proxy:
            # SQL compatibility
            _keys = ub.flatten([self._proxy.keys(), self._proxy['_unstructured'].keys()])
            return iter((k for k in _keys if k != '_unstructured'))
        else:
            return self._proxy.keys()

    def __getitem__(self, key):
        """
        Proxy getter attribute for underlying `self.img` dictionary

        CommandLine:
            xdoctest -m kwcoco.coco_image CocoImage.__getitem__

        Example:
            >>> import pytest
            >>> # without _unstructured populated
            >>> import kwcoco
            >>> self = kwcoco.CocoImage({'foo': 1})
            >>> assert self.get('foo') == 1
            >>> assert self.get('foo', None) == 1
            >>> # with _unstructured populated
            >>> self = kwcoco.CocoImage({'_unstructured': {'foo': 1}})
            >>> assert self.get('foo') == 1
            >>> assert self.get('foo', None) == 1
            >>> # without _unstructured empty
            >>> self = kwcoco.CocoImage({})
            >>> print('----')
            >>> with pytest.raises(KeyError):
            >>>     self.get('foo')
            >>> assert self.get('foo', None) is None
            >>> # with _unstructured empty
            >>> self = kwcoco.CocoImage({'_unstructured': {'bar': 1}})
            >>> with pytest.raises(KeyError):
            >>>     self.get('foo')
            >>> assert self.get('foo', None) is None
        """
        if AliasedDictProxy.__contains__(self, key):
            return AliasedDictProxy.__getitem__(self, key)
        else:
            _img = self._proxy
            if '_unstructured' in _img:
                # Workaround for sql-view, treat items in "_unstructured" as
                # if they are in the top level image.
                _extra = _img['_unstructured']
                if key in _extra:
                    return _extra[key]
                else:
                    raise KeyError(key)
            else:
                raise KeyError(key)

    @property
    def channels(self):
        # from delayed_image.channel_spec import FusedChannelSpec
        # from delayed_image.channel_spec import ChannelSpec
        img_parts = []
        for obj in self.iter_asset_objs():
            obj_parts = obj.get('channels', None)
            if obj_parts is not None:
                # obj_chan = FusedChannelSpec.coerce(obj_parts).normalize()
                obj_chan = FusedChannelSpec.coerce(obj_parts)
                img_parts.append(obj_chan.spec)
        if not img_parts:
            return None
            # return ChannelSpec.coerce('*')
        spec = ChannelSpec(','.join(img_parts))
        return spec

    @property
    def n_assets(self):
        """
        The number of on-disk files associated with this coco image
        """
        # Hacked because we have too many ways of having assets right now
        # really need a single assets table.
        has_main = int(self.get('file_name', None) is not None)
        num_group1 = len(self.img.get('assets', None) or [])
        num_group2 = len(self.img.get('auxiliary', None) or [])
        total = has_main + num_group1 + num_group2
        return total

    @property
    def num_channels(self):
        return self.channels.numel()
        # return sum(map(len, self.channels.streams()))

    @property
    def dsize(self):
        width = self.img.get('width', None)
        height = self.img.get('height', None)
        return width, height

    def image_filepath(self):
        """
        Note: this only returns a file path if it is directly associated with
        this image, it does not respect assets. This is intended to duck-type
        the method of a CocoAsset. Use :func:`primary_image_filepath` for this
        instead.
        """
        if self.bundle_dpath is None:
            raise Exception('Bundle dpath must be populated to use this method')
            # return self['file_name']
        else:
            return ub.Path(self.bundle_dpath) / self['file_name']

    def primary_image_filepath(self, requires=None):
        dpath = ub.Path(self.bundle_dpath)
        fname = self.primary_asset()['file_name']
        fpath = dpath / fname
        return fpath

    def primary_asset(self, requires=None, as_dict=True):
        """
        Compute a "main" image asset.

        Note:
            Uses a heuristic.

            * First, try to find the auxiliary image that has with the smallest
            distortion to the base image (if known via warp_aux_to_img)

            * Second, break ties by using the largest image if w / h is known

            * Last, if previous information not available use the first
              auxiliary image.

        Args:
            requires (List[str] | None):
                list of attribute that must be non-None to consider an object
                as the primary one.

            as_dict (bool):
                if True the return type is a raw dictionary. Otherwise use a newer
                object-oriented wrapper that should be duck-type swappable.
                In the future this default will change to False.

        Returns:
            None | dict : the asset dict or None if it is not found

        TODO:
            - [ ] Add in primary heuristics

        Example:
            >>> import kwarray
            >>> from kwcoco.coco_image import *  # NOQA
            >>> rng = kwarray.ensure_rng(0)
            >>> def random_asset(name, w=None, h=None):
            >>>     return {'file_name': name, 'width': w, 'height': h}
            >>> self = CocoImage({
            >>>     'auxiliary': [
            >>>         random_asset('1'),
            >>>         random_asset('2'),
            >>>         random_asset('3'),
            >>>     ]
            >>> })
            >>> assert self.primary_asset()['file_name'] == '1'
            >>> self = CocoImage({
            >>>     'auxiliary': [
            >>>         random_asset('1'),
            >>>         random_asset('2', 3, 3),
            >>>         random_asset('3'),
            >>>     ]
            >>> })
            >>> assert self.primary_asset()['file_name'] == '2'
            >>> #
            >>> # Test new object oriented output
            >>> self = CocoImage({
            >>>     'file_name': 'foo',
            >>>     'assets': [
            >>>         random_asset('1'),
            >>>         random_asset('2'),
            >>>         random_asset('3'),
            >>>     ],
            >>> })
            >>> assert self.primary_asset(as_dict=False) is self
            >>> self = CocoImage({
            >>>     'assets': [
            >>>         random_asset('1'),
            >>>         random_asset('3'),
            >>>     ],
            >>>     'auxiliary': [
            >>>         random_asset('1'),
            >>>         random_asset('2', 3, 3),
            >>>         random_asset('3'),
            >>>     ]
            >>> })
            >>> assert self.primary_asset(as_dict=False)['file_name'] == '2'
        """
        import kwimage
        if requires is None:
            requires = []
        img = self.img
        has_base_image = img.get('file_name', None) is not None
        candidates = []

        if has_base_image:
            obj = img
            if all(k in obj for k in requires):
                # Return the base image if we can
                if as_dict:
                    return obj
                else:
                    return self

        # Choose "best" auxiliary image based on a heuristic.
        eye = kwimage.Affine.eye().matrix
        asset_objs = img.get('auxiliary', img.get('assets', [])) or []
        for idx, obj in enumerate(asset_objs):
            # Take frobenius norm to get "distance" between transform and
            # the identity. We want to find the auxiliary closest to the
            # identity transform.
            warp_img_from_asset = kwimage.Affine.coerce(obj.get('warp_aux_to_img', None))
            fro_dist = np.linalg.norm(warp_img_from_asset - eye, ord='fro')
            w = obj.get('width', None) or 0
            h = obj.get('height', None) or 0
            if all(k in obj for k in requires):
                candidates.append({
                    'idx': idx,
                    'area': w * h,
                    'fro_dist': fro_dist,
                    'obj': obj,
                })

        if len(candidates) == 0:
            return None

        idx = ub.argmin(
            candidates, key=lambda val: (
                val['fro_dist'], -val['area'], val['idx'])
        )
        obj = candidates[idx]['obj']
        if as_dict:
            return obj
        else:
            return CocoAsset(obj, bundle_dpath=self.bundle_dpath)

    def iter_image_filepaths(self, with_bundle=True):
        """
        Could rename to iter_asset_filepaths

        Args:
            with_bundle (bool):
                If True, prepends the bundle dpath to fully specify the path.
                Otherwise, just returns the registered string in the file_name
                attribute of each asset.  Defaults to True.

        Yields:
            ub.Path
        """
        dpath = ub.Path(self.bundle_dpath)
        for obj in self.iter_asset_objs():
            fname = obj.get('file_name')
            fpath = dpath / fname
            yield fpath

    def iter_assets(self):
        """
        Iterate through assets (which could include the image itself it points to a file path).

        Object-oriented alternative to :func:`CocoImage.iter_asset_objs`

        Yields:
            CocoImage | CocoAsset:
                an asset object (or image object if it points to a file)

        Example:
            >>> import kwcoco
            >>> coco_img = kwcoco.CocoImage({'width': 128, 'height': 128})
            >>> assert len(list(coco_img.iter_assets())) == 0
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> self = dset.coco_image(1)
            >>> assert len(list(self.iter_assets())) > 1
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8')
            >>> self = dset.coco_image(1)
            >>> assert list(self.iter_assets()) == [self]
        """
        img = self.img
        has_base_image = img.get('file_name', None) is not None
        if has_base_image:
            yield self
        for obj in img.get('auxiliary', None) or []:
            yield CocoAsset(obj, self.bundle_dpath)
        for obj in img.get('assets', None) or []:
            yield CocoAsset(obj, self.bundle_dpath)

    def iter_asset_objs(self):
        """
        Iterate through base + auxiliary dicts that have file paths

        Note:
            In most cases prefer :func:`iter_assets` instead.

        Yields:
            dict: an image or auxiliary dictionary
        """
        img = self.img
        has_base_image = img.get('file_name', None) is not None
        if has_base_image:
            yield img
        for obj in img.get('auxiliary', None) or []:
            yield obj
        for obj in img.get('assets', None) or []:
            yield obj

    def find_asset(self, channels):
        """
        Find the asset dictionary with the specified channels

        Args:
            channels (str | FusedChannelSpec):
                channel names the asset must have.

        Returns:
            CocoImage | CocoAsset

        Example:
            >>> # A pathological example (test-case)
            >>> import kwcoco
            >>> self = kwcoco.CocoImage({
            >>>     'file_name': 'raw',
            >>>     'channels': 'red|green|blue',
            >>>     'assets': [
            >>>         {'file_name': '1', 'channels': 'spam'},
            >>>         {'file_name': '2', 'channels': 'eggs|jam'},
            >>>     ],
            >>>     'auxiliary': [
            >>>         {'file_name': '3', 'channels': 'foo'},
            >>>         {'file_name': '4', 'channels': 'bar|baz'},
            >>>     ]
            >>> })
            >>> assert self.find_asset('blah') is None
            >>> assert self.find_asset('red|green|blue') is self
            >>> self.find_asset('foo')['file_name'] == '3'
            >>> self.find_asset('baz')['file_name'] == '4'

        Example:
            >>> # A more standard test case
            >>> # In this case there is is a top-level base image, as well as
            >>> # additional assets.
            >>> import kwcoco
            >>> self = kwcoco.CocoImage({
            >>>     'file_name': 'path/to/rgbdata.jpg',
            >>>     'channels': 'red|green|blue',
            >>>     'assets': [
            >>>         {'file_name': 'path/to/depth/data.png', 'channels': 'depth'},
            >>>         {'file_name': 'path/to/opticalflow/data.tif', 'channels': 'flowx|flowy'},
            >>>     ],
            >>> })
            >>> # Searching for an asset that does not exist returns None
            >>> assert self.find_asset('does-not-exist') is None
            >>> # Searching for an asset finds the dictionary containing the channel
            >>> assert self.find_asset('flowy')['channels'] == 'flowx|flowy'
            >>> # The top level dict is considered an asset if it has channels
            >>> assert self.find_asset('red') is self
            >>> #
            >>> #
            >>> # Another common case is when the top-level dictionary has no
            >>> # file_name, and all image information is pointed to by the assets.
            >>> self = kwcoco.CocoImage({
            >>>     'assets': [
            >>>         {'file_name': 'path/to/rgbdata.jpg', 'channels': 'red|green|blue'},
            >>>         {'file_name': 'path/to/depth/data.png', 'channels': 'depth'},
            >>>         {'file_name': 'path/to/opticalflow/data.tif', 'channels': 'flowx|flowy'},
            >>>     ],
            >>> })
            >>> # Searching for an asset that does not exist returns None
            >>> assert self.find_asset('does-not-exist') is None
            >>> # Searching for an asset finds the dictionary containing the channel
            >>> assert self.find_asset('flowy|flowx')['channels'] == 'flowx|flowy'
            >>> assert self.find_asset('flowy|flowx')['channels'] == 'flowx|flowy'
            >>> # The top level dict is considered an asset if it has channels
            >>> assert self.find_asset('red') is not self
        """
        obj = self.find_asset_obj(channels)
        if obj is not None:
            if obj is self.img:
                return self
            return CocoAsset(obj, bundle_dpath=self.bundle_dpath)

    def find_asset_obj(self, channels):
        """
        Find the asset dictionary with the specified channels

        In most cases use :func:`CocoImge.find_asset` instead.

        Example:
            >>> import kwcoco
            >>> coco_img = kwcoco.CocoImage({'width': 128, 'height': 128})
            >>> coco_img.add_auxiliary_item(
            >>>     'rgb.png', channels='red|green|blue', width=32, height=32)
            >>> assert coco_img.find_asset_obj('red') is not None
            >>> assert coco_img.find_asset_obj('green') is not None
            >>> assert coco_img.find_asset_obj('blue') is not None
            >>> assert coco_img.find_asset_obj('red|blue') is not None
            >>> assert coco_img.find_asset_obj('red|green|blue') is not None
            >>> assert coco_img.find_asset_obj('red|green|blue') is not None
            >>> assert coco_img.find_asset_obj('black') is None
            >>> assert coco_img.find_asset_obj('r') is None

        Example:
            >>> # Test with concise channel code
            >>> import kwcoco
            >>> coco_img = kwcoco.CocoImage({'width': 128, 'height': 128})
            >>> coco_img.add_auxiliary_item(
            >>>     'msi.png', channels='foo.0:128', width=32, height=32)
            >>> assert coco_img.find_asset_obj('foo') is None
            >>> assert coco_img.find_asset_obj('foo.3') is not None
            >>> assert coco_img.find_asset_obj('foo.3:5') is not None
            >>> assert coco_img.find_asset_obj('foo.3000') is None

        Example:
            >>> # Test a mallformed case. If using this function each
            >>> # dictionary with file_name must have channels
            >>> import kwcoco
            >>> self = kwcoco.CocoImage({
            >>>     'file_name': 'path/to/rgbdata.jpg',
            >>>     'channels': None,
            >>>     'assets': [
            >>>         {'file_name': 'path/to/depth/data.png', 'channels': 'depth'},
            >>>         {'file_name': 'path/to/opticalflow/data.tif', 'channels': 'flowx|flowy'},
            >>>     ],
            >>> })
            >>> # Searching for an asset that does not exist returns None
            >>> import pytest
            >>> with pytest.raises(TypeError):
            >>>     self.find_asset_obj('depth')
        """
        # from delayed_image.channel_spec import FusedChannelSpec
        channels = FusedChannelSpec.coerce(channels)
        for obj in self.iter_asset_objs():
            obj_channels = FusedChannelSpec.coerce(obj['channels'])
            if (obj_channels & channels).numel():
                return obj

    def _assets_key(self):
        """
        Internal helper for transition from auxiliary -> assets in the image
        spec
        """
        if 'auxiliary' in self:
            return 'auxiliary'
        elif 'assets' in self:
            return 'assets'
        else:
            return 'auxiliary'

    def add_annotation(self, **ann):
        """
        Adds an annotation to this image.

        This is a convenience method, and requires that this CocoImage is still
        connected to a parent dataset.

        Args:
            **ann: annotation attributes (e.g. bbox, category_id)

        Returns:
            int: the new annotation id

        SeeAlso:
            :func:`kwcoco.CocoDataset.add_annotation`
        """
        if self.dset is None:
            raise RuntimeError(
                'Can only add an annotation through a CocoImage '
                'if it is connected to its parent CocoDataset')
        return self.dset.add_annotation(image_id=self.img['id'], **ann)

    def add_asset(self, file_name=None, channels=None, imdata=None,
                  warp_aux_to_img=None, width=None, height=None,
                  imwrite=False, image_id=None, **kw):
        """
        Adds an auxiliary / asset item to the image dictionary.

        This operation can be done purely in-memory (the default), or the image
        data can be written to a file on disk (via the imwrite=True flag).

        Args:
            file_name (str | PathLike | None):
                The name of the file relative to the bundle directory. If
                unspecified, imdata must be given.

            channels (str | kwcoco.FusedChannelSpec | None):
                The channel code indicating what each of the bands represents.
                These channels should be disjoint wrt to the existing data in
                this image (this is not checked).

            imdata (ndarray | None):
                The underlying image data this auxiliary item represents.  If
                unspecified, it is assumed file_name points to a path on disk
                that will eventually exist. If imdata, file_name, and the
                special imwrite=True flag are specified, this function will
                write the data to disk.

            warp_aux_to_img (kwimage.Affine | None):
                The transformation from this auxiliary space to image space.
                If unspecified, assumes this item is related to image space by
                only a scale factor.

            width (int | None):
                Width of the data in auxiliary space (inferred if unspecified)

            height (int | None):
                Height of the data in auxiliary space (inferred if unspecified)

            imwrite (bool):
                If specified, both imdata and file_name must be specified, and
                this will write the data to disk. Note: it it recommended that
                you simply call imwrite yourself before or after calling this
                function. This lets you better control imwrite parameters.

            image_id (int | None):
                An asset dictionary contains an image-id, but it should *not*
                be specified here. If it is, then it *must* agree with this
                image's id.

            **kw : stores arbitrary key/value pairs in this new asset.

        TODO:
            - [ ] Allow imwrite to specify an executor that is used to
            return a Future so the imwrite call does not block.

        Example:
            >>> from kwcoco.coco_image import *  # NOQA
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> coco_img = dset.coco_image(1)
            >>> imdata = np.random.rand(32, 32, 5)
            >>> channels = kwcoco.FusedChannelSpec.coerce('Aux:5')
            >>> coco_img.add_asset(imdata=imdata, channels=channels)

        Example:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset()
            >>> gid = dset.add_image(name='my_image_name', width=200, height=200)
            >>> coco_img = dset.coco_image(gid)
            >>> coco_img.add_asset('path/img1_B0.tif', channels='B0', width=200, height=200)
            >>> coco_img.add_asset('path/img1_B1.tif', channels='B1', width=200, height=200)
            >>> coco_img.add_asset('path/img1_B2.tif', channels='B2', width=200, height=200)
            >>> coco_img.add_asset('path/img1_TCI.tif', channels='r|g|b', width=200, height=200)
        """
        import kwimage
        from os.path import isabs, join  # NOQA
        # from delayed_image.channel_spec import FusedChannelSpec

        img = self.img

        if imdata is None and file_name is None:
            raise ValueError('must specify file_name or imdata')

        # Check type of resolution inputs.
        if not isinstance(width, int) and width is not None:
            raise TypeError(f'"width" input is neither an int or None variable but type: "{type(width)}"')

        if not isinstance(height, int) and height is not None:
            raise TypeError(f'"height" input is neither an int or None variable but type: "{type(height)}"')

        # Infer resolution inputs from image data.
        if width is None and imdata is not None:
            width = imdata.shape[1]

        if height is None and imdata is not None:
            height = imdata.shape[0]

        if warp_aux_to_img is None:
            img_width = img.get('width', None)
            img_height = img.get('height', None)
            if img_width is None or img_height is None:
                raise ValueError('Parent image canvas has an unknown size. '
                                 'Need to set width/height')
            if width is None or height is None:
                raise ValueError('Unable to infer warp_aux_to_img without width')
            # Assume we can just scale up the auxiliary data to match the image
            # space unless the user says otherwise
            warp_aux_to_img = kwimage.Affine.scale((
                img_width / width, img_height / height))
        else:
            warp_aux_to_img = kwimage.Affine.coerce(warp_aux_to_img)

        # Normalize for json serializability
        if channels is not None:
            channels = FusedChannelSpec.coerce(channels).spec

        if file_name is not None:
            file_name = os.fspath(file_name)

        # Make the asset info dict
        parent_image_id = img.get('id', None)
        if image_id is not None:
            if parent_image_id is not None:
                assert image_id == parent_image_id, (
                    f'The specified image_id ({image_id}) did not match the '
                    f'parent image id ({parent_image_id}) property.'
                )
        else:
            image_id = parent_image_id

        obj = {
            'image_id': image_id,  # for when assets move to their own table
            'file_name': file_name,
            'height': height,
            'width': width,
            'channels': channels,
            'warp_aux_to_img': warp_aux_to_img.concise(),
        }
        if imdata is not None:
            if imwrite:
                if __debug__ and file_name is None:
                    raise ValueError(
                        'file_name must be given if imwrite is True')
                # if self.dset is None:
                #     fpath = file_name
                #     if not isabs(fpath):
                #         raise ValueError(ub.paragraph(
                #             '''
                #             Got relative file_name, but no dataset is attached
                #             to this coco image. Attach a dataset or use an
                #             absolute path.
                #             '''))
                # else:
                fpath = join(self.bundle_dpath, file_name)
                kwimage.imwrite(fpath, imdata)
            else:
                obj['imdata'] = imdata

        obj.update(**kw)

        assets_key = self._assets_key()
        asset_list = img.get(assets_key, None)
        if asset_list is None:
            asset_list = img[assets_key] = []
        asset_list.append(obj)
        if self.dset is not None:
            self.dset._invalidate_hashid()

    def imdelay(self, channels=None, space='image', resolution=None,
                bundle_dpath=None, interpolation='linear', antialias=True,
                nodata_method=None, RESOLUTION_KEY=None):
        """
        Perform a delayed load on the data in this image.

        The delayed load can load a subset of channels, and perform lazy
        warping operations. If the underlying data is in a tiled format this
        can reduce the amount of disk IO needed to read the data if only a
        small crop or lower resolution view of the data is needed.

        Note:
            This method is experimental and relies on the delayed load
            proof-of-concept.

        Args:
            gid (int): image id to load

            channels (kwcoco.FusedChannelSpec): specific channels to load.
                if unspecified, all channels are loaded.

            space (str):
                can either be "image" for loading in image space, or
                "video" for loading in video space.

            resolution (None | str | float):
                If specified, applies an additional scale factor to the result
                such that the data is loaded at this specified resolution.
                This requires that the image / video has a registered
                resolution attribute and that its units agree with this
                request.

        TODO:
            - [ ] This function could stand to have a better name. Maybe imread
                  with a delayed=True flag? Or maybe just delayed_load?

        Example:
            >>> from kwcoco.coco_image import *  # NOQA
            >>> import kwcoco
            >>> gid = 1
            >>> #
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> self = CocoImage(dset.imgs[gid], dset)
            >>> delayed = self.imdelay()
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize()))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize()))
            >>> #
            >>> dset = kwcoco.CocoDataset.demo('shapes8')
            >>> delayed = dset.coco_image(gid).imdelay()
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize()))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize()))

            >>> crop = delayed.crop((slice(0, 3), slice(0, 3)))
            >>> crop.finalize()

            >>> # TODO: should only select the "red" channel
            >>> dset = kwcoco.CocoDataset.demo('shapes8')
            >>> delayed = CocoImage(dset.imgs[gid], dset).imdelay(channels='r')

            >>> import kwcoco
            >>> gid = 1
            >>> #
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> delayed = dset.coco_image(gid).imdelay(channels='B1|B2', space='image')
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize()))
            >>> delayed = dset.coco_image(gid).imdelay(channels='B1|B2|B11', space='image')
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize()))
            >>> delayed = dset.coco_image(gid).imdelay(channels='B8|B1', space='video')
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize()))

            >>> delayed = dset.coco_image(gid).imdelay(channels='B8|foo|bar|B1', space='video')
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize()))

        Example:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo()
            >>> coco_img = dset.coco_image(1)
            >>> # Test case where nothing is registered in the dataset
            >>> delayed = coco_img.imdelay()
            >>> final = delayed.finalize()
            >>> assert final.shape == (512, 512, 3)

            >>> delayed = coco_img.imdelay()
            >>> final = delayed.finalize()
            >>> print('final.shape = {}'.format(ub.urepr(final.shape, nl=1)))
            >>> assert final.shape == (512, 512, 3)

        Example:
            >>> # Test that delay works when imdata is stored in the image
            >>> # dictionary itself.
            >>> from kwcoco.coco_image import *  # NOQA
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> coco_img = dset.coco_image(1)
            >>> imdata = np.random.rand(6, 6, 5)
            >>> imdata[:] = np.arange(5)[None, None, :]
            >>> channels = kwcoco.FusedChannelSpec.coerce('Aux:5')
            >>> coco_img.add_auxiliary_item(imdata=imdata, channels=channels)
            >>> delayed = coco_img.imdelay(channels='B1|Aux:2:4')
            >>> final = delayed.finalize()

        Example:
            >>> # Test delay when loading in asset space
            >>> from kwcoco.coco_image import *  # NOQA
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-msi-multisensor', rng=0)
            >>> coco_img = dset.coco_image(1)
            >>> # Find a stream where the asset is at a different scale
            >>> # (currently this is hacked and depends on the rng being nice)
            >>> stream1 = coco_img.channels.streams()[0]
            >>> stream2 = coco_img.channels.streams()[1]
            >>> asset_delayed = coco_img.imdelay(stream2, space='asset')
            >>> img_delayed = coco_img.imdelay(stream2, space='image')
            >>> vid_delayed = coco_img.imdelay(stream2, space='video')
            >>> #
            >>> aux_imdata = asset_delayed.as_xarray().finalize()
            >>> img_imdata = img_delayed.as_xarray().finalize()
            >>> assert aux_imdata.shape != img_imdata.shape
            >>> # Cannot load multiple asset items at the same time in
            >>> # asset space
            >>> import pytest
            >>> fused_channels = stream1 | stream2
            >>> from delayed_image.delayed_nodes import CoordinateCompatibilityError
            >>> with pytest.raises(CoordinateCompatibilityError):
            >>>     aux_delayed2 = coco_img.imdelay(fused_channels, space='asset')

        Example:
            >>> # Test loading at a specific resolution.
            >>> from kwcoco.coco_image import *  # NOQA
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-msi-multisensor')
            >>> coco_img = dset.coco_image(1)
            >>> coco_img.img['resolution'] = '1 meter'
            >>> img_delayed1 = coco_img.imdelay(space='image')
            >>> vid_delayed1 = coco_img.imdelay(space='video')
            >>> # test with unitless request
            >>> img_delayed2 = coco_img.imdelay(space='image', resolution=3.1)
            >>> vid_delayed2 = coco_img.imdelay(space='video', resolution='3.1 meter')
            >>> np.ceil(img_delayed1.shape[0] / 3.1) == img_delayed2.shape[0]
            >>> np.ceil(vid_delayed1.shape[0] / 3.1) == vid_delayed2.shape[0]
            >>> # test with unitless data
            >>> coco_img.img['resolution'] = 1
            >>> img_delayed2 = coco_img.imdelay(space='image', resolution=3.1)
            >>> vid_delayed2 = coco_img.imdelay(space='video', resolution='3.1 meter')
            >>> np.ceil(img_delayed1.shape[0] / 3.1) == img_delayed2.shape[0]
            >>> np.ceil(vid_delayed1.shape[0] / 3.1) == vid_delayed2.shape[0]
        """
        # from kwimage.transform import Affine
        # from delayed_image.channel_spec import FusedChannelSpec
        if bundle_dpath is None:
            bundle_dpath = self.bundle_dpath

        img = self.img
        requested = channels
        if requested is not None:
            requested = FusedChannelSpec.coerce(requested)
            requested = requested.normalize()

        # Get info about the primary image and check if its channels are
        # requested (if it even has any)
        img_info = _delay_load_imglike(bundle_dpath, img,
                                       nodata_method=nodata_method)
        obj_info_list = [(img_info, img)]
        asset_list = img.get('auxiliary', img.get('assets', [])) or []
        for asset in asset_list:
            info = _delay_load_imglike(bundle_dpath, asset,
                                       nodata_method=nodata_method)
            obj_info_list.append((info, asset))

        chan_list = []
        for info, obj in obj_info_list:
            if info.get('chan_construct', None) is not None:
                include_flag = requested is None
                if not include_flag:
                    if requested.intersection(info['channels']):
                        include_flag = True
                if include_flag:
                    chncls, chnkw = info['chan_construct']
                    chan = chncls(**chnkw)
                    quant = info.get('quantization', None)
                    if quant is not None:
                        chan = chan.dequantize(quant)
                    if space not in {'auxiliary', 'asset'}:
                        warp_img_from_asset = obj.get('warp_aux_to_img', None)
                        # warp_img_from_asset = Affine.coerce(warp_img_from_asset)
                        chan = chan.warp(
                            warp_img_from_asset, dsize=img_info['dsize'],
                            interpolation=interpolation, antialias=antialias,
                            lazy=True,
                        )
                    chan_list.append(chan)

        num_parts = len(chan_list)

        if space == 'video':
            video = self.video or {}
            width = video.get('width', img.get('width', None))
            height = video.get('height', img.get('height', None))
        elif space in {'asset', 'auxiliary'}:
            if num_parts == 0:
                width = img.get('width', None)
                height = img.get('height', None)
            else:
                # TODO: should check these are all in the same space
                width, height = chan_list[0].dsize
        elif space == 'image':
            width = img.get('width', None)
            height = img.get('height', None)
        else:
            raise KeyError(space)

        dsize = (width, height)

        if num_parts == 0:
            if requested is not None:
                # Handle case where the image doesnt have the requested
                # channels.
                # TODO: We should use a NoData node instead that
                # can switch between nans and masked images
                # from delayed_image import DelayedNans
                # from delayed_image import DelayedChannelConcat
                # delayed = DelayedNodata(dsize=dsize, channels=requested, nodata_method=nodata_method)
                delayed = DelayedNans(dsize=dsize, channels=requested)
                delayed = DelayedChannelConcat([delayed])
            else:
                raise ValueError('no data registered in kwcoco image')
        # elif num_parts == 1:
        #     delayed = chan_list[0]
        #     # Reorder channels in the requested order
        #     delayed = delayed.take_channels(requested, lazy=True)

        #     if space in {'image', 'auxiliary', 'asset'}:
        #         pass
        #     elif space == 'video':
        #         warp_vid_from_img = self.img.get('warp_img_to_vid', None)
        #         delayed = delayed.warp(
        #             warp_vid_from_img, dsize=dsize,
        #             interpolation=interpolation,
        #             antialias=antialias,
        #             lazy=True)
        #     else:
        #         raise KeyError('space = {}'.format(space))
        else:
            # from delayed_image import DelayedChannelConcat
            delayed = DelayedChannelConcat(chan_list)

            # Reorder channels in the requested order
            if requested is not None:
                delayed = delayed.take_channels(requested)

            if hasattr(delayed, 'components'):
                if len(delayed.components) == 1:
                    delayed = delayed.components[0]

            if space in {'image', 'auxiliary', 'asset'}:
                pass
            elif space == 'video':
                warp_vid_from_img = self.img.get('warp_img_to_vid', None)
                delayed = delayed.warp(
                    warp_vid_from_img, dsize=dsize,
                    interpolation=interpolation,
                    antialias=antialias,
                    lazy=True)
            else:
                raise KeyError('space = {}'.format(space))

        if resolution is not None:
            # Adjust to the requested resolution
            factor = self._scalefactor_for_resolution(
                space=space, resolution=resolution,
                RESOLUTION_KEY=RESOLUTION_KEY)
            delayed = delayed.scale(
                factor, antialias=antialias, interpolation=interpolation)

        return delayed

    @ub.memoize_method
    def valid_region(self, space='image'):
        """
        If this image has a valid polygon, return it in image, or video space

        Returns:
            None | kwimage.MultiPolygon
        """
        import kwimage
        valid_coco_poly = self.img.get('valid_region', None)
        if valid_coco_poly is None:
            valid_poly = None
        else:
            kw_poly_img = kwimage.MultiPolygon.coerce(valid_coco_poly)
            if kw_poly_img is None:
                valid_poly = None
            else:
                if space == 'image':
                    valid_poly = kw_poly_img
                elif space == 'video':
                    warp_vid_from_img = self.warp_vid_from_img
                    valid_poly = kw_poly_img.warp(warp_vid_from_img)
                else:
                    # To warp it into an auxiliary space we need to know which one
                    raise NotImplementedError(space)
        return valid_poly

    @ub.memoize_property
    def warp_vid_from_img(self):
        """
        Affine transformation that warps image space -> video space.

        Returns:
            kwimage.Affine: The transformation matrix
        """
        import kwimage
        warp_img_to_vid = kwimage.Affine.coerce(self.img.get('warp_img_to_vid', None))
        if warp_img_to_vid.matrix is None:
            # Hack to ensure the matrix property always is an array
            warp_img_to_vid.matrix = np.asarray(warp_img_to_vid)
        return warp_img_to_vid

    @ub.memoize_property
    def warp_img_from_vid(self):
        """
        Affine transformation that warps video space -> image space.

        Returns:
            kwimage.Affine: The transformation matrix
        """
        return self.warp_vid_from_img.inv()

    def _warp_for_resolution(self, space, resolution=None):
        """
        Compute a transform from image-space to the requested space at a
        target resolution.
        """
        import kwimage
        if space == 'image':
            warp_space_from_img = kwimage.Affine(None)
        elif space == 'video':
            warp_space_from_img = self.warp_vid_from_img
        else:
            raise NotImplementedError(space)  # auxiliary/asset space

        if resolution is None:
            warp_final_from_img = warp_space_from_img
        else:
            # Requested the annotation at a resolution, so we need to apply a
            # scale factor
            scale = self._scalefactor_for_resolution(space=space,
                                                     resolution=resolution)
            warp_final_from_space = kwimage.Affine.scale(scale)
            warp_final_from_img = warp_final_from_space @ warp_space_from_img
        return warp_final_from_img

    def _annot_segmentation(self, ann, space='video', resolution=None):
        """"
        Load annotation segmentations in a requested space at a target resolution.

        Example:
            >>> from kwcoco.coco_image import *  # NOQA
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-msi-multisensor')
            >>> coco_img = dset.coco_image(1)
            >>> coco_img.img['resolution'] = '1 meter'
            >>> ann = coco_img.annots().objs[0]
            >>> img_sseg = coco_img._annot_segmentation(ann, space='image')
            >>> vid_sseg = coco_img._annot_segmentation(ann, space='video')
            >>> img_sseg_2m = coco_img._annot_segmentation(ann, space='image', resolution='2 meter')
            >>> vid_sseg_2m = coco_img._annot_segmentation(ann, space='video', resolution='2 meter')
            >>> print(f'img_sseg.area    = {img_sseg.area}')
            >>> print(f'vid_sseg.area    = {vid_sseg.area}')
            >>> print(f'img_sseg_2m.area = {img_sseg_2m.area}')
            >>> print(f'vid_sseg_2m.area = {vid_sseg_2m.area}')
        """
        import kwimage
        warp_final_from_img = self._warp_for_resolution(space=space, resolution=resolution)
        img_sseg = kwimage.MultiPolygon.coerce(ann['segmentation'])
        warped_sseg = img_sseg.warp(warp_final_from_img)
        return warped_sseg

    def _annot_segmentations(self, anns, space='video', resolution=None):
        """"
        Load multiple annotation segmentations in a requested space at a target
        resolution.

        Example:
            >>> from kwcoco.coco_image import *  # NOQA
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-msi-multisensor')
            >>> coco_img = dset.coco_image(1)
            >>> coco_img.img['resolution'] = '1 meter'
            >>> ann = coco_img.annots().objs[0]
            >>> img_sseg = coco_img._annot_segmentations([ann], space='image')
            >>> vid_sseg = coco_img._annot_segmentations([ann], space='video')
            >>> img_sseg_2m = coco_img._annot_segmentations([ann], space='image', resolution='2 meter')
            >>> vid_sseg_2m = coco_img._annot_segmentations([ann], space='video', resolution='2 meter')
            >>> print(f'img_sseg.area    = {img_sseg[0].area}')
            >>> print(f'vid_sseg.area    = {vid_sseg[0].area}')
            >>> print(f'img_sseg_2m.area = {img_sseg_2m[0].area}')
            >>> print(f'vid_sseg_2m.area = {vid_sseg_2m[0].area}')
        """
        import kwimage
        warp_final_from_img = self._warp_for_resolution(space=space, resolution=resolution)
        warped_ssegs = []
        for ann in anns:
            img_sseg = kwimage.MultiPolygon.coerce(ann['segmentation'])
            warped_sseg = img_sseg.warp(warp_final_from_img)
            warped_ssegs.append(warped_sseg)
        return warped_ssegs

    def resolution(self, space='image', channel=None, RESOLUTION_KEY=None):
        """
        Returns the resolution of this CocoImage in the requested space if
        known. Errors if this information is not registered.

        Args:
            space (str): the space to the resolution of.
                Can be either "image", "video", or "asset".

            channel (str | kwcoco.FusedChannelSpec | None):
                a channel that identifies a single asset, only relevant if
                asking for asset space

        Returns:
            Dict:
                has items mag (with the magnitude of the resolution) and
                unit, which is a convenience and only loosely enforced.

        Example:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> self = dset.coco_image(1)
            >>> self.img['resolution'] = 1
            >>> self.resolution()
            >>> self.img['resolution'] = '1 meter'
            >>> assert self.resolution(space='video') == {'mag': (1.0, 1.0), 'unit': 'meter'}
            >>> self.resolution(space='asset', channel='B11')
            >>> self.resolution(space='asset', channel='B1')
        """
        import kwimage
        # Compute the offset transform from the requested space
        # Handle the cases where resolution is specified at the image or at the
        # video level.

        if RESOLUTION_KEY is None:
            RESOLUTION_KEY = DEFAULT_RESOLUTION_KEYS

        def aliased_get(d, keys, default=None):
            if not ub.iterable(keys):
                return d.get(keys, default)
            else:
                found = 0
                for key in keys:
                    if key in d:
                        found = 1
                        val = d[key]
                        break
                if not found:
                    val = default
                return val

        if space == 'video':
            vid_resolution_expr = aliased_get(self.video, RESOLUTION_KEY, None)
            if vid_resolution_expr is None:
                # Do we have an image level resolution?
                img_resolution_expr = aliased_get(self.img, RESOLUTION_KEY, None)
                assert img_resolution_expr is not None
                img_resolution_info = coerce_resolution(img_resolution_expr)
                img_resolution_mat = kwimage.Affine.scale(img_resolution_info['mag'])
                vid_resolution = (self.warp_vid_from_img @ img_resolution_mat.inv()).inv()
                vid_resolution_info = {
                    'mag': vid_resolution.decompose()['scale'],
                    'unit': img_resolution_info['unit']
                }
            else:
                vid_resolution_info = coerce_resolution(vid_resolution_expr)
            space_resolution_info = vid_resolution_info
        elif space == 'image':
            img_resolution_expr = aliased_get(self.img, RESOLUTION_KEY, None)
            if img_resolution_expr is None:
                # Do we have an image level resolution?
                vid_resolution_expr = aliased_get(self.video, RESOLUTION_KEY, None)
                assert vid_resolution_expr is not None
                vid_resolution_info = coerce_resolution(vid_resolution_expr)
                vid_resolution_mat = kwimage.Affine.scale(vid_resolution_info['mag'])
                img_resolution = (self.warp_img_from_vid @ vid_resolution_mat.inv()).inv()
                img_resolution_info = {
                    'mag': img_resolution.decompose()['scale'],
                    'unit': vid_resolution_info['unit']
                }
            else:
                img_resolution_info = coerce_resolution(img_resolution_expr)
            space_resolution_info = img_resolution_info
        elif space in {'asset', 'auxiliary'}:
            if channel is None:
                raise ValueError('must specify a channel to ask for the asset resolution')
            # Use existing code to get the resolution of the image (could be more efficient)
            space_resolution_info = self.resolution('image', RESOLUTION_KEY=RESOLUTION_KEY).copy()
            # Adjust the image resolution based on the asset scale factor
            warp_img_from_aux = kwimage.Affine.coerce(self.find_asset_obj(channel).get('warp_aux_to_img', None))
            img_res_mat = kwimage.Affine.scale(space_resolution_info['mag'])
            aux_res_mat = img_res_mat @ warp_img_from_aux
            space_resolution_info['mag'] = np.array(aux_res_mat.decompose()['scale'])
        else:
            raise KeyError(space)
        return space_resolution_info

    def _scalefactor_for_resolution(self, space, resolution, channel=None, RESOLUTION_KEY=None):
        """
        Given image or video space, compute the scale factor needed to achieve the
        target resolution.

        # Use this to implement
        scale_resolution_from_img
        scale_resolution_from_vid

        Args:
            space (str): the space to the resolution of.
                Can be either "image", "video", or "asset".

            resolution (str | float | int):
                the resolution (ideally with units) you want.

            channel (str | kwcoco.FusedChannelSpec | None):
                a channel that identifies a single asset, only relevant if
                asking for asset space

        Returns:
            Tuple[float, float]:
                the x and y scale factor that can be used to scale the
                underlying "space" to achieve the requested resolution.

        Ignore:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> self = dset.coco_image(1)
            >>> self.img['resolution'] = "3 puppies"
            >>> scale_factor = self._scalefactor_for_resolution(space='asset', channel='B11', resolution="7 puppies")
            >>> print('scale_factor = {}'.format(ub.urepr(scale_factor, precision=4, nl=0)))
            scale_factor = (1.2857, 1.2857)
        """
        if resolution is None:
            return (1., 1.)
        space_resolution_info = self.resolution(space=space, channel=channel, RESOLUTION_KEY=RESOLUTION_KEY)
        request_resolution_info = coerce_resolution(resolution)
        # If units are unspecified, assume they are compatible
        if space_resolution_info['unit'] is not None:
            if request_resolution_info['unit'] is not None:
                assert space_resolution_info['unit'] == request_resolution_info['unit']
        x1, y1 = request_resolution_info['mag']
        x2, y2 = space_resolution_info['mag']
        scale_factor = (x2 / x1, y2 / y1)
        return scale_factor

    def _detections_for_resolution(coco_img, space='video', resolution=None,
                                   aids=None, RESOLUTION_KEY=None):
        """
        This is slightly less than ideal in terms of API, but it will work for
        now.
        """
        import kwimage
        assert space == 'video', 'other cases are not handled'
        # Build transform from image to requested space
        warp_vid_from_img = coco_img.warp_vid_from_img
        scale = coco_img._scalefactor_for_resolution(space='video',
                                                     resolution=resolution,
                                                     RESOLUTION_KEY=RESOLUTION_KEY)
        warp_req_from_vid = kwimage.Affine.scale(scale)
        warp_req_from_img = warp_req_from_vid @ warp_vid_from_img

        # Get annotations in "Image Space"
        if aids is None:
            annots = coco_img.dset.annots(image_id=coco_img.img['id'])
        else:
            # We already know the annots we want (assume the user
            # specified them correctly and they belong to this image).
            annots = coco_img.dset.annots(aids)
        imgspace_dets = annots.detections

        # Warp them into the requested space
        reqspace_dets = imgspace_dets.warp(warp_req_from_img)
        reqspace_dets.data['aids'] = np.array(list(annots))
        return reqspace_dets

    # Deprecated aliases
    add_auxiliary_item = deprecated_function_alias(
        'kwcoco', 'add_auxiliary_item', deprecate='now', new_func=add_asset)

    delay = deprecated_function_alias(
        'kwcoco', 'delay', new_func=imdelay, deprecate='now')

    def show(self, **kwargs):
        """
        Show the image with matplotlib if possible

        SeeAlso:
            :func:`kwcoco.CocoDataset.show_image`

        Example:
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> self = dset.coco_image(1)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autoplt()
            >>> self.show()

        """
        if self.dset is None:
            raise Exception('Currently requires a connected dataset. '
                            'This may be relaxed in the future')
        return self.dset.show_image(self['id'], **kwargs)

    def draw(self, **kwargs):
        """
        Draw the image on an ndarray using opencv

        SeeAlso:
            :func:`kwcoco.CocoDataset.draw_image`

        Example:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> self = dset.coco_image(1)
            >>> canvas = self.draw()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)

        """
        if self.dset is None:
            raise Exception('Currently requires a connected dataset. '
                            'This may be relaxed in the future')
        return self.dset.draw_image(self['id'], **kwargs)


class CocoAsset(_CocoObject):
    """
    A Coco Asset / Auxiliary Item

    Represents one 2D image file relative to a parent img.

    Could be a single asset, or an image with sub-assets, but sub-assets are
    ignored here.

    Initially we called these "auxiliary" items, but I think we should
    change their name to "assets", which better maps with STAC terminology.

    Example:
        >>> from kwcoco.coco_image import *  # NOQA
        >>> self = CocoAsset({'warp_aux_to_img': 'foo'})
        >>> assert 'warp_aux_to_img' in self
        >>> assert 'warp_img_from_asset' in self
        >>> assert 'warp_wld_from_asset' not in self
        >>> assert 'warp_to_wld' not in self
        >>> self['warp_aux_to_img'] = 'bar'
        >>> assert self._proxy == {'warp_aux_to_img': 'bar'}
    """

    # To maintain backwards compatibility we register aliases of properties
    # The main key should be the primary property.
    __alias_to_primary__ = {
        'warp_img_from_asset': 'warp_aux_to_img',
        'warp_wld_from_asset': 'warp_to_wld',
    }

    def __init__(self, asset, bundle_dpath=None):
        super().__init__(asset, bundle_dpath=bundle_dpath)

    def __nice__(self):
        return repr(self.__json__())

    def image_filepath(self):
        if self.bundle_dpath is None:
            raise Exception('Bundle dpath must be populated to use this method')
            # return self['file_name']
        else:
            return ub.Path(self.bundle_dpath) / self['file_name']


class CocoVideo(_CocoObject):
    """
    Object representing a single video.

    Example:
        >>> from kwcoco.coco_image import *  # NOQA
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('vidshapes1')
        >>> obj = dset.videos().objs[0]
        >>> self = CocoVideo(obj, dset)
        >>> print(f'self={self}')
    """
    __alias_to_primary__ = {}

    def __nice__(self):
        return repr(self.__json__())


class CocoAnnotation(_CocoObject):
    """
    Object representing a single annotation.

    Example:
        >>> from kwcoco.coco_image import *  # NOQA
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('vidshapes1')
        >>> obj = dset.annots().objs[0]
        >>> self = CocoAnnotation(obj, dset)
        >>> print(f'self={self}')
    """
    __alias_to_primary__ = {}

    def __nice__(self):
        return repr(self.__json__())


class CocoCategory(_CocoObject):
    """
    Object representing a single category.

    Example:
        >>> from kwcoco.coco_image import *  # NOQA
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('vidshapes1')
        >>> obj = dset.categories().objs[0]
        >>> self = CocoCategory(obj, dset)
        >>> print(f'self={self}')
    """
    __alias_to_primary__ = {}

    def __nice__(self):
        return repr(self.__json__())


class CocoTrack(_CocoObject):
    """
    Object representing a single track.

    Example:
        >>> from kwcoco.coco_image import *  # NOQA
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('vidshapes1')
        >>> obj = dset.tracks().objs[0]
        >>> self = CocoTrack(obj, dset)
        >>> print(f'self={self}')
    """
    __alias_to_primary__ = {}

    def __nice__(self):
        return repr(self.__json__())

    def annots(self):
        assert self.dset is not None
        return self.dset.annots(track_id=self['id'])


def _delay_load_imglike(bundle_dpath, obj, nodata_method=None):
    # from os.path import join
    # from delayed_image.channel_spec import FusedChannelSpec
    # from delayed_image import DelayedLoad, DelayedIdentity
    info = {}
    fname = obj.get('file_name', None)
    imdata = obj.get('imdata', None)
    channels_ = obj.get('channels', None)
    if channels_ is not None:
        channels_ = FusedChannelSpec.coerce(channels_)
        channels_ = channels_.normalize()
    info['channels'] = channels_
    width = obj.get('width', None)
    height = obj.get('height', None)
    num_overviews = obj.get('num_overviews', None)
    if height is not None and width is not None:
        info['dsize'] = dsize = (width, height)
    else:
        info['dsize'] = dsize = (None, None)

    quantization = obj.get('quantization', None)
    if imdata is not None:
        _kwargs = dict(data=imdata, channels=channels_, dsize=dsize)
        info['chan_construct'] = (DelayedIdentity, _kwargs)
    elif fname is not None:
        fpath = join(bundle_dpath, fname)
        _kwargs = dict(fpath=fpath, channels=channels_, dsize=dsize,
                       nodata_method=nodata_method,
                       num_overviews=num_overviews)
        info['chan_construct'] = (DelayedLoad, _kwargs)
    info['quantization'] = quantization
    return info


def parse_quantity(expr):
    import re
    expr_pat = re.compile(
        r'^(?P<magnitude>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
        '(?P<spaces> *)'
        '(?P<unit>.*)$')
    match = expr_pat.match(expr.strip())
    if match is None:
        raise ValueError(f'Unable to parse {expr!r}')
    return match.groupdict()


def coerce_resolution(expr):
    if isinstance(expr, str):
        result = parse_quantity(expr)
        unit = result['unit']
        x = y = float(result['magnitude'])
    else:
        x = y = float(expr)
        unit = None

    parsed = {
        'mag': (x, y),
        'unit': unit,
    }
    return parsed
