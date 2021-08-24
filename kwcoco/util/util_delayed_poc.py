"""
This module is ported from ndsampler, and will likely eventually move to
kwimage and be refactored using pymbolic

The classes in this file represent a tree of delayed operations.

Proof of concept for delayed chainable transforms in Python.

There are several optimizations that could be applied.

This is similar to GDAL's virtual raster table, but it works in memory and I
think it is easier to chain operations.

SeeAlso:
    ../../dev/symbolic_delayed.py


WARNING:
    As the name implies this is a proof of concept, and the actual
    implementation was hacked together too quickly. Serious refactoring will be
    necessary.


Concepts:

    Each class should be a layer that adds a new transformation on top of
    underlying nested layers. Adding new layers should be quick, and there
    should always be the option to "finalize" a stack of layers, chaining the
    transforms / operations and then applying one final efficient transform at
    the end.


Conventions:

    * dsize = (always in width / height), no channels are present

    * shape for images is always (height, width, channels)

    * channels are always the last dimension of each image, if no channel
      dim is specified, finalize will add it.

    * Videos must be the last process in the stack, and add a leading
        time dimension to the shape. dsize is still width, height, but shape
        is now: (time, height, width, chan)


Example:
    >>> # Example demonstrating the modivating use case
    >>> # We have multiple aligned frames for a video, but each of
    >>> # those frames is in a different resolution. Furthermore,
    >>> # each of the frames consists of channels in different resolutions.
    >>> # Create raw channels in some "native" resolution for frame 1
    >>> f1_chan1 = DelayedIdentity.demo('astro', chan=0, dsize=(300, 300))
    >>> f1_chan2 = DelayedIdentity.demo('astro', chan=1, dsize=(200, 200))
    >>> f1_chan3 = DelayedIdentity.demo('astro', chan=2, dsize=(10, 10))
    >>> # Create raw channels in some "native" resolution for frame 2
    >>> f2_chan1 = DelayedIdentity.demo('carl', dsize=(64, 64), chan=0)
    >>> f2_chan2 = DelayedIdentity.demo('carl', dsize=(260, 260), chan=1)
    >>> f2_chan3 = DelayedIdentity.demo('carl', dsize=(10, 10), chan=2)
    >>> #
    >>> # Delayed warp each channel into its "image" space
    >>> # Note: the images never actually enter this space we transform through it
    >>> f1_dsize = np.array((3, 3))
    >>> f2_dsize = np.array((2, 2))
    >>> f1_img = DelayedChannelConcat([
    >>>     f1_chan1.delayed_warp(Affine.scale(f1_dsize / f1_chan1.dsize), dsize=f1_dsize),
    >>>     f1_chan2.delayed_warp(Affine.scale(f1_dsize / f1_chan2.dsize), dsize=f1_dsize),
    >>>     f1_chan3.delayed_warp(Affine.scale(f1_dsize / f1_chan3.dsize), dsize=f1_dsize),
    >>> ])
    >>> f2_img = DelayedChannelConcat([
    >>>     f2_chan1.delayed_warp(Affine.scale(f2_dsize / f2_chan1.dsize), dsize=f2_dsize),
    >>>     f2_chan2.delayed_warp(Affine.scale(f2_dsize / f2_chan2.dsize), dsize=f2_dsize),
    >>>     f2_chan3.delayed_warp(Affine.scale(f2_dsize / f2_chan3.dsize), dsize=f2_dsize),
    >>> ])
    >>> # Combine frames into a video
    >>> vid_dsize = np.array((280, 280))
    >>> vid = DelayedFrameConcat([
    >>>     f1_img.delayed_warp(Affine.scale(vid_dsize / f1_img.dsize), dsize=vid_dsize),
    >>>     f2_img.delayed_warp(Affine.scale(vid_dsize / f2_img.dsize), dsize=vid_dsize),
    >>> ])
    >>> vid.nesting
    >>> print('vid.nesting = {}'.format(ub.repr2(vid.__json__(), nl=-2)))
    >>> final = vid.finalize(interpolation='nearest')
    >>> # xdoctest: +REQUIRES(--show)
    >>> import kwplot
    >>> kwplot.autompl()
    >>> kwplot.imshow(final[0], pnum=(1, 2, 1), fnum=1)
    >>> kwplot.imshow(final[1], pnum=(1, 2, 2), fnum=1)

Example:
    >>> import kwcoco
    >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
    >>> delayed = dset.delayed_load(1)
    >>> from kwcoco.util.util_delayed_poc import *  # NOQA
    >>> astro = DelayedLoad.demo('astro')
    >>> print('MSI = ' + ub.repr2(delayed.__json__(), nl=-3, sort=0))
    >>> print('ASTRO = ' + ub.repr2(astro.__json__(), nl=2, sort=0))

    >>> subchan = delayed.take_channels('B1|B8')
    >>> subcrop = subchan.delayed_crop((slice(10, 80), slice(30, 50)))
    >>> #
    >>> subcrop.nesting()
    >>> subchan.nesting()
    >>> subchan.finalize()
    >>> subcrop.finalize()
    >>> #
    >>> msi_crop = delayed.delayed_crop((slice(10, 80), slice(30, 50)))
    >>> subdata = msi_crop.delayed_warp(kwimage.Affine.scale(3), dsize='auto').take_channels('B11|B1')
    >>> subdata.finalize()


Example:
    >>> # test case where an auxiliary image does not map entirely on the image.
    >>> from kwcoco.util.util_delayed_poc import *  # NOQA
    >>> import kwimage
    >>> from os.path import join
    >>> dpath = ub.ensure_app_cache_dir('kwcoco/tests/delayed_poc')
    >>> chan1_fpath = join(dpath, 'chan1.tiff')
    >>> chan2_fpath = join(dpath, 'chan2.tiff')
    >>> chan3_fpath = join(dpath, 'chan2.tiff')
    >>> chan1_raw = np.random.rand(128, 128, 1)
    >>> chan2_raw = np.random.rand(64, 64, 1)
    >>> chan3_raw = np.random.rand(256, 256, 1)
    >>> kwimage.imwrite(chan1_fpath, chan1_raw)
    >>> kwimage.imwrite(chan2_fpath, chan2_raw)
    >>> kwimage.imwrite(chan3_fpath, chan3_raw)
    >>> #
    >>> c1 = channel_spec.FusedChannelSpec.coerce('c1')
    >>> c2 = channel_spec.FusedChannelSpec.coerce('c2')
    >>> c3 = channel_spec.FusedChannelSpec.coerce('c2')
    >>> aux1 = DelayedLoad(chan1_fpath, dsize=chan1_raw.shape[0:2][::-1], channels=c1, num_bands=1)
    >>> aux2 = DelayedLoad(chan2_fpath, dsize=chan2_raw.shape[0:2][::-1], channels=c2, num_bands=1)
    >>> aux3 = DelayedLoad(chan3_fpath, dsize=chan3_raw.shape[0:2][::-1], channels=c3, num_bands=1)
    >>> #
    >>> img_dsize = (128, 128)
    >>> transform1 = kwimage.Affine.coerce(scale=0.5)
    >>> transform2 = kwimage.Affine.coerce(theta=0.5, shear=0.01, offset=(-20, -40))
    >>> transform3 = kwimage.Affine.coerce(offset=(64, 0)) @ kwimage.Affine.random(rng=10)
    >>> part1 = aux1.delayed_warp(np.eye(3), dsize=img_dsize)
    >>> part2 = aux2.delayed_warp(transform2, dsize=img_dsize)
    >>> part3 = aux3.delayed_warp(transform3, dsize=img_dsize)
    >>> delayed = DelayedChannelConcat([part1, part2, part3])
    >>> #
    >>> delayed_crop = delayed.crop((slice(0, 10), slice(0, 10)))
    >>> delayed_final = delayed_crop.finalize()
    >>> # xdoctest: +REQUIRES(--show)
    >>> import kwplot
    >>> kwplot.autompl()
    >>> final = delayed.finalize()
    >>> kwplot.imshow(final, fnum=1, pnum=(1, 2, 1))
    >>> kwplot.imshow(delayed_final, fnum=1, pnum=(1, 2, 2))


    comp = delayed_crop.components[2]

    comp.sub_data.finalize()

    data = np.array([[0]]).astype(np.float32)
    kwimage.warp_affine(data, np.eye(3), dsize=(32, 32))
    kwimage.warp_affine(data, np.eye(3))

    kwimage.warp_affine(data[0:0], np.eye(3))

    transform = kwimage.Affine.coerce(scale=0.1)
    data = np.array([[0]]).astype(np.float32)

    data = np.array([[]]).astype(np.float32)
    kwimage.warp_affine(data, transform, dsize=(0, 2), antialias=True)

    data = np.array([[]]).astype(np.float32)
    kwimage.warp_affine(data, transform, dsize=(10, 10))

    data = np.array([[0]]).astype(np.float32)
    kwimage.warp_affine(data, transform, dsize=(0, 2), antialias=True)

    data = np.array([[0]]).astype(np.float32)
    kwimage.warp_affine(data, transform, dsize=(10, 10))

    cv2.warpAffine(
        kwimage.grab_test_image(dsize=(1, 1)),
        kwimage.Affine.coerce(scale=0.1).matrix[0:2],
        dsize=(0, 1),
    )
"""
import ubelt as ub
import numpy as np
import kwimage
import kwarray
from kwimage.transform import Affine
from kwcoco import channel_spec

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


# DEBUG_PRINT = ub.identity
# DEBUG_PRINT = print


class DelayedVisionOperation(ub.NiceRepr):
    """
    Base class for nodes in a tree of delayed computer-vision operations
    """
    def __nice__(self):
        channels = self.channels
        return '{}, {}'.format(self.shape, channels)

    def finalize(self):
        raise NotImplementedError

    def children(self):
        """
        Abstract method, which should generate all of the direct children of a
        node in the operation tree.
        """
        raise NotImplementedError

    def _optimize_paths(self, **kwargs):
        """
        Iterate through the leaf nodes, which are virtually transformed into
        the root space.

        This returns some sort of hueristically optimized leaf repr wrt warps.
        """
        # DEBUG_PRINT('DelayedVisionOperation._optimize_paths {}'.format(type(self)))
        for child in self.children():
            yield from child._optimize_paths(**kwargs)

    def __json__(self):
        from kwcoco.util.util_json import ensure_json_serializable
        json_dict = ensure_json_serializable(self.nesting())
        return json_dict

    def nesting(self):
        def _child_nesting(child):
            if hasattr(child, 'nesting'):
                return child.nesting()
            elif isinstance(child, np.ndarray):
                return {
                    'type': 'ndarray',
                    'shape': self.sub_data.shape,
                }
        children = [_child_nesting(child) for child in self.children()]
        item = {
            'type': self.__class__.__name__,
            'meta': self.meta,
        }
        if children:
            item['children'] = children
        return item

    def warp(self, *args, **kwargs):
        """ alias for delayed_warp, might change to this API in the future """
        return self.delayed_warp(*args, **kwargs)

    def crop(self, *args, **kwargs):
        """ alias for delayed_crop, might change to this API in the future """
        return self.delayed_crop(*args, **kwargs)


class DelayedVideoOperation(DelayedVisionOperation):
    pass


class DelayedImageOperation(DelayedVisionOperation):
    """
    Operations that pertain only to images
    """

    def delayed_crop(self, region_slices):
        """
        Create a new delayed image that performs a crop in the transformed
        "self" space.

        Args:
            region_slices (Tuple[slice, slice]): y-slice and x-slice.

        Note:
            Returns a heuristically "simplified" tree. In the current
            implementation there are only 3 operations, cat, warp, and crop.
            All cats go at the top, all crops go at the bottom, all warps are
            in the middle.

        Returns:
            DelayedWarp: lazy executed delayed transform

        Example:
            >>> dsize = (100, 100)
            >>> tf2 = Affine.affine(scale=3).matrix
            >>> self = DelayedWarp(np.random.rand(33, 33), tf2, dsize)
            >>> region_slices = (slice(5, 10), slice(1, 12))
            >>> delayed_crop = self.delayed_crop(region_slices)
            >>> print(ub.repr2(delayed_crop.nesting(), nl=-1, sort=0))
            >>> delayed_crop.finalize()

        Example:
            >>> chan1 = DelayedLoad.demo('astro')
            >>> chan2 = DelayedLoad.demo('carl')
            >>> warped1a = chan1.delayed_warp(Affine.scale(1.2).matrix)
            >>> warped2a = chan2.delayed_warp(Affine.scale(1.5))
            >>> warped1b = warped1a.delayed_warp(Affine.scale(1.2).matrix)
            >>> warped2b = warped2a.delayed_warp(Affine.scale(1.5))
            >>> #
            >>> region_slices = (slice(97, 677), slice(5, 691))
            >>> self = warped2b
            >>> #
            >>> crop1 = warped1b.delayed_crop(region_slices)
            >>> crop2 = warped2b.delayed_crop(region_slices)
            >>> print(ub.repr2(warped1b.nesting(), nl=-1, sort=0))
            >>> print(ub.repr2(warped2b.nesting(), nl=-1, sort=0))
            >>> # Notice how the crop merges the two nesting layers
            >>> # (via the hueristic optimize step)
            >>> print(ub.repr2(crop1.nesting(), nl=-1, sort=0))
            >>> print(ub.repr2(crop2.nesting(), nl=-1, sort=0))
            >>> frame1 = crop1.finalize(dsize=(500, 500))
            >>> frame2 = crop2.finalize(dsize=(500, 500))
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(frame1, pnum=(1, 2, 1), fnum=1)
            >>> kwplot.imshow(frame2, pnum=(1, 2, 2), fnum=1)

        """
        # DEBUG_PRINT('DelayedImageOperation.delayed_crop: {}'.format(type(self)))
        if region_slices is None:
            return self
        components = []

        for delayed_leaf in self._optimize_paths():
            # DEBUG_PRINT('delayed_leaf = {!r}'.format(delayed_leaf))
            # DEBUG_PRINT('delayed_leaf.sub_data = {!r}'.format(delayed_leaf.sub_data))
            # Compute, sub_crop_slices, and new tf_newleaf_to_newroot
            assert isinstance(delayed_leaf, DelayedWarp)  # HACK
            tf_leaf_to_root = delayed_leaf.meta['transform']

            root_region_box = kwimage.Boxes.from_slice(
                region_slices, shape=delayed_leaf.shape)
            root_region_bounds = root_region_box.to_polygons()[0]

            w = root_region_box.width.ravel()[0]
            h = root_region_box.height.ravel()[0]
            root_dsize = (w, h)

            leaf_crop_slices, tf_newleaf_to_newroot = _compute_leaf_subcrop(
                root_region_bounds, tf_leaf_to_root)

            delayed_leaf.sub_data

            if isinstance(delayed_leaf.sub_data, (DelayedLoad, DelayedNans)):
                # if hasattr(delayed_leaf.sub_data, 'delayed_crop'):
                # Hack
                crop = delayed_leaf.sub_data.delayed_crop(leaf_crop_slices)
            else:
                crop = DelayedCrop(delayed_leaf.sub_data, leaf_crop_slices)

            warp = DelayedWarp(crop, tf_newleaf_to_newroot, dsize=root_dsize)
            components.append(warp)

        if len(components) == 0:
            print('self = {!r}'.format(self))
            raise ValueError('Did not find any componets')
        if len(components) == 1:
            return components[0]
        else:
            return DelayedChannelConcat(components)

    def delayed_warp(self, transform, dsize=None):
        """
        Delayedly transform the underlying data.

        Note:
            this deviates from kwimage warp functions because instead of
            "output_dims" (specified in c-style shape) we specify dsize (w, h).

        Returns:
            DelayedWarp : new delayed transform a chained transform
        """
        warped = DelayedWarp(self, transform=transform, dsize=dsize)
        return warped

    def take_channels(self, channels):
        raise NotImplementedError


class DelayedIdentity(DelayedImageOperation):
    """
    Noop leaf that does nothing. Mostly used in tests atm.

    Typically used to just hold raw data.

    DelayedIdentity.demo('astro', chan=0, dsize=(32, 32))
    """
    __hack_dont_optimize__ = True

    def __init__(self, sub_data):
        self.sub_data = sub_data
        self.meta = {}
        self.cache = {}
        h, w = self.sub_data.shape[0:2]
        self.dsize = (w, h)
        if len(self.sub_data.shape) == 2:
            num_bands = 1
        elif len(self.sub_data.shape) == 3:
            num_bands = self.sub_data.shape[2]
        else:
            raise ValueError(
                'Data may only have 2 space dimensions and 1 channel '
                'dimension')
        self.num_bands = num_bands
        self.shape = (h, w, self.num_bands)
        self.meta['dsize'] = self.dsize
        self.meta['shape'] = self.shape

    @classmethod
    def demo(cls, key='astro', chan=None, dsize=None):
        if key == 'checkerboard':
            # https://stackoverflow.com/questions/2169478/how-to-make-a-checkerboard-in-numpy
            num_squares = 8
            num_pairs = num_squares // 2
            img_size = 512
            b = img_size // num_squares
            img = np.kron([[1, 0] * num_pairs, [0, 1] * num_pairs] * num_pairs, np.ones((b, b)))
            sub_data = img
        else:
            sub_data = kwimage.grab_test_image(key, dsize=dsize)
        if chan is not None:
            sub_data = sub_data[..., chan]
        self = cls(sub_data)
        return self

    def children(self):
        yield from []

    # def delayed_crop(self, region_slices):
    #     return DelayedCrop(self, region_slices)

    def _optimize_paths(self, **kwargs):
        # DEBUG_PRINT('DelayedIdentity._optimize_paths')
        # Hack
        yield DelayedWarp(self, Affine(None), dsize=self.dsize)

    def finalize(self):
        final = self.sub_data
        final = kwarray.atleast_nd(final, 3, front=False)
        return final


class DelayedNans(DelayedImageOperation):
    """
    Constructs nan channels as needed

    Example:
        self = DelayedNans((10, 10), channel_spec.FusedChannelSpec.coerce('rgb'))
        region_slices = (slice(5, 10), slice(1, 12))
        delayed = self.delayed_crop(region_slices)

    """
    def __init__(self, dsize=None, channels=None):
        self.meta = {}
        self.meta['dsize'] = dsize
        self.meta['channels'] = channels

        if channels is not None:
            # hack
            self.meta['channels'] = self.meta['channels'].normalize()
            self.meta['num_bands'] = len(self.meta['channels'].unique())

    @property
    def shape(self):
        dsize = self.dsize
        if dsize is None:
            w, h = None, None
        else:
            w, h = dsize
        c = self.num_bands
        return (h, w, c)

    @property
    def num_bands(self):
        return self.meta.get('num_bands', None)

    @property
    def dsize(self):
        dsize = self.meta.get('dsize', None)
        return dsize

    @property
    def channels(self):
        return self.meta.get('channels', None)

    def children(self):
        yield from []

    def _optimize_paths(self, **kwargs):
        # DEBUG_PRINT('DelayedLoad._optimize_paths')
        # hack
        yield DelayedWarp(self, Affine(None), dsize=self.dsize)

    def finalize(self, **kwargs):
        if 'dsize' in kwargs:
            shape = tuple(kwargs['dsize'])[::-1] + (self.num_bands,)
        else:
            shape = self.shape
        final = np.full(shape, fill_value=np.nan)

        as_xarray = kwargs.get('as_xarray', False)
        if as_xarray:
            import xarray as xr
            channels = self.channels
            coords = {}
            if channels is not None:
                coords['c'] = channels.code_list()
            final = xr.DataArray(final, dims=('y', 'x', 'c'), coords=coords)
        return final

    def delayed_crop(self, region_slices):
        # DEBUG_PRINT('DelayedNans.delayed_crop')
        channels = self.channels
        dsize = self.dsize
        data_dims = dsize[::-1]
        data_slice, extra_pad = kwarray.embed_slice(region_slices, data_dims)
        box = kwimage.Boxes.from_slice(data_slice)
        new_width = box.width.ravel()[0]
        new_height = box.height.ravel()[0]

        new_dsize = (new_width, new_height)
        new = self.__class__(new_dsize, channels=channels)
        return new

    def delayed_warp(self, transform, dsize=None):
        new = self.__class__(dsize, channels=self.channels)
        return new


class DelayedLoad(DelayedImageOperation):
    """
    A load operation for a specific sub-region and sub-bands in a specified
    image.

    Note:
        This class contains support for fusing certain lazy operations into
        this layer, namely cropping, scaling, and channel selection.

        For now these are named ``immediates``

    Example:
        >>> fpath = kwimage.grab_test_image_fpath()
        >>> self = DelayedLoad(fpath)
        >>> print('self = {!r}'.format(self))
        >>> self.load_shape()
        >>> print('self = {!r}'.format(self))
        >>> self.finalize()

        >>> f1_img = DelayedLoad.demo('astro', dsize=(300, 300))
        >>> f2_img = DelayedLoad.demo('carl', dsize=(256, 320))
        >>> print('f1_img = {!r}'.format(f1_img))
        >>> print('f2_img = {!r}'.format(f2_img))
        >>> print(f2_img.finalize().shape)
        >>> print(f1_img.finalize().shape)

        >>> fpath = kwimage.grab_test_image_fpath()
        >>> channels = channel_spec.FusedChannelSpec.coerce('rgb')
        >>> self = DelayedLoad(fpath, channels=channels)
    """
    __hack_dont_optimize__ = True

    def __init__(self, fpath, channels=None,
                 # Extra params to allow certain operations as close to the
                 # disk as possible.
                 dsize=None,
                 num_bands=None,
                 immediate_crop=None,
                 immediate_chan_idxs=None,
                 immediate_dsize=None):
        # self.data = {}
        self.meta = {}
        self.cache = {}

        if immediate_dsize is not None:
            dsize = immediate_dsize

        self.meta['fpath'] = fpath
        self.meta['dsize'] = dsize
        self.meta['channels'] = channels

        self._immediates = {
            'crop': immediate_crop,
            'dsize': immediate_dsize,
            'chan_idxs': immediate_chan_idxs,
        }

        if num_bands is not None:
            self.meta['num_bands'] = num_bands

        if channels is not None:
            # hack
            self.meta['channels'] = self.meta['channels'].normalize()
            if num_bands is None:
                self.meta['num_bands'] = len(self.meta['channels'].unique())

    @classmethod
    def demo(DelayedLoad, key='astro', dsize=None):
        fpath = kwimage.grab_test_image_fpath(key)
        self = DelayedLoad(fpath, immediate_dsize=dsize)
        return self

    @classmethod
    def coerce(cls, data):
        raise NotImplementedError

    def children(self):
        yield from []

    def nesting(self):
        item = {
            'type': self.__class__.__name__,
            'meta': self.meta,
            '_immediates': self._immediates,
        }
        return item

    def _optimize_paths(self, **kwargs):
        # DEBUG_PRINT('DelayedLoad._optimize_paths')
        # hack
        yield DelayedWarp(self, Affine(None), dsize=self.dsize)
        # raise AssertionError('hack so this is not called')

    def load_shape(self, use_channel_heuristic=False):
        disk_shape = kwimage.load_image_shape(self.fpath)
        if self.meta.get('num_bands', None) is None:
            num_bands = disk_shape[2] if len(disk_shape) == 3 else 1
            self.meta['num_bands'] = num_bands
        if self.meta.get('dsize', None) is None:
            h, w = disk_shape[0:2]
            self.meta['dsize'] = (w, h)
        if self.meta.get('channels', None) is None:
            if self.meta['num_bands'] == 3:
                self.meta['channels'] = channel_spec.FusedChannelSpec.coerce('r|g|b')
        self.meta['_raw_shape'] = disk_shape
        return self

    def _ensure_dsize(self):
        dsize = self.dsize
        if dsize is None:
            self.load_shape()
            dsize = self.dsize
        return dsize

    @property
    def shape(self):
        dsize = self.dsize
        if dsize is None:
            w, h = None, None
        else:
            w, h = dsize
        c = self.num_bands
        return (h, w, c)

    @property
    def num_bands(self):
        return self.meta.get('num_bands', None)

    @property
    def dsize(self):
        dsize = self.meta.get('dsize', None)
        return dsize

    @property
    def channels(self):
        return self.meta.get('channels', None)

    @property
    def fpath(self):
        return self.meta.get('fpath', None)

    def finalize(self, **kwargs):
        final = self.cache.get('final', None)
        if final is None:
            if have_gdal():
                # TODO: warn if we dont have a COG.
                pre_final = LazyGDalFrameFile(self.fpath)
            else:
                import warnings
                warnings.warn('DelayedLoad may not be efficient without gdal')
                # TODO: delay even further with gdal
                pre_final = kwimage.imread(self.fpath)
                pre_final = kwarray.atleast_nd(pre_final, 3)

            chan_idxs = self._immediates.get('chan_idxs', None)
            space_slice = self._immediates.get('crop', None)
            if chan_idxs is None:
                chan_slice = tuple([slice(None)])
            else:
                chan_slice = tuple([chan_idxs])
            if space_slice is None:
                space_slice = tuple([slice(None), slice(None)])
            sl = space_slice + chan_slice

            final = pre_final[sl]

            dsize = self._immediates.get('dsize', None)
            if dsize is not None:
                final = kwimage.imresize(final, dsize=dsize, antialias=True)
            self.cache['final'] = final

        as_xarray = kwargs.get('as_xarray', False)
        if as_xarray:
            # FIXME: might not work with
            import xarray as xr
            channels = self.channels
            coords = {}
            if channels is not None:
                coords['c'] = channels.code_list()
            final = xr.DataArray(final, dims=('y', 'x', 'c'), coords=coords)
        return final

    @profile
    def delayed_crop(self, region_slices):
        """
        Args:
            region_slices (Tuple[slice, slice]): y-slice and x-slice.

        Returns:
            DelayedLoad : a new delayed load object with a fused crop operation

        Example:
            >>> # Test chained crop operations
            >>> from kwcoco.util.util_delayed_poc import *  # NOQA
            >>> self = orig = DelayedLoad.demo('astro').load_shape()
            >>> region_slices = slices1 = (slice(0, 90), slice(30, 60))
            >>> self = crop1 = orig.delayed_crop(slices1)
            >>> region_slices = slices2 = (slice(10, 21), slice(10, 22))
            >>> self = crop2 = crop1.delayed_crop(slices2)
            >>> region_slices = slices3 = (slice(3, 20), slice(5, 20))
            >>> crop3 = crop2.delayed_crop(slices3)
            >>> # Spot check internals
            >>> print('orig = {}'.format(ub.repr2(orig.__json__(), nl=2)))
            >>> print('crop1 = {}'.format(ub.repr2(crop1.__json__(), nl=2)))
            >>> print('crop2 = {}'.format(ub.repr2(crop2.__json__(), nl=2)))
            >>> print('crop3 = {}'.format(ub.repr2(crop3.__json__(), nl=2)))
            >>> # Test internals
            >>> assert crop3._immediates['crop'][0].start == 13
            >>> assert crop3._immediates['crop'][0].stop == 21
            >>> # Test shapes work out correctly
            >>> assert crop3.finalize().shape == (8, 7, 3)
            >>> assert crop2.finalize().shape == (11, 12, 3)
            >>> assert crop1.take_channels([1, 2]).finalize().shape == (90, 30, 2)
            >>> assert orig.finalize().shape == (512, 512, 3)

        Note:

            .. code::

                This chart gives an intuition on how new absolute slice coords
                are computed from existing absolute coords ane relative coords.

                      5 7    <- new
                      3 5    <- rel
                   --------
                   01234567  <- relative coordinates
                   --------
                   2      9  <- curr
                 ----------
                 0123456789  <- absolute coordinates
                 ----------
        """
        # DEBUG_PRINT('DelayedLoad.delayed_crop')
        # Check if there is already a delayed crop operation
        curr_slices = self._immediates['crop']
        if curr_slices is None:
            data_dims = self._ensure_dsize()[::-1]
            curr_slices = (slice(0, data_dims[0]), slice(0, data_dims[1]))

        rel_ysl, rel_xsl = region_slices
        curr_ysl, curr_xsl = curr_slices

        # Apply the new relative slice to the current absolute slice
        new_xstart = min(curr_xsl.start + rel_xsl.start, curr_xsl.stop)
        new_xstop = min(curr_xsl.start + rel_xsl.stop, curr_xsl.stop)

        new_ystart = min(curr_ysl.start + rel_ysl.start, curr_ysl.stop)
        new_ystop = min(curr_ysl.start + rel_ysl.stop, curr_ysl.stop)

        new_crop = (slice(new_ystart, new_ystop),
                    slice(new_xstart, new_xstop))

        new_dsize = (new_xstop - new_xstart,
                     new_ystop - new_ystart)

        assert self._immediates['dsize'] is None, 'does not handle'

        new = self.__class__(
            fpath=self.meta['fpath'],
            num_bands=self.meta['num_bands'],
            channels=self.meta['channels'],
            dsize=new_dsize,
            immediate_crop=new_crop,
            immediate_chan_idxs=self._immediates['chan_idxs'],
        )
        return new

    @profile
    def take_channels(self, channels):
        """
        This method returns a subset of the vision data with only the
        specified bands / channels.

        Args:
            channels (List[int] | slice | channel_spec.FusedChannelSpec):
                List of integers indexes, a slice, or a channel spec, which is
                typically a pipe (`|`) delimited list of channel codes. See
                kwcoco.ChannelSpec for more detials.

        Returns:
            DelayedLoad:
                a new delayed load with a fused take channel operation

        Note:
            The channel subset must exist here or it will raise an error.
            A better implementation (via pymbolic) might be able to do better

        Example:
            >>> from kwcoco.util.util_delayed_poc import *  # NOQA
            >>> import kwcoco
            >>> self = DelayedLoad.demo('astro').load_shape()
            >>> channels = [2, 0]
            >>> new = self.take_channels(channels)
            >>> new3 = new.take_channels([1, 0])

            >>> final1 = self.finalize()
            >>> final2 = new.finalize()
            >>> final3 = new3.finalize()
            >>> assert np.all(final1[..., 2] == final2[..., 0])
            >>> assert np.all(final1[..., 0] == final2[..., 1])
            >>> assert final2.shape[2] == 2

            >>> assert np.all(final1[..., 2] == final3[..., 1])
            >>> assert np.all(final1[..., 0] == final3[..., 0])
            >>> assert final3.shape[2] == 2
        """
        if isinstance(channels, list):
            top_idx_mapping = channels
        else:
            channels = channel_spec.FusedChannelSpec.coerce(channels)
            # Computer subindex integer mapping
            request_codes = channels.as_list()
            top_codes = self.channels.as_oset()
            top_idx_mapping = [
                top_codes.index(code)
                for code in request_codes
            ]

        if self._immediates['chan_idxs'] is not None:
            new_chan_ixs = list(ub.take(self._immediates['chan_idxs'],
                                        top_idx_mapping))
        else:
            new_chan_ixs = top_idx_mapping

        channels = self.meta['channels']
        if channels is not None:
            new_chan_parsed = list(ub.take(channels.parsed, top_idx_mapping))
            channels = channel_spec.FusedChannelSpec(new_chan_parsed)

        num_bands = len(new_chan_ixs)

        new = self.__class__(
            fpath=self.meta['fpath'],
            num_bands=num_bands,
            channels=channels,
            dsize=self.dsize,
            immediate_dsize=self._immediates['dsize'],
            immediate_crop=self._immediates['crop'],
            immediate_chan_idxs=new_chan_ixs
        )
        return new


def have_gdal():
    try:
        from osgeo import gdal
    except ImportError:
        return False
    else:
        return gdal is not None


_GDAL_DTYPE_LUT = {
    1: np.uint8,     2: np.uint16,
    3: np.int16,     4: np.uint32,      5: np.int32,
    6: np.float32,   7: np.float64,     8: np.complex_,
    9: np.complex_,  10: np.complex64,  11: np.complex128
}


class LazyGDalFrameFile(ub.NiceRepr):
    """
    TODO:
        - [ ] Move to its own backend module
        - [ ] When used with COCO, allow the image metadata to populate the
              height, width, and channels if possible.

    Example:
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> self = LazyGDalFrameFile.demo()
        >>> print('self = {!r}'.format(self))
        >>> self[0:3, 0:3]
        >>> self[:, :, 0]
        >>> self[0]
        >>> self[0, 3]

        >>> # import kwplot
        >>> # kwplot.imshow(self[:])

    Example:
        >>> # See if we can reproduce the INTERLEAVE bug

        data = np.random.rand(128, 128, 64)
        import kwimage
        import ubelt as ub
        from os.path import join
        dpath = ub.ensure_app_cache_dir('kwcoco/tests/reader')
        fpath = join(dpath, 'foo.tiff')
        kwimage.imwrite(fpath, data, backend='skimage')
        recon1 = kwimage.imread(fpath)
        recon1.shape

        self = LazyGDalFrameFile(fpath)
        self.shape
        self[:]



    """
    def __init__(self, fpath):
        self.fpath = fpath

    @ub.memoize_property
    def _ds(self):
        from osgeo import gdal
        from os.path import exists
        if not exists(self.fpath):
            raise Exception('File does not exist: {}'.format(self.fpath))
        ds = gdal.Open(self.fpath, gdal.GA_ReadOnly)
        return ds

    @classmethod
    def demo(cls, key='astro', dsize=None):
        """
        Ignore:
            >>> self = LazyGDalFrameFile.demo(dsize=(6600, 4400))
        """
        from os.path import join
        cache_dpath = ub.ensure_app_cache_dir('kwcoco/demo')
        fpath = join(cache_dpath, key + '.cog.tiff')
        depends = ub.odict(dsize=dsize)
        cfgstr = ub.hash_data(depends)
        stamp = ub.CacheStamp(fname=key, cfgstr=cfgstr, dpath=cache_dpath,
                              product=[fpath])
        if stamp.expired():
            img = kwimage.grab_test_image(key, dsize=dsize)
            kwimage.imwrite(fpath, img, backend='gdal')
            stamp.renew()
        self = cls(fpath)
        return self

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def shape(self):
        # if 0:
        #     ds = self.ds
        #     INTERLEAVE = ds.GetMetadata('IMAGE_STRUCTURE').get('INTERLEAVE', '')  # handle INTERLEAVE=BAND
        #     if INTERLEAVE == 'BAND':
        #         pass
        #     ds.GetMetadata('')  # handle TIFFTAG_IMAGEDESCRIPTION
        #     from osgeo import gdal
        #     subdataset_infos = ds.GetSubDatasets()
        #     subdatasets = []
        #     for subinfo in subdataset_infos:
        #         path = subinfo[0]
        #         sub_ds = gdal.Open(path, gdal.GA_ReadOnly)
        #         subdatasets.append(sub_ds)
        #     for sub in subdatasets:
        #         sub.ReadAsArray()
        #         print((sub.RasterXSize, sub.RasterYSize, sub.RasterCount))
        #     sub = subdatasets[0][0]
        ds = self._ds
        width = ds.RasterXSize
        height = ds.RasterYSize
        C = ds.RasterCount
        return (height, width, C)

    @property
    def dtype(self):
        main_band = self._ds.GetRasterBand(1)
        dtype = _GDAL_DTYPE_LUT[main_band.DataType]
        return dtype

    def __nice__(self):
        from os.path import basename
        return '.../' + basename(self.fpath)

    def __getitem__(self, index):
        """
        References:
            https://gis.stackexchange.com/questions/162095/gdal-driver-create-typeerror

        Ignore:
            >>> self = LazyGDalFrameFile.demo(dsize=(6600, 4400))
            >>> index = [slice(2100, 2508, None), slice(4916, 5324, None), None]
            >>> img_part = self[index]
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(img_part)
        """
        ds = self._ds
        width = ds.RasterXSize
        height = ds.RasterYSize
        C = ds.RasterCount

        if 1:
            INTERLEAVE = ds.GetMetadata('IMAGE_STRUCTURE').get('INTERLEAVE', '')
            if INTERLEAVE == 'BAND':
                if len(ds.GetSubDatasets()) > 0:
                    raise NotImplementedError('Cannot handle interleaved files yet')

        if not ub.iterable(index):
            index = [index]

        index = list(index)
        if len(index) < 3:
            n = (3 - len(index))
            index = index + [None] * n

        ypart = _rectify_slice_dim(index[0], height)
        xpart = _rectify_slice_dim(index[1], width)
        channel_part = _rectify_slice_dim(index[2], C)
        trailing_part = [channel_part]

        if len(trailing_part) == 1:
            channel_part = trailing_part[0]
            if isinstance(channel_part, list):
                band_indices = channel_part
            else:
                band_indices = range(*channel_part.indices(C))
        else:
            band_indices = range(C)
            assert len(trailing_part) <= 1

        ystart, ystop = map(int, [ypart.start, ypart.stop])
        xstart, xstop = map(int, [xpart.start, xpart.stop])

        ysize = ystop - ystart
        xsize = xstop - xstart

        gdalkw = dict(xoff=xstart, yoff=ystart,
                      win_xsize=xsize, win_ysize=ysize)

        PREALLOC = 1
        if PREALLOC:
            # preallocate like kwimage.im_io._imread_gdal
            from kwimage.im_io import _gdal_to_numpy_dtype
            shape = (ysize, xsize, len(band_indices))
            bands = [ds.GetRasterBand(1 + band_idx)
                     for band_idx in band_indices]
            gdal_dtype = bands[0].DataType
            dtype = _gdal_to_numpy_dtype(gdal_dtype)
            img_part = np.empty(shape, dtype=dtype)
            for out_idx, band in enumerate(bands):
                img_part[:, :, out_idx] = band.ReadAsArray(**gdalkw)
        else:
            channels = []
            for band_idx in band_indices:
                band = ds.GetRasterBand(1 + band_idx)
                channel = band.ReadAsArray(**gdalkw)
                channels.append(channel)
            img_part = np.dstack(channels)
        return img_part

    def __array__(self):
        """
        Allow this object to be passed to np.asarray

        References:
            https://numpy.org/doc/stable/user/basics.dispatch.html
        """
        return self[:]


def validate_nonzero_data(file):
    """
    Test to see if the image is all black.

    May fail on all-black images

    Example:
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> import kwimage
        >>> gpath = kwimage.grab_test_image_fpath()
        >>> file = LazyGDalFrameFile(gpath)
        >>> validate_nonzero_data(file)
    """
    try:
        import numpy as np
        # Find center point of the image
        cx, cy = np.array(file.shape[0:2]) // 2
        center = [cx, cy]
        # Check if the center pixels have data, look at more data if needbe
        sizes = [8, 512, 2048, 5000]
        for d in sizes:
            index = tuple(slice(c - d, c + d) for c in center)
            partial_data = file[index]
            total = partial_data.sum()
            if total > 0:
                break
        if total == 0:
            total = file[:].sum()
        has_data = total > 0
    except Exception:
        has_data = False
    return has_data


def _rectify_slice_dim(part, D):
    if part is None:
        return slice(0, D)
    elif isinstance(part, slice):
        start = 0 if part.start is None else max(0, part.start)
        stop = D if part.stop is None else min(D, part.stop)
        if stop < 0:
            stop = D + stop
        assert part.step is None
        part = slice(start, stop)
        return part
    elif isinstance(part, int):
        part = slice(part, part + 1)
    elif isinstance(part, list):
        part = part
    else:
        raise TypeError(part)
    return part


class DelayedFrameConcat(DelayedVideoOperation):
    """
    Represents multiple frames in a video

    Note:

        .. code::

            Video[0]:
                Frame[0]:
                    Chan[0]: (32) +--------------------------------+
                    Chan[1]: (16) +----------------+
                    Chan[2]: ( 8) +--------+
                Frame[1]:
                    Chan[0]: (30) +------------------------------+
                    Chan[1]: (14) +--------------+
                    Chan[2]: ( 6) +------+

    TODO:
        - [ ] Support computing the transforms when none of the data is loaded

    Example:
        >>> # Simpler case with fewer nesting levels
        >>> rng = kwarray.ensure_rng(None)
        >>> # Delayed warp each channel into its "image" space
        >>> # Note: the images never enter the space we transform through
        >>> f1_img = DelayedLoad.demo('astro', (300, 300))
        >>> f2_img = DelayedLoad.demo('carl', (256, 256))
        >>> # Combine frames into a video
        >>> vid_dsize = np.array((100, 100))
        >>> self = vid = DelayedFrameConcat([
        >>>     f1_img.delayed_warp(Affine.scale(vid_dsize / f1_img.dsize)),
        >>>     f2_img.delayed_warp(Affine.scale(vid_dsize / f2_img.dsize)),
        >>> ], dsize=vid_dsize)
        >>> print(ub.repr2(vid.nesting(), nl=-1, sort=0))
        >>> final = vid.finalize(interpolation='nearest', dsize=(32, 32))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(final[0], pnum=(1, 2, 1), fnum=1)
        >>> kwplot.imshow(final[1], pnum=(1, 2, 2), fnum=1)
        >>> region_slices = (slice(0, 90), slice(30, 60))
    """
    def __init__(self, frames, dsize=None):
        self.frames = frames
        if dsize is None:
            dsize_cands = [frame.dsize for frame in self.frames]
            dsize = _largest_shape(dsize_cands)

        self.dsize = dsize
        nband_cands = [frame.num_bands for frame in self.frames]
        if any(c is None for c in nband_cands):
            num_bands = None
        if ub.allsame(nband_cands):
            num_bands = nband_cands[0]
        else:
            raise ValueError(
                'components must all have the same delayed size')
        self.num_bands = num_bands
        self.num_frames = len(self.frames)
        self.meta = {
            'num_bands': self.num_bands,
            'num_frames': self.num_frames,
            'shape': self.shape,
        }

    def children(self):
        yield from self.frames

    @property
    def channels(self):
        # Assume all channels are the same, or at least aligned via nans?
        return self.frames[0].channels

    @property
    def shape(self):
        w, h = self.dsize
        return (self.num_frames, h, w, self.num_bands)

    def finalize(self, **kwargs):
        """
        Execute the final transform
        """
        # Add in the video axis
        # as_xarray = kwargs.get('as_xarray', False)

        stack = [frame.finalize(**kwargs)[None, :]
                 for frame in self.frames]
        stack_shapes = np.array([s.shape for s in stack])

        stack_whc = stack_shapes[:, 1:4]
        max_whc = stack_whc.max(axis=0)
        delta_whc = max_whc - stack_whc

        stack2 = []
        for delta, item in zip(delta_whc, stack):
            pad_width = [(0, 0)] + list(zip([0] * len(delta), delta))
            item = np.pad(item, pad_width=pad_width,)
            stack2.append(item)

        final = np.concatenate(stack2, axis=0)
        return final

    def delayed_crop(self, region_slices):
        """
        Example:
            >>> from kwcoco.util.util_delayed_poc import *  # NOQA
            >>> # Create raw channels in some "native" resolution for frame 1
            >>> f1_chan1 = DelayedIdentity.demo('astro', chan=(1, 0), dsize=(300, 300))
            >>> f1_chan2 = DelayedIdentity.demo('astro', chan=2, dsize=(10, 10))
            >>> # Create raw channels in some "native" resolution for frame 2
            >>> f2_chan1 = DelayedIdentity.demo('carl', dsize=(64, 64), chan=(1, 0))
            >>> f2_chan2 = DelayedIdentity.demo('carl', dsize=(10, 10), chan=2)
            >>> #
            >>> f1_dsize = np.array(f1_chan1.dsize)
            >>> f2_dsize = np.array(f2_chan1.dsize)
            >>> f1_img = DelayedChannelConcat([
            >>>     f1_chan1.delayed_warp(Affine.scale(f1_dsize / f1_chan1.dsize), dsize=f1_dsize),
            >>>     f1_chan2.delayed_warp(Affine.scale(f1_dsize / f1_chan2.dsize), dsize=f1_dsize),
            >>> ])
            >>> f2_img = DelayedChannelConcat([
            >>>     f2_chan1.delayed_warp(Affine.scale(f2_dsize / f2_chan1.dsize), dsize=f2_dsize),
            >>>     f2_chan2.delayed_warp(Affine.scale(f2_dsize / f2_chan2.dsize), dsize=f2_dsize),
            >>> ])
            >>> vid_dsize = np.array((280, 280))
            >>> full_vid = DelayedFrameConcat([
            >>>     f1_img.delayed_warp(Affine.scale(vid_dsize / f1_img.dsize), dsize=vid_dsize),
            >>>     f2_img.delayed_warp(Affine.scale(vid_dsize / f2_img.dsize), dsize=vid_dsize),
            >>> ])
            >>> region_slices = (slice(80, 200), slice(80, 200))
            >>> print(ub.repr2(full_vid.nesting(), nl=-1, sort=0))
            >>> crop_vid = full_vid.delayed_crop(region_slices)
            >>> final_full = full_vid.finalize(interpolation='nearest')
            >>> final_crop = crop_vid.finalize(interpolation='nearest')
            >>> import pytest
            >>> with pytest.raises(ValueError):
            >>>     # should not be able to crop a crop yet
            >>>     crop_vid.delayed_crop(region_slices)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(final_full[0], pnum=(2, 2, 1), fnum=1)
            >>> kwplot.imshow(final_full[1], pnum=(2, 2, 2), fnum=1)
            >>> kwplot.imshow(final_crop[0], pnum=(2, 2, 3), fnum=1)
            >>> kwplot.imshow(final_crop[1], pnum=(2, 2, 4), fnum=1)
        """
        # DEBUG_PRINT('DelayedFrameConcat.delayed_crop')
        new_frames = []
        for frame in self.frames:
            new_frame = frame.delayed_crop(region_slices)
            new_frames.append(new_frame)
        new = DelayedFrameConcat(new_frames)
        return new


class DelayedChannelConcat(DelayedImageOperation):
    """
    Represents multiple channels in an image that could be concatenated

    Attributes:
        components (List[DelayedWarp]): a list of stackable channels. Each
            component may be comprised of multiple channels.

    TODO:
        - [ ] can this be generalized into a delayed concat?
        - [ ] can all concats be delayed until the very end?

    Example:
        >>> comp1 = DelayedWarp(np.random.rand(11, 7))
        >>> comp2 = DelayedWarp(np.random.rand(11, 7, 3))
        >>> comp3 = DelayedWarp(
        >>>     np.random.rand(3, 5, 2),
        >>>     transform=Affine.affine(scale=(7/5, 11/3)).matrix,
        >>>     dsize=(7, 11)
        >>> )
        >>> components = [comp1, comp2, comp3]
        >>> chans = DelayedChannelConcat(components)
        >>> final = chans.finalize()
        >>> assert final.shape == chans.shape
        >>> assert final.shape == (11, 7, 6)

        >>> # We should be able to nest DelayedChannelConcat inside virutal images
        >>> frame1 = DelayedWarp(
        >>>     chans, transform=Affine.affine(scale=2.2).matrix,
        >>>     dsize=(20, 26))
        >>> frame2 = DelayedWarp(
        >>>     np.random.rand(3, 3, 6), dsize=(20, 26))
        >>> frame3 = DelayedWarp(
        >>>     np.random.rand(3, 3, 6), dsize=(20, 26))

        >>> print(ub.repr2(frame1.nesting(), nl=-1, sort=False))
        >>> frame1.finalize()
        >>> vid = DelayedFrameConcat([frame1, frame2, frame3])
        >>> print(ub.repr2(vid.nesting(), nl=-1, sort=False))
    """
    def __init__(self, components, dsize=None):
        if len(components) == 0:
            raise ValueError('No components to concatenate')
        self.components = components
        if dsize is None:
            dsize_cands = [comp.dsize for comp in self.components]
            if not ub.allsame(dsize_cands):
                raise ValueError(
                    'components must all have the same delayed size')
            dsize = dsize_cands[0]
        self.dsize = dsize
        self.num_bands = sum(comp.num_bands for comp in self.components)
        self.meta = {
            'shape': self.shape,
            'num_bands': self.num_bands,
        }

    def children(self):
        yield from self.components

    @classmethod
    def random(cls, num_parts=3, rng=None):
        """
        Example:
            >>> self = DelayedChannelConcat.random()
            >>> print('self = {!r}'.format(self))
            >>> print(ub.repr2(self.nesting(), nl=-1, sort=0))
        """
        rng = kwarray.ensure_rng(rng)
        self_w = rng.randint(8, 64)
        self_h = rng.randint(8, 64)
        components = []
        for _ in range(num_parts):
            subcomp = DelayedWarp.random(rng=rng)
            tf = Affine.random(rng=rng).matrix
            comp = subcomp.delayed_warp(tf, dsize=(self_w, self_h))
            components.append(comp)
        self = DelayedChannelConcat(components)
        return self

    @property
    def channels(self):
        sub_channs = []
        for comp in self.components:
            sub_channs.append(comp.channels)
        channs = channel_spec.FusedChannelSpec.concat(sub_channs)
        return channs

    @property
    def shape(self):
        w, h = self.dsize
        return (h, w, self.num_bands)

    def finalize(self, **kwargs):
        """
        Execute the final transform
        """
        as_xarray = kwargs.get('as_xarray', False)
        stack = [comp.finalize(**kwargs) for comp in self.components]
        if len(stack) == 1:
            final = stack[0]
        else:
            if as_xarray:
                import xarray as xr
                final = xr.concat(stack, dim='c')
            else:
                final = np.concatenate(stack, axis=2)
        return final

    @profile
    def take_channels(self, channels):
        """
        This method returns a subset of the vision data with only the
        specified bands / channels.

        Args:
            channels (List[int] | slice | channel_spec.FusedChannelSpec):
                List of integers indexes, a slice, or a channel spec, which is
                typically a pipe (`|`) delimited list of channel codes. See
                kwcoco.ChannelSpec for more detials.

        Returns:
            DelayedVisionOperation:
                a delayed vision operation that only operates on the following
                channels.

        Example:
            >>> from kwcoco.util.util_delayed_poc import *  # NOQA
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> self = delayed = dset.delayed_load(1)
            >>> channels = 'B11|B8|B1|B10'
            >>> new = self.take_channels(channels)

        Example:
            >>> # Complex case
            >>> import kwcoco
            >>> from kwcoco.util.util_delayed_poc import *  # NOQA
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> delayed = dset.delayed_load(1)
            >>> astro = DelayedLoad.demo('astro').load_shape(use_channel_heuristic=True)
            >>> aligned = astro.warp(kwimage.Affine.scale(600 / 512), dsize='auto')
            >>> self = combo = DelayedChannelConcat(delayed.components + [aligned])
            >>> channels = 'B1|r|B8|g'
            >>> new = self.take_channels(channels)
            >>> new_cropped = new.crop((slice(10, 200), slice(12, 350)))
            >>> datas = new_cropped.finalize()
            >>> vizable = kwimage.normalize_intensity(datas, axis=2)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> stacked = kwimage.stack_images(vizable.transpose(2, 0, 1))
            >>> kwplot.imshow(stacked)

        Example:
            >>> # Test case where requested channel does not exist
            >>> import kwcoco
            >>> from kwcoco.util.util_delayed_poc import *  # NOQA
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> self = dset.delayed_load(1)
            >>> channels = 'B1|foobar|bazbiz|B8'
            >>> new = self.take_channels(channels)
            >>> new_cropped = new.crop((slice(10, 200), slice(12, 350)))
            >>> fused = new_cropped.finalize()
            >>> assert fused.shape == (190, 338, 4)
            >>> assert np.all(np.isnan(fused[..., 1:3]))
            >>> assert not np.any(np.isnan(fused[..., 0]))
            >>> assert not np.any(np.isnan(fused[..., 3]))
        """
        if isinstance(channels, list):
            top_idx_mapping = channels
            top_codes = self.channels.as_list()
            request_codes = None
        else:
            channels = channel_spec.FusedChannelSpec.coerce(channels)
            # Computer subindex integer mapping
            request_codes = channels.as_list()
            top_codes = self.channels.as_oset()
            top_idx_mapping = []
            for code in request_codes:
                try:
                    top_idx_mapping.append(top_codes.index(code))
                except KeyError:
                    top_idx_mapping.append(None)

        # Rearange subcomponents into the specified channel representation
        # I am not confident that this logic is the best way to do this.
        # This may be a bottleneck
        subindexer = kwarray.FlatIndexer([
            comp.num_bands for comp in self.components])

        accum = []
        class ContiguousSegment(object):
            def __init__(self, comp, start):
                self.comp = comp
                self.start = start
                self.stop = start + 1
                self.codes = []

        curr = None
        for request_idx, idx in enumerate(top_idx_mapping):
            if idx is None:
                # Requested channel does not exist in our data stack
                comp = None
                inner = 0
                if curr is not None and curr.comp is None:
                    inner = curr.stop
            else:
                # Requested channel exists in our data stack
                outer, inner = subindexer.unravel(idx)
                comp = self.components[outer]
            if curr is None:
                curr = ContiguousSegment(comp, inner)
            else:
                is_contiguous = curr.comp is comp and (inner == curr.stop)
                if is_contiguous:
                    # extend the previous contiguous segment
                    curr.stop = inner + 1
                else:
                    # accept previous segment and start a new one
                    accum.append(curr)
                    curr = ContiguousSegment(comp, inner)

            # Hack for nans
            if request_codes is not None:
                curr.codes.append(request_codes[request_idx])

        # Accumulate final segment
        if curr is not None:
            accum.append(curr)

        # Execute the delayed operation
        new_components = []
        for curr in accum:
            comp = curr.comp
            if comp is None:
                # Requested component did not exist, return nans
                if request_codes is not None:
                    nan_chan = channel_spec.FusedChannelSpec(curr.codes)
                else:
                    nan_chan = None
                comp = DelayedNans(self.dsize, channels=nan_chan)
                new_components.append(comp)
            else:
                if curr.start == 0 and curr.stop == comp.num_bands:
                    # Entire component is valid, no need for sub-operation
                    new_components.append(comp)
                else:
                    # Only part of the component is taken, need to sub-operate
                    # It would be nice if we only loaded the file once if we need
                    # multiple parts discontiguously.
                    sub_idxs = list(range(curr.start, curr.stop))
                    sub_comp = comp.take_channels(sub_idxs)
                    new_components.append(sub_comp)

        new = DelayedChannelConcat(new_components)
        return new


class DelayedWarp(DelayedImageOperation):
    """
    POC for chainable transforms

    Note:
        "sub" is used to refer to the underlying data in its native coordinates
        and resolution.

        "self" is used to refer to the data in the transformed coordinates that
        are exposed by this class.

    Attributes:

        sub_data (DelayedWarp | ArrayLike):
            array-like image data at a naitive resolution

        transform (Transform):
            transforms data from native "sub"-image-space to
            "self"-image-space.

    Example:
        >>> dsize = (12, 12)
        >>> tf1 = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        >>> tf2 = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 1]])
        >>> tf3 = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 1]])
        >>> band1 = DelayedWarp(np.random.rand(6, 6), tf1, dsize)
        >>> band2 = DelayedWarp(np.random.rand(4, 4), tf2, dsize)
        >>> band3 = DelayedWarp(np.random.rand(3, 3), tf3, dsize)
        >>> #
        >>> # Execute a crop in a one-level transformed space
        >>> region_slices = (slice(5, 10), slice(0, 12))
        >>> delayed_crop = band2.delayed_crop(region_slices)
        >>> final_crop = delayed_crop.finalize()
        >>> #
        >>> # Execute a crop in a nested transformed space
        >>> tf4 = np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1]])
        >>> chained = DelayedWarp(band2, tf4, (18, 18))
        >>> delayed_crop = chained.delayed_crop(region_slices)
        >>> final_crop = delayed_crop.finalize()
        >>> #
        >>> tf4 = np.array([[.5, 0, 0], [0, .5, 0], [0, 0, 1]])
        >>> chained = DelayedWarp(band2, tf4, (6, 6))
        >>> delayed_crop = chained.delayed_crop(region_slices)
        >>> final_crop = delayed_crop.finalize()
        >>> #
        >>> region_slices = (slice(1, 5), slice(2, 4))
        >>> delayed_crop = chained.delayed_crop(region_slices)
        >>> final_crop = delayed_crop.finalize()

    Example:
        >>> dsize = (17, 12)
        >>> tf = np.array([[5.2, 0, 1.1], [0, 3.1, 2.2], [0, 0, 1]])
        >>> self = DelayedWarp(np.random.rand(3, 5, 13), tf, dsize=dsize)
        >>> self.finalize().shape
    """
    def __init__(self, sub_data, transform=None, dsize=None):
        transform = Affine.coerce(transform)

        # TODO: We probably don't need to track sub-bounds, size, shape
        # or any of that anywhere except at the root and leaf.

        try:
            if hasattr(sub_data, 'bounds'):
                sub_shape = sub_data.shape
                sub_bounds = sub_data.bounds
            else:
                sub_shape = sub_data.shape
                sub_h, sub_w = sub_shape[0:2]
                sub_bounds = kwimage.Coords(
                    np.array([[0,     0], [sub_w, 0],
                              [0, sub_h], [sub_w, sub_h]])
                )
            self.bounds = sub_bounds.warp(transform.matrix)
            if dsize is ub.NoParam:
                pass
            elif dsize is None:
                (h, w) = sub_shape[0:2]
                dsize = (w, h)
            elif isinstance(dsize, str):
                if dsize == 'auto':
                    max_xy = np.ceil(self.bounds.data.max(axis=0))
                    max_x = int(max_xy[0])
                    max_y = int(max_xy[1])
                    dsize = (max_x, max_y)
                else:
                    raise KeyError(dsize)
            else:
                if isinstance(dsize, np.ndarray):
                    dsize = tuple(map(int, dsize))
                dsize = dsize

            if len(sub_data.shape) == 2:
                num_bands = 1
            elif len(sub_data.shape) == 3:
                num_bands = sub_data.shape[2]
            else:
                raise ValueError(
                    'Data may only have 2 space dimensions and 1 channel '
                    'dimension')
        except Exception:
            num_bands = None

        self.sub_data = sub_data
        self.meta = {
            'dsize': dsize,
            'num_bands': num_bands,
            'transform': transform,
        }

    @classmethod
    def random(cls, nesting=(2, 5), rng=None):
        """
        Example:
            >>> self = DelayedWarp.random(nesting=(4, 7))
            >>> print('self = {!r}'.format(self))
            >>> print(ub.repr2(self.nesting(), nl=-1, sort=0))
        """
        from kwarray.distributions import DiscreteUniform, Uniform
        rng = kwarray.ensure_rng(rng)
        chan_distri = DiscreteUniform.coerce((1, 5), rng=rng)
        nest_distri = DiscreteUniform.coerce(nesting, rng=rng)
        size_distri = DiscreteUniform.coerce((8, 64), rng=rng)
        raw_distri = Uniform(rng=rng)
        leaf_c = chan_distri.sample()
        leaf_w = size_distri.sample()
        leaf_h = size_distri.sample()
        raw = raw_distri.sample(leaf_h, leaf_w, leaf_c)
        layer = raw
        depth = nest_distri.sample()
        for _ in range(depth):
            tf = Affine.random(rng=rng).matrix
            layer = DelayedWarp(layer, tf, dsize='auto')
        self = layer
        return self

    @property
    def channels(self):
        if hasattr(self.sub_data, 'channels'):
            return self.sub_data.channels
        else:
            return None

    def children(self):
        yield self.sub_data

    @property
    def dsize(self):
        return self.meta['dsize']

    @property
    def num_bands(self):
        return self.meta['num_bands']

    @property
    def shape(self):
        # trailing_shape = self.sub_data.shape[2:]
        # trailing shape should only be allowed to have 0 or 1 dimension
        if self.meta['dsize'] is None:
            w = h = None
        else:
            w, h = self.meta['dsize']
        return (h, w, self.meta['num_bands'])

    def _optimize_paths(self, **kwargs):
        """
        Example:
            >>> self = DelayedWarp.random()
            >>> leafs = list(self._optimize_paths())
            >>> print('leafs = {!r}'.format(leafs))
        """
        # DEBUG_PRINT('DelayedWarp._optimize_paths')
        dsize = kwargs.get('dsize', None)
        transform = kwargs.get('transform', None)
        if dsize is None:
            dsize = self.meta['dsize']
        if transform is None:
            transform = self.meta['transform']
        else:
            transform = kwargs.get('transform', None) @ self.meta['transform']
        kwargs['dsize'] = dsize
        kwargs['transform'] = transform
        sub_data = self.sub_data
        flag = getattr(sub_data, '__hack_dont_optimize__', False)
        if hasattr(sub_data, '_optimize_paths') and not flag:
            yield from sub_data._optimize_paths(
                transform=transform, dsize=dsize)
        else:
            leaf = DelayedWarp(sub_data, transform, dsize=dsize)
            yield leaf

    def finalize(self, transform=None, dsize=None, interpolation='linear',
                 **kwargs):
        """
        Execute the final transform

        Can pass a parent transform to augment this underlying transform.

        Args:
            transform (Transform): an additional transform to perform
            dsize (Tuple[int, int]): overrides destination canvas size

        Example:
            >>> tf = np.array([[0.9, 0, 3.9], [0, 1.1, -.5], [0, 0, 1]])
            >>> raw = kwimage.grab_test_image(dsize=(54, 65))
            >>> raw = kwimage.ensure_float01(raw)
            >>> # Test nested finalize
            >>> layer1 = raw
            >>> num = 10
            >>> for _ in range(num):
            ...     layer1  = DelayedWarp(layer1, tf, dsize='auto')
            >>> final1 = layer1.finalize()
            >>> # Test non-nested finalize
            >>> layer2 = list(layer1._optimize_paths())[0]
            >>> final2 = layer2.finalize()
            >>> #
            >>> print(ub.repr2(layer1.nesting(), nl=-1, sort=0))
            >>> print(ub.repr2(layer2.nesting(), nl=-1, sort=0))
            >>> print('final1 = {!r}'.format(final1))
            >>> print('final2 = {!r}'.format(final2))
            >>> print('final1.shape = {!r}'.format(final1.shape))
            >>> print('final2.shape = {!r}'.format(final2.shape))
            >>> assert np.allclose(final1, final2)
            >>> #
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(raw, pnum=(1, 3, 1), fnum=1)
            >>> kwplot.imshow(final1, pnum=(1, 3, 2), fnum=1)
            >>> kwplot.imshow(final2, pnum=(1, 3, 3), fnum=1)
            >>> kwplot.show_if_requested()

        Example:
            >>> # Test aliasing
            >>> s = DelayedIdentity.demo()
            >>> s = DelayedIdentity.demo('checkerboard')
            >>> a = s.delayed_warp(Affine.scale(0.05), dsize='auto')
            >>> b = s.delayed_warp(Affine.scale(3), dsize='auto')

            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> # It looks like downsampling linear and area is the same
            >>> # Does warpAffine have no alias handling?
            >>> pnum_ = kwplot.PlotNums(nRows=2, nCols=4)
            >>> kwplot.imshow(a.finalize(interpolation='area'), pnum=pnum_(), title='warpAffine area')
            >>> kwplot.imshow(a.finalize(interpolation='linear'), pnum=pnum_(), title='warpAffine linear')
            >>> kwplot.imshow(a.finalize(interpolation='nearest'), pnum=pnum_(), title='warpAffine nearest')
            >>> kwplot.imshow(a.finalize(interpolation='nearest', antialias=False), pnum=pnum_(), title='warpAffine nearest AA=0')
            >>> kwplot.imshow(kwimage.imresize(s.finalize(), dsize=a.dsize, interpolation='area'), pnum=pnum_(), title='resize area')
            >>> kwplot.imshow(kwimage.imresize(s.finalize(), dsize=a.dsize, interpolation='linear'), pnum=pnum_(), title='resize linear')
            >>> kwplot.imshow(kwimage.imresize(s.finalize(), dsize=a.dsize, interpolation='nearest'), pnum=pnum_(), title='resize nearest')
            >>> kwplot.imshow(kwimage.imresize(s.finalize(), dsize=a.dsize, interpolation='cubic'), pnum=pnum_(), title='resize cubic')
        """
        # todo: needs to be extended for the case where the sub_data is a
        # nested chain of transforms.
        # import cv2
        # from kwimage import im_cv2
        if dsize is None:
            dsize = self.meta['dsize']
        transform = Affine.coerce(transform) @ self.meta['transform']
        sub_data = self.sub_data
        flag = getattr(sub_data, '__hack_dont_optimize__', False)
        if flag:
            sub_data = sub_data.finalize()

        if hasattr(sub_data, 'finalize'):
            # Branch finalize
            final = sub_data.finalize(transform=transform, dsize=dsize,
                                      interpolation=interpolation, **kwargs)
            # Ensure that the last dimension is channels
            final = kwarray.atleast_nd(final, 3, front=False)
        else:
            as_xarray = kwargs.get('as_xarray', False)
            # Leaf finalize
            # flags = im_cv2._coerce_interpolation(interpolation)
            if dsize == (None, None):
                dsize = None
            sub_data_ = np.asarray(sub_data)
            M = np.asarray(transform)
            antialias = kwargs.get('antialias', True)
            final = kwimage.warp_affine(sub_data_, M, dsize=dsize,
                                        interpolation=interpolation,
                                        antialias=antialias)
            # final = cv2.warpPerspective(sub_data_, M, dsize=dsize, flags=flags)
            # Ensure that the last dimension is channels
            final = kwarray.atleast_nd(final, 3, front=False)
            if as_xarray:
                import xarray as xr
                channels = self.channels
                coords = {}
                if channels is not None:
                    coords['c'] = channels.code_list()
                final = xr.DataArray(final, dims=('y', 'x', 'c'), coords=coords)

        return final

    def take_channels(self, channels):
        new_subdata = self.sub_data.take_channels(channels)
        new = self.__class__(new_subdata, transform=self.meta['transform'],
                             dsize=self.meta['dsize'])
        return new


class DelayedCrop(DelayedImageOperation):
    """
    Represent a delayed crop operation

    Example:
        >>> sub_data = DelayedLoad.demo()
        >>> sub_slices = (slice(5, 10), slice(1, 12))
        >>> self = DelayedCrop(sub_data, sub_slices)
        >>> print(ub.repr2(self.nesting(), nl=-1, sort=0))
        >>> final = self.finalize()
        >>> print('final.shape = {!r}'.format(final.shape))

    Example:
        >>> sub_data = DelayedLoad.demo()
        >>> sub_slices = (slice(5, 10), slice(1, 12))
        >>> crop1 = DelayedCrop(sub_data, sub_slices)
        >>> import pytest
        >>> # Should only error while huristics are in use.
        >>> with pytest.raises(ValueError):
        >>>     crop2 = DelayedCrop(crop1, sub_slices)
    """

    __hack_dont_optimize__ = True

    def __init__(self, sub_data, sub_slices):
        if isinstance(sub_data, (DelayedCrop, DelayedWarp, DelayedChannelConcat)):
            raise ValueError('cant crop generally yet')

        self.sub_data = sub_data
        self.sub_slices = sub_slices

        sl_x, sl_y = sub_slices[0:2]
        width = sl_x.stop - sl_x.start
        height = sl_y.stop - sl_y.start
        if hasattr(sub_data, 'num_bands'):
            num_bands = sub_data.num_bands
        else:
            num_bands = kwimage.num_channels(self.sub_data)
        self.num_bands = num_bands
        self.shape = (height, width, num_bands)
        self.meta = {
            'shape': self.shape,
            'sub_slices': self.sub_slices,
            'num_bands': self.num_bands,
        }

    @property
    def channels(self):
        if hasattr(self.sub_data, 'channels'):
            return self.sub_data.channels
        else:
            return None

    def children(self):
        yield self.sub_data

    def finalize(self, **kwargs):
        if hasattr(self.sub_data, 'finalize'):
            return self.sub_data.finalize(**kwargs)[self.sub_slices]
        else:
            return self.sub_data[self.sub_slices]

    def _optimize_paths(self, **kwargs):
        raise NotImplementedError('cant look at leafs through crop atm')


def _compute_leaf_subcrop(root_region_bounds, tf_leaf_to_root):
    r"""
    Given a region in a "root" image and a trasnform between that "root" and
    some "leaf" image, compute the appropriate quantized region in the "leaf"
    image and the adjusted transformation between that root and leaf.

    Example:
        >>> region_slices = (slice(33, 100), slice(22, 62))
        >>> region_shape = (100, 100, 1)
        >>> root_region_box = kwimage.Boxes.from_slice(region_slices, shape=region_shape)
        >>> root_region_bounds = root_region_box.to_polygons()[0]
        >>> tf_leaf_to_root = Affine.affine(scale=7).matrix
        >>> slices, tf_new = _compute_leaf_subcrop(root_region_bounds, tf_leaf_to_root)
        >>> print('tf_new =\n{!r}'.format(tf_new))
        >>> print('slices = {!r}'.format(slices))

    Ignore:

        root_region_bounds = kwimage.Coords.random(4)
        tf_leaf_to_root = np.eye(3)
        tf_leaf_to_root[0, 2] = -1e-11

    """
    # Transform the region bounds into the sub-image space
    tf_root_to_leaf = np.asarray(Affine.coerce(tf_leaf_to_root).inv())
    leaf_region_bounds = root_region_bounds.warp(tf_root_to_leaf)
    leaf_region_box = leaf_region_bounds.bounding_box().to_ltrb()

    # Quantize to a region that is possible to sample from
    leaf_crop_box = leaf_region_box.quantize()

    # is this ok?
    leaf_crop_box = leaf_crop_box.clip(0, 0, None, None)

    # Because we sampled a large quantized region, we need to modify the
    # transform to nudge it a bit to the left, undoing the quantization,
    # which has a bit of extra padding on the left, before applying the
    # final transform.
    # subpixel_offset = leaf_region_box.data[0, 0:2]
    crop_offset = leaf_crop_box.data[0, 0:2]
    root_offset = root_region_bounds.exterior.data.min(axis=0)

    tf_root_to_newroot = Affine.affine(offset=root_offset).inv().matrix
    tf_newleaf_to_leaf = Affine.affine(offset=crop_offset).matrix

    # Resample the smaller region to align it with the root region
    # Note: The right most transform is applied first
    tf_newleaf_to_newroot = (
        tf_root_to_newroot @
        tf_leaf_to_root @
        tf_newleaf_to_leaf
    )

    lt_x, lt_y, rb_x, rb_y = leaf_crop_box.data[0, 0:4]
    leaf_crop_slices = (slice(lt_y, rb_y), slice(lt_x, rb_x))

    return leaf_crop_slices, tf_newleaf_to_newroot


def _largest_shape(shapes):
    """
    Finds maximum over all shapes

    Example:
        >>> shapes = [
        >>>     (10, 20), None, (None, 30), (40, 50, 60, None), (100,)
        >>> ]
        >>> largest = _largest_shape(shapes)
        >>> print('largest = {!r}'.format(largest))
        >>> assert largest == (100, 50, 60, None)
    """
    def _nonemax(a, b):
        if a is None or b is None:
            return a or b
        return max(a, b)
    import itertools as it
    largest = []
    for shape in shapes:
        if shape is not None:
            largest = [
                _nonemax(c1, c2)
                for c1, c2 in it.zip_longest(largest, shape, fillvalue=None)
            ]
    largest = tuple(largest)
    return largest


def _devcheck_corner():
    self = DelayedWarp.random(rng=0)
    print(self.nesting())
    region_slices = (slice(40, 90), slice(20, 62))
    region_box = kwimage.Boxes.from_slice(region_slices, shape=self.shape)
    region_bounds = region_box.to_polygons()[0]

    for leaf in self._optimize_paths():
        pass

    tf_leaf_to_root = leaf['transform']
    tf_root_to_leaf = np.linalg.inv(tf_leaf_to_root)

    leaf_region_bounds = region_bounds.warp(tf_root_to_leaf)
    leaf_region_box = leaf_region_bounds.bounding_box().to_ltrb()
    leaf_crop_box = leaf_region_box.quantize()
    lt_x, lt_y, rb_x, rb_y = leaf_crop_box.data[0, 0:4]

    root_crop_corners = leaf_crop_box.to_polygons()[0].warp(tf_leaf_to_root)

    # leaf_crop_slices = (slice(lt_y, rb_y), slice(lt_x, rb_x))

    crop_offset = leaf_crop_box.data[0, 0:2]
    corner_offset = leaf_region_box.data[0, 0:2]
    offset_xy = crop_offset - corner_offset

    tf_root_to_leaf

    # NOTE:

    # Cropping applies a translation in whatever space we do it in
    # We need to save the bounds of the crop.
    # But now we need to adjust the transform so it points to the
    # cropped-leaf-space not just the leaf-space, so we invert the implicit
    # crop

    tf_crop_to_leaf = Affine.affine(offset=crop_offset)

    # tf_newroot_to_root = Affine.affine(offset=region_box.data[0, 0:2])
    tf_root_to_newroot = Affine.affine(offset=region_box.data[0, 0:2]).inv()

    tf_crop_to_leaf = Affine.affine(offset=crop_offset)
    tf_crop_to_newroot = tf_root_to_newroot @ tf_leaf_to_root @ tf_crop_to_leaf
    tf_newroot_to_crop = tf_crop_to_newroot.inv()

    # tf_leaf_to_crop
    # tf_corner_offset = Affine.affine(offset=offset_xy)

    subpixel_offset = Affine.affine(offset=offset_xy).matrix
    tf_crop_to_leaf = subpixel_offset
    # tf_crop_to_root = tf_leaf_to_root @ tf_crop_to_leaf
    # tf_root_to_crop = np.linalg.inv(tf_crop_to_root)

    if 1:
        import kwplot
        kwplot.autoplt()

        lw, lh = leaf['sub_data_shape'][0:2]
        leaf_box = kwimage.Boxes([[0, 0, lw, lh]], 'xywh')
        root_box = kwimage.Boxes([[0, 0, self.dsize[0], self.dsize[1]]], 'xywh')

        ax1 = kwplot.figure(fnum=1, pnum=(2, 2, 1), doclf=1).gca()
        ax2 = kwplot.figure(fnum=1, pnum=(2, 2, 2)).gca()
        ax3 = kwplot.figure(fnum=1, pnum=(2, 2, 3)).gca()
        ax4 = kwplot.figure(fnum=1, pnum=(2, 2, 4)).gca()
        root_box.draw(setlim=True, ax=ax1)
        leaf_box.draw(setlim=True, ax=ax2)

        region_bounds.draw(ax=ax1, color='green', alpha=.4)
        leaf_region_bounds.draw(ax=ax2, color='green', alpha=.4)
        leaf_crop_box.draw(ax=ax2, color='purple')
        root_crop_corners.draw(ax=ax1, color='purple', alpha=.4)

        new_w = region_box.to_xywh().data[0, 2]
        new_h = region_box.to_xywh().data[0, 3]
        ax3.set_xlim(0, new_w)
        ax3.set_ylim(0, new_h)

        crop_w = leaf_crop_box.to_xywh().data[0, 2]
        crop_h = leaf_crop_box.to_xywh().data[0, 3]
        ax4.set_xlim(0, crop_w)
        ax4.set_ylim(0, crop_h)

        pts3_ = kwimage.Points.random(3).scale((new_w, new_h))
        pts3 = kwimage.Points(xy=np.vstack([[[0, 0], [5, 5], [0, 49], [40, 45]], pts3_.xy]))
        pts4 = pts3.warp(tf_newroot_to_crop.matrix)
        pts3.draw(ax=ax3)
        pts4.draw(ax=ax4)

    # delayed_crop = band2.delayed_crop(region_slices)
    # final_crop = delayed_crop.finalize()

if __name__ == '__main__':
    import xdoctest
    xdoctest.doctest_module(__file__)
