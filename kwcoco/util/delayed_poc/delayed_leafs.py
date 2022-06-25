"""
Leaf nodes for delayed operations
"""
import ubelt as ub
import numpy as np
import kwimage
import kwarray
from kwcoco import channel_spec
from kwcoco.util.delayed_poc.delayed_base import DelayedImage

try:
    import xarray as xr
except ImportError:
    xr = None

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


class DelayedNans(DelayedImage):
    """
    Constructs nan channels as needed

    Example:
        self = DelayedNans((10, 10), channel_spec.FusedChannelSpec.coerce('rgb'))
        region_slices = (slice(5, 10), slice(1, 12))
        delayed = self.crop(region_slices)

    Example:
        >>> from kwcoco.util.delayed_poc.delayed_leafs import *  # NOQA
        >>> from kwcoco.util.delayed_poc.delayed_nodes import *  # NOQA
        >>> import kwcoco
        >>> dsize = (307, 311)
        >>> c1 = DelayedNans(dsize=dsize, channels=kwcoco.FusedChannelSpec.coerce('foo'))
        >>> c2 = DelayedLoad.demo('astro', dsize=dsize).load_shape(True)
        >>> cat = DelayedChannelStack([c1, c2])
        >>> warped_cat = cat.warp(kwimage.Affine.scale(1.07), dsize=(328, 332))
        >>> warped_cat.finalize()

        #>>> cropped = warped_cat.crop((slice(0, 300), slice(0, 100)))
        #>>> cropped.finalize().shape
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
        """
        Yields:
            Any
        """
        yield from []

    def _optimize_paths(self, **kwargs):
        # DEBUG_PRINT('DelayedLoad._optimize_paths')
        # hack
        # if 'dsize' in kwargs:
        #     dsize = tuple(kwargs['dsize'])
        # else:
        from kwcoco.util.delayed_poc.delayed_nodes import DelayedWarp
        dsize = self.dsize
        yield DelayedWarp(self, kwimage.Affine(None), dsize=dsize)

    @profile
    def finalize(self, **kwargs):
        if 'dsize' in kwargs:
            shape = tuple(kwargs['dsize'])[::-1] + (self.num_bands,)
        else:
            shape = self.shape
        final = np.full(shape, fill_value=np.nan)

        as_xarray = kwargs.get('as_xarray', False)
        if as_xarray:
            channels = self.channels
            coords = {}
            if channels is not None:
                coords['c'] = channels.code_list()
            final = xr.DataArray(final, dims=('y', 'x', 'c'), coords=coords)
        return final

    def crop(self, region_slices):
        # DEBUG_PRINT('DelayedNans.crop')
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

    def warp(self, transform, dsize=None):
        # Warping does nothing to nans, except maybe changing the dsize
        new = self.__class__(dsize, channels=self.channels)
        return new


class DelayedLoad(DelayedImage):
    """
    A load operation for a specific sub-region and sub-bands in a specified
    image.

    Note:
        This class contains support for fusing certain lazy operations into
        this layer, namely cropping, scaling, and channel selection.

        For now these are named ``immediates``

    Example:
        >>> from kwcoco.util.delayed_poc.delayed_leafs import DelayedLoad
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

    Example:
        >>> # Test with quantization
        >>> from kwcoco.util.delayed_poc.delayed_leafs import DelayedLoad
        >>> fpath = kwimage.grab_test_image_fpath()
        >>> channels = channel_spec.FusedChannelSpec.coerce('rgb')
        >>> self = DelayedLoad(fpath, channels=channels, quantization={
        >>>     'orig_min': 0.,
        >>>     'orig_max': 1.,
        >>>     'quant_min': 0,
        >>>     'quant_max': 256,
        >>>     'nodata': None,
        >>> })
        >>> final1 = self.finalize(dequantize=False)
        >>> final2 = self.finalize(dequantize=True)
        >>> assert final1.dtype.kind == 'u'
        >>> assert final2.dtype.kind == 'f'
        >>> assert final2.max() <= 1
    """
    __hack_dont_optimize__ = True

    def __init__(self, fpath, channels=None,
                 # Extra params to allow certain operations as close to the
                 # disk as possible.
                 dsize=None,
                 num_bands=None,
                 immediate_crop=None,
                 immediate_chan_idxs=None,
                 immediate_dsize=None,
                 quantization=None,
                 num_overviews=None):

        # self.data = {}
        self.meta = {}
        self.cache = {}

        if immediate_dsize is not None:
            dsize = immediate_dsize

        self.meta['fpath'] = fpath
        self.meta['dsize'] = dsize
        self.meta['channels'] = channels
        self.meta['num_overviews'] = num_overviews

        self._immediates = {
            'crop': immediate_crop,
            'dsize': immediate_dsize,
            'chan_idxs': immediate_chan_idxs,
        }
        self.quantization = quantization

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
        """
        Yields:
            Any
        """
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
        # if 'dsize' in kwargs:
        #     dsize = tuple(kwargs['dsize'])
        # else:
        dsize = self.dsize
        from kwcoco.util.delayed_poc.delayed_nodes import DelayedWarp
        yield DelayedWarp(self, kwimage.Affine(None), dsize=dsize)
        # raise AssertionError('hack so this is not called')

    def load_shape(self, use_channel_heuristic=False):
        disk_shape = kwimage.load_image_shape(self.fpath)
        if self.meta.get('num_bands', None) is None:
            num_bands = disk_shape[2] if len(disk_shape) == 3 else 1
            self.meta['num_bands'] = num_bands
        if self.meta.get('dsize', None) is None:
            h, w = disk_shape[0:2]
            self.meta['dsize'] = (w, h)

        # This is not robust. This should be removed.
        # if self.meta.get('channels', None) is None:
        #     if self.meta['num_bands'] == 3:
        #         self.meta['channels'] = channel_spec.FusedChannelSpec.coerce('r|g|b')

        self.meta['_raw_shape'] = disk_shape
        return self

    def _ensure_dsize(self):
        dsize = self.dsize
        if dsize is None:
            self.load_shape()
            dsize = self.dsize
        return dsize

    def _ensure_num_overviews(self):
        # Try and find the number of overviews
        num_overviews = self.num_overviews
        if num_overviews is None:
            # Hack that requires a separate load of the data. Can we reuse some
            # cache here?
            from kwcoco.util import lazy_frame_backends
            using_gdal = lazy_frame_backends.LazyGDalFrameFile.available()
            if using_gdal:
                pre_final = lazy_frame_backends.LazyGDalFrameFile(self.fpath)
                num_overviews = pre_final.num_overviews
            else:
                num_overviews = 0
            self.meta['num_overviews'] = num_overviews
        return self.num_overviews

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

    @property
    def num_overviews(self):
        return self.meta.get('num_overviews', None)

    @profile
    def finalize(self, **kwargs):
        """
        TODO:
            - [ ] Load from overviews if a scale will be necessary

        Args:
            **kwargs:
                nodata : if specified this data item is treated as nodata, the
                    data is then converted to floats and the nodata value is
                    replaced with nan.
        """
        nodata = kwargs.get('nodata', None)
        overview = kwargs.get('overview', None)

        # Probably should not use a cache here?
        # final = self.cache.get('final', None)
        final = None
        if final is None:
            from kwcoco.util import lazy_frame_backends
            using_gdal = lazy_frame_backends.LazyGDalFrameFile.available()
            if lazy_frame_backends.LazyGDalFrameFile.available():
                # TODO: warn if we dont have a COG.
                pre_final = lazy_frame_backends.LazyGDalFrameFile(self.fpath,
                                                                  nodata=nodata,
                                                                  overview=overview)
                pre_final._ds
                # pre_final = LazyGDalFrameFile(self.fpath)
                # TODO: choose the fastest lazy backend for the file
                # pre_final = lazy_frame_backends.LazyRasterIOFrameFile(self.fpath)  # which is faster?
                # pre_final = lazy_frame_backends.LazySpectralFrameFile(self.fpath)  # which is faster?
            else:
                if nodata == 'auto':
                    raise Exception('need gdal for auto no-data')
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

            # Handle nan
            if not using_gdal:
                if nodata is not None and isinstance(nodata, int):
                    if final.dtype.kind != 'f':
                        final = final.astype(np.float32)
                    final[final == nodata] = np.nan

            dequantize_ = kwargs.get('dequantize', True)
            if self.quantization is not None and dequantize_:
                # Note: this is very inefficient on crop
                final = dequantize(final, self.quantization)

            dsize = self._immediates.get('dsize', None)
            if dsize is not None:
                final = kwimage.imresize(final, dsize=dsize, antialias=True)
            # self.cache['final'] = final

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
    def crop(self, region_slices):
        """
        Args:
            region_slices (Tuple[slice, slice]): y-slice and x-slice.

        Returns:
            DelayedLoad : a new delayed load object with a fused crop operation

        Example:
            >>> # Test chained crop operations
            >>> from kwcoco.util.delayed_poc.delayed_leafs import *  # NOQA
            >>> from kwcoco.util.delayed_poc.delayed_nodes import *  # NOQA
            >>> self = orig = DelayedLoad.demo('astro').load_shape()
            >>> region_slices = slices1 = (slice(0, 90), slice(30, 60))
            >>> self = crop1 = orig.crop(slices1)
            >>> region_slices = slices2 = (slice(10, 21), slice(10, 22))
            >>> self = crop2 = crop1.crop(slices2)
            >>> region_slices = slices3 = (slice(3, 20), slice(5, 20))
            >>> crop3 = crop2.crop(slices3)
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
        # DEBUG_PRINT('DelayedLoad.crop')
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

        new_crop = (slice(new_ystart, new_ystop), slice(new_xstart, new_xstop))
        new_dsize = (new_xstop - new_xstart, new_ystop - new_ystart)

        # TODO: it might be ok to remove this line
        assert self._immediates['dsize'] is None, 'does not handle'

        new = self.__class__(
            fpath=self.meta['fpath'],
            num_bands=self.meta['num_bands'],
            channels=self.meta['channels'],
            num_overviews=self.meta['num_overviews'],
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
            >>> from kwcoco.util.delayed_poc.delayed_leafs import *  # NOQA
            >>> from kwcoco.util.delayed_poc.delayed_nodes import *  # NOQA
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
            num_overviews=self.meta['num_overviews'],
            immediate_dsize=self._immediates['dsize'],
            immediate_crop=self._immediates['crop'],
            immediate_chan_idxs=new_chan_ixs,
            quantization=self.quantization,
        )
        return new


def dequantize(quant_data, quantization):
    """
    Helper for dequantization

    Args:
        quant_data (ndarray):
            data to dequantize

        quantization (Dict[str, Any]):
            quantization information dictionary to undo.
            Expected keys are:
            orig_type (str)
            orig_min (float)
            orig_max (float)
            quant_min (float)
            quant_max (float)
            nodata (None | int)

    Returns:
        ndarray : dequantized data

    Example:
        >>> quant_data = (np.random.rand(4, 4) * 256).astype(np.uint8)
        >>> quantization = {
        >>>     'orig_dtype': 'float32',
        >>>     'orig_min': 0,
        >>>     'orig_max': 1,
        >>>     'quant_min': 0,
        >>>     'quant_max': 255,
        >>>     'nodata': None,
        >>> }
        >>> dequantize(quant_data, quantization)

    Example:
        >>> quant_data = np.ones((4, 4), dtype=np.uint8)
        >>> quantization = {
        >>>     'orig_dtype': 'float32',
        >>>     'orig_min': 0,
        >>>     'orig_max': 1,
        >>>     'quant_min': 1,
        >>>     'quant_max': 1,
        >>>     'nodata': None,
        >>> }
        >>> dequantize(quant_data, quantization)
    """
    orig_dtype = quantization.get('orig_dtype', 'float32')
    orig_min = quantization.get('orig_min', 0)
    orig_max = quantization.get('orig_max', 1)
    quant_min = quantization.get('quant_min', 0)
    quant_max = quantization['quant_max']
    nodata = quantization.get('nodata', None)
    orig_extent = orig_max - orig_min
    quant_extent = quant_max - quant_min
    if quant_extent == 0:
        scale = 0
    else:
        scale = (orig_extent / quant_extent)
    dequant = quant_data.astype(orig_dtype)
    dequant = (dequant - quant_min) * scale + orig_min
    if nodata is not None:
        mask = quant_data == nodata
        dequant[mask] = np.nan
    return dequant


class DelayedIdentity(DelayedImage):
    """
    Noop leaf that does nothing. Can be used to hold raw data.

    Typically used to just hold raw data.

    DelayedIdentity.demo('astro', chan=0, dsize=(32, 32))

    Example:
        >>> from kwcoco.util.util_delayed_poc import *  # NOQA
        >>> sub_data = np.random.rand(31, 37, 3)
        >>> self = DelayedIdentity(sub_data)
        >>> self = DelayedIdentity(sub_data, channels='L|a|b')

        >>> # test with quantization
        >>> rng = kwarray.ensure_rng(32)
        >>> sub_data_quant = (rng.rand(31, 37, 3) * 1000).astype(np.int16)
        >>> sub_data_quant[0, 0] = -9999
        >>> self = DelayedIdentity(sub_data_quant, channels='L|a|b', quantization={
        >>>     'orig_min': 0.,
        >>>     'orig_max': 1.,
        >>>     'quant_min': 0,
        >>>     'quant_max': 1000,
        >>>     'nodata': -9999,
        >>> })
        >>> final1 = self.finalize(dequantize=True)
        >>> final2 = self.finalize(dequantize=False)
        >>> assert np.all(np.isnan(final1[0, 0]))
        >>> scale = final2 / final1
        >>> scales = scale[scale > 0]
        >>> assert np.all(np.isclose(scales, 1000))
        >>> # check that take channels works
        >>> new_subdata = self.take_channels('a')
        >>> sub_final1 = new_subdata.finalize(dequantize=True)
        >>> sub_final2 = new_subdata.finalize(dequantize=False)
        >>> assert sub_final1.dtype.kind == 'f'
        >>> assert sub_final2.dtype.kind == 'i'
    """
    __hack_dont_optimize__ = True

    def __init__(self, sub_data, dsize=None, channels=None, quantization=None):
        self.sub_data = sub_data
        self.meta = {}
        self.cache = {}
        h, w = self.sub_data.shape[0:2]
        if dsize is None:
            dsize = (w, h)
        self.dsize = dsize
        self.quantization = quantization
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
        self.meta['quantization'] = self.quantization
        if channels is None:
            # self.channels = channel_spec.FusedChannelSpec.coerce(num_bands)
            self.channels = None
        else:
            self.channels = channel_spec.FusedChannelSpec.coerce(channels)

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
        """
        Yields:
            Any
        """
        yield from []

    # def crop(self, region_slices):
    #     return DelayedCrop(self, region_slices)

    def _optimize_paths(self, **kwargs):
        # DEBUG_PRINT('DelayedIdentity._optimize_paths')
        # Hack
        from kwcoco.util.delayed_poc.delayed_nodes import DelayedWarp
        yield DelayedWarp(self, kwimage.Affine(None), dsize=self.dsize)

    @profile
    def finalize(self, **kwargs):
        dequantize_ = kwargs.get('dequantize', True)
        final = self.sub_data
        final = kwarray.atleast_nd(final, 3, front=False)
        if self.quantization is not None and dequantize_:
            # Note: this is very inefficient on crop
            final = dequantize(final, self.quantization)
        return final

    def take_channels(self, channels):
        """
        Returns:
            DelayedIdentity
        """
        if not isinstance(self.sub_data, np.ndarray):
            return super().take_channels(channels)

        # Perform operation immediately
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

        new_chan_ixs = top_idx_mapping
        channels = self.channels
        if channels is not None:
            new_chan_parsed = list(ub.take(channels.parsed, top_idx_mapping))
            channels = channel_spec.FusedChannelSpec(new_chan_parsed)

        new_data = self.sub_data[..., new_chan_ixs]

        new = self.__class__(
            sub_data=new_data,
            dsize=self.dsize,
            channels=channels,
            quantization=self.quantization,
        )
        return new
