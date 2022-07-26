"""
Terminal nodes
"""

import kwarray
import kwimage
import numpy as np
import warnings
from kwcoco.util.delayed_ops.delayed_nodes import DelayedImage2
# from kwcoco.util.delayed_ops.delayed_nodes import DelayedArray2

try:
    from xdev import profile
except ImportError:
    from ubelt import identity as profile


class DelayedImageLeaf2(DelayedImage2):

    def get_transform_from_leaf(self):
        """
        Returns the transformation that would align data with the leaf

        Returns:
            kwimage.Affine
        """
        return kwimage.Affine.eye()

    @profile
    def optimize(self):
        return self


class DelayedLoad2(DelayedImageLeaf2):
    """
    Reads an image from disk.

    If a gdal backend is available, and the underlying image is in the
    appropriate formate (e.g. COG) this will return a lazy reference that
    enables fast overviews and crops.

    Example:
        >>> from kwcoco.util.delayed_ops import *  # NOQA
        >>> self = DelayedLoad2.demo(dsize=(16, 16)).prepare()
        >>> data1 = self.finalize()

    Example:
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> # Demo code to develop support for overviews
        >>> from kwcoco.util.delayed_ops import *  # NOQA
        >>> import kwimage
        >>> import ubelt as ub
        >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
        >>> self = DelayedLoad2(fpath, channels='r|g|b').prepare()
        >>> print(f'self={self}')
        >>> print('self.meta = {}'.format(ub.repr2(self.meta, nl=1)))
        >>> quantization = {
        >>>     'quant_max': 255,
        >>>     'nodata': 0,
        >>> }
        >>> node0 = self
        >>> node1 = node0.get_overview(2)
        >>> node2 = node1[13:900, 11:700]
        >>> node3 = node2.dequantize(quantization)
        >>> node4 = node3.warp({'scale': 0.05})
        >>> #
        >>> data0 = node0._validate().finalize()
        >>> data1 = node1._validate().finalize()
        >>> data2 = node2._validate().finalize()
        >>> data3 = node3._validate().finalize()
        >>> data4 = node4._validate().finalize()
        >>> node4.write_network_text()

    Example:
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> # Test delayed ops with int16 and nodata values
        >>> from kwcoco.util.delayed_ops import *  # NOQA
        >>> import kwimage
        >>> from kwcoco.util.delayed_ops.helpers import quantize_float01
        >>> import ubelt as ub
        >>> dpath = ub.Path.appdir('kwcoco/tests/test_delay_nodata').ensuredir()
        >>> fpath = dpath / 'data.tif'
        >>> data = kwimage.ensure_float01(kwimage.grab_test_image())
        >>> poly = kwimage.Polygon.random(rng=321032).scale(data.shape[0])
        >>> poly.fill(data, np.nan)
        >>> data_uint16, quantization = quantize_float01(data)
        >>> nodata = quantization['nodata']
        >>> kwimage.imwrite(fpath, data_uint16, nodata=nodata, backend='gdal', overviews=3)
        >>> # Test loading the data
        >>> self = DelayedLoad2(fpath, channels='r|g|b', nodata_method='float').prepare()
        >>> node0 = self
        >>> node1 = node0.dequantize(quantization)
        >>> node2 = node1.warp({'scale': 0.51}, interpolation='lanczos')
        >>> node3 = node2[13:900, 11:700]
        >>> node4 = node3.warp({'scale': 0.9}, interpolation='lanczos')
        >>> node4.write_network_text()
        >>> node5 = node4.optimize()
        >>> node5.write_network_text()
        >>> node6 = node5.warp({'scale': 8}, interpolation='lanczos').optimize()
        >>> node6.write_network_text()
        >>> #
        >>> data0 = node0._validate().finalize()
        >>> data1 = node1._validate().finalize()
        >>> data2 = node2._validate().finalize()
        >>> data3 = node3._validate().finalize()
        >>> data4 = node4._validate().finalize()
        >>> data5 = node5._validate().finalize()
        >>> data6 = node6._validate().finalize()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> stack1 = kwimage.stack_images([data1, data2, data3, data4, data5])
        >>> stack2 = kwimage.stack_images([stack1, data6], axis=1)
        >>> kwplot.imshow(stack2)
    """
    def __init__(self, fpath, channels=None, dsize=None, nodata_method=None):
        """
        Args:
            fpath (str | PathLike):
                URI pointing at the image data to load

            channels (int | str | kwcoco.FusedChannelSpec | None):
                the underlying channels of the image if known a-priori

            dsize (Tuple[int, int]):
                The width / height of the image if known a-priori

            nodata_method (str | None):
                How to handle nodata values in the file itself.
                Can be "auto", "float", or "ma".

        """
        super().__init__(channels=channels, dsize=dsize)
        self.meta['fpath'] = fpath
        self.meta['nodata_method'] = nodata_method
        self.lazy_ref = None

    @property
    def fpath(self):
        return self.meta['fpath']

    @classmethod
    def demo(DelayedLoad2, key='astro', dsize=None, channels=None):
        fpath = kwimage.grab_test_image_fpath(key, dsize=dsize)
        self = DelayedLoad2(fpath, channels=channels)
        return self

    def _load_reference(self):
        nodata_method = self.meta.get('nodata_method', None)
        if self.lazy_ref is None:
            from kwcoco.util import lazy_frame_backends
            using_gdal = lazy_frame_backends.LazyGDalFrameFile.available()
            if using_gdal:
                # the nodata arg here isn't named that great
                self.lazy_ref = lazy_frame_backends.LazyGDalFrameFile(
                    self.fpath, nodata_method=nodata_method)
            else:
                if nodata_method == 'auto':
                    raise Exception('need gdal for auto no-data')
                self.lazy_ref = NotImplemented
        return self

    def prepare(self):
        """
        If metadata is missing, perform minimal IO operations in order to
        prepopulate metadata that could help us better optimize the operation
        tree.

        Returns:
            DelayedLoad2
        """
        self._load_metadata()
        return self

    def _load_metadata(self):
        self._load_reference()
        if self.lazy_ref is NotImplemented:
            shape = kwimage.load_image_shape(self.fpath)
            if len(shape) == 2:
                shape = shape + (1,)
            num_overviews = 0
        else:
            shape = self.lazy_ref.shape
            num_overviews = self.lazy_ref.num_overviews
        h, w, c = shape
        if self.dsize is None:
            self.meta['dsize'] = (w, h)
        if self.num_channels is None:
            self.meta['num_channels'] = c
        self.meta['num_overviews'] = num_overviews
        return self

    def _finalize(self):
        """
        Returns:
            ArrayLike

        Example:
            >>> # Check difference between finalize and _finalize
            >>> from kwcoco.util.delayed_ops.delayed_leafs import *  # NOQA
            >>> self = DelayedLoad2.demo().prepare()
            >>> final_arr = self.finalize()
            >>> assert isinstance(final_arr, np.ndarray), 'finalize should always return an array'
            >>> final_ref = self._finalize()
            >>> if self.lazy_ref is not NotImplemented:
            >>>     assert not isinstance(final_ref, np.ndarray), (
            >>>         'A pure load with gdal should return a reference that is '
            >>>         'similiar to but not quite an array')
        """
        self._load_reference()
        if self.lazy_ref is NotImplemented:
            warnings.warn('DelayedLoad2 may not be efficient without gdal')
            pre_final = kwimage.imread(self.fpath)
            pre_final = kwarray.atleast_nd(pre_final, 3)
            return pre_final
        else:
            return self.lazy_ref


class DelayedNans2(DelayedImageLeaf2):
    """
    Constructs nan channels as needed

    Example:
        self = DelayedNans((10, 10), channel_spec.FusedChannelSpec.coerce('rgb'))
        region_slices = (slice(5, 10), slice(1, 12))
        delayed = self.crop(region_slices)

    Example:
        >>> from kwcoco.util.delayed_ops import *  # NOQA
        >>> import kwcoco
        >>> dsize = (307, 311)
        >>> c1 = DelayedNans2(dsize=dsize, channels='foo')
        >>> c2 = DelayedLoad2.demo('astro', dsize=dsize, channels='R|G|B').prepare()
        >>> cat = DelayedChannelConcat2([c1, c2])
        >>> warped_cat = cat.warp({'scale': 1.07}, dsize=(328, 332))._validate()
        >>> warped_cat._validate().optimize().finalize()
    """
    def __init__(self, dsize=None, channels=None):
        super().__init__(channels=channels, dsize=dsize)

    def _finalize(self):
        """
        Returns:
            ArrayLike
        """
        shape = self.shape
        final = np.full(shape, fill_value=np.nan)
        return final

    def _optimized_crop(self, space_slice=None, chan_idxs=None):
        """
        Crops an image along integer pixel coordinates.

        Args:
            space_slice (Tuple[slice, slice]): y-slice and x-slice.
            chan_idxs (List[int]): indexes of bands to take

        Returns:
            DelayedImage2
        """
        if chan_idxs is None:
            channels = self.channels
        else:
            channels = self.channels[chan_idxs]
        dsize = self.dsize
        data_dims = dsize[::-1]
        data_slice, extra_pad = kwarray.embed_slice(space_slice, data_dims)
        box = kwimage.Boxes.from_slice(data_slice)
        new_width = box.width.ravel()[0]
        new_height = box.height.ravel()[0]
        new_dsize = (new_width, new_height)
        new = self.__class__(new_dsize, channels=channels)
        return new

    def _optimized_warp(self, transform, dsize=None, antialias=True, interpolation='linear', border_value='auto'):
        """
        Returns:
            DelayedImage2
        """
        # Warping does nothing to nans, except maybe changing the dsize
        new = self.__class__(dsize, channels=self.channels)
        return new


class DelayedIdentity2(DelayedImageLeaf2):
    """
    Returns an ndarray as-is

    Example:
        self = DelayedNans((10, 10), channel_spec.FusedChannelSpec.coerce('rgb'))
        region_slices = (slice(5, 10), slice(1, 12))
        delayed = self.crop(region_slices)

    Example:
        >>> from kwcoco.util.delayed_ops import *  # NOQA
        >>> import kwcoco
        >>> arr = kwimage.checkerboard()
        >>> self = DelayedIdentity2(arr, channels='gray')
        >>> warp = self.warp({'scale': 1.07})
        >>> warp.optimize().finalize()
    """
    def __init__(self, data, channels=None, dsize=None):
        super().__init__(channels=channels)
        self.data = data
        self.meta['num_channels'] = kwimage.num_channels(data)
        if dsize is None:
            dsize = data.shape[0:2][::-1]
        self.meta['dsize'] = dsize

    def _finalize(self):
        """
        Returns:
            ArrayLike
        """
        return self.data
