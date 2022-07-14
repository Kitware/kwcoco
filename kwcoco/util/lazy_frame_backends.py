"""
Ducktyped interfaces for loading subregions of images with standard slice
syntax
"""
import os
import ubelt as ub
import numpy as np
import kwimage
from os.path import join, exists

from collections import OrderedDict

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


class CacheDict(OrderedDict):
    """
    Dict with a limited length, ejecting LRUs as needed.

    Example:
        >>> c = CacheDict(cache_len=2)
        >>> c[1] = 1
        >>> c[2] = 2
        >>> c[3] = 3
        >>> c
        CacheDict([(2, 2), (3, 3)])
        >>> c[2]
        2
        >>> c[4] = 4
        >>> c
        CacheDict([(2, 2), (4, 4)])
        >>>

    References:
        https://gist.github.com/davesteele/44793cd0348f59f8fadd49d7799bd306
    """

    def __init__(self, *args, cache_len: int = 10, **kwargs):
        assert cache_len > 0
        self.cache_len = cache_len
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().move_to_end(key)
        while len(self) > self.cache_len:
            oldkey = next(iter(self))
            super().__delitem__(oldkey)

    def __getitem__(self, key):
        val = super().__getitem__(key)
        super().move_to_end(key)
        return val


# Only can use this cache if we assume we are in readonly mode
# GLOBAL_GDAL_CACHE = CacheDict(cache_len=32)
GLOBAL_GDAL_CACHE = None


@ub.memoize
def _have_gdal():
    try:
        from osgeo import gdal
    except ImportError:
        return False
    else:
        return gdal is not None


@ub.memoize
def _have_rasterio():
    try:
        import rasterio
    except ImportError:
        return False
    else:
        return rasterio is not None


@ub.memoize
def _have_spectral():
    try:
        import spectral
    except ImportError:
        return False
    else:
        return spectral is not None


_GDAL_DTYPE_LUT = {
    1: np.uint8,     2: np.uint16,
    3: np.int16,     4: np.uint32,      5: np.int32,
    6: np.float32,   7: np.float64,     8: np.complex_,
    9: np.complex_,  10: np.complex64,  11: np.complex128
}


class LazySpectralFrameFile(ub.NiceRepr):
    """
    Potentially faster than GDAL for HDR formats.
    """
    def __init__(self, fpath):
        self.fpath = fpath

    @ub.memoize_property
    def _ds(self):
        import spectral
        from os.path import exists
        if not exists(self.fpath):
            raise Exception('File does not exist: {}'.format(self.fpath))
        ds = spectral.envi.open(os.fspath(self.fpath))
        return ds

    @classmethod
    def available(self):
        """
        Returns True if this backend is available
        """
        return _have_spectral()

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def shape(self):
        return self._ds.shape

    @property
    def dtype(self):
        return self._ds.dtype

    def __nice__(self):
        from os.path import basename
        return '.../' + basename(self.fpath)

    @profile
    def __getitem__(self, index):
        ds = self._ds

        height, width, C = ds.shape
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

        img_part = ds.read_subregion(
            row_bounds=(ystart, ystop), col_bounds=(xstart, xstop),
            bands=band_indices)
        return img_part


class LazyRasterIOFrameFile(ub.NiceRepr):
    """

    fpath = '/home/joncrall/.cache/kwcoco/demo/large_hyperspectral/big_img_128.bsq'
    lazy_rio = LazyRasterIOFrameFile(fpath)
    ds = lazy_rio._ds

    Ignore:
        # Can rasterio read multiple bands at once?
        # Seems like that is an overhead for hyperspectral images

        import rasterio
        riods = rasterio.open(fpath)
        import timerit

        ti = timerit.Timerit(1, bestof=1, verbose=2)
        b = tuple(range(1, riods.count + 1))
        for timer in ti.reset('rasterio'):
            with timer:
                riods.read(b)

        lazy_rio = LazyRasterIOFrameFile(fpath)
        for timer in ti.reset('LazyRasterIOFrameFile'):
            with timer:
                lazy_rio[:]

        lazy_gdal = LazyGDalFrameFile(fpath)
        for timer in ti.reset('LazyGDalFrameFile'):
            with timer:
                lazy_gdal[:]

    """
    def __init__(self, fpath):
        self.fpath = fpath

    @classmethod
    def available(self):
        """
        Returns True if this backend is available
        """
        return _have_rasterio()

    @ub.memoize_property
    def _ds(self):
        import rasterio
        from os.path import exists
        if not exists(self.fpath):
            raise Exception('File does not exist: {}'.format(self.fpath))
        ds = rasterio.open(self.fpath, mode='r')
        return ds

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def shape(self):
        ds = self._ds
        return (ds.height, ds.width, ds.count)

    @property
    def dtype(self):
        # Assume the first is the same as the rest
        ds = self._ds
        dtype = getattr(np, ds.dtypes[0])
        return dtype

    def __nice__(self):
        from os.path import basename
        return '.../' + basename(self.fpath)

    @profile
    def __getitem__(self, index):
        ds = self._ds
        width = ds.width
        height = ds.height
        C = ds.count

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

        indexes = [b + 1 for b in band_indices]
        img_part = ds.read(indexes=indexes, window=((ystart, ystop), (xstart, xstop)))
        img_part = img_part.transpose(1, 2, 0)
        return img_part


def _demo_geoimg_with_nodata():
    """
    Example:
        from kwcoco.util.lazy_frame_backends import *  # NOQA
        fpath = _demo_geoimg_with_nodata()
        self = LazyGDalFrameFile.demo()

    """
    import kwimage
    from osgeo import osr
    # gdal.UseExceptions()

    # Make a dummy geotiff
    imdata = kwimage.grab_test_image('airport')
    dpath = ub.Path.appdir('kwcoco/test/geotiff').ensuredir()
    geo_fpath = dpath / 'dummy_geotiff.tif'

    # compute dummy values for a geotransform to CRS84
    img_h, img_w = imdata.shape[0:2]
    img_box = kwimage.Boxes([[0, 0, img_w, img_h]], 'xywh')
    wld_box = kwimage.Boxes([[-73.7595528, 42.6552404, 0.0001, 0.0001]], 'xywh')
    img_corners = img_box.corners()
    wld_corners = wld_box.corners()
    transform = kwimage.Affine.fit(img_corners, wld_corners)

    nodata = -9999

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    crs = srs.ExportToWkt()

    # Set a region to be nodata
    imdata = imdata.astype(np.int16)
    imdata[-100:] = nodata
    imdata[0:200:, -200:-180] = nodata

    kwimage.imwrite(geo_fpath, imdata, backend='gdal', nodata=-9999, crs=crs, transform=transform)
    return geo_fpath


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

    Args:
        fpath (str): the path to the file to load
        nodata_method (None | int | str): how to handle nodata
        overview (int): The overview level to load (zero is no overview)

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
    def __init__(self, fpath, nodata_method=None, overview=None):
        self.fpath = fpath
        self.nodata_method = nodata_method
        self.overview = overview
        if self.overview is None:
            self.overview = 0
        self._ds_cache = None

    @classmethod
    def available(self):
        """
        Returns True if this backend is available
        """
        return _have_gdal()

    @profile
    def _reload_cache(self):
        from osgeo import gdal
        _fpath = os.fspath(self.fpath)
        if _fpath.endswith('.hdr'):
            # Use spectral-like process to point gdal to the correct file given
            # the hdr
            ext = '.' + _read_envi_header(_fpath)['interleave']
            _fpath = ub.augpath(_fpath, ext=ext)

        if GLOBAL_GDAL_CACHE is not None and _fpath in GLOBAL_GDAL_CACHE:
            self._ds_cache = GLOBAL_GDAL_CACHE[_fpath]
        else:
            ds = gdal.Open(_fpath, gdal.GA_ReadOnly)
            if ds is None:
                if not exists(self.fpath):
                    raise Exception('File does not exist: {}'.format(self.fpath))
                raise Exception((
                    'GDAL Failed to open the fpath={!r} for an unknown reason. '
                    'Call gdal.UseExceptions() beforehand to get the '
                    'real exception').format(self.fpath))
            self._ds_cache = ds
            if GLOBAL_GDAL_CACHE is not None:
                GLOBAL_GDAL_CACHE[_fpath] = ds

    def get_overview(self, overview):
        """
        Returns the overview relative to this one.
        """
        return self.get_absolute_overview(overview + self.overview)

    def get_absolute_overview(self, overview):
        """
        Returns the overview relative to the base
        """
        new = self.__class__(self.fpath, nodata_method=self.nodata_method,
                             overview=overview)
        new._ds_cache = self._ds_cache
        return new

    @property
    def _ds(self):
        if self._ds_cache is None:
            self._reload_cache()
        ds = self._ds_cache
        return ds

    @classmethod
    def demo(cls, key='astro', dsize=None):
        """
        Ignore:
            >>> self = LazyGDalFrameFile.demo(dsize=(6600, 4400))
        """
        cache_dpath = ub.ensure_app_cache_dir('kwcoco/demo')
        fpath = join(cache_dpath, key + '.cog.tiff')
        depends = ub.odict(dsize=dsize)
        stamp = ub.CacheStamp(fname=key, depends=depends, dpath=cache_dpath,
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

    @ub.memoize_property
    def num_overviews(self):
        return self.num_absolute_overviews - self.overview

    @ub.memoize_property
    def num_absolute_overviews(self):
        ds = self._ds
        default_band0 = ds.GetRasterBand(1)
        num_overviews = default_band0.GetOverviewCount()
        return num_overviews

    @ub.memoize_property
    def shape(self):
        ds = self._ds
        default_band0 = ds.GetRasterBand(1)

        if self.overview:
            num_overviews = default_band0.GetOverviewCount()
            self.load_overview = min(self.overview, num_overviews)
            if self.load_overview:
                self.post_overview = self.overview - self.load_overview
                if self.post_overview != 0:
                    raise ValueError('unhandled: overview does not exist')
                # Overviews are zero indexed in gdal, inconsistent, I know
                band0 = default_band0.GetOverview(self.load_overview - 1)
            else:
                band0 = default_band0
        else:
            band0 = default_band0
        self.num_channels = ds.RasterCount
        self.width = band0.XSize
        self.height = band0.YSize
        width = self.width
        height = self.height
        C = self.num_channels
        return (height, width, C)

    @property
    def dtype(self):
        main_band = self._ds.GetRasterBand(1)
        dtype = _GDAL_DTYPE_LUT[main_band.DataType]
        return dtype

    def __nice__(self):
        from os.path import basename
        return '.../' + basename(self.fpath)

    @profile
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

            >>> self = LazyGDalFrameFile.demo(dsize=(6600, 4400))
            >>> self.nodata_method = 0
            >>> index = [slice(2100, 2508, None), slice(4916, 5324, None), None]
            >>> img_part = self[index]
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(img_part / 255)

        Example:
            >>> # Test nodata works correctly
            >>> # xdoctest: +REQUIRES(module:osgeo)
            >>> from kwcoco.util.lazy_frame_backends import *  # NOQA
            >>> from kwcoco.util.lazy_frame_backends import _demo_geoimg_with_nodata
            >>> fpath = _demo_geoimg_with_nodata()
            >>> self = LazyGDalFrameFile(fpath, nodata_method='auto')
            >>> imdata = self[:]
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> import kwarray
            >>> kwplot.autompl()
            >>> imdata = kwimage.normalize_intensity(imdata)
            >>> imdata = np.nan_to_num(imdata)
            >>> kwplot.imshow(imdata)

        Example:
            >>> # xdoctest: +REQUIRES(--ipfs)
            >>> # Demo code to develop support for overviews
            >>> import kwimage
            >>> from kwcoco.util.lazy_frame_backends import *  # NOQA
            >>> fpath = ub.grabdata('https://ipfs.io/ipfs/QmaFcb565HM9FV8f41jrfCZcu1CXsZZMXEosjmbgeBhFQr', fname='PXL_20210411_150641385.jpg')
            >>> self = LazyGDalFrameFile(fpath, overview=2)
            >>> print(LazyGDalFrameFile(fpath, overview=0).shape)
            >>> print(LazyGDalFrameFile(fpath, overview=1).shape)
            >>> print(LazyGDalFrameFile(fpath, overview=2).shape)
            >>> print(LazyGDalFrameFile(fpath, overview=3).shape)
            >>> # print(LazyGDalFrameFile(fpath, overview=4).shape)
            >>> print(LazyGDalFrameFile(fpath, overview=0)[:].shape)
            >>> print(LazyGDalFrameFile(fpath, overview=1)[:].shape)
            >>> print(LazyGDalFrameFile(fpath, overview=2)[:].shape)
            >>> print(LazyGDalFrameFile(fpath, overview=3)[:].shape)
            >>> print(LazyGDalFrameFile(fpath, overview=3)[0:20].shape)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> import kwarray
            >>> import timerit
            >>> kwplot.autompl()
            >>> datas = []
            >>> ti = timerit.Timerit(50, bestof=3, verbose=0)
            >>> for overview in range(0, 4):
            >>>     import timerit
            >>>     for timer in ti.reset('time'):
            >>>         with timer:
            >>>             self = LazyGDalFrameFile(fpath, overview=overview)
            >>>             imdata = self[100:200, 100:200]
            >>>     sec = ti.mean()
            >>>     imdata = kwimage.draw_header_text(imdata, f'{overview=}\n{self.shape}\n[100:200, 100:200]\n{sec:0.6f}', fit='shrink')
            >>>     datas.append(imdata)
            >>> for overview in range(0, 4):
            >>>     import timerit
            >>>     aff = kwimage.Affine.coerce(scale=1 / (2 ** overview))
            >>>     crop_box = kwimage.Boxes.from_slice((slice(100, 200), slice(100, 200)))
            >>>     raw_crop = crop_box.warp(aff.inv()).quantize().to_slices()[0]
            >>>     for timer in ti.reset('time'):
            >>>         with timer:
            >>>             self = LazyGDalFrameFile(fpath, overview=0)
            >>>             raw_imdata = self[raw_crop]
            >>>             imdata = kwimage.warp_affine(raw_imdata, aff, dsize=(100, 100), interpolation='lanczos', antialias=1)
            >>>     sec = ti.mean()
            >>>     def _slicestr(sl):
            >>>         return '{}:{}'.format(sl.start, sl.stop)
            >>>     crop_str = '[{}, {}]'.format(_slicestr(raw_crop[0]), _slicestr(raw_crop[1]))
            >>>     imdata = kwimage.draw_header_text(imdata, f'simulate overview\n{self.shape}\n{crop_str}\n{sec:0.6f}', fit='shrink')
            >>>     datas.append(imdata)
            >>> # TODO: time the alternative case where you load everything and then crop
            >>> canvas = kwimage.stack_images_grid(datas, chunksize=4, axis=0, pad=10)
            >>> canvas = kwimage.draw_header_text(canvas, 'demo of lazy crops with overviews', fit='shrink')
            >>> kwplot.imshow(canvas)
        """
        ds = self._ds
        height, width, C = self.shape

        if 1:
            INTERLEAVE = ds.GetMetadata('IMAGE_STRUCTURE').get('INTERLEAVE', '')
            if INTERLEAVE == 'BAND':
                if len(ds.GetSubDatasets()) > 0:
                    raise NotImplementedError('Cannot handle interleaved files yet')

        if not ub.iterable(index):
            index = [index]
        else:
            index = list(index)

        TOTAL_DIMS = 3  # (always H, W, C)

        # Handle ellipsis
        num_ellipsis = index.count(Ellipsis)
        if num_ellipsis > 1:
            raise Exception('an index can only have a single ellipsis')
        elif num_ellipsis == 1:
            # Expand the ellipsis
            ell_idx = index.index(Ellipsis)
            n = (1 + TOTAL_DIMS - len(index))
            if n > 0:
                index = index[:ell_idx] + ([slice(None, None)] * n) + index[ell_idx + 1:]
                # index = index[:ell_idx] + ([None] * n) + index[ell_idx + 1:]
        else:
            # Expand trailing dims
            if len(index) < TOTAL_DIMS:
                n = (TOTAL_DIMS - len(index))
                index = index + [slice(None, None)] * n
                # index = index + [None] * n

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

        from kwimage.im_io import _gdal_read
        gdal_dset = ds
        nodata_method = self.nodata_method
        if nodata_method == 'auto':
            nodata_method = 'float'  # just use floats
        ignore_color_table = True
        overview = self.overview
        imdata, _ = _gdal_read(gdal_dset=gdal_dset, overview=overview,
                               nodata_method=nodata_method,
                               nodata_value=None,
                               ignore_color_table=ignore_color_table,
                               band_indices=band_indices, gdalkw=gdalkw)
        return imdata

    def __array__(self):
        """
        Allow this object to be passed to np.asarray

        References:
            https://numpy.org/doc/stable/user/basics.dispatch.html
        """
        return self[:]


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


def _validate_nonzero_data(file):
    """
    Test to see if the image is all black.

    May fail on all-black images

    Example:
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> import kwimage
        >>> gpath = kwimage.grab_test_image_fpath()
        >>> file = LazyGDalFrameFile(gpath)
        >>> _validate_nonzero_data(file)
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


def _read_envi_header(file):
    """
    USAGE: hdr = _read_envi_header(file)

    Reads an ENVI ".hdr" file header and returns the parameters in a
    dictionary as strings.  Header field names are treated as case
    insensitive and all keys in the dictionary are lowercase.

    Modified from spectral/io/envi.py

    References:
        https://github.com/spectralpython/spectral
    """
    f = open(file, 'r')

    try:
        starts_with_ENVI = f.readline().strip().startswith('ENVI')
    except UnicodeDecodeError:
        msg = (
            'File does not appear to be an ENVI header (appears to be a '
            'binary file).')
        f.close()
        raise Exception(msg)
    else:
        if not starts_with_ENVI:
            msg = ('File does not appear to be an ENVI header (missing "ENVI" '
                   'at beginning of first line).')
            f.close()
            raise Exception(msg)

    lines = f.readlines()
    f.close()

    dict = {}
    have_nonlowercase_param = False
    support_nonlowercase_params = False
    try:
        while lines:
            line = lines.pop(0)
            if line.find('=') == -1:
                continue
            if line[0] == ';':
                continue

            (key, sep, val) = line.partition('=')
            key = key.strip()
            if not key.islower():
                have_nonlowercase_param = True
                if not support_nonlowercase_params:
                    key = key.lower()
            val = val.strip()
            if val and val[0] == '{':
                str = val.strip()
                while str[-1] != '}':
                    line = lines.pop(0)
                    if line[0] == ';':
                        continue
                    str += '\n' + line.strip()
                if key == 'description':
                    dict[key] = str.strip('{}').strip()
                else:
                    vals = str[1:-1].split(',')
                    for j in range(len(vals)):
                        vals[j] = vals[j].strip()
                    dict[key] = vals
            else:
                dict[key] = val

        if have_nonlowercase_param and not support_nonlowercase_params:
            import warnings
            msg = 'Parameters with non-lowercase names encountered ' \
                  'and converted to lowercase. To retain source file ' \
                  'parameter name capitalization, set ' \
                  'spectral.settings.envi_support_nonlowercase_params to ' \
                  'True.'
            warnings.warn(msg)
        return dict
    except Exception:
        raise
