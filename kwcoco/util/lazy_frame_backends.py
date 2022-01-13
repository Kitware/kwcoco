"""
Ducktyped interfaces for loading subregions of images with standard slice
syntax
"""
import os
import ubelt as ub
import numpy as np
import kwimage
from os.path import join, exists

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


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

    @classmethod
    def available(self):
        """
        Returns True if this backend is available
        """
        return _have_gdal()

    @ub.memoize_property
    def _ds(self):
        from osgeo import gdal
        if not exists(self.fpath):
            raise Exception('File does not exist: {}'.format(self.fpath))
        _fpath = os.fspath(self.fpath)
        if _fpath.endswith('.hdr'):
            # Use spectral-like process to point gdal to the correct file given
            # the hdr
            ext = '.' + _read_envi_header(_fpath)['interleave']
            _fpath = ub.augpath(_fpath, ext=ext)
        ds = gdal.Open(_fpath, gdal.GA_ReadOnly)
        if ds is None:
            raise Exception((
                'GDAL Failed to open the fpath={!r} for an unknown reason. '
                'Call gdal.UseExceptions() beforehand to get the '
                'real exception').format(self.fpath))
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
            try:
                img_part = np.empty(shape, dtype=dtype)
            except ValueError:
                print('ERROR')
                print('self.fpath = {!r}'.format(self.fpath))
                print('dtype = {!r}'.format(dtype))
                print('shape = {!r}'.format(shape))
                raise
            for out_idx, band in enumerate(bands):
                buf = band.ReadAsArray(**gdalkw)
                if buf is None:
                    raise IOError(ub.paragraph(
                        '''
                        GDAL was unable to read band: {}, {}, with={}
                        from fpath={!r}
                        '''.format(out_idx, band, gdalkw, self.fpath)))
                img_part[:, :, out_idx] = buf
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
