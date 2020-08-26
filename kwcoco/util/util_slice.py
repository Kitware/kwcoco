import numpy as np
import ubelt as ub


def padded_slice(data, in_slice, ndim=None, pad_slice=None,
                 pad_mode='constant', **padkw):
    """
    Allows slices with out-of-bound coordinates.  Any out of bounds coordinate
    will be sampled via padding.

    Note:
        Negative slices have a different meaning here then they usually do.
        Normally, they indicate a wrap-around or a reversed stride, but here
        they index into out-of-bounds space (which depends on the pad mode).
        For example a slice of -2:1 literally samples two pixels to the left of
        the data and one pixel from the data, so you get two padded values and
        one data value.

    Args:
        data (Sliceable[T]): data to slice into. Any channels must be the last dimension.
        in_slice (Tuple[slice, ...]): slice for each dimensions
        ndim (int): number of spatial dimensions
        pad_slice (List[int|Tuple]): additional padding of the slice

    Returns:
        Tuple[Sliceable, Dict] :

            data_sliced: subregion of the input data (possibly with padding,
                depending on if the original slice went out of bounds)

            transform : information on how to return to the original coordinates

                Currently a dict containing:
                    st_dims: a list indicating the low and high space-time
                        coordinate values of the returned data slice.

    Example:
        >>> data = np.arange(5)
        >>> in_slice = [slice(-2, 7)]

        >>> data_sliced, transform = padded_slice(data, in_slice)
        >>> print(ub.repr2(data_sliced, with_dtype=False))
        np.array([0, 0, 0, 1, 2, 3, 4, 0, 0])

        >>> data_sliced, transform = padded_slice(data, in_slice, pad_slice=(3, 3))
        >>> print(ub.repr2(data_sliced, with_dtype=False))
        np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0])

        >>> data_sliced, transform = padded_slice(data, slice(3, 4), pad_slice=[(1, 0)])
        >>> print(ub.repr2(data_sliced, with_dtype=False))
        np.array([2, 3])

    """
    if isinstance(in_slice, slice):
        in_slice = [in_slice]

    ndim = len(in_slice)

    data_dims = data.shape[:ndim]

    low_dims = [sl.start for sl in in_slice]
    high_dims = [sl.stop for sl in in_slice]

    data_slice, extra_padding = _rectify_slice(data_dims, low_dims, high_dims,
                                               pad=pad_slice)

    in_slice_clipped = tuple(slice(*d) for d in data_slice)
    # Get the parts of the image that are in bounds
    data_clipped = data[in_slice_clipped]

    # Add any padding that is needed to behave like negative dims exist
    if sum(map(sum, extra_padding)) == 0:
        # The slice was completely in bounds
        data_sliced = data_clipped
    else:
        if len(data.shape) != len(extra_padding):
            extra_padding = extra_padding + [(0, 0)]
        data_sliced = np.pad(data_clipped, extra_padding, mode=pad_mode,
                             **padkw)

    st_dims = data_slice[0:ndim]
    pad_dims = extra_padding[0:ndim]

    st_dims = [(s - pad[0], t + pad[1])
               for (s, t), pad in zip(st_dims, pad_dims)]

    # TODO: return a better transform back to the original space
    transform = {
        'st_dims': st_dims,
        'st_offset': [d[0] for d in st_dims]
    }
    return data_sliced, transform


def _rectify_slice(data_dims, low_dims, high_dims, pad=None):
    """
    Given image dimensions, bounding box dimensions, and a padding get the
    corresponding slice from the image and any extra padding needed to achieve
    the requested window size.

    Args:
        data_dims (tuple): n-dimension data sizes (e.g. 2d height, width)
        low_dims (tuple): bounding box low values (e.g. 2d ymin, xmin)
        high_dims (tuple): bounding box high values (e.g. 2d ymax, xmax)
        pad (tuple): (List[int|Tuple]):
            pad applied to (left and right) / (both) sides of each slice dim

    Returns:
        Tuple:
            data_slice - low and high values of a fancy slice corresponding to
                the image with shape `data_dims`. This slice may not correspond
                to the full window size if the requested bounding box goes out
                of bounds.
            extra_padding - extra padding needed after slicing to achieve
                the requested window size.

    Example:
        >>> # Case where 2D-bbox is inside the data dims on left edge
        >>> # Comprehensive 1D-cases are in the unit-test file
        >>> data_dims  = [300, 300]
        >>> low_dims   = [0, 0]
        >>> high_dims  = [10, 10]
        >>> pad        = [10, 5]
        >>> a, b = _rectify_slice(data_dims, low_dims, high_dims, pad)
        >>> print('data_slice = {!r}'.format(a))
        >>> print('extra_padding = {!r}'.format(b))
        data_slice = [(0, 20), (0, 15)]
        extra_padding = [(10, 0), (5, 0)]
    """
    # Determine the real part of the image that can be sliced out
    data_slice = []
    extra_padding = []
    if pad is None:
        pad = 0
    if isinstance(pad, int):
        pad = [pad] * len(data_dims)
    # Normalize to left/right pad value for each dim
    pad_slice = [p if ub.iterable(p) else [p, p] for p in pad]

    # Determine the real part of the image that can be sliced out
    for D_img, d_low, d_high, d_pad in zip(data_dims, low_dims, high_dims, pad_slice):
        if d_low > d_high:
            raise ValueError('d_low > d_high: {} > {}'.format(d_low, d_high))
        # Determine where the bounds would be if the image size was inf
        raw_low = d_low - d_pad[0]
        raw_high = d_high + d_pad[1]
        # Clip the slice positions to the real part of the image
        sl_low = min(D_img, max(0, raw_low))
        sl_high = min(D_img, max(0, raw_high))
        data_slice.append((sl_low, sl_high))

        # Add extra padding when the window extends past the real part
        low_diff = sl_low - raw_low
        high_diff = raw_high - sl_high

        # Hand the case where both raw coordinates are out of bounds
        extra_low = max(0, low_diff + min(0, high_diff))
        extra_high = max(0, high_diff + min(0, low_diff))
        extra = (extra_low, extra_high)
        extra_padding.append(extra)
    return data_slice, extra_padding
