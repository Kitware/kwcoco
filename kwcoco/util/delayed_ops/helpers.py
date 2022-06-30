import kwimage
import ubelt as ub
import numpy as np


try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


def _auto_dsize(transform, sub_dsize):
    sub_w, sub_h = sub_dsize
    sub_bounds = kwimage.Coords(
        np.array([[0,     0], [sub_w, 0],
                  [0, sub_h], [sub_w, sub_h]])
    )
    bounds = sub_bounds.warp(transform.matrix)
    max_xy = np.ceil(bounds.data.max(axis=0))
    max_x = int(max_xy[0])
    max_y = int(max_xy[1])
    dsize = (max_x, max_y)
    return dsize


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


@profile
def _swap_warp_after_crop(root_region_bounds, tf_leaf_to_root):
    r"""
    Given a warp followed by a crop, compute the corresponding crop followed by
    a warp.

    Given a region in a "root" image and a trasnform between that "root" and
    some "leaf" image, compute the appropriate quantized region in the "leaf"
    image and the adjusted transformation between that root and leaf.

    Args:
        root_region_bounds (kwimage.Polygon):
            region representing the crop that happens after the warp

        tf_leaf_to_root (kwimage.Affine):
            the warp that happens before the input crop

    Returns:
        Tuple[Tuple[slice, slice], kwimage.Affine]:
            leaf_crop_slices - the crop that happens before the warp
            tf_newleaf_to_newroot - warp that happens after the crop.

    Example:
        >>> region_slices = (slice(33, 100), slice(22, 62))
        >>> region_shape = (100, 100, 1)
        >>> root_region_box = kwimage.Boxes.from_slice(region_slices, shape=region_shape)
        >>> root_region_bounds = root_region_box.to_polygons()[0]
        >>> tf_leaf_to_root = kwimage.Affine.affine(scale=7).matrix
        >>> slices, tf_new = _swap_warp_after_crop(root_region_bounds, tf_leaf_to_root)
        >>> print('tf_new =\n{!r}'.format(tf_new))
        >>> print('slices = {!r}'.format(slices))
    """
    # Transform the region bounds into the sub-image space
    tf_leaf_to_root = kwimage.Affine.coerce(tf_leaf_to_root)
    tf_root_to_leaf = tf_leaf_to_root.inv()
    tf_root_to_leaf = tf_root_to_leaf.__array__()
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

    tf_root_to_newroot = kwimage.Affine.affine(offset=-root_offset).matrix
    tf_newleaf_to_leaf = kwimage.Affine.affine(offset=crop_offset).matrix

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


@profile
def _swap_crop_after_warp(inner_region, outer_transform):
    r"""
    Given a crop followed by a warp (usually an overview), compute the
    corresponding warp followed by a crop followed by a small correction warp.

    Note that in general it is not possible to ensure the crop is the last
    operation, there may need to be a small warp after it.

    However, this is generally only useful when the warp being pushed early in
    the operation chain corresponds to an overview, and often - but not always
    - the final warp will simply be the identity.

    Args:
        inner_region (kwimage.Polygon):
            region representing the crop that happens before the warp

        outer_transform (kwimage.Affine):
            the warp that happens after the input crop

    Returns:
        Tuple[kwimage.Affine, Tuple[slice, slice], kwimage.Affine]:

            new_inner_warp - the new warp to happen before the crop

            outer_crop - the new crop after the main warp

            new_outer_warp - a small subpixel alignment warp to happen last

    Example:
        >>> from kwcoco.util.delayed_ops.helpers import *  # NOQA
        >>> region_slices = (slice(33, 100), slice(22, 62))
        >>> region_shape = (100, 100, 1)
        >>> inner_region = kwimage.Boxes.from_slice(region_slices)
        >>> inner_region = inner_region.to_polygons()[0]
        >>> outer_transform = kwimage.Affine.affine(scale=1/4)
        >>> new_inner_warp, outer_crop, new_outer_warp = _swap_crop_after_warp(inner_region, outer_transform)
        >>> print('new_inner_warp = {}'.format(ub.repr2(new_inner_warp, nl=1)))
        >>> print('outer_crop = {}'.format(ub.repr2(outer_crop, nl=1)))
        >>> print('new_outer_warp = {}'.format(ub.repr2(new_outer_warp, nl=1)))
    """
    # Find where the inner region maps to after the transform is applied
    outer_region = inner_region.warp(outer_transform)

    # Transform the region bounds into the sub-image space
    outer_box = outer_region.bounding_box().to_ltrb()

    # Quantize to a region that is possible to sample from
    outer_crop_box = outer_box.quantize()

    # is this ok?
    outer_crop_box = outer_crop_box.clip(0, 0, None, None)

    # Because the new crop might not be perfectly aligned, we might need to
    # nudge it a bit after we crop out its bounds.
    crop_offset = outer_crop_box.data[0, 0:2]
    outer_offset = outer_region.exterior.data.min(axis=0)

    # Compute the extra transform that will realign the quantized croped data
    # with the original warped inner crop.
    tf_crop_to_box = kwimage.Affine.affine(
        offset=crop_offset - outer_offset
    )

    lt_x, lt_y, rb_x, rb_y = outer_crop_box.data[0, 0:4]
    outer_crop = (slice(lt_y, rb_y), slice(lt_x, rb_x))
    new_outer_warp = tf_crop_to_box

    # The inner warp will be the same as the original outer warp.
    new_inner_warp = outer_transform

    return new_inner_warp, outer_crop, new_outer_warp


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


def quantize_float01(imdata, old_min=0, old_max=1, quantize_dtype=np.int16):
    """
    Note:
        Setting old_min / old_max indicates the possible extend of the input
        data (and it will be clipped to it). It does not mean that the input
        data has to have those min and max values, but it should be between
        them.

    Example:
        >>> from kwcoco.util.delayed_ops.helpers import *  # NOQA
        >>> # Test error when input is not nicely between 0 and 1
        >>> imdata = (np.random.randn(32, 32, 3) - 1.) * 2.5
        >>> quant1, quantization1 = quantize_float01(imdata, old_min=0, old_max=1)
        >>> recon1 = dequantize(quant1, quantization1)
        >>> error1 = np.abs((recon1 - imdata)).sum()
        >>> print('error1 = {!r}'.format(error1))
        >>> #
        >>> for i in range(1, 20):
        >>>     print('i = {!r}'.format(i))
        >>>     quant2, quantization2 = quantize_float01(imdata, old_min=-i, old_max=i)
        >>>     recon2 = dequantize(quant2, quantization2)
        >>>     error2 = np.abs((recon2 - imdata)).sum()
        >>>     print('error2 = {!r}'.format(error2))

    Example:
        >>> # Test dequantize with uint8
        >>> from watch.tasks.fusion.predict import *  # NOQA
        >>> from kwcoco.util.util_delayed_poc import dequantize
        >>> imdata = np.random.randn(32, 32, 3)
        >>> quant1, quantization1 = quantize_float01(imdata, old_min=0, old_max=1, quantize_dtype=np.uint8)
        >>> recon1 = dequantize(quant1, quantization1)
        >>> error1 = np.abs((recon1 - imdata)).sum()
        >>> print('error1 = {!r}'.format(error1))

    Example:
        >>> # Test quantization with different signed / unsigned combos
        >>> from watch.tasks.fusion.predict import *  # NOQA
        >>> print(quantize_float01(None, 0, 1, np.int16))
        >>> print(quantize_float01(None, 0, 1, np.int8))
        >>> print(quantize_float01(None, 0, 1, np.uint8))
        >>> print(quantize_float01(None, 0, 1, np.uint16))

    """
    # old_min = 0
    # old_max = 1
    quantize_iinfo = np.iinfo(quantize_dtype)
    quantize_max = quantize_iinfo.max
    if quantize_iinfo.kind == 'u':
        # Unsigned quantize
        quantize_nan = 0
        quantize_min = 1
    elif quantize_iinfo.kind == 'i':
        # Signed quantize
        quantize_min = 0
        quantize_nan = max(-9999, quantize_iinfo.min)

    quantization = {
        'orig_min': old_min,
        'orig_max': old_max,
        'quant_min': quantize_min,
        'quant_max': quantize_max,
        'nodata': quantize_nan,
    }

    old_extent = (old_max - old_min)
    new_extent = (quantize_max - quantize_min)
    quant_factor = new_extent / old_extent

    if imdata is not None:
        invalid_mask = np.isnan(imdata)
        new_imdata = (imdata.clip(old_min, old_max) - old_min) * quant_factor + quantize_min
        new_imdata = new_imdata.astype(quantize_dtype)
        new_imdata[invalid_mask] = quantize_nan
    else:
        new_imdata = None

    return new_imdata, quantization
