"""
These functions might be added to kwimage
"""
import numpy as np
import cv2
import ubelt as ub

# Backwards compat definition. Code was ported to kwimage.
from kwimage import connected_components  # NOQA
from kwimage import draw_header_text  # NOQA
from kwimage import Box  # NOQA
import kwarray


def _auto_kernel_sigma(kernel=None, sigma=None, autokernel_mode='ours'):
    """
    Attempt to determine sigma and kernel size from heuristics

    Example:
        >>> _auto_kernel_sigma(None, None)
        >>> _auto_kernel_sigma(3, None)
        >>> _auto_kernel_sigma(None, 0.8)
        >>> _auto_kernel_sigma(7, None)
        >>> _auto_kernel_sigma(None, 1.4)

    Ignore:
        >>> # xdoctest: +REQUIRES(--demo)
        >>> rows = []
        >>> for k in np.arange(3, 101, 2):
        >>>     s = _auto_kernel_sigma(k, None)[1][0]
        >>>     rows.append({'k': k, 's': s, 'type': 'auto_sigma'})
        >>> #
        >>> sigmas = np.array([r['s'] for r in rows])
        >>> other = np.linspace(0, sigmas.max() + 1, 100)
        >>> sigmas = np.unique(np.hstack([sigmas, other]))
        >>> sigmas.sort()
        >>> for s in sigmas:
        >>>     k = _auto_kernel_sigma(None, s, autokernel_mode='cv2')[0][0]
        >>>     rows.append({'k': k, 's': s, 'type': 'auto_kernel (cv2)'})
        >>> #
        >>> for s in sigmas:
        >>>     k = _auto_kernel_sigma(None, s, autokernel_mode='ours')[0][0]
        >>>     rows.append({'k': k, 's': s, 'type': 'auto_kernel (ours)'})
        >>> import pandas as pd
        >>> df = pd.DataFrame(rows)
        >>> p = df.pivot(index=['s'], columns=['type'], values=['k'])
        >>> print(p[~p.droplevel(0, axis=1).auto_sigma.isnull()])
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> sns = kwplot.autosns()
        >>> sns.lineplot(data=df, x='s', y='k', hue='type')
    """
    import numbers
    if kernel is None and sigma is None:
        kernel = 3

    if kernel is not None:
        if isinstance(kernel, numbers.Integral):
            k_x = k_y = kernel
        else:
            k_x, k_y = kernel

    if sigma is None:
        # https://github.com/egonSchiele/OpenCV/blob/09bab41/modules/imgproc/src/smooth.cpp#L344
        sigma_x = 0.3 * ((k_x - 1) * 0.5 - 1) + 0.8
        sigma_y = 0.3 * ((k_y - 1) * 0.5 - 1) + 0.8
    else:
        if isinstance(sigma, numbers.Number):
            sigma_x = sigma_y = sigma
        else:
            sigma_x, sigma_y = sigma

    if kernel is None:
        if autokernel_mode == 'zero':
            # When 0 computed internally via cv2
            k_x = k_y = 0
        elif autokernel_mode == 'cv2':
            # if USE_CV2_DEF:
            # This is the CV2 definition
            # https://github.com/egonSchiele/OpenCV/blob/09bab41/modules/imgproc/src/smooth.cpp#L387
            depth_factor = 3  # or 4 for non-uint8
            k_x = int(round(sigma_x * depth_factor * 2 + 1)) | 1
            k_y = int(round(sigma_y * depth_factor * 2 + 1)) | 1
        elif autokernel_mode == 'ours':
            # But I think this definition makes more sense because it keeps
            # sigma and the kernel in agreement more often
            """
            # Our hueristic is computed via solving the sigma heuristic for k
            import sympy as sym
            s, k = sym.symbols('s, k', rational=True)
            sa = sym.Rational('3 / 10') * ((k - 1) / 2 - 1) + sym.Rational('8 / 10')
            sym.solve(sym.Eq(s, sa), k)
            """
            k_x = max(3, round(20 * sigma_x / 3 - 7 / 3)) | 1
            k_y = max(3, round(20 * sigma_y / 3 - 7 / 3)) | 1
        else:
            raise KeyError(autokernel_mode)
    sigma = (sigma_x, sigma_y)
    kernel = (k_x, k_y)
    return kernel, sigma


@ub.memoize
def upweight_center_mask(shape):
    """
    Example:
        >>> from geowatch.utils.util_kwimage import *  # NOQA
        >>> shapes = [32, 64, 96, 128, 256]
        >>> results = {}
        >>> for shape in shapes:
        >>>     results[str(shape)] = upweight_center_mask(shape)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nSubplots=len(results))
        >>> for k, result in results.items():
        >>>     kwplot.imshow(result, pnum=pnum_(), title=k)
        >>> kwplot.show_if_requested()
    """
    import kwimage
    shape, sigma = _auto_kernel_sigma(kernel=shape)
    sigma_x, sigma_y = sigma
    weights = kwimage.gaussian_patch(shape, sigma=(sigma_x, sigma_y))
    weights = weights / weights.max()
    # weights = kwimage.ensure_uint255(weights)
    kernel = np.maximum(np.array(shape) // 8, 3)
    kernel = kernel + (1 - (kernel % 2))
    weights = kwimage.morphology(
        weights, kernel=kernel, mode='dilate', element='rect', iterations=1)
    weights = kwimage.ensure_float01(weights)
    weights = np.maximum(weights, 0.001)
    return weights


def perchannel_colorize(data, channel_colors=None):
    """
    Note: this logic semi-exists in kwimage.Heatmap.
    It would be good to consolidate it.

    Args:
        data (ndarray): the last dimension should be chanels, and they should
            be probabilities between zero and one.

    Example:
        >>> from geowatch.utils.util_kwimage import *  # NOQA
        >>> import itertools as it
        >>> import kwarray
        >>> channel_colors = ['tomato', 'gold', 'lime', 'darkturquoise']
        >>> c = len(channel_colors)
        >>> s = 32
        >>> cx_combos = list(ub.flatten(it.combinations(range(c), n) for n in range(0, c + 1)))
        >>> w = s // len(cx_combos)
        >>> data = np.zeros((s, s, c), dtype=np.float32)
        >>> y_slider = kwarray.SlidingWindow((s, s), (w, s,))
        >>> x_slider = kwarray.SlidingWindow((s, s), (s, w,))
        >>> for idx, cxs in enumerate(cx_combos):
        >>>     for cx in cxs:
        >>>         data[x_slider[idx] + (cx,)] =  1
        >>>         data[y_slider[idx] + (cx,)] =  0.5
        >>> canvas = perchannel_colorize(data, channel_colors)[..., 0:3]
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas, docla=1)

    Example:
        >>> from geowatch.utils.util_kwimage import *  # NOQA
        >>> import itertools as it
        >>> import kwarray
        >>> channel_colors = ['blue']
        >>> data = np.linspace(0, 1, 512 * 512).reshape(512, 512, 1)
        >>> canvas = perchannel_colorize(data, channel_colors)[..., 0:3]
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas, docla=1)
    """
    import kwimage

    num_channels = data.shape[2]

    if len(data.shape) == 2:
        # add in prefix channel if its not there
        data = data[None, :, :]

    existing_colors = [
        kwimage.Color.coerce(c).as01()
        for c in channel_colors if c is not None
    ]

    # Define default colors
    fill_colors = kwimage.Color.distinct(
        num_channels - len(existing_colors),
        existing=existing_colors)
    fill_color_iter = iter(fill_colors)

    resolved_channel_colors = []
    for c in channel_colors:
        if c is None:
            c = next(fill_color_iter)
        else:
            c = kwimage.Color.coerce(c).as01()
        resolved_channel_colors.append(c)

    # Each class gets its own color, and modulates the alpha
    sumtotal = np.nansum(data, axis=2)
    sumtotal[np.isnan(sumtotal)] = 1
    sumtotal[sumtotal == 0] = 1
    sumtotal = np.maximum(sumtotal, 1)
    layers = []
    for cidx in range(num_channels):
        chan = data[:, :, cidx]
        alpha = chan / sumtotal
        # alpha = chan / num_channels
        color = resolved_channel_colors[cidx]
        layer = np.empty(tuple(chan.shape) + (4,))
        layer[..., 3] = alpha
        layer[..., 0:3] = color
        layers.append(layer)

    background = np.zeros_like(layer)
    background[..., 3] = 1
    layers.append(background)
    colormask = kwimage.overlay_alpha_layers(layers, keepalpha=False)
    # colormask[..., 3] *= with_alpha
    return colormask


def ensure_false_color(canvas, method='ortho'):
    """
    Given a canvas with more than 3 colors, (or 2 colors) do
    something to get it into a colorized space.

    TODO:
        - [ ] I have no idea how well this works. Probably better methods exist. Find them.

    Example:
        >>> import kwimage
        >>> import numpy as np
        >>> demo_img = kwimage.ensure_float01(kwimage.grab_test_image('astro'))
        >>> canvas = demo_img @ np.random.rand(3, 2)
        >>> rgb_canvas2 = ensure_false_color(canvas)
        >>> canvas = np.tile(demo_img, (1, 1, 10))
        >>> rgb_canvas10 = ensure_false_color(canvas)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(rgb_canvas2, pnum=(1, 2, 1))
        >>> kwplot.imshow(rgb_canvas10, pnum=(1, 2, 2))
    """
    import kwarray
    import numpy as np
    import kwimage
    canvas = kwarray.atleast_nd(canvas, 3)

    if canvas.shape[2] in {1, 3}:
        rgb_canvas = canvas
    # elif canvas.shape[2] == 2:
    #     # Use LAB to colorize
    #     L_part = np.ones_like(canvas[..., 0:1]) * 50
    #     a_min = -86.1875
    #     a_max = 98.234375
    #     b_min = -107.859375
    #     b_max = 94.46875
    #     a_part = (canvas[..., 0:1] - a_min) / (a_max - a_min)
    #     b_part = (canvas[..., 1:2] - b_min) / (b_max - b_min)
    #     lab_canvas = np.concatenate([L_part, a_part, b_part], axis=2)
    #     rgb_canvas = kwimage.convert_colorspace(lab_canvas, src_space='lab', dst_space='rgb')
    else:

        if method == 'ortho':
            rng = kwarray.ensure_rng(canvas.shape[2])
            seedmat = rng.rand(canvas.shape[2], 3).T
            h, tau = np.linalg.qr(seedmat, mode='raw')
            false_colored = (canvas @ h)
            rgb_canvas = kwarray.normalize(false_colored)
        elif method.lower() == 'pca':
            import sklearn
            ndim = canvas.ndim
            dims = canvas.shape[0:2]
            if ndim == 2:
                in_channels = 1
            else:
                in_channels = canvas.shape[2]

            if in_channels > 1:
                model = sklearn.decomposition.PCA(1)
                X = canvas.reshape(-1, in_channels)
                X_ = model.fit_transform(X)
                gray = X_.reshape(dims)
                viz = kwimage.make_heatmask(gray, with_alpha=1)[:, :, 0:3]
            else:
                gray = canvas.reshape(dims)
                viz = gray
            return viz
    return rgb_canvas


def colorize_label_image(labels, with_legend=True, label_mapping=None,
                         label_to_color=None, legend_dpi=200):
    """
    Rename to draw_label_image?

    Replace an image with integer labels with colors

    Args:
        labels (ndarray): a label image
        with_legend (bool):
        legend_mapping (dict):
            maps the label used in the label image to what should appear in the
            legend.

    CommandLine:
        python -X importtime -m xdoctest geowatch.utils.util_kwimage colorize_label_image

    Example:
        >>> from geowatch.utils.util_kwimage import *  # NOQA
        >>> labels = (np.random.rand(32, 32) * 10).astype(np.uint8) % 5
        >>> label_to_color = {0: 'black'}
        >>> label_mapping = {0: 'background'}
        >>> with_legend = True
        >>> canvas1 = colorize_label_image(labels, with_legend,
        >>>     label_mapping=label_mapping, label_to_color=label_to_color)
        >>> canvas2 = colorize_label_image(labels, with_legend,
        >>>     label_mapping=label_mapping, label_to_color=None)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas1, pnum=(1, 2, 1), fnum=1)
        >>> kwplot.imshow(canvas2, pnum=(1, 2, 2), fnum=1)

    Example:
        >>> from geowatch.utils.util_kwimage import *  # NOQA
        >>> labels = (np.random.rand(4, 4) * 10) % 5
        >>> labels[0:2] = np.nan
        >>> label_to_color = {0: 'black'}
        >>> label_mapping = {0: 'background'}
        >>> with_legend = True
        >>> canvas1 = colorize_label_image(labels, with_legend,
        >>>     label_mapping=label_mapping, label_to_color=label_to_color)
        >>> canvas2 = colorize_label_image(labels, with_legend,
        >>>     label_mapping=label_mapping, label_to_color=None)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas1, pnum=(1, 2, 1), fnum=1)
        >>> kwplot.imshow(canvas2, pnum=(1, 2, 2), fnum=1)
    """
    import kwimage
    unique_labels, inv = np.unique(labels, return_inverse=True)

    if np.isnan(unique_labels).any():
        # need specialized nan handling because we are going to use unique
        # values as keys in a dictionary.
        import math
        unique_labels = ['nan' if math.isnan(f) else f for f in unique_labels]

    if label_to_color is not None:
        used_labels = set(label_to_color) & set(unique_labels)
        label_to_color_ = {k: kwimage.Color(c).as01() for k, c in label_to_color.items()}
        existing = list(label_to_color_.values())
        uncolored_labels = list(set(unique_labels) - set(used_labels))
    else:
        existing = None
        uncolored_labels = unique_labels
        used_labels = set()
        label_to_color_ = {}

    # When there are a lot of unique colors this takes a very long time
    new_label_colors = kwimage.Color.distinct(len(uncolored_labels), existing=existing)
    label_to_color_.update(ub.dzip(uncolored_labels, new_label_colors))

    unique_label_colors = [label_to_color_[c] for c in unique_labels]
    unique_label_colors = np.array(unique_label_colors)
    colored_label_img = unique_label_colors[inv].reshape(labels.shape + (unique_label_colors.shape[-1],))

    # index_to_color = np.array([kwimage.Color('black').as01()] + label_colors)
    # colored_label_img = index_to_color[labels]
    if with_legend:
        import kwplot

        label_to_color = ub.dzip(unique_labels, unique_label_colors)
        if label_mapping:
            label_to_color = {str(k) if k not in label_mapping else str(k) + ': ' + str(label_mapping[k]): v
                              for k, v in label_to_color.items()}

        legend = kwplot.make_legend_img(label_to_color, dpi=legend_dpi)

        h1, w1 = legend.shape[0:2]
        h2, w2 = colored_label_img.shape[0:2]
        box1 = kwimage.Box.from_dsize((w1, h1))
        box2 = kwimage.Box.from_dsize((w2, h2))

        if 1:
            box3 = box1.copy()
            box3 = box3.scale(box2.width / box1.width)
            if box3.height > box2.height:
                box3 = box3.scale(box2.height / box3.height)
            # print(f'box2={box2}')
            # print(f'box3={box3}')
            # print(f'box1={box1}')
            sf = box3.width / box1.width
            legend = kwimage.imresize(legend, scale=sf)
            # if box3.width > box2.width:
            #     box3 = box3.scale(box2.height / box3.height)
            # if box1.
            # if w1 > w2:
            #     legend = kwimage.imresize(legend, dsize=(w2, None))
            #     h1, w1 = legend.shape[0, 1]
            # if h1 > h2:
            #     legend = kwimage.imresize(legend, dsize=(None, h2))

        canvas = kwimage.stack_images([colored_label_img, legend], axis=1,
                                      bg_value='gray')
        # resize='smaller')
    else:
        canvas = colored_label_img
    return canvas


def local_variance(image, kernel, handle_nans=True):
    """
    The local variance at each point in the image (take the sqrt to get the
    local std)

    Args:
        image (ndarray)
        kernel (int | Tuple[int, int]) kernel size (w, h)

    Returns:
        ndarray: the image with the variance at each point

    References:
        https://answers.opencv.org/question/193393/local-mean-and-variance/
        https://stackoverflow.com/questions/11456565/opencv-mean-sd-filter

    Example:
        >>> # Test with nans
        >>> from geowatch.utils.util_kwimage import *  # NOQA
        >>> import kwimage
        >>> import kwarray
        >>> shape = (512, 512)
        >>> dsize = shape[::-1]
        >>> image = np.zeros(shape + (3,), dtype=np.uint8)
        >>> image = kwimage.ensure_float01(image)
        >>> rng = kwarray.ensure_rng(0)
        >>> image[:, 256:, :] = rng.rand(512, 256, 3)  # high frequency noise
        >>> poly1 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly2 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly3 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly1.draw_on(image, color='kitware_blue')
        >>> poly2.draw_on(image, color='pink')
        >>> poly3.draw_on(image, color='kitware_green')
        >>> image[50:70, :, :] = 1  # a line of ones
        >>> image[150:170, :, :] = np.nan  # a line of nans
        >>> #image = kwimage.convert_colorspace(image, 'rgb', 'gray')
        >>> varimg = local_variance(image, kernel=7)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(kwimage.fill_nans_with_checkers(image), pnum=(1, 2, 1), title='input image')
        >>> kwplot.imshow(kwimage.fill_nans_with_checkers(kwarray.normalize(varimg)), pnum=(1, 2, 2), title='variance image')
    """
    ksize = (kernel, kernel) if isinstance(kernel, int) else kernel

    image_f = image.astype(np.float32, copy=True)
    invalid_mask = np.isnan(image_f)
    has_mask = np.any(invalid_mask)
    if has_mask:
        if len(invalid_mask.shape) > 2:
            invalid_mask = invalid_mask.any(axis=2)
        # What is a good replacement value?
        image_f[invalid_mask] = 0
    local_mean = cv2.boxFilter(image_f, ddepth=-1, ksize=ksize)
    diff = (image_f - local_mean)
    square_diff = diff * diff
    local_vari = cv2.boxFilter(square_diff, ddepth=-1, ksize=ksize)
    if has_mask:
        local_vari[invalid_mask] = np.nan
    return local_vari


def find_lowvariance_regions(image, kernel=7):
    """
    The idea is that we want to detect large region in an image that are filled
    entirely with the same color.

    The approach is that we are going to find the local variance of the image
    in a KxK window (K is the size of a kernel and corresponds to a minimum
    size of homogenous region that we care to segment).  Then we are going to
    find all regions with zero variance. The connected components of that
    binary image should be roughly what we want.

    We can postprocess this with floodfills to get nearly exacly what we want.

    Example:
        >>> from geowatch.utils.util_kwimage import *  # NOQA
        >>> import kwimage
        >>> import kwarray
        >>> shape = (512, 512)
        >>> dsize = shape[::-1]
        >>> image = np.zeros(shape + (3,), dtype=np.uint8)
        >>> rng = kwarray.ensure_rng(0)
        >>> image[:, 256:, :] = (rng.rand(512, 256, 3) * 255)  # high frequency noise
        >>> poly1 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly2 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly3 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly1.draw_on(image, color='kitware_blue')
        >>> poly2.draw_on(image, color='pink')
        >>> poly3.draw_on(image, color='kitware_green')
        >>> image[50:70, :, :] = 255  # a "thin" line
        >>> labels = find_lowvariance_regions(image)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> canvas = colorize_label_image(labels)
        >>> kwplot.imshow(image, pnum=(1, 2, 1), title='input image')
        >>> kwplot.imshow(canvas, pnum=(1, 2, 2), title='labeled regions')

    Example:
        >>> # Test with nans
        >>> from geowatch.utils.util_kwimage import *  # NOQA
        >>> import kwimage
        >>> import kwarray
        >>> shape = (512, 512)
        >>> dsize = shape[::-1]
        >>> image = np.zeros(shape + (3,), dtype=np.uint8)
        >>> image = kwimage.ensure_float01(image)
        >>> rng = kwarray.ensure_rng(0)
        >>> image[:, 256:, :] = rng.rand(512, 256, 3)  # high frequency noise
        >>> poly1 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly2 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly3 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly1.draw_on(image, color='kitware_blue')
        >>> poly2.draw_on(image, color='pink')
        >>> poly3.draw_on(image, color='kitware_green')
        >>> image[50:70, :, :] = 1  # a line of ones
        >>> image[150:170, :, :] = np.nan  # a line of nans
        >>> image = kwimage.convert_colorspace(image, 'rgb', 'gray')
        >>> labels = find_lowvariance_regions(image, kernel=7)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> canvas = colorize_label_image(labels)
        >>> kwplot.imshow(image, pnum=(1, 2, 1), title='input image')
        >>> kwplot.imshow(canvas, pnum=(1, 2, 2), title='labeled regions')
    """
    h, w = image.shape[0:2]
    # standard deviation filter
    # https://stackoverflow.com/questions/11456565/opencv-mean-sd-filter
    kernel = (kernel, kernel) if isinstance(kernel, int) else kernel
    vari_image = local_variance(image, kernel)

    if len(vari_image.shape) > 2:
        binary_image = (vari_image == 0).all(axis=2).astype(np.uint8)
    else:
        binary_image = (vari_image == 0).astype(np.uint8)
    # import kwimage
    labels, info = connected_components(binary_image, with_stats=False)
    return labels


def find_samecolor_regions(image, min_region_size=49, seed_method='grid',
                           connectivity=8, scale=1.0, grid_stride='auto',
                           PRINT_STEPS=0, values=None):
    """
    Find large spatially connected regions in an image where all pixels have
    the same value.

    Works by selecting a set of initial seed points. A flood fill is run at
    each seed point to find potential large samedata regions.

    More specifically, we find seed points using a regular grid, or finding
    pixels in regions with low spatial variance.  Then for each candidate seed
    point, we do a flood-fill to see if it in a large samecolor region. Each
    connected region that satisfies the requested criteria is given an integer
    label. The return value is a label-mask where each pixel that is part of a
    region is given a value corresponding to that region's integer label. Any
    pixel not part of a region is given a label of zero. For technical reasons,
    returned labels will start at 1.

    Args:
        image (ndarray):
            image to find regions of the same color

        min_region_size (int):
            the minimum number of pixels in a region for it to be
            considered valid.

        seed_method (str): can be grid or variance

        connectivity (int): cc connectivity. Either 4 or 8.

        scale (float): scale at which the computation is done.
            Should be a value between 0 and 1. The default is 1.  Setting to
            less than 1 will resize the image, perform the computation, and
            then upsample the output. This can cause a significant speed
            increase at the cost of some accuracy.

        values (None | List):
            if specified, only finds the samecolor regions with this intensity
            value, otherwise any value will be considered.

    References:
        https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga366aae45a6c1289b341d140839f18717

    Example:
        >>> from geowatch.utils.util_kwimage import *  # NOQA
        >>> import kwimage
        >>> import kwarray
        >>> shape = (512, 512)
        >>> dsize = shape[::-1]
        >>> image = np.zeros(shape + (3,), dtype=np.uint8)
        >>> rng = kwarray.ensure_rng(0)
        >>> image[:, 256:, :] = (rng.rand(512, 256, 3) * 255)  # high frequency noise
        >>> poly1 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly2 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly3 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly1.draw_on(image, color='kitware_blue')
        >>> poly2.draw_on(image, color='pink')
        >>> poly3.draw_on(image, color='kitware_green')
        >>> image[50:70, :, :] = 255  # a "thin" line
        >>> #labels = find_samecolor_regions(image, seed_method='grid')
        >>> labels = find_samecolor_regions(image, seed_method='variance')
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> canvas = colorize_label_image(labels)
        >>> kwplot.imshow(image, pnum=(1, 2, 1), title='input image')
        >>> kwplot.imshow(canvas, pnum=(1, 2, 2), title='labeled regions')

    Example:
        >>> # Test with nans
        >>> from geowatch.utils.util_kwimage import *  # NOQA
        >>> import kwimage
        >>> import kwarray
        >>> shape = (512, 512)
        >>> dsize = shape[::-1]
        >>> image = np.zeros(shape + (3,), dtype=np.uint8)
        >>> image = kwimage.ensure_float01(image).astype(np.float32)
        >>> rng = kwarray.ensure_rng(0)
        >>> image[:, 256:, :] = rng.rand(512, 256, 3)  # high frequency noise
        >>> poly1 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly2 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly3 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly1.draw_on(image, color='kitware_blue')
        >>> poly2.draw_on(image, color='pink')
        >>> poly3.draw_on(image, color='kitware_green')
        >>> image[50:70, :, :] = 1  # a line of ones
        >>> image[150:170, :, :] = np.nan  # a line of nans
        >>> image = kwimage.convert_colorspace(image, 'rgb', 'gray')
        >>> labels = find_samecolor_regions(image, seed_method='grid')
        >>> #labels = find_samecolor_regions(image, seed_method='variance')
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> canvas = colorize_label_image(labels)
        >>> kwplot.imshow(image, pnum=(1, 2, 1), title='input image')
        >>> kwplot.imshow(canvas, pnum=(1, 2, 2), title='labeled regions')

    Example:
        >>> from geowatch.utils.util_kwimage import *  # NOQA
        >>> w, h = 5, 4
        >>> image = (np.arange(w * h).reshape(h, w)).astype(np.uint8)
        >>> image[2, 2] = 0
        >>> image[2, 3] = 0
        >>> image[3, 4] = 0
        >>> min_region_size = 2
        >>> seed_method = 'grid'
        >>> connectivity = 8
        >>> scale = 1.0
        >>> grid_stride = 1
        >>> labels = find_samecolor_regions(
        >>>     image, min_region_size, seed_method, connectivity, scale,
        >>>     grid_stride, PRINT_STEPS=0)
        >>> print(labels)
        >>> print(image)
        >>> assert (labels > 0).sum() == 3

    Example:
        >>> # Check dtypes
        >>> from geowatch.utils.util_kwimage import *  # NOQA
        >>> dtypes = [np.uint8, np.float32, np.float64, int, np.int16, np.uint16]
        >>> failed = []
        >>> for dtype in dtypes:
        ...    image = (np.random.rand(32, 32) * 512).astype(dtype)
        ...    try:
        ...        find_samecolor_regions(image, min_region_size=10)
        ...    except Exception:
        ...        failed.append(dtype)
        >>> print(f'failed={failed}')
        >>> assert len(failed) == 0

    Example:
        >>> # Check specifying valid values
        >>> from geowatch.utils.util_kwimage import *  # NOQA
        >>> w, h = 32, 32
        >>> image = np.zeros((h, w), dtype=np.uint8)
        >>> image[0:8, :] = 1
        >>> image[8:16, :] = 2
        >>> image[16:32, :] = 3
        >>> image[:, 16:32] = 4
        >>> image[24:32, :] = 1
        >>> image[2, 3] = 5
        >>> image[7, 11] = 13
        >>> image[17, 19] = 23
        >>> image[29, 31] = 37
        >>> #image = kwimage.imresize(image, dsize=(256, 256), interpolation='lanczos')
        >>> min_region_size = 2
        >>> seed_method = 'grid'
        >>> connectivity = 8
        >>> scale = 1.0
        >>> grid_stride = 1
        >>> labels1 = find_samecolor_regions(
        >>>     image, min_region_size, seed_method, connectivity, scale,
        >>>     grid_stride, PRINT_STEPS=0)
        >>> labels2 = find_samecolor_regions(
        >>>     image, min_region_size, seed_method, connectivity, scale,
        >>>     grid_stride, PRINT_STEPS=0, values={1})
        >>> #labels = find_samecolor_regions(image, seed_method='variance')
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> import kwarray
        >>> kwplot.imshow(image, pnum=(1, 3, 1), title='input image', cmap='viridis')
        >>> kwplot.imshow(colorize_label_image(labels1), pnum=(1, 3, 2), title='values=None')
        >>> kwplot.imshow(colorize_label_image(labels2), pnum=(1, 3, 3), title='values={1}')

    Returns:
        ndarray: a label array where 0 indicates background and a
            non-zero label is a samecolor region.

    TODO:
        - [ ] Could generalize this to search for values within a tolerence, in
        which case we would get a fuzzyfind sort of function.

    Note:
        this is a kwimage candidate

    Optimizing:
        >>> import xdev
        >>> xdev.profile_now(find_lowvariance_regions)(image)
        >>> xdev.profile_now(find_samecolor_regions)(image)
        >>> find_samecolor_regions(image)
        >>> #
        >>> import timerit
        >>> ti = timerit.Timerit(30, bestof=3, verbose=2)
        >>> for timer in ti.reset('find_lowvariance_regions'):
        >>>     with timer:
        >>>         find_lowvariance_regions(image)
        >>> #
        >>> ti = timerit.Timerit(30, bestof=3, verbose=2)
        >>> for timer in ti.reset('find_samecolor_regions'):
        >>>     with timer:
        >>>         labels = find_samecolor_regions(image)
        >>> #
        >>> # Test to see the overhead compared to different levels of downscale / upscale
        >>> ti = timerit.Timerit(30, bestof=3, verbose=2)
        >>> for timer in ti.reset('find_samecolor_regions + resize'):
        >>>     with timer:
        >>>         labels = find_samecolor_regions(image, scale=0.5)
        >>> #
        >>> # Test to see the overhead compared to different levels of downscale / upscale
        >>> ti = timerit.Timerit(30, bestof=3, verbose=2)
        >>> for timer in ti.reset('find_samecolor_regions + resize'):
        >>>     with timer:
        >>>         labels = find_samecolor_regions(image, scale=0.25)

        >>> from geowatch.utils.util_kwimage import *  # NOQA
        >>> import kwimage
        >>> image = kwimage.grab_test_image('amazon', dsize=(512, 512))[..., 0:1]
        >>> poly = kwimage.Polygon.random().scale(image.shape[0:2][::-1])
        >>> image = poly.fill(image, value=0)
        >>> mask = (np.random.rand(*image.shape[0:2]) > 0.999)
        >>> rs, cs = np.where(mask)
        >>> image[rs[0:5], cs[0:5]] = 0
        >>> ti = timerit.Timerit(2, bestof=2, verbose=2)
        >>> for timer in ti.reset('find_samecolor_regions, with values'):
        >>>     labels = find_samecolor_regions(image, values={0})
        >>> for timer in ti.reset('find_samecolor_regions, with values, seed_method=values'):
        >>>     labels = find_samecolor_regions(image, values={0}, seed_method='values')
        >>> for timer in ti.reset('find_samecolor_regions, with values, seed_method=variance'):
        >>>     labels = find_samecolor_regions(image, values={0}, seed_method='variance')
        >>> for timer in ti.reset('find_samecolor_regions, without values'):
        >>>     labels = find_samecolor_regions(image, values=None)
    """
    import cv2
    import kwimage

    if scale != 1.0:
        assert 0 < scale <= 1, 'scale should be in the range (0, 1]'
        orig_dsize = image.shape[0:2][::-1]
        image = kwimage.imresize(image, scale=scale, interpolation='nearest')

    values_of_interest = values
    h, w = image.shape[0:2]

    # if len(image.shape) > 2:
    #     num_channels = image.shape[2]
    #     if num_channels not in {1, 3}:
    #         raise Exception(f'Need 1 or 3 channels, got {num_channels}')

    # floodFill only accepts uint8 and float32, we we need to cast to float
    # here for other data types.

    if 0:
        from kwimage.im_cv2 import _cv2_input_fixer_v2
        image, final_dtype = _cv2_input_fixer_v2(
            image, allowed_types='uint8,float32', contiguous=True)
    else:
        if image.dtype.kind == 'f':
            if image.dtype.itemsize != 4:
                image = image.astype(np.float32)
        else:
            if (image.dtype.kind != 'u') or (image.dtype.itemsize != 1):
                image = image.astype(np.float32)

        if not image.flags['C_CONTIGUOUS'] or not image.flags['OWNDATA']:
            # Cv2 only likes certain types of numpy arrays
            image = np.ascontiguousarray(image).copy()

    # Enumerate a set of pixel positions that we will try to flood fill.
    if seed_method == 'grid':
        # Seed method, uniform grid
        # This method is a lot faster, but it will miss any component
        # that a sampling point doesn't land on.
        if grid_stride == 'auto':
            # stride = int(np.ceil(min_region_size / 2))
            # stride = int(np.ceil(min_region_size / 2))
            stride = int(np.ceil(np.sqrt(min_region_size)))
        else:
            stride = grid_stride
        x_grid = np.arange(0, w, stride)
        y_grid = np.arange(0, h, stride)
        x_locs, y_locs = np.meshgrid(x_grid, y_grid)
        x_locs = x_locs.ravel()
        y_locs = y_locs.ravel()
        check_xy = np.stack([x_locs, y_locs], axis=1)
    elif seed_method == 'variance':
        # Seed method, low variance
        ksize = int(np.ceil(np.sqrt(min_region_size)))
        ksize = ksize + (1 - (ksize % 2))
        kernel = (ksize, ksize)
        seed_labels = find_lowvariance_regions(image, kernel)
        unique_labels, unique_pos = np.unique(seed_labels, return_index=True)
        seed_y, seed_x = np.unravel_index(unique_pos, seed_labels.shape)
        seed_xy = np.stack([seed_x, seed_y], axis=1)
        check_xy = seed_xy[unique_labels > 0]
        # return seed_labels
    elif seed_method == 'values':
        # If specific value are given, just use them
        check_mask = kwarray.isect_flags(image, values_of_interest)
        check_mask = exactly_1channel(check_mask)
        check_xy = np.stack(np.where(check_mask), axis=1)[:, ::-1]
    else:
        raise KeyError(seed_method)

    # Initialize the floodfill mask and our output labels
    accum_labels = np.zeros((h + 2, w + 2), dtype=np.uint8)
    mask = accum_labels.copy()
    mask[0, :] = 1
    mask[-1, :] = 1
    mask[:, 0] = 1
    mask[:, -1] = 1

    # Initialize floodfill flags
    ff_flags_base = 0
    ff_flags_base |= connectivity
    ff_flags_base |= cv2.FLOODFILL_FIXED_RANGE  # only consider difference between the seed and the point to be filled
    ff_flags_base |= cv2.FLOODFILL_MASK_ONLY

    prev_mask = mask.copy()

    if PRINT_STEPS:
        import rich
        regions_found = 0

    if values_of_interest is not None and seed_method != 'values':
        # Filter out any grid positions based on values of interest
        grid_values = image[check_xy.T[1], check_xy.T[0]]
        grid_values = kwarray.atleast_nd(grid_values, n=2)
        flags = None
        for v in values_of_interest:
            if flags is None:
                flags = (grid_values == v).all(axis=1)
            else:
                flags |= (grid_values == v).all(axis=1)
        check_xy = check_xy[flags]

    def special_grid_xy_iter(check_xy):
        if seed_method == 'values':
            check_xy_ = check_xy.copy()
            check_xy_p1 = check_xy_ + 1
            while len(check_xy_):
                # Execute loop with the current top most position
                yield check_xy_[0]
                # Remove that position
                check_xy_ = check_xy_[1:]
                check_xy_p1 = check_xy_p1[1:]

                # TODO: only do check if the caller sends a message saying we
                # found a big region.

                # Check if other remaining positions can be removed
                keep_flags = accum_labels[check_xy_p1[:, 1], check_xy_p1[:, 0]] == 0
                if not np.all(keep_flags):
                    check_xy_ = check_xy_[keep_flags]
                    check_xy_p1 = check_xy_p1[keep_flags]
        else:
            yield from iter(check_xy)

    # Start at 2 because 1 is used as an internal value
    cluster_label = 2
    # for check_x, check_y in check_xy:
    for check_x, check_y in special_grid_xy_iter(check_xy):
        already_filled = accum_labels[check_y + 1, check_x + 1]

        if PRINT_STEPS:
            print('')
            print('----')
            check_position = np.full_like(image, dtype=str, fill_value='.')
            check_position[check_y, check_x] = 'x'
            if already_filled:
                rich.print(f'seed xys = ({check_x}, {check_y})')
                rich.print('[yellow] already filled')
                rich.print(ub.hzcat(list(map(str, ['\n' + str(image), ' ', '\n' + str(check_position), ' ', mask, ' ', accum_labels]))))

        if not already_filled:
            seed_point = (check_x, check_y)
            # The value of the mask is specified in the flags Note: we can only
            # handle 254 different regions, which should be fine, but its a
            # limitaiton (we could work around it if needed)
            ff_flags = ff_flags_base | (cluster_label << 8)

            num, im, mask, rect = cv2.floodFill(
                image, mask=mask, seedPoint=seed_point, newVal=1, loDiff=0, upDiff=0,
                # rect=None,
                flags=ff_flags)

            fx, fy, fw, fh = rect
            sl = (slice(fy, fy + fh + 1), slice(fx, fx + fw + 1))
            # sl = kwimage.Box.coerce(np.array([
            #     [fx, fy, fw + 1, fh + 1]]), 'xywh').to_slice()

            if PRINT_STEPS:
                print('')
                print('----')
                rich.print(f'xy = ({check_x}, {check_y})')
                rich.print(f'num = {num}')
                if num > min_region_size:
                    rich.print('[green] found a region')
                else:
                    rich.print('[red] not a region')
                delta = mask - prev_mask
                rich.print(ub.hzcat(list(map(str, ['\n' + str(image), ' ', '\n' + str(check_position), ' ', mask, ' ', accum_labels, ' ', delta]))))

            if num > min_region_size:
                # use delta to work around an issue where the cluster label is
                # not incremented on every iteration. i.e. if we find a
                # cluster, we would otherwise inadvertently take data from
                # previous non-clusters as they are given the same mask label.
                # Accept this as a cluster of similar colors
                if 1:
                    # Faster method where we only copy data in the filled region
                    mask_part = mask[sl] - prev_mask[sl]
                    label_part = accum_labels[sl]
                    label_part[mask_part == cluster_label] = cluster_label
                else:
                    delta = mask - prev_mask
                    accum_labels[delta == cluster_label] = cluster_label
                cluster_label += 1

            # Update the previous mask
            prev_mask[sl] = mask[sl]

        if PRINT_STEPS and True:
            regions_found += 1
            print(f'regions_found={regions_found}')
            mask = accum_labels[1:-1, 1:-1] > 0
            current_unique_maked_values = np.unique(image[mask])
            print(f'current_unique_maked_values={current_unique_maked_values}')

    final_labels = accum_labels[1:-1, 1:-1]
    if PRINT_STEPS:
        print('Final Labels')
        rich.print(final_labels)
    # is_labeled = final_labels
    # Make labeles start at 1 instead of 2.
    # final_labels[is_labeled] = final_labels[is_labeled] - 1

    if scale != 1.0:
        final_labels = kwimage.imresize(
            final_labels, dsize=orig_dsize, interpolation='nearest')
    return final_labels


def find_high_frequency_values(image, values=None, abs_thresh=0.2,
                               rel_thresh=None):
    """
    Values that appear in the image very often, may be indicative of an
    artifact that we should remove.

    Args:
        values (None | List): the values of interest to find.
            if unspecified, any highly frequent value is flagged.

    Ignore:
        >>> # Without value restriction
        >>> from geowatch.utils.util_kwimage import *  # NOQA
        >>> import kwimage
        >>> image1 = kwimage.grab_test_image(dsize=(256, 256))
        >>> dsize = image1.shape[0:2][::-1]
        >>> poly1 =kwimage.Polygon.random(rng=3).scale(dsize)
        >>> poly2 =kwimage.Polygon.random(rng=2).scale(dsize)
        >>> image2 = image1.copy()[..., 0]
        >>> image2 = poly1.draw_on(image2, color=[0])
        >>> image2 = poly2.draw_on(image2, color=[0])
        >>> with ub.Timer() as t2:
        >>>     mask2 = find_high_frequency_values(image2)
        >>> with ub.Timer() as t3:
        >>>     mask3 = find_samecolor_regions(image2)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(image2, doclf=1, pnum=(1, 3, 1))
        >>> kwplot.imshow(colorize_label_image(mask2), pnum=(1, 3, 2), title=f'find_high_frequency_values: @ {t2.elapsed:0.4}s')
        >>> kwplot.imshow(colorize_label_image(mask3), pnum=(1, 3, 3), title=f'find_samecolor_regions @ {t3.elapsed:0.4}s')
        >>> kwplot.show_if_requested()

    Ignore:
        >>> # With value restriction
        >>> from geowatch.utils.util_kwimage import *  # NOQA
        >>> import kwimage
        >>> image1 = kwimage.grab_test_image(dsize=(256, 256))
        >>> dsize = image1.shape[0:2][::-1]
        >>> poly1 =kwimage.Polygon.random(rng=3).scale(dsize)
        >>> poly2 =kwimage.Polygon.random(rng=2).scale(dsize)
        >>> image2 = image1.copy()[..., 0]
        >>> image2 = poly1.draw_on(image2, color=[0])
        >>> image2 = poly2.draw_on(image2, color=[0])
        >>> with ub.Timer() as t2:
        >>>     mask2 = find_high_frequency_values(image2, values={0})
        >>> with ub.Timer() as t3:
        >>>     mask3 = find_samecolor_regions(image2, values={0})
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(image2, doclf=1, pnum=(1, 3, 1))
        >>> kwplot.imshow(colorize_label_image(mask2), pnum=(1, 3, 2), title=f'find_high_frequency_values: @ {t2.elapsed:0.4}s')
        >>> kwplot.imshow(colorize_label_image(mask3), pnum=(1, 3, 3), title=f'find_samecolor_regions @ {t3.elapsed:0.4}s')
        >>> kwplot.show_if_requested()

    Ignore:
        >>> # With value restriction
        >>> from geowatch.utils.util_kwimage import *  # NOQA
        >>> image = np.random.rand(32, 32) + 1
        >>> values = {0}
        >>> abs_thresh = 0.2
        >>> mask1 = find_high_frequency_values(image, values)
        >>> assert not np.any(mask1)
        >>> image[0:16] = 0
        >>> mask2 = find_high_frequency_values(image, values)
        >>> assert np.all(mask2[:16])
        >>> assert not np.any(mask2[16:])
    """
    def ratios(data):
        return data[:-1] / data[1:]
    import kwarray
    import numpy as np

    values_of_interest = values
    if values_of_interest is not None and len(values_of_interest) == 1:
        # Optimization for a single bad values we care about.
        value_of_interest = ub.peek(values_of_interest)
        flags = (image == value_of_interest)
        if len(flags.shape) > 2:
            axis = tuple(range(2, len(flags.shape)))
            flags = flags.all(axis=axis)
        abs_score = flags.sum() / flags.size
        if abs_thresh is not None and abs_score > abs_thresh:
            mask = flags
        else:
            mask = np.zeros_like(flags)

        if rel_thresh is not None:
            raise NotImplementedError
    else:
        raw_values, raw_counts = np.unique(image, return_counts=True)
        valid_mask = ~np.isnan(raw_values)
        values = raw_values[valid_mask]
        counts = raw_counts[valid_mask]

        if values_of_interest is None:
            max_bad_values = 10
        else:
            max_bad_values = len(values_of_interest) + 1

        ranked_idxs = counts.argsort()[::-1]
        ranked_counts = counts[ranked_idxs[:max_bad_values]]
        ranked_values = values[ranked_idxs[:max_bad_values]]

        abs_score = ranked_counts / image.size
        rel_score = ratios(ranked_counts)
        abs_score = abs_score[:len(rel_score)]
        ranked_values = ranked_values[:len(rel_score)]

        if abs_thresh is not None:
            flags = abs_score > abs_thresh
        else:
            flags = np.zeros(len(abs_score), dtype=bool)

        if rel_thresh is not None:
            flags |= (rel_score > rel_thresh)

        bad_values = ranked_values[flags]
        image = kwarray.atleast_nd(image, 3)

        mask = kwarray.isect_flags(image, bad_values)
        mask = mask.reshape(image.shape)
        mask = mask.any(axis=2)
    return mask


def polygon_distance_transform(poly, shape):
    """
    The API needs work, but I think the idea could be useful

    Args:
        poly (kwimage.Polygon): polygon to create distance weights for
        shape (Tuple[int, int]): size of canvas to draw onto

    Returns:
        Tuple[ndarray, ndarray] -
            dist - pixels inside the polygon contain the distance to the edge of the polygon.
            poly_mask - a binary mask where 1s indicate where the polygon is.

    Example:
        >>> from geowatch.utils.util_kwimage import *  # NOQA
        >>> import cv2
        >>> import kwimage
        >>> poly = kwimage.Polygon.random().scale(32)
        >>> shape = (32, 32)
        >>> dist, poly_mask = polygon_distance_transform(poly, shape)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(dist, cmap='viridis', doclf=1, pnum=(1, 2, 1), title='distance weights')
        >>> poly.draw(fill=0, border=1)
        >>> kwplot.imshow(poly_mask.astype(np.float32), pnum=(1, 2, 2), title='poly-mask')
    """
    import cv2
    poly_mask = np.zeros(shape, dtype=np.uint8)
    poly_mask = poly.fill(poly_mask, value=1)
    dist = cv2.distanceTransform(
        src=poly_mask, distanceType=cv2.DIST_L2, maskSize=3)
    return dist, poly_mask


def multiple_polygon_distance_transform_weighting(polys, shape):
    """
    Does a distance tranform on multiple polygons independently and then
    combines their weights such that each pixels uses the maximum distance
    to a polygon it is contained in.

    Args:
        polys (list[kwimage.Polygon]): polygons to draw.
        shape (Tuple[int, int]): size of canvas to draw onto

    Returns:
        Tuple[ndarray, ndarray] -
            dist - pixels inside the polygon contain the distance to the edge of the polygon.
            poly_mask - a binary mask where 1s indicate where the polygon is.

    CommandLine:
        xdoctest -m geowatch.utils.util_kwimage multiple_polygon_distance_transform_weighting --show

    Example:
        >>> from geowatch.utils.util_kwimage import *  # NOQA
        >>> import kwimage
        >>> poly1 = kwimage.Polygon.random(rng=0).scale(32)
        >>> poly2 = poly1.translate((5, 5))
        >>> poly3 = poly2.translate((5, 5))
        >>> poly4 = poly3.translate((5, 5))
        >>> poly5 = poly4.translate((5, 5))
        >>> polys = [poly1, poly2, poly3, poly4, poly5]
        >>> shape = (32, 32)
        >>> dist, poly_mask = multiple_polygon_distance_transform_weighting(polys, shape)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(dist, cmap='viridis', doclf=1, pnum=(1, 2, 1), title='distance weights')
        >>> for poly in polys:
        >>>     poly.draw(fill=0, border=1)
        >>> kwplot.imshow(poly_mask.astype(np.float32), pnum=(1, 2, 2), title='poly-mask')
        >>> kwplot.show_if_requested()
    """
    dist_accum = np.zeros(shape, dtype=np.float32)
    poly_accum = np.zeros(shape, dtype=np.uint8)
    for poly in polys:
        dist, poly_mask = polygon_distance_transform(poly, shape)
        max_dist = dist.max()
        if max_dist > 0:
            dist_weight = dist / max_dist
            poly_accum = np.maximum(poly_accum, poly_mask, out=poly_accum)
            dist_accum = np.maximum(dist_weight, dist_accum, out=dist_accum)

    dist, poly_mask = dist_accum, poly_accum
    return dist, poly_mask


def devcheck_frame_poly_weights(poly, shape, dtype=np.uint8):
    """
    import kwimage
    import kwplot
    kwplot.autompl()
    from geowatch.utils import util_kwimage
    space_shape = (380, 380)
    weights1 = util_kwimage.upweight_center_mask(space_shape)
    weights2 = kwarray.normalize(kwimage.gaussian_patch(space_shape))
    sigma3 = 4.8 * ((space_shape[0] - 1) * 0.5 - 1) + 0.8
    weights3 = kwarray.normalize(kwimage.gaussian_patch(space_shape, sigma=sigma3))

    min_spacetime_weight = 0.5

    weights1 = np.maximum(weights1, min_spacetime_weight)
    weights2 = np.maximum(weights2, min_spacetime_weight)
    weights3 = np.maximum(weights3, min_spacetime_weight)

    # Hack so color bar goes to 0
    weights3[0, 0] = 0
    weights2[0, 0] = 0
    weights1[0, 0] = 0

    kwplot.imshow(weights1, pnum=(1, 3, 1), title='current', cmap='viridis', data_colorbar=1)
    kwplot.imshow(weights2, pnum=(1, 3, 2), title='variant1', cmap='viridis', data_colorbar=1)
    kwplot.imshow(weights3, pnum=(1, 3, 3), title='variant2', cmap='viridis', data_colorbar=1)
    """
    import kwimage
    space_shape = (128, 128)
    space_dsize = space_shape[::-1]
    polys = [
        kwimage.Polygon.random().scale(space_dsize).scale(0.25, about='center'),
        kwimage.Polygon.random().scale(space_dsize).scale(0.25, about='center'),
        kwimage.Polygon.random().scale(space_dsize).scale(0.25, about='center'),
        kwimage.Polygon.random().scale(space_dsize).scale(0.25, about='center'),
    ]

    frame_poly_weights_v1 = np.ones(space_shape, dtype=np.float32)
    frame_poly_weights_v2 = np.zeros(space_shape, dtype=np.float32)
    for poly in polys:
        dist, poly_mask = polygon_distance_transform(poly, space_shape)
        max_dist = dist.max()
        if max_dist > 0:
            dist_weight = dist / max_dist
            weight_mask = dist_weight + (1 - poly_mask)
            frame_poly_weights_v1 = frame_poly_weights_v1 * weight_mask
            frame_poly_weights_v2 = np.maximum(frame_poly_weights_v2, dist_weight)

    sigma = (
        (4.8 * ((space_shape[1] - 1) * 0.5 - 1) + 0.8),
        (4.8 * ((space_shape[0] - 1) * 0.5 - 1) + 0.8),
    )
    min_spacetime_weight = 0.5
    frame_poly_weights = frame_poly_weights_v2
    frame_poly_weights = np.maximum(frame_poly_weights, min_spacetime_weight)
    space_weights = kwarray.normalize(kwimage.gaussian_patch(space_shape, sigma=sigma))
    import kwplot
    kwplot.autompl()
    kwplot.imshow(frame_poly_weights_v1, pnum=(1, 3, 1))
    kwplot.imshow(frame_poly_weights, pnum=(1, 3, 2))
    kwplot.imshow(np.maximum(frame_poly_weights, space_weights), pnum=(1, 3, 3))


def find_low_overlap_covering_boxes(polygons, scale, min_box_dim, max_box_dim,
                                    merge_thresh=0.001, max_iters=100,
                                    verbose=1):
    """
    Given a set of polygons we want to find a small set of boxes that
    completely cover all of those polygons.

    We are going to do some set-cover shenanigans by making a bunch of
    candidate boxes based on some hueristics and find a set cover of those.

    Then we will search for small boxes that can be merged, and iterate.

    Args:
        polygons (List[Polygon): the input shapes that need clustering
        scale (float): scale factor for context we want around each polygon.
        min_box_dim (float): minimum side length of a returned box
        max_box_dim (float): maximum side length of a returned box

    Returns:
        Tuple[Boxes, List[ndarray]]:
            keep_bbs: The chosen boxes that cover the inputs
            overlap_idxs: Corresponding list indicating which of the original
                inputs overlaps the each covering box.


    References:
        https://aip.scitation.org/doi/pdf/10.1063/1.5090003?cookieSet=1
        Mercantile - https://pypi.org/project/mercantile/0.4/
        BingMapsTiling - XYZ Tiling for webmap services
        https://mercantile.readthedocs.io/en/stable/api/mercantile.html#mercantile.bounding_tile
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.713.6709&rep=rep1&type=pdf

    Example:
        >>> # Create random polygons as test data
        >>> from geowatch.utils.util_kwimage import *  # NOQA
        >>> import kwimage
        >>> import kwarray
        >>> from kwarray import distributions
        >>> rng = kwarray.ensure_rng(934602708841)
        >>> num = 200
        >>> #
        >>> canvas_width = 2000
        >>> offset_distri = distributions.Uniform(canvas_width, rng=rng)
        >>> scale_distri = distributions.Uniform(10, 150, rng=rng)
        >>> #
        >>> polygons = []
        >>> for _ in range(num):
        >>>     poly = kwimage.Polygon.random(rng=rng)
        >>>     poly = poly.scale(scale_distri.sample())
        >>>     poly = poly.translate(offset_distri.sample(2))
        >>>     polygons.append(poly)
        >>> polygons = kwimage.PolygonList(polygons)
        >>> #
        >>> scale = 1.0
        >>> min_box_dim = 240
        >>> max_box_dim = 500
        >>> #
        >>> keep_bbs, overlap_idxs = find_low_overlap_covering_boxes(polygons, scale, min_box_dim, max_box_dim)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> kwplot.figure(fnum=1, doclf=1)
        >>> polygons.draw(color='pink')
        >>> keep_bbs.draw(color='orange', setlim=1)
        >>> plt.gca().set_title('find_low_overlap_covering_boxes')

    Example:
        >>> # Empty test case
        >>> from geowatch.utils.util_kwimage import *  # NOQA
        >>> keep_bbs, overlap_idxs = find_low_overlap_covering_boxes([], 1, 0, 1)
        >>> assert len(keep_bbs) == 0
        >>> assert len(overlap_idxs) == 0
    """
    import kwimage
    import kwarray
    import numpy as np
    import geopandas as gpd
    import ubelt as ub
    from kwgis.utils import util_gis
    import networkx as nx

    if len(polygons) == 0:
        empty_boxes = kwimage.Boxes(np.empty((0, 4)), 'xywh')
        empty_ixs = []
        return empty_boxes, empty_ixs

    polygons_sh = [p.to_shapely() for p in polygons]
    polygons_gdf = gpd.GeoDataFrame(geometry=polygons_sh)

    polybbs = kwimage.Boxes.concatenate([p.to_boxes() for p in polygons])
    initial_candiate_bbs = polybbs.scale(scale, about='center')
    initial_candiate_bbs = initial_candiate_bbs.to_cxywh()
    initial_candiate_bbs.data[..., 2] = np.maximum(initial_candiate_bbs.data[..., 2], min_box_dim)
    initial_candiate_bbs.data[..., 3] = np.maximum(initial_candiate_bbs.data[..., 3], min_box_dim)

    candidate_bbs = initial_candiate_bbs

    def refine_candidates(candidate_bbs, iter_idx):
        # Add some translated boxes to the mix to see if they do any better
        extras = [
            candidate_bbs.translate((-min_box_dim / 10, 0)),
            candidate_bbs.translate((+min_box_dim / 10, 0)),
            candidate_bbs.translate((0, -min_box_dim / 10)),
            candidate_bbs.translate((0, +min_box_dim / 10)),
            candidate_bbs.translate((-min_box_dim / 3, 0)),
            candidate_bbs.translate((+min_box_dim / 3, 0)),
            candidate_bbs.translate((0, -min_box_dim / 3)),
            candidate_bbs.translate((0, +min_box_dim / 3)),
        ]
        candidate_bbs = kwimage.Boxes.concatenate([candidate_bbs] + extras)

        # Find the minimum boxes that cover all of the regions
        # xs, ys = centroids.T
        # ws = hs = np.full(len(xs), fill_value=site_meters)
        # utm_boxes = kwimage.Boxes(np.stack([xs, ys, ws, hs], axis=1), 'cxywh').to_xywh()

        boxes_gdf = gpd.GeoDataFrame(geometry=candidate_bbs.to_shapely(), crs=polygons_gdf.crs)
        box_poly_overlap = util_gis.geopandas_pairwise_overlaps(boxes_gdf, polygons_gdf, predicate='contains')
        cover_idxs = list(kwarray.setcover(box_poly_overlap).keys())
        keep_bbs = candidate_bbs.take(cover_idxs)
        box_ious = keep_bbs.ious(keep_bbs)

        if iter_idx > 0:
            # Dont do it on the first iter to compare to old algo
            laplace = box_ious - np.diag(np.diag(box_ious))
            mergable = laplace > merge_thresh
            g = nx.Graph()
            g.add_edges_from(list(zip(*np.where(mergable))))
            cliques = sorted(nx.find_cliques(g), key=len)[::-1]

            used = set()
            merged_boxes = []
            for clique in cliques:
                if used & set(clique):
                    continue

                new_box = keep_bbs.take(clique).bounding_box()
                w = new_box.width.ravel()[0]
                h = new_box.height.ravel()[0]
                if w < max_box_dim and h < max_box_dim:
                    merged_boxes.append(new_box)
                    used.update(clique)

            unused = sorted(set(range(len(keep_bbs))) - used)
            post_merge_bbs = kwimage.Boxes.concatenate([keep_bbs.take(unused)] + merged_boxes)

            boxes_gdf = gpd.GeoDataFrame(geometry=post_merge_bbs.to_shapely(), crs=polygons_gdf.crs)
            box_poly_overlap = util_gis.geopandas_pairwise_overlaps(boxes_gdf, polygons_gdf, predicate='contains')
            cover_idxs = list(kwarray.setcover(box_poly_overlap).keys())
            new_cand_bbs = post_merge_bbs.take(cover_idxs)
        else:
            new_cand_bbs = keep_bbs

        new_cand_overlaps = list(ub.take(box_poly_overlap, cover_idxs))
        return new_cand_bbs, new_cand_overlaps

    new_cand_overlaps = None

    for iter_idx in range(max_iters):
        old_candidate_bbs = candidate_bbs
        candidate_bbs, new_cand_overlaps = refine_candidates(candidate_bbs, iter_idx)
        num_old = len(old_candidate_bbs)
        num_new = len(candidate_bbs)
        if num_old == num_new:
            residual = (old_candidate_bbs.data - candidate_bbs.data).max()
            if residual > 0:
                if verbose:
                    print('improving residual = {}'.format(ub.urepr(residual, nl=1)))
            else:
                if verbose:
                    print('converged')
                break
        else:
            if verbose:
                print(f'improving: {num_old} -> {num_new}')
    else:
        if verbose:
            print('did not converge')
    keep_bbs = candidate_bbs
    overlap_idxs = new_cand_overlaps

    if 0:
        import kwplot
        kwplot.autoplt()
        kwplot.figure(fnum=1, doclf=1)
        polygons.draw(color='pink')
        # candidate_bbs.draw(color='blue', setlim=1)
        keep_bbs.draw(color='orange', setlim=1)

    return keep_bbs, overlap_idxs


def find_low_overlap_covering_boxes_optimize(polygons, scale, min_box_dim, max_box_dim, merge_thresh=0.001, max_iters=100):
    """
    A variant of the covering problem that doesn't work that well, but might in
    the future with tweaks.

    Ignore:
        >>> # Create random polygons as test data
        >>> import kwimage
        >>> import kwarray
        >>> from kwarray import distributions
        >>> rng = kwarray.ensure_rng(934602708841)
        >>> num = 200
        >>> #
        >>> canvas_width = 2000
        >>> offset_distri = distributions.Uniform(canvas_width, rng=rng)
        >>> scale_distri = distributions.Uniform(10, 150, rng=rng)
        >>> #
        >>> polygons = []
        >>> for _ in range(num):
        >>>     poly = kwimage.Polygon.random(rng=rng)
        >>>     poly = poly.scale(scale_distri.sample())
        >>>     poly = poly.translate(offset_distri.sample(2))
        >>>     polygons.append(poly)
        >>> polygons = kwimage.PolygonList(polygons)
        >>> #
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> kwplot.figure(doclf=1)
        >>> plt.gca().set_xlim(0, canvas_width)
        >>> plt.gca().set_ylim(0, canvas_width)
        >>> _ = polygons.draw(fill=0, border=1, color='pink')
        >>> #
        >>> scale = 1.0
        >>> min_box_dim = 240
        >>> max_box_dim = 500
        >>> #

    """
    import kwimage
    # import kwarray

    start_scale = 2.0
    polygon_boxes = kwimage.Boxes.concatenate([p.to_boxes() for p in polygons]).to_ltrb()
    candidate_bbs = polygon_boxes.scale(start_scale, about='center').to_ltrb()
    orig_candidates = candidate_bbs.copy()
    import torch

    device = 'cpu'
    # device = 0
    polygon_ltrb = polygon_boxes.tensor().data.float().to(device)
    candidate_ltrb = torch.nn.Parameter(candidate_bbs.tensor().data.float().to(device))

    # These will be soft bits that will indicate 1 or 0, and we will try to
    # force into an integer solution via rounding.
    indicator_logits = torch.nn.Parameter(
        torch.rand(len(candidate_ltrb), dtype=torch.float, device=candidate_ltrb.device) * 0.2 + 0.8
    )

    parameters = [
        candidate_ltrb,
        indicator_logits,
    ]
    import torch_optimizer as optim
    optimizer = optim.RangerQH(parameters, lr=1e-2)
    # from torch.optim import SGD
    # optimizer = SGD(parameters, lr=1-1, weight_decay=1e-7)
    # from torch.optim import AdamW
    # optimizer = AdamW(parameters, lr=1e-3, weight_decay=1e-6)

    target_boxes = kwimage.Boxes(polygon_ltrb, 'ltrb')
    cover_boxes = kwimage.Boxes(candidate_ltrb, 'ltrb')

    # ltrb1 = target_boxes.data
    # ltrb2 = cover_boxes.data
    # _impl = target_boxes._impl

    # num_targets = len(target_boxes)
    target_area = target_boxes.area.sum()
    # eps = kwarray.dtype_info(target_area.dtype).eps

    denom = target_area

    disatisfaction_penalty = 100

    def forward():
        # TODO: restrict what boxes can cover what objects via grouping to
        # reduce computational complexity here.
        iooa = target_boxes.iooas(cover_boxes)

        areas = target_boxes.area
        self_ious = target_boxes.ious(target_boxes, impl='py')

        indicator_bits = indicator_logits.sigmoid()
        chosen_area = (indicator_bits * (areas / denom)).sum()
        chosen_self_overlap = (indicator_bits * self_ious).sum()

        # We want to minimize...
        objective = (
            # Total chosen area covered
            chosen_area +
            # Overlap of the chosen boxes
            chosen_self_overlap
        )

        # Subject to the constraint (which we relax for optimization)
        relaxed_iooa = iooa * indicator_bits[:, None]

        # TODO: Getting this loss right is the key to this problem.
        # The current version doesn't work that well. But a more numerically
        # stable version might do better.

        # All of the polygons must be completely covered by at least one box
        sat_critical = relaxed_iooa.max(dim=0)[0].min()
        sat_overall = relaxed_iooa.max(dim=0)[0].mean()
        satisfaction = (sat_critical + sat_overall) / 2

        bottom_line_loss = disatisfaction_penalty * (1 - sat_critical)
        overall_sat_loss = disatisfaction_penalty * (1 - sat_overall)
        loss = objective / (satisfaction + 0.01) + bottom_line_loss + overall_sat_loss
        # loss = bottom_line_loss + overall_sat_loss

        outputs = {
            'item_losses': {
                'chosen_area': chosen_area[None, None, ...],
                'chosen_self_overlap': chosen_self_overlap[None, None, ...],
                'sat_critical': sat_critical[None, None, ...],
                'sat_overall': sat_overall[None, None, ...],
                'satisfaction': satisfaction[None, None, ...],
                'total': loss[None, None, ...],
            },
            'loss': loss,
        }
        return outputs

    if 1:
        prog = ub.ProgIter(range(100000))
        for i in prog:
            optimizer.zero_grad()
            outputs = forward()
            loss = outputs['loss']
            loss.backward()
            total_grad = candidate_ltrb.grad.sum()
            mean_grad = total_grad / candidate_ltrb.numel()
            drift = (candidate_ltrb.data - orig_candidates.data).abs().max().item()
            sat_overall = outputs['item_losses']['sat_overall'].sum().item()
            sat_critical = outputs['item_losses']['sat_critical'].sum().item()
            prog.set_extra(f'{loss=} {total_grad=} {mean_grad=} {drift=} {sat_critical=} {sat_overall=}')
            optimizer.step()

    else:

        def draw_batch():
            import kwarray
            indicator_bits = kwarray.ArrayAPI.numpy(indicator_logits.sigmoid())
            orig_candidates.draw(color='red', linewidth=6)
            cover_boxes.numpy().draw(color='blue', setlim=1, alpha=indicator_bits, u=3)
            target_boxes.numpy().draw(color='orange', setlim='grow', linwidth=2)

        import kwplot
        sns = kwplot.autosns()
        plt = kwplot.autoplt()
        kwplot.figure(fnum=1, doclf=1)
        draw_batch()
        plt.gca().set_title('find_low_overlap_covering_boxes')

        import xdev
        fnum = 2
        fig = kwplot.figure(fnum=fnum, doclf=True)
        fig.set_size_inches(15, 6)
        fig.subplots_adjust(left=0.05, top=0.9)
        prev = None
        _frame_idx = 0

        loss_records = []
        loss_records = [g[0] for g in ub.group_items(loss_records, lambda x: x['step']).values()]
        step = 0
        _frame_idx = 0

        for _frame_idx in xdev.InteractiveIter(list(range(_frame_idx + 1, 1000))):
            # for _frame_idx in list(range(_frame_idx, 1000)):
            num_steps = 100
            ex = None
            prog = ub.ProgIter(range(num_steps), desc='optimize')
            for _i in prog:
                optimizer.zero_grad()
                outputs = forward()
                loss = outputs['loss'].sum()
                if torch.any(torch.isnan(loss)):
                    print('NAN OUTPUT!!!')
                    print('loss = {!r}'.format(loss))
                    print('prev = {!r}'.format(prev))
                    ex = Exception('prev = {!r}'.format(prev))
                    break
                # elif loss > 1e4:
                #     # Turn down the learning rate when loss gets huge
                #     scale = (loss / 1e4).detach()
                #     loss /= scale
                prev = loss
                # import torch.utils.data as torch_data
                # default_collate = torch_data.dataloader.default_collate
                # item_losses_ = default_collate(outputs['item_losses'])
                item_losses_ = outputs['item_losses']
                item_losses = ub.map_vals(lambda x: sum(x).item(), item_losses_)
                loss_records.extend([{'part': key, 'val': val, 'step': step} for key, val in item_losses.items()])
                loss.backward()
                total_grad = candidate_ltrb.grad.sum()
                mean_grad = total_grad / candidate_ltrb.numel()
                drift = (candidate_ltrb.data - orig_candidates.data).abs().max().item()
                sat_overall = outputs['item_losses']['sat_overall'].sum().item()
                sat_critical = outputs['item_losses']['sat_critical'].sum().item()
                prog.set_extra(f'{loss=} {total_grad=} {mean_grad=} {drift=} {sat_critical=} {sat_overall=}')
                optimizer.step()
                step += 1

            draw_batch()

            kwplot.figure(pnum=(1, 2, 1), fnum=fnum, docla=1)
            draw_batch()

            fig = kwplot.figure(fnum=fnum, pnum=(1, 2, 2))
            #kwplot.imshow(canvas, pnum=(1, 2, 1))
            import pandas as pd
            df = pd.DataFrame(loss_records)
            total_df = dict(list((df.groupby('part'))))['total']
            print(total_df)
            ax = sns.lineplot(data=total_df, x='step', y='val', hue='part')
            ax
            # ax.set_ylim(0, df.groupby('part')['val'].median().max())
            # try:
            #     ax.set_yscale('logit')
            # except Exception:
            #     ...
            # from kwutil.slugify_ext import smart_truncate
            # from kwplot.mpl_make import render_figure_to_image
            # fig.suptitle(smart_truncate(str(optimizer).replace('\n', ''), max_length=64))
            # img = render_figure_to_image(fig)
            # img = kwimage.convert_colorspace(img, src_space='bgr', dst_space='rgb')
            # fpath = join(dpath, 'frame_{:04d}.png'.format(_frame_idx))
            #kwimage.imwrite(fpath, img)
            xdev.InteractiveIter.draw()
            if ex:
                raise ex

    # polygon_boxes.tensor()
    # boxes_gdf = gpd.GeoDataFrame(geometry=candidate_bbs.to_shapely(), crs=polygons_gdf.crs)
    # box_poly_overlap = util_gis.geopandas_pairwise_overlaps(boxes_gdf, polygons_gdf, predicate='contains')
    # cover_idxs = list(kwarray.setcover(box_poly_overlap).keys())
    # keep_bbs = candidate_bbs.take(cover_idxs)
    # box_ious = keep_bbs.ious(keep_bbs)
    # import pulp
    # prob = pulp.LpProblem("Set Cover", pulp.LpMinimize)


def exactly_1channel(image, ndim=2):
    """
    Like atleast_3channels, exactly_1channel returns a 2D image as either
    a 2D or 3D array, depending on if ndim is 2 or 3. For a 3D array the last
    dimension is always 1.  An error is thrown if assumptions are not met.

    PORTED TO kwimage in version 0.9.14

    Args:
        image (ndarray):
        ndim (int): either 2 or 3

    Example:
        >>> assert exactly_1channel(np.empty((3, 3)), ndim=2).shape == (3, 3)
        >>> assert exactly_1channel(np.empty((3, 3)), ndim=3).shape == (3, 3, 1)
        >>> assert exactly_1channel(np.empty((3, 3, 1)), ndim=2).shape == (3, 3)
        >>> assert exactly_1channel(np.empty((3, 3, 1)), ndim=3).shape == (3, 3, 1)
    """
    if len(image.shape) == 3:
        assert image.shape[2] == 1
        if ndim == 2:
            image = image[:, :, 0]
        else:
            assert ndim == 3
    else:
        assert len(image.shape) == 2
        if ndim == 3:
            image = image[:, :, None]
    return image


def load_image_shape(fpath, backend='auto', include_channels=True):
    """
    Version from kwimage dev/0.9.26

    Determine the height/width/channels of an image without reading the entire
    file.

    Args:
        fpath (str): path to an image
        backend (str | List[str]): can be "auto", "pil", or "gdal".
            Can also be a list of which backends to try in which order.
        include_channels (bool): if False, only reads the height, width.

    Returns:
        Tuple[int, int, int] - shape of the image
            Recall this library uses the convention that "shape" is refers to
            height,width,channels array-style ordering and "size" is
            width,height cv2-style ordering.

    Example:
        >>> from geowatch.utils.util_kwimage import *  # NOQA
        >>> import ubelt as ub
        >>> import kwimage
        >>> dpath = ub.Path.appdir('kwimage/tests', type='cache').ensuredir()
        >>> fpath = dpath / 'foo.tif'
        >>> kwimage.imwrite(fpath, np.random.rand(64, 64, 3))
        >>> shape1 = load_image_shape(fpath, backend=['pil', 'gdal'])
        >>> shape2 = load_image_shape(fpath, backend=['gdal', 'pil'])
        >>> assert shape1 == shape2 == (64, 64, 3)
    """
    import os

    if backend == 'auto':
        backend = ['pil', 'gdal']

    if isinstance(backend, list):
        candidate_errors = []
        success = False
        for candidate_backend in backend:
            if candidate_backend == 'gdal':
                if not _have_gdal():
                    continue
            try:
                shape = load_image_shape(fpath, backend=candidate_backend,
                                         include_channels=include_channels)
            except Exception as ex:
                candidate_errors.append((candidate_backend, ex))
            else:
                success = True
                break
        if not success:
            if len(candidate_errors) == 0:
                raise Exception('Unable to try an candidates')
            else:
                raise candidate_errors[-1]
    elif backend == 'pil':
        # TODO: can we prevent pil from logging to stdout here on failure?
        # This will often print "More samples per pixel than can be decoded"
        # which gives little context and is ultimately not an issue if we can
        # fallback on gdal.
        from PIL import Image
        fpath = os.fspath(fpath)
        with Image.open(fpath) as pil_img:
            width, height = pil_img.size
            if include_channels:
                num_channels = len(pil_img.getbands())
                shape = (height, width, num_channels)
            else:
                shape = (height, width)
    elif backend == 'gdal':
        from osgeo import gdal
        fpath = os.fspath(fpath)
        gdal_dset = gdal.Open(fpath, gdal.GA_ReadOnly)
        if gdal_dset is None:
            raise Exception(gdal.GetLastErrorMsg())
        width = gdal_dset.RasterXSize
        height = gdal_dset.RasterYSize
        if include_channels:
            num_channels = gdal_dset.RasterCount
            shape = (height, width, num_channels)
        else:
            shape = (height, width)
        gdal_dset = None
    elif backend == 'imagesize':
        import imagesize
        if include_channels:
            raise NotImplementedError('no way to get number of channels with imagesize')
        width, height = imagesize.get(fpath)
        shape = (height, width)
    else:
        raise KeyError(backend)
    return shape


def _have_gdal():
    try:
        from osgeo import gdal  # NOQA
    except Exception:
        return False
    else:
        return True


def draw_multiclass_clf_on_image(im, classes, probs=None, true_ohe=None, top_k=3, border=1):
    """
    Draws multiclass classification label on an image.

    Works best with image chips sized between 200x200 and 500x500

    Args:
        im (ndarray): the image
        classes (Sequence[str] | kwcoco.CategoryTree): list of class names
        true_ohe (int): true class indicator vector
        probs (ndarray): predicted class probs for each class

    Ignore:
        from geowatch.utils.util_kwimage import *  # NOQA
        im = None
        classes = ['dog', 'cat', 'beagle', 'boxer', 'tabby', 'ragdoll', 'gooddog', 'baddog']
        probs = [0.8, 0.3, 0.2, 0.8, 0.2, 0.2, 0.1, 0.9]
        true_ohe = [1, 0, 1, 0, 0, 0, 1, 0]
        border = 1
        canvas = draw_multiclass_clf_on_image(im, classes, probs, true_ohe)

        import kwplot
        kwplot.autompl()
        kwplot.imshow(canvas)
    """
    import kwimage
    import kwarray

    if true_ohe is not None:
        true_ohe = kwarray.ArrayAPI.numpy(true_ohe)
        true_idxs = np.where(true_ohe)[0]

    lines = []

    toshow_class_idxs = []
    if probs is not None:
        probs = kwarray.ArrayAPI.numpy(probs)
        top_pred_idxs = probs.argsort()[::-1][:top_k]
        missing_true_idxs = sorted(set(true_idxs) - set(top_pred_idxs))
        toshow_class_idxs.extend(top_pred_idxs)
        toshow_class_idxs.extend(missing_true_idxs)

    for cidx in toshow_class_idxs:
        class_name = classes[cidx]
        pred_score = probs[cidx]
        is_true = true_ohe[cidx]
        if is_true:
            label = (f't:{class_name}@{pred_score:.2f}: {is_true}')
        else:
            label = (f'p:{class_name}@{pred_score:.2f}: {is_true}')
        lines.append(label)
    text = '\n'.join(lines)

    fontkw = {
        'fontScale': 1.0,
        'thickness': 2
    }
    # color = 'dodgerblue' if pcx == tcx else 'orangered'
    if im is not None:
        im_ = kwimage.atleast_3channels(im)
        # w, h = im.shape[0:2][::-1]
    else:
        im_ = None

    org2 = np.array((2, 5))
    canvas = kwimage.draw_text_on_image(im_, text, org=org2, color='kitware_green',
                                        valign='top', border=border, **fontkw)
    return canvas
