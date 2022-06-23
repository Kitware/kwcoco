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


TODO:
    - [x] Need to handle masks / nodata values when warping. Might need to
          rely more on gdal / rasterio for this.


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
    >>> from kwcoco.util.delayed_poc import *  # NOQA
    >>> import ubelt as ub
    >>> import numpy as np
    >>> import kwimage
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
    >>>     f1_chan1.delayed_warp(kwimage.Affine.scale(f1_dsize / f1_chan1.dsize), dsize=f1_dsize),
    >>>     f1_chan2.delayed_warp(kwimage.Affine.scale(f1_dsize / f1_chan2.dsize), dsize=f1_dsize),
    >>>     f1_chan3.delayed_warp(kwimage.Affine.scale(f1_dsize / f1_chan3.dsize), dsize=f1_dsize),
    >>> ])
    >>> f2_img = DelayedChannelConcat([
    >>>     f2_chan1.delayed_warp(kwimage.Affine.scale(f2_dsize / f2_chan1.dsize), dsize=f2_dsize),
    >>>     f2_chan2.delayed_warp(kwimage.Affine.scale(f2_dsize / f2_chan2.dsize), dsize=f2_dsize),
    >>>     f2_chan3.delayed_warp(kwimage.Affine.scale(f2_dsize / f2_chan3.dsize), dsize=f2_dsize),
    >>> ])
    >>> # Combine frames into a video
    >>> vid_dsize = np.array((280, 280))
    >>> vid = DelayedFrameConcat([
    >>>     f1_img.delayed_warp(kwimage.Affine.scale(vid_dsize / f1_img.dsize), dsize=vid_dsize),
    >>>     f2_img.delayed_warp(kwimage.Affine.scale(vid_dsize / f2_img.dsize), dsize=vid_dsize),
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
    >>> import kwimage
    >>> import ubelt as ub
    >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
    >>> delayed = dset.delayed_load(1)
    >>> from kwcoco.util.delayed_poc import *  # NOQA
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
    >>> msi_warp = msi_crop.delayed_warp(kwimage.Affine.scale(3), dsize='auto')
    >>> subdata = msi_warp.take_channels('B11|B1')
    >>> final = subdata.finalize()
    >>> assert final.shape == (210, 60, 2)


Example:
    >>> # test case where an auxiliary image does not map entirely on the image.
    >>> from kwcoco.util.delayed_poc import *  # NOQA
    >>> import kwimage
    >>> import kwcoco
    >>> from os.path import join
    >>> import ubelt as ub
    >>> import numpy as np
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
    >>> c1 = kwcoco.FusedChannelSpec.coerce('c1')
    >>> c2 = kwcoco.FusedChannelSpec.coerce('c2')
    >>> c3 = kwcoco.FusedChannelSpec.coerce('c2')
    >>> aux1 = DelayedLoad(chan1_fpath, dsize=chan1_raw.shape[0:2][::-1], channels=c1, num_bands=1)
    >>> aux2 = DelayedLoad(chan2_fpath, dsize=chan2_raw.shape[0:2][::-1], channels=c2, num_bands=1)
    >>> aux3 = DelayedLoad(chan3_fpath, dsize=chan3_raw.shape[0:2][::-1], channels=c3, num_bands=1)
    >>> #
    >>> img_dsize = (128, 128)
    >>> transform1 = kwimage.Affine.coerce(scale=0.5)
    >>> transform2 = kwimage.Affine.coerce(theta=0.5, shearx=0.01, offset=(-20, -40))
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


__mkinit__ = """
mkinit -m kwcoco.util.delayed_poc
"""


from kwcoco.util.delayed_poc import delayed_base
from kwcoco.util.delayed_poc import delayed_leafs
from kwcoco.util.delayed_poc import delayed_nodes

from kwcoco.util.delayed_poc.delayed_base import (DelayedImageOperation,
                                                  DelayedVideoOperation,
                                                  DelayedVisionOperation,)
from kwcoco.util.delayed_poc.delayed_leafs import (DelayedLoad, DelayedNans,
                                                   DelayedIdentity, dequantize)
from kwcoco.util.delayed_poc.delayed_nodes import (DelayedChannelConcat,
                                                   DelayedCrop,
                                                   DelayedFrameConcat,
                                                   DelayedWarp, JaggedArray,)

__all__ = ['DelayedChannelConcat', 'DelayedCrop', 'DelayedFrameConcat',
           'DelayedImageOperation', 'DelayedLoad', 'DelayedNans',
           'DelayedVideoOperation', 'DelayedVisionOperation', 'DelayedWarp',
           'JaggedArray', 'delayed_base', 'delayed_leafs', 'delayed_nodes',
           'dequantize', 'DelayedIdentity']
