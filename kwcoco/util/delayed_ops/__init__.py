r"""
A rewrite of the delayed operations

Note:
    The classes in this submodule will have their names changed when the old
    POC delayed operations are deprecated.

TODO:
    The optimize logic could likley be better expressed as some sort of
    AST transformer.

Example:
    >>> # xdoctest: +REQUIRES(module:gdal)
    >>> from kwcoco.util.delayed_ops import *  # NOQA
    >>> import kwimage
    >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
    >>> dimg = DelayedLoad2(fpath, channels='r|g|b').prepare()
    >>> quantization = {'quant_max': 255, 'nodata': 0}
    >>> #
    >>> # Make a complex chain of operations
    >>> dimg = dimg.dequantize(quantization)
    >>> dimg = dimg.warp({'scale': 1.1})
    >>> dimg = dimg.warp({'scale': 1.1})
    >>> dimg = dimg[0:400, 1:400]
    >>> dimg = dimg.warp({'scale': 0.5})
    >>> dimg = dimg[0:800, 1:800]
    >>> dimg = dimg.warp({'scale': 0.5})
    >>> dimg = dimg[0:800, 1:800]
    >>> dimg = dimg.warp({'scale': 0.5})
    >>> dimg = dimg.warp({'scale': 1.1})
    >>> dimg = dimg.warp({'scale': 1.1})
    >>> dimg = dimg.warp({'scale': 2.1})
    >>> dimg = dimg[0:200, 1:200]
    >>> dimg = dimg[1:200, 2:200]
    >>> dimg.write_network_text()
    ╙── Crop dsize=(128,130),space_slice=(slice(1,200,None),slice(2,200,None))
        └─╼ Crop dsize=(130,131),space_slice=(slice(0,200,None),slice(1,200,None))
            └─╼ Warp dsize=(131,131),transform={scale=2.1},antialias=True,interpolation=linear
                └─╼ Warp dsize=(62,62),transform={scale=1.1},antialias=True,interpolation=linear
                    └─╼ Warp dsize=(56,56),transform={scale=1.1},antialias=True,interpolation=linear
                        └─╼ Warp dsize=(50,50),transform={scale=0.5},antialias=True,interpolation=linear
                            └─╼ Crop dsize=(99,100),space_slice=(slice(0,800,None),slice(1,800,None))
                                └─╼ Warp dsize=(100,100),transform={scale=0.5},antialias=True,interpolation=linear
                                    └─╼ Crop dsize=(199,200),space_slice=(slice(0,800,None),slice(1,800,None))
                                        └─╼ Warp dsize=(200,200),transform={scale=0.5},antialias=True,interpolation=linear
                                            └─╼ Crop dsize=(399,400),space_slice=(slice(0,400,None),slice(1,400,None))
                                                └─╼ Warp dsize=(621,621),transform={scale=1.1},antialias=True,interpolation=linear
                                                    └─╼ Warp dsize=(564,564),transform={scale=1.1},antialias=True,interpolation=linear
                                                        └─╼ Dequantize dsize=(512,512),quantization={quant_max=255,nodata=0}
                                                            └─╼ Load channels=<FusedChannelSpec(r|g|b)>,num_channels=3,dsize=(512,512),fpath=.../demodata/astro_overviews=3.tif,num_overviews=0
    >>> # Optimize the chain
    >>> dopt = dimg.optimize()
    >>> dopt.write_network_text()
    ╙── Warp dsize=(128,130),transform={offset=(-0.227133...,-0.231347...),scale=0.384326...},antialias=True,interpolation=linear
        └─╼ Dequantize dsize=(318,329),quantization={quant_max=255,nodata=0}
            └─╼ Crop dsize=(318,329),space_slice=(slice(2,331,None),slice(13,331,None))
                └─╼ Load channels=<FusedChannelSpec(r|g|b)>,num_channels=3,dsize=(512,512),fpath=.../demodata/astro_overviews=3.tif,num_overviews=0
    >>> final0 = dimg.finalize()
    >>> final1 = dopt.finalize()
    >>> # xdoctest: +REQUIRES(--show)
    >>> import kwplot
    >>> kwplot.autompl()
    >>> kwplot.imshow(final0, pnum=(1, 2, 1), fnum=1, title='raw')
    >>> kwplot.imshow(final1, pnum=(1, 2, 2), fnum=1, title='optimized')

Example:
    >>> # xdoctest: +REQUIRES(module:gdal)
    >>> from kwcoco.util.delayed_ops import *  # NOQA
    >>> import kwimage
    >>> # Sometimes we want to manipulate data in a space, but then remove all
    >>> # warps in order to get a sample without any data artifacts.  This is
    >>> # handled by adding a new transform that inverts everything and optimizing
    >>> # it, which results in all warps canceling each other out.
    >>> fpath = kwimage.grab_test_image_fpath()
    >>> base = DelayedLoad2(fpath, channels='r|g|b').prepare()
    >>> warp = kwimage.Affine.random(rng=321, offset=0)
    >>> warp = kwimage.Affine.scale(0.5)
    >>> orig = base.get_overview(1).warp(warp)[16:96, 24:128]
    >>> delayed = orig.optimize()
    >>> print('Orig')
    >>> orig.write_network_text()
    >>> print('Delayed')
    >>> delayed.write_network_text()
    >>> # Get the transform that would bring us back to the leaf
    >>> tf_root_from_leaf = delayed.get_transform_from_leaf()
    >>> print('tf_root_from_leaf =\n{}'.format(ub.repr2(tf_root_from_leaf, nl=1)))
    >>> undo_all = tf_root_from_leaf.inv()
    >>> print('undo_all =\n{}'.format(ub.repr2(undo_all, nl=1)))
    >>> undo_scale = kwimage.Affine.coerce(ub.dict_diff(undo_all.concise(), ['offset']))
    >>> print('undo_scale =\n{}'.format(ub.repr2(undo_scale, nl=1)))
    >>> print('Undone All')
    >>> undone_all = delayed.warp(undo_all).optimize()
    >>> undone_all.write_network_text()
    >>> # Discard translation components
    >>> print('Undone Scale')
    >>> undone_scale = delayed.warp(undo_scale).optimize()
    >>> undone_scale.write_network_text()

    >>> import kwplot
    >>> kwplot.autompl()
    >>> to_stack = []
    >>> to_stack.append(base.finalize())
    >>> to_stack.append(orig.finalize())
    >>> to_stack.append(delayed.finalize())
    >>> to_stack.append(undone_all.finalize())
    >>> to_stack.append(undone_scale.finalize())
    >>> kwplot.autompl()
    >>> stack = kwimage.stack_images(to_stack, axis=1, bg_value=(5, 100, 10), pad=10)
    >>> kwplot.imshow(stack)


Example:
    >>> # xdoctest: +REQUIRES(module:gdal)
    >>> from kwcoco.util.delayed_ops import *  # NOQA
    >>> import kwimage
    >>> # Demo case where we have different channels at different resolutions
    >>> fpath = kwimage.grab_test_image_fpath()
    >>> base = DelayedLoad2.demo(dsize=(600, 600), channels='r|g|b').prepare()
    >>> bandR = DelayedLoad2.demo(dsize=(100, 100), channels='r|g|b').prepare()[:, :, 0]
    >>> bandG = DelayedLoad2.demo(dsize=(300, 300), channels='r|g|b').prepare()[:, :, 1]
    >>> bandB = DelayedLoad2.demo(dsize=(600, 600), channels='r|g|b').prepare()[:, :, 2]
    >>> chan1 = bandR.warp({'scale': 6})
    >>> chan2 = bandG.warp({'scale': 2})
    >>> chan3 = bandB
    >>> orig = DelayedChannelConcat2([chan1, chan2, chan3]).warp({'scale': 0.35})[10:80, 50:135].warp({'scale': 3})
    >>> delayed = orig.optimize()
    >>> print('Orig')
    >>> orig.write_network_text()
    >>> print('Delayed')
    >>> delayed.write_network_text()
    >>> row0 = []
    >>> row1 = []
    >>> row2 = []
    >>> row0.append(base.finalize())
    >>> row1.append(orig.finalize())
    >>> row1.append(delayed.finalize())
    >>> row2.append(bandR.finalize())
    >>> row2.append(bandG.finalize())
    >>> row2.append(bandB.finalize())
    >>> row3 = []
    >>> # Get the transform that would bring us back to the leaf
    >>> for chosen_band in delayed.parts:
    >>>     spec = chosen_band.channels.spec
    >>>     lut = {c[0]: c for c in ['red', 'green', 'blue']}
    >>>     color = lut[spec]
    >>>     print(ub.color_text('============', color))
    >>>     print(ub.color_text(spec, color))
    >>>     print(ub.color_text('============', color))
    >>>     chosen_band.write_network_text()
    >>>     tf_root_from_leaf = chosen_band.get_transform_from_leaf()
    >>>     print('tf_root_from_leaf =\n{}'.format(ub.repr2(tf_root_from_leaf, nl=1)))
    >>>     undo_all = tf_root_from_leaf.inv()
    >>>     print('undo_all =\n{}'.format(ub.repr2(undo_all, nl=1)))
    >>>     undo_scale = kwimage.Affine.coerce(ub.dict_diff(undo_all.concise(), ['offset']))
    >>>     print('undo_scale =\n{}'.format(ub.repr2(undo_scale, nl=1)))
    >>>     print('Undone All')
    >>>     undone_all = chosen_band.warp(undo_all).optimize()
    >>>     undone_all.write_network_text()
    >>>     # Discard translation components
    >>>     print('Undone Scale')
    >>>     undone_scale = chosen_band.warp(undo_scale).optimize()
    >>>     undone_scale.write_network_text()
    >>>     undone_scale.write_network_text()
    >>>     row3.append(undone_all.finalize())
    >>>     row3.append(undone_scale.finalize())
    >>>     print(ub.color_text('============', color))
    >>> #
    >>> import kwplot
    >>> kwplot.autompl()
    >>> kwplot.autompl()
    >>> stack0 = kwimage.stack_images(row0, axis=1, bg_value=(5, 100, 10), pad=10)
    >>> stack1 = kwimage.stack_images(row1, axis=1, bg_value=(5, 100, 10), pad=10)
    >>> stack2 = kwimage.stack_images(row2, axis=1, bg_value=(5, 100, 10), pad=10)
    >>> stack3 = kwimage.stack_images(row3, axis=1, bg_value=(5, 100, 10), pad=10)
    >>> stack = kwimage.stack_images([stack0, stack1, stack2, stack3], axis=0, bg_value=(5, 100, 10), pad=10)
    >>> kwplot.imshow(stack, title='notice how the "undone all" crops are shifted to the right such that they align with the original image')

"""


__mkinit__ = """
mkinit -m kwcoco.util.delayed_ops
"""


__private__ = [
    '_tests',
]


__protected__ = [
    'helpers',
]


from kwcoco.util.delayed_ops import delayed_base
from kwcoco.util.delayed_ops import delayed_leafs
from kwcoco.util.delayed_ops import delayed_nodes
from kwcoco.util.delayed_ops import helpers

from kwcoco.util.delayed_ops.delayed_base import (DelayedNaryOperation2,
                                                  DelayedOperation2,
                                                  DelayedUnaryOperation2,)
from kwcoco.util.delayed_ops.delayed_leafs import (DelayedIdentity2,
                                                   DelayedImageLeaf2,
                                                   DelayedLoad2, DelayedNans2,)
from kwcoco.util.delayed_ops.delayed_nodes import (DelayedArray2,
                                                   DelayedChannelConcat2,
                                                   DelayedConcat2,
                                                   DelayedCrop2,
                                                   DelayedDequantize2,
                                                   DelayedFrameStack2,
                                                   DelayedImage2,
                                                   DelayedOverview2,
                                                   DelayedStack2, DelayedWarp2,
                                                   JaggedArray2,)

__all__ = ['DelayedArray2', 'DelayedChannelConcat2', 'DelayedConcat2',
           'DelayedCrop2', 'DelayedDequantize2', 'DelayedFrameStack2',
           'DelayedIdentity2', 'DelayedImage2', 'DelayedImageLeaf2',
           'DelayedLoad2', 'DelayedNans2', 'DelayedNaryOperation2',
           'DelayedOperation2', 'DelayedOverview2', 'DelayedStack2',
           'DelayedUnaryOperation2', 'DelayedWarp2', 'JaggedArray2',
           'delayed_base', 'delayed_leafs', 'delayed_nodes', 'helpers']
