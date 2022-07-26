r"""
A rewrite of the delayed operations

Note:
    The classes in this submodule will have their names changed when the old
    POC delayed operations are deprecated.

TODO:
    The optimize logic could likley be better expressed as some sort of
    AST transformer.

Example:
    >>> # xdoctest: +REQUIRES(module:osgeo)
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
    ╙── Crop dsize=(128,130),space_slice=(slice(1,131,None),slice(2,130,None))
        └─╼ Crop dsize=(130,131),space_slice=(slice(0,131,None),slice(1,131,None))
            └─╼ Warp dsize=(131,131),transform={scale=2.1000}
                └─╼ Warp dsize=(62,62),transform={scale=1.1000}
                    └─╼ Warp dsize=(56,56),transform={scale=1.1000}
                        └─╼ Warp dsize=(50,50),transform={scale=0.5000}
                            └─╼ Crop dsize=(99,100),space_slice=(slice(0,100,None),slice(1,100,None))
                                └─╼ Warp dsize=(100,100),transform={scale=0.5000}
                                    └─╼ Crop dsize=(199,200),space_slice=(slice(0,200,None),slice(1,200,None))
                                        └─╼ Warp dsize=(200,200),transform={scale=0.5000}
                                            └─╼ Crop dsize=(399,400),space_slice=(slice(0,400,None),slice(1,400,None))
                                                └─╼ Warp dsize=(621,621),transform={scale=1.1000}
                                                    └─╼ Warp dsize=(564,564),transform={scale=1.1000}
                                                        └─╼ Dequantize dsize=(512,512),quantization={quant_max=255,nodata=0}
                                                            └─╼ Load channels=r|g|b,dsize=(512,512),num_overviews=3,fname=astro_overviews=3.tif

    >>> # Optimize the chain
    >>> dopt = dimg.optimize()
    >>> dopt.write_network_text()
    ╙── Warp dsize=(128,130),transform={offset=(-0.6...,-1.0...),scale=1.5373}
        └─╼ Dequantize dsize=(80,83),quantization={quant_max=255,nodata=0}
            └─╼ Crop dsize=(80,83),space_slice=(slice(0,83,None),slice(3,83,None))
                └─╼ Overview dsize=(128,128),overview=2
                    └─╼ Load channels=r|g|b,dsize=(512,512),num_overviews=3,fname=astro_overviews=3.tif

    >>> final0 = dimg.finalize(optimize=False)
    >>> final1 = dopt.finalize()
    >>> assert final0.shape == final1.shape
    >>> # xdoctest: +REQUIRES(--show)
    >>> import kwplot
    >>> kwplot.autompl()
    >>> kwplot.imshow(final0, pnum=(1, 2, 1), fnum=1, title='raw')
    >>> kwplot.imshow(final1, pnum=(1, 2, 2), fnum=1, title='optimized')

Example:
    >>> # xdoctest: +REQUIRES(module:osgeo)
    >>> from kwcoco.util.delayed_ops import *  # NOQA
    >>> import ubelt as ub
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
    >>> # xdoctest: +REQUIRES(--show)
    >>> import kwplot
    >>> kwplot.autompl()
    >>> to_stack = []
    >>> to_stack.append(base.finalize(optimize=False))
    >>> to_stack.append(orig.finalize(optimize=False))
    >>> to_stack.append(delayed.finalize(optimize=False))
    >>> to_stack.append(undone_all.finalize(optimize=False))
    >>> to_stack.append(undone_scale.finalize(optimize=False))
    >>> kwplot.autompl()
    >>> stack = kwimage.stack_images(to_stack, axis=1, bg_value=(5, 100, 10), pad=10)
    >>> kwplot.imshow(stack)

CommandLine:
    xdoctest -m /home/joncrall/code/kwcoco/kwcoco/util/delayed_ops/__init__.py __doc__:2

Example:
    >>> # xdoctest: +REQUIRES(module:osgeo)
    >>> from kwcoco.util.delayed_ops import *  # NOQA
    >>> import ubelt as ub
    >>> import kwimage
    >>> import kwarray
    >>> import numpy as np
    >>> # Demo case where we have different channels at different resolutions
    >>> base = DelayedLoad2.demo(channels='r|g|b').prepare().dequantize({'quant_max': 255})
    >>> bandR = base[:, :, 0].scale(100 / 512)[:, :-50].evaluate()
    >>> bandG = base[:, :, 1].scale(300 / 512).warp({'theta': np.pi / 8, 'about': (150, 150)}).evaluate()
    >>> bandB = base[:, :, 2].scale(600 / 512)[:150, :].evaluate()
    >>> # Align the bands in "video" space
    >>> delayed_vidspace = DelayedChannelConcat2([
    >>>     bandR.scale(6, dsize=(600, 600)).optimize(),
    >>>     bandG.warp({'theta': -np.pi / 8, 'about': (150, 150)}).scale(2, dsize=(600, 600)).optimize(),
    >>>     bandB.scale(1, dsize=(600, 600)).optimize(),
    >>> ]).warp(
    >>>   #{'scale': 0.35, 'theta': 0.3, 'about': (30, 50), 'offset': (-10, -80)}
    >>>   {'scale': 0.7}
    >>> )
    >>> #delayed_vidspace._set_nested_params(border_value=0)
    >>> vidspace_box = kwimage.Boxes([[100, 10, 270, 160]], 'ltrb')
    >>> vidspace_poly = vidspace_box.to_polygons()[0]
    >>> vidspace_slice = vidspace_box.to_slices()[0]
    >>> crop_vidspace = delayed_vidspace[vidspace_slice]
    >>> crop_vidspace._set_nested_params(interpolation='lanczos')
    >>> # Note: this only works because the graph is lazilly optimized
    >>> crop_vidspace_box = vidspace_box.warp(crop_vidspace._transform_from_subdata())
    >>> crop_vidspace_poly = vidspace_poly.warp(crop_vidspace._transform_from_subdata())
    >>> opt_crop_vidspace = crop_vidspace.optimize()
    >>> print('Original: Video Space')
    >>> delayed_vidspace.write_network_text()
    >>> print('Original Crop: Video Space')
    >>> crop_vidspace.write_network_text()
    >>> print('Optimized Crop: Video Space')
    >>> opt_crop_vidspace.write_network_text()
    >>> tostack_grid = []
    >>> # Drop boxes in asset space
    >>> tostack_grid.append([]); row = tostack_grid[-1]
    >>> row.append(kwimage.draw_text_on_image(None, text='Underlying asset bands (imagine these are on disk)'))
    >>> tostack_grid.append([]); row = tostack_grid[-1]
    >>> delayed_vidspace_opt = delayed_vidspace.optimize()
    >>> tf_vidspace_to_rband = delayed_vidspace_opt.parts[0].get_transform_from_leaf().inv()
    >>> tf_vidspace_to_gband = delayed_vidspace_opt.parts[1].get_transform_from_leaf().inv()
    >>> tf_vidspace_to_bband = delayed_vidspace_opt.parts[2].get_transform_from_leaf().inv()
    >>> rband_box = vidspace_box.warp(tf_vidspace_to_rband)
    >>> gband_box = vidspace_box.warp(tf_vidspace_to_gband)
    >>> bband_box = vidspace_box.warp(tf_vidspace_to_bband)
    >>> rband_poly = vidspace_poly.warp(tf_vidspace_to_rband)
    >>> gband_poly = vidspace_poly.warp(tf_vidspace_to_gband)
    >>> bband_poly = vidspace_poly.warp(tf_vidspace_to_bband)
    >>> row.append(kwimage.draw_header_text(rband_poly.draw_on(rband_box.draw_on(bandR.finalize()), edgecolor='b', fill=0), 'R'))
    >>> row.append(kwimage.draw_header_text(gband_poly.draw_on(gband_box.draw_on(bandG.finalize()), edgecolor='b', fill=0), 'asset G-band'))
    >>> row.append(kwimage.draw_header_text(bband_poly.draw_on(bband_box.draw_on(bandB.finalize()), edgecolor='b', fill=0), 'asset B-band'))
    >>> # Draw the box in image space
    >>> tostack_grid.append([]); row = tostack_grid[-1]
    >>> row.append(kwimage.draw_text_on_image(None, text='A Box in Virtual Video Space (This space is conceptually easy to work in)'))
    >>> tostack_grid.append([]); row = tostack_grid[-1]
    >>> def _tocanvas(img):
    ...     if img.dtype.kind == 'u':
    ...         return img
    ...     return kwimage.ensure_uint255(kwimage.fill_nans_with_checkers(img))
    >>> row.append(kwimage.draw_header_text(vidspace_box.draw_on(_tocanvas(delayed_vidspace.finalize())), 'vidspace'))
    >>> # Draw finalized aligned crops
    >>> tostack_grid.append([]); row = tostack_grid[-1]
    >>> row.append(kwimage.draw_text_on_image(None, text='Finalized delayed warp/crop. Left-to-Right: Original, Optimized, Difference'))
    >>> tostack_grid.append([]); row = tostack_grid[-1]
    >>> crop_opt_final = opt_crop_vidspace.finalize()
    >>> crop_raw_final = crop_vidspace.finalize(optimize=False)
    >>> row.append(crop_raw_final)
    >>> row.append(crop_opt_final)
    >>> row.append(kwimage.ensure_uint255(kwarray.normalize(np.linalg.norm(kwimage.ensure_float01(crop_opt_final) - kwimage.ensure_float01(crop_raw_final), axis=2))))
    >>> # Get the transform that would bring us back to the leaf
    >>> tostack_grid.append([]); row = tostack_grid[-1]
    >>> row.append(kwimage.draw_text_on_image(None, text='The "Unwarped" / "Unscaled" cropped regions'))
    >>> tostack_grid.append([]); row = tostack_grid[-1]
    >>> for chosen_band in opt_crop_vidspace.parts:
    >>>     spec = chosen_band.channels.spec
    >>>     lut = {c[0]: c for c in ['red', 'green', 'blue']}
    >>>     color = lut[spec]
    >>>     print(ub.color_text('============', color))
    >>>     print(ub.color_text(spec, color))
    >>>     print(ub.color_text('============', color))
    >>>     chosen_band.write_network_text()
    >>>     tf_root_from_leaf = chosen_band.get_transform_from_leaf()
    >>>     tf_leaf_from_root = tf_root_from_leaf.inv()
    >>>     undo_all = tf_leaf_from_root
    >>>     undo_scale = kwimage.Affine.coerce(ub.dict_diff(undo_all.concise(), ['offset', 'theta']))
    >>>     print('tf_root_from_leaf = {}'.format(ub.repr2(tf_root_from_leaf.concise(), nl=1)))
    >>>     print('undo_all = {}'.format(ub.repr2(undo_all.concise(), nl=1)))
    >>>     print('undo_scale = {}'.format(ub.repr2(undo_scale.concise(), nl=1)))
    >>>     print('Undone All')
    >>>     undone_all = chosen_band.warp(undo_all, interpolation='lanczos').optimize()
    >>>     undone_all.write_network_text()
    >>>     # Discard translation components
    >>>     print('Undone Scale')
    >>>     undone_scale = chosen_band.warp(undo_scale).optimize()
    >>>     undone_scale.write_network_text()
    >>>     undone_all_canvas = undone_all.finalize()
    >>>     undone_scale_canvas = undone_scale.finalize()
    >>>     undone_all_canvas = crop_vidspace_box.warp(undo_all).draw_on(undone_all_canvas)
    >>>     undone_scale_canvas = crop_vidspace_box.warp(undo_scale).draw_on(undone_scale_canvas)
    >>>     undone_all_canvas = crop_vidspace_poly.warp(undo_all).draw_on(undone_all_canvas, edgecolor='b', fill=0)
    >>>     undone_scale_canvas = crop_vidspace_poly.warp(undo_scale).draw_on(undone_scale_canvas, edgecolor='b', fill=0)
    >>>     #row.append(kwimage.stack_images([undone_all_canvas, undone_scale_canvas], axis=0, bg_value=(5, 100, 10), pad=10))
    >>>     row.append(undone_all_canvas)
    >>>     row.append(undone_scale_canvas)
    >>>     print(ub.color_text('============', color))
    >>> #
    >>> # xdoctest: +REQUIRES(--show)
    >>> import kwplot
    >>> kwplot.autompl()
    >>> tostack_grid = [[_tocanvas(c) for c in cols] for cols in tostack_grid]
    >>> tostack_rows  = [kwimage.stack_images(cols, axis=1, bg_value=(5, 100, 10), pad=10) for cols in tostack_grid if cols]
    >>> stack = kwimage.stack_images(tostack_rows, axis=0, bg_value=(5, 100, 10), pad=10)
    >>> kwplot.imshow(stack, title='notice how the "undone all" crops are shifted to the right such that they align with the original image')
    >>> kwplot.show_if_requested()

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
                                                   )

__all__ = ['DelayedArray2', 'DelayedChannelConcat2', 'DelayedConcat2',
           'DelayedCrop2', 'DelayedDequantize2', 'DelayedFrameStack2',
           'DelayedIdentity2', 'DelayedImage2', 'DelayedImageLeaf2',
           'DelayedLoad2', 'DelayedNans2', 'DelayedNaryOperation2',
           'DelayedOperation2', 'DelayedOverview2', 'DelayedStack2',
           'DelayedUnaryOperation2', 'DelayedWarp2',
           'delayed_base', 'delayed_leafs', 'delayed_nodes', 'helpers']
