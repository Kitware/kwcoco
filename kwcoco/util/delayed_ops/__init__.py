"""
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
    >>> dimg = DelayedLoad2(fpath, channels='r|g|b')._load_metadata()
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
