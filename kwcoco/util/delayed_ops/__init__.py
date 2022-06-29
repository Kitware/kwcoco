"""
A rewrite of the delayed operations

TODO:
    The optimize logic could likley be better expressed as some sort of
    AST transformer.

Example:
    >>> # Make a complex chain of operations and optimize it
    >>> from kwcoco.util.delayed_ops import *  # NOQA
    >>> import kwimage
    >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
    >>> dimg = DelayedLoad2(fpath, channels='r|g|b')._load_metadata()
    >>> quantization = {'quant_max': 255, 'nodata': 0}
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
    >>> dopt = dimg.optimize()
    >>> if 1:
    >>>     import networkx as nx
    >>>     dimg.write_network_text()
    >>>     dopt.write_network_text()
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
    'delayed_tests',
]


from kwcoco.util.delayed_ops import delayed_base
from kwcoco.util.delayed_ops import delayed_leafs
from kwcoco.util.delayed_ops import delayed_nodes
from kwcoco.util.delayed_ops import delayed_tests
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
from kwcoco.util.delayed_ops.helpers import (dequantize, profile,)

__all__ = ['DelayedArray2', 'DelayedChannelConcat2', 'DelayedConcat2',
           'DelayedCrop2', 'DelayedDequantize2', 'DelayedFrameStack2',
           'DelayedIdentity2', 'DelayedImage2', 'DelayedImageLeaf2',
           'DelayedLoad2', 'DelayedNans2', 'DelayedNaryOperation2',
           'DelayedOperation2', 'DelayedOverview2', 'DelayedStack2',
           'DelayedUnaryOperation2', 'DelayedWarp2', 'JaggedArray2',
           'delayed_base', 'delayed_leafs', 'delayed_nodes',
           'delayed_tests', 'dequantize', 'helpers', 'profile']
