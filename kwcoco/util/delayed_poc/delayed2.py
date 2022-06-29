"""
A rewrite of the delayed operations

TODO:
    The optimize logic could likley be better expressed as some sort of
    AST transformer.

Example:
    >>> # Make a complex chain of operations and optimize it
    >>> from kwcoco.util.delayed_poc.delayed2 import *  # NOQA
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
import ubelt as ub
import numpy as np
import kwimage
import kwarray
import copy
import warnings
from kwcoco import exceptions
from kwcoco import channel_spec
from kwcoco.util.util_monkey import Reloadable  # NOQA


try:
    from xdev import profile
except Exception:
    from ubelt import identity as profile

DEBUG = ub.identity
# DEBUG = print


# ----------------
# Abstract Classes
# ----------------


# @Reloadable.developing
class DelayedOperation2(ub.NiceRepr):

    def __init__(self):
        self.meta = {}

    def __nice__(self):
        return '{}'.format(self.shape)

    def nesting(self):
        def _child_nesting(child):
            if hasattr(child, 'nesting'):
                return child.nesting()
            elif isinstance(child, np.ndarray):
                return {
                    'type': 'ndarray',
                    'shape': self.subdata.shape,
                }
        item = {
            'type': self.__class__.__name__,
            'meta': self.meta,
        }
        child_nodes = list(self.children())
        if child_nodes:
            child_nestings = [_child_nesting(child) for child in child_nodes]
            item['children'] = child_nestings
        return item

    def as_graph(self):
        """
        """
        import networkx as nx
        graph = nx.DiGraph()
        stack = [self]
        while stack:
            item = stack.pop()
            node_id = id(item)
            graph.add_node(node_id)
            sub_meta = {k: v for k, v in item.meta.items() if v is not None}
            if 'transform' in sub_meta:
                sub_meta['transform'] = sub_meta['transform'].concise()
                sub_meta['transform'].pop('type')
            param_key = ub.repr2(sub_meta, sort=0, compact=1, nl=0)
            name = item.__class__.__name__.replace('Delayed', '')
            node_data = graph.nodes[node_id]
            node_data['label'] = f'{name} {param_key}'
            node_data['name'] = name
            node_data['meta'] = sub_meta
            for child in item.children():
                child_id = id(child)
                graph.add_edge(node_id, child_id)
                stack.append(child)
        return graph

    def write_network_text(self, with_labels=True):
        from cmd_queue.util import graph_str
        graph = self.as_graph()
        # import networkx as nx
        # nx.write_network_text(graph)
        print(graph_str(graph, with_labels=with_labels))

    @property
    def shape(self):
        raise NotImplementedError

    def children(self):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError


class DelayedNaryOperation2(DelayedOperation2):
    """
    For operations that have multiple input arrays
    """
    def __init__(self, parts):
        super().__init__()
        self.parts = parts

    def children(self):
        yield from iter(self.parts)


class DelayedUnaryOperation2(DelayedOperation2):
    """
    For operations that have a single input array
    """
    def __init__(self, subdata):
        super().__init__()
        self.subdata = subdata

    def children(self):
        if self.subdata is not None:
            yield self.subdata


# --------
# Stacking
# --------


class DelayedStack2(DelayedNaryOperation2):
    """
    Stacks multiple arrays together.
    """

    def __init__(self, parts, axis):
        super().__init__(parts=parts)
        self.meta['axis'] = axis

    def __nice__(self):
        return '{}'.format(self.shape)

    @property
    def shape(self):
        shape = self.subdata.shape
        return shape


class DelayedConcat2(DelayedNaryOperation2):
    """
    Stacks multiple arrays together.
    """

    def __init__(self, parts, axis):
        super().__init__(parts=parts)
        self.meta['axis'] = axis

    def __nice__(self):
        return '{}'.format(self.shape)

    @property
    def shape(self):
        shape = self.subdata.shape
        return shape


class DelayedFrameStack2(DelayedStack2):
    """
    Stacks multiple arrays together.
    """

    def __init__(self, parts):
        super().__init__(parts=parts, axis=0)

# ------
# Images
# ------


class JaggedArray2(ub.NiceRepr):
    """
    The result of an unaligned concatenate
    """
    def __init__(self, parts, axis):
        self.parts = parts
        self.axis = axis

    def __nice__(self):
        return '{}, axis={}'.format(self.shape, self.axis)

    @property
    def shape(self):
        shapes = [p.shape for p in self.parts]
        return shapes


class DelayedChannelConcat2(DelayedConcat2):
    """
    Stacks multiple arrays together.

    Example:
        >>> from kwcoco.util.delayed_poc.delayed2 import *  # NOQA
        >>> import kwcoco
        >>> dsize = (307, 311)
        >>> c1 = DelayedNans2(dsize=dsize, channels='foo')
        >>> c2 = DelayedLoad2.demo('astro', dsize=dsize, channels='R|G|B').prepare()
        >>> cat = DelayedChannelConcat2([c1, c2])
        >>> warped_cat = cat.warp({'scale': 1.07}, dsize=(328, 332))
        >>> warp.optimize().finalize()

        >>> dsize = (307, 311)
        >>> c1 = DelayedNans2(dsize=dsize, channels='foo')
        >>> c2 = DelayedLoad2.demo('astro', channels='R|G|B').prepare()
        >>> cat = DelayedChannelConcat2([c1, c2], jagged=True)
        >>> warped_cat = cat.warp({'scale': 1.07}, dsize=(328, 332))
        >>> warp.optimize().finalize()
    """

    def __init__(self, parts, dsize=None, jagged=False):
        super().__init__(parts=parts, axis=2)
        self.meta['jagged'] = jagged
        if dsize is None and not jagged:
            dsize_cands = [comp.dsize for comp in self.parts]
            if not ub.allsame(dsize_cands):
                raise exceptions.CoordinateCompatibilityError(
                    # 'parts must all have the same delayed size')
                    'parts must all have the same delayed size: got {}'.format(dsize_cands))
            dsize = dsize_cands[0]
        self.dsize = dsize
        try:
            self.num_bands = sum(comp.num_bands for comp in self.parts)
        except TypeError:
            if any(comp.num_bands is None for comp in self.parts):
                self.num_bands = None
            else:
                raise

    @property
    def channels(self):
        import kwcoco
        sub_channs = []
        for comp in self.parts:
            comp_channels = comp.channels
            if comp_channels is None:
                return None
            sub_channs.append(comp_channels)
        channs = kwcoco.FusedChannelSpec.concat(sub_channs)
        return channs

    @property
    def shape(self):
        if self.meta['jagged']:
            w = h = None
        else:
            w, h = self.dsize
        return (h, w, self.num_bands)

    def finalize(self):
        """
        Execute the final transform
        """
        stack = [comp.finalize() for comp in self.parts]
        if len(stack) == 1:
            final = stack[0]
        else:
            if self.meta['jagged']:
                final = JaggedArray2(stack, axis=2)
            else:
                final = np.concatenate(stack, axis=2)
        return final

    @profile
    def take_channels(self, channels):
        """
        This method returns a subset of the vision data with only the
        specified bands / channels.

        Args:
            channels (List[int] | slice | channel_spec.FusedChannelSpec):
                List of integers indexes, a slice, or a channel spec, which is
                typically a pipe (`|`) delimited list of channel codes. See
                kwcoco.ChannelSpec for more detials.

        Returns:
            DelayedArray2:
                a delayed vision operation that only operates on the following
                channels.

        Example:
            >>> from kwcoco.util.delayed_poc.delayed_nodes import *  # NOQA
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> self = delayed = dset.delayed_load(1)
            >>> channels = 'B11|B8|B1|B10'
            >>> new = self.take_channels(channels)

        Example:
            >>> # Complex case
            >>> import kwcoco
            >>> from kwcoco.util.delayed_poc.delayed_nodes import *  # NOQA
            >>> from kwcoco.util.delayed_poc.delayed_leafs import DelayedLoad
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> delayed = dset.delayed_load(1)
            >>> astro = DelayedLoad.demo('astro').load_shape()
            >>> astro.meta['channels'] = kwcoco.FusedChannelSpec.coerce('r|g|b')
            >>> aligned = astro.warp(kwimage.Affine.scale(600 / 512), dsize='auto')
            >>> self = combo = DelayedChannelStack(delayed.parts + [aligned])
            >>> channels = 'B1|r|B8|g'
            >>> new = self.take_channels(channels)
            >>> new_cropped = new.crop((slice(10, 200), slice(12, 350)))
            >>> datas = new_cropped.finalize()
            >>> vizable = kwimage.normalize_intensity(datas, axis=2)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> stacked = kwimage.stack_images(vizable.transpose(2, 0, 1))
            >>> kwplot.imshow(stacked)

        Example:
            >>> # Test case where requested channel does not exist
            >>> import kwcoco
            >>> from kwcoco.util.delayed_poc.delayed_nodes import *  # NOQA
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral', use_cache=1, verbose=100)
            >>> self = dset.delayed_load(1)
            >>> channels = 'B1|foobar|bazbiz|B8'
            >>> new = self.take_channels(channels)
            >>> new_cropped = new.crop((slice(10, 200), slice(12, 350)))
            >>> fused = new_cropped.finalize()
            >>> assert fused.shape == (190, 338, 4)
            >>> assert np.all(np.isnan(fused[..., 1:3]))
            >>> assert not np.any(np.isnan(fused[..., 0]))
            >>> assert not np.any(np.isnan(fused[..., 3]))
        """
        from kwcoco.util.delayed_poc.delayed_leafs import DelayedNans
        if isinstance(channels, list):
            top_idx_mapping = channels
            top_codes = self.channels.as_list()
            request_codes = None
        else:
            channels = channel_spec.FusedChannelSpec.coerce(channels)
            # Computer subindex integer mapping
            request_codes = channels.as_list()
            top_codes = self.channels.as_oset()
            top_idx_mapping = []
            for code in request_codes:
                try:
                    top_idx_mapping.append(top_codes.index(code))
                except KeyError:
                    top_idx_mapping.append(None)

        # Rearange subcomponents into the specified channel representation
        # I am not confident that this logic is the best way to do this.
        # This may be a bottleneck
        subindexer = kwarray.FlatIndexer([
            comp.num_bands for comp in self.parts])

        accum = []
        class ContiguousSegment(object):
            def __init__(self, comp, start):
                self.comp = comp
                self.start = start
                self.stop = start + 1
                self.codes = []

        curr = None
        for request_idx, idx in enumerate(top_idx_mapping):
            if idx is None:
                # Requested channel does not exist in our data stack
                comp = None
                inner = 0
                if curr is not None and curr.comp is None:
                    inner = curr.stop
            else:
                # Requested channel exists in our data stack
                outer, inner = subindexer.unravel(idx)
                comp = self.parts[outer]
            if curr is None:
                curr = ContiguousSegment(comp, inner)
            else:
                is_contiguous = curr.comp is comp and (inner == curr.stop)
                if is_contiguous:
                    # extend the previous contiguous segment
                    curr.stop = inner + 1
                else:
                    # accept previous segment and start a new one
                    accum.append(curr)
                    curr = ContiguousSegment(comp, inner)

            # Hack for nans
            if request_codes is not None:
                curr.codes.append(request_codes[request_idx])

        # Accumulate final segment
        if curr is not None:
            accum.append(curr)

        # Execute the delayed operation
        new_components = []
        for curr in accum:
            comp = curr.comp
            if comp is None:
                # Requested component did not exist, return nans
                if request_codes is not None:
                    nan_chan = channel_spec.FusedChannelSpec(curr.codes)
                else:
                    nan_chan = None
                comp = DelayedNans(self.dsize, channels=nan_chan)
                new_components.append(comp)
            else:
                if curr.start == 0 and curr.stop == comp.num_bands:
                    # Entire component is valid, no need for sub-operation
                    new_components.append(comp)
                else:
                    # Only part of the component is taken, need to sub-operate
                    # It would be nice if we only loaded the file once if we need
                    # multiple parts discontiguously.
                    sub_idxs = list(range(curr.start, curr.stop))
                    sub_comp = comp.take_channels(sub_idxs)
                    new_components.append(sub_comp)

        new = DelayedChannelConcat2(new_components, jagged=self.meta['jagged'])
        return new

    def crop(self, space_slice):
        return self.__class__([c.crop(space_slice) for c in self.parts])

    def warp(self, transform, dsize='auto', antialias=True, interpolation='linear'):
        return self.__class__([c.warp(transform, dsize, antialias, interpolation) for c in self.parts])

    def dequantize(self, quantization):
        return self.__class__([c.dequantize(quantization) for c in self.parts])

    def get_overview(self, overview):
        return self.__class__([c.get_overview(overview) for c in self.parts])


class DelayedArray2(DelayedUnaryOperation2):
    """
    A generic NDArray.
    """
    def __init__(self, subdata=None):
        super().__init__(subdata=subdata)

    def __nice__(self):
        return '{}'.format(self.shape)

    @property
    def shape(self):
        shape = self.subdata.shape
        return shape


class DelayedImage2(DelayedArray2):
    """
    For the case where an array represents a 2D image with multiple channels
    """
    def __init__(self, subdata=None, dsize=None, channels=None):
        super().__init__(subdata)
        self.channels = channels
        self.meta['dsize'] = dsize

    def __nice__(self):
        if self.channels is None:
            return '{}'.format(self.shape)
        else:
            return '{}, {}'.format(self.shape, self.channels)

    @property
    def shape(self):
        dsize = self.dsize
        if dsize is None:
            dsize = (None, None)
        w, h = dsize
        c = self.num_bands
        return (h, w, c)

    @property
    def num_bands(self):
        num_bands = self.meta.get('num_bands', None)
        if num_bands is None and self.subdata is not None:
            num_bands = self.subdata.num_bands
        return num_bands

    @property
    def dsize(self):
        return self.meta.get('dsize', None)

    @property
    def channels(self):
        channels = self.meta.get('channels', None)
        if channels is None and self.subdata is not None:
            channels = self.subdata.channels
        return channels

    @channels.setter
    def channels(self, channels):
        if channels is None:
            num_bands = None
        else:
            if isinstance(channels, int):
                num_bands = channels
                channels = None
            else:
                import kwcoco
                channels = kwcoco.FusedChannelSpec.coerce(channels)
                num_bands = channels.normalize().numel()
        self.meta['channels'] = channels
        self.meta['num_bands'] = num_bands

    @property
    def num_overviews(self):
        num_overviews = self.meta.get('num_overviews', None)
        if num_overviews is None and self.subdata is not None:
            num_overviews = self.subdata.num_overviews
        return num_overviews

    def __getitem__(self, sl):
        assert len(sl) == 2
        sl_y, sl_x = sl
        space_slice = (sl_y, sl_x)
        return self.crop(space_slice)

    def take_channels(self, channels):
        raise NotImplementedError

    def crop(self, space_slice):
        """
        Crops an image along integer pixel coordinates.

        Args:
            space_slice (Tuple[slice, slice]): y-slice and x-slice.
        """
        new = DelayedCrop2(self, space_slice)
        return new

    def warp(self, transform, dsize='auto', antialias=True, interpolation='linear'):
        """
        Applys an affine transformation to the image

        Args:
            transform (ndarray | dict | kwimage.Affine):
                a coercable affine matrix.  See :class:`kwimage.Affine` for
                details on what can be coerced.

            dsize (Tuple[int, int] | str):
                The width / height of the output canvas. If 'auto', dsize is
                computed such that the positive coordinates of the warped image
                will fit in the new canvas. In this case, any pixel that maps
                to a negative coordinate will be clipped.  This has the
                property that the input transformation is not modified.

            antialias (bool):
                if True determines if the transform is downsampling and applies
                antialiasing via gaussian a blur. Defaults to False

            interpolation (str):
                interpolation code or cv2 integer. Interpolation codes are linear,
                nearest, cubic, lancsoz, and area. Defaults to "linear".
        """
        new = DelayedWarp2(self, transform, dsize=dsize, antialias=antialias,
                           interpolation=interpolation)
        return new

    def dequantize(self, quantization):
        """
        Rescales image intensities from int to floats.

        Args:
            quantization (Dict):
                see :func:`kwcoco.util.delayed_poc.helpers.dequantize`
        """
        new = DelayedDequantize2(self, quantization)
        return new

    def get_overview(self, overview):
        """
        Downsamples an image by a factor of two.

        Args:
            overview (int): the overview to use (assuming it exists)
        """
        new = DelayedOverview2(self, overview)
        return new


class DelayedWarp2(DelayedImage2):
    """
    Applies an affine transform to an image.

    Example:
        >>> from kwcoco.util.delayed_poc.delayed2 import *  # NOQA
        >>> self = DelayedLoad2.demo(dsize=(16, 16))._load_metadata()
        >>> warp1 = self.warp({'scale': 3})
        >>> warp2 = warp1.warp({'theta': 0.1})
        >>> warp3 = warp2._opt_fuse_warps()
        >>> print(ub.repr2(warp2.nesting(), nl=-1, sort=0))
        >>> print(ub.repr2(warp3.nesting(), nl=-1, sort=0))
    """
    def __init__(self, subdata, transform, dsize='auto', antialias=True,
                 interpolation='linear'):
        """
        Args:
            subdata (DelayedArray): data to operate on

            transform (ndarray | dict | kwimage.Affine):
                a coercable affine matrix.  See :class:`kwimage.Affine` for
                details on what can be coerced.

            dsize (Tuple[int, int] | str):
                The width / height of the output canvas. If 'auto', dsize is
                computed such that the positive coordinates of the warped image
                will fit in the new canvas. In this case, any pixel that maps
                to a negative coordinate will be clipped.  This has the
                property that the input transformation is not modified.

            antialias (bool):
                if True determines if the transform is downsampling and applies
                antialiasing via gaussian a blur. Defaults to False

            interpolation (str):
                interpolation code or cv2 integer. Interpolation codes are linear,
                nearest, cubic, lancsoz, and area. Defaults to "linear".
        """
        super().__init__(subdata)
        transform = kwimage.Affine.coerce(transform)
        if dsize == 'auto':
            from kwcoco.util.delayed_poc.helpers import _auto_dsize
            dsize = _auto_dsize(transform, self.subdata.dsize)
        self.meta['transform'] = transform
        self.meta['antialias'] = antialias
        self.meta['interpolation'] = interpolation
        self.meta['dsize'] = dsize

    def finalize(self):
        dsize = self.dsize
        if dsize == (None, None):
            dsize = None
        antialias = self.meta['antialias']
        transform = self.meta['transform']
        interpolation = self.meta['interpolation']

        prewarp = self.subdata.finalize()
        prewarp = np.asarray(prewarp)

        M = np.asarray(transform)
        final = kwimage.warp_affine(prewarp, M, dsize=dsize,
                                    interpolation=interpolation,
                                    antialias=antialias)
        # final = cv2.warpPerspective(sub_data_, M, dsize=dsize, flags=flags)
        # Ensure that the last dimension is channels
        final = kwarray.atleast_nd(final, 3, front=False)
        return final

    @profile
    def optimize(self):
        DEBUG(f'Optimize {self.__class__.__name__}')
        new = copy.copy(self)
        new.subdata = self.subdata.optimize()
        if isinstance(new.subdata, DelayedWarp2):
            new = new._opt_fuse_warps()
        split = new._opt_split_warp_overview()
        if new is not split:
            new = split
            new.subdata = new.subdata.optimize()
        return new

    def _opt_fuse_warps(self):
        """
        Combine two consecutive warps into a single operation.
        """
        DEBUG('Optimize _opt_fuse_warps')
        assert isinstance(self.subdata, DelayedWarp2)
        inner_data = self.subdata.subdata
        tf1 = self.subdata.meta['transform']
        tf2 = self.meta['transform']
        # TODO: could ensure the metadata is compatable, for now just take the
        # most recent
        dsize = self.meta['dsize']
        common_meta = ub.dict_isect(self.meta, {'antialias', 'interpolation'})
        new_transform = tf2 @ tf1
        new = self.__class__(inner_data, new_transform, dsize=dsize,
                             **common_meta)
        return new

    def _opt_split_warp_overview(self):
        """
        Split this node into a warp and an overview if possible

        Example:
            >>> # xdoctest: +REQUIRES(module:gdal)
            >>> from kwcoco.util.delayed_poc.delayed2 import *  # NOQA
            >>> import kwimage
            >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
            >>> self = DelayedLoad2(fpath, channels='r|g|b')._load_metadata()
            >>> print(f'self={self}')
            >>> print('self.meta = {}'.format(ub.repr2(self.meta, nl=1)))
            >>> warp0 = self.warp({'scale': 0.2})
            >>> warp1 = warp0.optimize_warp_overview()
            >>> warp2 = self.warp({'scale': 0.25}).optimize_warp_overview()
            >>> print(ub.repr2(warp0.nesting(), nl=-1, sort=0))
            >>> print(ub.repr2(warp1.nesting(), nl=-1, sort=0))
            >>> print(ub.repr2(warp2.nesting(), nl=-1, sort=0))

        Example:
            >>> # xdoctest: +REQUIRES(module:gdal)
            >>> from kwcoco.util.delayed_poc.delayed2 import *  # NOQA
            >>> import kwimage
            >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
            >>> self = DelayedLoad2(fpath, channels='r|g|b')._load_metadata()
            >>> warp0 = self.warp({'scale': 1 / 2 ** 6})
            >>> opt = warp0.optimize()
            >>> print(ub.repr2(warp0.nesting(), nl=-1, sort=0))
            >>> print(ub.repr2(opt.nesting(), nl=-1, sort=0))
        """
        DEBUG('Optimize _opt_split_warp_overview')
        inner_data = self.subdata
        num_overviews = inner_data.num_overviews
        if not num_overviews:
            return self

        # Check if there is a strict downsampling component
        transform = self.meta['transform']
        params = transform.decompose()
        sx, sy = params['scale']
        if sx >= 1 or sy >= 1:
            return self

        # Check how many pyramid downs we could replace downsampling with
        from kwimage.im_cv2 import _prepare_scale_residual
        num_downs_possible, _, _ = _prepare_scale_residual(sx, sy, fudge=0)
        # But only use as many downs as we have overviews
        num_downs = min(num_overviews, num_downs_possible)
        if num_downs == 0:
            return self

        # Given the overview, find the residual to reconstruct the original
        overview_transform = kwimage.Affine.scale(1 / (2 ** num_downs))
        # Let T=original, O=overview, R=residual
        # T = R @ O
        # T @ O.inv = R @ O @ O.inv
        # T @ O.inv = R
        residual_transform = transform @ overview_transform.inv()
        new_transform = residual_transform
        dsize = self.meta['dsize']
        overview = inner_data.get_overview(num_downs)
        if new_transform.isclose_identity():
            new = overview
        else:
            common_meta = ub.dict_isect(self.meta, {
                'antialias', 'interpolation'})
            new = overview.warp(new_transform, dsize=dsize, **common_meta)
        return new


class DelayedDequantize2(DelayedImage2):
    """
    Rescales image intensities from int to floats.

    The output is usually between 0 and 1. This also handles transforming
    nodata into nan values.
    """
    def __init__(self, subdata, quantization):
        """
        Args:
            subdata (DelayedArray): data to operate on
            quantization (Dict):
                see :func:`kwcoco.util.delayed_poc.helpers.dequantize`
        """
        super().__init__(subdata)
        self.meta['quantization'] = quantization
        self.meta['dsize'] = subdata.dsize

    def finalize(self):
        from kwcoco.util.delayed_poc.helpers import dequantize
        quantization = self.meta['quantization']
        final = self.subdata.finalize()
        final = kwarray.atleast_nd(final, 3, front=False)
        if quantization is not None:
            final = dequantize(final, quantization)
        return final

    @profile
    def optimize(self):
        """
        Example:
            >>> # Test a case that caused an error in development
            >>> from kwcoco.util.delayed_poc.delayed2 import *  # NOQA
            >>> fpath = kwimage.grab_test_image_fpath()
            >>> base = DelayedLoad2(fpath, channels='r|g|b')._load_metadata()
            >>> quantization = {'quant_max': 255, 'nodata': 0}
            >>> self = base.get_overview(1).dequantize(quantization)
            >>> self.write_network_text()
            >>> opt = self.optimize()
        """
        DEBUG(f'Optimize {self.__class__.__name__}')
        new = copy.copy(self)
        new.subdata = self.subdata.optimize()

        if isinstance(new.subdata, DelayedDequantize2):
            raise AssertionError('Dequantization is only allowed once')

        if isinstance(new.subdata, DelayedWarp2):
            # Swap order so quantize is before the warp
            new = new._opt_dequant_before_other()
            new = new.optimize()

        return new

    def _opt_dequant_before_other(self):
        DEBUG('Optimize _opt_dequant_before_other')
        quantization = self.meta['quantization']
        new = copy.copy(self.subdata)
        new.subdata = new.subdata.dequantize(quantization)
        return new


class DelayedCrop2(DelayedImage2):
    """
    Crops an image along integer pixel coordinates.

    Example:
        >>> from kwcoco.util.delayed_poc.delayed2 import *  # NOQA
        >>> self = DelayedLoad2.demo(dsize=(16, 16))._load_metadata()
        >>> crop1 = self[4:12, 0:16]
        >>> crop2 = crop1[2:6, 0:8]
        >>> crop3 = crop2._opt_fuse_crops()
        >>> print(ub.repr2(crop3.nesting(), nl=-1, sort=0))
    """
    def __init__(self, subdata, space_slice):
        """
        Args:
            subdata (DelayedArray): data to operate on
            space_slice (Tuple[slice, slice]): y-slice and x-slice.
        """
        super().__init__(subdata)
        # TODO: are we doing infinite padding or clipping?
        # This assumes infinite padding
        in_w, in_h = subdata.dsize
        space_dims = (in_h, in_w)
        space_slice, _pad = kwarray.embed_slice(space_slice, space_dims)

        sl_y, sl_x = space_slice[0:2]
        width = sl_x.stop - sl_x.start
        height = sl_y.stop - sl_y.start
        self.meta['dsize'] = (width, height)
        self.meta['space_slice'] = space_slice

    def finalize(self):
        space_slice = self.meta['space_slice']
        sub_final = self.subdata.finalize()
        final = sub_final[space_slice]
        return final

    @profile
    def optimize(self):
        DEBUG(f'Optimize {self.__class__.__name__}')
        new = copy.copy(self)
        new.subdata = self.subdata.optimize()
        if isinstance(new.subdata, DelayedCrop2):
            new = new._opt_fuse_crops()

        if isinstance(new.subdata, DelayedWarp2):
            new = new._opt_warp_after_crop()
            new = new.optimize()
        elif isinstance(new.subdata, DelayedDequantize2):
            new = new._opt_dequant_after_crop()
            new = new.optimize()
        return new

    def _opt_fuse_crops(self):
        """
        Combine two consecutive crops into a single operation.
        """
        DEBUG('Optimize _opt_fuse_crops')
        assert isinstance(self.subdata, DelayedCrop2)
        # Inner is the data closer to the leaf (disk), outer is the data closer
        # to the user (output).
        inner_data = self.subdata.subdata
        inner_slices = self.subdata.meta['space_slice']
        outer_slices = self.meta['space_slice']

        outer_ysl, outer_xsl = outer_slices
        inner_ysl, inner_xsl = inner_slices

        # Apply the new relative slice to the current absolute slice
        new_xstart = min(inner_xsl.start + outer_xsl.start, inner_xsl.stop)
        new_xstop = min(inner_xsl.start + outer_xsl.stop, inner_xsl.stop)
        new_ystart = min(inner_ysl.start + outer_ysl.start, inner_ysl.stop)
        new_ystop = min(inner_ysl.start + outer_ysl.stop, inner_ysl.stop)

        new_crop = (slice(new_ystart, new_ystop), slice(new_xstart, new_xstop))
        new = self.__class__(inner_data, new_crop)
        return new

    def _opt_warp_after_crop(self):
        """
        If the child node is a warp, move it after the crop.

        This is more efficient because:
            1. The crop is closer to the load.
            2. we are warping with less data.

        Example:
            >>> from kwcoco.util.delayed_poc.delayed2 import *  # NOQA
            >>> fpath = kwimage.grab_test_image_fpath()
            >>> node0 = DelayedLoad2(fpath, channels='r|g|b')._load_metadata()
            >>> node1 = node0.warp({'scale': 0.432, 'theta': np.pi / 3, 'about': (80, 80), 'shearx': .3, 'offset': (-50, -50)})
            >>> node2 = node1[10:50, 1:40]
            >>> self = node2
            >>> new_outer = node2._opt_warp_after_crop()
            >>> print(ub.repr2(node2.nesting(), nl=-1, sort=0))
            >>> print(ub.repr2(new_outer.nesting(), nl=-1, sort=0))
            >>> final0 = self.finalize()
            >>> final1 = new_outer.finalize()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(final0, pnum=(2, 2, 1), fnum=1, title='raw')
            >>> kwplot.imshow(final1, pnum=(2, 2, 2), fnum=1, title='optimized')

        Example:
            >>> # xdoctest: +REQUIRES(module:gdal)
            >>> from kwcoco.util.delayed_poc.delayed2 import *  # NOQA
            >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
            >>> node0 = DelayedLoad2(fpath, channels='r|g|b')._load_metadata()
            >>> node1 = node0.warp({'scale': 1000 / 512})
            >>> node2 = node1[250:750, 0:500]
            >>> self = node2
            >>> new_outer = node2._opt_warp_after_crop()
            >>> print(ub.repr2(node2.nesting(), nl=-1, sort=0))
            >>> print(ub.repr2(new_outer.nesting(), nl=-1, sort=0))
        """
        DEBUG('Optimize _opt_warp_after_crop')
        assert isinstance(self.subdata, DelayedWarp2)
        # Inner is the data closer to the leaf (disk), outer is the data closer
        # to the user (output).
        outer_slices = self.meta['space_slice']
        inner_transform = self.subdata.meta['transform']

        outer_region = kwimage.Boxes.from_slice(outer_slices)
        outer_region = outer_region.to_polygons()[0]

        from kwcoco.util.delayed_poc.helpers import _swap_warp_after_crop
        inner_slice, outer_transform = _swap_warp_after_crop(
            outer_region, inner_transform)

        warp_meta = ub.dict_isect(self.meta, {
            'dsize', 'antialias', 'interpolation'})

        new_inner = self.subdata.subdata.crop(inner_slice)
        new_outer = new_inner.warp(outer_transform, **warp_meta)
        return new_outer

    def _opt_dequant_after_crop(self):
        # Swap order so dequantize is after the crop
        assert isinstance(self.subdata, DelayedDequantize2)
        quantization = self.subdata.meta['quantization']
        new = copy.copy(self)
        new.subdata = self.subdata.subdata  # Remove the dequantization
        new = new.dequantize(quantization)  # Push it after the crop
        return new


class DelayedOverview2(DelayedImage2):
    """
    Downsamples an image by a factor of two.

    If the underlying image being loaded has precomputed overviews it simply
    loads these instead of downsampling the original image, which is more
    efficient.

    Example:
        >>> # Make a complex chain of operations and optimize it
        >>> from kwcoco.util.delayed_poc.delayed2 import *  # NOQA
        >>> import kwimage
        >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
        >>> dimg = DelayedLoad2(fpath, channels='r|g|b')._load_metadata()
        >>> dimg = dimg.get_overview(1)
        >>> dimg = dimg.get_overview(1)
        >>> dimg = dimg.get_overview(1)
        >>> dopt = dimg.optimize()
        >>> if 1:
        >>>     import networkx as nx
        >>>     nx.write_network_text(dimg.as_graph())
        >>>     nx.write_network_text(dopt.as_graph())
        >>> print(ub.repr2(dopt.nesting(), nl=-1, sort=0))
        >>> final0 = dimg.finalize()
        >>> final1 = dopt.finalize()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(final0, pnum=(1, 2, 1), fnum=1, title='raw')
        >>> kwplot.imshow(final1, pnum=(1, 2, 2), fnum=1, title='optimized')
    """
    def __init__(self, subdata, overview):
        """
        Args:
            subdata (DelayedArray): data to operate on
            overview (int): the overview to use (assuming it exists)
        """
        super().__init__(subdata)
        self.meta['overview'] = overview
        w, h = subdata.dsize
        sf = 1 / (2 ** overview)
        """
        Ignore:
            # Check how gdal handles overviews for odd sized images.
            imdata = np.random.rand(31, 29)
            kwimage.imwrite('foo.tif', imdata, backend='gdal', overviews=3)
            ub.cmd('gdalinfo foo.tif', verbose=3)
        """
        # The rounding operation for gdal overviews is ceiling
        def iceil(x):
            return int(np.ceil(x))
        w = iceil(sf * w)
        h = iceil(sf * h)
        self.meta['dsize'] = (w, h)

    @property
    def num_overviews(self):
        # This operation reduces the number of available overviews
        num_remain = self.subdata.num_overviews - self.meta['overview']
        return num_remain

    def finalize(self):
        sub_final = self.subdata.finalize()
        if not hasattr(sub_final, 'get_overview'):
            warnings.warn(ub.paragraph(
                '''
                The underlying data does not have overviews.
                Simulating the overview using a imresize operation.
                '''
            ))
            final = kwimage.imresize(
                sub_final,
                scale=1 / 2 ** self.meta['overview'],
                interpolation='nearest',
                # antialias=True
            )
        else:
            final = sub_final.get_overview(self.meta['overview'])
        return final

    @profile
    def optimize(self):
        DEBUG(f'Optimize {self.__class__.__name__}')
        new = copy.copy(self)
        new.subdata = self.subdata.optimize()
        if isinstance(new.subdata, DelayedOverview2):
            new = new._opt_fuse_overview()

        if isinstance(new.subdata, DelayedCrop2):
            new = new._opt_crop_after_overview()
            new = new.optimize()
        elif isinstance(new.subdata, DelayedWarp2):
            new = new._opt_warp_after_overview()
            new = new.optimize()
        elif isinstance(new.subdata, DelayedDequantize2):
            new = new._opt_dequant_after_overview()
            new = new.optimize()
        return new

    def _opt_crop_after_overview(self):
        """
        Given an outer overview and an inner crop, switch places. We want the
        overview to be as close to the load as possible.

        Example:
            >>> # xdoctest: +REQUIRES(module:gdal)
            >>> from kwcoco.util.delayed_poc.delayed2 import *  # NOQA
            >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
            >>> node0 = DelayedLoad2(fpath, channels='r|g|b')._load_metadata()
            >>> node1 = node0[100:400, 120:450]
            >>> node2 = node1.get_overview(2)
            >>> self = node2
            >>> new_outer = node2.optimize()
            >>> print(ub.repr2(node2.nesting(), nl=-1, sort=0))
            >>> print(ub.repr2(new_outer.nesting(), nl=-1, sort=0))
            >>> final0 = self.finalize()
            >>> final1 = new_outer.finalize()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(final0, pnum=(1, 2, 1), fnum=1, title='raw')
            >>> kwplot.imshow(final1, pnum=(1, 2, 2), fnum=1, title='optimized')
        """
        from kwcoco.util.delayed_poc.helpers import _swap_crop_after_warp
        assert isinstance(self.subdata, DelayedCrop2)
        # Inner is the data closer to the leaf (disk), outer is the data closer
        # to the user (output).
        outer_overview = self.meta['overview']
        inner_slices = self.subdata.meta['space_slice']

        sf = 1 / 2 ** outer_overview
        outer_transform = kwimage.Affine.scale(sf)

        inner_region = kwimage.Boxes.from_slice(inner_slices)
        inner_region = inner_region.to_polygons()[0]

        new_inner_warp, outer_crop, new_outer_warp = _swap_crop_after_warp(
            inner_region, outer_transform)

        # Move the overview to the inside, it should be unchanged
        new = self.subdata.subdata.get_overview(outer_overview)

        # Move the crop to the outside
        new = new.crop(outer_crop)

        if not np.all(np.isclose(np.eye(3), new_outer_warp)):
            # we might have to apply an additional warp at the end.
            new = new.warp(new_outer_warp)
        return new

    def _opt_fuse_overview(self):
        assert isinstance(self.subdata, DelayedOverview2)
        outer_overview = self.meta['overview']
        inner_overrview = self.subdata.meta['overview']
        new_overview = inner_overrview + outer_overview
        new = self.subdata.subdata.get_overview(new_overview)
        return new

    def _opt_dequant_after_overview(self):
        # Swap order so dequantize is after the crop
        assert isinstance(self.subdata, DelayedDequantize2)
        quantization = self.subdata.meta['quantization']
        new = copy.copy(self)
        new.subdata = self.subdata.subdata  # Remove the dequantization
        new = new.dequantize(quantization)  # Push it after the crop
        return new

    def _opt_warp_after_overview(self):
        """
        Given an warp followed by an overview, move the warp to the outer scope
        such that the overview is first.

        Example:
            >>> # xdoctest: +REQUIRES(module:gdal)
            >>> from kwcoco.util.delayed_poc.delayed2 import *  # NOQA
            >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
            >>> node0 = DelayedLoad2(fpath, channels='r|g|b')._load_metadata()
            >>> node1 = node0.warp({'scale': (2.1, .7), 'offset': (20, 40)})
            >>> node2 = node1.get_overview(2)
            >>> self = node2
            >>> new_outer = node2.optimize()
            >>> print(ub.repr2(node2.nesting(), nl=-1, sort=0))
            >>> print(ub.repr2(new_outer.nesting(), nl=-1, sort=0))
            >>> final0 = self.finalize()
            >>> final1 = new_outer.finalize()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(final0, pnum=(1, 2, 1), fnum=1, title='raw')
            >>> kwplot.imshow(final1, pnum=(1, 2, 2), fnum=1, title='optimized')
        """
        assert isinstance(self.subdata, DelayedWarp2)
        outer_overview = self.meta['overview']
        inner_transform = self.subdata.meta['transform']

        sf = 1 / 2 ** outer_overview
        outer_transform = kwimage.Affine.scale(sf)

        A = outer_transform
        B = inner_transform
        # We have: A @ B, and we want that to equal C @ A
        # where the overview A left as-is and moved inside, we modify the new
        # outer transform C to accomodate this.
        # So C = A @ B @ A.inv()
        C = A @ B @ A.inv()
        new_outer = C
        new_inner_overview = outer_overview
        new = self.subdata.subdata.get_overview(new_inner_overview)
        new = new.warp(new_outer)
        return new


# -----
# Leafs
# -----

class DelayedImageLeaf2(DelayedImage2):

    @profile
    def optimize(self):
        DEBUG(f'Optimize {self.__class__.__name__}')
        return self


class DelayedLoad2(DelayedImageLeaf2):
    """
    Reads an image from disk.

    If a gdal backend is available, and the underlying image is in the
    appropriate formate (e.g. COG) this will return a lazy reference that
    enables fast overviews and crops.

    Example:
        >>> from kwcoco.util.delayed_poc.delayed2 import *  # NOQA
        >>> self = DelayedLoad2.demo(dsize=(16, 16))._load_metadata()
        >>> data1 = self.finalize()

    Example:
        >>> # xdoctest: +REQUIRES(module:gdal)
        >>> # Demo code to develop support for overviews
        >>> from kwcoco.util.delayed_poc.delayed2 import *  # NOQA
        >>> import kwimage
        >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
        >>> self = DelayedLoad2(fpath, channels='r|g|b')._load_metadata()
        >>> print(f'self={self}')
        >>> print('self.meta = {}'.format(ub.repr2(self.meta, nl=1)))
        >>> quantization = {
        >>>     'quant_max': 255,
        >>>     'nodata': 0,
        >>> }
        >>> node0 = self
        >>> node1 = node0.get_overview(2)
        >>> node2 = node1[13:900, 11:700]
        >>> node3 = node2.dequantize(quantization)
        >>> node4 = node3.warp({'scale': 0.05})
        >>> #
        >>> data0 = node0.finalize()
        >>> data1 = node1.finalize()
        >>> data2 = node2.finalize()
        >>> data3 = node3.finalize()
        >>> data4 = node4.finalize()
        >>> print(ub.repr2(node4.nesting(), nl=-1, sort=0))
        >>> print(f'{data0.shape=} {data0.dtype=}')
        >>> print(f'{data1.shape=} {data1.dtype=}')
        >>> print(f'{data2.shape=} {data2.dtype=}')
        >>> print(f'{data3.shape=} {data3.dtype=}')
        >>> print(f'{data4.shape=} {data4.dtype=}')
    """
    def __init__(self, fpath, channels=None, dsize=None):
        """
        Args:
            fpath (str | PathLike):
                URI pointing at the image data to load

            channels (int | str | kwcoco.FusedChannelSpec | None):
                the underlying channels of the image if known a-priori

            dsize (Tuple[int, int]):
                The width / height of the image if known a-priori
        """
        super().__init__(channels=channels, dsize=dsize)
        self.meta['fpath'] = fpath
        self.lazy_ref = None

    @property
    def fpath(self):
        return self.meta['fpath']

    @classmethod
    def demo(DelayedLoad2, key='astro', dsize=None, channels=None):
        fpath = kwimage.grab_test_image_fpath(key, dsize=dsize)
        self = DelayedLoad2(fpath, channels=channels)
        return self

    def _load_reference(self):
        if self.lazy_ref is None:
            from kwcoco.util import lazy_frame_backends
            using_gdal = lazy_frame_backends.LazyGDalFrameFile.available()
            if using_gdal:
                self.lazy_ref = lazy_frame_backends.LazyGDalFrameFile(self.fpath)
            else:
                self.lazy_ref = NotImplemented
        return self

    def prepare(self):
        return self._load_metadata()

    def _load_metadata(self):
        self._load_reference()
        if self.lazy_ref is NotImplemented:
            shape = kwimage.load_image_shape(self.fpath)
            if len(shape) == 2:
                shape = shape + (1,)
            num_overviews = 0
        else:
            shape = self.lazy_ref.shape
            num_overviews = self.lazy_ref.num_overviews
        h, w, c = shape
        if self.dsize is None:
            self.meta['dsize'] = (w, h)
        if self.num_bands is None:
            self.meta['num_bands'] = c
        self.meta['num_overviews'] = num_overviews
        return self

    def finalize(self):
        self._load_reference()
        if self.lazy_ref is NotImplemented:
            warnings.warn('DelayedLoad2 may not be efficient without gdal')
            pre_final = kwimage.imread(self.fpath)
            pre_final = kwarray.atleast_nd(pre_final, 3)
        else:
            return self.lazy_ref


class DelayedNans2(DelayedImageLeaf2):
    """
    Constructs nan channels as needed

    Example:
        self = DelayedNans((10, 10), channel_spec.FusedChannelSpec.coerce('rgb'))
        region_slices = (slice(5, 10), slice(1, 12))
        delayed = self.crop(region_slices)

    Example:
        >>> from kwcoco.util.delayed_poc.delayed2 import *  # NOQA
        >>> import kwcoco
        >>> dsize = (307, 311)
        >>> c1 = DelayedNans2(dsize=dsize, channels='foo')
        >>> c2 = DelayedLoad2.demo('astro', dsize=dsize, channels='R|G|B').prepare()
        >>> cat = DelayedChannelConcat2([c1, c2])
        >>> warped_cat = cat.warp({'scale': 1.07}, dsize=(328, 332))
        >>> warp.optimize().finalize()

        #>>> cropped = warped_cat.crop((slice(0, 300), slice(0, 100)))
        #>>> cropped.finalize().shape
    """
    def __init__(self, dsize=None, channels=None):
        super().__init__(channels=channels, dsize=dsize)

    def finalize(self):
        shape = self.shape
        final = np.full(shape, fill_value=np.nan)
        return final

    def crop(self, region_slices):
        channels = self.channels
        dsize = self.dsize
        data_dims = dsize[::-1]
        data_slice, extra_pad = kwarray.embed_slice(region_slices, data_dims)
        box = kwimage.Boxes.from_slice(data_slice)
        new_width = box.width.ravel()[0]
        new_height = box.height.ravel()[0]
        new_dsize = (new_width, new_height)
        new = self.__class__(new_dsize, channels=channels)
        return new

    def warp(self, transform, dsize=None, antialias=True, interpolation='linear'):
        # Warping does nothing to nans, except maybe changing the dsize
        new = self.__class__(dsize, channels=self.channels)
        return new


class DelayedIdentity(DelayedImageLeaf2):
    """
    Returns an ndarray as-is

    Example:
        self = DelayedNans((10, 10), channel_spec.FusedChannelSpec.coerce('rgb'))
        region_slices = (slice(5, 10), slice(1, 12))
        delayed = self.crop(region_slices)

    Example:
        >>> from kwcoco.util.delayed_poc.delayed2 import *  # NOQA
        >>> import kwcoco
        >>> arr = kwimage.checkerboard()
        >>> self = DelayedIdentity(arr, channels='gray')
        >>> warp = self.warp({'scale': 1.07})
        >>> warp.optimize().finalize()
    """
    def __init__(self, data, channels=None):
        super().__init__(channels=channels)
        self.data = data
        self.meta['num_bands'] = kwimage.num_channels(data)
        self.meta['dsize'] = data.shape[0:2][::-1]

    def finalize(self):
        return self.data


# -----
# Tests
# -----


@profile
def shuffle_operations_test():
    """
    CommandLine:
        XDEV_PROFILE=1 xdoctest -m /home/joncrall/code/kwcoco/kwcoco/util/delayed_poc/delayed2.py shuffle_operations_test
    """
    # Try putting operations in differnet orders and ensure optimize always
    # fixes it.

    fpath = kwimage.grab_test_image_fpath(overviews=3)
    base = DelayedLoad2(fpath, channels='r|g|b')._load_metadata()
    quantization = {'quant_max': 255, 'nodata': 0}
    base.get_overview(1).dequantize(quantization).optimize()

    operations = [
        ('warp', {'scale': 1}),
        ('crop', (slice(None), slice(None))),
        ('get_overview', 1),
        ('dequantize', quantization),
    ]

    dequant_idx = [t[0] for t in operations].index('dequantize')

    rng = kwarray.ensure_rng(None)

    # Repeat the test multiple times.
    num_times = 10
    for _ in range(num_times):

        num_ops = rng.randint(1, 30)
        op_idxs = rng.randint(0, len(operations), size=num_ops)

        # Don't allow dequantize more than once
        keep_flags = op_idxs != dequant_idx
        if not np.all(keep_flags):
            keeper = rng.choice(np.where(~keep_flags)[0])
            keep_flags[keeper] = True
        op_idxs = op_idxs[keep_flags]

        delayed = base
        for idx in op_idxs:
            name, args = operations[idx]
            func = getattr(delayed, name)
            delayed = func(args)

        # delayed.write_network_text(with_labels="name")
        opt = delayed.optimize()
        # opt.write_network_text(with_labels="name")

        # We always expect that we will get a sequence in the form
        expected_sequence = ['Warp2', 'Dequantize2', 'Crop2', 'Overview2', 'Load2']
        # But we are allowed to skip steps
        import networkx as nx
        graph = opt.as_graph()
        node_order = list(nx.topological_sort(graph))
        opname_order = [graph.nodes[n]['name'] for n in node_order]
        if opname_order[-1] != expected_sequence[-1]:
            raise AssertionError('Unexpected sequence')
        prev_idx = -1
        for opname in opname_order:
            this_idx = expected_sequence.index(opname)
            if this_idx <= prev_idx:
                raise AssertionError('Unexpected sequence')
            prev_idx = this_idx
