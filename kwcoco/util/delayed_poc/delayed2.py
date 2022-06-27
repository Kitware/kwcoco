"""
A rewrite of the delayed operations

TODO:
    The optimize logic could likley be better expressed as some sort of
    AST transformer.
"""

import ubelt as ub
import numpy as np
import kwarray
import kwimage
import copy


DEBUG = ub.identity
# DEBUG = print


class DelayedArray2(ub.NiceRepr):
    def __init__(self, subdata=None):
        self.subdata = subdata
        self.meta = {}

    def __nice__(self):
        return ''
        # return '{}'.format(self.shape)

    # @property
    # def shape(self):
    #     # shape = self.meta.get('shape', None)
    #     # if shape is None and self.subdata is not None:
    #     shape = self.subdata.shape
    #     return shape

    def nesting(self):
        def _child_nesting(child):
            if hasattr(child, 'nesting'):
                return child.nesting()
            elif isinstance(child, np.ndarray):
                return {
                    'type': 'ndarray',
                    'shape': self.subdata.shape,
                }
        if self.subdata is None:
            child_nodes = []
        else:
            child_nodes = [self.subdata]
        child_nestings = [_child_nesting(child) for child in child_nodes]
        item = {
            'type': self.__class__.__name__,
            'meta': self.meta,
        }
        if child_nestings:
            item['children'] = child_nestings
        return item

    def as_graph(self):
        """
        graph = self.as_graph()
        nx.write_network_text(graph)
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

            graph.nodes[node_id]['label'] = f'{name} {param_key}'

            if item.subdata is None:
                child_nodes = []
            else:
                child_nodes = [item.subdata]

            for child in child_nodes:
                child_id = id(child)
                graph.add_edge(node_id, child_id)
                stack.append(child)
        return graph

    def write_network_text(self):
        import networkx as nx
        graph = self.as_graph()
        nx.write_network_text(graph)

        # child_nestings = [_child_nesting(child) for child in child_nodes]
        # item = {
        #     'type': self.__class__.__name__,
        #     'meta': self.meta,
        # }
        # if child_nestings:
        #     item['children'] = child_nestings


class DelayedImage2(DelayedArray2):
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
        Args:
            space_slice (Tuple[slice, slice]): y-slice and x-slice.
        """
        new = DelayedCrop2(self, space_slice)
        return new

    def warp(self, transform, dsize='auto', antialias=True, interpolation='linear'):
        new = DelayedWarp2(self, transform, dsize=dsize, antialias=antialias,
                           interpolation=interpolation)
        return new

    def dequantize(self, quantization):
        new = DelayedDequantize2(self, quantization)
        return new

    def get_overview(self, overview):
        new = DelayedOverview2(self, overview)
        return new

    def finalize(self):
        raise NotImplementedError

    def optimize(self):
        DEBUG(f'Optimize {self.__class__.__name__}')
        raise NotImplementedError


class DelayedWarp2(DelayedImage2):
    """
    Example:
        >>> from kwcoco.util.delayed_poc.delayed2 import *  # NOQA
        >>> self = DelayedLoad2.demo(dsize=(16, 16))._load_metadata()
        >>> warp1 = self.warp({'scale': 3})
        >>> warp2 = warp1.warp({'theta': 0.1})
        >>> warp3 = warp2.optimize_fused_warps()
        >>> print(ub.repr2(warp2.nesting(), nl=-1, sort=0))
        >>> print(ub.repr2(warp3.nesting(), nl=-1, sort=0))
    """
    def __init__(self, subdata, transform, antialias=True, dsize='auto',
                 interpolation='linear'):
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

    def optimize_fused_warps(self):
        """
        Combine two consecutive warps into a single node.
        """
        DEBUG('Optimize optimize_fused_warps')
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

    def optimize_split_overview(self):
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
        DEBUG('Optimize optimize_split_overview')
        inner_data = self.subdata
        num_overviews = inner_data.num_overviews
        if num_overviews:
            transform = self.meta['transform']
            params = transform.decompose()
            sx, sy = params['scale']
            if sx < 1 and sy < 1:
                # Note: we don't know how many overviews the underlying
                # image will have, if any, we we can't assume this will get
                # us anywhere.
                from kwimage.im_cv2 import _prepare_scale_residual
                num_downs, residual_sx, residual_sy = _prepare_scale_residual(
                    sx, sy, fudge=0)
                # Only use as many downs as we have overviews
                can_do = min(num_overviews, num_downs)
                if can_do > 0:
                    overview_transform = kwimage.Affine.scale(1 / (2 ** can_do))
                    # This is the transform that is implicit in using an overview
                    # Let T be the original tranform we want
                    # Let O be our implicit overview transform
                    # Let R be the residual that we would need to apply to get
                    # our full transform if we started from a pre-scaled
                    # overview.
                    # T = R @ O
                    # ...
                    # T @ O.inv = R @ O @ O.inv
                    # T @ O.inv = R
                    residual_transform = transform @ overview_transform.inv()
                    new_transform = residual_transform

                    dsize = self.meta['dsize']
                    common_meta = ub.dict_isect(self.meta, {'antialias', 'interpolation'})

                    overview = inner_data.get_overview(can_do)
                    if np.allclose(np.array(new_transform) - np.eye(3), 0):
                        new = overview
                    else:
                        new = overview.warp(new_transform, dsize=dsize, **common_meta)
                    return new
        return self

    def optimize(self):
        DEBUG(f'Optimize {self.__class__.__name__}')
        new = copy.copy(self)
        new.subdata = self.subdata.optimize()
        if isinstance(new.subdata, DelayedWarp2):
            new = new.optimize_fused_warps()
            # new = new.optimize()
        if not isinstance(new.subdata, DelayedOverview2):
            split = new.optimize_split_overview()
            if new is not split:
                new = split
                new.subdata = new.subdata.optimize()
        # if isinstance(new.subdata, DelayedDequantize2):
        #     # Swap order so dequantize is after the crop
        #     quantization = new.subdata.meta['quantization']
        #     new.subdata = new.subdata.subdata.optimize()
        #     new = new.dequantize(quantization)
        #     new = new.optimize()
        return new


class DelayedDequantize2(DelayedImage2):
    def __init__(self, subdata, quantization):
        super().__init__(subdata)
        self.meta['quantization'] = quantization
        self.meta['dsize'] = subdata.dsize

    def finalize(self):
        from kwcoco.util.delayed_poc.helpers import dequantize  # NOQA
        quantization = self.meta['quantization']
        final = self.subdata.finalize()
        final = kwarray.atleast_nd(final, 3, front=False)
        if quantization is not None:
            final = dequantize(final, quantization)
        return final

    def optimize(self):
        DEBUG(f'Optimize {self.__class__.__name__}')
        new = copy.copy(self)
        new.subdata = self.subdata.optimize()

        if isinstance(new.subdata, DelayedWarp2):
            # Swap order so quantize is inside the warp
            new = copy.copy(new.subdata)
            new.subdata = new.subdata.dequantize(self.meta['quantization'])
            new = new.optimize()

        if isinstance(new.subdata, DelayedOverview2):
            # Swap order so quantize is inside the warp
            new = copy.copy(new.subdata)
            new.subdata = new.subdata.dequantize(self.meta['quantization'])
            new = new.optimize()
        return self


class DelayedCrop2(DelayedImage2):
    """
    Example:
        >>> from kwcoco.util.delayed_poc.delayed2 import *  # NOQA
        >>> self = DelayedLoad2.demo(dsize=(16, 16))._load_metadata()
        >>> crop1 = self[4:12, 0:16]
        >>> crop2 = crop1[2:6, 0:8]
        >>> crop3 = crop2.optimize_fused_crops()
        >>> print(ub.repr2(crop3.nesting(), nl=-1, sort=0))
    """
    def __init__(self, subdata, space_slice):
        super().__init__(subdata)
        # TODO: are we doing infinite padding or clipping?
        # This assumes infinite padding
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

    def optimize_fused_crops(self):
        """
        Combine two consecutive crops into a single node.
        """
        DEBUG('Optimize optimize_fused_crops')
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

    def optimize_warp_crop(self):
        """
        Given a inner warp and an outer crop, swap the order to an inner crop
        and an outer warp. This moves the crop closer to the leaf (i.e. data on
        disk), which is more efficient.

        Example:
            >>> from kwcoco.util.delayed_poc.delayed2 import *  # NOQA
            >>> fpath = kwimage.grab_test_image_fpath()
            >>> node0 = DelayedLoad2(fpath, channels='r|g|b')._load_metadata()
            >>> node1 = node0.warp({'scale': 0.432, 'theta': np.pi / 3, 'about': (80, 80), 'shearx': .3, 'offset': (-50, -50)})
            >>> node2 = node1[10:50, 1:40]
            >>> self = node2
            >>> new_outer = node2.optimize_warp_crop()
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
            >>> new_outer = node2.optimize_warp_crop()
            >>> print(ub.repr2(node2.nesting(), nl=-1, sort=0))
            >>> print(ub.repr2(new_outer.nesting(), nl=-1, sort=0))
        """
        DEBUG('Optimize optimize_warp_crop')
        assert isinstance(self.subdata, DelayedWarp2)
        # Inner is the data closer to the leaf (disk), outer is the data closer
        # to the user (output).
        outer_slices = self.meta['space_slice']
        inner_transform = self.subdata.meta['transform']

        outer_region = kwimage.Boxes.from_slice(outer_slices)
        outer_region = outer_region.to_polygons()[0]

        from kwcoco.util.delayed_poc.helpers import _compute_leaf_subcrop
        inner_slice, outer_transform = _compute_leaf_subcrop(
            outer_region, inner_transform)

        warp_meta = ub.dict_isect(self.meta, {
            'dsize', 'antialias', 'interpolation'})

        new_inner = self.subdata.subdata.crop(inner_slice)
        new_outer = new_inner.warp(outer_transform, **warp_meta)
        return new_outer

    def optimize(self):
        DEBUG(f'Optimize {self.__class__.__name__}')
        new = copy.copy(self)
        new.subdata = self.subdata.optimize()
        if isinstance(new.subdata, DelayedCrop2):
            new = new.optimize_fused_crops()
            # new = new.optimize()
        if isinstance(new.subdata, DelayedWarp2):
            new = new.optimize_warp_crop()
            new = new.optimize()
        if isinstance(new.subdata, DelayedDequantize2):
            # Swap order so dequantize is after the overview
            quantization = new.subdata.meta['quantization']
            new.subdata = new.subdata.subdata.optimize()
            new = new.dequantize(quantization)
            new = new.optimize()
        return new


class DelayedOverview2(DelayedImage2):
    """
    Ignore:
        # check the formula for overview
        imdata = np.random.rand(31, 29)
        kwimage.imwrite('foo.tif', imdata, backend='gdal', overviews=3)
        ub.cmd('gdalinfo foo.tif', verbose=3)

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
        super().__init__(subdata)
        self.meta['overview'] = overview
        w, h = subdata.dsize
        sf = 1 / (2 ** overview)
        # The rounding operation for gdal overviews is ceiling
        def iceil(x):
            return int(np.ceil(x))
        w = iceil(sf * w)
        h = iceil(sf * h)
        self.meta['dsize'] = (w, h)

    @property
    def num_overviews(self):
        num_remain = self.subdata.num_overviews - self.meta['overview']
        return num_remain

    def finalize(self):
        sub_final = self.subdata.finalize()
        if not hasattr(sub_final, 'get_overview'):
            import warnings
            warnings.warn('the underlying data does not have overviews. Recasting as a resize')
            final = kwimage.imresize(
                sub_final,
                scale=1 / 2 ** self.meta['overview'],
                interpolation='nearest',
                # antialias=True
            )
        else:
            final = sub_final.get_overview(self.meta['overview'])
        return final

    def optimize_crop_overview(self):
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
        assert isinstance(self.subdata, DelayedCrop2)
        # Inner is the data closer to the leaf (disk), outer is the data closer
        # to the user (output).
        outer_overview = self.meta['overview']
        inner_slices = self.subdata.meta['space_slice']

        sf = 1 / 2 ** outer_overview
        outer_transform = kwimage.Affine.scale(sf)

        inner_region = kwimage.Boxes.from_slice(inner_slices)
        inner_region = inner_region.to_polygons()[0]

        from kwcoco.util.delayed_poc.helpers import _swap_crop_transform2
        new_inner_warp, outer_crop, new_outer_warp = _swap_crop_transform2(
            inner_region, outer_transform)

        # Move the overview to the inside, it should be unchanged
        new = self.subdata.subdata.get_overview(outer_overview)

        # Move the crop to the outside
        new = new.crop(outer_crop)

        if not np.all(np.isclose(np.eye(3), new_outer_warp)):
            # we might have to apply an additional warp at the end.
            new = new.warp(new_outer_warp)
        return new

    def optimize_fuse_overviews(self):
        assert isinstance(self.subdata, DelayedOverview2)
        outer_overview = self.meta['overview']
        inner_overrview = self.subdata.meta['overview']
        new_overview = inner_overrview + outer_overview
        new = self.subdata.subdata.get_overview(new_overview)
        return new

    def optimize_overview_warp(self):
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

    def optimize(self):
        DEBUG(f'Optimize {self.__class__.__name__}')
        new = copy.copy(self)
        new.subdata = self.subdata.optimize()
        if isinstance(new.subdata, DelayedOverview2):
            new = new.optimize_fuse_overviews()

        if isinstance(new.subdata, DelayedCrop2):
            new = new.optimize_crop_overview()
            new = new.optimize()

        if isinstance(new.subdata, DelayedWarp2):
            new = new.optimize_overview_warp()
            new = new.optimize()
        return new


class DelayedLoad2(DelayedImage2):
    """
    Example:
        from kwcoco.util.delayed_poc.delayed2 import *  # NOQA
        self = DelayedLoad2.demo(dsize=(16, 16))._load_metadata()
        print(f'self={self}')
        print('self.meta = {}'.format(ub.repr2(self.meta, nl=1)))
        data1 = self.finalize()


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
        >>>     'orig_dtype': 'float32',
        >>>     'orig_min': 0,
        >>>     'orig_max': 1,
        >>>     'quant_min': 0,
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
        super().__init__(channels=channels, dsize=dsize)
        self.fpath = fpath
        self.lazy_ref = None

    @classmethod
    def demo(DelayedLoad2, key='astro', dsize=None):
        fpath = kwimage.grab_test_image_fpath(key, dsize=dsize)
        self = DelayedLoad2(fpath)
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
            import kwarray
            import warnings
            warnings.warn('DelayedLoad2 may not be efficient without gdal')
            # TODO: delay even further with gdal
            pre_final = kwimage.imread(self.fpath)
            pre_final = kwarray.atleast_nd(pre_final, 3)
        else:
            return self.lazy_ref

    def optimize(self):
        """
        Example:
            >>> # Make a complex chain of operations and optimize it
            >>> from kwcoco.util.delayed_poc.delayed2 import *  # NOQA
            >>> import kwimage
            >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
            >>> dimg = DelayedLoad2(fpath, channels='r|g|b')._load_metadata()
            >>> quantization = {
            >>>     'orig_dtype': 'float32',
            >>>     'orig_min': 0,
            >>>     'orig_max': 1,
            >>>     'quant_min': 0,
            >>>     'quant_max': 255,
            >>>     'nodata': 0,
            >>> }
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
            >>> print(ub.repr2(dimg.nesting(), nl=-1, sort=0))
            >>> dopt = dimg.optimize()
            >>> if 1:
            >>>     import networkx as nx
            >>>     dimg.write_network_text()
            >>>     dopt.write_network_text()
            >>> print(ub.repr2(dopt.nesting(), nl=-1, sort=0))
            >>> final0 = dimg.finalize()
            >>> final1 = dopt.finalize()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(final0, pnum=(1, 2, 1), fnum=1, title='raw')
            >>> kwplot.imshow(final1, pnum=(1, 2, 2), fnum=1, title='optimized')
        """
        DEBUG(f'Optimize {self.__class__.__name__}')
        return self
