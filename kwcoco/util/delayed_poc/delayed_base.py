"""
Base classes for delayed operations
"""
import ubelt as ub
import numpy as np
import kwimage
# from abc import ABC, abstractmethod


try:
    import xarray as xr
except ImportError:
    xr = None

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


# DEBUG_PRINT = ub.identity
# DEBUG_PRINT = print


class DelayedArray(ub.NiceRepr):
    """
    Generalized ndoperations
    """

    def __nice__(self):
        return '{}'.format(self.shape)

    # @abstractmethod
    def finalize(self, **kwargs):
        raise NotImplementedError

    # @abstractmethod
    def children(self):
        """
        Abstract method, which should generate all of the direct children of a
        node in the operation tree.

        Yields:
            DelayedVisionMixin:
        """
        raise NotImplementedError

    def _optimize_paths(self, **kwargs):
        """
        Iterate through the leaf nodes, which are virtually transformed into
        the root space.

        This returns some sort of hueristically optimized leaf repr wrt warps.
        """
        # DEBUG_PRINT('DelayedVisionMixin._optimize_paths {}'.format(type(self)))
        for child in self.children():
            yield from child._optimize_paths(**kwargs)

    def __json__(self):
        from kwcoco.util.util_json import ensure_json_serializable
        json_dict = ensure_json_serializable(self.nesting())
        return json_dict

    def nesting(self):
        def _child_nesting(child):
            if hasattr(child, 'nesting'):
                return child.nesting()
            elif isinstance(child, np.ndarray):
                return {
                    'type': 'ndarray',
                    'shape': self.sub_data.shape,
                }
        children = [_child_nesting(child) for child in self.children()]
        item = {
            'type': self.__class__.__name__,
            'meta': self.meta,
        }
        if children:
            item['children'] = children
        return item


class DelayedVisionMixin:
    """
    Specific case of array operations pertaining to vision use-cases

    Base class for nodes in a tree of delayed computer-vision operations
    """
    def __nice__(self):
        channels = self.channels
        return '{}, {}'.format(self.shape, channels)

    # @abstractmethod
    def warp(self, *args, **kwargs):
        raise NotImplementedError

    # @abstractmethod
    def crop(self, *args, **kwargs):
        raise NotImplementedError

    # Backwards compatibility
    def delayed_warp(self, *args, **kwargs):
        return self.warp(*args, **kwargs)

    def delayed_crop(self, *args, **kwargs):
        return self.crop(*args, **kwargs)


class DelayedVideo(DelayedArray, DelayedVisionMixin):
    """
    Specific vision use case for stacked 2D images in over time as a 3D tube
    """
    pass


class DelayedImage(DelayedArray, DelayedVisionMixin):
    """
    Specific vision use case for 2D images with some number of channels
    """

    @profile
    def crop(self, region_slices):
        """
        Create a new delayed image that performs a crop in the transformed
        "self" space.

        Args:
            region_slices (Tuple[slice, slice]): y-slice and x-slice.

        Note:
            Returns a heuristically "simplified" tree. In the current
            implementation there are only 3 operations, cat, warp, and crop.
            All cats go at the top, all crops go at the bottom, all warps are
            in the middle.

        Returns:
            DelayedImage: lazy executed delayed transform

        Example:
            >>> from kwcoco.util.delayed_poc.delayed_nodes import DelayedWarp
            >>> dsize = (100, 100)
            >>> tf2 = kwimage.Affine.affine(scale=3).matrix
            >>> self = DelayedWarp(np.random.rand(33, 33), tf2, dsize)
            >>> region_slices = (slice(5, 10), slice(1, 12))
            >>> crop = self.crop(region_slices)
            >>> print(ub.repr2(crop.nesting(), nl=-1, sort=0))
            >>> crop.finalize()

        Example:
            >>> from kwcoco.util.delayed_poc.delayed_leafs import DelayedLoad
            >>> chan1 = DelayedLoad.demo('astro')
            >>> chan2 = DelayedLoad.demo('carl')
            >>> warped1a = chan1.warp(kwimage.Affine.scale(1.2).matrix)
            >>> warped2a = chan2.warp(kwimage.Affine.scale(1.5))
            >>> warped1b = warped1a.warp(kwimage.Affine.scale(1.2).matrix)
            >>> warped2b = warped2a.warp(kwimage.Affine.scale(1.5))
            >>> #
            >>> region_slices = (slice(97, 677), slice(5, 691))
            >>> self = warped2b
            >>> #
            >>> crop1 = warped1b.crop(region_slices)
            >>> crop2 = warped2b.crop(region_slices)
            >>> print(ub.repr2(warped1b.nesting(), nl=-1, sort=0))
            >>> print(ub.repr2(warped2b.nesting(), nl=-1, sort=0))
            >>> # Notice how the crop merges the two nesting layers
            >>> # (via the hueristic optimize step)
            >>> print(ub.repr2(crop1.nesting(), nl=-1, sort=0))
            >>> print(ub.repr2(crop2.nesting(), nl=-1, sort=0))
            >>> frame1 = crop1.finalize(dsize=(500, 500))
            >>> frame2 = crop2.finalize(dsize=(500, 500))
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(frame1, pnum=(1, 2, 1), fnum=1)
            >>> kwplot.imshow(frame2, pnum=(1, 2, 2), fnum=1)

        """
        from kwcoco.util.delayed_poc.delayed_nodes import DelayedWarp
        from kwcoco.util.delayed_poc.delayed_nodes import DelayedCrop
        from kwcoco.util.delayed_poc.delayed_nodes import DelayedChannelStack
        from kwcoco.util.delayed_poc.delayed_leafs import DelayedLoad
        from kwcoco.util.delayed_poc.delayed_leafs import DelayedNans
        from kwcoco.util.delayed_poc.delayed_nodes import _compute_leaf_subcrop
        # DEBUG_PRINT('DelayedImage.crop: {}'.format(type(self)))
        if region_slices is None:
            return self
        components = []

        for delayed_leaf in self._optimize_paths():
            # DEBUG_PRINT('delayed_leaf = {!r}'.format(delayed_leaf))
            # DEBUG_PRINT('delayed_leaf.sub_data = {!r}'.format(delayed_leaf.sub_data))
            # Compute, sub_crop_slices, and new tf_newleaf_to_newroot
            assert isinstance(delayed_leaf, DelayedWarp)  # HACK
            tf_leaf_to_root = delayed_leaf.meta['transform']

            root_region_box = kwimage.Boxes.from_slice(
                region_slices, shape=delayed_leaf.shape)
            root_region_bounds = root_region_box.to_polygons()[0]

            w = root_region_box.width.ravel()[0]
            h = root_region_box.height.ravel()[0]
            root_dsize = (w, h)

            leaf_crop_slices, tf_newleaf_to_newroot = _compute_leaf_subcrop(
                root_region_bounds, tf_leaf_to_root)

            delayed_leaf.sub_data

            if isinstance(delayed_leaf.sub_data, (DelayedLoad, DelayedNans)):
                # if hasattr(delayed_leaf.sub_data, 'crop'):
                # Hack
                crop = delayed_leaf.sub_data.crop(leaf_crop_slices)
            else:
                crop = DelayedCrop(delayed_leaf.sub_data, leaf_crop_slices)

            warp = DelayedWarp(crop, tf_newleaf_to_newroot, dsize=root_dsize)
            components.append(warp)

        if len(components) == 0:
            print('self = {!r}'.format(self))
            raise ValueError('Did not find any componets')
        if len(components) == 1:
            return components[0]
        else:
            return DelayedChannelStack(components)

    def warp(self, transform, dsize=None):
        """
        Delayed transform the underlying data.

        Note:
            this deviates from kwimage warp functions because instead of
            "output_dims" (specified in c-style shape) we specify dsize (w, h).

        Args:
            transform (Any): coercable transform
            dsize (Tuple[int, int] | None): output width/height

        Returns:
            kwcoco.util.delayed_poc.delayed_nodes.DelayedWarp: new delayed transform a chained transform
        """
        from kwcoco.util.delayed_poc.delayed_nodes import DelayedWarp
        warped = DelayedWarp(self, transform=transform, dsize=dsize)
        return warped

    # @abstractmethod
    def take_channels(self, channels):
        """
        Args:
            channels (Any):

        Returns:
            DelayedVisionMixin :
                delayed operation only on specified channels
        """
        raise NotImplementedError
