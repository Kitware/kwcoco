"""
Intermediate nodes for delayed operations
"""
import ubelt as ub
import numpy as np
import kwimage
import kwarray
from kwcoco import channel_spec
from kwcoco import exceptions
from kwcoco.util.delayed_poc.delayed_base import DelayedImage
from kwcoco.util.delayed_poc.delayed_base import DelayedVideo
# from kwcoco.util.delayed_poc.delayed_base import DelayedArrayOperation
from kwcoco.util.delayed_poc.delayed_base import DelayedVisionMixin  # NOQA
from kwcoco.util.delayed_poc.delayed_base import DelayedArray  # NOQA
from kwcoco.util.delayed_poc.helpers import _auto_dsize
from kwcoco.util.delayed_poc.helpers import _largest_shape
from kwcoco.util.delayed_poc.helpers import _swap_warp_after_crop  # NOQA

try:
    import xarray as xr
except ImportError:
    xr = None

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


class JaggedArray(ub.NiceRepr):
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


class StackMixin:
    def __init__(self, parts, axis):
        self.parts = parts
        self.axis = axis

    def children(self):
        """
        Yields:
            Any:
        """
        yield from self.parts

    @property
    def shape(self):
        stack_dims = [part.shape for part in self.parts]
        n = len(self.parts)
        axis = self.axis
        common_shape = [max(stack_dim) for stack_dim in zip(*stack_dims)]
        shape = tuple([*common_shape[0:axis], n, *common_shape[axis:]])
        return shape

    @profile
    def finalize(self, **kwargs):
        """
        Execute the final transform
        """
        # Add in the video axis
        # as_xarray = kwargs.get('as_xarray', False)
        stack = [part.finalize(**kwargs) for part in self.parts]
        stack_dims = np.array([s.shape for s in stack])
        max_dims = stack_dims.max(axis=0)
        delta_dims = max_dims - stack_dims
        if np.any(delta_dims):
            padded_stack = []
            for delta, item in zip(delta_dims, stack):
                pad_width = list(zip([0] * len(delta), delta))
                item = np.pad(item, pad_width=pad_width)
                padded_stack.append(item)
            final = np.stack(padded_stack, axis=self.axis)
        else:
            final = np.stack(stack, axis=self.axis)
        return final


class DelayedStack(StackMixin, DelayedArray):
    """
    Generalize stacking. Its like concat, but in a new dimension.

    Example:
        >>> from kwcoco.util import delayed_poc as delayarr
        >>> arr1 = delayarr.DelayedIdentity.demo(dsize=(32, 32), chan=[0, 2])
        >>> arr2 = delayarr.DelayedIdentity.demo(dsize=(32, 32), chan=[1, 0])
        >>> self = DelayedStack([arr1, arr2], axis=0)
        >>> print(f'self={self}')
        >>> final = self.finalize()
        >>> print(f'final.shape={final.shape}')

        >>> from kwcoco.util import delayed_poc as delayarr
        >>> arr1 = delayarr.DelayedIdentity.demo(dsize=(32, 32), chan=[0, 2])
        >>> arr2 = delayarr.DelayedIdentity.demo(dsize=(8, 32), chan=[1, 0])
        >>> arr3 = delayarr.DelayedIdentity.demo(dsize=(16, 42), chan=[1, 0])
        >>> arr4 = delayarr.DelayedIdentity.demo(dsize=(64, 8), chan=[1, 0])
        >>> self = DelayedStack([arr1, arr2, arr3, arr4], axis=0)
        >>> print(f'self={self}')
        >>> final = self.finalize()
        >>> print(f'final.shape={final.shape}')
    """


class DelayedFrameStack(StackMixin, DelayedVideo):
    """
    Represents multiple frames in a video

    TODO:
        rename to DelayedFrameStack

    Note:

        .. code::

            Video[0]:
                Frame[0]:
                    Chan[0]: (32) +--------------------------------+
                    Chan[1]: (16) +----------------+
                    Chan[2]: ( 8) +--------+
                Frame[1]:
                    Chan[0]: (30) +------------------------------+
                    Chan[1]: (14) +--------------+
                    Chan[2]: ( 6) +------+

    TODO:
        - [ ] Support computing the transforms when none of the data is loaded

    Example:
        >>> # Simpler case with fewer nesting levels
        >>> from kwcoco.util.delayed_poc.delayed_nodes import *  # NOQA
        >>> from kwcoco.util.delayed_poc.delayed_leafs import DelayedLoad
        >>> import kwarray
        >>> rng = kwarray.ensure_rng(None)
        >>> # Delayed warp each channel into its "image" space
        >>> # Note: the images never enter the space we transform through
        >>> f1_img = DelayedLoad.demo('astro', (300, 300))
        >>> f2_img = DelayedLoad.demo('carl', (256, 256))
        >>> # Combine frames into a video
        >>> vid_dsize = np.array((100, 100))
        >>> self = vid = DelayedFrameStack([
        >>>     f1_img.warp(kwimage.Affine.scale(vid_dsize / f1_img.dsize)),
        >>>     f2_img.warp(kwimage.Affine.scale(vid_dsize / f2_img.dsize)),
        >>> ], dsize=vid_dsize)
        >>> print(ub.repr2(vid.nesting(), nl=-1, sort=0))
        >>> final = vid.finalize(interpolation='nearest', dsize=(32, 32))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(final[0], pnum=(1, 2, 1), fnum=1)
        >>> kwplot.imshow(final[1], pnum=(1, 2, 2), fnum=1)
        >>> region_slices = (slice(0, 90), slice(30, 60))
    """
    def __init__(self, frames, dsize=None):
        self.parts = frames
        self.axis = 0
        if dsize is None:
            dsize_cands = [frame.dsize for frame in self.parts]
            dsize = _largest_shape(dsize_cands)

        self.dsize = dsize
        nband_cands = [frame.num_bands for frame in self.parts]
        if any(c is None for c in nband_cands):
            num_bands = None
        if ub.allsame(nband_cands):
            num_bands = nband_cands[0]
        else:
            raise exceptions.CoordinateCompatibilityError(
                'components must all have the same delayed size: got {}'.format(nband_cands))
        self.num_bands = num_bands
        self.num_frames = len(self.parts)
        self.meta = {
            'num_bands': self.num_bands,
            'num_frames': self.num_frames,
            'shape': self.shape,
        }

    @property
    def frames(self):
        return self.parts

    def children(self):
        """
        Yields:
            Any:
        """
        yield from self.parts

    @property
    def channels(self):
        # Assume all channels are the same, or at least aligned via nans?
        return self.frames[0].channels

    @property
    def shape(self):
        w, h = self.dsize
        return (self.num_frames, h, w, self.num_bands)

    def crop(self, region_slices):
        """
        Example:
            >>> from kwcoco.util.delayed_poc.delayed_nodes import *  # NOQA
            >>> from kwcoco.util.delayed_poc.delayed_leafs import DelayedIdentity
            >>> # Create raw channels in some "native" resolution for frame 1
            >>> f1_chan1 = DelayedIdentity.demo('astro', chan=(1, 0), dsize=(300, 300))
            >>> f1_chan2 = DelayedIdentity.demo('astro', chan=2, dsize=(10, 10))
            >>> # Create raw channels in some "native" resolution for frame 2
            >>> f2_chan1 = DelayedIdentity.demo('carl', dsize=(64, 64), chan=(1, 0))
            >>> f2_chan2 = DelayedIdentity.demo('carl', dsize=(10, 10), chan=2)
            >>> #
            >>> f1_dsize = np.array(f1_chan1.dsize)
            >>> f2_dsize = np.array(f2_chan1.dsize)
            >>> f1_img = DelayedChannelStack([
            >>>     f1_chan1.warp(kwimage.Affine.scale(f1_dsize / f1_chan1.dsize), dsize=f1_dsize),
            >>>     f1_chan2.warp(kwimage.Affine.scale(f1_dsize / f1_chan2.dsize), dsize=f1_dsize),
            >>> ])
            >>> f2_img = DelayedChannelStack([
            >>>     f2_chan1.warp(kwimage.Affine.scale(f2_dsize / f2_chan1.dsize), dsize=f2_dsize),
            >>>     f2_chan2.warp(kwimage.Affine.scale(f2_dsize / f2_chan2.dsize), dsize=f2_dsize),
            >>> ])
            >>> vid_dsize = np.array((280, 280))
            >>> full_vid = DelayedFrameStack([
            >>>     f1_img.warp(kwimage.Affine.scale(vid_dsize / f1_img.dsize), dsize=vid_dsize),
            >>>     f2_img.warp(kwimage.Affine.scale(vid_dsize / f2_img.dsize), dsize=vid_dsize),
            >>> ])
            >>> region_slices = (slice(80, 200), slice(80, 200))
            >>> print(ub.repr2(full_vid.nesting(), nl=-1, sort=0))
            >>> crop_vid = full_vid.crop(region_slices)
            >>> final_full = full_vid.finalize(interpolation='nearest')
            >>> final_crop = crop_vid.finalize(interpolation='nearest')
            >>> import pytest
            >>> with pytest.raises(ValueError):
            >>>     # should not be able to crop a crop yet
            >>>     crop_vid.crop(region_slices)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(final_full[0], pnum=(2, 2, 1), fnum=1)
            >>> kwplot.imshow(final_full[1], pnum=(2, 2, 2), fnum=1)
            >>> kwplot.imshow(final_crop[0], pnum=(2, 2, 3), fnum=1)
            >>> kwplot.imshow(final_crop[1], pnum=(2, 2, 4), fnum=1)
        """
        # DEBUG_PRINT('DelayedFrameStack.crop')
        new_frames = []
        for frame in self.frames:
            new_frame = frame.crop(region_slices)
            new_frames.append(new_frame)
        new = DelayedFrameStack(new_frames)
        return new

    def warp(self, transform, dsize=None):
        """
        Delayed transform the underlying data.

        Note:
            this deviates from kwimage warp functions because instead of
            "output_dims" (specified in c-style shape) we specify dsize (w, h).

        Returns:
            DelayedWarp : new delayed transform a chained transform
        """
        # warped = DelayedWarp(self, transform=transform, dsize=dsize)
        # return warped

        if dsize is None:
            dsize = self.dsize
        elif isinstance(dsize, str):
            if dsize == 'auto':
                dsize = _auto_dsize(transform, self.dsize)
        new_frames = []
        for frame in self.frames:
            new_frame = frame.warp(transform, dsize=dsize)
            new_frames.append(new_frame)
        new = DelayedFrameStack(new_frames)
        return new


class DelayedChannelStack(StackMixin, DelayedImage):
    """
    Represents multiple channels in an image that could be concatenated

    Attributes:
        components (List[DelayedWarp]): a list of stackable channels. Each
            component may be comprised of multiple channels.

    TODO:
        - [ ] can this be generalized into a delayed concat and combined with DelayedFrameStack?
        - [ ] can all concats be delayed until the very end?

    Example:
        >>> from kwcoco.util.delayed_poc.delayed_nodes import *  # NOQA
        >>> # Create 3 delayed operations to concatenate
        >>> comp1 = DelayedWarp(np.random.rand(11, 7))
        >>> comp2 = DelayedWarp(np.random.rand(11, 7, 3))
        >>> comp3 = DelayedWarp(
        >>>     np.random.rand(3, 5, 2),
        >>>     transform=kwimage.Affine.affine(scale=(7/5, 11/3)).matrix,
        >>>     dsize=(7, 11)
        >>> )
        >>> components = [comp1, comp2, comp3]
        >>> chans = DelayedChannelStack(components)
        >>> final = chans.finalize()
        >>> assert final.shape == chans.shape
        >>> assert final.shape == (11, 7, 6)

        >>> # We should be able to nest DelayedChannelStack inside virutal images
        >>> frame1 = DelayedWarp(
        >>>     chans, transform=kwimage.Affine.affine(scale=2.2).matrix,
        >>>     dsize=(20, 26))
        >>> frame2 = DelayedWarp(
        >>>     np.random.rand(3, 3, 6), dsize=(20, 26))
        >>> frame3 = DelayedWarp(
        >>>     np.random.rand(3, 3, 6), dsize=(20, 26))

        >>> print(ub.repr2(frame1.nesting(), nl=-1, sort=False))
        >>> frame1.finalize()
        >>> vid = DelayedFrameStack([frame1, frame2, frame3])
        >>> print(ub.repr2(vid.nesting(), nl=-1, sort=False))

    Example:
        >>> from kwcoco.util.delayed_poc.delayed_nodes import *  # NOQA
        >>> # If requested, we can return arrays of the different sizes.
        >>> # but usually this will raise an error.
        >>> comp1 = DelayedWarp.random(dsize=(32, 32), nesting=(1, 5), channels=1)
        >>> comp2 = DelayedWarp.random(dsize=(8, 8), nesting=(1, 5), channels=1)
        >>> comp3 = DelayedWarp.random(dsize=(64, 64), nesting=(1, 5), channels=1)
        >>> components = [comp1, comp2, comp3]
        >>> self = DelayedChannelStack(components, jagged=True)
        >>> final = self.finalize()
        >>> print('final = {!r}'.format(final))
    """
    def __init__(self, components, dsize=None, jagged=False):
        if len(components) == 0:
            raise ValueError('No components to concatenate')
        self.components = components
        self.jagged = jagged
        if dsize is None and not jagged:
            dsize_cands = [comp.dsize for comp in self.components]
            if not ub.allsame(dsize_cands):
                raise exceptions.CoordinateCompatibilityError(
                    # 'components must all have the same delayed size')
                    'components must all have the same delayed size: got {}'.format(dsize_cands))
            dsize = dsize_cands[0]
        self.dsize = dsize
        try:
            self.num_bands = sum(comp.num_bands for comp in self.components)
        except TypeError:
            if any(comp.num_bands is None for comp in self.components):
                self.num_bands = None
            else:
                raise
        self.meta = {
            'shape': self.shape,
            'num_bands': self.num_bands,
        }

    def children(self):
        """
        Yields:
            Any
        """
        yield from self.components

    @classmethod
    def random(cls, num_parts=3, rng=None):
        """
        Example:
            >>> self = DelayedChannelStack.random()
            >>> print('self = {!r}'.format(self))
            >>> print(ub.repr2(self.nesting(), nl=-1, sort=0))
        """
        rng = kwarray.ensure_rng(rng)
        self_w = rng.randint(8, 64)
        self_h = rng.randint(8, 64)
        components = []
        for _ in range(num_parts):
            subcomp = DelayedWarp.random(rng=rng)
            tf = kwimage.Affine.random(rng=rng).matrix
            comp = subcomp.warp(tf, dsize=(self_w, self_h))
            components.append(comp)
        self = DelayedChannelStack(components)
        return self

    @property
    def channels(self):
        sub_channs = []
        for comp in self.components:
            comp_channels = comp.channels
            if comp_channels is None:
                return None
            sub_channs.append(comp_channels)
        channs = channel_spec.FusedChannelSpec.concat(sub_channs)
        return channs

    @property
    def shape(self):
        if self.jagged:
            w = h = None
        else:
            w, h = self.dsize
        return (h, w, self.num_bands)

    @profile
    def finalize(self, **kwargs):
        """
        Execute the final transform
        """
        as_xarray = kwargs.get('as_xarray', False)
        stack = [comp.finalize(**kwargs) for comp in self.components]
        if len(stack) == 1:
            final = stack[0]
        else:
            if self.jagged:
                final = JaggedArray(stack, axis=2)
            else:
                if as_xarray:
                    import xarray as xr
                    final = xr.concat(stack, dim='c')
                else:
                    final = np.concatenate(stack, axis=2)
        return final

    def warp(self, transform, dsize=None):
        """
        Delayed transform the underlying data.

        Note:
            this deviates from kwimage warp functions because instead of
            "output_dims" (specified in c-style shape) we specify dsize (w, h).

        Returns:
            DelayedWarp : new delayed transform a chained transform
        """
        if dsize is None:
            dsize = self.dsize
        elif isinstance(dsize, str):
            if dsize == 'auto':
                dsize = _auto_dsize(transform, self.dsize)
        new_parts = []
        for part in self.components:
            new_frame = part.warp(transform, dsize=dsize)
            new_parts.append(new_frame)
        new = DelayedChannelStack(new_parts)
        return new

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
            DelayedArray:
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
            >>> self = combo = DelayedChannelStack(delayed.components + [aligned])
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

        CommandLine:
            xdoctest -m /home/joncrall/code/kwcoco/kwcoco/util/delayed_poc.delayed_nodes.py DelayedChannelStack.take_channels:2 --profile

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
            comp.num_bands for comp in self.components])

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
                comp = self.components[outer]
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

        new = DelayedChannelStack(new_components, jagged=self.jagged)
        return new


class DelayedWarp(DelayedImage):
    """
    POC for chainable transforms

    Note:
        "sub" is used to refer to the underlying data in its native coordinates
        and resolution.

        "self" is used to refer to the data in the transformed coordinates that
        are exposed by this class.

    Attributes:

        sub_data (DelayedWarp | ArrayLike):
            array-like image data at a naitive resolution

        transform (kwimage.Transform):
            transforms data from native "sub"-image-space to
            "self"-image-space.

    Example:
        >>> dsize = (12, 12)
        >>> tf1 = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        >>> tf2 = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 1]])
        >>> tf3 = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 1]])
        >>> band1 = DelayedWarp(np.random.rand(6, 6), tf1, dsize)
        >>> band2 = DelayedWarp(np.random.rand(4, 4), tf2, dsize)
        >>> band3 = DelayedWarp(np.random.rand(3, 3), tf3, dsize)
        >>> #
        >>> # Execute a crop in a one-level transformed space
        >>> region_slices = (slice(5, 10), slice(0, 12))
        >>> crop = band2.crop(region_slices)
        >>> final_crop = crop.finalize()
        >>> #
        >>> # Execute a crop in a nested transformed space
        >>> tf4 = np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1]])
        >>> chained = DelayedWarp(band2, tf4, (18, 18))
        >>> crop = chained.crop(region_slices)
        >>> final_crop = crop.finalize()
        >>> #
        >>> tf4 = np.array([[.5, 0, 0], [0, .5, 0], [0, 0, 1]])
        >>> chained = DelayedWarp(band2, tf4, (6, 6))
        >>> crop = chained.crop(region_slices)
        >>> final_crop = crop.finalize()
        >>> #
        >>> region_slices = (slice(1, 5), slice(2, 4))
        >>> crop = chained.crop(region_slices)
        >>> final_crop = crop.finalize()

    Example:
        >>> dsize = (17, 12)
        >>> tf = np.array([[5.2, 0, 1.1], [0, 3.1, 2.2], [0, 0, 1]])
        >>> self = DelayedWarp(np.random.rand(3, 5, 13), tf, dsize=dsize)
        >>> self.finalize().shape
    """

    _DEFAULT_AUTO_OVERVIEW = 0

    def __init__(self, sub_data, transform=None, dsize=None):
        transform = kwimage.Affine.coerce(transform)

        # TODO: We probably don't need to track sub-bounds, size, shape
        # or any of that anywhere except at the root and leaf.

        try:
            if hasattr(sub_data, 'bounds'):
                sub_shape = sub_data.shape
                sub_bounds = sub_data.bounds
            else:
                sub_shape = sub_data.shape
                sub_h, sub_w = sub_shape[0:2]
                sub_bounds = kwimage.Coords(
                    np.array([[0,     0], [sub_w, 0],
                              [0, sub_h], [sub_w, sub_h]])
                )
            self.bounds = sub_bounds.warp(transform.matrix)
            if dsize is ub.NoParam:
                pass
            elif dsize is None:
                (h, w) = sub_shape[0:2]
                dsize = (w, h)
            elif isinstance(dsize, str):
                if dsize == 'auto':
                    # TODO: could use _auto_dsize
                    max_xy = np.ceil(self.bounds.data.max(axis=0))
                    max_x = int(max_xy[0])
                    max_y = int(max_xy[1])
                    dsize = (max_x, max_y)
                else:
                    raise KeyError(dsize)
            else:
                if isinstance(dsize, np.ndarray):
                    dsize = tuple(map(int, dsize))
                dsize = dsize

            if len(sub_data.shape) == 2:
                num_bands = 1
            elif len(sub_data.shape) == 3:
                num_bands = sub_data.shape[2]
            else:
                raise ValueError(
                    'Data may only have 2 space dimensions and 1 channel '
                    'dimension')
        except Exception:
            num_bands = None

        self.sub_data = sub_data
        self.meta = {
            'dsize': dsize,
            'num_bands': num_bands,
            'transform': transform,
        }

    @classmethod
    def random(cls,
               dsize=None,
               raw_width=(8, 64),
               raw_height=(8, 64),
               channels=(1, 5),
               nesting=(2, 5),
               rng=None):
        """
        Create a random delayed warp operation for testing / demo

        Args:
            dsize (Tuple[int, int] | None):
                The width and height of the finalized data.
                If unspecified, it will be a function of the random warps.

            raw_width (int | Tuple[int, int]):
                The exact or min / max width of the random raw data

            raw_height (int | Tuple[int, int]):
                The exact or min / max height of the random raw data

            nesting (Tuple[int, int]):
                The exact or min / max random depth of warp nestings

            channels (int | Tuple[int, int]):
                The exact or min / max number of random channels.

        Returns:
            DelayedWarp

        Example:
            >>> self = DelayedWarp.random(nesting=(4, 7))
            >>> print('self = {!r}'.format(self))
            >>> print(ub.repr2(self.nesting(), nl=-1, sort=0))

        Ignore:
            import kwplot
            kwplot.autompl()
            self = DelayedWarp.random(dsize=(32, 32), nesting=(1, 5), channels=3)
            data = self.finalize()
            kwplot.imshow(data)
        """
        from kwarray.distributions import DiscreteUniform, Uniform, Constant
        rng = kwarray.ensure_rng(rng)

        def distribution_coercion_rules(scalar, pair):
            # The rules for how to coerce a kwarray distribution are ambiguous
            # but perhaps we can define a nice way to define what they are on a
            # per-case basis, where domain knowledge can break the ambiguity?
            def coerce(arg, rng=None):
                # Choose the class to coerce into based on the rules
                cls = None
                if not ub.iterable(arg):
                    cls = scalar
                    arg = [arg]
                else:
                    if len(arg) == 2:
                        cls = pair
                if cls is None:
                    raise TypeError(type(arg))
                return cls.coerce(arg, rng=rng)
            return coerce

        coercer = distribution_coercion_rules(scalar=Constant, pair=DiscreteUniform)

        chan_distri = coercer(channels, rng=rng)
        nest_distri = coercer(nesting, rng=rng)
        rw_distri = coercer(raw_width, rng=rng)
        rh_distri = coercer(raw_height, rng=rng)
        raw_distri = Uniform(rng=rng)
        leaf_c = chan_distri.sample()
        leaf_w = rw_distri.sample()
        leaf_h = rh_distri.sample()

        raw = raw_distri.sample(leaf_h, leaf_w, leaf_c).astype(np.float32)
        layer = raw
        depth = nest_distri.sample()
        for _ in range(depth):
            tf = kwimage.Affine.random(rng=rng).matrix
            layer = DelayedWarp(layer, tf, dsize='auto')

        # Final warp to the desired output size
        if dsize is not None:
            ow, oh = dsize
            w, h = layer.dsize
            tf = kwimage.Affine.scale((h / oh, w / ow))
            layer = DelayedWarp(layer, tf, dsize=dsize)

        self = layer
        return self

    @property
    def channels(self):
        if hasattr(self.sub_data, 'channels'):
            return self.sub_data.channels
        else:
            return None

    def children(self):
        """
        Yields:
            Any:
        """
        yield self.sub_data

    @property
    def dsize(self):
        return self.meta['dsize']

    @property
    def num_bands(self):
        return self.meta['num_bands']

    @property
    def shape(self):
        # trailing_shape = self.sub_data.shape[2:]
        # trailing shape should only be allowed to have 0 or 1 dimension
        if self.meta['dsize'] is None:
            w = h = None
        else:
            w, h = self.meta['dsize']
        return (h, w, self.meta['num_bands'])

    def _optimize_paths(self, **kwargs):
        """
        Example:
            >>> self = DelayedWarp.random()
            >>> leafs = list(self._optimize_paths())
            >>> print('leafs = {!r}'.format(leafs))
        """
        # DEBUG_PRINT('DelayedWarp._optimize_paths')
        dsize = kwargs.get('dsize', None)
        transform = kwargs.get('transform', None)
        if dsize is None:
            dsize = self.meta['dsize']
        if transform is None:
            transform = self.meta['transform']
        else:
            transform = kwargs.get('transform', None) @ self.meta['transform']
        kwargs['dsize'] = dsize
        kwargs['transform'] = transform
        sub_data = self.sub_data
        flag = getattr(sub_data, '__hack_dont_optimize__', False)
        if hasattr(sub_data, '_optimize_paths') and not flag:
            yield from sub_data._optimize_paths(
                transform=transform, dsize=dsize)
        else:
            leaf = DelayedWarp(sub_data, transform, dsize=dsize)
            yield leaf

    @profile
    def finalize(self, transform=None, dsize=None, interpolation='linear',
                 **kwargs):
        """
        Execute the final transform

        Can pass a parent transform to augment this underlying transform.

        Args:
            transform (kwimage.Transform): an additional transform to perform
            dsize (Tuple[int, int]): overrides destination canvas size

        Example:
            >>> tf = np.array([[0.9, 0, 3.9], [0, 1.1, -.5], [0, 0, 1]])
            >>> raw = kwimage.grab_test_image(dsize=(54, 65))
            >>> raw = kwimage.ensure_float01(raw)
            >>> # Test nested finalize
            >>> layer1 = raw
            >>> num = 10
            >>> for _ in range(num):
            ...     layer1  = DelayedWarp(layer1, tf, dsize='auto')
            >>> final1 = layer1.finalize()
            >>> # Test non-nested finalize
            >>> layer2 = list(layer1._optimize_paths())[0]
            >>> final2 = layer2.finalize()
            >>> #
            >>> print(ub.repr2(layer1.nesting(), nl=-1, sort=0))
            >>> print(ub.repr2(layer2.nesting(), nl=-1, sort=0))
            >>> print('final1 = {!r}'.format(final1))
            >>> print('final2 = {!r}'.format(final2))
            >>> print('final1.shape = {!r}'.format(final1.shape))
            >>> print('final2.shape = {!r}'.format(final2.shape))
            >>> assert np.allclose(final1, final2)
            >>> #
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(raw, pnum=(1, 3, 1), fnum=1)
            >>> kwplot.imshow(final1, pnum=(1, 3, 2), fnum=1)
            >>> kwplot.imshow(final2, pnum=(1, 3, 3), fnum=1)
            >>> kwplot.show_if_requested()

        Example:
            >>> # Test aliasing
            >>> from kwcoco.util.delayed_poc.delayed_leafs import DelayedIdentity
            >>> s = DelayedIdentity.demo()
            >>> s = DelayedIdentity.demo('checkerboard')
            >>> a = s.warp(kwimage.Affine.scale(0.05), dsize='auto')
            >>> b = s.warp(kwimage.Affine.scale(3), dsize='auto')

            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> # It looks like downsampling linear and area is the same
            >>> # Does warpAffine have no alias handling?
            >>> pnum_ = kwplot.PlotNums(nRows=2, nCols=4)
            >>> kwplot.imshow(a.finalize(interpolation='area'), pnum=pnum_(), title='warpAffine area')
            >>> kwplot.imshow(a.finalize(interpolation='linear'), pnum=pnum_(), title='warpAffine linear')
            >>> kwplot.imshow(a.finalize(interpolation='nearest'), pnum=pnum_(), title='warpAffine nearest')
            >>> kwplot.imshow(a.finalize(interpolation='nearest', antialias=False), pnum=pnum_(), title='warpAffine nearest AA=0')
            >>> kwplot.imshow(kwimage.imresize(s.finalize(), dsize=a.dsize, interpolation='area'), pnum=pnum_(), title='resize area')
            >>> kwplot.imshow(kwimage.imresize(s.finalize(), dsize=a.dsize, interpolation='linear'), pnum=pnum_(), title='resize linear')
            >>> kwplot.imshow(kwimage.imresize(s.finalize(), dsize=a.dsize, interpolation='nearest'), pnum=pnum_(), title='resize nearest')
            >>> kwplot.imshow(kwimage.imresize(s.finalize(), dsize=a.dsize, interpolation='cubic'), pnum=pnum_(), title='resize cubic')

        Example:
            >>> # xdoctest: +REQUIRES(--ipfs)
            >>> # Demo code to develop support for overviews
            >>> import kwimage
            >>> from kwcoco.util.delayed_poc.delayed_nodes import *  # NOQA
            >>> from kwcoco.util.delayed_poc.delayed_leafs import DelayedLoad
            >>> fpath = ub.grabdata('https://ipfs.io/ipfs/QmaFcb565HM9FV8f41jrfCZcu1CXsZZMXEosjmbgeBhFQr', fname='PXL_20210411_150641385.jpg')
            >>> data0 = kwimage.imread(fpath, overview=0, backend='gdal')
            >>> data1 = kwimage.imread(fpath, overview=1, backend='gdal')
            >>> #transform1 = kwimage.Affine.affine(scale=0.1)
            >>> transform1 = kwimage.Affine.affine(scale=0.1, theta=-np.pi / 16, about=(0, 0))
            >>> transform2 = kwimage.Affine.affine(scale=2.0)
            >>> delayed_load = DelayedLoad(fpath=fpath)
            >>> delayed_load._ensure_dsize()
            >>> self = delayed_load.warp(transform1, dsize='auto')
            >>> #
            >>> orig_imdata1 = delayed_load.finalize(auto_overview=0)
            >>> shrunk_imdata1 = self.finalize(auto_overview=0)
            >>> rescale_imdata1 = self.finalize(transform=transform2, auto_overview=0)
            >>> #
            >>> orig_imdata2 = delayed_load.finalize(auto_overview=1)
            >>> shrunk_imdata2 = self.finalize(auto_overview=1)
            >>> rescale_imdata2 = self.finalize(transform=transform2, auto_overview=1)
            >>> # With a crop
            >>> crop = self.crop((slice(10, 60), slice(10, 60)))
            >>> crop_imdata1 = crop.finalize(auto_overview=0)
            >>> rescale_crop_imdata1 = crop.finalize(transform=transform2, auto_overview=0)
            >>> #
            >>> crop_imdata2 = crop.finalize(auto_overview=1)
            >>> rescale_crop_imdata2 = crop.finalize(transform=transform2, auto_overview=1)

            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> pnum_ = kwplot.PlotNums(nRows=2, nCols=5)
            >>> kwplot.imshow(orig_imdata1, pnum=pnum_(), title=f'orig, auto_overview=0: {orig_imdata1.shape}')
            >>> kwplot.imshow(shrunk_imdata1, pnum=pnum_(), title=f'shrunk, auto_overview=0: {shrunk_imdata1.shape}')
            >>> kwplot.imshow(rescale_imdata1, pnum=pnum_(), title=f'rescaled, auto_overview=0: {rescale_imdata1.shape}')
            >>> kwplot.imshow(crop_imdata1, pnum=pnum_(), title=f'rescaled, auto_overview=0: {crop_imdata1.shape}')
            >>> kwplot.imshow(rescale_crop_imdata1, pnum=pnum_(), title=f'rescaled, auto_overview=0: {rescale_crop_imdata1.shape}')
            >>> kwplot.imshow(orig_imdata2, pnum=pnum_(), title=f'orig, auto_overview=1: {orig_imdata2.shape}')
            >>> kwplot.imshow(shrunk_imdata2, pnum=pnum_(), title=f'shrunk, auto_overview=1: {shrunk_imdata2.shape}')
            >>> kwplot.imshow(rescale_imdata2, pnum=pnum_(), title=f'rescaled, auto_overview=1: {rescale_imdata2.shape}')
            >>> kwplot.imshow(crop_imdata2, pnum=pnum_(), title=f'crop, auto_overview=1: {crop_imdata2.shape}')
            >>> kwplot.imshow(rescale_crop_imdata2, pnum=pnum_(), title=f'crop-rescaled, auto_overview=1: {rescale_crop_imdata2.shape}')

        Ignore:
            # benchmark
            import timerit
            ti = timerit.Timerit(3, verbose=1)
            ti.reset('auto-overview=0').call(self.finalize, auto_overview=0)
            ti.reset('auto-overview=0 + tf').call(self.finalize, transform=transform2, auto_overview=0)
            ti.reset('auto-overview=1').call(delayed_load.finalize, auto_overview=1)
            ti.reset('auto-overview=1 + tf').call(delayed_load.finalize, transform=transform2, auto_overview=1)
        """
        # todo: needs to be extended for the case where the sub_data is a
        # nested chain of transforms.
        # import cv2
        # from kwimage import im_cv2
        if dsize is None:
            dsize = self.meta['dsize']
        transform = kwimage.Affine.coerce(transform) @ self.meta['transform']
        sub_data = self.sub_data

        # Errr, do we have to handle this here?
        # This probably means that we are about to perform an IO or data
        # generation operation. These are often more expensive than the
        # transforms themselves, so if can do things like generate less data,
        # or use IO overviews, we should do that and modify the transform.
        probably_io = getattr(sub_data, '__hack_dont_optimize__', False)
        if probably_io:
            subkw = ub.dict_diff(kwargs, {'as_xarray'})

            # Auto overview support
            AUTO_OVERVIEWS = kwargs.get('auto_overview', self._DEFAULT_AUTO_OVERVIEW)
            if AUTO_OVERVIEWS:
                try:
                    num_overviews = sub_data._ensure_num_overviews()
                except AttributeError:
                    num_overviews = 0
            else:
                num_overviews = 0

            if num_overviews:
                params = transform.decompose()
                sx, sy = params['scale']
                if sx < 1 and sy < 1:
                    # Note: we don't know how many overviews the underlying
                    # image will have, if any, we we can't assume this will get
                    # us anywhere.
                    num_downs, residual_sx, residual_sy = kwimage.im_cv2._prepare_scale_residual(sx, sy, fudge=0)
                    # Only use as many downs as we have overviews
                    can_do = min(num_overviews, num_downs)
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
                    transform = residual_transform
                    # subkw['overview'] = subkw.get('overview', 0) + can_do
                    subkw['overview'] = can_do

            sub_data = sub_data.finalize(**subkw)

        if hasattr(sub_data, 'finalize'):
            # This is not the final transform. Pass our transform down the
            # pipeline to delay it as long as possible.

            # Branch finalize
            final = sub_data.finalize(transform=transform, dsize=dsize,
                                      interpolation=interpolation, **kwargs)
            if len(final.shape) < 3:
                # HACK: we are assuming xarray never hits this case
                # Ensure that the last dimension is channels
                final = kwarray.atleast_nd(final, 3, front=False)
        else:
            as_xarray = kwargs.get('as_xarray', False)
            # Leaf finalize
            # flags = im_cv2._coerce_interpolation(interpolation)
            if dsize == (None, None):
                dsize = None
            sub_data_ = np.asarray(sub_data)
            M = np.asarray(transform)
            antialias = kwargs.get('antialias', True)

            final = kwimage.warp_affine(sub_data_, M, dsize=dsize,
                                        interpolation=interpolation,
                                        antialias=antialias)
            # final = cv2.warpPerspective(sub_data_, M, dsize=dsize, flags=flags)
            # Ensure that the last dimension is channels
            final = kwarray.atleast_nd(final, 3, front=False)
            if as_xarray:
                import xarray as xr
                channels = self.channels
                coords = {}
                if channels is not None:
                    coords['c'] = channels.code_list()
                final = xr.DataArray(final, dims=('y', 'x', 'c'), coords=coords)

        return final

    def take_channels(self, channels):
        new_subdata = self.sub_data.take_channels(channels)
        new = self.__class__(new_subdata, transform=self.meta['transform'],
                             dsize=self.meta['dsize'])
        return new


class DelayedCrop(DelayedImage):
    """
    Represent a delayed crop operation

    Example:
        >>> from kwcoco.util.delayed_poc.delayed_leafs import DelayedLoad
        >>> sub_data = DelayedLoad.demo()
        >>> sub_slices = (slice(5, 10), slice(1, 12))
        >>> self = DelayedCrop(sub_data, sub_slices)
        >>> print(ub.repr2(self.nesting(), nl=-1, sort=0))
        >>> final = self.finalize()
        >>> print('final.shape = {!r}'.format(final.shape))

    Example:
        >>> from kwcoco.util.delayed_poc.delayed_leafs import DelayedLoad
        >>> sub_data = DelayedLoad.demo()
        >>> sub_slices = (slice(5, 10), slice(1, 12))
        >>> crop1 = DelayedCrop(sub_data, sub_slices)
        >>> import pytest
        >>> # Should only error while huristics are in use.
        >>> with pytest.raises(ValueError):
        >>>     crop2 = DelayedCrop(crop1, sub_slices)
    """

    __hack_dont_optimize__ = True

    def __init__(self, sub_data, sub_slices):
        if isinstance(sub_data, (DelayedCrop, DelayedWarp, DelayedChannelStack)):
            raise ValueError('cant crop generally yet')

        self.sub_data = sub_data
        self.sub_slices = sub_slices

        sl_x, sl_y = sub_slices[0:2]
        width = sl_x.stop - sl_x.start
        height = sl_y.stop - sl_y.start
        if hasattr(sub_data, 'num_bands'):
            num_bands = sub_data.num_bands
        else:
            num_bands = kwimage.num_channels(self.sub_data)
        self.num_bands = num_bands
        self.shape = (height, width, num_bands)
        self.meta = {
            'shape': self.shape,
            'sub_slices': self.sub_slices,
            'num_bands': self.num_bands,
        }

    @property
    def channels(self):
        if hasattr(self.sub_data, 'channels'):
            return self.sub_data.channels
        else:
            return None

    def children(self):
        """
        Yields:
            Any:
        """
        yield self.sub_data

    @profile
    def finalize(self, **kwargs):
        if hasattr(self.sub_data, 'finalize'):
            assert not kwargs.get('overview', None), 'overview is not handled in crop logic yet'
            final = self.sub_data.finalize(**kwargs)
            return final[self.sub_slices]
        else:
            return self.sub_data[self.sub_slices]

    def _optimize_paths(self, **kwargs):
        raise NotImplementedError('cant look at leafs through crop atm')
