"""
Intermediate operations
"""
import kwarray
import kwimage
import copy
import numpy as np
import ubelt as ub
import warnings
from kwcoco import exceptions
from kwcoco import channel_spec
from kwcoco.util.delayed_ops.delayed_base import DelayedNaryOperation2, DelayedUnaryOperation2
from kwcoco.util.delayed_ops import delayed_leafs


try:
    from xdev import profile
except Exception:
    from ubelt import identity as profile

# --------
# Stacking
# --------


class DelayedStack2(DelayedNaryOperation2):
    """
    Stacks multiple arrays together.
    """

    def __init__(self, parts, axis):
        """
        Args:
            parts (List[DelayedArray2]): data to stack
            axis (int): axes to stack on
        """
        super().__init__(parts=parts)
        self.meta['axis'] = axis

    def __nice__(self):
        """
        Returns:
            str
        """
        return '{}'.format(self.shape)

    @property
    def shape(self):
        """
        Returns:
            None | Tuple[int | None, ...]
        """
        shape = self.subdata.shape
        return shape


class DelayedConcat2(DelayedNaryOperation2):
    """
    Stacks multiple arrays together.
    """

    def __init__(self, parts, axis):
        """
        Args:
            parts (List[DelayedArray2]): data to concat
            axis (int): axes to concat on
        """
        super().__init__(parts=parts)
        self.meta['axis'] = axis

    def __nice__(self):
        return '{}'.format(self.shape)

    @property
    def shape(self):
        """
        Returns:
            None | Tuple[int | None, ...]
        """
        shape = self.subdata.shape
        return shape


class DelayedFrameStack2(DelayedStack2):
    """
    Stacks multiple arrays together.
    """

    def __init__(self, parts):
        """
        Args:
            parts (List[DelayedArray2]): data to stack
        """
        super().__init__(parts=parts, axis=0)

# ------
# Images
# ------


class ImageOpsMixin:

    def crop(self, space_slice=None, chan_idxs=None):
        """
        Crops an image along integer pixel coordinates.

        Args:
            space_slice (Tuple[slice, slice]): y-slice and x-slice.
            chan_idxs (List[int]): indexes of bands to take

        Returns:
            DelayedImage2
        """
        new = DelayedCrop2(self, space_slice, chan_idxs)
        return new

    def warp(self, transform, dsize='auto', antialias=True,
             interpolation='linear', border_value='auto'):
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

            border_value (int | float | str):
                if auto will be nan for float and 0 for int.

        Returns:
            DelayedImage2
        """
        new = DelayedWarp2(self, transform, dsize=dsize, antialias=antialias,
                           interpolation=interpolation)
        return new

    def dequantize(self, quantization):
        """
        Rescales image intensities from int to floats.

        Args:
            quantization (Dict[str, Any]):
                see :func:`kwcoco.util.delayed_ops.helpers.dequantize`

        Returns:
            DelayedDequantize2
        """
        new = DelayedDequantize2(self, quantization)
        return new

    def get_overview(self, overview):
        """
        Downsamples an image by a factor of two.

        Args:
            overview (int): the overview to use (assuming it exists)

        Returns:
            DelayedOverview2
        """
        new = DelayedOverview2(self, overview)
        return new

    def as_xarray(self):
        """
        Returns:
            DelayedAsXarray2
        """
        return DelayedAsXarray2(self)


class DelayedChannelConcat2(ImageOpsMixin, DelayedConcat2):
    """
    Stacks multiple arrays together.

    CommandLine:
        xdoctest -m /home/joncrall/code/kwcoco/kwcoco/util/delayed_ops/delayed_nodes.py DelayedChannelConcat2:1

    Example:
        >>> from kwcoco.util.delayed_ops import *  # NOQA
        >>> from kwcoco.util.delayed_ops.delayed_leafs import DelayedLoad2
        >>> import kwcoco
        >>> dsize = (307, 311)
        >>> c1 = DelayedNans2(dsize=dsize, channels='foo')
        >>> c2 = DelayedLoad2.demo('astro', dsize=dsize, channels='R|G|B').prepare()
        >>> cat = DelayedChannelConcat2([c1, c2])
        >>> warped_cat = cat.warp({'scale': 1.07}, dsize=(328, 332))
        >>> warped_cat._validate()
        >>> warped_cat.finalize()

    Example:
        >>> # Test case that failed in initial implementation
        >>> # Due to incorrectly pushing channel selection under the concat
        >>> from kwcoco.util.delayed_ops import *  # NOQA
        >>> import kwimage
        >>> fpath = kwimage.grab_test_image_fpath()
        >>> base1 = DelayedLoad2(fpath, channels='r|g|b').prepare()
        >>> base2 = DelayedLoad2(fpath, channels='x|y|z').prepare().scale(2)
        >>> base3 = DelayedLoad2(fpath, channels='i|j|k').prepare().scale(2)
        >>> bands = [base2, base1[:, :, 0].scale(2).evaluate(),
        >>>          base1[:, :, 1].evaluate().scale(2),
        >>>          base1[:, :, 2].evaluate().scale(2), base3]
        >>> delayed = DelayedChannelConcat2(bands)
        >>> delayed = delayed.warp({'scale': 2})
        >>> delayed = delayed[0:100, 0:55, [0, 2, 4]]
        >>> delayed.write_network_text()
        >>> delayed.optimize()
    """

    def __init__(self, parts, dsize=None):
        """
        Args:
            parts (List[DelayedArray2]): data to concat
            dsize (Tuple[int, int] | None): size if known a-priori
        """
        super().__init__(parts=parts, axis=2)
        if dsize is None:
            dsize_cands = [comp.dsize for comp in self.parts]
            if not ub.allsame(dsize_cands):
                raise exceptions.CoordinateCompatibilityError(
                    # 'parts must all have the same delayed size')
                    'parts must all have the same delayed size: got {}'.format(dsize_cands))
            if len(dsize_cands) == 0:
                dsize = None
            else:
                dsize = dsize_cands[0]
        self.dsize = dsize
        try:
            self.num_channels = sum(comp.num_channels for comp in self.parts)
        except TypeError:
            if any(comp.num_channels is None for comp in self.parts):
                self.num_channels = None
            else:
                raise

    def __nice__(self):
        """
        Returns:
            str
        """
        if self.channels is None:
            return '{}'.format(self.shape)
        else:
            return '{}, {}'.format(self.shape, self.channels)

    @property
    def channels(self):
        """
        Returns:
            None | kwcoco.FusedChannelSpec
        """
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
        """
        Returns:
            Tuple[int | None, int | None, int | None]
        """
        w, h = self.dsize
        return (h, w, self.num_channels)

    def _finalize(self):
        """
        Returns:
            ArrayLike
        """
        stack = [comp._finalize() for comp in self.parts]
        if len(stack) == 1:
            final = stack[0]
        else:
            stack = [kwarray.atleast_nd(s, 3) for s in stack]
            final = np.concatenate(stack, axis=2)
        return final

    def optimize(self):
        """
        Returns:
            DelayedImage2
        """
        new_parts = [part.optimize() for part in self.parts]
        kw = ub.dict_isect(self.meta, ['dsize'])
        new = self.__class__(new_parts, **kw)
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
                :class:`kwcoco.ChannelSpec` for more detials.

        Returns:
            DelayedArray2:
                a delayed vision operation that only operates on the following
                channels.

        Example:
            >>> from kwcoco.util.delayed_ops.delayed_nodes import *  # NOQA
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> self = delayed = dset.coco_image(1).delay(mode=1)
            >>> channels = 'B11|B8|B1|B10'
            >>> new = self.take_channels(channels)

        Example:
            >>> # Complex case
            >>> import kwcoco
            >>> from kwcoco.util.delayed_ops.delayed_nodes import *  # NOQA
            >>> from kwcoco.util.delayed_ops.delayed_leafs import DelayedLoad2
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> delayed = dset.coco_image(1).delay(mode=1)
            >>> astro = DelayedLoad2.demo('astro', channels='r|g|b').prepare()
            >>> aligned = astro.warp(kwimage.Affine.scale(600 / 512), dsize='auto')
            >>> self = combo = DelayedChannelConcat2(delayed.parts + [aligned])
            >>> channels = 'B1|r|B8|g'
            >>> new = self.take_channels(channels)
            >>> new_cropped = new.crop((slice(10, 200), slice(12, 350)))
            >>> new_opt = new_cropped.optimize()
            >>> datas = new_opt.finalize()
            >>> if 1:
            >>>     new_cropped.write_network_text(with_labels='name')
            >>>     new_opt.write_network_text(with_labels='name')
            >>> vizable = kwimage.normalize_intensity(datas, axis=2)
            >>> self._validate()
            >>> new._validate()
            >>> new_cropped._validate()
            >>> new_opt._validate()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> stacked = kwimage.stack_images(vizable.transpose(2, 0, 1))
            >>> kwplot.imshow(stacked)

        Example:
            >>> # Test case where requested channel does not exist
            >>> import kwcoco
            >>> from kwcoco.util.delayed_ops.delayed_nodes import *  # NOQA
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral', use_cache=1, verbose=100)
            >>> self = delayed = dset.coco_image(1).delay(mode=1)
            >>> channels = 'B1|foobar|bazbiz|B8'
            >>> new = self.take_channels(channels)
            >>> new_cropped = new.crop((slice(10, 200), slice(12, 350)))
            >>> fused = new_cropped.finalize()
            >>> assert fused.shape == (190, 338, 4)
            >>> assert np.all(np.isnan(fused[..., 1:3]))
            >>> assert not np.any(np.isnan(fused[..., 0]))
            >>> assert not np.any(np.isnan(fused[..., 3]))
        """
        if channels is None:
            return self
        from kwcoco.util.delayed_ops.delayed_leafs import DelayedNans2
        current_channels = self.channels

        if isinstance(channels, list):
            top_idx_mapping = channels
            top_codes = self.channels.as_list()
            request_codes = None
        else:
            channels = channel_spec.FusedChannelSpec.coerce(channels)
            # Computer subindex integer mapping
            request_codes = channels.as_list()
            top_codes = current_channels.as_oset()
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
            comp.num_channels for comp in self.parts])

        accum = []
        class _ContiguousSegment(object):
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
                curr = _ContiguousSegment(comp, inner)
            else:
                is_contiguous = curr.comp is comp and (inner == curr.stop)
                if is_contiguous:
                    # extend the previous contiguous segment
                    curr.stop = inner + 1
                else:
                    # accept previous segment and start a new one
                    accum.append(curr)
                    curr = _ContiguousSegment(comp, inner)

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
                comp = DelayedNans2(self.dsize, channels=nan_chan)
                new_components.append(comp)
            else:
                if curr.start == 0 and curr.stop == comp.num_channels:
                    # Entire component is valid, no need for sub-operation
                    new_components.append(comp)
                else:
                    # Only part of the component is taken, need to sub-operate
                    # It would be nice if we only loaded the file once if we need
                    # multiple parts discontiguously.
                    sub_idxs = list(range(curr.start, curr.stop))
                    sub_comp = comp.take_channels(sub_idxs)
                    new_components.append(sub_comp)

        new = DelayedChannelConcat2(new_components)
        return new

    def __getitem__(self, sl):
        if not isinstance(sl, tuple):
            raise TypeError('slice must be given as tuple')
        if len(sl) == 2:
            sl_y, sl_x = sl
            chan_idxs = None
        elif len(sl) == 3:
            sl_y, sl_x, chan_idxs = sl
        else:
            raise ValueError('Slice must have 2 or 3 dims')
        space_slice = (sl_y, sl_x)
        return self.crop(space_slice, chan_idxs)

    @property
    def num_overviews(self):
        """
        Returns:
            int
        """
        num_overviews = self.meta.get('num_overviews', None)
        if num_overviews is None and self.parts is not None:
            cand = [p.num_overviews for p in self.parts]
            if ub.allsame(cand):
                num_overviews = cand[0]
            else:
                import warnings
                warnings.warn('inconsistent overviews')
                num_overviews = None
        return num_overviews

    def as_xarray(self):
        """
        Returns:
            DelayedAsXarray2
        """
        return DelayedAsXarray2(self)

    def _push_operation_under(self, op, kwargs):
        # Note: we can't do this with a crop that has band selection
        # But spatial operations should be ok.
        return self.__class__([op(p, **kwargs) for p in self.parts])

    def _validate(self):
        """
        Check that the delayed metadata corresponds with the finalized data
        """
        final = self._finalize()
        # meta_dsize = self.dsize
        meta_shape = self.shape

        final_shape = final.shape

        correspondences = {
            'shape': (final_shape, meta_shape)
        }
        for k, tup in correspondences.items():
            v1, v2 = tup
            if v1 != v2:
                raise AssertionError(
                    f'final and meta {k} does not agree {v1!r} != {v2!r}')
        return self

    def undo_warps(self, remove=None, retain=None, squash_nans=False, return_warps=False):
        """
        Attempts to "undo" warping for each concatenated channel and returns a
        list of delayed operations that are cropped to the right regions.

        Typically you will retrain offset, theta, and shear to remove scale.
        This ensures the data is spatially aligned up to a scale factor.

        Args:
            remove (List[str]): if specified, list components of the warping to
                remove. Can include: "offset", "scale", "shearx", "theta".
                Typically set this to ["scale"].

            retain (List[str]): if specified, list components of the warping to
                retain. Can include: "offset", "scale", "shearx", "theta".
                Mutually exclusive with "remove". If neither remove or retain
                is specified, retain is set to ``[]``.

            squash_nans (bool):
                if True, pure nan channels are squashed into a 1x1 array as
                they do not correspond to a real source.

            return_warps (bool):
                if True, return the transforms we applied.
                This is useful when you need to warp objects in the original
                space into the jagged space.

        Example:
            >>> from kwcoco.util.delayed_ops.delayed_nodes import *  # NOQA
            >>> from kwcoco.util.delayed_ops.delayed_leafs import DelayedLoad2
            >>> from kwcoco.util.delayed_ops.delayed_leafs import DelayedNans2
            >>> import ubelt as ub
            >>> import kwimage
            >>> import kwarray
            >>> import numpy as np
            >>> # Demo case where we have different channels at different resolutions
            >>> base = DelayedLoad2.demo(channels='r|g|b').prepare().dequantize({'quant_max': 255})
            >>> bandR = base[:, :, 0].scale(100 / 512)[:, :-50].evaluate()
            >>> bandG = base[:, :, 1].scale(300 / 512).warp({'theta': np.pi / 8, 'about': (150, 150)}).evaluate()
            >>> bandB = base[:, :, 2].scale(600 / 512)[:150, :].evaluate()
            >>> bandN = DelayedNans2((600, 600), channels='N')
            >>> # Make a concatenation of images of different underlying native resolutions
            >>> delayed_vidspace = DelayedChannelConcat2([
            >>>     bandR.scale(6, dsize=(600, 600)).optimize(),
            >>>     bandG.warp({'theta': -np.pi / 8, 'about': (150, 150)}).scale(2, dsize=(600, 600)).optimize(),
            >>>     bandB.scale(1, dsize=(600, 600)).optimize(),
            >>>     bandN,
            >>> ]).warp({'scale': 0.7}).optimize()
            >>> vidspace_box = kwimage.Boxes([[100, 10, 270, 160]], 'ltrb')
            >>> vidspace_poly = vidspace_box.to_polygons()[0]
            >>> vidspace_slice = vidspace_box.to_slices()[0]
            >>> self = delayed_vidspace[vidspace_slice].optimize()
            >>> print('--- Aligned --- ')
            >>> self.write_network_text()
            >>> squash_nans = True
            >>> undone_all_parts, tfs1 = self.undo_warps(squash_nans=squash_nans, return_warps=True)
            >>> undone_scale_parts, tfs2 = self.undo_warps(remove=['scale'], squash_nans=squash_nans, return_warps=True)
            >>> stackable_aligned = self.finalize().transpose(2, 0, 1)
            >>> stackable_undone_all = []
            >>> stackable_undone_scale = []
            >>> print('--- Undone All --- ')
            >>> for undone in undone_all_parts:
            ...     undone.write_network_text()
            ...     stackable_undone_all.append(undone.finalize())
            >>> print('--- Undone Scale --- ')
            >>> for undone in undone_scale_parts:
            ...     undone.write_network_text()
            ...     stackable_undone_scale.append(undone.finalize())
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> canvas0 = kwimage.stack_images(stackable_aligned, axis=1)
            >>> canvas1 = kwimage.stack_images(stackable_undone_all, axis=1)
            >>> canvas2 = kwimage.stack_images(stackable_undone_scale, axis=1)
            >>> canvas0 = kwimage.draw_header_text(canvas0, 'Rescaled Aligned Channels')
            >>> canvas1 = kwimage.draw_header_text(canvas1, 'Unwarped Channels')
            >>> canvas2 = kwimage.draw_header_text(canvas2, 'Unscaled Channels')
            >>> canvas = kwimage.stack_images([canvas0, canvas1, canvas2], axis=0)
            >>> canvas = kwimage.fill_nans_with_checkers(canvas)
            >>> kwplot.imshow(canvas)
        """
        valid_keys = {"offset", "scale", "shearx", "theta"}
        if remove is not None and retain is not None:
            raise ValueError('Mutex')
        if remove is not None:
            retain = valid_keys - set(remove)
        else:
            if retain is None:
                retain = set()
        unwarped_parts = []
        jagged_warps = []
        for part in self.parts:
            tf_root_from_leaf = part.get_transform_from_leaf()
            tf_leaf_from_root = tf_root_from_leaf.inv()
            undo_all = tf_leaf_from_root
            all_undo_components = undo_all.concise()
            undo_components = ub.dict_diff(all_undo_components, retain)
            undo_warp = kwimage.Affine.coerce(undo_components)
            undone_part = part.warp(undo_warp).optimize()
            if squash_nans:
                if return_warps:
                    # hack the return undo_warp
                    w, h = undone_part.dsize
                    undo_warp = kwimage.Affine.scale((1 / w, 1 / h)) @ undo_warp
                if isinstance(undone_part, delayed_leafs.DelayedNans2):
                    undone_part = undone_part[0:1, 0:1].optimize()
            unwarped_parts.append(undone_part)
            if return_warps:
                jagged_warps.append(undo_warp)
        if return_warps:
            return unwarped_parts, jagged_warps
        else:
            return unwarped_parts


class DelayedArray2(DelayedUnaryOperation2):
    """
    A generic NDArray.
    """
    def __init__(self, subdata=None):
        """
        Args:
            subdata (DelayedArray2):
        """
        super().__init__(subdata=subdata)

    def __nice__(self):
        """
        Returns:
            str
        """
        return '{}'.format(self.shape)

    @property
    def shape(self):
        """
        Returns:
            None | Tuple[int | None, ...]
        """
        shape = self.subdata.shape
        return shape


class DelayedImage2(ImageOpsMixin, DelayedArray2):
    """
    For the case where an array represents a 2D image with multiple channels
    """
    def __init__(self, subdata=None, dsize=None, channels=None):
        """
        Args:
            subdata (DelayedArray2):
            dsize (None | Tuple[int | None, int | None]): overrides subdata dsize
            channels (None | int | kwcoco.FusedChannelSpec): overrides subdata channels
        """
        super().__init__(subdata)
        self.channels = channels
        self.meta['dsize'] = dsize

    def __nice__(self):
        """
        Returns:
            str
        """
        if self.channels is None:
            return '{}'.format(self.shape)
        else:
            return '{}, {}'.format(self.shape, self.channels)

    @property
    def shape(self):
        """
        Returns:
            None | Tuple[int | None, int | None, int | None]
        """
        dsize = self.dsize
        if dsize is None:
            dsize = (None, None)
        w, h = dsize
        c = self.num_channels
        return (h, w, c)

    @property
    def num_channels(self):
        """
        Returns:
            None | int
        """
        num_channels = self.meta.get('num_channels', None)
        if num_channels is None and self.subdata is not None:
            num_channels = self.subdata.num_channels
        return num_channels

    @property
    def dsize(self):
        """
        Returns:
            None | Tuple[int | None, int | None]
        """
        # return self.meta.get('dsize', None)
        dsize = self.meta.get('dsize', None)
        if dsize is None and self.subdata is not None:
            dsize = self.subdata.dsize
        return dsize

    @property
    def channels(self):
        """
        Returns:
            None | kwcoco.FusedChannelSpec
        """
        channels = self.meta.get('channels', None)
        if channels is None and self.subdata is not None:
            channels = self.subdata.channels
        return channels

    @channels.setter
    def channels(self, channels):
        if channels is None:
            num_channels = None
        else:
            if isinstance(channels, int):
                num_channels = channels
                channels = None
            else:
                import kwcoco
                channels = kwcoco.FusedChannelSpec.coerce(channels)
                num_channels = channels.normalize().numel()
        self.meta['channels'] = channels
        self.meta['num_channels'] = num_channels

    @property
    def num_overviews(self):
        """
        Returns:
            int
        """
        num_overviews = self.meta.get('num_overviews', None)
        if num_overviews is None and self.subdata is not None:
            num_overviews = self.subdata.num_overviews
        return num_overviews

    def __getitem__(self, sl):
        if not isinstance(sl, tuple):
            raise TypeError('slice must be given as tuple')
        if len(sl) == 2:
            sl_y, sl_x = sl
            chan_idxs = None
        elif len(sl) == 3:
            sl_y, sl_x, chan_idxs = sl
        else:
            raise ValueError('Slice must have 2 or 3 dims')
        space_slice = (sl_y, sl_x)
        return self.crop(space_slice, chan_idxs)

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
            DelayedCrop2:
                a new delayed load with a fused take channel operation

        Note:
            The channel subset must exist here or it will raise an error.
            A better implementation (via pymbolic) might be able to do better

        Example:
            >>> #
            >>> # Test Channel Select Via Code
            >>> from kwcoco.util.delayed_ops.delayed_nodes import *  # NOQA
            >>> from kwcoco.util.delayed_ops import DelayedLoad2
            >>> self = DelayedLoad2.demo(dsize=(16, 16), channels='r|g|b').prepare()
            >>> channels = 'r|b'
            >>> new = self.take_channels(channels)._validate()
            >>> new2 = new[:, :, [1, 0]]._validate()
            >>> new3 = new2[:, :, [1]]._validate()

        Example:
            >>> from kwcoco.util.delayed_ops.delayed_nodes import *  # NOQA
            >>> from kwcoco.util.delayed_ops import DelayedLoad2
            >>> import kwcoco
            >>> self = DelayedLoad2.demo('astro').prepare()
            >>> channels = [2, 0]
            >>> new = self.take_channels(channels)
            >>> new3 = new.take_channels([1, 0])
            >>> new._validate()
            >>> new3._validate()

            >>> final1 = self.finalize()
            >>> final2 = new.finalize()
            >>> final3 = new3.finalize()
            >>> assert np.all(final1[..., 2] == final2[..., 0])
            >>> assert np.all(final1[..., 0] == final2[..., 1])
            >>> assert final2.shape[2] == 2

            >>> assert np.all(final1[..., 2] == final3[..., 1])
            >>> assert np.all(final1[..., 0] == final3[..., 0])
            >>> assert final3.shape[2] == 2
        """
        if isinstance(channels, list):
            top_idx_mapping = channels
        else:
            current_channels = self.channels
            if current_channels is None:
                raise ValueError(
                    'The channel spec for this node are unknown. '
                    'Cannot use a spec to select them'
                )
            channels = channel_spec.FusedChannelSpec.coerce(channels)
            # Computer subindex integer mapping
            request_codes = channels.as_list()
            top_codes = current_channels.as_oset()
            top_idx_mapping = [
                top_codes.index(code)
                for code in request_codes
            ]
        new_chan_ixs = top_idx_mapping
        new = self.crop(None, new_chan_ixs)
        return new

    def scale(self, scale, dsize='auto', antialias=True, interpolation='linear'):
        return self.warp({'scale': scale}, dsize=dsize)

    def _validate(self):
        """
        Check that the delayed metadata corresponds with the finalized data
        """
        opt = self.optimize()
        opt_shape = opt.shape

        final = self._finalize()
        # meta_dsize = self.dsize
        meta_shape = self.shape

        final_shape = final.shape

        correspondences = {
            'opt_chans': (self.channels, opt.channels),
            'opt_nbands': (self.num_channels, opt.num_channels),
            'final_shape': (final_shape, meta_shape),
            'opt_shape': (opt_shape, meta_shape),
        }
        for k, tup in correspondences.items():
            v1, v2 = tup
            if v1 != v2:
                raise AssertionError(
                    f'final and meta {k} does not agree {v1!r} != {v2!r}')
        return self

    def _transform_from_subdata(self):
        raise NotImplementedError

    def get_transform_from_leaf(self):
        """
        Returns the transformation that would align data with the leaf
        """
        subdata_from_leaf = self.subdata.get_transform_from_leaf()
        self_from_subdata = self._transform_from_subdata()
        self_from_leaf = self_from_subdata @ subdata_from_leaf
        return self_from_leaf

    def evaluate(self):
        """
        Evaluate this node and return the data as an identity.

        Returns:
            DelayedIdentity2
        """
        from kwcoco.util.delayed_ops.delayed_leafs import DelayedIdentity2
        final = self.finalize()
        new = DelayedIdentity2(final, dsize=self.dsize, channels=self.channels)
        return new

    def _opt_push_under_concat(self):
        assert isinstance(self.subdata, DelayedChannelConcat2)
        kwargs = ub.compatible(self.meta, self.__class__.__init__)
        new = self.subdata._push_operation_under(self.__class__, kwargs)
        return new


class DelayedAsXarray2(DelayedImage2):
    """
    Casts the data to an xarray object in the finalize step

    Example;
        >>> from kwcoco.util.delayed_ops.delayed_nodes import *  # NOQA
        >>> from kwcoco.util.delayed_ops import DelayedLoad2
        >>> # without channels
        >>> base = DelayedLoad2.demo(dsize=(16, 16)).prepare()
        >>> self = base.as_xarray()
        >>> final = self._validate().finalize()
        >>> assert len(final.coords) == 0
        >>> assert final.dims == ('y', 'x', 'c')
        >>> # with channels
        >>> base = DelayedLoad2.demo(dsize=(16, 16), channels='r|g|b').prepare()
        >>> self = base.as_xarray()
        >>> final = self._validate().finalize()
        >>> assert final.coords.indexes['c'].tolist() == ['r', 'g', 'b']
        >>> assert final.dims == ('y', 'x', 'c')
    """

    def _finalize(self):
        """
        Returns:
            ArrayLike
        """
        import xarray as xr
        subfinal = np.asarray(self.subdata._finalize())
        channels = self.subdata.channels
        coords = {}
        if channels is not None:
            coords['c'] = channels.code_list()
        final = xr.DataArray(subfinal, dims=('y', 'x', 'c'), coords=coords)
        return final

    def optimize(self):
        """
        Returns:
            DelayedImage2
        """
        return self.subdata.optimize().as_xarray()


class DelayedWarp2(DelayedImage2):
    """
    Applies an affine transform to an image.

    Example:
        >>> from kwcoco.util.delayed_ops.delayed_nodes import *  # NOQA
        >>> from kwcoco.util.delayed_ops import DelayedLoad2
        >>> self = DelayedLoad2.demo(dsize=(16, 16)).prepare()
        >>> warp1 = self.warp({'scale': 3})
        >>> warp2 = warp1.warp({'theta': 0.1})
        >>> warp3 = warp2._opt_fuse_warps()
        >>> warp3._validate()
        >>> print(ub.repr2(warp2.nesting(), nl=-1, sort=0))
        >>> print(ub.repr2(warp3.nesting(), nl=-1, sort=0))
    """
    def __init__(self, subdata, transform, dsize='auto', antialias=True,
                 interpolation='linear', border_value='auto'):
        """
        Args:
            subdata (DelayedArray2): data to operate on

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
            from kwcoco.util.delayed_ops.helpers import _auto_dsize
            dsize = _auto_dsize(transform, self.subdata.dsize)
        self.meta['transform'] = transform
        self.meta['antialias'] = antialias
        self.meta['interpolation'] = interpolation
        self.meta['dsize'] = dsize
        self.meta['border_value'] = border_value

    @property
    def transform(self):
        """
        Returns:
            kwimage.Affine
        """
        return self.meta['transform']

    def _finalize(self):
        """
        Returns:
            ArrayLike
        """
        dsize = self.dsize
        if dsize == (None, None):
            dsize = None
        antialias = self.meta['antialias']
        transform = self.meta['transform']
        interpolation = self.meta['interpolation']

        prewarp = self.subdata._finalize()
        prewarp = np.asarray(prewarp)

        # TODO: we could configure this, but forcing nans on floats seems like
        # a pretty nice default border behavior. It would be even nicer to have
        # masked arrays for ints.
        # The scalar / explicit functionality will be handled inside warp_affine
        # in the future, so some of this can be removed.
        num_chan = kwimage.num_channels(prewarp)
        if self.meta['border_value'] == 'auto':
            if prewarp.dtype.kind == 'f':
                border_value = np.nan
            else:
                border_value = 0
        else:
            border_value = self.meta['border_value']
        if not ub.iterable(border_value):
            # Odd OpenCV behavior: https://github.com/opencv/opencv/issues/22283
            # Can only have at most 4 components to border_value and
            # then they start to wrap around. This is fine if we are only
            # specifying a single number for all channels
            border_value = (border_value,) * min(4, num_chan)
        if len(border_value) > 4:
            raise ValueError('borderValue cannot have more than 4 components. '
                             'OpenCV #22283 describes why')

        # HACK:
        # the border value only correctly applies to the first 4 channels for
        # whatever reason.
        border_value = border_value[0:4]

        M = np.asarray(transform)
        final = kwimage.warp_affine(prewarp, M, dsize=dsize,
                                    interpolation=interpolation,
                                    antialias=antialias,
                                    border_value=border_value)
        # final = cv2.warpPerspective(sub_data_, M, dsize=dsize, flags=flags)
        # Ensure that the last dimension is channels
        final = kwarray.atleast_nd(final, 3, front=False)
        return final

    @profile
    def optimize(self):
        """
        Returns:
            DelayedImage2

        Example:
            >>> # Demo optimization that removes a noop warp
            >>> from kwcoco.util.delayed_ops import DelayedLoad2
            >>> import kwimage
            >>> base = DelayedLoad2.demo(channels='r|g|b').prepare()
            >>> self = base.warp(kwimage.Affine.eye())
            >>> new = self.optimize()
            >>> assert len(self.as_graph().nodes) == 2
            >>> assert len(new.as_graph().nodes) == 1

        Example:
            >>> # Test optimize nans
            >>> from kwcoco.util.delayed_ops import DelayedNans2
            >>> import kwimage
            >>> base = DelayedNans2(dsize=(100, 100), channels='a|b|c')
            >>> self = base.warp(kwimage.Affine.scale(0.1))
            >>> # Should simply return a new nan generator
            >>> new = self.optimize()
            >>> assert len(new.as_graph().nodes) == 1
        """
        new = copy.copy(self)
        new.subdata = self.subdata.optimize()
        if isinstance(new.subdata, DelayedWarp2):
            new = new._opt_fuse_warps()

        ### The tolerance should be very strict by default, but
        ### we also might want to be able to parameterize it
        if new.transform.isclose_identity(rtol=0, atol=0) and new.dsize == new.subdata.dsize:
            new = new.subdata
        elif isinstance(new.subdata, DelayedChannelConcat2):
            new = new._opt_push_under_concat().optimize()
        elif hasattr(new.subdata, '_optimized_warp'):
            # The subdata knows how to optimize itself wrt a warp
            warp_kwargs = ub.dict_isect(self.meta, {
                'transform', 'dsize', 'antialias', 'interpolation',
                'border_value'})
            new = new.subdata._optimized_warp(**warp_kwargs).optimize()
        else:
            split = new._opt_split_warp_overview()
            if new is not split:
                new = split
                new.subdata = new.subdata.optimize()
                new = new.optimize()
            else:
                new = new._opt_absorb_overview()
        return new

    def _transform_from_subdata(self):
        return self.transform

    def _opt_fuse_warps(self):
        """
        Combine two consecutive warps into a single operation.
        """
        assert isinstance(self.subdata, DelayedWarp2)
        inner_data = self.subdata.subdata
        tf1 = self.subdata.meta['transform']
        tf2 = self.meta['transform']
        # TODO: could ensure the metadata is compatable, for now just take the
        # most recent
        dsize = self.meta['dsize']
        common_meta = ub.dict_isect(self.meta, {'antialias', 'interpolation', 'border_value'})
        new_transform = tf2 @ tf1
        new = self.__class__(inner_data, new_transform, dsize=dsize,
                             **common_meta)
        return new

    def _opt_absorb_overview(self):
        """
        Remove the overview if we can get a higher resolution without it

        Example:
            >>> # xdoctest: +REQUIRES(module:osgeo)
            >>> from kwcoco.util.delayed_ops.delayed_nodes import *  # NOQA
            >>> from kwcoco.util.delayed_ops import DelayedLoad2
            >>> import kwimage
            >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
            >>> base = DelayedLoad2(fpath, channels='r|g|b').prepare()
            >>> # Case without any operations between the overview and warp
            >>> self = base.get_overview(1).warp({'scale': 4})
            >>> self.write_network_text()
            >>> opt = self._opt_absorb_overview()._validate()
            >>> opt.write_network_text()
            >>> opt_data = [d for n, d in opt.as_graph().nodes(data=True)]
            >>> assert 'DelayedOverview2' not in [d['type'] for d in opt_data]
            >>> # Case with a chain of operations between overview and warp
            >>> self = base.get_overview(1)[0:101, 0:100].warp({'scale': 4})
            >>> self.write_network_text()
            >>> opt = self._opt_absorb_overview()._validate()
            >>> opt.write_network_text()
            >>> opt_data = [d for n, d in opt.as_graph().nodes(data=True)]
            >>> assert opt_data[1]['meta']['space_slice'] == (slice(0, 202, None), slice(0, 200, None))
            >>> # Any sort of complex chain does prevents this optimization
            >>> # from running.
            >>> self = base.get_overview(1)[0:101, 0:100][0:50, 0:50].warp({'scale': 4})
            >>> opt = self._opt_absorb_overview()._validate()
            >>> opt.write_network_text()
            >>> opt_data = [d for n, d in opt.as_graph().nodes(data=True)]
            >>> assert 'DelayedOverview2' in [d['type'] for d in opt_data]
        """
        # Check if there is a strict downsampling component
        transform = self.meta['transform']
        params = transform.decompose()
        sx, sy = params['scale']
        if sx < 2 and sy < 2:
            return self

        # Lookahead to see if there is a nearby overview operation that can be
        # absorbed, remember the chain of operations between the warp and the
        # overview, as it will need to be modified.
        parent = self
        subdata = None
        chain = []
        num_dc = 0
        for i in range(4):
            subdata = parent.subdata
            if subdata is None:
                break
            elif isinstance(subdata, DelayedWarp2):
                subdata = None
                break
            elif isinstance(subdata, DelayedOverview2):
                # We found an overview node
                overview = subdata
                break
            elif isinstance(subdata, DelayedDequantize2):
                pass
            elif isinstance(subdata, DelayedCrop2):
                num_dc += 1
            else:
                subdata = None
                break
            chain.append(subdata)
            parent = subdata
        else:
            subdata = None

        if subdata is None:
            return self

        if num_dc > 1:
            return self

        # Replace the overview node with a warp node that mimics it.
        # This has no impact on the function of the operation stack.
        mimic_overview = overview._opt_overview_as_warp()
        tf1 = mimic_overview.meta['transform']

        # Handle any nodes between the warp and the overview.
        # This should be at most one quantization and one crop operation,
        # but we may generalize that in the future.
        if not chain:
            # The overview is directly after this warp
            new_head = mimic_overview.subdata
        else:
            # Copy the chain so this does not mutate the input
            chain = [copy.copy(n) for n in chain]
            for u, v in ub.iter_window(chain, 2):
                u.subdata = v
            tail = chain[-1]
            tail.subdata = mimic_overview
            # Check if the tail of the chain is a crop.
            if hasattr(tail, '_opt_warp_after_crop'):
                # This modifies the tail so it is now a warp followed by a
                # crop. Note that the warp may be different than the mimiced
                # overview, so use this new transform instead.
                # (Actually, I think this can't make the new crop non integral,
                # so it probably wont matter)
                modified_tail = tail._opt_warp_after_crop()
                new_chain_dsize = modified_tail.meta['dsize']
                tf1 = modified_tail.meta['transform']
                # Remove the modified warp
                tail_parent = chain[-2] if len(chain) > 1 else self
                new_tail = modified_tail.subdata
                tail_parent.subdata = new_tail
                chain[-1] = new_tail
                for notcrop in chain[:-1]:
                    notcrop.meta['dsize'] = new_chain_dsize
            else:
                # The chain does not contain a crop operation, we can safely
                # remove it.
                # Finally remove the overview transform entirely
                tail.subdata = mimic_overview.subdata
                new_chain_dsize = mimic_overview.subdata.meta['dsize']

            # The dsize within the chain might be wrong due to our
            # modification. I **think** its ok to just directly set it to the
            # new dsize as it should only be operations that do not change the
            # dsize, but it would be nice to find a more ellegant
            # implementation.
            for notcrop in chain[:-1]:
                notcrop.meta['dsize'] = new_chain_dsize
            new_head = chain[0]

        warp_meta = ub.dict_isect(self.meta, {'antialias', 'interpolation', 'border_value'})
        tf2 = self.meta['transform']
        dsize = self.meta['dsize']
        new_transform = tf2 @ tf1
        new = self.__class__(new_head, new_transform, dsize=dsize, **warp_meta)
        return new

    def _opt_split_warp_overview(self):
        """
        Split this node into a warp and an overview if possible

        Example:
            >>> # xdoctest: +REQUIRES(module:osgeo)
            >>> from kwcoco.util.delayed_ops.delayed_nodes import *  # NOQA
            >>> from kwcoco.util.delayed_ops import DelayedLoad2
            >>> import kwimage
            >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
            >>> self = DelayedLoad2(fpath, channels='r|g|b').prepare()
            >>> print(f'self={self}')
            >>> print('self.meta = {}'.format(ub.repr2(self.meta, nl=1)))
            >>> warp0 = self.warp({'scale': 0.2})
            >>> warp1 = warp0._opt_split_warp_overview()
            >>> warp2 = self.warp({'scale': 0.25})._opt_split_warp_overview()
            >>> print(ub.repr2(warp0.nesting(), nl=-1, sort=0))
            >>> print(ub.repr2(warp1.nesting(), nl=-1, sort=0))
            >>> print(ub.repr2(warp2.nesting(), nl=-1, sort=0))

        Example:
            >>> # xdoctest: +REQUIRES(module:osgeo)
            >>> from kwcoco.util.delayed_ops.delayed_nodes import *  # NOQA
            >>> from kwcoco.util.delayed_ops import DelayedLoad2
            >>> import kwimage
            >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
            >>> self = DelayedLoad2(fpath, channels='r|g|b').prepare()
            >>> warp0 = self.warp({'scale': 1 / 2 ** 6})
            >>> opt = warp0.optimize()
            >>> print(ub.repr2(warp0.nesting(), nl=-1, sort=0))
            >>> print(ub.repr2(opt.nesting(), nl=-1, sort=0))
        """
        inner_data = self.subdata
        num_overviews = inner_data.num_overviews
        if not num_overviews:
            return self

        # Check if there is a strict downsampling component
        transform = self.meta['transform']
        params = transform.decompose()
        sx, sy = params['scale']
        if sx > 0.5 or sy > 0.5:
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
                'antialias', 'interpolation', 'border_value'})
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
            subdata (DelayedArray2): data to operate on
            quantization (Dict):
                see :func:`kwcoco.util.delayed_ops.helpers.dequantize`
        """
        super().__init__(subdata)
        self.meta['quantization'] = quantization
        self.meta['dsize'] = subdata.dsize

    def _finalize(self):
        """
        Returns:
            ArrayLike
        """
        from kwcoco.util.delayed_ops.helpers import dequantize
        quantization = self.meta['quantization']
        final = self.subdata._finalize()
        final = kwarray.atleast_nd(final, 3, front=False)
        if quantization is not None:
            final = dequantize(final, quantization)
        return final

    @profile
    def optimize(self):
        """

        Returns:
            DelayedImage2

        Example:
            >>> # Test a case that caused an error in development
            >>> from kwcoco.util.delayed_ops.delayed_nodes import *  # NOQA
            >>> from kwcoco.util.delayed_ops import DelayedLoad2
            >>> fpath = kwimage.grab_test_image_fpath()
            >>> base = DelayedLoad2(fpath, channels='r|g|b').prepare()
            >>> quantization = {'quant_max': 255, 'nodata': 0}
            >>> self = base.get_overview(1).dequantize(quantization)
            >>> self.write_network_text()
            >>> opt = self.optimize()
        """
        new = copy.copy(self)
        new.subdata = self.subdata.optimize()

        if isinstance(new.subdata, DelayedDequantize2):
            raise AssertionError('Dequantization is only allowed once')

        if isinstance(new.subdata, DelayedWarp2):
            # Swap order so quantize is before the warp
            new = new._opt_dequant_before_other()
            new = new.optimize()

        if isinstance(new.subdata, DelayedChannelConcat2):
            new = new._opt_push_under_concat().optimize()
        return new

    def _opt_dequant_before_other(self):
        quantization = self.meta['quantization']
        new = copy.copy(self.subdata)
        new.subdata = new.subdata.dequantize(quantization)
        return new

    def _transform_from_subdata(self):
        return kwimage.Affine.eye()


class DelayedCrop2(DelayedImage2):
    """
    Crops an image along integer pixel coordinates.

    Example:
        >>> from kwcoco.util.delayed_ops.delayed_nodes import *  # NOQA
        >>> from kwcoco.util.delayed_ops import DelayedLoad2
        >>> base = DelayedLoad2.demo(dsize=(16, 16)).prepare()
        >>> # Test Fuse Crops Space Only
        >>> crop1 = base[4:12, 0:16]
        >>> self = crop1[2:6, 0:8]
        >>> opt = self._opt_fuse_crops()
        >>> self.write_network_text()
        >>> opt.write_network_text()
        >>> #
        >>> # Test Channel Select Via Index
        >>> self = base[:, :, [0]]
        >>> self.write_network_text()
        >>> final = self._finalize()
        >>> assert final.shape == (16, 16, 1)
        >>> assert base[:, :, [0, 1]].finalize().shape == (16, 16, 2)
        >>> assert base[:, :, [2, 0, 1]].finalize().shape == (16, 16, 3)
    """
    def __init__(self, subdata, space_slice=None, chan_idxs=None):
        """
        Args:
            subdata (DelayedArray2): data to operate on

            space_slice (Tuple[slice, slice]):
                if speficied, take this y-slice and x-slice.

            chan_idxs (List[int] | None):
                if specified, take these channels / bands
        """
        super().__init__(subdata)
        # TODO: are we doing infinite padding or clipping?
        # This assumes infinite padding
        in_w, in_h = subdata.dsize
        if space_slice is not None:
            space_dims = (in_h, in_w)
            slice_box = kwimage.Boxes.from_slice(
                space_slice, space_dims, wrap=True, clip=True)
            space_slice = slice_box.to_slices()[0]
            # width = slice_box.width.ravel()[0]
            # height = slice_box.height.ravel()[0]
            space_slice, _pad = kwarray.embed_slice(space_slice, space_dims)
            sl_y, sl_x = space_slice[0:2]
            width = sl_x.stop - sl_x.start
            height = sl_y.stop - sl_y.start
            self.meta['dsize'] = (width, height)
        else:
            space_slice = (slice(0, in_h), slice(0, in_w))
            self.meta['dsize'] = (in_w, in_h)

        if chan_idxs is not None:
            current_channels = self.channels
            if current_channels is not None:
                new_channels = current_channels[chan_idxs]
            else:
                new_channels = len(chan_idxs)
            self.channels = new_channels

        self.meta['space_slice'] = space_slice
        self.meta['chan_idxs'] = chan_idxs

    def _finalize(self):
        """
        Returns:
            ArrayLike
        """
        space_slice = self.meta['space_slice']
        chan_idxs = self.meta['chan_idxs']
        sub_final = self.subdata._finalize()
        if chan_idxs is None:
            full_slice = space_slice
        else:
            full_slice = space_slice + (chan_idxs,)
        # final = sub_final[space_slice]
        final = sub_final[full_slice]
        final = kwarray.atleast_nd(final, 3)
        return final

    @profile
    def optimize(self):
        """
        Returns:
            DelayedImage2

        Example:
            >>> # Test optimize nans
            >>> from kwcoco.util.delayed_ops import DelayedNans2
            >>> import kwimage
            >>> base = DelayedNans2(dsize=(100, 100), channels='a|b|c')
            >>> self = base[0:10, 0:5]
            >>> # Should simply return a new nan generator
            >>> new = self.optimize()
            >>> self.write_network_text()
            >>> new.write_network_text()
            >>> assert len(new.as_graph().nodes) == 1
        """
        new = copy.copy(self)
        new.subdata = self.subdata.optimize()
        if isinstance(new.subdata, DelayedCrop2):
            new = new._opt_fuse_crops()

        if hasattr(new.subdata, '_optimized_crop'):
            # The subdata knows how to optimize itself wrt this node
            crop_kwargs = ub.dict_isect(self.meta, {'space_slice', 'chan_idxs'})
            new = new.subdata._optimized_crop(**crop_kwargs).optimize()
        if isinstance(new.subdata, DelayedWarp2):
            new = new._opt_warp_after_crop()
            new = new.optimize()
        elif isinstance(new.subdata, DelayedDequantize2):
            new = new._opt_dequant_after_crop()
            new = new.optimize()

        if isinstance(new.subdata, DelayedChannelConcat2):
            if isinstance(new, DelayedCrop2):
                # We have to be careful if there we have band selection
                chan_idxs = new.meta.get('chan_idxs', None)
                space_slice = new.meta.get('space_slice', None)
                taken = new.subdata
                if chan_idxs is not None:
                    taken = new.subdata.take_channels(chan_idxs).optimize()
                if space_slice is not None:
                    taken = taken.crop(space_slice)._opt_push_under_concat().optimize()
                new = taken
            else:
                new = new._opt_push_under_concat().optimize()

        return new

    def _transform_from_subdata(self):
        sl_y, sl_x = self.meta['space_slice']
        offset = -sl_x.start, -sl_y.start
        self_from_subdata = kwimage.Affine.translate(offset)
        return self_from_subdata

    def _opt_fuse_crops(self):
        """
        Combine two consecutive crops into a single operation.

        Example:
            >>> from kwcoco.util.delayed_ops.delayed_nodes import *  # NOQA
            >>> from kwcoco.util.delayed_ops.delayed_leafs import DelayedLoad2
            >>> base = DelayedLoad2.demo(dsize=(16, 16)).prepare()
            >>> # Test Fuse Crops Space Only
            >>> crop1 = base[4:12, 0:16]
            >>> crop2 = self = crop1[2:6, 0:8]
            >>> opt = crop2._opt_fuse_crops()
            >>> self.write_network_text()
            >>> opt.write_network_text()
            >>> opt._validate()
            >>> self._validate()

        Example:
            >>> # Test Fuse Crops Channels Only
            >>> from kwcoco.util.delayed_ops.delayed_nodes import *  # NOQA
            >>> from kwcoco.util.delayed_ops.delayed_leafs import DelayedLoad2
            >>> base = DelayedLoad2.demo(dsize=(16, 16)).prepare()
            >>> crop1 = base.crop(chan_idxs=[0, 2, 1])
            >>> crop2 = crop1.crop(chan_idxs=[1, 2])
            >>> crop3 = self = crop2.crop(chan_idxs=[0, 1])
            >>> opt = self._opt_fuse_crops()._opt_fuse_crops()
            >>> self.write_network_text()
            >>> opt.write_network_text()
            >>> finalB = base._validate()._finalize()
            >>> final1 = opt._validate()._finalize()
            >>> final2 = self._validate()._finalize()
            >>> assert np.all(final2[..., 0] == finalB[..., 2])
            >>> assert np.all(final2[..., 1] == finalB[..., 1])
            >>> assert np.all(final2[..., 0] == final1[..., 0])
            >>> assert np.all(final2[..., 1] == final1[..., 1])

        Example:
            >>> # Test Fuse Crops Space  And Channels
            >>> from kwcoco.util.delayed_ops.delayed_nodes import *  # NOQA
            >>> from kwcoco.util.delayed_ops.delayed_leafs import DelayedLoad2
            >>> base = DelayedLoad2.demo(dsize=(16, 16)).prepare()
            >>> crop1 = base[4:12, 0:16, [1, 2]]
            >>> self = crop1[2:6, 0:8, [1]]
            >>> opt = self._opt_fuse_crops()
            >>> self.write_network_text()
            >>> opt.write_network_text()
            >>> self._validate()
            >>> crop1._validate()
        """
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

        # Handle bands
        inner_chan_idxs = self.subdata.meta['chan_idxs']
        outer_chan_idxs = self.meta['chan_idxs']
        if outer_chan_idxs is None and inner_chan_idxs is None:
            new_chan_idxs = None
        elif outer_chan_idxs is None:
            new_chan_idxs = inner_chan_idxs
        elif inner_chan_idxs is None:
            new_chan_idxs = outer_chan_idxs
        else:
            new_chan_idxs = list(ub.take(inner_chan_idxs, outer_chan_idxs))
        new_crop = (slice(new_ystart, new_ystop), slice(new_xstart, new_xstop))
        new = self.__class__(inner_data, new_crop, new_chan_idxs)
        return new

    def _opt_warp_after_crop(self):
        """
        If the child node is a warp, move it after the crop.

        This is more efficient because:
            1. The crop is closer to the load.
            2. we are warping with less data.

        Example:
            >>> from kwcoco.util.delayed_ops.delayed_nodes import *  # NOQA
            >>> from kwcoco.util.delayed_ops.delayed_leafs import DelayedLoad2
            >>> fpath = kwimage.grab_test_image_fpath()
            >>> node0 = DelayedLoad2(fpath, channels='r|g|b').prepare()
            >>> node1 = node0.warp({'scale': 0.432, 'theta': np.pi / 3, 'about': (80, 80), 'shearx': .3, 'offset': (-50, -50)})
            >>> node2 = node1[10:50, 1:40]
            >>> self = node2
            >>> new_outer = node2._opt_warp_after_crop()
            >>> print(ub.repr2(node2.nesting(), nl=-1, sort=0))
            >>> print(ub.repr2(new_outer.nesting(), nl=-1, sort=0))
            >>> final0 = self._finalize()
            >>> final1 = new_outer._finalize()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(final0, pnum=(2, 2, 1), fnum=1, title='raw')
            >>> kwplot.imshow(final1, pnum=(2, 2, 2), fnum=1, title='optimized')

        Example:
            >>> # xdoctest: +REQUIRES(module:osgeo)
            >>> from kwcoco.util.delayed_ops import *  # NOQA
            >>> from kwcoco.util.delayed_ops.delayed_leafs import DelayedLoad2
            >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
            >>> node0 = DelayedLoad2(fpath, channels='r|g|b').prepare()
            >>> node1 = node0.warp({'scale': 1000 / 512})
            >>> node2 = node1[250:750, 0:500]
            >>> self = node2
            >>> new_outer = node2._opt_warp_after_crop()
            >>> print(ub.repr2(node2.nesting(), nl=-1, sort=0))
            >>> print(ub.repr2(new_outer.nesting(), nl=-1, sort=0))
        """
        assert isinstance(self.subdata, DelayedWarp2)
        # Inner is the data closer to the leaf (disk), outer is the data closer
        # to the user (output).
        outer_slices = self.meta['space_slice']
        outer_chan_idxs = self.meta['chan_idxs']
        inner_transform = self.subdata.meta['transform']

        outer_region = kwimage.Boxes.from_slice(outer_slices)
        outer_region = outer_region.to_polygons()[0]

        from kwcoco.util.delayed_ops.helpers import _swap_warp_after_crop
        inner_slice, outer_transform = _swap_warp_after_crop(
            outer_region, inner_transform)

        warp_meta = ub.dict_isect(self.meta, {'dsize'})
        warp_meta.update(ub.dict_isect(
            self.subdata.meta, {'antialias', 'interpolation', 'border_value'}))

        new_inner = self.subdata.subdata.crop(inner_slice, outer_chan_idxs)
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
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> # Make a complex chain of operations and optimize it
        >>> from kwcoco.util.delayed_ops import *  # NOQA
        >>> import kwimage
        >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
        >>> dimg = DelayedLoad2(fpath, channels='r|g|b').prepare()
        >>> dimg = dimg.get_overview(1)
        >>> dimg = dimg.get_overview(1)
        >>> dimg = dimg.get_overview(1)
        >>> dopt = dimg.optimize()
        >>> if 1:
        >>>     import networkx as nx
        >>>     dimg.write_network_text()
        >>>     dopt.write_network_text()
        >>> print(ub.repr2(dopt.nesting(), nl=-1, sort=0))
        >>> final0 = dimg._finalize()[:]
        >>> final1 = dopt._finalize()[:]
        >>> assert final0.shape == final1.shape
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(final0, pnum=(1, 2, 1), fnum=1, title='raw')
        >>> kwplot.imshow(final1, pnum=(1, 2, 2), fnum=1, title='optimized')
    """
    def __init__(self, subdata, overview):
        """
        Args:
            subdata (DelayedArray2): data to operate on
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
        """
        Returns:
            int
        """
        # This operation reduces the number of available overviews
        num_remain = self.subdata.num_overviews - self.meta['overview']
        return num_remain

    def _finalize(self):
        """
        Returns:
            ArrayLike
        """
        sub_final = self.subdata._finalize()
        overview = self.meta['overview']
        if hasattr(sub_final, 'get_overview'):
            # This should be a lazy gdal frame
            if sub_final.num_overviews >= overview:
                final = sub_final.get_overview(overview)
                return final

        warnings.warn(ub.paragraph(
            '''
            The underlying data does not have overviews.
            Simulating the overview using a imresize operation.
            '''
        ))
        sub_final = np.asarray(sub_final)
        final = kwimage.imresize(
            sub_final,
            scale=1 / 2 ** overview,
            interpolation='nearest',
            # antialias=True
        )
        return final

    @profile
    def optimize(self):
        """
        Returns:
            DelayedImage2
        """
        new = copy.copy(self)
        new.subdata = self.subdata.optimize()
        if isinstance(new.subdata, DelayedOverview2):
            new = new._opt_fuse_overview()

        if new.meta['overview'] == 0:
            new = new.subdata
        elif isinstance(new.subdata, DelayedCrop2):
            new = new._opt_crop_after_overview()
            new = new.optimize()
        elif isinstance(new.subdata, DelayedWarp2):
            new = new._opt_warp_after_overview()
            new = new.optimize()
        elif isinstance(new.subdata, DelayedDequantize2):
            new = new._opt_dequant_after_overview()
            new = new.optimize()
        if isinstance(new.subdata, DelayedChannelConcat2):
            new = new._opt_push_under_concat().optimize()
        return new

    def _transform_from_subdata(self):
        scale = 1 / 2 ** self.meta['overview']
        return kwimage.Affine.scale(scale)

    def _opt_overview_as_warp(self):
        """
        Sometimes it is beneficial to replace an overview with a warp as an
        intermediate optimization step.
        """
        transform = self._transform_from_subdata()
        dsize = self.meta['dsize']
        new = self.subdata.warp(transform, dsize=dsize)
        return new

    def _opt_crop_after_overview(self):
        """
        Given an outer overview and an inner crop, switch places. We want the
        overview to be as close to the load as possible.

        Example:
            >>> # xdoctest: +REQUIRES(module:osgeo)
            >>> from kwcoco.util.delayed_ops import *  # NOQA
            >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
            >>> node0 = DelayedLoad2(fpath, channels='r|g|b').prepare()
            >>> node1 = node0[100:400, 120:450]
            >>> node2 = node1.get_overview(2)
            >>> self = node2
            >>> new_outer = node2.optimize()
            >>> print(ub.repr2(node2.nesting(), nl=-1, sort=0))
            >>> print(ub.repr2(new_outer.nesting(), nl=-1, sort=0))
            >>> final0 = self._finalize()
            >>> final1 = new_outer._finalize()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(final0, pnum=(1, 2, 1), fnum=1, title='raw')
            >>> kwplot.imshow(final1, pnum=(1, 2, 2), fnum=1, title='optimized')
        """
        from kwcoco.util.delayed_ops.helpers import _swap_crop_after_warp
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
            >>> # xdoctest: +REQUIRES(module:osgeo)
            >>> from kwcoco.util.delayed_ops import *  # NOQA
            >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
            >>> node0 = DelayedLoad2(fpath, channels='r|g|b').prepare()
            >>> node1 = node0.warp({'scale': (2.1, .7), 'offset': (20, 40)})
            >>> node2 = node1.get_overview(2)
            >>> self = node2
            >>> new_outer = node2.optimize()
            >>> print(ub.repr2(node2.nesting(), nl=-1, sort=0))
            >>> print(ub.repr2(new_outer.nesting(), nl=-1, sort=0))
            >>> final0 = self._finalize()
            >>> final1 = new_outer._finalize()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(final0, pnum=(1, 2, 1), fnum=1, title='raw')
            >>> kwplot.imshow(final1, pnum=(1, 2, 2), fnum=1, title='optimized')
        """
        assert isinstance(self.subdata, DelayedWarp2)
        outer_overview = self.meta['overview']
        inner_transform = self.subdata.meta['transform']
        outer_transform = self._transform_from_subdata()
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
