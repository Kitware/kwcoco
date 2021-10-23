import ubelt as ub


class CocoImage(ub.NiceRepr):
    """
    An object-oriented representation of a coco image.

    It provides helper methods that are specific to a single image.

    This operates directly on a single coco image dictionary, but it can
    optionally be connected to a parent dataset, which allows it to use
    CocoDataset methods to query about relationships and resolve pointers.

    This is different than the Images class in coco_object1d, which is just a
    vectorized interface to multiple objects.

    Example:
        >>> import kwcoco
        >>> dset1 = kwcoco.CocoDataset.demo('shapes8')
        >>> dset2 = kwcoco.CocoDataset.demo('vidshapes8-multispectral')

        >>> self = CocoImage(dset1.imgs[1], dset1)
        >>> print('self = {!r}'.format(self))
        >>> print('self.channels = {}'.format(ub.repr2(self.channels, nl=1)))

        >>> self = CocoImage(dset2.imgs[1], dset2)
        >>> print('self.channels = {}'.format(ub.repr2(self.channels, nl=1)))
        >>> self.primary_asset()
    """

    def __init__(self, img, dset=None):
        self.img = img
        self.dset = dset

    @classmethod
    def from_gid(cls, dset, gid):
        img = dset.index.imgs[gid]
        self = cls(img, dset=dset)
        return self

    def __nice__(self):
        """
        Example:
            >>> from kwcoco.coco_image import *  # NOQA
            >>> import kwcoco
            >>> with ub.CaptureStdout() as cap:
            ...     dset = kwcoco.CocoDataset.demo('shapes8')
            >>> self = CocoImage(dset.dataset['images'][0], dset)
            >>> print('self = {!r}'.format(self))

            >>> dset = kwcoco.CocoDataset.demo()
            >>> self = CocoImage(dset.dataset['images'][0], dset)
            >>> print('self = {!r}'.format(self))
        """
        from kwcoco.util.util_truncate import smart_truncate
        from functools import partial
        stats = self.stats()
        stats = ub.map_vals(str, stats)
        stats = ub.map_vals(
            partial(smart_truncate, max_length=32, trunc_loc=0.5),
            stats)
        return ub.repr2(stats, compact=1, nl=0, sort=0)

    def stats(self):
        """
        """
        key_attrname = [
            ('wh', 'dsize'),
            ('n_chan', 'num_channels'),
            ('channels', 'channels'),
        ]
        stats = {}
        for key, attrname in key_attrname:
            try:
                stats[key] = getattr(self, attrname)
            except Exception as ex:
                stats[key] = repr(ex)
        return stats

    def __getitem__(self, key):
        return self.img[key]

    def keys(self):
        return self.img.keys()

    def get(self, key, default=ub.NoParam):
        """
        Duck type some of the dict interface
        """
        if default is ub.NoParam:
            return self.img.get(key)
        else:
            return self.img.get(key, default)

    @property
    def channels(self):
        from kwcoco.channel_spec import FusedChannelSpec
        from kwcoco.channel_spec import ChannelSpec
        img_parts = []
        for obj in self.iter_asset_objs():
            obj_parts = obj.get('channels', None)
            obj_chan = FusedChannelSpec.coerce(obj_parts).normalize()
            img_parts.append(obj_chan.spec)
        spec = ChannelSpec(','.join(img_parts))
        return spec

    @property
    def num_channels(self):
        return self.channels.numel()
        # return sum(map(len, self.channels.streams()))

    @property
    def dsize(self):
        width = self.img.get('width', None)
        height = self.img.get('height', None)
        return width, height

    def primary_asset(self, requires=[]):
        """
        Compute a "main" image asset.

        Args:
            requires (List[str]):
                list of attribute that must be non-None to consider an object
                as the primary one.

        TODO:
            - [ ] Add in primary heuristics
        """
        import kwimage
        import numpy as np
        img = self.img
        has_base_image = img.get('file_name', None) is not None
        candidates = []

        if has_base_image:
            obj = img
            if all(k in obj for k in requires):
                # Return the base image if we can
                return obj

        # Choose "best" auxiliary image based on a hueristic.
        eye = kwimage.Affine.eye().matrix
        for obj in img.get('auxiliary', []):
            # Take frobenius norm to get "distance" between transform and
            # the identity. We want to find the auxiliary closest to the
            # identity transform.
            warp_aux_to_img = kwimage.Affine.coerce(obj.get('warp_aux_to_img', None))
            fro_dist = np.linalg.norm(warp_aux_to_img.matrix - eye, ord='fro')

            if all(k in obj for k in requires):
                candidates.append({
                    'area': obj['width'] * obj['height'],
                    'fro_dist': fro_dist,
                    'obj': obj,
                })

        if len(candidates) == 0:
            return None

        idx = ub.argmin(
            candidates, key=lambda val: (val['fro_dist'], -val['area'])
        )
        obj = candidates[idx]['obj']
        return obj

    def iter_asset_objs(self):
        """
        Iterate through base + auxiliary dicts that have file paths
        """
        img = self.img
        has_base_image = img.get('file_name', None) is not None
        if has_base_image:
            obj = img
            # cant remove auxiliary otherwise inplace modification doesnt work
            # obj = ub.dict_diff(img, {'auxiliary'})
            yield obj
        for obj in img.get('auxiliary', []):
            yield obj

    def delay(self, channels=None, space='image', bundle_dpath=None):
        """
        Experimental method

        Args:
            gid (int): image id to load

            channels (FusedChannelSpec): specific channels to load.
                if unspecified, all channels are loaded.

            space (str):
                can either be "image" for loading in image space, or
                "video" for loading in video space.

        TODO:
            - [X] Currently can only take all or none of the channels from each
                base-image / auxiliary dict. For instance if the main image is
                r|g|b you can't just select g|b at the moment.

            - [X] The order of the channels in the delayed load should
                match the requested channel order.

            - [ ] TODO: add nans to bands that don't exist or throw an error

        Example:
            >>> import kwcoco
            >>> gid = 1
            >>> #
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> self = CocoImage(dset.imgs[gid], dset)
            >>> delayed = self.delay()
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize()))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize(as_xarray=True)))
            >>> #
            >>> dset = kwcoco.CocoDataset.demo('shapes8')
            >>> delayed = dset.delayed_load(gid)
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize()))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize(as_xarray=True)))

            >>> crop = delayed.delayed_crop((slice(0, 3), slice(0, 3)))
            >>> crop.finalize()
            >>> crop.finalize(as_xarray=True)

            >>> # TODO: should only select the "red" channel
            >>> dset = kwcoco.CocoDataset.demo('shapes8')
            >>> delayed = CocoImage(dset.imgs[gid], dset).delay(channels='r')

            >>> import kwcoco
            >>> gid = 1
            >>> #
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> delayed = dset.delayed_load(gid, channels='B1|B2', space='image')
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize(as_xarray=True)))
            >>> delayed = dset.delayed_load(gid, channels='B1|B2|B11', space='image')
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize(as_xarray=True)))
            >>> delayed = dset.delayed_load(gid, channels='B8|B1', space='video')
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize(as_xarray=True)))

            >>> delayed = dset.delayed_load(gid, channels='B8|foo|bar|B1', space='video')
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize(as_xarray=True)))
        """
        from kwcoco.util.util_delayed_poc import DelayedLoad, DelayedChannelConcat
        from kwimage.transform import Affine
        from kwcoco.channel_spec import FusedChannelSpec
        if bundle_dpath is None:
            bundle_dpath = self.dset.bundle_dpath

        img = self.img
        requested = channels
        if requested is not None:
            requested = FusedChannelSpec.coerce(requested).normalize()

        def _delay_load_imglike(obj):
            from os.path import join
            info = {}
            fname = obj.get('file_name', None)
            channels_ = obj.get('channels', None)
            if channels_ is not None:
                channels_ = FusedChannelSpec.coerce(channels_).normalize()
            info['channels'] = channels_
            width = obj.get('width', None)
            height = obj.get('height', None)
            if height is not None and width is not None:
                info['dsize'] = dsize = (width, height)
            else:
                info['dsize'] = None
            if fname is not None:
                info['fpath'] = fpath = join(bundle_dpath, fname)
                info['chan'] = DelayedLoad(fpath, channels=channels_, dsize=dsize)
            return info

        # obj = img
        info = img_info = _delay_load_imglike(img)

        chan_list = []
        if info.get('chan', None) is not None:
            include_flag = requested is None
            if not include_flag:
                if requested.intersection(info['channels']):
                    include_flag = True
            if include_flag:
                chan_list.append(info.get('chan', None))

        for aux in img.get('auxiliary', []):
            info = _delay_load_imglike(aux)
            aux_to_img = Affine.coerce(aux.get('warp_aux_to_img', None))
            chan = info['chan']

            include_flag = requested is None
            if not include_flag:
                if requested.intersection(info['channels']):
                    include_flag = True
            if include_flag:
                chan = chan.delayed_warp(
                    aux_to_img, dsize=img_info['dsize'])
                chan_list.append(chan)

        if len(chan_list) == 0:
            raise ValueError('no data')
        else:
            delayed = DelayedChannelConcat(chan_list)

        # Reorder channels in the requested order
        if requested is not None:
            delayed = delayed.take_channels(requested)

        if hasattr(delayed, 'components'):
            if len(delayed.components) == 1:
                delayed = delayed.components[0]

        if space == 'image':
            pass
        elif space == 'video':
            vidid = img['video_id']
            video = self.dset.index.videos[vidid]
            width = video.get('width', img.get('width', None))
            height = video.get('height', img.get('height', None))
            video_dsize = (width, height)
            img_to_vid = Affine.coerce(img.get('warp_img_to_vid', None))
            delayed = delayed.delayed_warp(img_to_vid, dsize=video_dsize)
        else:
            raise KeyError('space = {}'.format(space))
        return delayed
