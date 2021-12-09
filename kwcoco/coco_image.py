import ubelt as ub
import numpy as np
from os.path import join


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
        self._bundle_dpath = None
        self._video = None

    @classmethod
    def from_gid(cls, dset, gid):
        img = dset.index.imgs[gid]
        self = cls(img, dset=dset)
        return self

    @property
    def bundle_dpath(self):
        if self.dset is not None:
            return self.dset.bundle_dpath
        else:
            return self._bundle_dpath

    @bundle_dpath.setter
    def bundle_dpath(self, value):
        self._bundle_dpath = value

    @property
    def video(self):
        """
        Helper to grab the video for this image if it exists
        """
        if self._video is None and self.dset is not None:
            vidid = self.img.get('video_id', None)
            if vidid is None:
                video = None
            else:
                video = self.dset.index.videos[vidid]
        else:
            video = self._video
        return video

    @video.setter
    def video(self, value):
        self._video = value

    def detach(self):
        """
        Removes references to the underlying coco dataset, but keeps special
        information such that it wont be needed.
        """
        self._bundle_dpath = self.bundle_dpath
        self._video = self.video
        self.dset = None
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
            # obj_chan = FusedChannelSpec.coerce(obj_parts).normalize()
            obj_chan = FusedChannelSpec.coerce(obj_parts)
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

    def primary_image_filepath(self, requires=None):
        dpath = self.bundle_dpath
        fname = self.primary_asset()['file_name']
        fpath = join(dpath, fname)
        return fpath

    def primary_asset(self, requires=None):
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
        if requires is None:
            requires = []
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

    def iter_image_filepaths(self):
        """
        """
        dpath = self.bundle_dpath
        for obj in self.iter_asset_objs():
            fname = obj.get('file_name')
            fpath = join(dpath, fname)
            yield fpath

    def iter_asset_objs(self):
        """
        Iterate through base + auxiliary dicts that have file paths

        Yields:
            dict: an image or auxiliary dictionary
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

    def add_auxiliary_item(self, file_name=None, channels=None,
                           imdata=None, warp_aux_to_img=None, width=None,
                           height=None, imwrite=False):
        """
        Adds an auxiliary item to the image dictionary.

        This operation can be done purely in-memory (the default), or the image
        data can be written to a file on disk (via the imwrite=True flag).

        Args:
            file_name (str | None):
                The name of the file relative to the bundle directory. If
                unspecified, imdata must be given.

            channels (str | kwcoco.FusedChannelSpec):
                The channel code indicating what each of the bands represents.
                These channels should be disjoint wrt to the existing data in
                this image (this is not checked).

            imdata (ndarray | None):
                The underlying image data this auxiliary item represents.  If
                unspecified, it is assumed file_name points to a path on disk
                that will eventually exist. If imdata, file_name, and the
                special imwrite=True flag are specified, this function will
                write the data to disk.

            warp_aux_to_img (kwimage.Affine):
                The transformation from this auxiliary space to image space.
                If unspecified, assumes this item is related to image space by
                only a scale factor.

            width (int):
                Width of the data in auxiliary space (inferred if unspecified)

            height (int):
                Height of the data in auxiliary space (inferred if unspecified)

            imwrite (bool):
                If specified, both imdata and file_name must be specified, and
                this will write the data to disk. Note: it it recommended that
                you simply call imwrite yourself before or after calling this
                function. This lets you better control imwrite parameters.

        TODO:
            - [ ] Allow imwrite to specify an executor that is used to
            return a Future so the imwrite call does not block.

        Example:
            >>> from kwcoco.coco_image import *  # NOQA
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> coco_img = dset.coco_image(1)
            >>> imdata = np.random.rand(32, 32, 5)
            >>> channels = kwcoco.FusedChannelSpec.coerce('Aux:5')
            >>> coco_img.add_auxiliary_item(imdata=imdata, channels=channels)
        """
        from os.path import isabs, join  # NOQA
        import kwimage
        from kwcoco.channel_spec import FusedChannelSpec

        img = self.img

        if imdata is None and file_name is None:
            raise ValueError('must specify file_name or imdata')

        if width is None and imdata is not None:
            width = imdata.shape[1]

        if height is None and imdata is not None:
            height = imdata.shape[0]

        if warp_aux_to_img is None:
            if width is None or height is None:
                raise ValueError('unable to infer aux_to_img without width')
            # Assume we can just scale up the auxiliary data to match the image
            # space unless the user says otherwise
            warp_aux_to_img = kwimage.Affine.scale((
                img['width'] / width, img['height'] / height))

        if channels is not None:
            channels = FusedChannelSpec.coerce(channels).spec

        # Make the aux info dict
        aux = {
            'file_name': file_name,
            'height': height,
            'width': width,
            'channels': channels,
            'warp_aux_to_img': warp_aux_to_img.concise(),
        }
        if imdata is not None:
            if imwrite:
                if __debug__ and file_name is None:
                    raise ValueError(
                        'file_name must be given if imwrite is True')
                # if self.dset is None:
                #     fpath = file_name
                #     if not isabs(fpath):
                #         raise ValueError(ub.paragraph(
                #             '''
                #             Got relative file_name, but no dataset is attached
                #             to this coco image. Attatch a dataset or use an
                #             absolute path.
                #             '''))
                # else:
                fpath = join(self.bundle_dpath, file_name)
                kwimage.imwrite(fpath, imdata)
            else:
                aux['imdata'] = imdata

        auxiliary = img.get('auxiliary', None)
        if auxiliary is None:
            auxiliary = img['auxiliary'] = []
        auxiliary.append(aux)
        if self.dset is not None:
            self.dset._invalidate_hashid()

    def delay(self, channels=None, space='image', bundle_dpath=None):
        """
        Perform a delayed load on the data in this image.

        The delayed load can load a subset of channels, and perform lazy
        warping operations. If the underlying data is in a tiled format this
        can reduce the amount of disk IO needed to read the data if only a
        small crop or lower resolution view of the data is needed.

        Note:
            This method is experimental and relies on the delayed load
            proof-of-concept.

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

            - [X] TODO: add nans to bands that don't exist or throw an error

            - [ ] This function could stand to have a better name. Maybe imread
                  with a delayed=True flag? Or maybe just delayed_load?

        Example:
            >>> from kwcoco.coco_image import *  # NOQA
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

        Example:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo()
            >>> coco_img = dset.coco_image(1)
            >>> # Test case where nothing is registered in the dataset
            >>> delayed = coco_img.delay()
            >>> final = delayed.finalize()
            >>> assert final.shape == (512, 512, 3)

        Example:
            >>> # Test that delay works when imdata is stored in the image
            >>> # dictionary itself.
            >>> from kwcoco.coco_image import *  # NOQA
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> coco_img = dset.coco_image(1)
            >>> imdata = np.random.rand(6, 6, 5)
            >>> imdata[:] = np.arange(5)[None, None, :]
            >>> channels = kwcoco.FusedChannelSpec.coerce('Aux:5')
            >>> coco_img.add_auxiliary_item(imdata=imdata, channels=channels)
            >>> delayed = coco_img.delay(channels='B1|Aux:2:4')
            >>> final = delayed.finalize()
        """
        from kwcoco.util.util_delayed_poc import DelayedChannelConcat
        from kwcoco.util.util_delayed_poc import DelayedNans
        from kwimage.transform import Affine
        from kwcoco.channel_spec import FusedChannelSpec
        if bundle_dpath is None:
            bundle_dpath = self.bundle_dpath

        img = self.img
        requested = channels
        if requested is not None:
            requested = FusedChannelSpec.coerce(requested).normalize()

        # Get info about the primary image and check if its channels are
        # requested (if it even has any)
        img_info = _delay_load_imglike(bundle_dpath, img)
        obj_info_list = [(img_info, img)]
        for aux in img.get('auxiliary', []):
            info = _delay_load_imglike(bundle_dpath, aux)
            obj_info_list.append((info, aux))

        chan_list = []
        for info, obj in obj_info_list:
            if info.get('chan_construct', None) is not None:
                include_flag = requested is None
                if not include_flag:
                    if requested.intersection(info['channels']):
                        include_flag = True
                if include_flag:
                    aux_to_img = Affine.coerce(obj.get('warp_aux_to_img', None))
                    chncls, chnkw = info['chan_construct']
                    chan = chncls(**chnkw)
                    chan = chan.delayed_warp(
                        aux_to_img, dsize=img_info['dsize'])
                    chan_list.append(chan)

        if space == 'video':
            video = self.video
            width = video.get('width', img.get('width', None))
            height = video.get('height', img.get('height', None))
        else:
            width = img.get('width', None)
            height = img.get('height', None)
        dsize = (width, height)

        if len(chan_list) == 0:
            if requested is not None:
                # Handle case where the image doesnt have the requested
                # channels.
                delayed = DelayedNans(dsize=dsize, channels=requested)
                return delayed
            else:
                raise ValueError('no data registered in kwcoco image')
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
            img_to_vid = Affine.coerce(img.get('warp_img_to_vid', None))
            delayed = delayed.delayed_warp(img_to_vid, dsize=dsize)
        else:
            raise KeyError('space = {}'.format(space))
        return delayed


def _delay_load_imglike(bundle_dpath, obj):
    from kwcoco.util.util_delayed_poc import DelayedLoad, DelayedIdentity
    from os.path import join
    from kwcoco.channel_spec import FusedChannelSpec
    info = {}
    fname = obj.get('file_name', None)
    imdata = obj.get('imdata', None)
    channels_ = obj.get('channels', None)
    if channels_ is not None:
        channels_ = FusedChannelSpec.coerce(channels_)
        channels_ = channels_.normalize()
    info['channels'] = channels_
    width = obj.get('width', None)
    height = obj.get('height', None)
    if height is not None and width is not None:
        info['dsize'] = dsize = (width, height)
    else:
        info['dsize'] = dsize = (None, None)
    if imdata is not None:
        info['chan_construct'] = (DelayedIdentity, dict(
            sub_data=imdata, channels=channels_, dsize=dsize))
    elif fname is not None:
        info['fpath'] = fpath = join(bundle_dpath, fname)
        # Delaying this gives us a small speed boost
        info['chan_construct'] = (DelayedLoad, dict(
            fpath=fpath, channels=channels_, dsize=dsize))
    return info
