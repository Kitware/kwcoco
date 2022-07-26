import ubelt as ub
import numpy as np
from os.path import join

try:
    from xdev import profile
except Exception:
    profile = ub.identity


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

    def __contains__(self, key):
        return key in self.keys()

    def __getitem__(self, key):
        """
        Proxy getter attribute for underlying `self.img` dictionary
        """
        return self.get(key)

    def keys(self):
        """
        Proxy getter attribute for underlying `self.img` dictionary
        """
        if 'extra' in self.img:
            # SQL compatibility
            _keys = ub.flatten([self.img.keys(), self.img['extra'].keys()])
            return iter((k for k in _keys if k != 'extra'))
        else:
            return self.img.keys()

    def get(self, key, default=ub.NoParam):
        """
        Proxy getter attribute for underlying `self.img` dictionary

        Example:
            >>> import pytest
            >>> # without extra populated
            >>> import kwcoco
            >>> self = kwcoco.CocoImage({'foo': 1})
            >>> assert self.get('foo') == 1
            >>> assert self.get('foo', None) == 1
            >>> # with extra populated
            >>> self = kwcoco.CocoImage({'extra': {'foo': 1}})
            >>> assert self.get('foo') == 1
            >>> assert self.get('foo', None) == 1
            >>> # without extra empty
            >>> self = kwcoco.CocoImage({})
            >>> with pytest.raises(KeyError):
            >>>     self.get('foo')
            >>> assert self.get('foo', None) is None
            >>> # with extra empty
            >>> self = kwcoco.CocoImage({'extra': {'bar': 1}})
            >>> with pytest.raises(KeyError):
            >>>     self.get('foo')
            >>> assert self.get('foo', None) is None
        """
        _img = self.img
        if default is ub.NoParam:
            if 'extra' in _img:
                # Workaround for sql-view
                if key in _img:
                    return _img[key]
                else:
                    _extra = _img['extra']
                    if key in _extra:
                        return _extra[key]
                    else:
                        raise KeyError(key)
            else:
                return _img[key]
        else:
            if 'extra' in _img:
                # Workaround for sql-view
                if key in _img:
                    return _img.get(key, default)
                else:
                    return _img['extra'].get(key, default)
            else:
                return _img.get(key, default)

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

        Notes:
            Uses a heuristic.

            * First, try to find the auxiliary image that has with the smallest
            distortion to the base image (if known via warp_aux_to_img)

            * Second, break ties by using the largest image if w / h is known

            * Last, if previous information not available use the first
              auxiliary image.

        Args:
            requires (List[str]):
                list of attribute that must be non-None to consider an object
                as the primary one.

        TODO:
            - [ ] Add in primary heuristics

        Example:
            >>> import kwarray
            >>> from kwcoco.coco_image import *  # NOQA
            >>> rng = kwarray.ensure_rng(0)
            >>> def random_auxiliary(name, w=None, h=None):
            >>>     return {'file_name': name, 'width': w, 'height': h}
            >>> self = CocoImage({
            >>>     'auxiliary': [
            >>>         random_auxiliary('1'),
            >>>         random_auxiliary('2'),
            >>>         random_auxiliary('3'),
            >>>     ]
            >>> })
            >>> assert self.primary_asset()['file_name'] == '1'
            >>> self = CocoImage({
            >>>     'auxiliary': [
            >>>         random_auxiliary('1'),
            >>>         random_auxiliary('2', 3, 3),
            >>>         random_auxiliary('3'),
            >>>     ]
            >>> })
            >>> assert self.primary_asset()['file_name'] == '2'
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
        asset_objs = img.get('auxiliary', img.get('assets', []))
        for idx, obj in enumerate(asset_objs):
            # Take frobenius norm to get "distance" between transform and
            # the identity. We want to find the auxiliary closest to the
            # identity transform.
            warp_aux_to_img = kwimage.Affine.coerce(obj.get('warp_aux_to_img', None))
            fro_dist = np.linalg.norm(warp_aux_to_img - eye, ord='fro')
            w = obj.get('width', None) or 0
            h = obj.get('height', None) or 0
            if all(k in obj for k in requires):
                candidates.append({
                    'idx': idx,
                    'area': w * h,
                    'fro_dist': fro_dist,
                    'obj': obj,
                })

        if len(candidates) == 0:
            return None

        idx = ub.argmin(
            candidates, key=lambda val: (
                val['fro_dist'], -val['area'], val['idx'])
        )
        obj = candidates[idx]['obj']
        return obj

    def iter_image_filepaths(self, with_bundle=True):
        """
        Could rename to iter_asset_filepaths

        Args:
            with_bundle (bool):
                If True, prepends the bundle dpath to fully specify the path.
                Otherwise, just returns the registered string in the file_name
                attribute of each asset.  Defaults to True.
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
        for obj in img.get('assets', []):
            yield obj

    def find_asset_obj(self, channels):
        """
        Find the asset dictionary with the specified channels

        Example:
            >>> import kwcoco
            >>> coco_img = kwcoco.CocoImage({'width': 128, 'height': 128})
            >>> coco_img.add_auxiliary_item(
            >>>     'rgb.png', channels='red|green|blue', width=32, height=32)
            >>> assert coco_img.find_asset_obj('red') is not None
            >>> assert coco_img.find_asset_obj('green') is not None
            >>> assert coco_img.find_asset_obj('blue') is not None
            >>> assert coco_img.find_asset_obj('red|blue') is not None
            >>> assert coco_img.find_asset_obj('red|green|blue') is not None
            >>> assert coco_img.find_asset_obj('red|green|blue') is not None
            >>> assert coco_img.find_asset_obj('black') is None
            >>> assert coco_img.find_asset_obj('r') is None

        Example:
            >>> # Test with concise channel code
            >>> import kwcoco
            >>> coco_img = kwcoco.CocoImage({'width': 128, 'height': 128})
            >>> coco_img.add_auxiliary_item(
            >>>     'msi.png', channels='foo.0:128', width=32, height=32)
            >>> assert coco_img.find_asset_obj('foo') is None
            >>> assert coco_img.find_asset_obj('foo.3') is not None
            >>> assert coco_img.find_asset_obj('foo.3:5') is not None
            >>> assert coco_img.find_asset_obj('foo.3000') is None
        """
        from kwcoco.channel_spec import FusedChannelSpec
        channels = FusedChannelSpec.coerce(channels)
        found = None
        for obj in self.iter_asset_objs():
            obj_channels = FusedChannelSpec.coerce(obj['channels'])
            if (obj_channels & channels).numel():
                found = obj
                break
        return found

    def _assets_key(self):
        """
        Internal helper for transition from auxiliary -> assets in the image
        spec
        """
        if 'auxiliary' in self:
            return 'auxiliary'
        elif 'assets' in self:
            return 'assets'
        else:
            return 'auxiliary'

    def add_auxiliary_item(self, file_name=None, channels=None,
                           imdata=None, warp_aux_to_img=None, width=None,
                           height=None, imwrite=False):
        """
        Adds an auxiliary / asset item to the image dictionary.

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

        Example:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset()
            >>> gid = dset.add_image(name='my_image_name', width=200, height=200)
            >>> coco_img = dset.coco_image(gid)
            >>> coco_img.add_auxiliary_item('path/img1_B0.tif', channels='B0', width=200, height=200)
            >>> coco_img.add_auxiliary_item('path/img1_B1.tif', channels='B1', width=200, height=200)
            >>> coco_img.add_auxiliary_item('path/img1_B2.tif', channels='B2', width=200, height=200)
            >>> coco_img.add_auxiliary_item('path/img1_TCI.tif', channels='r|g|b', width=200, height=200)
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
            img_width = img.get('width', None)
            img_height = img.get('height', None)
            if img_width is None or img_height is None:
                raise ValueError('Parent image canvas has an unknown size. '
                                 'Need to set width/height')
            if width is None or height is None:
                raise ValueError('Unable to infer aux_to_img without width')
            # Assume we can just scale up the auxiliary data to match the image
            # space unless the user says otherwise
            warp_aux_to_img = kwimage.Affine.scale((
                img_width / width, img_height / height))
        else:
            warp_aux_to_img = kwimage.Affine.coerce(warp_aux_to_img)

        if channels is not None:
            channels = FusedChannelSpec.coerce(channels).spec

        # Make the aux info dict
        obj = {
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
                obj['imdata'] = imdata

        assets_key = self._assets_key()
        asset_list = img.get(assets_key, None)
        if asset_list is None:
            asset_list = img[assets_key] = []
        asset_list.append(obj)
        if self.dset is not None:
            self.dset._invalidate_hashid()

    # Alias for add_auxiliary_item (which will eventually be deprecated)
    add_asset = add_auxiliary_item

    @profile
    def delay(self, channels=None, space='image', bundle_dpath=None,
              interpolation='linear', antialias=True, nodata_method=None,
              mode=1):
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

            channels (kwcoco.FusedChannelSpec): specific channels to load.
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
wc
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
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize()))
            >>> #
            >>> dset = kwcoco.CocoDataset.demo('shapes8')
            >>> delayed = dset.coco_image(gid).delay()
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize()))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize()))

            >>> crop = delayed.crop((slice(0, 3), slice(0, 3)))
            >>> crop.finalize()

            >>> # TODO: should only select the "red" channel
            >>> dset = kwcoco.CocoDataset.demo('shapes8')
            >>> delayed = CocoImage(dset.imgs[gid], dset).delay(channels='r')

            >>> import kwcoco
            >>> gid = 1
            >>> #
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> delayed = dset.coco_image(gid).delay(channels='B1|B2', space='image')
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize()))
            >>> delayed = dset.coco_image(gid).delay(channels='B1|B2|B11', space='image')
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize()))
            >>> delayed = dset.coco_image(gid).delay(channels='B8|B1', space='video')
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize()))

            >>> delayed = dset.coco_image(gid).delay(channels='B8|foo|bar|B1', space='video')
            >>> print('delayed = {!r}'.format(delayed))
            >>> print('delayed.finalize() = {!r}'.format(delayed.finalize()))

        Example:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo()
            >>> coco_img = dset.coco_image(1)
            >>> # Test case where nothing is registered in the dataset
            >>> delayed = coco_img.delay()
            >>> final = delayed.finalize()
            >>> assert final.shape == (512, 512, 3)

            >>> delayed = coco_img.delay(mode=1)
            >>> final = delayed.finalize()
            >>> print('final.shape = {}'.format(ub.repr2(final.shape, nl=1)))
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
            >>> delayed = coco_img.delay(channels='B1|Aux:2:4', mode=1)
            >>> final = delayed.finalize()

        Example:
            >>> # Test delay when loading in asset space
            >>> from kwcoco.coco_image import *  # NOQA
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-msi-multisensor')
            >>> coco_img = dset.coco_image(1)
            >>> stream1 = coco_img.channels.streams()[0]
            >>> stream2 = coco_img.channels.streams()[1]
            >>> aux_delayed = coco_img.delay(stream1, space='asset')
            >>> img_delayed = coco_img.delay(stream1, space='image')
            >>> vid_delayed = coco_img.delay(stream1, space='video')
            >>> #
            >>> aux_imdata = aux_delayed.as_xarray().finalize()
            >>> img_imdata = img_delayed.as_xarray().finalize()
            >>> assert aux_imdata.shape != img_imdata.shape
            >>> # Cannot load multiple asset items at the same time in
            >>> # asset space
            >>> import pytest
            >>> fused_channels = stream1 | stream2
            >>> with pytest.raises(kwcoco.exceptions.CoordinateCompatibilityError):
            >>>     aux_delayed2 = coco_img.delay(fused_channels, space='asset')
        """
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
        img_info = _delay_load_imglike(bundle_dpath, img, mode=mode,
                                       nodata_method=nodata_method)
        obj_info_list = [(img_info, img)]
        auxlist = img.get('auxiliary', img.get('assets', []))
        for aux in auxlist:
            info = _delay_load_imglike(bundle_dpath, aux, mode=mode,
                                       nodata_method=nodata_method)
            obj_info_list.append((info, aux))

        chan_list = []
        for info, obj in obj_info_list:
            if info.get('chan_construct', None) is not None:
                include_flag = requested is None
                if not include_flag:
                    if requested.intersection(info['channels']):
                        include_flag = True
                if include_flag:
                    chncls, chnkw = info['chan_construct']
                    chan = chncls(**chnkw)
                    if mode == 1:
                        quant = info.get('quantization', None)
                        if quant is not None:
                            chan = chan.dequantize(quant)
                    if space not in {'auxiliary', 'asset'}:
                        aux_to_img = Affine.coerce(obj.get('warp_aux_to_img', None))
                        chan = chan.warp(
                            aux_to_img, dsize=img_info['dsize'])
                    chan_list.append(chan)

        # TODO: allow load in auxiliary/asset space

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
                if mode == 0:
                    from kwcoco.util.util_delayed_poc import DelayedNans
                    delayed = DelayedNans(dsize=dsize, channels=requested)
                elif mode == 1:
                    from kwcoco.util.delayed_ops import DelayedNans2
                    from kwcoco.util.delayed_ops import DelayedChannelConcat2
                    delayed = DelayedNans2(dsize=dsize, channels=requested)
                    delayed = DelayedChannelConcat2([delayed])
                else:
                    raise KeyError(mode)
                return delayed
            else:
                raise ValueError('no data registered in kwcoco image')
        else:
            if mode == 0:
                from kwcoco.util.util_delayed_poc import DelayedChannelConcat
                delayed = DelayedChannelConcat(chan_list)
            elif mode == 1:
                from kwcoco.util.delayed_ops import DelayedChannelConcat2
                delayed = DelayedChannelConcat2(chan_list)
            else:
                raise KeyError(mode)

        # Reorder channels in the requested order
        if requested is not None:
            delayed = delayed.take_channels(requested)

        if hasattr(delayed, 'components'):
            if len(delayed.components) == 1:
                delayed = delayed.components[0]

        if space in {'image', 'auxiliary', 'asset'}:
            pass
        elif space == 'video':
            img_to_vid = self.warp_vid_from_img
            # img_to_vid = Affine.coerce(img.get('warp_img_to_vid', None))
            if mode == 1:
                delayed = delayed.warp(img_to_vid, dsize=dsize,
                                       interpolation=interpolation,
                                       antialias=antialias)
            else:
                delayed = delayed.warp(img_to_vid, dsize=dsize)
        else:
            raise KeyError('space = {}'.format(space))

        # if mode == 1:
        #     delayed = delayed.optimize()

        return delayed

    @ub.memoize_method
    def valid_region(self, space='image'):
        """
        If this image has a valid polygon, return it in image, or video space
        """
        import kwimage
        valid_coco_poly = self.img.get('valid_region', None)
        if valid_coco_poly is None:
            valid_poly = None
        else:
            kw_poly_img = kwimage.MultiPolygon.coerce(valid_coco_poly)
            if kw_poly_img is None:
                valid_poly = None
            else:
                if space == 'image':
                    valid_poly = kw_poly_img
                elif space == 'video':
                    warp_vid_from_img = self.warp_vid_from_img
                    valid_poly = kw_poly_img.warp(warp_vid_from_img)
                else:
                    # To warp it into an auxiliary space we need to know which one
                    raise NotImplementedError(space)
        return valid_poly

    # def warp_vid_from_img(self):
    #     pass

    # def warp_vid_from_img(self):
    #     pass

    @ub.memoize_property
    def warp_vid_from_img(self):
        import kwimage
        warp_img_to_vid = kwimage.Affine.coerce(self.img.get('warp_img_to_vid', None))
        if warp_img_to_vid.matrix is None:
            # Hack to ensure the matrix property always is an array
            warp_img_to_vid.matrix = np.asarray(warp_img_to_vid)
        return warp_img_to_vid

    @ub.memoize_property
    def warp_img_from_vid(self):
        return self.warp_vid_from_img.inv()

    def _annot_segmentation(self, ann, space='video'):
        import kwimage
        warp_vid_from_img = self.warp_vid_from_img
        img_sseg = kwimage.MultiPolygon.coerce(ann['segmentation'])
        if space == 'image':
            warped_sseg = img_sseg
            pass
        elif space == 'video':
            vid_sseg = img_sseg.warp(warp_vid_from_img)
            warped_sseg = vid_sseg
        else:
            raise NotImplementedError(space)  # auxiliary/asset space
        return warped_sseg


class CocoAsset(object):
    """
    A Coco Asset / Auxiliary Item

    Represents one 2D image file relative to a parent img.

    Could be a single asset, or an image with sub-assets, but sub-assets are
    ignored here.

    Initially we called these "auxiliary" items, but I think we should
    change their name to "assets", which better maps with STAC terminology.
    """

    def __getitem__(self, key):
        """
        Proxy getter attribute for underlying `self.obj` dictionary
        """
        return self.obj[key]

    def keys(self):
        """
        Proxy getter attribute for underlying `self.obj` dictionary
        """
        return self.obj.keys()

    def get(self, key, default=ub.NoParam):
        """
        Proxy getter attribute for underlying `self.obj` dictionary
        """
        if default is ub.NoParam:
            return self.obj.get(key)
        else:
            return self.obj.get(key, default)


def _delay_load_imglike(bundle_dpath, obj, mode=0, nodata_method=None):
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

    if mode == 0:
        from kwcoco.util.util_delayed_poc import DelayedLoad, DelayedIdentity
        quantization = obj.get('quantization', None)
        if imdata is not None:
            info['chan_construct'] = (DelayedIdentity, dict(
                sub_data=imdata, channels=channels_, dsize=dsize,
                quantization=quantization))
        elif fname is not None:
            info['fpath'] = fpath = join(bundle_dpath, fname)
            # Delaying this gives us a small speed boost
            info['chan_construct'] = (DelayedLoad, dict(
                fpath=fpath, channels=channels_, dsize=dsize,
                quantization=quantization))
    elif mode == 1:
        from kwcoco.util.delayed_ops import DelayedLoad2, DelayedIdentity2
        quantization = obj.get('quantization', None)
        if imdata is not None:
            info['chan_construct'] = (DelayedIdentity2, dict(
                data=imdata, channels=channels_, dsize=dsize))
        elif fname is not None:
            info['fpath'] = fpath = join(bundle_dpath, fname)
            info['chan_construct'] = (
                DelayedLoad2,
                dict(fpath=fpath, channels=channels_, dsize=dsize,
                     nodata_method=nodata_method))
        info['quantization'] = quantization

    return info
