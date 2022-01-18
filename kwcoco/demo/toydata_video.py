"""
Generates "toydata" for demo and testing purposes.

This is the video version of the toydata generator and should be prefered to
the loose image version in toydata_image.

"""
from os.path import join
import numpy as np
import ubelt as ub
import kwarray
import kwimage
from kwcoco.demo.toypatterns import CategoryPatterns


try:
    from xdev import profile
except Exception:
    profile = ub.identity


TOYDATA_VIDEO_VERSION = 21


@profile
def random_video_dset(
        num_videos=1, num_frames=2, num_tracks=2, anchors=None,
        image_size=(600, 600), verbose=3, render=False, aux=None,
        multispectral=False, multisensor=False, rng=None, dpath=None,
        max_speed=0.01, channels=None, **kwargs):
    """
    Create a toy Coco Video Dataset

    Args:
        num_videos (int) : number of videos

        num_frames (int) : number of images per video

        num_tracks (int) : number of tracks per video

        image_size (Tuple[int, int]):
            The width and height of the generated images

        render (bool | dict):
            if truthy the toy annotations are synthetically rendered. See
            :func:`render_toy_image` for details.

        rng (int | None | RandomState): random seed / state

        dpath (str): only used if render is truthy, place to write rendered
            images.

        verbose (int, default=3): verbosity mode

        aux (bool): if True generates dummy auxiliary channels

        multispectral (bool): similar to aux, but does not have the concept of
            a "main" image.

        max_speed (float): max speed of movers

        channels (str): experimental new way to get MSI with specific
            band distributions.

        **kwargs : used for old backwards compatible argument names
            gsize - alias for image_size

    SeeAlso:
        random_single_video_dset

    Example:
        >>> from kwcoco.demo.toydata_video import *  # NOQA
        >>> dset = random_video_dset(render=True, num_videos=3, num_frames=2,
        >>>                          num_tracks=5, image_size=(128, 128))
        >>> # xdoctest: +REQUIRES(--show)
        >>> dset.show_image(1, doclf=True)
        >>> dset.show_image(2, doclf=True)

        >>> from kwcoco.demo.toydata_video import *  # NOQA
        dset = random_video_dset(render=False, num_videos=3, num_frames=2,
            num_tracks=10)
        dset._tree()
        dset.imgs[1]

        dset = random_single_video_dset()
        dset._tree()
        dset.imgs[1]

        from kwcoco.demo.toydata_video import *  # NOQA
        dset = random_video_dset(render=True, num_videos=3, num_frames=2,
           num_tracks=10)
        print(dset.imgs[1])
        print('dset.bundle_dpath = {!r}'.format(dset.bundle_dpath))
        dset._tree()

        import xdev
        globals().update(xdev.get_func_kwargs(random_video_dset))
        num_videos = 2
    """
    if 'gsize' in kwargs:  # nocover
        if 0:
            # TODO: enable this warning
            import warnings
            warnings.warn('gsize is deprecated. Use image_size param instead',
                          DeprecationWarning)
        image_size = kwargs.pop('gsize')
    if len(kwargs) != 0:
        raise ValueError('unknown kwargs={}'.format(kwargs))

    rng = kwarray.ensure_rng(rng)
    subsets = []
    tid_start = 1
    gid_start = 1
    for vidid in range(1, num_videos + 1):
        if verbose > 2:
            print('generate vidid = {!r}'.format(vidid))

        dset = random_single_video_dset(
            image_size=image_size, num_frames=num_frames,
            num_tracks=num_tracks, tid_start=tid_start, anchors=anchors,
            gid_start=gid_start, video_id=vidid, render=False, autobuild=False,
            aux=aux, multispectral=multispectral, multisensor=multisensor,
            max_speed=max_speed, channels=channels, rng=rng, verbose=verbose)
        try:
            gid_start = dset.dataset['images'][-1]['id'] + 1
            tid_start = dset.dataset['annotations'][-1]['track_id'] + 1
        except IndexError:
            pass
        subsets.append(dset)

    if num_videos == 0:
        raise AssertionError
    if num_videos == 1:
        dset = subsets[0]
    else:
        import kwcoco
        assert len(subsets) > 1, '{}'.format(len(subsets))
        dset = kwcoco.CocoDataset.union(*subsets)

    # The dataset has been prepared, now we just render it and we have
    # a nice video dataset.
    renderkw = {
        'dpath': None,
    }
    if isinstance(render, dict):
        renderkw.update(render)
    else:
        if not render:
            renderkw = None

    if renderkw:
        if dpath is None:
            dpath = ub.ensure_app_cache_dir('kwcoco', 'demo_vidshapes')

        if verbose > 2:
            print('rendering')
        render_toy_dataset(dset, rng=rng, dpath=dpath, renderkw=renderkw,
                           verbose=verbose)
        dset.fpath = join(dpath, 'data.kwcoco.json')

    dset._build_index()
    return dset


@profile
def random_single_video_dset(image_size=(600, 600), num_frames=5,
                             num_tracks=3, tid_start=1, gid_start=1,
                             video_id=1, anchors=None, rng=None, render=False,
                             dpath=None, autobuild=True, verbose=3, aux=None,
                             multispectral=False, max_speed=0.01,
                             channels=None, multisensor=False, **kwargs):
    """
    Create the video scene layout of object positions.

    Note:
        Does not render the data unless specified.

    Args:
        image_size (Tuple[int, int]): size of the images

        num_frames (int): number of frames in this video

        num_tracks (int): number of tracks in this video

        tid_start (int, default=1): track-id start index

        gid_start (int, default=1): image-id start index

        video_id (int, default=1): video-id of this video

        anchors (ndarray | None): base anchor sizes of the object boxes we will
            generate.

        rng (RandomState): random state / seed

        render (bool | dict): if truthy, does the rendering according to
            provided params in the case of dict input.

        autobuild (bool, default=True): prebuild coco lookup indexes

        verbose (int): verbosity level

        aux (bool | List[str]): if specified generates auxiliary channels

        multispectral (bool): if specified simulates multispectral imagry
            This is similar to aux, but has no "main" file.

        max_speed (float):
            max speed of movers

        channels (str | None | ChannelSpec):
            if specified generates multispectral images with dummy channels

        multisensor (bool):
            if True, generates demodata from "multiple sensors", in
                other words, observations may have different "bands".

        **kwargs : used for old backwards compatible argument names
            gsize - alias for image_size

    TODO:
        - [ ] Need maximum allowed object overlap measure
        - [ ] Need better parameterized path generation

    Example:
        >>> from kwcoco.demo.toydata_video import *  # NOQA
        >>> anchors = np.array([ [0.3, 0.3],  [0.1, 0.1]])
        >>> dset = random_single_video_dset(render=True, num_frames=10, num_tracks=10, anchors=anchors, max_speed=0.2)
        >>> # xdoctest: +REQUIRES(--show)
        >>> # Show the tracks in a single image
        >>> import kwplot
        >>> kwplot.autompl()
        >>> annots = dset.annots()
        >>> tids = annots.lookup('track_id')
        >>> tid_to_aids = ub.group_items(annots.aids, tids)
        >>> paths = []
        >>> track_boxes = []
        >>> for tid, aids in tid_to_aids.items():
        >>>     boxes = dset.annots(aids).boxes.to_cxywh()
        >>>     path = boxes.data[:, 0:2]
        >>>     paths.append(path)
        >>>     track_boxes.append(boxes)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> ax = plt.gca()
        >>> ax.cla()
        >>> #
        >>> import kwimage
        >>> colors = kwimage.Color.distinct(len(track_boxes))
        >>> for i, boxes in enumerate(track_boxes):
        >>>     color = colors[i]
        >>>     path = boxes.data[:, 0:2]
        >>>     boxes.draw(color=color, centers={'radius': 0.01}, alpha=0.5)
        >>>     ax.plot(path.T[0], path.T[1], 'x-', color=color)

    Example:
        >>> from kwcoco.demo.toydata_video import *  # NOQA
        >>> anchors = np.array([ [0.2, 0.2],  [0.1, 0.1]])
        >>> gsize = np.array([(600, 600)])
        >>> print(anchors * gsize)
        >>> dset = random_single_video_dset(render=True, num_frames=10,
        >>>                             anchors=anchors, num_tracks=10,
        >>>                             image_size='random')
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> plt.clf()
        >>> gids = list(dset.imgs.keys())
        >>> pnums = kwplot.PlotNums(nSubplots=len(gids))
        >>> for gid in gids:
        >>>     dset.show_image(gid, pnum=pnums(), fnum=1, title=False)
        >>> pnums = kwplot.PlotNums(nSubplots=len(gids))

    Example:
        >>> from kwcoco.demo.toydata_video import *  # NOQA
        >>> dset = random_single_video_dset(num_frames=10, num_tracks=10, aux=True)
        >>> assert 'auxiliary' in dset.imgs[1]
        >>> assert dset.imgs[1]['auxiliary'][0]['channels']
        >>> assert dset.imgs[1]['auxiliary'][1]['channels']

    Example:
        >>> from kwcoco.demo.toydata_video import *  # NOQA
        >>> multispectral = True
        >>> dset = random_single_video_dset(num_frames=1, num_tracks=1, multispectral=True)
        >>> dset._check_json_serializable()
        >>> dset.dataset['images']
        >>> assert dset.imgs[1]['auxiliary'][1]['channels']
        >>> # test that we can render
        >>> render_toy_dataset(dset, rng=0, dpath=None, renderkw={})

    Example:
        >>> from kwcoco.demo.toydata_video import *  # NOQA
        >>> dset = random_single_video_dset(num_frames=4, num_tracks=1, multispectral=True, multisensor=True, image_size='random', rng=2338)
        >>> dset._check_json_serializable()
        >>> assert dset.imgs[1]['auxiliary'][1]['channels']
        >>> # Print before and after render
        >>> #print('multisensor-images = {}'.format(ub.repr2(dset.dataset['images'], nl=-2)))
        >>> #print('multisensor-images = {}'.format(ub.repr2(dset.dataset, nl=-2)))
        >>> print(ub.hash_data(dset.dataset))
        >>> # test that we can render
        >>> render_toy_dataset(dset, rng=0, dpath=None, renderkw={})
        >>> #print('multisensor-images = {}'.format(ub.repr2(dset.dataset['images'], nl=-2)))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> from kwcoco.demo.toydata_video import _draw_video_sequence  # NOQA
        >>> gids = [1, 2, 3, 4]
        >>> final = _draw_video_sequence(dset, gids)
        >>> print('dset.fpath = {!r}'.format(dset.fpath))
        >>> kwplot.imshow(final)

    Ignore:
        import xdev
        globals().update(xdev.get_func_kwargs(random_single_video_dset))

        dset = random_single_video_dset(render=True, num_frames=10,
            anchors=anchors, num_tracks=10)
        dset._tree()
    """
    import pandas as pd
    import kwcoco
    from kwarray import distributions
    rng = kwarray.ensure_rng(rng)

    if 'gsize' in kwargs:  # nocover
        if 0:
            # TODO: enable this warning
            import warnings
            warnings.warn('gsize is deprecated. Use image_size param instead',
                          DeprecationWarning)
        image_size = kwargs.pop('gsize')
    assert len(kwargs) == 0, 'unknown kwargs={}'.format(**kwargs)

    image_ids = list(range(gid_start, num_frames + gid_start))
    track_ids = list(range(tid_start, num_tracks + tid_start))

    if isinstance(image_size, str):
        if image_size == 'random':
            image_size = (
                distributions.Uniform(200, 800, rng=rng),
                distributions.Uniform(200, 800, rng=rng),
            )

    coercable_width = image_size[0]
    coercable_height = image_size[1]

    if isinstance(coercable_width, distributions.Distribution):
        video_width = int(coercable_width.sample())
        video_height = int(coercable_height.sample())
        image_width_distri = coercable_width
        image_height_distri = coercable_height
    else:
        video_width = coercable_width
        video_height = coercable_height
        image_width_distri = distributions.Constant(coercable_width)
        image_height_distri = distributions.Constant(coercable_height)

    video_dsize = (video_width, video_height)
    video_window = kwimage.Boxes([[0, 0, video_width, video_height]], 'xywh')

    dset = kwcoco.CocoDataset(autobuild=False)
    dset.add_video(
        name='toy_video_{}'.format(video_id),
        width=video_width,
        height=video_height,
        id=video_id,
    )

    if bool(multispectral) + bool(aux) + bool(channels) > 1:
        raise ValueError('can only have one of multispectral, aux, or channels')

    # backwards compat
    no_main_image = False
    if channels is None:
        if aux is True:
            channels = 'disparity,flowx|flowy'
        if multispectral:
            channels = 'B1,B8,B8a,B10,B11'
            no_main_image = True
    else:
        no_main_image = True

    special_fusedbands_to_scale = {
        'disparity': 1,
        'flowx|flowy': 1,
        'B1': 1,
        'B8': 1 / 6,
        'B8a': 1 / 3,
        'B10': 1,
        'B11': 1 / 3,
    }

    sensor_to_channels = {}
    if channels is not None:
        channels = kwcoco.ChannelSpec.coerce(channels)
        sensor_to_channels['sensor1'] = channels

    if multisensor:
        assert channels is not None
        # todo: give users a way to specify (1) how many sensors, and (2)
        # what the channels for each sensor should be.
        sensor_to_channels['sensor2'] = kwcoco.ChannelSpec.coerce('r|g|b,disparity,gauss,B8|B11')
        sensor_to_channels['sensor3'] = kwcoco.ChannelSpec.coerce('r|g|b,flowx|flowy,distri,B10|B11')
        sensor_to_channels['sensor4'] = kwcoco.ChannelSpec.coerce('B11,X.2,Y:2:6')

    sensors = sorted(sensor_to_channels.keys())

    for frame_idx, gid in enumerate(image_ids):
        if verbose > 2:
            print('generate gid = {!r}'.format(gid))
        image_height = int(image_height_distri.sample())
        image_width = int(image_width_distri.sample())

        warp_vid_from_img = kwimage.Affine.coerce(
            scale=(video_width / image_width, video_height / image_height)
        )
        warp_img_from_vid = warp_vid_from_img.inv()

        img = {
            'id': gid,
            'file_name': '<todo-generate-{}-{}>'.format(video_id, frame_idx),
            'width': image_width,
            'height': image_height,
            'warp_img_to_vid': warp_vid_from_img.concise(),
            'frame_index': frame_idx,
            'video_id': video_id,
        }
        if no_main_image:
            # TODO: can we do better here?
            img['name'] = 'generated-{}-{}'.format(video_id, frame_idx)
            img['file_name'] = None

        if sensors:
            frame_sensor_idx = rng.randint(0, len(sensors))
            frame_sensor = sensors[frame_sensor_idx]
            frame_channels = sensor_to_channels[frame_sensor]

            img['auxiliary'] = []
            for stream in frame_channels.streams():
                scale = special_fusedbands_to_scale.get(stream.spec, None)
                if scale is not None:
                    warp_img_from_aux = kwimage.Affine.scale(scale=1 / scale)
                else:
                    # about = (image_width / 2, image_height / 2)
                    # params = kwimage.Affine.random_params(rng=rng)
                    # params['about'] = about
                    # warp_img_from_aux = kwimage.Affine.coerce(params)
                    warp_img_from_aux = kwimage.Affine.random(rng=rng)

                warp_aux_from_img = warp_img_from_aux.inv()
                warp_aux_from_vid = warp_aux_from_img @ warp_img_from_vid
                aux_window = video_window.warp(warp_aux_from_vid).quantize()

                aux_width = aux_window.width.item()
                aux_height = aux_window.height.item()

                auxitem = {
                    'file_name': '<todo-generate-{}-{}>'.format(video_id, frame_idx),
                    'channels': stream.spec,
                    'width': aux_width,
                    'height': aux_height,
                    'warp_aux_to_img': warp_img_from_aux.concise(),
                }
                # hack for dtype
                if stream.spec.startswith('B'):
                    auxitem['dtype'] = 'uint16'
                else:
                    auxitem['dtype'] = 'uint8'
                img['auxiliary'].append(auxitem)

        dset.add_image(**img)

    classes = ['star', 'superstar', 'eff']
    for catname in classes:
        dset.ensure_category(name=catname)

    catpats = CategoryPatterns.coerce(classes, rng=rng)

    if True:
        # TODO: add ensure keypoint category to dset
        # Add newstyle keypoint categories
        kpname_to_id = {}
        dset.dataset['keypoint_categories'] = []
        for kpcat in catpats.keypoint_categories:
            dset.dataset['keypoint_categories'].append(kpcat)
            kpname_to_id[kpcat['name']] = kpcat['id']

    # Generate paths in a way that they are dependant on each other
    paths = random_multi_object_path(
        num_frames=num_frames,
        num_objects=num_tracks, rng=rng,
        max_speed=max_speed)

    def warp_within_bounds(self, x_min, y_min, x_max, y_max):
        """
        Translate / scale the boxes to fit in the bounds

        FIXME: do something reasonable

        Example:
            >>> from kwimage.structs.boxes import *  # NOQA
            >>> self = Boxes.random(10).scale(1).translate(-10)
            >>> x_min, y_min, x_max, y_max = 10, 10, 20, 20
            >>> x_min, y_min, x_max, y_max = 0, 0, 20, 20
            >>> print('self = {!r}'.format(self))
            >>> scaled = warp_within_bounds(self, x_min, y_min, x_max, y_max)
            >>> print('scaled = {!r}'.format(scaled))
        """
        tlbr = self.to_tlbr()
        tl_x, tl_y, br_x, br_y = tlbr.components
        tl_xy_min = np.c_[tl_x, tl_y].min(axis=0)
        br_xy_max = np.c_[br_x, br_y].max(axis=0)
        tl_xy_lb = np.array([x_min, y_min])
        br_xy_ub = np.array([x_max, y_max])

        size_ub = br_xy_ub - tl_xy_lb
        size_max = br_xy_max - tl_xy_min

        tl_xy_over = np.minimum(tl_xy_lb - tl_xy_min, 0)
        # tl_xy_over = -tl_xy_min
        # Now at the minimum coord
        tmp = tlbr.translate(tl_xy_over)
        _tl_x, _tl_y, _br_x, _br_y = tmp.components
        tmp_tl_xy_min = np.c_[_tl_x, _tl_y].min(axis=0)
        # tmp_br_xy_max = np.c_[_br_x, _br_y].max(axis=0)

        tmp.translate(-tmp_tl_xy_min)
        sf = np.minimum(size_ub / size_max, 1)
        out = tmp.scale(sf).translate(tmp_tl_xy_min)
        return out

    if verbose > 2:
        print('generate tracks')

    for tid, path in zip(track_ids, paths):
        if anchors is None:
            anchors_ = anchors
        else:
            anchors_ = np.array([anchors[rng.randint(0, len(anchors))]])

        # Box scale
        video_boxes = kwimage.Boxes.random(
            num=num_frames, scale=1.0, format='cxywh', rng=rng,
            anchors=anchors_)

        # Smooth out varying box sizes
        alpha = rng.rand() * 0.1
        wh = pd.DataFrame(video_boxes.data[:, 2:4], columns=['w', 'h'])

        ar = wh['w'] / wh['h']
        min_ar = 0.25
        max_ar = 1 / min_ar

        wh['w'][ar < min_ar] = wh['h'] * 0.25
        wh['h'][ar > max_ar] = wh['w'] * 0.25

        box_dims = wh.ewm(alpha=alpha, adjust=False).mean()

        video_boxes.data[:, 0:2] = path
        video_boxes.data[:, 2:4] = box_dims.values

        video_boxes = video_boxes.scale(video_dsize, about='origin')
        video_boxes = video_boxes.scale(0.9, about='center')

        # oob_pad = -20  # allow some out of bounds
        # oob_pad = 20  # allow some out of bounds
        # video_boxes = video_boxes.to_tlbr()
        # TODO: need better path distributions
        # video_boxes = warp_within_bounds(video_boxes, 0 - oob_pad, 0 - oob_pad, image_size[0] + oob_pad, image_size[1] + oob_pad)
        video_boxes = video_boxes.to_xywh()

        video_boxes.data = video_boxes.data.round(1)
        cidx = rng.randint(0, len(classes))
        video_dets = kwimage.Detections(
            boxes=video_boxes,
            class_idxs=np.array([cidx] * len(video_boxes)),
            classes=classes,
        )

        WITH_KPTS_SSEG = True
        if WITH_KPTS_SSEG:
            kpts = []
            ssegs = []
            ddims = video_boxes.data[:, 2:4].astype(int)[:, ::-1]
            offsets = video_boxes.data[:, 0:2].astype(int)
            cnames = [classes[cidx] for cidx in video_dets.class_idxs]
            for dims, xy_offset, cname in zip(ddims, offsets, cnames):
                info = catpats._todo_refactor_geometric_info(cname, xy_offset, dims)
                kpts.append(info['kpts'])

                if False and rng.rand() > 0.5:
                    # sseg dropout
                    ssegs.append(None)
                else:
                    ssegs.append(info['segmentation'])

            video_dets.data['keypoints'] = kwimage.PointsList(kpts)
            video_dets.data['segmentations'] = kwimage.SegmentationList(ssegs)

        start_frame = 0
        for frame_index, video_det in enumerate(video_dets, start=start_frame):
            frame_img = dset.dataset['images'][frame_index]
            warp_vid_from_img = kwimage.Affine.coerce(frame_img['warp_img_to_vid'])
            warp_img_from_vid = warp_vid_from_img.inv()
            image_det = video_det.warp(warp_img_from_vid)
            image_ann = list(image_det.to_coco(dset=dset, style='new'))[0]
            image_ann['track_id'] = tid
            image_ann['image_id'] = frame_img['id']
            dset.add_annotation(**image_ann)

    HACK_FIX_COCO_KPTS_FORMAT = True
    if HACK_FIX_COCO_KPTS_FORMAT:
        # Hack to fix coco formatting from Detections.to_coco
        kpt_cats = dset.keypoint_categories()
        for ann in dset.dataset['annotations']:
            for kpt in ann.get('keypoints', []):
                if 'keypoint_category_id' not in kpt:
                    kp_catname = kpt.pop('keypoint_category')
                    kpt['keypoint_category_id'] = kpt_cats.node_to_id[kp_catname]

    # The dataset has been prepared, now we just render it and we have
    # a nice video dataset.
    renderkw = {
        'dpath': dpath,
    }
    if isinstance(render, dict):
        renderkw.update(render)
    else:
        if not render:
            renderkw = None
    if renderkw is not None:
        if verbose > 2:
            print('rendering')
        render_toy_dataset(dset, rng=rng, dpath=dpath, renderkw=renderkw,
                           verbose=verbose)
    if autobuild:
        dset._build_index()
    return dset


def _draw_video_sequence(dset, gids):
    """
    Helper to draw a multi-sensor sequence

    Ignore:
        from kwcoco.demo.toydata_video import _draw_video_sequence  # NOQA
        gids = [1, 2, 3]
        final = _draw_video_sequence(dset, gids)
        import kwplot
        kwplot.autompl()
        kwplot.imshow(final)
    """
    horizontal_stack = []
    max_width = 256
    images = dset.images(gids)
    for coco_img in images.coco_images:
        chan_names = coco_img.channels.fuse()
        chan_hwc = coco_img.delay(space='video').finalize()
        chan_chw = chan_hwc.transpose(2, 0, 1)
        cells = []
        for raw_data, chan_name in zip(chan_chw, chan_names):
            # norm_data = kwimage.normalize_intensity(raw_data.astype(np.float32)).clip(0, 1)
            norm_data = kwimage.normalize(raw_data.astype(np.float32)).clip(0, 1)
            cells.append({
                'norm_data': norm_data,
                'raw_data': raw_data,
                'text': chan_name,
            })
        vertical_stack = []
        header_dims = {'width': max_width}
        header_part = kwimage.draw_header_text(
            image=header_dims, fit=False,
            text='t={frame_index} gid={id}'.format(**coco_img.img),
            color='salmon')
        vertical_stack.append(header_part)
        for cell in cells:
            norm_data = cell['norm_data']
            cell_canvas = kwimage.imresize(norm_data, dsize=(max_width, None))
            cell_canvas = cell_canvas.clip(0, 1)
            cell_canvas = kwimage.atleast_3channels(cell_canvas)
            cell_canvas = kwimage.draw_text_on_image(
                cell_canvas, cell['text'], (1, 1), valign='top',
                color='white', border=3)
            vertical_stack.append(cell_canvas)

        vertical_stack = [kwimage.ensure_uint255(d) for d in vertical_stack]
        column_img = kwimage.stack_images(vertical_stack, axis=0)

        horizontal_stack.append(column_img)
    final = kwimage.stack_images(horizontal_stack, axis=1)
    return final


@profile
def render_toy_dataset(dset, rng, dpath=None, renderkw=None, verbose=0):
    """
    Create toydata_video renderings for a preconstructed coco dataset.

    Args:
        dset (CocoDataset):
            A dataset that contains special "renderable" annotations. (e.g.
            the demo shapes). Each image can contain special fields that
            influence how an image will be rendered.

            Currently this process is simple, it just creates a noisy image
            with the shapes superimposed over where they should exist as
            indicated by the annotations. In the future this may become more
            sophisticated.

            Each item in `dset.dataset['images']` will be modified to add
            the "file_name" field indicating where the rendered data is writen.

        rng (int | None | RandomState): random state

        dpath (str):
            The location to write the images to. If unspecified, it is written
            to the rendered folder inside the kwcoco cache directory.

        renderkw (dict): See :func:`render_toy_image` for details.
            Also takes imwrite keywords args only handled in this function.
            TODO better docs.

    Example:
        >>> from kwcoco.demo.toydata_video import *  # NOQA
        >>> import kwarray
        >>> rng = None
        >>> rng = kwarray.ensure_rng(rng)
        >>> num_tracks = 3
        >>> dset = random_video_dset(rng=rng, num_videos=3, num_frames=5,
        >>>                          num_tracks=num_tracks, image_size=(128, 128))
        >>> dset = render_toy_dataset(dset, rng)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> plt.clf()
        >>> gids = list(dset.imgs.keys())
        >>> pnums = kwplot.PlotNums(nSubplots=len(gids), nRows=num_tracks)
        >>> for gid in gids:
        >>>     dset.show_image(gid, pnum=pnums(), fnum=1, title=False)
        >>> pnums = kwplot.PlotNums(nSubplots=len(gids))
    """
    import kwcoco
    rng = kwarray.ensure_rng(rng)
    dset._build_index()

    if 0:
        dset._ensure_json_serializable()
    hashid = dset._build_hashid()[0:24]

    if dpath is None:
        dset_name = 'rendered_{}'.format(hashid)
        bundle_dpath = ub.ensure_app_cache_dir('kwcoco', 'rendered', dset_name)
        dset.fpath = join(bundle_dpath, 'data.kwcoco.json')
        bundle_dpath = dset.bundle_dpath
    else:
        bundle_dpath = dpath

    img_dpath = ub.ensuredir((bundle_dpath, '_assets/images'))

    imwrite_kw = {}
    imwrite_ops = {'compress', 'blocksize', 'interleave', 'options'}
    if renderkw is None:
        renderkw = {}
    imwrite_kw = ub.dict_isect(renderkw, imwrite_ops)
    if imwrite_kw:
        # imwrite_kw['backend'] = 'gdal'
        # imwrite_kw['space'] = None
        # imwrite kw requries gdal
        from osgeo import gdal  # NOQA

    for gid in ub.ProgIter(dset.imgs.keys(), desc='render gid', verbose=verbose > 2):
        # Render data inside the image
        img = render_toy_image(dset, gid, rng=rng, renderkw=renderkw)

        # Extract the data from memory and write it to disk
        fname = 'img_{:05d}.png'.format(gid)
        imdata = img.pop('imdata', None)
        if imdata is not None:
            img_fpath = join(img_dpath, fname)
            img.update({
                'file_name': img_fpath,
                'channels': 'r|g|b',
            })
            kwimage.imwrite(img_fpath, imdata)

        auxiliaries = img.pop('auxiliary', None)
        if auxiliaries is not None:
            for auxdict in auxiliaries:
                chan_part = kwcoco.ChannelSpec.coerce(auxdict['channels']).as_path()
                aux_dpath = ub.ensuredir(
                    (bundle_dpath, '_assets', 'aux', 'aux_' + chan_part))
                aux_fpath = ub.augpath(join(aux_dpath, fname), ext='.tif')
                ub.ensuredir(aux_dpath)
                auxdict['file_name'] = aux_fpath
                auxdata = auxdict.pop('imdata', None)
                try:
                    from osgeo import gdal  # NOQA
                    kwimage.imwrite(
                        aux_fpath, auxdata, backend='gdal', space=None,
                        **imwrite_kw)
                except Exception:
                    kwimage.imwrite(
                        aux_fpath, auxdata, space=None, **imwrite_kw)
            img['auxiliary'] = auxiliaries

    dset._build_index()
    return dset


@profile
def render_toy_image(dset, gid, rng=None, renderkw=None):
    """
    Modifies dataset inplace, rendering synthetic annotations.

    This does not write to disk. Instead this writes to placeholder values in
    the image dictionary.

    Args:
        dset (CocoDataset): coco dataset with renderable anotations / images
        gid (int): image to render
        rng (int | None | RandomState): random state
        renderkw (dict): rendering config
             gray (boo): gray or color images
             fg_scale (float): foreground noisyness (gauss std)
             bg_scale (float): background noisyness (gauss std)
             fg_intensity (float): foreground brightness (gauss mean)
             bg_intensity (float): background brightness (gauss mean)
             newstyle (bool): use new kwcoco datastructure formats
             with_kpts (bool): include keypoint info
             with_sseg (bool): include segmentation info

    Returns:
        Dict: the inplace-modified image dictionary

    Example:
        >>> from kwcoco.demo.toydata_video import *  # NOQA
        >>> image_size=(600, 600)
        >>> num_frames=5
        >>> verbose=3
        >>> rng = None
        >>> import kwarray
        >>> rng = kwarray.ensure_rng(rng)
        >>> aux = 'mx'
        >>> dset = random_single_video_dset(
        >>>     image_size=image_size, num_frames=num_frames, verbose=verbose, aux=aux, rng=rng)
        >>> print('dset.dataset = {}'.format(ub.repr2(dset.dataset, nl=2)))
        >>> gid = 1
        >>> renderkw = {}
        >>> render_toy_image(dset, gid, rng, renderkw=renderkw)
        >>> img = dset.imgs[gid]
        >>> canvas = img['imdata']
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas, doclf=True, pnum=(1, 2, 1))
        >>> dets = dset.annots(gid=gid).detections
        >>> dets.draw()

        >>> auxdata = img['auxiliary'][0]['imdata']
        >>> aux_canvas = false_color(auxdata)
        >>> kwplot.imshow(aux_canvas, pnum=(1, 2, 2))
        >>> _ = dets.draw()

        >>> # xdoctest: +REQUIRES(--show)
        >>> img, anns = demodata_toy_img(image_size=(172, 172), rng=None, aux=True)
        >>> print('anns = {}'.format(ub.repr2(anns, nl=1)))
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img['imdata'], pnum=(1, 2, 1), fnum=1)
        >>> auxdata = img['auxiliary'][0]['imdata']
        >>> kwplot.imshow(auxdata, pnum=(1, 2, 2), fnum=1)
        >>> kwplot.show_if_requested()

    Example:
        >>> from kwcoco.demo.toydata_video import *  # NOQA
        >>> multispectral = True
        >>> dset = random_single_video_dset(num_frames=1, num_tracks=1, multispectral=True)
        >>> gid = 1
        >>> dset.imgs[gid]
        >>> rng = kwarray.ensure_rng(0)
        >>> renderkw = {'with_sseg': True}
        >>> img = render_toy_image(dset, gid, rng=rng, renderkw=renderkw)
    """
    rng = kwarray.ensure_rng(rng)

    if renderkw is None:
        renderkw = {}

    gray = renderkw.get('gray', 1)
    fg_scale = renderkw.get('fg_scale', 0.5)
    bg_scale = renderkw.get('bg_scale', 0.8)
    bg_intensity = renderkw.get('bg_intensity', 0.1)
    fg_intensity = renderkw.get('fg_intensity', 0.9)
    newstyle = renderkw.get('newstyle', True)
    with_kpts = renderkw.get('with_kpts', False)
    with_sseg = renderkw.get('with_sseg', False)

    categories = list(dset.name_to_cat.keys())
    catpats = CategoryPatterns.coerce(categories, fg_scale=fg_scale,
                                      fg_intensity=fg_intensity, rng=rng)

    if with_kpts and newstyle:
        # TODO: add ensure keypoint category to dset
        # Add newstyle keypoint categories
        # kpname_to_id = {}
        dset.dataset['keypoint_categories'] = []
        for kpcat in catpats.keypoint_categories:
            dset.dataset['keypoint_categories'].append(kpcat)
            # kpname_to_id[kpcat['name']] = kpcat['id']

    img = dset.imgs[gid]
    gw, gh = img['width'], img['height']
    dims = (gh, gw)
    annots = dset.annots(gid=gid)

    imdata, chan_to_auxinfo = render_background(img, rng, gray, bg_intensity, bg_scale)
    imdata, chan_to_auxinfo = render_foreground(imdata, chan_to_auxinfo, dset,
                                                annots, catpats, with_sseg,
                                                with_kpts, dims, newstyle,
                                                gray, rng)

    if imdata is not None:
        imdata = (imdata * 255).astype(np.uint8)
        imdata = kwimage.atleast_3channels(imdata)
        main_channels = 'gray' if gray else 'r|g|b'
        img.update({
            'imdata': imdata,
            'channels': main_channels,
        })

    for auxinfo in img.get('auxiliary', []):
        # Postprocess the auxiliary data so it looks interesting
        # It would be really cool if we could do this based on what
        # the simulated channel was.
        import kwcoco
        chankey = auxinfo['channels']
        auxdata = chan_to_auxinfo[chankey]['imdata']
        auxchan = kwcoco.FusedChannelSpec.coerce(chankey)
        for chan_idx, chan_name in enumerate(auxchan.as_list()):
            if chan_name == 'flowx':
                auxdata[..., chan_idx] = np.gradient(auxdata[..., chan_idx], axis=0)
            elif chan_name == 'flowy':
                auxdata[..., chan_idx] = np.gradient(auxdata[..., chan_idx], axis=1)
            elif chan_name.startswith('B') or 1:
                mask = rng.rand(*auxdata[..., chan_idx].shape[0:2]) > 0.5
                auxdata[..., chan_idx] = kwimage.fourier_mask(auxdata[..., chan_idx], mask)[..., 0]
        auxdata = kwarray.normalize(auxdata)
        auxdata = auxdata.clip(0, 1)
        _dtype = auxinfo.pop('dtype', 'uint8').lower()
        if _dtype == 'uint8':
            auxdata = (auxdata * int((2 ** 8) - 1)).astype(np.uint8)
        elif _dtype == 'uint16':
            auxdata = (auxdata * int((2 ** 16) - 1)).astype(np.uint16)
        else:
            raise KeyError(_dtype)
        auxinfo['imdata'] = auxdata

    return img


@profile
def render_foreground(imdata, chan_to_auxinfo, dset, annots, catpats,
                      with_sseg, with_kpts, dims, newstyle, gray, rng):
    """
    Renders demo annoations on top of a demo background
    """
    boxes = annots.boxes

    tlbr_boxes = boxes.to_tlbr().quantize().data.astype(int)

    # Render coco-style annotation dictionaries
    for ann, tlbr in zip(annots.objs, tlbr_boxes):
        catname = dset._resolve_to_cat(ann['category_id'])['name']
        tl_x, tl_y, br_x, br_y = tlbr

        # hack
        if tl_x == br_x:
            tl_x = tl_x - 1
            br_x = br_x + 1
        if tl_y == br_y:
            tl_y = tl_y - 1
            br_y = br_y + 1

        chip_index = tuple([slice(tl_y, br_y), slice(tl_x, br_x)])
        if imdata is None:
            chip = None
        else:
            data_slice, padding = kwarray.embed_slice(chip_index, imdata.shape)
            # TODO: could have a kwarray function to expose this inverse slice
            # functionality. Also having a top-level call to apply an embedded
            # slice would be good
            chip = kwarray.padded_slice(imdata, chip_index)
            inverse_slice = (
                slice(padding[0][0], chip.shape[0] - padding[0][1]),
                slice(padding[1][0], chip.shape[1] - padding[1][1]),
            )

        size = (br_x - tl_x, br_y - tl_y)
        xy_offset = (tl_x, tl_y)

        if chip is None or chip.size:
            # todo: no need to make kpts / sseg if not requested
            info = catpats.render_category(
                catname, chip, xy_offset, dims, newstyle=newstyle,
                size=size)

            if imdata is not None:
                fgdata = info['data']
                if gray:
                    fgdata = fgdata.mean(axis=2, keepdims=True)
                imdata[data_slice] = fgdata[inverse_slice]

            if with_sseg:
                ann['segmentation'] = info['segmentation']
            if with_kpts:
                ann['keypoints'] = info['keypoints']

            if chan_to_auxinfo is not None:
                # chan_to_auxinfo.keys()
                coco_sseg = ann.get('segmentation', None)
                if coco_sseg:
                    seg = kwimage.Segmentation.coerce(coco_sseg)
                    seg = seg.to_multi_polygon()
                    for chankey, auxinfo in chan_to_auxinfo.items():
                        val = rng.uniform(0.2, 1.0)
                        # transform annotation into aux space
                        warp_aux_to_img = auxinfo.get('warp_aux_to_img', None)
                        if warp_aux_to_img is not None:
                            warp_aux_from_img = kwimage.Affine.coerce(warp_aux_to_img).inv().matrix
                            seg_ = seg.warp(warp_aux_from_img)
                        else:
                            seg_ = seg
                        c = kwimage.num_channels(auxinfo['imdata'])
                        if c < 4:
                            # hack work around bug in kwimage, where only first
                            # channel was filled
                            val = (val,) * c
                        auxinfo['imdata'] = seg_.fill(auxinfo['imdata'], value=val)
    return imdata, chan_to_auxinfo


@profile
def render_background(img, rng, gray, bg_intensity, bg_scale):
    # This is 2x as fast for image_size=(300,300)
    gw, gh = img['width'], img['height']
    if img.get('file_name', None) is None:
        imdata = None
    else:
        if gray:
            gshape = (gh, gw, 1)
            imdata = kwarray.standard_normal(gshape, mean=bg_intensity, std=bg_scale,
                                               rng=rng, dtype=np.float32)
        else:
            gshape = (gh, gw, 3)
            # imdata = kwarray.standard_normal(gshape, mean=bg_intensity, std=bg_scale,
            #                                    rng=rng, dtype=np.float32)
            # hack because 3 channels is slower
            imdata = kwarray.uniform(0, 1, gshape, rng=rng, dtype=np.float32)
        np.clip(imdata, 0, 1, out=imdata)

    chan_to_auxinfo = {}
    for auxinfo in img.get('auxiliary', []):
        import kwcoco
        chankey = auxinfo['channels']
        aux_bands = kwcoco.ChannelSpec.coerce(chankey).numel()
        aux_width = auxinfo.get('width', gw)
        aux_height = auxinfo.get('height', gh)
        auxshape = (aux_height, aux_width, aux_bands)
        if chankey == 'gauss':
            auxdata = np.stack([
                kwimage.gaussian_patch(auxshape[0:2])
                for b in range(aux_bands)], axis=2)
            auxdata = kwarray.normalize(auxdata) * rng.rand()
        elif chankey == 'distri':
            # Random distribution currently broken, fix later
            # random_distri = kwarray.distributions.Distribution.random(rng=rng)
            # auxdata = random_distri.sample(*auxshape)
            scale = rng.randint(0, 100) * rng.rand()
            auxdata = rng.exponential(scale, size=auxshape)
            auxdata = kwarray.normalize(auxdata)
        else:
            auxdata = kwarray.uniform(0, 0.4, auxshape, rng=rng, dtype=np.float32)

        # if True or True or True:
        #     auxdata[:] = 0
        # auxdata = (auxdata - auxdata.min())
        # auxdata = (auxdata / max(1e-8, auxdata.min()))
        auxinfo['imdata'] = auxdata
        chan_to_auxinfo[chankey] = auxinfo

    return imdata, chan_to_auxinfo


def false_color(twochan):
    """
    TODO: the function ensure_false_color will eventually be ported to kwimage
    use that instead.
    """
    if 0:
        import sklearn
        model = sklearn.decomposition.PCA(3)
        X = twochan.reshape(-1, 2)
        model.fit(X)
    else:
        import sklearn
        ndim = twochan.ndim
        dims = twochan.shape[0:2]
        if ndim == 2:
            in_channels = 1
        else:
            in_channels = twochan.shape[2]

        if in_channels > 1:
            model = sklearn.decomposition.PCA(1)
            X = twochan.reshape(-1, in_channels)
            X_ = model.fit_transform(X)
            gray = X_.reshape(dims)
            viz = kwimage.make_heatmask(gray, with_alpha=1)[:, :, 0:3]
        else:
            gray = twochan.reshape(dims)
            viz = gray
        return viz


@profile
def random_multi_object_path(num_objects, num_frames, rng=None, max_speed=0.01):
    """

    Ignore:
        >>> from kwcoco.demo.toydata_video import *  # NOQA
        >>> #
        >>> import kwarray
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> #
        >>> num_objects = 5
        >>> num_frames = 100
        >>> rng = kwarray.ensure_rng(0)
        >>> #
        >>> from kwcoco.demo.toydata_video import *  # NOQA
        >>> paths = random_multi_object_path(num_objects, num_frames, rng, max_speed=0.1)
        >>> #
        >>> from mpl_toolkits.mplot3d import Axes3D  # NOQA
        >>> ax = plt.gca(projection='3d')
        >>> ax.cla()
        >>> #
        >>> for path in paths:
        >>>     time = np.arange(len(path))
        >>>     ax.plot(time, path.T[0] * 1, path.T[1] * 1, 'o-')
        >>> ax.set_xlim(0, num_frames)
        >>> ax.set_ylim(-.01, 1.01)
        >>> ax.set_zlim(-.01, 1.01)

    Ignore:
        >>> from kwcoco.demo.toydata_video import *  # NOQA
        >>> num_objects = 1
        >>> num_frames = 2
        >>> rng = kwarray.ensure_rng(342)
        >>> paths = random_multi_object_path(num_objects, num_frames, rng, max_speed=0.1)
        >>> print('paths = {!r}'.format(paths))
        >>> print(ub.hash_data(paths))
    """
    import kwarray
    rng = kwarray.ensure_rng(rng)

    USE_BOIDS = 1

    if USE_BOIDS:
        from kwcoco.demo.boids import Boids
        config = {
            'perception_thresh': 0.2,
            'max_speed': max_speed,
            'max_force': 0.001,
            'damping': 0.99,
        }
        boids = Boids(num_objects, rng=rng, **config).initialize()
        paths = boids.paths(num_frames)
        return paths
    else:
        import torch
        from torch.nn import functional as F

        max_speed = rng.rand(num_objects, 1) * 0.01
        max_speed = np.concatenate([max_speed, max_speed], axis=1)

        max_speed = torch.from_numpy(max_speed)

        # TODO: can we do better?
        torch.optim.SGD
        class Positions(torch.nn.Module):
            def __init__(model):
                super().__init__()
                _pos = torch.from_numpy(rng.rand(num_objects, 2))
                model.pos = torch.nn.Parameter(_pos)

            def forward(model, noise):
                loss_parts = {}

                # Push objects away from each other
                utriu_dists = torch.nn.functional.pdist(model.pos)
                respulsive = (1 / (utriu_dists ** 2)).sum() / num_objects
                loss_parts['repulsive'] = 0.01 * respulsive.sum()

                # Push objects in random directions
                loss_parts['random'] = (model.pos * noise).sum()

                # Push objects away from the boundary
                margin = 0
                x = model.pos
                y = F.softplus((x - 0.5) + margin)
                y = torch.max(F.softplus(-(x - 0.5) + margin), y)
                loss_parts['boundary'] = y.sum() * 2

                return sum(loss_parts.values())

        model = Positions()
        params = list(model.parameters())
        optim = torch.optim.SGD(params, lr=1.00, momentum=0.9)

        positions = []
        for i in range(num_frames):
            optim.zero_grad()
            noise = torch.from_numpy(rng.rand(2))
            loss = model.forward(noise)
            loss.backward()

            if max_speed is not None:
                # Enforce a per-object speed limit
                model.pos.grad.data[:] = torch.min(model.pos.grad, max_speed)
                model.pos.grad.data[:] = torch.max(model.pos.grad, -max_speed)

            optim.step()

            # Enforce boundry conditions
            model.pos.data.clamp_(0, 1)
            positions.append(model.pos.data.numpy().copy())

    paths = np.concatenate([p[:, None] for p in positions], axis=1)
    return paths
    # path = np.array(positions) % 1


def random_path(num, degree=1, dimension=2, rng=None, mode='boid'):
    """
    Create a random path using a somem ethod curve.

    Args:
        num (int): number of points in the path
        degree (int, default=1): degree of curvieness of the path
        dimension (int, default=2): number of spatial dimensions
        mode (str): can be boid, walk, or bezier
        rng (RandomState, default=None): seed

    References:
        https://github.com/dhermes/bezier

    Example:
        >>> from kwcoco.demo.toydata_video import *  # NOQA
        >>> num = 10
        >>> dimension = 2
        >>> degree = 3
        >>> rng = None
        >>> path = random_path(num, degree, dimension, rng, mode='boid')
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> kwplot.multi_plot(xdata=path[:, 0], ydata=path[:, 1], fnum=1, doclf=1, xlim=(0, 1), ylim=(0, 1))
        >>> kwplot.show_if_requested()

    Example:
        >>> # xdoctest: +REQUIRES(--3d)
        >>> # xdoctest: +REQUIRES(module:bezier)
        >>> import kwarray
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> #
        >>> num= num_frames = 100
        >>> rng = kwarray.ensure_rng(0)
        >>> #
        >>> from kwcoco.demo.toydata_video import *  # NOQA
        >>> paths = []
        >>> paths.append(random_path(num, degree=3, dimension=3, mode='bezier'))
        >>> paths.append(random_path(num, degree=2, dimension=3, mode='bezier'))
        >>> paths.append(random_path(num, degree=4, dimension=3, mode='bezier'))
        >>> #
        >>> from mpl_toolkits.mplot3d import Axes3D  # NOQA
        >>> ax = plt.gca(projection='3d')
        >>> ax.cla()
        >>> #
        >>> for path in paths:
        >>>     time = np.arange(len(path))
        >>>     ax.plot(time, path.T[0] * 1, path.T[1] * 1, 'o-')
        >>> ax.set_xlim(0, num_frames)
        >>> ax.set_ylim(-.01, 1.01)
        >>> ax.set_zlim(-.01, 1.01)
        >>> ax.set_xlabel('x')
        >>> ax.set_ylabel('y')
        >>> ax.set_zlabel('z')
    """
    rng = kwarray.ensure_rng(rng)

    if mode == 'boid':
        from kwcoco.demo.boids import Boids
        boids = Boids(1, rng=rng).initialize()
        path = boids.paths(num)[0]
    elif mode == 'walk':
        # TODO: can we do better?
        import torch
        torch.optim.SGD
        class Position(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pos = torch.nn.Parameter(torch.from_numpy(rng.rand(2)))

            def forward(self, noise):
                return (self.pos * noise).sum()

        pos = Position()
        params = list(pos.parameters())
        optim = torch.optim.SGD(params, lr=0.01, momentum=0.9)

        positions = []
        for i in range(num):
            optim.zero_grad()
            noise = torch.from_numpy(rng.rand(2))
            loss = pos.forward(noise)
            loss.backward()
            optim.step()
            positions.append(pos.pos.data.numpy().copy())
        path = np.array(positions) % 1

    elif mode == 'bezier':
        import bezier
        # Create random bezier control points
        nodes_f = rng.rand(degree + 1, dimension).T  # F-contiguous
        curve = bezier.Curve(nodes_f, degree=degree)
        if 0:
            # TODO: https://stackoverflow.com/questions/18244305/how-to-redistribute-points-evenly-over-a-curve
            t = int(np.log2(num) + 1)

            def recsub(c, d):
                if d <= 0:
                    yield c
                else:
                    a, b = c.subdivide()
                    yield from recsub(a, d - 1)
                    yield from recsub(b, d - 1)
            c = curve
            subcurves = list(recsub(c, d=t))
            path_f = np.array([c.evaluate(0.0)[:, 0] for c in subcurves][0:num]).T
        else:
            # Evaluate path points
            s_vals = np.linspace(0, 1, num)
            # s_vals = np.linspace(*sorted(rng.rand(2)), num)
            path_f = curve.evaluate_multi(s_vals)
        path = path_f.T  # C-contiguous
    else:
        raise KeyError(mode)
    return path
