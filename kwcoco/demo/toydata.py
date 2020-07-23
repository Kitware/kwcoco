# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from os.path import join
import six
import glob
import numpy as np
import ubelt as ub
import kwarray
import kwimage
import skimage
import skimage.morphology  # NOQA
from kwcoco.toypatterns import CategoryPatterns


def demodata_toy_img(anchors=None, gsize=(104, 104), categories=None,
                     n_annots=(0, 50), fg_scale=0.5, bg_scale=0.8,
                     bg_intensity=0.1, fg_intensity=0.9,
                     gray=True, centerobj=None, exact=False,
                     newstyle=True, rng=None, aux=None):
    r"""
    Generate a single image with non-overlapping toy objects of available
    categories.

    Args:
        anchors (ndarray): Nx2 base width / height of boxes

        gsize (Tuple[int, int]): width / height of the image

        categories (List[str]): list of category names

        n_annots (Tuple | int): controls how many annotations are in the image.
            if it is a tuple, then it is interpreted as uniform random bounds

        fg_scale (float): standard deviation of foreground intensity

        bg_scale (float): standard deviation of background intensity

        bg_intensity (float): mean of background intensity

        fg_intensity (float): mean of foreground intensity

        centerobj (bool): if 'pos', then the first annotation will be in the
            center of the image, if 'neg', then no annotations will be in the
            center.

        exact (bool): if True, ensures that exactly the number of specified
            annots are generated.

        newstyle (bool): use new-sytle mscoco format

        rng (RandomState): the random state used to seed the process

        aux: if specified builds auxillary channels

    CommandLine:
        xdoctest -m kwcoco.demo.toydata demodata_toy_img:0 --profile
        xdoctest -m kwcoco.demo.toydata demodata_toy_img:1 --show

    Example:
        >>> from kwcoco.demo.toydata import *  # NOQA
        >>> img, anns = demodata_toy_img(gsize=(32, 32), anchors=[[.3, .3]], rng=0)
        >>> img['imdata'] = '<ndarray shape={}>'.format(img['imdata'].shape)
        >>> print('img = {}'.format(ub.repr2(img)))
        >>> print('anns = {}'.format(ub.repr2(anns, nl=2, cbr=True)))
        >>> # xdoctest: +IGNORE_WANT
        img = {
            'height': 32,
            'imdata': '<ndarray shape=(32, 32, 3)>',
            'width': 32,
        }
        anns = [{'bbox': [15, 10, 9, 8],
          'category_name': 'star',
          'keypoints': [],
          'segmentation': {'counts': '[`06j0000O20N1000e8', 'size': [32, 32]},},
         {'bbox': [11, 20, 7, 7],
          'category_name': 'star',
          'keypoints': [],
          'segmentation': {'counts': 'g;1m04N0O20N102L[=', 'size': [32, 32]},},
         {'bbox': [4, 4, 8, 6],
          'category_name': 'superstar',
          'keypoints': [{'keypoint_category': 'left_eye', 'xy': [7.25, 6.8125]}, {'keypoint_category': 'right_eye', 'xy': [8.75, 6.8125]}],
          'segmentation': {'counts': 'U4210j0300O01010O00MVO0ed0', 'size': [32, 32]},},
         {'bbox': [3, 20, 6, 7],
          'category_name': 'star',
          'keypoints': [],
          'segmentation': {'counts': 'g31m04N000002L[f0', 'size': [32, 32]},},]

    Example:
        >>> # xdoctest: +REQUIRES(--show)
        >>> img, anns = demodata_toy_img(gsize=(172, 172), rng=None, aux=True)
        >>> print('anns = {}'.format(ub.repr2(anns, nl=1)))
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img['imdata'], pnum=(1, 2, 1), fnum=1)
        >>> auxdata = img['auxillary'][0]['imdata']
        >>> kwplot.imshow(auxdata, pnum=(1, 2, 2), fnum=1)
        >>> kwplot.show_if_requested()

    Ignore:
        from kwcoco.demo.toydata import *
        import xinspect
        globals().update(xinspect.get_kwargs(demodata_toy_img))

    """
    if anchors is None:
        anchors = [[.20, .20]]
    anchors = np.asarray(anchors)

    rng = kwarray.ensure_rng(rng)
    catpats = CategoryPatterns.coerce(categories, fg_scale=fg_scale,
                                      fg_intensity=fg_intensity, rng=rng)

    if n_annots is None:
        n_annots = (0, 50)

    if isinstance(n_annots, tuple):
        num = rng.randint(*n_annots)
    else:
        num = n_annots

    assert centerobj in {None, 'pos', 'neg'}
    if exact:
        raise NotImplementedError

    while True:
        boxes = kwimage.Boxes.random(
            num=num, scale=1.0, format='xywh', rng=rng, anchors=anchors)
        boxes = boxes.scale(gsize)
        bw, bh = boxes.components[2:4]
        ar = np.maximum(bw, bh) / np.minimum(bw, bh)
        flags = ((bw > 1) & (bh > 1) & (ar < 4))
        boxes = boxes[flags.ravel()]

        if centerobj != 'pos' or len(boxes):
            # Ensure we generate at least one box when centerobj is true
            # TODO: if an exact number of boxes is specified, we
            # should ensure that that number is generated.
            break

    if centerobj:
        if centerobj == 'pos':
            assert len(boxes) > 0, 'oops, need to enforce at least one'
        if len(boxes) > 0:
            # Force the first box to be in the center
            cxywh = boxes.to_cxywh()
            cxywh.data[0, 0:2] = np.array(gsize) / 2
            boxes = cxywh.to_tlbr()

    # Make sure the first box is always kept.
    box_priority = np.arange(boxes.shape[0])[::-1].astype(np.float32)
    boxes.ious(boxes)

    nms_impls = ub.oset(['cython_cpu', 'numpy'])
    nms_impls = nms_impls & kwimage.algo.available_nms_impls()
    nms_impl = nms_impls[0]

    if len(boxes) > 1:
        tlbr_data = boxes.to_tlbr().data
        keep = kwimage.non_max_supression(
            tlbr_data, scores=box_priority, thresh=0.0, impl=nms_impl)
        boxes = boxes[keep]

    if centerobj == 'neg':
        # The center of the image should be negative so remove the center box
        boxes = boxes[1:]

    boxes = boxes.scale(.8).translate(.1 * min(gsize))
    boxes.data = boxes.data.astype(np.int)

    # Hack away zero width objects
    boxes = boxes.to_xywh(copy=False)
    boxes.data[..., 2:4] = np.maximum(boxes.data[..., 2:4], 1)

    gw, gh = gsize
    dims = (gh, gw)

    # This is 2x as fast for gsize=(300,300)
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

    if aux:
        auxdata = np.zeros(gshape, dtype=np.float32)
    else:
        auxdata = None

    catnames = []

    tlbr_boxes = boxes.to_tlbr().data
    xywh_boxes = boxes.to_xywh().data.tolist()

    # Construct coco-style annotation dictionaries
    anns = []
    for tlbr, xywh in zip(tlbr_boxes, xywh_boxes):
        tl_x, tl_y, br_x, br_y = tlbr
        chip_index = tuple([slice(tl_y, br_y), slice(tl_x, br_x)])
        chip = imdata[chip_index]
        xy_offset = (tl_x, tl_y)
        info = catpats.random_category(chip, xy_offset, dims,
                                       newstyle=newstyle)
        fgdata = info['data']
        if gray:
            fgdata = fgdata.mean(axis=2, keepdims=True)

        catnames.append(info['name'])
        imdata[tl_y:br_y, tl_x:br_x, :] = fgdata
        ann = {
            'category_name': info['name'],
            'segmentation': info['segmentation'],
            'keypoints': info['keypoints'],
            'bbox': xywh,
            'area': float(xywh[2] * xywh[3]),
        }
        anns.append(ann)

        if auxdata is not None:
            seg = kwimage.Segmentation.coerce(info['segmentation'])
            seg = seg.to_multi_polygon()
            val = rng.uniform(0.2, 1.0)
            # val = 1.0
            auxdata = seg.fill(auxdata, value=val)

    if 0:
        imdata.mean(axis=2, out=imdata[:, :, 0])
        imdata[:, :, 1] = imdata[:, :, 0]
        imdata[:, :, 2] = imdata[:, :, 0]

    imdata = (imdata * 255).astype(np.uint8)
    imdata = kwimage.atleast_3channels(imdata)

    main_channels = 'rgb'
    # main_channels = 'gray' if gray else 'rgb'

    img = {
        'width': gw,
        'height': gh,
        'imdata': imdata,
        'channels': main_channels,
    }

    if auxdata is not None:
        mask = rng.rand(*auxdata.shape[0:2]) > 0.5
        auxdata = kwimage.fourier_mask(auxdata, mask)
        auxdata = (auxdata - auxdata.min())
        auxdata = (auxdata / max(1e-8, auxdata.max()))
        auxdata = auxdata.clip(0, 1)
        # Hack aux data is always disparity for now
        img['auxillary'] = [{
            'imdata': auxdata,
            'channels': 'disparity',
        }]

    return img, anns


def demodata_toy_dset(gsize=(600, 600), n_imgs=5, verbose=3, rng=0,
                      newstyle=True, dpath=None, aux=None, cache=True):
    """
    Create a toy detection problem

    Args:
        gsize (Tuple): size of the images
        n_img (int): number of images to generate
        rng (int | RandomState): random number generator or seed
        newstyle (bool, default=True): create newstyle mscoco data
        dpath (str): path to the output image directory, defaults to using
            kwcoco cache dir

    Returns:
        dict: dataset in mscoco format

    CommandLine:
        xdoctest -m kwcoco.demo.toydata demodata_toy_dset --show

    Ignore:
        import xdev
        globals().update(xdev.get_func_kwargs(demodata_toy_dset))

    Example:
        >>> from kwcoco.demo.toydata import *
        >>> import kwcoco
        >>> dataset = demodata_toy_dset(gsize=(300, 300), aux=True, cache=False)
        >>> dpath = ub.ensure_app_cache_dir('kwcoco', 'toy_dset')
        >>> dset = kwcoco.CocoDataset(dataset)
        >>> # xdoctest: +REQUIRES(--show)
        >>> print(ub.repr2(dset.dataset, nl=2))
        >>> import kwplot
        >>> kwplot.autompl()
        >>> dset.show_image(gid=1)
        >>> ub.startfile(dpath)
    """
    if dpath is None:
        dpath = ub.ensure_app_cache_dir('kwcoco', 'toy_dset')
    else:
        ub.ensuredir(dpath)

    import kwarray
    rng = kwarray.ensure_rng(rng)

    catpats = CategoryPatterns.coerce([
        # 'box',
        # 'circle',
        'star',
        'superstar',
        'eff',
        # 'octagon',
        # 'diamond'
    ])

    anchors = np.array([
        [1, 1], [2, 2], [1.5, 1], [2, 1], [3, 1], [3, 2], [2.5, 2.5],
    ])
    anchors = np.vstack([anchors, anchors[:, ::-1]])
    anchors = np.vstack([anchors, anchors * 1.5])
    # anchors = np.vstack([anchors, anchors * 2.0])
    anchors /= (anchors.max() * 3)
    anchors = np.array(sorted(set(map(tuple, anchors.tolist()))))

    cfg = {
        'anchors': anchors,
        'gsize': gsize,
        'n_imgs': n_imgs,
        'categories': catpats.categories,
        'newstyle': newstyle,
        'keypoint_categories': catpats.keypoint_categories,
        'rng': ub.hash_data(rng),
        'aux': aux,
    }
    cacher = ub.Cacher('toy_dset_v3', dpath=ub.ensuredir(dpath, 'cache'),
                       cfgstr=ub.repr2(cfg), verbose=verbose, enabled=0)

    root_dpath = ub.ensuredir((dpath, 'shapes_{}_{}'.format(
        cfg['n_imgs'], cacher._condense_cfgstr())))

    img_dpath = ub.ensuredir((root_dpath, 'images'))

    n_have = len(list(glob.glob(join(img_dpath, '*.png'))))
    # Hack: Only allow cache loading if the data seems to exist
    cacher.enabled = (n_have == n_imgs) and cache

    bg_intensity = .1
    fg_scale = 0.5
    bg_scale = 0.8

    dataset = cacher.tryload(on_error='clear')
    if dataset is None:
        ub.delete(img_dpath)
        ub.ensuredir(img_dpath)
        dataset = {
            'images': [],
            'annotations': [],
            'categories': [],
        }

        dataset['categories'].append({
            'id': 0,
            'name': 'background',
        })

        name_to_cid = {}
        for cat in catpats.categories:
            dataset['categories'].append(cat)
            name_to_cid[cat['name']] = cat['id']

        if newstyle:
            # Add newstyle keypoint categories
            kpname_to_id = {}
            dataset['keypoint_categories'] = []
            for kpcat in catpats.keypoint_categories:
                dataset['keypoint_categories'].append(kpcat)
                kpname_to_id[kpcat['name']] = kpcat['id']

        for __ in ub.ProgIter(range(n_imgs), label='creating data'):

            # TODO: parallelize
            img, anns = demodata_toy_img(anchors, gsize=gsize,
                                         categories=catpats,
                                         newstyle=newstyle, fg_scale=fg_scale,
                                         bg_scale=bg_scale,
                                         bg_intensity=bg_intensity, rng=rng,
                                         aux=aux)
            imdata = img.pop('imdata')

            gid = len(dataset['images']) + 1
            fname = 'img_{:05d}.png'.format(gid)
            fpath = join(img_dpath, fname)
            img.update({
                'id': gid,
                'file_name': fpath,
                'channels': 'rgb',
            })
            auxillaries = img.pop('auxillary', None)
            if auxillaries is not None:
                for auxdict in auxillaries:
                    aux_dpath = ub.ensuredir(
                        (root_dpath, 'aux_' + auxdict['channels']))
                    aux_fpath = ub.augpath(join(aux_dpath, fname), ext='.tif')
                    ub.ensuredir(aux_dpath)
                    auxdata = (auxdict.pop('imdata') * 255).astype(np.uint8)
                    auxdict['file_name'] = aux_fpath

                    print(kwarray.stats_dict(auxdata))
                    try:
                        import gdal  # NOQA
                        kwimage.imwrite(aux_fpath, auxdata, backend='gdal')
                    except Exception:
                        kwimage.imwrite(aux_fpath, auxdata)

                img['auxillary'] = auxillaries

            dataset['images'].append(img)
            for ann in anns:
                if newstyle:
                    # rectify newstyle keypoint ids
                    for kpdict in ann.get('keypoints', []):
                        kpname = kpdict.pop('keypoint_category')
                        kpdict['keypoint_category_id'] = kpname_to_id[kpname]
                cid = name_to_cid[ann.pop('category_name')]
                ann.update({
                    'id': len(dataset['annotations']) + 1,
                    'image_id': gid,
                    'category_id': cid,
                })
                dataset['annotations'].append(ann)

            kwimage.imwrite(fpath, imdata)

        import json
        with open(join(dpath, 'toy_dset.mscoco.json'), 'w') as file:
            if six.PY2:
                json.dump(dataset, file, indent=4)
            else:
                json.dump(dataset, file, indent='    ')

        cacher.enabled = True
        cacher.save(dataset)

    return dataset


def random_video_dset(gsize=(600, 600), num_frames=2, verbose=3, num_tracks=2,
                      tid_start=1, gid_start=1, num_videos=1, render=False,
                      rng=None):
    """
    Example:
        >>> from kwcoco.demo.toydata import *  # NOQA
        >>> dset = random_video_dset(render=True, num_videos=3, num_frames=2, num_tracks=10)
        >>> # xdoctest: +REQUIRES(--show)
        >>> dset.show_image(1, doclf=True)
        >>> dset.show_image(2, doclf=True)

        import xdev
        globals().update(xdev.get_func_kwargs(random_video_dset))
        num_videos = 2
    """
    rng = kwarray.ensure_rng(rng)
    subsets = []
    tid_start = 1
    for vidid in range(1, num_videos + 1):
        dset = random_single_video_dset(
            gsize=gsize,
            num_frames=num_frames,
            num_tracks=num_tracks, tid_start=tid_start,
            gid_start=gid_start, video_id=vidid, render=False,
            autobuild=False, rng=rng)

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
        render_toy_dataset(dset, rng=rng, **renderkw)

    dset._build_index()
    return dset


def random_single_video_dset(gsize=(600, 600), num_frames=5, verbose=3,
                             num_tracks=3, tid_start=1, gid_start=1,
                             video_id=1, render=False, rng=None, autobuild=True):
    """
    Example:
        >>> from kwcoco.demo.toydata import *  # NOQA
        >>> dset = random_single_video_dset(render=True, num_frames=2, num_tracks=10)
        >>> # xdoctest: +REQUIRES(--show)
        >>> dset.show_image(1, doclf=True)
        >>> dset.show_image(2, doclf=True)
    """
    import pandas as pd
    rng = kwarray.ensure_rng(rng)

    image_ids = list(range(gid_start, num_frames + gid_start))
    track_ids = list(range(tid_start, num_tracks + tid_start))

    import kwcoco
    dset = kwcoco.CocoDataset(autobuild=False)
    dset.add_video(name='toy_video_{}'.format(video_id), id=video_id)

    for frame_idx, gid in enumerate(image_ids):
        dset.add_image(**{
            'id': gid,
            'file_name': '<todo-generate>',
            'width': gsize[0],
            'height': gsize[1],
            'frame_index': frame_idx,
            'video_id': video_id,
        })

    classes = ['star', 'superstar', 'eff']
    for catname in classes:
        dset.ensure_category(name=catname)

    tid_to_anns = {}
    for tid in track_ids:
        degree = rng.randint(1, 5)
        num = num_frames
        path = random_path(num, degree=degree, rng=rng)
        boxes = kwimage.Boxes.random(
            num=num, scale=1.0, format='cxywh', rng=rng)

        # Smooth out varying box sizes
        alpha = rng.rand() * 0.1
        wh = pd.DataFrame(boxes.data[:, 2:4], columns=['w', 'h'])

        ar = wh['w'] / wh['h']
        min_ar = 0.25
        max_ar = 1 / min_ar

        wh['w'][ar < min_ar] = wh['h'] * 0.25
        wh['h'][ar > max_ar] = wh['w'] * 0.25

        box_dims = wh.ewm(alpha=alpha, adjust=False).mean()
        boxes.data[:, 0:2] = path
        boxes.data[:, 2:4] = box_dims.values
        boxes = boxes.scale(gsize).scale(0.9, about='center')

        def warp_within_bounds(self, x_min, y_min, x_max, y_max):
            """
            Translate / scale the boxes to fit in the bounds

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

            tl_xy_over = np.maximum(tl_xy_lb - tl_xy_min, 0)
            # Now at the minimum coord
            tmp = tlbr.translate(tl_xy_over)
            _tl_x, _tl_y, _br_x, _br_y = tmp.components
            tmp_tl_xy_min = np.c_[_tl_x, _tl_y].min(axis=0)
            # tmp_br_xy_max = np.c_[_br_x, _br_y].max(axis=0)

            tmp.translate(-tmp_tl_xy_min)
            sf = np.minimum(size_ub / size_max, 1)
            out = tmp.scale(sf).translate(tmp_tl_xy_min)
            return out

        oob_pad = -20  # allow some out of bounds
        boxes = boxes.to_tlbr()
        boxes = boxes.clip(0, 0, gsize[0], gsize[1])
        boxes = warp_within_bounds(boxes, 0 - oob_pad, 0 - oob_pad, gsize[0] + oob_pad, gsize[1] + oob_pad)
        boxes = boxes.to_xywh()

        boxes.data = boxes.data.round(1)
        cidx = rng.randint(0, len(classes))
        dets = kwimage.Detections(
            boxes=boxes,
            class_idxs=np.array([cidx] * len(boxes)),
            classes=classes,
        )

        anns = list(dets.to_coco(dset=dset))
        start_frame = 0
        for frame_index, ann in enumerate(anns, start=start_frame):
            ann['track_id'] = tid
            ann['image_id'] = dset.dataset['images'][frame_index]['id']
            dset.add_annotation(**ann)
        tid_to_anns[tid] = anns

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
    if renderkw is not None:
        render_toy_dataset(dset, rng=rng, **renderkw)
    if autobuild:
        dset._build_index()
    return dset


def render_toy_dataset(dset, rng, dpath=None):
    """
    Create toydata renderings for a preconstructed coco dataset.

    Example:
        >>> from kwcoco.demo.toydata import *  # NOQA
        >>> import kwarray
        >>> rng = None
        >>> rng = kwarray.ensure_rng(rng)
        >>> dset = random_video_dset(rng=rng, num_frames=10, num_tracks=3)
        >>> dset = render_toy_dataset(dset, rng)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> gids = list(dset.imgs.keys())
        >>> pnums = kwplot.PlotNums(nSubplots=len(gids))
        >>> for gid in gids:
        >>>     dset.show_image(gid, pnum=pnums(), fnum=1)
        >>> pnums = kwplot.PlotNums(nSubplots=len(gids))
        >>> for gid in gids:
        >>>     canvas = dset.draw_image(gid)
        >>>     kwplot.imshow(canvas, pnum=pnums(), fnum=2)
    """
    rng = kwarray.ensure_rng(rng)
    dset._build_index()

    dset._ensure_json_serializable()
    hashid = dset._build_hashid()[0:24]

    dpath = None
    if dpath is None:
        dpath = ub.ensure_app_cache_dir('kwcoco', 'toy_dset')
    else:
        ub.ensuredir(dpath)
    root_dpath = ub.ensuredir((dpath, 'render_{}'.format(hashid)))
    img_dpath = ub.ensuredir((root_dpath, 'images'))

    for gid in dset.imgs.keys():

        render_toy_image(dset, gid, rng=rng)

        img = dset.imgs[gid]
        imdata = img.pop('imdata')
        fname = 'img_{:05d}.png'.format(gid)
        fpath = join(img_dpath, fname)
        img.update({
            'file_name': fpath,
            'channels': 'rgb',
        })
        auxillaries = img.pop('auxillary', None)
        if auxillaries is not None:
            for auxdict in auxillaries:
                aux_dpath = ub.ensuredir(
                    (root_dpath, 'aux_' + auxdict['channels']))
                aux_fpath = ub.augpath(join(aux_dpath, fname), ext='.tif')
                ub.ensuredir(aux_dpath)
                auxdata = (auxdict.pop('imdata') * 255).astype(np.uint8)
                auxdict['file_name'] = aux_fpath
                try:
                    import gdal  # NOQA
                    kwimage.imwrite(aux_fpath, auxdata, backend='gdal')
                except Exception:
                    kwimage.imwrite(aux_fpath, auxdata)
            img['auxillary'] = auxillaries

        kwimage.imwrite(fpath, imdata)
    dset._build_index()
    return dset


def render_toy_image(dset, gid, rng=None):
    """
    Example:
        >>> from kwcoco.demo.toydata import *  # NOQA
        >>> gsize=(600, 600)
        >>> num_frames=5
        >>> verbose=3
        >>> rng = None
        >>> import kwarray
        >>> rng = kwarray.ensure_rng(rng)
        >>> dset = random_video_dset(
        >>>     gsize=gsize, num_frames=num_frames, verbose=verbose, rng=rng)
        >>> gid = 1
        >>> render_toy_image(dset, gid, rng)
        >>> gid = 1
        >>> canvas = dset.imgs[gid]['imdata']
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.imshow(canvas, doclf=True)
        >>> dets = dset.annots(gid=gid).detections
        >>> dets.draw()
    """
    rng = kwarray.ensure_rng(rng)

    # bg_intensity = 0
    # bg_scale = 1
    # fg_scale = 1
    # fg_intensity = 1

    gray = 1

    fg_scale = 0.5
    bg_scale = 0.8
    bg_intensity = 0.1
    fg_intensity = 0.9

    newstyle = True

    categories = list(dset.name_to_cat.keys())
    catpats = CategoryPatterns.coerce(categories, fg_scale=fg_scale,
                                      fg_intensity=fg_intensity, rng=rng)

    if newstyle:
        # Add newstyle keypoint categories
        kpname_to_id = {}
        dset.dataset['keypoint_categories'] = []
        for kpcat in catpats.keypoint_categories:
            dset.dataset['keypoint_categories'].append(kpcat)
            kpname_to_id[kpcat['name']] = kpcat['id']

    img = dset.imgs[gid]
    gw, gh = img['width'], img['height']
    dims = (gh, gw)
    annots = dset.annots(gid=gid)

    def render_background():
        # This is 2x as fast for gsize=(300,300)
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

        aux = 0
        if aux:
            auxdata = np.zeros(gshape, dtype=np.float32)
        else:
            auxdata = None
        return imdata, auxdata

    def render_foreground(imdata, auxdata):
        boxes = annots.boxes
        tlbr_boxes = boxes.to_tlbr().clip(0, 0, None, None).data.round(0).astype(np.int)

        # Render coco-style annotation dictionaries
        for ann, tlbr in zip(annots.objs, tlbr_boxes):
            print('ann = {}'.format(ub.repr2(ann, nl=1)))
            catname = dset._resolve_to_cat(ann['category_id'])['name']
            tl_x, tl_y, br_x, br_y = tlbr
            chip_index = tuple([slice(tl_y, br_y), slice(tl_x, br_x)])
            chip = imdata[chip_index]
            xy_offset = (tl_x, tl_y)

            if chip.size:
                info = catpats.render_category(catname, chip, xy_offset, dims,
                                               newstyle=newstyle)

                fgdata = info['data']
                if gray:
                    fgdata = fgdata.mean(axis=2, keepdims=True)

                imdata[tl_y:br_y, tl_x:br_x, :] = fgdata
                ann.update({
                    # 'segmentation': info['segmentation'],
                    # 'keypoints': info['keypoints'],
                })

                if auxdata is not None:
                    seg = kwimage.Segmentation.coerce(info['segmentation'])
                    seg = seg.to_multi_polygon()
                    val = rng.uniform(0.2, 1.0)
                    # val = 1.0
                    auxdata = seg.fill(auxdata, value=val)
            else:
                ann.update({
                    # 'segmentation': None,
                    # 'keypoints': None,
                })

            # if newstyle:
            #     # rectify newstyle keypoint ids
            #     for kpdict in ann.get('keypoints', []):
            #         kpname = kpdict.pop('keypoint_category')
            #         kpdict['keypoint_category_id'] = kpname_to_id[kpname]
        return imdata, auxdata

    imdata, auxdata = render_background()
    imdata, auxdata = render_foreground(imdata, auxdata)

    imdata = (imdata * 255).astype(np.uint8)
    imdata = kwimage.atleast_3channels(imdata)

    main_channels = 'rgb'
    # main_channels = 'gray' if gray else 'rgb'

    img.update({
        # 'width': gw,
        # 'height': gh,
        'imdata': imdata,
        'channels': main_channels,
    })

    if auxdata is not None:
        mask = rng.rand(*auxdata.shape[0:2]) > 0.5
        auxdata = kwimage.fourier_mask(auxdata, mask)
        auxdata = (auxdata - auxdata.min())
        auxdata = (auxdata / max(1e-8, auxdata.max()))
        auxdata = auxdata.clip(0, 1)
        # Hack aux data is always disparity for now
        img['auxillary'] = [{
            'imdata': auxdata,
            'channels': 'disparity',
        }]


def random_path(num, degree=1, dimension=2, rng=None):
    """
    Create a random path using a bezier curve.

    Args:
        num (int): number of points in the path
        degree (int, default=1): degree of curvieness of the path
        dimension (int, default=2): number of spatial dimensions
        rng (RandomState, default=None): seed

    References:
        https://github.com/dhermes/bezier

    Example:
        >>> from kwcoco.demo.toydata import *  # NOQA
        >>> num = 50
        >>> dimension = 2
        >>> degree = 10
        >>> rng = 0
        >>> path = random_path(num, degree, dimension, rng)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> kwplot.multi_plot(xdata=path[:, 0], ydata=path[:, 1])
        >>> kwplot.show_if_requested()
    """
    import bezier
    rng = kwarray.ensure_rng(rng)
    # Create random bezier control points
    nodes_f = rng.rand(degree + 1, dimension).T  # F-contiguous
    curve = bezier.Curve(nodes_f, degree=degree)
    # Evaluate path points
    s_vals = np.linspace(0, 1, num)
    path_f = curve.evaluate_multi(s_vals)
    path = path_f.T  # C-contiguous
    return path
