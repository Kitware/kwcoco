# -*- coding: utf-8 -*-
"""
Notes:
    The implementation of `demodata_toy_img` and `demodata_toy_dset` should be
    redone using the tools built for `random_video_dset`, which have more
    extensible implementations.

"""
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

        aux: if specified builds auxiliary channels

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
        >>> auxdata = img['auxiliary'][0]['imdata']
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
        img['auxiliary'] = [{
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

    SeeAlso:
        random_video_dset

    CommandLine:
        xdoctest -m kwcoco.demo.toydata demodata_toy_dset --show

    Ignore:
        import xdev
        globals().update(xdev.get_func_kwargs(demodata_toy_dset))

    TODO:
        - [ ] Non-homogeneous images sizes

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
            auxiliaries = img.pop('auxiliary', None)
            if auxiliaries is not None:
                for auxdict in auxiliaries:
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

                img['auxiliary'] = auxiliaries

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


def random_video_dset(
        num_videos=1, num_frames=2, num_tracks=2, anchors=None,
        gsize=(600, 600), verbose=3, render=False, aux=None, rng=None,
        dpath=None):
    """
    Create a toy Coco Video Dataset

    Args:
        num_videos (int) : number of videos

        num_frames (int) : number of images per video

        num_tracks (int) : number of tracks per video

        gsize : image size

        render (bool | dict):
            if truthy the toy annotations are synthetically rendered. See
            :func:`render_toy_image` for details.

        rng (int | None | RandomState): random seed / state

        dpath (str): only used if render is truthy, place to write rendered
            images.

    SeeAlso:
        random_single_video_dset

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
    gid_start = 1
    for vidid in range(1, num_videos + 1):
        dset = random_single_video_dset(
            gsize=gsize, num_frames=num_frames, num_tracks=num_tracks,
            tid_start=tid_start, anchors=anchors, gid_start=gid_start,
            video_id=vidid, render=False, autobuild=False, aux=aux, rng=rng)

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
        render_toy_dataset(dset, rng=rng, dpath=dpath, renderkw=renderkw)

    dset._build_index()
    return dset


def random_single_video_dset(gsize=(600, 600), num_frames=5,
                             num_tracks=3, tid_start=1, gid_start=1,
                             video_id=1, anchors=None, rng=None, render=False,
                             autobuild=True, verbose=3, aux=None):
    """
    Create the video scene layout of object positions.

    Args:
        gsize (Tuple[int, int]): size of the images

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

    TODO:
        - [ ] Need maximum allowed object overlap measure
        - [ ] Need better parameterized path generation

    Ignore:
        import xdev
        globals().update(xdev.get_func_kwargs(random_single_video_dset))

    Example:
        >>> from kwcoco.demo.toydata import *  # NOQA
        >>> anchors = np.array([ [0.3, 0.3],  [0.1, 0.1]])
        >>> dset = random_single_video_dset(render=True, num_frames=10, num_tracks=10, anchors=anchors)
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
        >>> from kwcoco.demo.toydata import *  # NOQA
        >>> anchors = np.array([ [0.2, 0.2],  [0.1, 0.1]])
        >>> gsize = np.array([(600, 600)])
        >>> print(anchors * gsize)
        >>> dset = random_single_video_dset(render=True, num_frames=10, anchors=anchors, num_tracks=10)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> plt.clf()
        >>> gids = list(dset.imgs.keys())
        >>> pnums = kwplot.PlotNums(nSubplots=len(gids), nRows=1)
        >>> for gid in gids:
        >>>     dset.show_image(gid, pnum=pnums(), fnum=1, title=False)
        >>> pnums = kwplot.PlotNums(nSubplots=len(gids))

    Example:
        >>> from kwcoco.demo.toydata import *  # NOQA
        >>> dset = random_single_video_dset(num_frames=10, num_tracks=10, aux=True)
        >>> assert 'auxiliary' in dset.imgs[1]
        >>> assert dset.imgs[1]['auxiliary'][0]['channels']
        >>> assert dset.imgs[1]['auxiliary'][1]['channels']
    """
    import pandas as pd
    import kwcoco
    rng = kwarray.ensure_rng(rng)

    image_ids = list(range(gid_start, num_frames + gid_start))
    track_ids = list(range(tid_start, num_tracks + tid_start))

    dset = kwcoco.CocoDataset(autobuild=False)
    dset.add_video(name='toy_video_{}'.format(video_id), id=video_id)

    for frame_idx, gid in enumerate(image_ids):
        img = {
            'id': gid,
            'file_name': '<todo-generate-{}-{}>'.format(video_id, frame_idx),
            'width': gsize[0],
            'height': gsize[1],
            'frame_index': frame_idx,
            'video_id': video_id,
        }
        if aux:
            if aux is True:
                aux = ['disparity', 'flowx|flowy']
            aux = aux if ub.iterable(aux) else [aux]
            img['auxiliary'] = []
            for chaninfo in aux:
                if isinstance(chaninfo, str):
                    chaninfo = {
                        'channels': chaninfo,
                        'width': gsize[0],
                        'height': gsize[1],
                    }
                # Add placeholder for auxiliary image data
                auxitem = {
                    'file_name': '<todo-generate-{}-{}>'.format(video_id, frame_idx),
                }
                auxitem.update(chaninfo)
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
        num_objects=num_tracks, rng=rng)

    tid_to_anns = {}
    for tid, path in zip(track_ids, paths):
        # degree = rng.randint(1, 5)
        # num = num_frames
        # path = random_path(num, degree=degree, rng=rng)

        if anchors is None:
            anchors_ = anchors
        else:
            anchors_ = np.array([anchors[rng.randint(0, len(anchors))]])

        # Box scale
        boxes = kwimage.Boxes.random(
            num=num_frames, scale=1.0, format='cxywh', rng=rng,
            anchors=anchors_)

        # Smooth out varying box sizes
        alpha = rng.rand() * 0.1
        wh = pd.DataFrame(boxes.data[:, 2:4], columns=['w', 'h'])

        ar = wh['w'] / wh['h']
        min_ar = 0.25
        max_ar = 1 / min_ar

        wh['w'][ar < min_ar] = wh['h'] * 0.25
        wh['h'][ar > max_ar] = wh['w'] * 0.25

        box_dims = wh.ewm(alpha=alpha, adjust=False).mean()
        # print('path = {!r}'.format(path))
        boxes.data[:, 0:2] = path
        boxes.data[:, 2:4] = box_dims.values
        # boxes = boxes.scale(0.1, about='center')
        # boxes = boxes.scale(gsize).scale(0.5, about='center')
        boxes = boxes.scale(gsize, about='origin')
        boxes = boxes.scale(0.9, about='center')

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

        # oob_pad = -20  # allow some out of bounds
        # oob_pad = 20  # allow some out of bounds
        # boxes = boxes.to_tlbr()
        # TODO: need better path distributions
        # boxes = warp_within_bounds(boxes, 0 - oob_pad, 0 - oob_pad, gsize[0] + oob_pad, gsize[1] + oob_pad)
        boxes = boxes.to_xywh()

        boxes.data = boxes.data.round(1)
        cidx = rng.randint(0, len(classes))
        dets = kwimage.Detections(
            boxes=boxes,
            class_idxs=np.array([cidx] * len(boxes)),
            classes=classes,
        )

        WITH_KPTS_SSEG = True
        if WITH_KPTS_SSEG:
            kpts = []
            ssegs = []
            ddims = boxes.data[:, 2:4].astype(np.int)[:, ::-1]
            offsets = boxes.data[:, 0:2].astype(np.int)
            cnames = [classes[cidx] for cidx in dets.class_idxs]
            for dims, xy_offset, cname in zip(ddims, offsets, cnames):
                info = catpats._todo_refactor_geometric_info(cname, xy_offset, dims)
                kpts.append(info['kpts'])

                if False and rng.rand() > 0.5:
                    # sseg dropout
                    ssegs.append(None)
                else:
                    ssegs.append(info['segmentation'])

            dets.data['keypoints'] = kwimage.PointsList(kpts)
            dets.data['segmentations'] = kwimage.SegmentationList(ssegs)

        anns = list(dets.to_coco(dset=dset, style='new'))

        start_frame = 0
        for frame_index, ann in enumerate(anns, start=start_frame):
            ann['track_id'] = tid
            ann['image_id'] = dset.dataset['images'][frame_index]['id']
            dset.add_annotation(**ann)
        tid_to_anns[tid] = anns

    HACK_FIX_COCO_KPTS_FORMAT = True
    if HACK_FIX_COCO_KPTS_FORMAT:
        # Hack to fix coco formatting from Detections.to_coco
        kpt_cats = dset.keypoint_categories()
        for ann in dset.dataset['annotations']:
            for kpt in ann['keypoints']:
                if 'keypoint_category_id' not in kpt:
                    kp_catname = kpt.pop('keypoint_category')
                    kpt['keypoint_category_id'] = kpt_cats.node_to_id[kp_catname]

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
        render_toy_dataset(dset, rng=rng, renderkw=renderkw)
    if autobuild:
        dset._build_index()
    return dset


def render_toy_dataset(dset, rng, dpath=None, renderkw=None):
    """
    Create toydata renderings for a preconstructed coco dataset.

    Example:
        >>> from kwcoco.demo.toydata import *  # NOQA
        >>> import kwarray
        >>> rng = None
        >>> rng = kwarray.ensure_rng(rng)
        >>> num_tracks = 3
        >>> dset = random_video_dset(rng=rng, num_videos=3, num_frames=10, num_tracks=3)
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
        >>> #
        >>> # for gid in gids:
        >>> #    canvas = dset.draw_image(gid)
        >>> #    kwplot.imshow(canvas, pnum=pnums(), fnum=2)
    """
    rng = kwarray.ensure_rng(rng)
    dset._build_index()

    dset._ensure_json_serializable()
    hashid = dset._build_hashid()[0:24]

    print('dpath = {!r}'.format(dpath))
    if dpath is None:
        dpath = ub.ensure_app_cache_dir('kwcoco', 'toy_dset')
    else:
        ub.ensuredir(dpath)
    root_dpath = ub.ensuredir((dpath, 'render_{}'.format(hashid)))
    img_dpath = ub.ensuredir((root_dpath, 'images'))
    print('img_dpath = {!r}'.format(img_dpath))

    for gid in dset.imgs.keys():

        # Render data inside the image
        img = render_toy_image(dset, gid, rng=rng, renderkw=renderkw)

        # Extract the data from memory and write it to disk
        imdata = img.pop('imdata')
        fname = 'img_{:05d}.png'.format(gid)
        fpath = join(img_dpath, fname)
        img.update({
            'file_name': fpath,
            'channels': 'rgb',
        })
        auxiliaries = img.pop('auxiliary', None)
        if auxiliaries is not None:
            for auxdict in auxiliaries:
                aux_dpath = ub.ensuredir(
                    (root_dpath, 'aux_' + auxdict['channels']))
                aux_fpath = ub.augpath(join(aux_dpath, fname), ext='.tif')
                ub.ensuredir(aux_dpath)
                auxdata = (auxdict.pop('imdata') * 255).astype(np.uint8)
                auxdict['file_name'] = aux_fpath
                try:
                    import gdal  # NOQA
                    kwimage.imwrite(aux_fpath, auxdata, backend='gdal', space=None)
                except Exception:
                    kwimage.imwrite(aux_fpath, auxdata, space=None)
            img['auxiliary'] = auxiliaries

        kwimage.imwrite(fpath, imdata)
    dset._build_index()
    return dset


def render_toy_image(dset, gid, rng=None, renderkw=None):
    """
    Modifies dataset inplace, rendering synthetic annotations

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
        >>> from kwcoco.demo.toydata import *  # NOQA
        >>> gsize=(600, 600)
        >>> num_frames=5
        >>> verbose=3
        >>> rng = None
        >>> import kwarray
        >>> rng = kwarray.ensure_rng(rng)
        >>> aux = 'mx'
        >>> dset = random_single_video_dset(
        >>>     gsize=gsize, num_frames=num_frames, verbose=verbose, aux=aux, rng=rng)
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
        >>> img, anns = demodata_toy_img(gsize=(172, 172), rng=None, aux=True)
        >>> print('anns = {}'.format(ub.repr2(anns, nl=1)))
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img['imdata'], pnum=(1, 2, 1), fnum=1)
        >>> auxdata = img['auxiliary'][0]['imdata']
        >>> kwplot.imshow(auxdata, pnum=(1, 2, 2), fnum=1)
        >>> kwplot.show_if_requested()

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

    def render_foreground(imdata, chan_to_auxinfo):
        boxes = annots.boxes
        tlbr_boxes = boxes.to_tlbr().clip(0, 0, None, None).data.round(0).astype(np.int)

        # Render coco-style annotation dictionaries
        for ann, tlbr in zip(annots.objs, tlbr_boxes):
            catname = dset._resolve_to_cat(ann['category_id'])['name']
            tl_x, tl_y, br_x, br_y = tlbr

            # TODO: Use pad-infinite-slices to fix a bug here

            chip_index = tuple([slice(tl_y, br_y), slice(tl_x, br_x)])
            chip = imdata[chip_index]
            xy_offset = (tl_x, tl_y)

            if chip.size:
                # todo: no need to make kpts / sseg if not requested
                info = catpats.render_category(
                    catname, chip, xy_offset, dims, newstyle=newstyle)

                fgdata = info['data']
                if gray:
                    fgdata = fgdata.mean(axis=2, keepdims=True)

                imdata[tl_y:br_y, tl_x:br_x, :] = fgdata

                if with_sseg:
                    ann['segmentation'] = info['segmentation']
                if with_kpts:
                    ann['keypoints'] = info['keypoints']

                if chan_to_auxinfo is not None:
                    coco_sseg = ann.get('segmentation', None)
                    if coco_sseg:
                        seg = kwimage.Segmentation.coerce(coco_sseg)
                        seg = seg.to_multi_polygon()
                        for chankey, auxinfo in chan_to_auxinfo.items():
                            val = rng.uniform(0.2, 1.0)
                            auxinfo['imdata'] = seg.fill(auxinfo['imdata'], value=val)
        return imdata, chan_to_auxinfo

    imdata, chan_to_auxinfo = render_background(img, rng, gray, bg_intensity, bg_scale)
    imdata, chan_to_auxinfo = render_foreground(imdata, chan_to_auxinfo)

    imdata = (imdata * 255).astype(np.uint8)
    imdata = kwimage.atleast_3channels(imdata)

    main_channels = 'gray' if gray else 'rgb'

    img.update({
        # 'width': gw,
        # 'height': gh,
        'imdata': imdata,
        'channels': main_channels,
    })
    for auxinfo in img.get('auxiliary', []):
        # Postprocess the auxiliary data so it looks interesting
        # It would be really cool if we could do this based on what
        # the simulated channel was.
        chankey = auxinfo['channels']
        auxdata = chan_to_auxinfo[chankey]['imdata']
        mask = rng.rand(*auxdata.shape[0:2]) > 0.5
        auxdata = kwimage.fourier_mask(auxdata, mask)
        auxdata = (auxdata - auxdata.min())
        auxdata = (auxdata / max(1e-8, auxdata.max()))
        auxdata = auxdata.clip(0, 1)
        auxinfo['imdata'] = auxdata

    return img


def false_color(twochan):
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


def render_background(img, rng, gray, bg_intensity, bg_scale):
    # This is 2x as fast for gsize=(300,300)
    gw, gh = img['width'], img['height']
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
        chankey = auxinfo['channels']
        # TODO:
        # Need to incorporate ChannelSpec abstraction here
        if chankey in ['mx|my', 'motion']:
            aux_bands = 2
        elif chankey == 'flowx|flowy':
            aux_bands = 2
        elif chankey == 'disparity':
            aux_bands = 1
        else:
            aux_bands = 1
        # TODO: make non-aligned auxiliary information?
        aux_width = auxinfo.get('width', gw)
        aux_height = auxinfo.get('height', gh)
        auxshape = (aux_height, aux_width, aux_bands)
        # auxdata = np.zeros(auxshape, dtype=np.float32)
        auxdata = kwarray.uniform(0, 0.01, auxshape, rng=rng, dtype=np.float32)
        auxinfo['imdata'] = auxdata
        chan_to_auxinfo[chankey] = auxinfo

    return imdata, chan_to_auxinfo


def random_multi_object_path(num_objects, num_frames, rng=None):
    """

    import kwarray
    import kwplot
    plt = kwplot.autoplt()

    num_objects = 5
    num_frames = 100
    rng = kwarray.ensure_rng(0)

    from kwcoco.demo.toydata import *  # NOQA
    paths = random_multi_object_path(num_objects, num_frames, rng)

    from mpl_toolkits.mplot3d import Axes3D  # NOQA
    ax = plt.gca(projection='3d')
    ax.cla()

    for path in paths:
        time = np.arange(len(path))
        ax.plot(time, path.T[0] * 1, path.T[1] * 1, 'o-')
    ax.set_xlim(0, num_frames)
    ax.set_ylim(-.01, 1.01)
    ax.set_zlim(-.01, 1.01)

    """
    import kwarray
    rng = kwarray.ensure_rng(rng)

    USE_TORCH = 0

    if not USE_TORCH:
        from kwcoco.demo.boids import Boids
        boids = Boids(num_objects, rng=rng).initialize()
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
    """
    import bezier
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
