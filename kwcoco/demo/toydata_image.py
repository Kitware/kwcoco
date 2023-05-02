"""
Generates "toydata" for demo and testing purposes.

Loose image version of the toydata generators.

Note:
    The implementation of `demodata_toy_img` and `demodata_toy_dset` should be
    redone using the tools built for `random_video_dset`, which have more
    extensible implementations.
"""
import glob
from os.path import basename
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


# Updated when toydata is modified.
# Internal cachers use this to invalidate old caches
TOYDATA_IMAGE_VERSION = 20


@profile
def demodata_toy_dset(image_size=(600, 600),
                      n_imgs=5,
                      verbose=3,
                      rng=0,
                      newstyle=True,
                      dpath=None,
                      fpath=None,
                      bundle_dpath=None,
                      aux=None,
                      use_cache=True,
                      **kwargs):
    """
    Create a toy detection problem

    Args:
        image_size (Tuple[int, int]): The width and height of the generated images

        n_imgs (int): number of images to generate

        rng (int | RandomState | None):
            random number generator or seed. Defaults to 0.

        newstyle (bool): create newstyle kwcoco data. default=True

        dpath (str | PathLike | None):
            path to the directory that will contain the bundle, (defaults to a
            kwcoco cache dir). Ignored if `bundle_dpath` is given.

        fpath (str | PathLike | None):
            path to the kwcoco file. The parent will be the bundle if it is not
            specified. Should be a descendant of the dpath if specified.

        bundle_dpath (str | PathLike | None):
            path to the directory that will store images.  If specified, dpath
            is ignored. If unspecified, a bundle will be written inside
            `dpath`.

        aux (bool | None): if True generates dummy auxiliary channels

        verbose (int): verbosity mode. default=3

        use_cache (bool): if True caches the generated json in the
            `dpath`. Default=True

        **kwargs : used for old backwards compatible argument names
            gsize - alias for image_size

    Returns:
        kwcoco.CocoDataset :

    SeeAlso:
        random_video_dset

    CommandLine:
        xdoctest -m kwcoco.demo.toydata_image demodata_toy_dset --show

    TODO:
        - [ ] Non-homogeneous images sizes

    Example:
        >>> from kwcoco.demo.toydata_image import *
        >>> import kwcoco
        >>> dset = demodata_toy_dset(image_size=(300, 300), aux=True, use_cache=False)
        >>> # xdoctest: +REQUIRES(--show)
        >>> print(ub.urepr(dset.dataset, nl=2))
        >>> import kwplot
        >>> kwplot.autompl()
        >>> dset.show_image(gid=1)
        >>> ub.startfile(dset.bundle_dpath)

        dset._tree()

        >>> from kwcoco.demo.toydata_image import *
        >>> import kwcoco

        dset = demodata_toy_dset(image_size=(300, 300), aux=True, use_cache=False)
        print(dset.imgs[1])
        dset._tree()

        dset = demodata_toy_dset(image_size=(300, 300), aux=True, use_cache=False,
            bundle_dpath='test_bundle')
        print(dset.imgs[1])
        dset._tree()

        dset = demodata_toy_dset(
            image_size=(300, 300), aux=True, use_cache=False, dpath='test_cache_dpath')

    Ignore:
        import xdev
        globals().update(xdev.get_func_kwargs(demodata_toy_dset))

    """
    import kwcoco

    if 'gsize' in kwargs:  # nocover
        if 0:
            # TODO: enable this warning
            import warnings
            warnings.warn('gsize is deprecated. Use image_size param instead',
                          DeprecationWarning)
        image_size = kwargs.pop('gsize')
    assert len(kwargs) == 0, 'unknown kwargs={}'.format(kwargs)

    if bundle_dpath is None:
        if dpath is None:
            dpath = ub.Path.appdir('kwcoco', 'demodata_bundles').ensuredir()
        else:
            ub.ensuredir(dpath)
    else:
        ub.ensuredir(bundle_dpath)

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

    # This configuration dictionary is what the cache depends on
    cfg = {
        'anchors': anchors,
        'image_size': image_size,
        'n_imgs': n_imgs,
        'categories': catpats.categories,
        'newstyle': newstyle,
        'keypoint_categories': catpats.keypoint_categories,
        'rng': ub.hash_data(rng),
        'aux': aux,
    }

    depends = ub.hash_data(cfg, base='abc')[0:14]

    if bundle_dpath is None:
        bundle_dname = 'shapes_{}_{}'.format(cfg['n_imgs'], depends)
        bundle_dpath = ub.ensuredir((dpath, bundle_dname))

    from os.path import abspath
    bundle_dpath = abspath(bundle_dpath)

    if fpath is None:
        fpath = join(bundle_dpath, 'data.kwcoco.json')

    cache_dpath = ub.ensuredir((bundle_dpath, '_cache'))
    assets_dpath = ub.ensuredir((bundle_dpath, '_assets'))
    img_dpath = ub.ensuredir((assets_dpath, 'images'))
    dset_fpath = fpath

    img_dpath = ub.ensuredir(img_dpath)
    cache_dpath = ub.ensuredir(cache_dpath)

    stamp = ub.CacheStamp(
        'toy_dset_stamp_v{:03d}'.format(TOYDATA_IMAGE_VERSION),
        dpath=cache_dpath, depends=depends, verbose=verbose, enabled=0)

    n_have = len(list(glob.glob(join(img_dpath, '*.png'))))
    # Hack: Only allow cache loading if the data seems to exist
    stamp.cacher.enabled = (n_have == n_imgs) and use_cache

    # TODO: parametarize
    bg_intensity = .1
    fg_scale = 0.5
    bg_scale = 0.8

    if stamp.expired():
        ub.delete(img_dpath, verbose=1)
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

        try:
            from osgeo import gdal  # NOQA
        except Exception:
            imwrite_kwargs = {}
        else:
            imwrite_kwargs = {'backend': 'gdal'}

        for __ in ub.ProgIter(range(n_imgs), label='creating data'):
            # TODO: parallelize
            img, anns = demodata_toy_img(anchors, image_size=image_size,
                                         categories=catpats, newstyle=newstyle,
                                         fg_scale=fg_scale, bg_scale=bg_scale,
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
                        (assets_dpath, 'auxiliary', auxdict['channels']))
                    aux_fpath = ub.augpath(join(aux_dpath, fname), ext='.tif')
                    ub.ensuredir(aux_dpath)
                    auxdata = (auxdict.pop('imdata') * 255).astype(np.uint8)
                    auxdict['file_name'] = aux_fpath
                    kwimage.imwrite(aux_fpath, auxdata, **imwrite_kwargs)

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
            # print('write fpath = {!r}'.format(fpath))

        fname = basename(dset_fpath)
        dset = kwcoco.CocoDataset(dataset, bundle_dpath=bundle_dpath,
                                  fname=fname)
        dset.dset_fpath = dset_fpath
        print('dump dset.dset_fpath = {!r}'.format(dset.dset_fpath))
        dset.dump(dset.dset_fpath, newlines=True)
        stamp.renew()
    else:
        # otherwise load the data
        # bundle_dpath = dirname(dset_fpath)
        print('read dset_fpath = {!r}'.format(dset_fpath))
        dset = kwcoco.CocoDataset(dset_fpath, bundle_dpath=bundle_dpath)

    dset.tag = basename(bundle_dpath)
    dset.fpath = dset_fpath

    # print('dset.bundle_dpath = {!r}'.format(dset.bundle_dpath))
    # dset.reroot(dset.bundle_dpath)

    return dset


def demodata_toy_img(anchors=None, image_size=(104, 104), categories=None,
                     n_annots=(0, 50), fg_scale=0.5, bg_scale=0.8,
                     bg_intensity=0.1, fg_intensity=0.9,
                     gray=True, centerobj=None, exact=False,
                     newstyle=True, rng=None, aux=None, **kwargs):
    r"""
    Generate a single image with non-overlapping toy objects of available
    categories.

    TODO:
        DEPRECATE IN FAVOR OF
            random_single_video_dset + render_toy_image

    Args:
        anchors (ndarray | None): Nx2 base width / height of boxes

        gsize (Tuple[int, int]): width / height of the image

        categories (List[str] | None): list of category names

        n_annots (Tuple | int): controls how many annotations are in the image.
            if it is a tuple, then it is interpreted as uniform random bounds

        fg_scale (float): standard deviation of foreground intensity

        bg_scale (float): standard deviation of background intensity

        bg_intensity (float): mean of background intensity

        fg_intensity (float): mean of foreground intensity

        centerobj (bool | None):
            if 'pos', then the first annotation will be in the center of the
            image, if 'neg', then no annotations will be in the center.

        exact (bool): if True, ensures that exactly the number of specified
            annots are generated.

        newstyle (bool): use new-sytle kwcoco format

        rng (RandomState | int | None):
            the random state used to seed the process

        aux (bool | None): if specified builds auxiliary channels

        **kwargs : used for old backwards compatible argument names.
            gsize - alias for image_size

    CommandLine:
        xdoctest -m kwcoco.demo.toydata_image demodata_toy_img:0 --profile
        xdoctest -m kwcoco.demo.toydata_image demodata_toy_img:1 --show

    Example:
        >>> from kwcoco.demo.toydata_image import *  # NOQA
        >>> img, anns = demodata_toy_img(image_size=(32, 32), anchors=[[.3, .3]], rng=0)
        >>> img['imdata'] = '<ndarray shape={}>'.format(img['imdata'].shape)
        >>> print('img = {}'.format(ub.urepr(img)))
        >>> print('anns = {}'.format(ub.urepr(anns, nl=2, cbr=True)))
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
        >>> img, anns = demodata_toy_img(image_size=(172, 172), rng=None, aux=True)
        >>> print('anns = {}'.format(ub.urepr(anns, nl=1)))
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img['imdata'], pnum=(1, 2, 1), fnum=1)
        >>> auxdata = img['auxiliary'][0]['imdata']
        >>> kwplot.imshow(auxdata, pnum=(1, 2, 2), fnum=1)
        >>> kwplot.show_if_requested()
    Example:
        >>> # xdoctest: +REQUIRES(--show)
        >>> img, anns = demodata_toy_img(image_size=(172, 172), rng=None, aux=True)
        >>> print('anns = {}'.format(ub.urepr(anns, nl=1)))
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img['imdata'], pnum=(1, 2, 1), fnum=1)
        >>> auxdata = img['auxiliary'][0]['imdata']
        >>> kwplot.imshow(auxdata, pnum=(1, 2, 2), fnum=1)
        >>> kwplot.show_if_requested()

    Ignore:
        from kwcoco.demo.toydata_image import *
        import xinspect
        globals().update(xinspect.get_kwargs(demodata_toy_img))

    """
    if anchors is None:
        anchors = [[.20, .20]]
    anchors = np.asarray(anchors)

    if 'gsize' in kwargs:  # nocover
        if 0:
            # TODO: enable this warning
            import warnings
            warnings.warn('gsize is deprecated. Use image_size param instead',
                          DeprecationWarning)
        image_size = kwargs.pop('gsize')
    assert len(kwargs) == 0, 'unknown kwargs={}'.format(**kwargs)

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
        boxes = boxes.scale(image_size)
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
            cxywh.data[0, 0:2] = np.array(image_size) / 2
            boxes = cxywh.to_ltrb()

    # Make sure the first box is always kept.
    box_priority = np.arange(boxes.shape[0])[::-1].astype(np.float32)
    boxes.ious(boxes)

    nms_impls = ub.oset(['cython_cpu', 'numpy'])
    nms_impls = nms_impls & kwimage.algo.available_nms_impls()
    nms_impl = nms_impls[0]

    if len(boxes) > 1:
        tlbr_data = boxes.to_ltrb().data
        keep = kwimage.non_max_supression(
            tlbr_data, scores=box_priority, thresh=0.0, impl=nms_impl)
        boxes = boxes[keep]

    if centerobj == 'neg':
        # The center of the image should be negative so remove the center box
        boxes = boxes[1:]

    boxes = boxes.scale(.8).translate(.1 * min(image_size))
    boxes.data = boxes.data.astype(int)

    # Hack away zero width objects
    boxes = boxes.to_xywh(copy=False)
    boxes.data[..., 2:4] = np.maximum(boxes.data[..., 2:4], 1)

    gw, gh = image_size
    dims = (gh, gw)

    # This is 2x as fast for image_size=(300,300)
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

    tlbr_boxes = boxes.to_ltrb().data
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
