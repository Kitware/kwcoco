#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub
"""
TODO:
    - [ ] Port logic from validate.
    - [ ] Handle annotations with duplicate ids.
        * Case 1: if all other information in the annotation is exactly the same, then delete all but one of the annotations.
        * Case 2: if all other information in the annotation is different, keep both and give the duplicates new ids.
"""


class CocoFixup(scfg.DataConfig):
    __command__ = 'fixup'

    src = scfg.Value(None, position=1, help='path to input dataset')

    dst = scfg.Value(None, position=2, help='path to output dataset')

    missing_assets = scfg.Value(True, help='if True remove missing assets')

    corrupted_assets = scfg.Value(True, help=('if True remove corrupted assets. '
                                              'Can also be only_shape for a quicker check.'))

    inplace = scfg.Value(False, isflag=True, help=(
        'if True and dst is unspecified then the output will overwrite the input'))

    workers = scfg.Value(0)

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        CommandLine:
            xdoctest -m kwcoco.cli.coco_fixup CocoFixup.main

        Example:
            >>> from kwcoco.cli.coco_fixup import *  # NOQA
            >>> import kwcoco
            >>> import ubelt as ub
            >>> dpath = ub.Path.appdir('kwcoco/tests/coco_fixup')
            >>> dpath.delete().ensuredir()
            >>> src_dset = kwcoco.CocoDataset.demo('vidshapes2', image_size=(64, 64), rng=0, dpath=dpath)
            >>> fpath1 = src_dset.coco_image(1).primary_image_filepath()
            >>> fpath2 = src_dset.coco_image(2).primary_image_filepath()
            >>> fpath1.delete()  # remove an asset
            >>> fpath2.write_bytes(fpath2.read_bytes()[0::2])  # corrupt an asset
            >>> src = ub.Path(src_dset.fpath)
            >>> print(f'src_dset={src_dset}')
            >>> dst = src.augment(prefix='fixed-')
            >>> kwargs = dict(src=src, dst=dst)
            >>> cmdline = 0
            >>> cls = CocoFixup
            >>> cls.main(cmdline=cmdline, **kwargs)
            >>> assert dst.exists()
            >>> dst_dset = kwcoco.CocoDataset(dst)
            >>> print(f'dst_dset={dst_dset}')
            >>> assert dst_dset.n_images < src_dset.n_images
            >>> assert dst_dset.n_videos < src_dset.n_videos
        """
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)

        try:
            import rich
        except ImportError:
            rich_print = print
        else:
            rich_print = rich.print
        rich_print('config = {}'.format(ub.urepr(config, nl=1)))

        if config['src'] is None:
            raise Exception('must specify source: {}'.format(config['src']))
        if config['dst'] is None:
            if config['inplace']:
                config['dst'] = config['src']
            else:
                raise ValueError('must specify dst: {}'.format(config['dst']))

        try:
            from osgeo import gdal
            gdal.UseExceptions()
        except Exception:
            ...

        print('reading fpath = {!r}'.format(config['src']))
        import kwcoco
        dset = kwcoco.CocoDataset.coerce(config['src'])
        dst_fpath = ub.Path(config['dst'])

        workers = config.workers
        check_aux = True

        if config.missing_assets:
            remove_missing_assets(dset)

        if config.corrupted_assets:
            find_and_remove_corrupted_assets(dset, check_aux, workers, config.corrupted_assets)

        remove_empty_videos(dset)

        # corrupted = dset.corrupted_images(check_aux=True, verbose=verbose,
        #                                   workers=config.get('workers', 0))
        dset.fpath = dst_fpath
        print(f'Write dset.fpath={dset.fpath}')
        dset.dump()


def find_corrupted_assets(dset, check_aux=True, workers=0,
                          corrupted_assets='full'):
    from kwcoco._helpers import _image_corruption_check

    if corrupted_assets is True:
        corrupted_assets = 'full'

    if corrupted_assets == 'only_shape':
        only_shape = True
    elif corrupted_assets == 'full':
        only_shape = False
    else:
        raise Exception

    jobs = ub.JobPool(mode='process', max_workers=workers)

    bundle_dpath = ub.Path(dset.bundle_dpath)
    verbose = 3

    # imread_kwargs = dict(backend='auto')
    # imread_kwargs = dict(backend='gdal')
    # imread_kwargs = dict(backend='gdal', overview='coarsest')
    imread_kwargs = {}

    img_enum = enumerate(dset.dataset['images'])
    for img_idx, img in ub.ProgIter(img_enum,
                                    total=len(dset.dataset['images']),
                                    desc='submit corruption checks',
                                    verbose=verbose):
        gid = img.get('id', None)
        fname = img.get('file_name', None)
        if fname is not None:
            gpath = bundle_dpath / fname
            job = jobs.submit(_image_corruption_check, gpath,
                              imread_kwargs=imread_kwargs,
                              only_shape=only_shape)
            job.input_info = (img_idx, gpath, gid)

        if check_aux:
            for asset_idx, aux in img.get('auxiliary', []):
                gpath = bundle_dpath / aux['file_name']
                job = jobs.submit(_image_corruption_check, gpath, imread_kwargs=imread_kwargs)
                job.input_info = (img_idx, gpath, gid, 'auxiliary', asset_idx)
            for asset_idx, aux in enumerate(img.get('assets', [])):
                gpath = bundle_dpath / aux['file_name']
                job = jobs.submit(_image_corruption_check, gpath, imread_kwargs=imread_kwargs)
                job.input_info = (img_idx, gpath, gid, 'assets', asset_idx)

    corrupted_info = []
    for job in jobs.as_completed(desc='check corrupted images',
                                 progkw={'verbose': verbose}):
        info = job.result()
        if info['failed']:
            corrupted_info.append(job.input_info)

    return corrupted_info


def find_and_remove_corrupted_assets(dset, check_aux=True, workers=0,
                                     corrupted_assets='full'):

    # FIND PART
    corrupted_info = find_corrupted_assets(
        dset, check_aux=check_aux, workers=workers,
        corrupted_assets=corrupted_assets)

    # REMOVE PART
    gid_to_missing = ub.group_items(corrupted_info, key=lambda t: t[2])
    # print(f'gid_to_missing = {ub.urepr(gid_to_missing, nl=1)}')
    empty_gids = []
    for gid, missing in ub.ProgIter(gid_to_missing.items(), desc='removing corrupted', verbose=3):
        coco_img = dset.coco_image(gid)
        remove_main = coco_img_remove_empty_assets(coco_img, missing)
        # print(f'coco_img={coco_img}')
        # print(f'remove_main={remove_main}')
        # print(f'coco_img.n_assets={coco_img.n_assets}')
        if remove_main or coco_img.n_assets == 0:
            empty_gids.append(gid)

    if empty_gids:
        # print(f'Found {len(empty_gids)} images without assets, removing')
        # print(f'empty_gids={empty_gids}')
        dset.remove_images(empty_gids, verbose=3)


def coco_img_remove_empty_assets(coco_img, missing):
    import os
    from os.path import join
    img = coco_img.img

    missing_fpaths = {os.fspath(t[1]) for t in missing}
    to_remove_assets = []
    to_remove_auxiliary = []
    remove_main = False
    # print(f'missing_fpaths = {ub.urepr(missing_fpaths, nl=1)}')

    # TODO: Better API for asset removal, this is hacked to deal with
    # current issues
    has_base_image = img.get('file_name', None) is not None
    if has_base_image:
        fpath = join(coco_img.bundle_dpath, img['file_name'])
        if fpath in missing_fpaths:
            remove_main = True

    for idx, obj in enumerate(img.get('auxiliary', None) or []):
        fpath = join(coco_img.bundle_dpath, obj['file_name'])
        if fpath in missing_fpaths:
            to_remove_auxiliary.append(idx)

    for idx, obj in enumerate(img.get('assets', None) or []):
        fpath = join(coco_img.bundle_dpath, obj['file_name'])
        if fpath in missing_fpaths:
            to_remove_assets.append(idx)

    if remove_main:
        img['file_name'] = None

    # print(f'remove_main={remove_main}')
    # print(f'to_remove_auxiliary = {ub.urepr(to_remove_auxiliary, nl=1)}')
    # print(f'to_remove_assets = {ub.urepr(to_remove_assets, nl=1)}')
    auxiliary = img.get('auxiliary', [])
    for idx in sorted(to_remove_auxiliary)[::-1]:
        del auxiliary[idx]

    assets = img.get('assets', [])
    for idx in sorted(to_remove_assets)[::-1]:
        del assets[idx]

    return remove_main


def remove_empty_videos(dset):
    videos = dset.videos()
    empty_vidids = [v for v, gids in zip(videos, videos.images) if len(gids) == 0]
    if empty_vidids:
        dset.remove_videos(empty_vidids, verbose=3)


def remove_missing_assets(dset):
    """
    Remove the entire image if no assets remain
    Handle asset / auxiliary dict
    """
    flat_missing = dset.missing_images(check_aux=True)
    gid_to_missing = ub.group_items(flat_missing, key=lambda t: t[2])
    empty_gids = []
    for gid, missing in ub.ProgIter(gid_to_missing.items(), desc='removing missing'):
        coco_img = dset.coco_image(gid)
        remove_main = coco_img_remove_empty_assets(coco_img, missing)
        # print(f'remove_main={remove_main}')
        # print(f'coco_img.n_assets={coco_img.n_assets}')
        if remove_main or coco_img.n_assets == 0:
            empty_gids.append(gid)
    # print(f'empty_gids={empty_gids}')

    if empty_gids:
        # print(f'Found {len(empty_gids)} images without assets, removing')
        # print(f'empty_gids={empty_gids}')
        dset.remove_images(empty_gids, verbose=3)


__cli__ = CocoFixup
main = __cli__.main

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/kwcoco/kwcoco/cli/coco_fixup.py
        python -m kwcoco.cli.coco_fixup
    """
    main()
