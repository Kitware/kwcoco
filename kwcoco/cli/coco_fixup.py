#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub
"""
TODO:
    - [ ] Port logic from validate.
    - [ ] Handle annotations with duplicate ids.
        * Case 1: if all other information in the annotation is eactly the same, then delete all but one of the annotations.
        * Case 2: if all other information in the annotation is diffrent, keep both and give the duplicates new ids.
"""


class CocoFixup(scfg.DataConfig):
    __command__ = 'fixup'

    src = scfg.Value(None, position=1, help='path to input dataset')

    dst = scfg.Value(None, position=2, help='path to output dataset')

    missing_assets = scfg.Value(True, help='if True remove missing assets')

    corrupted_assets = scfg.Value(True, help='if True remove corrupted assets')

    inplace = scfg.Value(False, isflag=True, help=(
        'if True and dst is unspecified then the output will overwrite the input'))

    workers = scfg.Value(0)

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> from kwcoco.cli.coco_fixup import *  # NOQA
            >>> import kwcoco
            >>> import ubelt as ub
            >>> dpath = ub.Path.appdir('kwcoco/tests/coco_fixup')
            >>> dpath.delete().ensuredir()
            >>> dset = kwcoco.CocoDataset.demo('vidshapes2', image_size=(64, 64), rng=0, dpath=dpath)
            >>> fpath1 = dset.coco_image(1).primary_image_filepath()
            >>> fpath2 = dset.coco_image(2).primary_image_filepath()
            >>> fpath1.delete()  # remove an asset
            >>> fpath2.write_bytes(fpath2.read_bytes()[0::2])  # corrupt an asset
            >>> src = ub.Path(dset.fpath)
            >>> cmdline = 0
            >>> cls = CocoFixup
            >>> dst = src.augment(prefix='fixed-')
            >>> kwargs = dict(
            >>>     src=src,
            >>>     dst=dst,
            >>>     missing_assets=True,
            >>>     corrupted_assets=True,
            >>> )
            >>> cls.main(cmdline=cmdline, **kwargs)
            >>> assert dst.exists()
            >>> dst_dset = kwcoco.CocoDataset(dst)
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

        print('reading fpath = {!r}'.format(config['src']))
        import kwcoco
        dset = kwcoco.CocoDataset.coerce(config['src'])
        dst_fpath = ub.Path(config['dst'])

        workers = config.workers
        check_aux = True

        if config.missing_assets:
            remove_missing_assets(dset)

        if config.corrupted_assets:
            find_and_remove_corrupted_assets(dset, check_aux, workers)

        remove_empty_videos(dset)

        # corrupted = dset.corrupted_images(check_aux=True, verbose=verbose,
        #                                   workers=config.get('workers', 0))
        dset.fpath = dst_fpath
        print(f'Write dset.fpath={dset.fpath}')
        dset.dump()


def find_and_remove_corrupted_assets(dset, check_aux=True, workers=0):
    from kwcoco._helpers import _image_corruption_check

    jobs = ub.JobPool(mode='process', max_workers=workers)

    bundle_dpath = ub.Path(dset.bundle_dpath)

    img_enum = enumerate(dset.dataset['images'])
    for img_idx, img in ub.ProgIter(img_enum,
                                    total=len(dset.dataset['images']),
                                    desc='submit corruption checks',
                                    verbose=1):
        gid = img.get('id', None)
        fname = img.get('file_name', None)
        if fname is not None:
            gpath = bundle_dpath / fname
            job = jobs.submit(_image_corruption_check, gpath)
            job.input_info = (img_idx, gpath, gid)

        if check_aux:
            for asset_idx, aux in img.get('auxiliary', []):
                gpath = bundle_dpath / aux['file_name']
                job = jobs.submit(_image_corruption_check, gpath)
                job.input_info = (img_idx, gpath, gid, 'auxiliary', asset_idx)
            for asset_idx, aux in enumerate(img.get('assets', [])):
                gpath = bundle_dpath / aux['file_name']
                job = jobs.submit(_image_corruption_check, gpath)
                job.input_info = (img_idx, gpath, gid, 'assets', asset_idx)

    corrupted_info = []
    for job in jobs.as_completed(desc='check corrupted images',
                                 progkw={'verbose': 1}):
        info = job.result()
        if info['failed']:
            corrupted_info.append(job.input_info)

    gid_to_missing = ub.group_items(corrupted_info, key=lambda t: t[2])
    empty_gids = []
    for gid, missing in ub.ProgIter(gid_to_missing.items(), desc='removing corrupted'):
        coco_img = dset.coco_image(gid)
        remove_main = coco_img_remove_empty_assets(coco_img, missing)
        if remove_main or coco_img.n_assets == 0:
            empty_gids.append(gid)

    if empty_gids:
        print(f'Found {len(empty_gids)} images without assets, removing')
        dset.remove_images(empty_gids, verbose=3)


def coco_img_remove_empty_assets(coco_img, missing):
    from os.path import join
    img = coco_img.img

    missing_fpaths = {t[1] for t in missing}
    to_remove_assets = []
    to_remove_auxiliary = []
    remove_main = False

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
        if remove_main or coco_img.n_assets == 0:
            empty_gids.append(gid)

    if empty_gids:
        print(f'Found {len(empty_gids)} images without assets, removing')
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
