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

    src = scfg.Value(None, position=1, help='path to input dataset')

    dst = scfg.Value(None, position=1, help='path to output dataset')

    missing_assets = scfg.Value(False, help='if True remove missing assets')

    corrupted_assets = scfg.Value(False, help='if True remove corrupted assets')

    inplace = scfg.Value(False, isflag=True, help=(
        'if True and dst is unspecified then the output will overwrite the input'))

    workers = scfg.Value(0)

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from kwcoco.cli.coco_fixup import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict()
            >>> cls = CocoFixupCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
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

        # remove_missing_assets(dset)
        # corrupted = dset.corrupted_images(check_aux=True, verbose=verbose,
        #                                   workers=config.get('workers', 0))


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
        remove_main['file_name'] = None
        remove_main['channels'] = None
        raise NotImplementedError

    auxiliary = img.get('auxiliary', [])
    for idx in sorted(to_remove_auxiliary)[::-1]:
        del auxiliary[idx]

    assets = img.get('assets', [])
    for idx in sorted(to_remove_assets)[::-1]:
        del assets[idx]


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
        coco_img_remove_empty_assets(coco_img, missing)
        if coco_img.n_assets == 0:
            empty_gids.append(gid)

    if empty_gids:
        print(f'Found {len(empty_gids)} images without assets, removing')
        dset.remove_images(empty_gids, verbose=3)

    videos = dset.videos()
    empty_vidids = [v for v, gids in zip(videos, videos.images) if len(gids) == 0]
    if empty_vidids:
        dset.remove_videos(empty_gids, verbose=3)


__cli__ = CocoFixup
main = __cli__.main

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/kwcoco/kwcoco/cli/coco_fixup.py
        python -m kwcoco.cli.coco_fixup
    """
    main()
