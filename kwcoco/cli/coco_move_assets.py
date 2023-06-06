#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class CocoMoveAssetsCLI(scfg.DataConfig):
    """
    Move assets and update corresponding kwcoco files as well

    This modifies the kwcoco files inplace.

    This operation is not atomic and if it is interupted then your kwcoco
    bundle may be put into a bad state.
    """
    src = scfg.Value('source asset file or folder')
    dst = scfg.Value('destination asset file or folder')
    io_workers = scfg.Value(0, help='io workers')
    coco_fpaths = scfg.Value([], nargs='+', help='coco files modified by the move operation')


def main(cmdline=1, **kwargs):
    """
    Example:
        >>> # xdoctest: +SKIP
        >>> cmdline = 0
        >>> kwargs = dict(
        >>>     coco_fpaths=['*_E.kwcoco.zip', '*_mae.kwcoco.zip'],
        >>>     src='./_assets/teamfeats',
        >>>     dst='./teamfeats/mae',
        >>>     io_workers='avail',
        >>> )

        >>> cmdline = 0
        >>> kwargs = dict(
        >>>     coco_fpaths=['*_M.kwcoco.zip', '*_rutgers_material_seg_v4.kwcoco.zip'],
        >>>     src='./_teamfeats',
        >>>     dst='./teamfeats/materials',
        >>>     io_workers='avail',
        >>> )
        >>> main(cmdline=cmdline, **kwargs)
    """
    import rich
    config = CocoMoveAssetsCLI.cli(cmdline=cmdline, data=kwargs, strict=True)
    rich.print('config = ' + ub.urepr(config, nl=1))
    from watch.utils import util_path
    import kwcoco
    import os
    coco_fpaths = util_path.coerce_patterned_paths(config.coco_fpaths)
    dsets = list(kwcoco.CocoDataset.coerce_multiple(coco_fpaths, workers=config.io_workers))

    src_path = ub.Path(config.src).absolute()
    dst_path = ub.Path(config.dst).absolute()

    impacted_assets = []

    # Determine which assets are impacted by the move
    for dset in dsets:
        for coco_img in dset.images().coco_images:
            for asset in coco_img.assets:
                asset_fpath = asset.image_filepath().absolute()
                if asset_fpath.is_relative_to(src_path):
                    asset.dset = dset
                    asset.image_id = coco_img['id']
                    impacted_assets.append(asset)
    print(f'Found {len(impacted_assets)} impacted assets')

    impacted_dsets = {}
    for asset in impacted_assets:
        impacted_dsets[id(asset.dset)] = asset.dset
    print(f'Found {len(impacted_dsets)} impacted datasets')

    # Modify the kwcoco files in memory
    for asset in impacted_assets:
        old_asset_fname = asset['file_name']
        old_asset_fpath = asset.image_filepath().absolute()
        fpath_rel_src = old_asset_fpath.relative_to(src_path)
        new_asset_fpath = dst_path / fpath_rel_src

        if ub.Path(old_asset_fname).is_absolute():
            new_asset_fname = new_asset_fpath
        else:
            bundle_dpath = ub.Path(asset._bundle_dpath).absolute()
            new_asset_fname = new_asset_fpath.relative_to(bundle_dpath)
        asset['file_name'] = os.fspath(new_asset_fname)

    # Move the assets
    src_path.move(dst_path)

    # Check that the kwcoco files are working
    for dset in dsets:
        assert not dset.missing_images()

    # Rewrite the kwcoco files
    for dset in impacted_dsets.values():
        dset.dump()
        ...

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/kwcoco/cli/coco_move_assets.py
        python -m kwcoco.cli.coco_move_assets
    """
    main()
