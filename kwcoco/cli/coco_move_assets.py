#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class CocoMoveAssetsCLI(scfg.DataConfig):
    """
    Move assets and update corresponding kwcoco files as well

    NOTE:
        The options: src and dst refer to folderes of asset, NOT kwcoco files.
        Think about this the same way you think about moving files.  All kwcoco
        files that reference the moved assets need to be specified so they can
        have their paths updated. Unspecified kwcoco files may break.

    This modifies the kwcoco files inplace.

    This operation is not atomic and if it is interupted then your kwcoco
    bundle may be put into a bad state.
    """
    src = scfg.Value('source asset file or folder')
    dst = scfg.Value('destination asset file or folder')
    io_workers = scfg.Value(0, help='io workers')
    coco_fpaths = scfg.Value([], nargs='+', help='coco files modified by the move operation')


class CocoMoveAssetManager:
    def __init__(self, coco_dsets):
        self.jobs = []
        self.coco_dsets = coco_dsets
        self.impacted_assets = None
        self.impacted_dsets = None
        self._previous_moves = []

    def submit(self, src, dst):
        """
        Enqueue a move operation, or mark that one has already occurred.

        If dst exists we assume the move has already been done, and we will
        update any coco files that were impacted by this, but not updated.

        Otherwise we assume src needs to be moved to dst.
        """
        if dst.exists():
            # Tell the manager that the src was already move to the dst, but
            # the kwcoco files may need to be updated.
            assert not src.exists()
            self._previous_moves.append({'src': src, 'dst': dst})
        else:
            assert src.exists()
            self.jobs.append({'src': src, 'dst': dst})

    def find_impacted(self):
        impacted_assets = []

        src_dst_pairs = set()
        for job in self.jobs:
            _s = job['src'].absolute()
            _d = job['dst'].absolute()
            src_dst_pairs.add((_s, _d))

        for job in self._previous_moves:
            _s = job['src'].absolute()
            _d = job['dst'].absolute()
            src_dst_pairs.add((_s, _d))

        # Determine which assets are impacted by the move
        for dset in ub.ProgIter(self.coco_dsets):
            for coco_img in ub.ProgIter(dset.images().coco_images):
                for asset in coco_img.assets:
                    asset_fpath = asset.image_filepath().absolute()
                    for _s, _d in src_dst_pairs:
                        if asset_fpath.is_relative_to(_s):
                            asset.dset = dset
                            asset.image_id = coco_img['id']
                            impacted_assets.append((asset, _s, _d))
                            break
        print(f'Found {len(impacted_assets)} impacted assets')

        impacted_dsets = {}
        for asset, _s, _d in impacted_assets:
            impacted_dsets[id(asset.dset)] = asset.dset
        print(f'Found {len(impacted_dsets)} impacted datasets')
        self.impacted_dsets = impacted_dsets
        self.impacted_assets = impacted_assets

    def modify_datasets(self):
        # Modify the kwcoco files in memory
        import os
        for asset, s, d in self.impacted_assets:
            old_asset_fname = asset['file_name']
            old_asset_fpath = asset.image_filepath().absolute()
            fpath_rel_src = old_asset_fpath.relative_to(s)
            new_asset_fpath = d / fpath_rel_src

            if ub.Path(old_asset_fname).is_absolute():
                new_asset_fname = new_asset_fpath
            else:
                bundle_dpath = ub.Path(asset._bundle_dpath).absolute()
                new_asset_fname = new_asset_fpath.relative_to(bundle_dpath)
            asset['file_name'] = os.fspath(new_asset_fname)

    def move_files(self):
        for job in ub.ProgIter(self.jobs, desc='moving files'):
            s = job['src'].absolute()
            d = job['dst'].absolute()
            s.move(d)

    def dump_datasets(self):
        # Check that the kwcoco files are working
        for dset in self.impacted_dsets.values():
            assert not dset.missing_images()

        # Rewrite the kwcoco files
        for dset in self.impacted_dsets.values():
            dset.dump()

    def run(self):
        self.find_impacted()
        self.modify_datasets()
        self.move_files()
        self.dump_datasets()


# def _devcheck():
#     import fsspec
#     fs = fsspec.filesystem('file', asynchronous=True)


def main(cmdline=1, **kwargs):
    """
    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(module:kwutil)
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
    from kwutil import util_path
    import kwcoco
    coco_fpaths = util_path.coerce_patterned_paths(config.coco_fpaths)
    dsets = list(kwcoco.CocoDataset.coerce_multiple(coco_fpaths, workers=config.io_workers))

    mv_man = CocoMoveAssetManager(dsets)
    mv_man.submit(config.src, config.dst)
    mv_man.run()


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/kwcoco/cli/coco_move_assets.py
        python -m kwcoco.cli.coco_move_assets
    """
    main()
