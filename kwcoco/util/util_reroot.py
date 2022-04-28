"""
Rerooting is harder than you would think
"""
import ubelt as ub
import os


def special_reroot_single(dset, verbose=0):
    import kwcoco
    bundle_dpath = ub.Path(dset.bundle_dpath).absolute()
    resolved_bundle_dpath = ub.Path(dset.bundle_dpath).resolve()

    any_modified = []
    for gid in ub.ProgIter(dset.images(), desc='special reroot', verbose=verbose):
        coco_img: kwcoco.CocoImage = dset.coco_image(gid)
        for obj in coco_img.iter_asset_objs():
            old_fname = ub.Path(obj['file_name'])
            fpath = (bundle_dpath / old_fname)
            if fpath.exists():
                new_fname = resolve_relative_to(fpath, resolved_bundle_dpath)
                if new_fname != old_fname:
                    assert (bundle_dpath / new_fname).exists()
                    any_modified.append(f'{old_fname} -> {new_fname}')
                    obj['file_name'] = os.fspath(new_fname)
    return any_modified


def resolve_relative_to(path, dpath, strict=False):
    """
    Given a path, try to resolve its symlinks such that it is relative to the
    given dpath.

    Example:
        >>> from kwcoco.util.util_reroot import *  # NOQA
        >>> import os
        >>> def _symlink(self, target, verbose=0):
        >>>     return ub.Path(ub.symlink(target, self, verbose=verbose))
        >>> ub.Path._symlink = _symlink
        >>> #
        >>> # TODO: try to enumerate all basic cases
        >>> #
        >>> base = ub.Path.appdir('kwcoco/tests/reroot')
        >>> base.delete().ensuredir()
        >>> #
        >>> drive1 = (base / 'drive1').ensuredir()
        >>> drive2 = (base / 'drive2').ensuredir()
        >>> #
        >>> data_repo1 = (drive1 / 'data_repo1').ensuredir()
        >>> cache = (data_repo1 / '.cache').ensuredir()
        >>> real_file1 = (cache / 'real_file1').touch()
        >>> #
        >>> real_bundle = (data_repo1 / 'real_bundle').ensuredir()
        >>> real_assets = (real_bundle / 'assets').ensuredir()
        >>> #
        >>> # Symlink file outside of the bundle
        >>> link_file1 = (real_assets / 'link_file1')._symlink(real_file1)
        >>> real_file2 = (real_assets / 'real_file2').touch()
        >>> link_file2 = (real_assets / 'link_file2')._symlink(real_file2)
        >>> #
        >>> #
        >>> # A symlink to the data repo
        >>> data_repo2 = (drive1 / 'data_repo2')._symlink(data_repo1)
        >>> data_repo3 = (drive2 / 'data_repo3')._symlink(data_repo1)
        >>> data_repo4 = (drive2 / 'data_repo4')._symlink(data_repo2)
        >>> #
        >>> # A prediction repo TODO
        >>> pred_repo5 = (drive2 / 'pred_repo5').ensuredir()
        >>> #
        >>> # _ = ub.cmd(f'tree -a {base}', verbose=3)
        >>> #
        >>> fpaths = []
        >>> for r, ds, fs in os.walk(base, followlinks=True):
        >>>     for f in fs:
        >>>         if 'file' in f:
        >>>             fpath = ub.Path(r) / f
        >>>             fpaths.append(fpath)
        >>> #
        >>> #
        >>> dpath = real_bundle.resolve()
        >>> #
        >>> for path in fpaths:
        >>>     # print(f'{path}')
        >>>     # print(f'{path.resolve()=}')
        >>>     resolved_rel = resolve_relative_to(path, dpath)
        >>>     print('resolved_rel = {!r}'.format(resolved_rel))
    """
    try:
        resolved_abs = resolve_directory_symlinks(path)
        resolved_rel = resolved_abs.relative_to(dpath)
    except ValueError:
        if strict:
            raise
        else:
            return path
    return resolved_rel


def resolve_directory_symlinks(path):
    """
    Only resolve symlinks of directories, not the base file
    """
    return path.parent.resolve() / path.name
