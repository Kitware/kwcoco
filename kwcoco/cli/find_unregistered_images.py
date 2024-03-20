#!/usr/bin/env python3
"""

TODO:
    - [ ] Should this be renamed to find_unregsitered_assets?

Example:
    >>> # xdoctest: +REQUIRES(module:kwutil)
    >>> import ubelt as ub
    >>> import kwcoco
    >>> dpath = ub.Path.appdir('kwcoco/tests/cli/find_unregistered_images')
    >>> dpath.delete().ensuredir()
    >>> # Create a fresh dataset, and unregister 2 images
    >>> dset = kwcoco.CocoDataset.demo('vidshapes1', image_size=(8, 8), num_frames=10, dpath=dpath)
    >>> image_ids = list(dset.images())
    >>> dset.remove_images(list(ub.take(image_ids, [2, 5])))
    >>> dset.dump()
    >>> # Check that the original 10 images existed
    >>> assert len(list(ub.Path(dset.bundle_dpath).glob('**/*.png'))) == 10
    >>> from kwcoco.cli.find_unregistered_images import FindUnregisteredImagesCLI
    >>> cls = FindUnregisteredImagesCLI
    >>> cmdline = 0
    >>> #
    >>> # Calling FindUnregisteredImage with action=list just prints
    >>> kwargs = dict(src=dset.fpath, action='list')
    >>> cls.main(cmdline=cmdline, **kwargs)
    >>> assert len(list(ub.Path(dset.bundle_dpath).glob('**/*.png'))) == 10
    >>> # Calling FindUnregisteredImage with action=list with verbose=0 has pipeable output
    >>> kwargs = dict(src=dset.fpath, action='list', verbose=0)
    >>> cls.main(cmdline=cmdline, **kwargs)
    >>> assert len(list(ub.Path(dset.bundle_dpath).glob('**/*.png'))) == 10
    >>> #
    >>> # Calling FindUnregisteredImage with action=delete removes the data
    >>> kwargs = dict(src=dset.fpath, action='delete')
    >>> cls.main(cmdline=cmdline, **kwargs)
    >>> assert len(list(ub.Path(dset.bundle_dpath).glob('**/*.png'))) == 8
    >>> # Cleanup
    >>> dpath.delete()
"""
import scriptconfig as scfg
import ubelt as ub


class FindUnregisteredImagesCLI(scfg.DataConfig):
    """
    Find images in a kwcoco bundle that are not registered in a kwcoco file.

    Based on the value of "action" list these images or delete them.
    """
    __command__ = 'find_unregistered_images'

    src = scfg.Value(None, nargs='+', help='all kwcoco paths that register data in this bundle', position=1)

    image_dpath = scfg.Value(None, help=ub.paragraph(
        '''
        if specified, only the specified path(s) will have unregistered images
        removed, otherwise this argument will be inferred as the bundle
        directories belonging to the specified kwcoco files.
        '''))

    action = scfg.Value('ask', help='What to do when an unregistered images is found.', choices=['ask', 'delete', 'list'])

    io_workers = scfg.Value('avail', help='number of io workers')

    verbose = scfg.Value(1, help='set to 0 for quiet pipeable output')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from kwcoco.cli.find_unregistered_images import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict()
            >>> cls = FindUnregisteredImagesCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        from kwcoco.util.util_rich import rich_print
        if config.verbose:
            rich_print('config = ' + ub.urepr(config, nl=1))
        import os
        import kwcoco
        from kwutil import util_path
        fpaths = util_path.coerce_patterned_paths(config.src)

        if config.verbose:
            rich_print('Will read fpaths = {}'.format(ub.urepr(fpaths, nl=1)))
        datasets = list(kwcoco.CocoDataset.coerce_multiple(
            fpaths, workers=config.io_workers, verbose=config.verbose))

        if config.image_dpath is None:
            image_dpaths = [ub.Path(d.bundle_dpath).absolute() for d in datasets]
        else:
            image_dpaths = util_path.coerce_patterned_paths(config.image_dpath)
            image_dpaths = [d.absolute() for d in image_dpaths]

        image_dpaths = list(ub.unique(image_dpaths))
        if len(image_dpaths) > 1:
            # Should we remove this restriction?
            raise NotImplementedError(ub.paragraph(
                '''
                The case for more than 1 image dpath has not been tested. It
                might work, if you remove this error. Please submit an MR if
                that is the case
                '''))

        fpath_sets = find_unregistered_images(datasets, image_dpaths)
        fpath_sets = ub.udict(fpath_sets)
        fpath_set_sizes = fpath_sets.map_values(len)
        if config.verbose:
            rich_print('fpath_set_sizes = ' + ub.urepr(fpath_set_sizes, align=':'))

        unregistered_fpaths = fpath_sets['unregistered']

        if len(unregistered_fpaths) > 0:
            import rich.prompt
            if config.verbose:
                rich_print('unregistered_fpaths = {}'.format(ub.urepr([os.fspath(f) for f in unregistered_fpaths], nl=1)))
                rich_print('fpath_set_sizes = ' + ub.urepr(fpath_set_sizes, align=':'))
            else:
                for f in unregistered_fpaths:
                    print(os.fspath(f))

            action = config.action
            if action == 'ask':
                ans = rich.prompt.Confirm.ask(f'Delete these {len(unregistered_fpaths)} unregistered files?')
                action = 'delete' if ans else 'list'

            if action == 'list':
                ...
            elif action == 'delete':
                # ACTUALLY DELETE
                for p in ub.ProgIter(unregistered_fpaths, desc='deleting'):
                    p.delete()
                # Find and remove empty directories
                for image_dpath in image_dpaths:
                    _remove_empty_dirs(image_dpath)
            else:
                raise KeyError(action)
        else:
            print('No unregistered files')


def find_unregistered_images(datasets, image_dpaths):
    # Enumerate all unique images registered by the datasets
    all_registered_fpaths = []
    for dset in datasets:
        all_registered_fpaths.extend(_check_registered(dset))

    # Enumerate all unique images in the image directories
    all_image_fpaths = []
    for image_dpath in image_dpaths:
        all_image_fpaths.extend(_find_existing_images(image_dpath))

    # Check overlaps
    registered_fpaths = set(all_registered_fpaths)
    existing_fpaths = set(all_image_fpaths)

    missing_fpaths = registered_fpaths - existing_fpaths
    unregistered_fpaths = existing_fpaths - registered_fpaths

    fpath_sets = {
        'registered': registered_fpaths,
        'existing': existing_fpaths,
        'missing': missing_fpaths,
        'unregistered': unregistered_fpaths,
    }
    return fpath_sets


def _check_registered(dset):
    """
    Build list of paths registered in the coco file
    """
    registered_paths = []
    for gid in dset.images():
        coco_img = dset.coco_image(gid)
        registered_paths.extend(list(coco_img.iter_image_filepaths()))
    registered_paths = [ub.Path(p).absolute() for p in registered_paths]
    registered_dups = ub.find_duplicates(registered_paths)
    if registered_dups:
        print('ERROR: Duplicates')
        for fpath, idxs in registered_dups.items():

            found_dup_gids = []
            # No fast index for this.
            for gid in dset.images():
                coco_img = dset.coco_image(gid)
                paths = {ub.Path(p).absolute() for p in coco_img.iter_image_filepaths()}
                if fpath in paths:
                    found_dup_gids.append(gid)

            for gid in found_dup_gids:
                coco_img = dset.coco_image(gid)
                print('coco_img.video = {}'.format(ub.urepr(coco_img.video, nl=1)))
                print('coco_img.img = {}'.format(ub.urepr(coco_img.img, nl=1)))
                print(f'coco_img={coco_img}')
        raise AssertionError('Registered files have duplicates')
    return registered_paths


def _find_existing_images(image_dpath):
    """
    Find images in a directory
    """
    import kwimage
    existing_image_paths = []
    for r, ds, fs in image_dpath.walk():
        for f in fs:
            if f.lower().endswith(kwimage.im_io.IMAGE_EXTENSIONS):
                existing_image_paths.append(r / f)
    existing_image_paths = [p.absolute() for p in existing_image_paths]
    assert not ub.find_duplicates(existing_image_paths)
    return existing_image_paths


def _remove_empty_dirs(dpath):
    empty_dpaths = True
    while empty_dpaths:
        empty_dpaths = []
        for r, ds, fs in dpath.walk():
            if not ds and not fs:
                empty_dpaths.append(r)
        for d in empty_dpaths:
            d.rmdir()


__cli__ = FindUnregisteredImagesCLI
main = __cli__.main

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/kwcoco/kwcoco/cli/find_unregistered_images.py
        python -m kwcoco.cli.find_unregistered_images
    """
    main()
