#!/usr/bin/env python
import ubelt as ub
import scriptconfig as scfg


class CocoRerootCLI:
    name = 'reroot'

    class CocoRerootConfig(scfg.DataConfig):
        """
        Reroot image paths onto a new image root.

        Modify the root of a coco dataset such to either make paths relative to a
        new root or make paths absolute.

        TODO:
            - [ ] Evaluate that all tests cases work
        """

        __epilog__ = """

        Example Usage:
            kwcoco reroot --help
            kwcoco reroot --src=special:shapes8 --dst rerooted.json
            kwcoco reroot --src=special:shapes8 --new_prefix=foo --check=True --dst rerooted.json
        """
        src = scfg.Value(None, help=(
            'Input coco dataset path'), position=1)

        dst = scfg.Value(None, help=(
            'Output coco dataset path'), position=2)

        new_prefix = scfg.Value(None, help=(
            'New prefix to insert before every image file name.'))

        old_prefix = scfg.Value(None, help=(
            'Old prefix to remove from the start of every image file name.'))

        absolute = scfg.Value(True, help=(
            'If False, the output file uses relative paths'))

        check = scfg.Value(True, help=(
            'If True, checks that all data exists'))

        autofix = scfg.Value(False, isflag=True, help=(
            ub.paragraph(
                '''
                If True, attempts an automatic fix. This assumes that paths
                are prefixed with an absolute path belonging to a different
                machine, and it attempts to strip off a minimal prefix to
                find relative paths that do exist.
                ''')))

        compress = scfg.Value('auto', help='if True writes results with compression. DEPRECATED. Just use a .zip suffix.')

        inplace = scfg.Value(False, isflag=True, help=(
            'if True and dst is unspecified then the output will overwrite the input'))

    CLIConfig = CocoRerootConfig

    @classmethod
    def main(cls, cmdline=True, **kw):
        r"""
        Example:
            >>> # xdoctest: +SKIP
            >>> kw = {'src': 'special:shapes8'}
            >>> cmdline = False
            >>> cls = CocoRerootCLI
            >>> cls.main(cmdline, **kw)

        Ignore:
            python ~/code/kwcoco/kwcoco/cli/coco_reroot.py  \
                --src=$HOME/code/bioharn/fast_test/deep_training/training_truth.json \
                --dst=$HOME/code/bioharn/fast_test/deep_training/training_truth2.json \
                --new_prefix=/home/joncrall/code/bioharn/fast_test/training_data \
                --old_prefix=/run/media/matt/Storage/TEST/training_data

            python ~/code/kwcoco/kwcoco/cli/coco_reroot.py  \
                --src=$HOME/code/bioharn/fast_test/deep_training/validation_truth.json \
                --dst=$HOME/code/bioharn/fast_test/deep_training/validation_truth2.json \
                --new_prefix=/home/joncrall/code/bioharn/fast_test/training_data \
                --old_prefix=/run/media/matt/Storage/TEST/training_data

            cmdline = '''
                --src=$HOME/code/bioharn/fast_test/deep_training/training_truth.json
                --dst=$HOME/code/bioharn/fast_test/deep_training/training_truth2.json
                --new_prefix=/home/joncrall/code/bioharn/fast_test/training_data
                --old_prefix=/run/media/matt/Storage/TEST/training_data
            '''
                /run/media/matt/Storage/TEST/training_data
                --check=True --dst rerooted.json
        """
        import kwcoco
        from os.path import dirname, abspath
        config = cls.CLIConfig.cli(data=kw, cmdline=cmdline)

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

        dset = kwcoco.CocoDataset.coerce(config['src'])

        new_root = abspath(dirname(config['dst']))

        if config['absolute']:
            new_root = abspath(new_root)

        if config['autofix']:
            autfixer = find_reroot_autofix(dset)
            if autfixer is not None:
                print('Found autfixer = {}'.format(ub.urepr(autfixer, nl=1)))
                config['new_prefix'] = autfixer['new_prefix']
                config['old_prefix'] = autfixer['old_prefix']

        dset.reroot(
            new_root=new_root,
            new_prefix=config['new_prefix'],
            old_prefix=config['old_prefix'],
            absolute=config['absolute'],
            check=config['check']
        )

        dset.fpath = config['dst']
        print('dump dset.fpath = {!r}'.format(dset.fpath))
        dumpkw = {
            'newlines': True,
            'compress': config['compress'],
        }
        dset.dump(dset.fpath, **dumpkw)


def find_reroot_autofix(dset):
    import os
    # Given a set of missing images, is there a way we can autofix them?
    missing_tups = dset.missing_images()
    missing_gpaths = [t[1] for t in missing_tups]
    chosen = None
    if len(missing_gpaths) == 0:
        print('All paths look like they exist')
        return None

    print(f'Found {len(missing_tups)} missing images')
    if len(missing_gpaths) > 0:
        bundle_dpath = ub.Path(dset.bundle_dpath)
        print('bundle_dpath = {}'.format(ub.urepr(bundle_dpath, nl=1)))
        first = ub.Path(missing_gpaths[0])
        print('first = {}'.format(ub.urepr(first, nl=1)))
        first_parts = first.parts
        print('first_parts = {}'.format(ub.urepr(first_parts, nl=1)))
        candidates = []
        for i in range(len(first_parts)):
            cand_path = bundle_dpath / ub.Path(*first_parts[i:])
            if cand_path.exists():
                candidates.append({
                    'old_prefix': os.fspath(ub.Path(*first_parts[:i])) + '/',
                    'new_prefix': '',
                })
        if len(candidates) == 0:
            raise RuntimeError('Could not determine a valid autofix')

        # Check that the fix fixes everything or dont do it.
        for candidate in candidates:
            old_pref = os.fspath(candidate['old_prefix'])
            new_pref = os.fspath(candidate['new_prefix'])

            any_missing = False
            for gpath in missing_gpaths:
                new_gpath = bundle_dpath / ub.Path(os.fspath(gpath).replace(old_pref, new_pref))
                if not new_gpath.exists():
                    any_missing = True
                    break

            if any_missing:
                continue
            chosen = candidate

    if not chosen:
        raise RuntimeError('No candidate fixed all paths')
    return chosen


_CLI = CocoRerootCLI

if __name__ == '__main__':
    _CLI.main()
