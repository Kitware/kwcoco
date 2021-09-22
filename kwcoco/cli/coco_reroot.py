#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ubelt as ub
import scriptconfig as scfg


class CocoRerootCLI:
    name = 'reroot'

    class CLIConfig(scfg.Config):
        """
        Reroot image paths onto a new image root.

        Modify the root of a coco dataset such to either make paths relative to a
        new root or make paths absolute.

        TODO:
            - [ ] Evaluate that all tests cases work
        """

        epilog = """

        Example Usage:
            kwcoco reroot --help
            kwcoco reroot --src=special:shapes8 --dst rerooted.json
            kwcoco reroot --src=special:shapes8 --new_prefix=foo --check=True --dst rerooted.json
        """
        default = {
            'src': scfg.Value(None, help=(
                'Path to the coco dataset'), position=1),

            'dst': scfg.Value(None, help=(
                'Save the re-rooted dataset to a new file'), position=2),

            'new_prefix': scfg.Value(None, help=(
                'Path to the new image root.')),

            'old_prefix': scfg.Value(None, help=(
                'Previous root to remove.')),

            'absolute': scfg.Value(True, help=(
                'If False, the output file uses relative paths')),

            'check': scfg.Value(True, help=(
                'If True, checks that all data exists')),
        }

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
        config = cls.CLIConfig(kw, cmdline=cmdline)
        print('config = {}'.format(ub.repr2(dict(config), nl=1)))

        if config['src'] is None:
            raise Exception('must specify source: {}'.format(config['src']))
        if config['dst'] is None:
            raise Exception('must specify dest: {}'.format(config['dst']))

        dset = kwcoco.CocoDataset.coerce(config['src'])

        new_root = abspath(dirname(config['dst']))

        if config['absolute']:
            new_root = abspath(new_root)

        dset.reroot(
            new_root=new_root,
            new_prefix=config['new_prefix'],
            old_prefix=config['old_prefix'],
            absolute=config['absolute'],
            check=config['check']
        )

        dset.fpath = config['dst']
        print('dump dset.fpath = {!r}'.format(dset.fpath))
        dset.dump(dset.fpath, newlines=True)


_CLI = CocoRerootCLI

if __name__ == '__main__':
    _CLI.main()
