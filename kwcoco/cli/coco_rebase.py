#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ubelt as ub
import scriptconfig as scfg


class CocoRebaseCLI:
    """

    NOT WORKING YET. NEED TO WORK THROUGH DETAILS IN A CASE-BY-CASE BASIS

    What happens when:

        * Existing file paths are absolute, but are not correct.

        * Existing file paths are absolute, and correct.

        * Existing file paths are relative, but the original directory is
          unknown, but the new image root correctly places them.

        * todo: enumerate the rest
    """
    name = 'rebase'

    class CLIConfig(scfg.Config):
        """
        Rebase image paths onto a new image root.
        """
        epilog = """
        Example Usage:
            kwcoco rebase --help
            kwcoco rebase --src=special:shapes8 --dst rebased.json
            kwcoco rebase --src=special:shapes8 --img_root=foo --check=True --dst rebased.json
        """
        default = {
            'src': scfg.Value(None, help=(
                'Path to the coco dataset')),

            'img_root': scfg.Value(None, help=(
                'Path to the new image root.')),

            'absolute': scfg.Value(True, help=(
                'If False, the output file uses relative paths')),

            'check': scfg.Value(True, help=(
                'If True, checks that all data exists')),

            'dst': scfg.Value(None, help=(
                'Save the rebased dataset to a new file')),
        }

    @classmethod
    def main(cls, cmdline=True, **kw):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> kw = {'src': 'special:shapes8'}
            >>> cmdline = False
            >>> cls = CocoRebaseCLI
            >>> cls.main(cmdline, **kw)
        """
        import kwcoco
        config = cls.CLIConfig(kw, cmdline=cmdline)
        print('config = {}'.format(ub.repr2(dict(config), nl=1)))

        if config['src'] is None:
            raise Exception('must specify source: '.format(config['src']))

        dset = kwcoco.CocoDataset.coerce(config['src'])

        dset.rebase(
            img_root=config['img_root'],
            absolute=config['absolute'],
            check=config['check']
        )

        dset.dump(config['dst'], newlines=True)


_CLI = CocoRebaseCLI

if __name__ == '__main__':
    _CLI.main()
