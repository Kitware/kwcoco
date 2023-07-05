#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class CocoMove(scfg.DataConfig):
    """
    Move a kwcoco file to a new location while maintaining relative paths.
    This is equivalent to a regular copy followed by ``kwcoco reroot`` followed
    by a delete of the original.

    TODO: add option to move the assets as well?
    """
    __command__ = 'move'
    __alias__ = ['mv']

    src = scfg.Value(None, help='coco file to move', position=1)
    dst = scfg.Value(None, help='new location to move to', position=2)

    absolute = scfg.Value(False, help=('If False, the output file uses relative paths'))

    check = scfg.Value(True, help=(
        'If True, checks that all data exists'))

    @classmethod
    def main(CocoMove, cmdline=1, **kwargs):
        """
        Example:
            >>> import ubelt as ub
            >>> from kwcoco.cli import coco_move
            >>> import kwcoco
            >>> dpath = ub.Path.appdir('kwcoco/doctest/move')
            >>> dpath.delete().ensuredir()
            >>> dset = kwcoco.CocoDataset.demo('vidshapes2', dpath=dpath)
            >>> cmdline = 0
            >>> dst = (ub.Path(dset.bundle_dpath) / 'new_dpath').ensuredir()
            >>> kwargs = dict(src=dset.fpath, dst=dst)
            >>> coco_move.CocoMove.main(cmdline=cmdline, **kwargs)
            >>> assert dst.exists()
            >>> assert not ub.Path(dset.fpath).exists()
        """
        try:
            from rich import print as rich_print
        except ImportError:
            rich_print = print

        config = CocoMove.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich_print('config = ' + ub.urepr(config, nl=1))
        import kwcoco
        print('loading = {}'.format(ub.urepr(config.src, nl=1)))
        dset = kwcoco.CocoDataset.coerce(config.src)

        old_fpath = ub.Path(dset.fpath)

        dst = ub.Path(config.dst)
        if dst.is_dir():
            new_fpath = dst / old_fpath.name
        else:
            new_fpath = dst

        print('old_fpath = {}'.format(ub.urepr(old_fpath, nl=1)))
        print('new_fpath = {}'.format(ub.urepr(new_fpath, nl=1)))

        if not new_fpath.parent.exists():
            raise Exception('Destination directory does not exist')

        dset.reroot(
            new_root=new_fpath.parent,
            absolute=config.absolute,
            check=config.check,
            verbose=3,
        )
        print(f'Finished reroot, saving to: {new_fpath}')
        dumpkw = {
            'newlines': True,
        }
        dset.fpath = new_fpath
        dset.dump(dset.fpath, **dumpkw)
        if old_fpath.resolve() != new_fpath.resolve():
            print(f'Removing old fpath: {old_fpath}')
            old_fpath.delete()


__config__ = CocoMove

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/kwcoco/kwcoco/cli/coco_move.py
        python -m kwcoco.cli.coco_move
    """
    __config__.main()
