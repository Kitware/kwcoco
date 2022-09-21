#!/usr/bin/env python
from os.path import dirname
import ubelt as ub
import scriptconfig as scfg


class CocoUnionCLI(object):
    name = 'union'

    class CLIConfig(scfg.Config):
        """
        Combine multiple COCO datasets into a single merged dataset.
        """
        default = {
            'src': scfg.Value([], nargs='+', help='path to multiple input datasets', position=1),
            'dst': scfg.Value('combo.kwcoco.json', help='path to output dataset'),
            'absolute': scfg.Value(False, isflag=1, help='if True, converts paths to absolute paths before doing union')
        }
        epilog = """
        Example Usage:
            kwcoco union --src special:shapes8 special:shapes1 --dst=combo.kwcoco.json
        """

    @classmethod
    def main(cls, cmdline=True, **kw):
        """
        Example:
            >>> from kwcoco.cli.coco_union import *  # NOQA
            >>> import ubelt as ub
            >>> dpath = ub.Path.appdir('kwcoco/tests/cli/union').ensuredir()
            >>> dst_fpath = dpath / 'combo.kwcoco.json'
            >>> kw = {
            >>>     'src': ['special:shapes8', 'special:shapes1'],
            >>>     'dst': dst_fpath
            >>> }
            >>> cmdline = False
            >>> cls = CocoUnionCLI
            >>> cls.main(cmdline, **kw)
        """
        import kwcoco
        config = cls.CLIConfig(kw, cmdline=cmdline)
        print('config = {}'.format(ub.repr2(dict(config), nl=1)))

        if config['src'] is None:
            raise Exception('must specify sources: {}'.format(config['src']))

        if len(config['src']) == 0:
            raise ValueError('Must provide at least one input dataset')

        datasets = []
        for fpath in ub.ProgIter(config['src'], desc='reading datasets',
                                 verbose=1):
            print('reading fpath = {!r}'.format(fpath))
            dset = kwcoco.CocoDataset.coerce(fpath)

            if config['absolute']:
                dset.reroot(absolute=True)

            datasets.append(dset)

        combo = kwcoco.CocoDataset.union(*datasets)

        out_fpath = config['dst']
        out_dpath = dirname(out_fpath)
        if out_dpath:
            ub.ensuredir(out_dpath)
        print('Writing to out_fpath = {!r}'.format(out_fpath))
        combo.fpath = out_fpath
        combo.dump(combo.fpath, newlines=True)

_CLI = CocoUnionCLI

if __name__ == '__main__':
    """
    CommandLine:
        python -m kwcoco.cli.coco_union
    """
    _CLI._main()
