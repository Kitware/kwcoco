#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ubelt as ub
import scriptconfig as scfg


class CocoSplitCLI(object):
    name = 'split'

    class CLIConfig(scfg.Config):
        """
        Split a single COCO dataset into two sub-datasets.
        """
        default = {
            'src': scfg.Value(None, help='input dataset to split', position=1),
            'dst1': scfg.Value('split1.mscoco.json', help='output path1'),
            'dst2': scfg.Value('split2.mscoco.json', help='output path2'),
            'factor': scfg.Value(3, help='ratio of items put in dset1 vs dset2'),
            'rng': scfg.Value(None, help='random seed'),
        }
        epilog = """
        Example Usage:
            kwcoco split --src special:shapes8 --dst1=learn.mscoco.json --dst2=test.mscoco.json --factor=3 --rng=42
        """

    @classmethod
    def main(cls, cmdline=True, **kw):
        """
        Example:
            >>> kw = {'src': 'special:shapes8',
            >>>       'dst1': 'train.json', 'dst2': 'test.json'}
            >>> cmdline = False
            >>> cls = CocoSplitCLI
            >>> cls.main(cmdline, **kw)
        """
        import kwcoco
        import kwarray
        from kwcoco.util import util_sklearn

        config = cls.CLIConfig(kw, cmdline=cmdline)
        print('config = {}'.format(ub.repr2(dict(config), nl=1)))

        if config['src'] is None:
            raise Exception('must specify source: {}'.format(config['src']))

        print('reading fpath = {!r}'.format(config['src']))
        dset = kwcoco.CocoDataset.coerce(config['src'])
        annots = dset.annots()
        gids = annots.gids
        cids = annots.cids

        # Balanced category split
        rng = kwarray.ensure_rng(config['rng'])

        shuffle = rng is not None
        self = util_sklearn.StratifiedGroupKFold(n_splits=config['factor'],
                                                 random_state=rng,
                                                 shuffle=shuffle)
        split_idxs = list(self.split(X=gids, y=cids, groups=gids))
        idxs1, idxs2 = split_idxs[0]

        gids1 = sorted(ub.unique(ub.take(gids, idxs1)))
        gids2 = sorted(ub.unique(ub.take(gids, idxs2)))

        dset1 = dset.subset(gids1)
        dset2 = dset.subset(gids2)

        dset1.fpath = config['dst1']
        print('Writing dset1 = {!r}'.format(dset1.fpath))
        dset1.dump(dset1.fpath, newlines=True)

        dset2.fpath = config['dst2']
        print('Writing dset2 = {!r}'.format(dset2.fpath))
        dset2.dump(dset2.fpath, newlines=True)

_CLI = CocoSplitCLI

if __name__ == '__main__':
    _CLI.main()
