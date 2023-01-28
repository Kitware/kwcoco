#!/usr/bin/env python
import ubelt as ub
import scriptconfig as scfg


class CocoSplitCLI(object):
    """
    Splits a coco files into two parts base on some criteria.

    Useful for generating quick and dirty train/test splits, but in general
    users should opt for using ``kwcoco subset`` instead to explicitly
    construct these splits based on domain knowledge.
    """
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
            'balance_categories': scfg.Value(True, help='if True tries to balance annotation categories across splits'),
            'splitter': scfg.Value(
                'auto', help=ub.paragraph(
                    '''
                    Split method to use.
                    Using "image" will randomly assign each image to a partition.
                    Using "video" will randomly assign each video to a partition.
                    Using "auto" chooses "video" if there are any, otherwise "image".
                    '''), choices=['auto', 'image', 'video']),
            'compress': scfg.Value(False, help='if True writes results with compression'),
        }
        epilog = """
        Example Usage:
            kwcoco split --src special:shapes8 --dst1=learn.mscoco.json --dst2=test.mscoco.json --factor=3 --rng=42
        """

    @classmethod
    def main(cls, cmdline=True, **kw):
        """
        Example:
            >>> from kwcoco.cli.coco_split import *  # NOQA
            >>> import ubelt as ub
            >>> dpath = ub.Path.appdir('kwcoco/tests/cli/split').ensuredir()
            >>> kw = {'src': 'special:vidshapes8',
            >>>       'dst1': dpath / 'train.json',
            >>>       'dst2': dpath / 'test.json'}
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

        splitter = config['splitter']
        if splitter == 'auto':
            splitter = 'video' if dset.n_videos > 0 else 'image'

        images = dset.images()
        cids_per_image = images.annots.cids
        gids = images.lookup('id')

        if splitter == 'video':
            group_ids = images.lookup('video_id')
        elif splitter == 'image':
            group_ids = gids
        else:
            raise KeyError(splitter)

        final_group_ids = []
        final_group_gids = []
        final_group_cids = []

        unique_cids = set(ub.flatten(cids_per_image)) | {0}
        distinct_cid = max(unique_cids) + 11

        for group_id, gid, cids in zip(group_ids, gids, cids_per_image):
            if len(cids) == 0:
                final_group_ids.append(group_id)
                final_group_gids.append(gid)
                final_group_cids.append(distinct_cid)
            else:
                final_group_ids.extend([group_id] * len(cids))
                final_group_gids.extend([gid] * len(cids))
                final_group_cids.extend(cids)

        # Balanced category split
        rng = kwarray.ensure_rng(config['rng'])

        shuffle = rng is not None
        self = util_sklearn.StratifiedGroupKFold(n_splits=config['factor'],
                                                 random_state=rng,
                                                 shuffle=shuffle)

        if config['balance_categories']:
            split_idxs = list(self.split(X=final_group_gids, y=final_group_cids, groups=final_group_ids))
        else:
            split_idxs = list(self.split(X=final_group_gids, y=final_group_gids, groups=final_group_ids))
        idxs1, idxs2 = split_idxs[0]

        gids1 = sorted(ub.unique(ub.take(final_group_gids, idxs1)))
        gids2 = sorted(ub.unique(ub.take(final_group_gids, idxs2)))

        dset1 = dset.subset(gids1)
        dset2 = dset.subset(gids2)

        dumpkw = {
            'newlines': True,
            'compress': config['compress'],
        }
        dset1.fpath = config['dst1']
        print('Writing dset1 = {!r}'.format(dset1.fpath))
        dset1.dump(**dumpkw)

        dset2.fpath = config['dst2']
        print('Writing dset2 = {!r}'.format(dset2.fpath))
        dset2.dump(**dumpkw)

_CLI = CocoSplitCLI

if __name__ == '__main__':
    _CLI.main()
