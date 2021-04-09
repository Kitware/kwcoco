#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ubelt as ub
import scriptconfig as scfg


class CocoSubsetCLI(object):
    name = 'subset'

    class CLIConfig(scfg.Config):
        """
        Take a subset of this dataset and write it to a new file
        """
        default = {
            'src': scfg.Value(None, help='input dataset path', position=1),
            'dst': scfg.Value(None, help='output dataset path'),
            'include_categories': scfg.Value(
                None, type=str, help=ub.paragraph(
                    '''
                    a comma separated list of categories, if specified only
                    images containing these categories will be included.
                    ''')),  # TODO: pattern matching?

            'gids': scfg.Value(
                None, help=ub.paragraph(
                    '''
                    if specified, only take these image ids
                    ''')),

            # TODO: Add more filter criteria
            #
            # image size
            # image timestamp
            # image file name matches
            # annotations with segmentations / keypoints?
            # iamges/annotations that contain a special attribute?
            # images with a maximum / minimum number of annotations?

            # 'rng': scfg.Value(None, help='random seed'),
        }
        epilog = """
        Example Usage:
            kwcoco subset --src special:shapes8 --dst=foo.kwcoco.json
        """

    @classmethod
    def main(cls, cmdline=True, **kw):
        """
        Example:
            >>> kw = {'src': 'special:shapes8',
            >>>       'dst': 'subset.json', 'include_categories': 'superstar'}
            >>> cmdline = False
            >>> cls = CocoSubsetCLI
            >>> cls.main(cmdline, **kw)
        """
        import kwcoco

        config = cls.CLIConfig(kw, cmdline=cmdline)
        print('config = {}'.format(ub.repr2(dict(config), nl=1)))

        if config['src'] is None:
            raise Exception('must specify source: {}'.format(config['src']))

        print('reading fpath = {!r}'.format(config['src']))
        dset = kwcoco.CocoDataset.coerce(config['src'])

        valid_gids = set(dset.imgs.keys())

        if config['gids'] is not None:
            if isinstance(config['gids'], str):
                valid_gids &= set(map(int, config['gids'].split(',')))
            elif ub.iterable(config['gids']):
                valid_gids &= set(map(int, config['gids']))
            else:
                raise KeyError(config['gids'])

        if config['include_categories'] is not None:
            catnames = config['include_categories'].split(',')
            chosen_cids = []
            for cname in catnames:
                cid = dset._resolve_to_cat(cname)['id']
                chosen_cids.append(cid)

            category_gids = set(ub.flatten(ub.take(
                dset.index.cid_to_gids, set(chosen_cids))))

            valid_gids &= category_gids

        # Balanced category split
        # rng = kwarray.ensure_rng(config['rng'])
        new_dset = dset.subset(valid_gids)

        new_dset.fpath = config['dst']
        print('Writing new_dset.fpath = {!r}'.format(new_dset.fpath))
        new_dset.dump(new_dset.fpath, newlines=True)

_CLI = CocoSubsetCLI

if __name__ == '__main__':
    _CLI.main()
