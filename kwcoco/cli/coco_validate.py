#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ubelt as ub
import scriptconfig as scfg


class CocoValidateCLI:
    name = 'validate'

    class CLIConfig(scfg.Config):
        """
        Validate that a coco file conforms to the json schema, that assets
        exist, and potentially fix corrupted assets by removing them.
        """
        default = {
            'src': scfg.Value(['special:shapes8'], nargs='+', help='path to datasets', position=1),
            'schema': scfg.Value(True, help='If True check the json schema'),

            'missing': scfg.Value(True, help='If True check if all assets (e.g. images) exist'),
            'corrupted': scfg.Value(False, help='If True check the assets can be read'),

            'fix': scfg.Value(None, help=ub.paragraph(
                '''
                Code indicating strategy to attempt to fix the dataset.
                If None, do nothing.
                If remove, removes missing / corrupted images.
                Other strategies may be added in the future.

                This is a hueristic and does not always work. dst must be
                specified. And only one src dataset can be given.
                ''')),

            'dst': scfg.Value(None, help=ub.paragraph(
                '''
                Location to write a "fixed" coco file if a fix strategy is
                given.
                '''))

        }
        epilog = """
        Example Usage:
            kwcoco toydata --dst foo.json --key=special:shapes8
            kwcoco validate --src=foo.json --corrupted=True
        """

    @classmethod
    def main(cls, cmdline=True, **kw):
        """
        Example:
            >>> kw = {'src': 'special:shapes8'}
            >>> cmdline = False
            >>> cls = CocoValidateCLI
            >>> cls.main(cmdline, **kw)
        """
        import kwcoco
        from kwcoco.coco_schema import COCO_SCHEMA

        config = cls.CLIConfig(kw, cmdline=cmdline)
        print('config = {}'.format(ub.repr2(dict(config), nl=1)))

        if config['src'] is None:
            raise Exception('must specify source: {}'.format(config['src']))

        if isinstance(config['src'], str):
            fpaths = [config['src']]
        else:
            fpaths = config['src']

        fix_strat = set()
        if config['fix'] is not None:
            fix_strat = {c.lower() for c in config['fix'].split('+')}

        datasets = []
        for fpath in ub.ProgIter(fpaths, desc='reading datasets', verbose=1):
            print('reading fpath = {!r}'.format(fpath))
            dset = kwcoco.CocoDataset.coerce(fpath)
            datasets.append(dset)

        if config['schema']:
            for dset in datasets:
                print('Check schema: dset = {!r}'.format(dset))
                result = COCO_SCHEMA.validate(dset.dataset)
                print('result = {!r}'.format(result))

        if config['missing']:
            for dset in datasets:
                missing = dset.missing_images(check_aux=True, verbose=1)
                if missing:
                    print('missing = {}'.format(ub.repr2(missing, nl=1)))

                    if 'remove' in fix_strat:
                        bad_gids = [t[2] for t in missing]
                        status = dset.remove_images(bad_gids, verbose=1)
                        print('status = {}'.format(ub.repr2(status, nl=1)))
                    else:
                        raise Exception('missing assets')
                # print('Check assets exist: dset = {!r}'.format(dset))
                # all_gids = list(dset.imgs.keys())
                # for gid in all_gids:
                #     fpath = dset.get_image_fpath(gid)
                #     if not exists(fpath):
                #         raise Exception('fpath = {} does not exist'.format(fpath))

        if config['corrupted']:
            for dset in datasets:
                bad_gpaths = dset.corrupted_images(check_aux=True, verbose=1)
                if bad_gpaths:
                    print('bad_gpaths = {}'.format(ub.repr2(bad_gpaths, nl=1)))

                    if 'remove' in fix_strat:
                        bad_gids = [t[2] for t in bad_gpaths]
                        status = dset.remove_images(bad_gids, verbose=1)
                        print('status = {}'.format(ub.repr2(status, nl=1)))
                    else:
                        raise Exception('corrupted assets')

        if config['dst']:
            if len(datasets) != 1:
                raise Exception('can only specify 1 dataset in fix mode')
            dset = datasets[0]
            dset.dump(config['dst'], newlines=True)


_CLI = CocoValidateCLI

if __name__ == '__main__':
    """
    CommandLine:
        python -m kwcoco.cli.coco_stats --src=special:shapes8
    """
    _CLI.main()
