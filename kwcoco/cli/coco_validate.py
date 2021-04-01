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
            >>> from kwcoco.cli.coco_validate import *  # NOQA
            >>> kw = {'src': 'special:shapes8'}
            >>> cmdline = False
            >>> cls = CocoValidateCLI
            >>> cls.main(cmdline, **kw)
        """
        import kwcoco
        config = cls.CLIConfig(kw, cmdline=cmdline)
        print('config = {}'.format(ub.repr2(dict(config), nl=1)))

        if config['src'] is None:
            raise Exception('must specify source: {}'.format(config['src']))

        if isinstance(config['src'], str):
            fpaths = [config['src']]
        else:
            fpaths = config['src']

        if config['dst']:
            if len(fpaths) != 1:
                raise Exception('can only specify 1 dataset in fix mode')

        fix_strat = set()
        if config['fix'] is not None:
            fix_strat = {c.lower() for c in config['fix'].split('+')}

        for fpath in ub.ProgIter(fpaths, desc='reading datasets', verbose=1):
            print('reading fpath = {!r}'.format(fpath))
            dset = kwcoco.CocoDataset.coerce(fpath)

            config_ = ub.dict_diff(config, {'src', 'dst', 'fix'})
            result = dset.validate(**config_)

            if 'missing' in result:
                if 'remove' in fix_strat:
                    missing = result['missing']
                    bad_gids = [t[2] for t in missing]
                    status = dset.remove_images(bad_gids, verbose=1)
                    print('status = {}'.format(ub.repr2(status, nl=1)))

            if 'corrupted' in result:
                if 'remove' in fix_strat:
                    corrupted = result['corrupted']
                    bad_gids = [t[2] for t in corrupted]
                    status = dset.remove_images(bad_gids, verbose=1)
                    print('status = {}'.format(ub.repr2(status, nl=1)))

            if config['dst']:
                if len(fpaths) != 1:
                    raise Exception('can only specify 1 dataset in fix mode')
                dset.dump(config['dst'], newlines=True)

            errors = result['errors']
            if errors:
                print('result = {}'.format(ub.repr2(result, nl=-1)))
                raise Exception('\n'.join(errors))


_CLI = CocoValidateCLI

if __name__ == '__main__':
    """
    CommandLine:
        python -m kwcoco.cli.coco_stats --src=special:shapes8
    """
    _CLI.main()
