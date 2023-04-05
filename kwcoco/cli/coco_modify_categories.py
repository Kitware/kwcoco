#!/usr/bin/env python
import ubelt as ub
import scriptconfig as scfg


class CocoModifyCatsCLI:
    """
    Remove, rename, or coarsen categories.
    """
    name = 'modify_categories'

    class CLIConfig(scfg.Config):
        """
        Rename or remove categories
        """
        epilog = """
        Example Usage:
            kwcoco modify_categories --help
            kwcoco modify_categories --src=special:shapes8 --dst modcats.json
            kwcoco modify_categories --src=special:shapes8 --dst modcats.json --rename eff:F,star:sun
            kwcoco modify_categories --src=special:shapes8 --dst modcats.json --remove eff,star
            kwcoco modify_categories --src=special:shapes8 --dst modcats.json --keep eff,

            kwcoco modify_categories --src=special:shapes8 --dst modcats.json --keep=[] --keep_annots=True
        """
        __default__ = {
            'src': scfg.Value(None, help=(
                'Path to the coco dataset'), position=1),

            'dst': scfg.Value(None, help=(
                'Save the rebased dataset to a new file')),

            'keep_annots': scfg.Value(False, help=(
                'if False, removes annotations when categories are removed, '
                'otherwise the annotations category is simply unset')),

            'remove': scfg.Value(None, help='Category names to remove. Mutex with keep.'),

            'keep': scfg.Value(None, help='If specified, remove all other categories. Mutex with remove.'),

            'rename': scfg.Value(None, type=str, help='category mapping in the format. "old1:new1,old2:new2"'),

            'compress': scfg.Value('auto', help='if True writes results with compression'),
        }

    @classmethod
    def main(cls, cmdline=True, **kw):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> kw = {'src': 'special:shapes8'}
            >>> cmdline = False
            >>> cls = CocoModifyCatsCLI
            >>> cls.main(cmdline, **kw)
        """
        import kwcoco
        config = cls.CLIConfig(kw, cmdline=cmdline)
        print('config = {}'.format(ub.urepr(dict(config), nl=1)))

        if config['src'] is None:
            raise Exception('must specify source: {}'.format(config['src']))

        dset = kwcoco.CocoDataset.coerce(config['src'])
        print('dset = {!r}'.format(dset))

        import networkx as nx
        print('Input Categories:')
        print(nx.forest_str(dset.object_categories().graph))

        if config['rename'] is not None:
            # parse rename string
            mapper = dict([p.split(':') for p in config['rename'].split(',')])
            print('mapper = {}'.format(ub.urepr(mapper, nl=1)))
            dset.rename_categories(mapper)

        if config['keep'] is not None:
            classes = set(dset.name_to_cat.keys())
            if isinstance(config['keep'], str):
                import warnings
                warnings.warn(
                    'Keep is specified as a string. '
                    'Did you mean to input a list?')
            remove = list(classes - set(config['keep']))
        else:
            remove = config['remove']

        if remove is not None:
            remove_cids = []
            for catname in remove:
                try:
                    cid = dset._resolve_to_cid(catname)
                except KeyError:
                    import warnings
                    warnings.warn('unable to lookup catname={!r}'.format(catname))
                else:
                    remove_cids.append(cid)
            dset.remove_categories(
                remove_cids, keep_annots=config['keep_annots'], verbose=1)

        print('Output Categories: ')
        print(nx.forest_str(dset.object_categories().graph))

        if config['dst'] is None:
            print('dry run')
        else:
            dset.fpath = config['dst']
            print('dset.fpath = {!r}'.format(dset.fpath))
            dumpkw = {
                'newlines': True,
                'compress': config['compress'],
            }
            dset.dump(dset.fpath, **dumpkw)


_CLI = CocoModifyCatsCLI

if __name__ == '__main__':
    _CLI.main()
