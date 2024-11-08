#!/usr/bin/env python
import ubelt as ub
import scriptconfig as scfg


class CocoModifyCatsCLI(scfg.DataConfig):
    """
    Remove, rename, or coarsen categories.
    """
    __command__ = 'modify_categories'
    __epilog__ = """
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
            'Save the modified dataset to a new file')),

        'keep_annots': scfg.Value(False, help=(
            'if False, removes annotations when categories are removed, '
            'otherwise the annotations category is simply unset')),

        'remove_empty_images': scfg.Value(False, isflag=True, help=(
            'if True, removes images when categories are removed, '
            'otherwise the images are simply kept as is')),

        'remove': scfg.Value(None, help='Category names to remove. Mutex with keep.'),

        'keep': scfg.Value(None, help='If specified, remove all other categories. Mutex with remove.'),

        'rename': scfg.Value(None, type=str, help='category mapping in the format. "old1:new1,old2:new2"'),

        'compress': scfg.Value('auto', help='if True writes results with compression'),
    }

    @classmethod
    def main(cls, cmdline=True, **kw):
        """
        Example:
            >>> from kwcoco.cli.coco_modify_categories import *  # NOQA
            >>> import kwcoco
            >>> import ubelt as ub
            >>> dpath = ub.Path.appdir('kwcoco/tests/coco_modify_categories').ensuredir()
            >>> old_dset = kwcoco.CocoDataset.demo('special:shapes8')
            >>> dst_fpath = dpath / 'modified_category.kwcoco.zip'
            >>> kw = {'src': old_dset.fpath, 'dst': dst_fpath, 'keep': []}
            >>> cmdline = False
            >>> cls = CocoModifyCatsCLI
            >>> cls.main(cmdline=cmdline, **kw)
            >>> assert dst_fpath.exists()
            >>> new_dset = kwcoco.CocoDataset(dst_fpath)
            >>> assert len(new_dset.cats) == 0

        Example:
            >>> # xdoctest: +SKIP
            >>> kw = {'src': 'special:shapes8'}
            >>> cmdline = False
            >>> cls = CocoModifyCatsCLI
            >>> cls.main(cmdline, **kw)
        """
        import kwcoco
        config = cls.cli(data=kw, cmdline=cmdline, strict=True)
        print('config = {}'.format(ub.urepr(dict(config), nl=1)))

        if config['src'] is None:
            raise Exception('must specify source: {}'.format(config['src']))

        dset = kwcoco.CocoDataset.coerce(config['src'])
        print('dset = {!r}'.format(dset))

        import networkx as nx
        import warnings
        print('Input Categories:')
        try:
            print(nx.forest_str(dset.object_categories().graph))
        except AttributeError:
            print(nx.write_network_text(dset.object_categories().graph))

        if config['rename'] is not None:
            # parse rename string
            mapper = dict([p.split(':') for p in config['rename'].split(',')])
            print('mapper = {}'.format(ub.urepr(mapper, nl=1)))
            dset.rename_categories(mapper)

        keep = config['keep']
        if keep is not None:
            classes = set(dset.name_to_cat.keys())
            try:
                import kwutil
                keep = kwutil.Yaml.coerce(keep)
            except ImportError:
                warnings.warn('kwutil is not available')
            if isinstance(keep, str):
                warnings.warn(
                    'Keep is specified as a string. '
                    'Did you mean to input a list?')
            remove = list(classes - set(keep))
        else:
            remove = config['remove']

        if remove is not None:
            try:
                import kwutil
                remove = kwutil.Yaml.coerce(remove)
            except ImportError:
                warnings.warn('kwutil is not available')
            remove_cids = []
            for catname in remove:
                try:
                    cid = dset._resolve_to_cid(catname)
                except KeyError:
                    warnings.warn('unable to lookup catname={!r}'.format(catname))
                else:
                    remove_cids.append(cid)
            dset.remove_categories(
                remove_cids, keep_annots=config['keep_annots'], verbose=1)

        if config['remove_empty_images']:
            noannot_images = [gid for gid, aids in dset.index.gid_to_aids.items() if len(aids) == 0]
            dset.remove_images(noannot_images, verbose=3)

        print('Output Categories: ')
        try:
            print(nx.forest_str(dset.object_categories().graph))
        except AttributeError:
            print(nx.write_network_text(dset.object_categories().graph))

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


__cli__ = CocoModifyCatsCLI

if __name__ == '__main__':
    __cli__.main()
