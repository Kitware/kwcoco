#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ubelt as ub
import scriptconfig as scfg


class CocoStatsCLI:
    name = 'stats'

    class CLIConfig(scfg.Config):
        """
        Compute summary statistics about a COCO dataset
        """
        default = {
            'src': scfg.Value(['special:shapes8'], nargs='+', help='path to dataset'),
            'basic': scfg.Value(True, help='show basic stats'),
            'extended': scfg.Value(True, help='show extended stats'),
            'catfreq': scfg.Value(True, help='show category frequency stats'),
            'boxes': scfg.Value(False, help='show bounding box stats'),
            'annot_attrs': scfg.Value(False, help='show annot attribute information'),

            'embed': scfg.Value(False, help='embed into interactive shell'),
        }
        epilog = """
        Example Usage:
            kwcoco stats --src=special:shapes8
            kwcoco stats --src=special:shapes8 --boxes=True
        """

    @classmethod
    def main(cls, cmdline=True, **kw):
        """
        Example:
            >>> kw = {'src': 'special:shapes8'}
            >>> cmdline = False
            >>> cls = CocoStatsCLI
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

        datasets = []
        for fpath in ub.ProgIter(fpaths, desc='reading datasets', verbose=1):
            print('reading fpath = {!r}'.format(fpath))
            dset = kwcoco.CocoDataset.coerce(fpath)
            datasets.append(dset)

        try:
            from kwcoco.category_tree import _print_forest
            for dset in datasets:
                print('dset = {!r}'.format(dset))
                print('Category Heirarchy: ')
                _print_forest(dset.object_categories().graph)
        except Exception:
            pass

        import pandas as pd
        pd.set_option('max_colwidth', 256)

        if config['basic']:
            tag_to_stats = {}
            for dset in datasets:
                tag_to_stats[dset.tag] = dset.basic_stats()
            df = pd.DataFrame.from_dict(tag_to_stats)
            print(df.to_string(float_format=lambda x: '%0.3f' % x))

        if config['extended']:
            tag_to_ext_stats = {}
            for dset in datasets:
                tag_to_ext_stats[dset.tag] = dset.extended_stats()

            for key in ['annots_per_img', 'annots_per_cat']:
                print('{!r}'.format(key))
                df = pd.DataFrame.from_dict(
                    {k: v[key] for k, v in tag_to_ext_stats.items()})
                print(df.to_string(float_format=lambda x: '%0.3f' % x))

        if config['catfreq']:
            tag_to_freq = {}
            for dset in datasets:
                tag_to_freq[dset.tag] = dset.category_annotation_frequency()
            df = pd.DataFrame.from_dict(tag_to_freq)
            print(df.to_string(float_format=lambda x: '%0.3f' % x))

        if config['annot_attrs']:
            print('Annot Attrs')
            for dset in datasets:
                print('dset.tag = {!r}'.format(dset.tag))

                attrs = ub.ddict(lambda: 0)
                for ann in dset.anns.values():
                    for key, value in ann.items():
                        if value:
                            attrs[key] += 1

                print('annot attrs = {!r}'.format(attrs))

        if config['boxes']:
            print('Box stats')
            for dset in datasets:
                print('dset.tag = {!r}'.format(dset.tag))
                print(ub.repr2(dset.boxsize_stats(), nl=-1, precision=2))

        if config['embed']:
            # Hidden hack
            import xdev
            xdev.embed()

        # for dset in datasets:
        #     # dset = datasets[0]
        #     # kwcoco.CocoDataset.coerce(config['src'])
        #     print('dset.fpath = {!r}'.format(dset.fpath))

        #     if config['basic']:
        #         basic = dset.basic_stats()
        #         print('basic = {}'.format(ub.repr2(basic, nl=1)))

        #     if config['extended']:
        #         extended = dset.extended_stats()
        #         print('extended = {}'.format(ub.repr2(extended, nl=1, precision=2)))

        #     if config['catfreq']:
        #         print('Category frequency')
        #         freq = dset.category_annotation_frequency()
        #         import pandas as pd
        #         df = pd.DataFrame.from_dict({str(dset.tag): freq})
        #         pd.set_option('max_colwidth', 256)
        #         print(df.to_string(float_format=lambda x: '%0.3f' % x))

        #     if config['boxes']:
        #         print('Box stats')
        #         print(ub.repr2(dset.boxsize_stats(), nl=-1, precision=2))


_CLI = CocoStatsCLI

if __name__ == '__main__':
    """
    CommandLine:
        python -m kwcoco.coco_stats --src=special:shapes8
    """
    _CLI._main()
