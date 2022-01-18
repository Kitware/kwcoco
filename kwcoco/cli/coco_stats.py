#!/usr/bin/env python
import ubelt as ub
import scriptconfig as scfg


class CocoStatsCLI:
    name = 'stats'

    class CLIConfig(scfg.Config):
        """
        Compute summary statistics about a COCO dataset
        """
        default = {
            'src': scfg.Value(['special:shapes8'], nargs='+', help='path to dataset', position=1),
            'basic': scfg.Value(True, help='show basic stats'),
            'extended': scfg.Value(True, help='show extended stats'),
            'catfreq': scfg.Value(True, help='show category frequency stats'),
            'boxes': scfg.Value(False, help=ub.paragraph(
                '''
                show bounding box stats in width-height format.
                ''')),

            'annot_attrs': scfg.Value(False, help='show annotation attribute information'),
            'image_attrs': scfg.Value(False, help='show image attribute information'),
            'video_attrs': scfg.Value(False, help='show video attribute information'),

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
            dset = kwcoco.CocoDataset.coerce(fpath)
            datasets.append(dset)
        print('Finished reading datasets')

        # hack dataset tags
        dset_tags = [dset.tag for dset in datasets]
        if len(set(dset_tags)) < len(dset_tags):
            from os.path import commonprefix
            dset_fpaths = [dset.fpath for dset in datasets]
            toremove = commonprefix(dset_fpaths)
            for dset in datasets:
                dset.tag = dset.fpath.replace(toremove, '')

        try:
            import networkx as nx
            for dset in datasets:
                print('dset = {!r}'.format(dset))
                print('Category Heirarchy: ')
                print(nx.forest_str(dset.object_categories().graph))
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

            # allkeys = ['annots_per_img', 'annots_per_cat']
            allkeys = sorted(set(ub.flatten(s.keys() for s in tag_to_ext_stats.values())))
            # print('allkeys = {!r}'.format(allkeys))

            for key in allkeys:
                print('\n--{!r}'.format(key))
                df = pd.DataFrame.from_dict(
                    {k: v[key] for k, v in tag_to_ext_stats.items()})
                print(df.to_string(float_format=lambda x: '%0.3f' % x))

        if config['catfreq']:
            tag_to_freq = {}
            for dset in datasets:
                tag_to_freq[dset.tag] = dset.category_annotation_frequency()
            df = pd.DataFrame.from_dict(tag_to_freq)
            print(df.to_string(float_format=lambda x: '%0.3f' % x))

        if config['video_attrs']:
            print('Video Attrs')
            for dset in datasets:
                attrs = dset.videos().attribute_frequency()
                print('video_attrs = {}'.format(ub.repr2(attrs, nl=1)))

        if config['image_attrs']:
            print('Image Attrs')
            for dset in datasets:
                print('dset.tag = {!r}'.format(dset.tag))
                attrs = dset.images().attribute_frequency()
                print('image_attrs = {}'.format(ub.repr2(attrs, nl=1)))

        if config['annot_attrs']:
            print('Annot Attrs')
            for dset in datasets:
                print('dset.tag = {!r}'.format(dset.tag))
                attrs = dset.annots().attribute_frequency()
                print('annot_attrs = {}'.format(ub.repr2(attrs, nl=1)))

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
        python -m kwcoco.cli.coco_stats --src=special:shapes8
    """
    _CLI.main()
