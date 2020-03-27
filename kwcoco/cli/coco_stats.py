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
            'src': scfg.Value('special:shapes8', help='path to dataset'),
            'basic': scfg.Value(True, help='show basic stats'),
            'extended': scfg.Value(True, help='show extended stats'),
            'catfreq': scfg.Value(True, help='show category frequency stats'),
            'boxes': scfg.Value(False, help='show bounding box stats'),
        }

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
            raise Exception('must specify source: '.format(config['src']))

        dset = kwcoco.CocoDataset.coerce(config['src'])
        print('dset.fpath = {!r}'.format(dset.fpath))

        if config['basic']:
            basic = dset.basic_stats()
            print('basic = {}'.format(ub.repr2(basic, nl=1)))

        if config['extended']:
            extended = dset.extended_stats()
            print('extended = {}'.format(ub.repr2(extended, nl=1, precision=2)))

        if config['catfreq']:
            print('Category frequency')
            freq = dset.category_annotation_frequency()
            import pandas as pd
            df = pd.DataFrame.from_dict({str(dset.tag): freq})
            pd.set_option('max_colwidth', 256)
            print(df.to_string(float_format=lambda x: '%0.3f' % x))

        if config['boxes']:
            print('Box stats')
            print(ub.repr2(dset.boxsize_stats(), nl=-1, precision=2))


_CLI = CocoStatsCLI

if __name__ == '__main__':
    """
    CommandLine:
        python -m kwcoco.coco_stats --src=special:shapes8
    """
    _CLI._main()
