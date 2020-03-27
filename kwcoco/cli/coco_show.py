#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ubelt as ub
import scriptconfig as scfg


class CocoShowCLI:
    name = 'show'

    class CLIConfig(scfg.Config):
        """
        Visualize a COCO image
        """
        default = {
            'src': scfg.Value(None, help='path to dataset'),
            'gid': scfg.Value(None, help='image id to show'),
            'aid': scfg.Value(None, help='annotation id to show'),

            'dst': scfg.Value(None, help='write image to file if specified'),
        }

    @classmethod
    def main(cls, cmdline=True, **kw):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> kw = {'src': 'special:shapes8'}
            >>> cmdline = False
            >>> cls = CocoShowCLI
            >>> cls.main(cmdline, **kw)
        """
        import kwcoco
        config = cls.CLIConfig(kw, cmdline=cmdline)
        print('config = {}'.format(ub.repr2(dict(config), nl=1)))

        if config['src'] is None:
            raise Exception('must specify source: '.format(config['src']))

        dset = kwcoco.CocoDataset.coerce(config['src'])
        print('dset.fpath = {!r}'.format(dset.fpath))

        import kwplot
        plt = kwplot.autoplt()

        gid = config['gid']
        aid = config['aid']

        out_fpath = config['dst']

        if gid is None and aid is None:
            gid = ub.peek(dset.imgs)

        ax = dset.show_image(gid=gid, aid=aid)
        print('ax.figure = {!r}'.format(ax.figure))
        if out_fpath is None:
            plt.show()
        else:
            ax.figure.savefig(out_fpath)


_CLI = CocoShowCLI

if __name__ == '__main__':
    """
    CommandLine:
        python -m kwcoco.cli.coco_show --src=special:shapes8 --gid=1
        python -m kwcoco.cli.coco_show --src=special:shapes8 --gid=1
        python -m kwcoco.cli.coco_show --src=special:shapes8 --gid=1 --dst out.png
    """
    _CLI.main()
