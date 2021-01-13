#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ubelt as ub
import scriptconfig as scfg


class CocoShowCLI:
    name = 'show'

    class CLIConfig(scfg.Config):
        """
        Visualize a COCO image using matplotlib, optionally writing it to disk
        """
        epilog = """
        Example Usage:
            kwcoco show --help
            kwcoco show --src=special:shapes8 --gid=1
            kwcoco show --src=special:shapes8 --gid=1 --dst out.png
        """
        default = {
            'src': scfg.Value(None, help=(
                'Path to the coco dataset')),
            'gid': scfg.Value(None, help=(
                'Image id to show, if unspecified the first image is shown')),
            'aid': scfg.Value(None, help=(
                'Annotation id to show, mutually exclusive with gid')),
            'dst': scfg.Value(None, help=(
                'Save the image to the specified file. '
                'If unspecified, the image is shown with pyplot')),

            'mode': scfg.Value(
                'matplotlib', choices=['matplotlib', 'opencv'],
                help='method used to draw the image'
            ),

            'channels': scfg.Value(
                None, type=str, help=ub.paragraph(
                    '''
                    By default uses the default channels (usually this is rgb),
                    otherwise specify the name of an auxillary channels
                    ''')
            ),

            'show_annots': scfg.Value(True, help=(
                'Overlay annotations on dispaly')),
        }

    @classmethod
    def main(cls, cmdline=True, **kw):
        """
        TODO:
            - [ ] Visualize auxillary data

        Example:
            >>> # xdoctest: +SKIP
            >>> kw = {'src': 'special:shapes8'}
            >>> cmdline = False
            >>> cls = CocoShowCLI
            >>> cls.main(cmdline, **kw)
        """
        import kwcoco
        import kwimage
        import kwplot

        config = cls.CLIConfig(kw, cmdline=cmdline)
        print('config = {}'.format(ub.repr2(dict(config), nl=1)))

        if config['src'] is None:
            raise Exception('must specify source: {}'.format(config['src']))

        dset = kwcoco.CocoDataset.coerce(config['src'])
        print('dset.fpath = {!r}'.format(dset.fpath))

        plt = kwplot.autoplt()

        gid = config['gid']
        aid = config['aid']

        out_fpath = config['dst']

        if gid is None and aid is None:
            gid = ub.peek(dset.imgs)

        if config['mode'] == 'matplotlib':
            show_kw = {
                'show_annots': config['show_annots'],
            }
            if config['show_annots'] == 'both':
                show_kw.pop('show_annots')
                show_kw['title'] = ''
                ax = dset.show_image(gid=gid, aid=aid, pnum=(1, 2, 1),
                                     show_annots=False, **show_kw)
                ax = dset.show_image(gid=gid, aid=aid, pnum=(1, 2, 2),
                                     show_annots=True, **show_kw)
            else:
                ax = dset.show_image(gid=gid, aid=aid, **show_kw)
            if out_fpath is None:
                plt.show()
            else:
                ax.figure.savefig(out_fpath)
        elif config['mode'] == 'opencv':
            canvas = dset.draw_image(gid, channels=config['channels'])
            if out_fpath is None:
                kwplot.imshow(canvas)
            else:
                kwimage.imwrite(out_fpath, canvas)

        else:
            raise KeyError(config['mode'])


_CLI = CocoShowCLI

if __name__ == '__main__':
    """
    CommandLine:
        python -m kwcoco.cli.coco_show --src=special:shapes8 --gid=1
        python -m kwcoco.cli.coco_show --src=special:shapes8 --gid=1
        python -m kwcoco.cli.coco_show --src=special:shapes8 --gid=1 --dst out.png
    """
    _CLI.main()
