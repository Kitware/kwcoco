from __future__ import annotations

#!/usr/bin/env python
import ubelt as ub
import kwconf
from kwcoco.cli._base import KWCocoConfig


class CocoShowCLI(KWCocoConfig):
    """
    Visualize a COCO image using matplotlib or opencv, optionally writing
    it to disk.
    """

    __command__ = 'show'
    __epilog__ = """
    Example Usage:
        kwcoco show --help
        kwcoco show --src=special:shapes8 --gid=1
        kwcoco show --src=special:shapes8 --gid=1 --dst out.png
    """
    __default__ = {
        'src': kwconf.Value(None, help=('Path to the coco dataset'), position=1),
        'gid': kwconf.Value(
            None, help=('Image id to show, if unspecified the first image is shown')
        ),
        'aid': kwconf.Value(
            None, help=('Annotation id to show, mutually exclusive with gid')
        ),
        'dst': kwconf.Value(
            None,
            help=(
                'Save the image to the specified file. '
                'If unspecified, the image is shown with pyplot'
            ),
        ),
        'mode': kwconf.Value(
            'matplotlib',
            choices=['matplotlib', 'opencv'],
            help='method used to draw the image',
        ),
        'channels': kwconf.Value(
            None,
            parser=str,
            help=ub.paragraph(
                """
                By default uses the default channels (usually this is rgb),
                otherwise specify the name of an auxiliary channels
                """
            ),
        ),
        'show_annots': kwconf.Value(
            True, isflag=True, help=('Overlay annotations on display')
        ),
        'show_labels': kwconf.Value(
            False, isflag=True, help=('Overlay labels on annotations')
        ),
    }

    @classmethod
    def main(cls, argv=True, **kw):
        """
        TODO:
            - [ ] Visualize multispectral data with assets

        CommandLine:
            xdoctest -m kwcoco.cli.coco_show CocoShowCLI.main:0

        Example:
            >>> # xdoctest: +REQUIRES(module:cv2)
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> from kwcoco.cli.coco_show import *  # NOQA
            >>> import kwcoco
            >>> import ubelt as ub
            >>> dpath = ub.Path.appdir('kwcoco/tests/coco_show/').ensuredir()
            >>> dset = kwcoco.CocoDataset.demo('special:shapes8')
            >>> fpath = dpath / 'show_output.png'
            >>> kw = {'src': dset.fpath, 'mode': 'opencv', 'dst': fpath}
            >>> argv = False
            >>> cls = CocoShowCLI
            >>> cls.main(argv=argv, **kw)
            >>> assert fpath.exists()

        Example:
            >>> # xdoctest: +SKIP
            >>> kw = {'src': 'special:shapes8'}
            >>> argv = False
            >>> cls = CocoShowCLI
            >>> cls.main(argv=argv, **kw)
        """
        import kwcoco
        import kwimage
        import kwplot

        config = cls.cli(data=kw, argv=argv, strict=True)
        print('config = {}'.format(ub.urepr(dict(config), nl=1)))

        if config['src'] is None:
            raise Exception('must specify source: {}'.format(config['src']))

        dset = kwcoco.CocoDataset.coerce(config['src'])
        print('dset.fpath = {!r}'.format(dset.fpath))

        plt = kwplot.autoplt()

        gid = config['gid']
        aid = config['aid']

        out_fpath = config['dst']

        gid_sequence = None

        if gid is None and aid is None:
            gid = ub.peek(dset.imgs)
            if dset.n_videos > 0:
                gid_sequence = list(ub.flatten(dset.videos().images))
                if len(gid_sequence):
                    gid = gid_sequence[0]

        if config['mode'] == 'matplotlib':
            show_kw = {
                'show_annots': config['show_annots'],
                'show_labels': config['show_labels'],
                'channels': config['channels'],
            }
            if config['show_annots'] == 'both':
                show_kw.pop('show_annots')
                show_kw['title'] = ''
                ax = dset.show_image(
                    gid=gid, aid=aid, pnum=(1, 2, 1), show_annots=False, **show_kw
                )
                ax = dset.show_image(
                    gid=gid, aid=aid, pnum=(1, 2, 2), show_annots=True, **show_kw
                )
            else:
                ax = dset.show_image(gid=gid, aid=aid, **show_kw)
            if out_fpath is None:
                if 1:
                    try:
                        import xdev
                    except Exception:
                        pass
                    else:
                        if gid_sequence is None:
                            gid_sequence = [gid] + list(set(dset.imgs.keys()) - {gid})
                        for gid in xdev.InteractiveIter(gid_sequence):
                            ax = dset.show_image(gid=gid, aid=aid, **show_kw)
                            xdev.InteractiveIter.draw()
                            plt.show(block=False)
                plt.show()
            else:
                ax.figure.savefig(out_fpath)
        elif config['mode'] == 'opencv':
            canvas = dset.draw_image(gid, channels=config['channels'])
            if out_fpath is None:
                kwplot.imshow(canvas)
                plt.show()
            else:
                kwimage.imwrite(out_fpath, canvas)

        else:
            raise KeyError(config['mode'])


__cli__ = CocoShowCLI

if __name__ == '__main__':
    """
    CommandLine:
        python -m kwcoco.cli.coco_show --src=special:shapes8 --gid=1
        python -m kwcoco.cli.coco_show --src=special:shapes8 --gid=1
        python -m kwcoco.cli.coco_show --src=special:shapes8 --gid=1 --dst out.png
    """
    __cli__.main()
