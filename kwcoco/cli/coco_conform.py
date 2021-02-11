#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ubelt as ub
import scriptconfig as scfg


class CocoConformCLI:
    """
    Make the COCO file conform to the spec.

    Populates inferable information such as image size, annotation area, etc.
    """
    name = 'conform'

    class CLIConfig(scfg.Config):
        """
        Reroot image paths onto a new image root.
        """
        epilog = """
        Example Usage:
            kwcoco conform --help
            kwcoco conform --src=special:shapes8 --dst conformed.json
        """
        default = {
            'src': scfg.Value(None, help=(
                'Path to the coco dataset'), position=1),

            'ensure_imgsize': scfg.Value(True, help=ub.paragraph(
                '''
                ensure each image has height and width attributes
                ''')),

            'pycocotools_info': scfg.Value(True, help=ub.paragraph(
                '''
                ensure information needed for pycocotools
                ''')),

            'workers': scfg.Value(
                8, help='number of background workers for bigger checks'),

            'dst': scfg.Value(None, help=(
                'Save the rebased dataset to a new file')),
        }

    @classmethod
    def main(cls, cmdline=True, **kw):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> kw = {'src': 'special:shapes8'}
            >>> cmdline = False
            >>> cls = CocoConformCLI
            >>> cls.main(cmdline, **kw)
        """
        import kwcoco
        import kwimage

        config = cls.CLIConfig(kw, cmdline=cmdline)
        print('config = {}'.format(ub.repr2(dict(config), nl=1)))

        if config['src'] is None:
            raise Exception('must specify source: {}'.format(config['src']))
        if config['dst'] is None:
            raise Exception('must specify dest: {}'.format(config['dst']))

        dset = kwcoco.CocoDataset.coerce(config['src'])

        if config['ensure_imgsize']:
            dset._ensure_imgsize(workers=config['workers'])

        if config['pycocotools_info']:
            for ann in ub.ProgIter(dset.dataset['annotations'], desc='update anns'):
                if 'iscrowd' not in ann:
                    ann['iscrowd'] = False

                if 'ignore' not in ann:
                    ann['ignore'] = ann.get('weight', 1.0) < .5

                if 'area' not in ann:
                    # Use segmentation if available
                    if 'segmentation' in ann:
                        poly = kwimage.Polygon.from_coco(ann['segmentation'][0])
                        ann['area'] = float(poly.to_shapely().area)
                    else:
                        x, y, w, h = ann['bbox']
                        ann['area'] = w * h

        dset.fpath = config['dst']
        print('dump dset.fpath = {!r}'.format(dset.fpath))
        dset.dump(dset.fpath, newlines=True)


_CLI = CocoConformCLI

if __name__ == '__main__':
    _CLI.main()
