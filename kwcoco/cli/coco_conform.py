#!/usr/bin/env python
import ubelt as ub
import scriptconfig as scfg


class CocoConformCLI(scfg.DataConfig):
    """
    Infer properties to make the COCO file conform to different specs.

    Arguments can be used to control which information is inferred.  By
    default, information such as image size, annotation area, are added to
    the file.

    Other arguments like ``--legacy`` and ``--mmlab`` can be used to
    conform to specifications expected by external tooling.
    """
    __command__ = 'conform'

    __epilog__ = """
    Example Usage:
        kwcoco conform --help
        kwcoco conform --src=special:shapes8 --dst conformed.json
        kwcoco conform special:shapes8 conformed.json
    """

    src = scfg.Value(None, position=1, help='the path to the input coco dataset')
    # FIXME: required is broken with --help

    dst = scfg.Value(None, position=2, help='the location to save the output dataset.')

    ensure_imgsize = scfg.Value(True, help=ub.paragraph(
            '''
            ensure each image has height and width attributes
            '''))

    pycocotools_info = scfg.Value(True, help=ub.paragraph(
            '''
            ensure information needed for pycocotools
            '''))

    legacy = scfg.Value(False, help=ub.paragraph(
            '''
            if True tries to convert to the original ms-coco format
            '''))

    mmlab = scfg.Value(False, help=ub.paragraph(
            '''
            if True tries to convert data to be compatible with open-
            mmlab tooling
            '''))

    compress = scfg.Value('auto', help='if True writes results with compression')

    workers = scfg.Value(8, help=ub.paragraph(
            '''
            number of background workers used for IO bound checks
            '''))

    inplace = scfg.Value(False, isflag=True, help=(
        'if True and dst is unspecified then the output will overwrite the input'))

    @classmethod
    def main(cls, cmdline=True, **kw):
        """
        Example:
            >>> from kwcoco.cli.coco_conform import *  # NOQA
            >>> import kwcoco
            >>> import ubelt as ub
            >>> dpath = ub.Path.appdir('kwcoco/tests/cli/conform').ensuredir()
            >>> dst = dpath / 'out.kwcoco.json'
            >>> kw = {'src': 'special:shapes8', 'dst': dst, 'compress': True}
            >>> cmdline = False
            >>> cls = CocoConformCLI
            >>> cls.main(cmdline, **kw)
        """
        import kwcoco

        config = cls.cli(data=kw, cmdline=cmdline, strict=True)
        print('config = {}'.format(ub.urepr(dict(config), nl=1)))

        if config['src'] is None:
            raise Exception('must specify source: {}'.format(config['src']))
        if config['dst'] is None:
            if config['inplace']:
                config['dst'] = config['src']
            else:
                raise ValueError('must specify dst: {}'.format(config['dst']))

        dset = kwcoco.CocoDataset.coerce(config['src'])

        config_ = ub.udict(config) - {'src', 'dst', 'compress'}
        dset.conform(**config_)

        dset.fpath = config['dst']
        print('dump dset.fpath = {!r}'.format(dset.fpath))
        dumpkw = {
            'newlines': True,
            'compress': config['compress'],
        }
        dset.dump(dset.fpath, **dumpkw)


__cli__ = CocoConformCLI

if __name__ == '__main__':
    __cli__.main()
