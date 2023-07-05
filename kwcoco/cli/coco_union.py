#!/usr/bin/env python
import ubelt as ub
import scriptconfig as scfg


class CocoUnionCLI(object):
    name = 'union'

    class CLIConfig(scfg.DataConfig):
        """
        Combine multiple COCO datasets into a single merged dataset.
        """
        __command__ = 'union'

        src = scfg.Value([], position=1, help='path to multiple input datasets', nargs='+')

        dst = scfg.Value('combo.kwcoco.json', help='path to output dataset')

        absolute = scfg.Value(False, isflag=1, help=ub.paragraph(
                '''
                if True, converts paths to absolute paths before doing union
                '''))

        remember_parent = scfg.Value(False, isflag=True, help=ub.paragraph(
                '''
                if True adds a union_parent item to each coco image and
                video that indicate which file it is from
                '''))

        io_workers = scfg.Value('avail-2', help=ub.paragraph(
            '''
            number of workers to load input datasets. By default will use
            available CPUs minus 2.
            '''))

        compress = scfg.Value('auto', help='if True writes results with compression')

        __epilog__ = """
        Example Usage:
            kwcoco union --src special:shapes8 special:shapes1 --dst=combo.kwcoco.json
        """

    @classmethod
    def main(cls, cmdline=True, **kw):
        """
        Example:
            >>> from kwcoco.cli.coco_union import *  # NOQA
            >>> import ubelt as ub
            >>> dpath = ub.Path.appdir('kwcoco/tests/cli/union').ensuredir()
            >>> dst_fpath = dpath / 'combo.kwcoco.json'
            >>> kw = {
            >>>     'src': ['special:shapes8', 'special:shapes1'],
            >>>     'dst': dst_fpath
            >>> }
            >>> cmdline = False
            >>> cls = CocoUnionCLI
            >>> cls.main(cmdline, **kw)
        """
        config = cls.CLIConfig.cli(data=kw, cmdline=cmdline)
        import kwcoco
        print('config = {}'.format(ub.urepr(config, nl=1)))

        if config.src is None:
            raise Exception('must specify sources: {}'.format(config.src))

        if len(config.src) == 0:
            raise ValueError('Must provide at least one input dataset')

        from kwcoco.util.util_parallel import coerce_num_workers
        io_workers = config.io_workers
        io_workers = coerce_num_workers(io_workers)
        io_workers = min(io_workers, len(config.src))
        if io_workers == 1:
            io_workers = 0

        if config.absolute:
            postprocess = _postprocess_absolute
        else:
            postprocess = None

        datasets = list(kwcoco.CocoDataset.coerce_multiple(
            config.src, postprocess=postprocess, ordered=True,
            workers=io_workers, mode='process', autobuild=False,
        ))

        print('Finished loading. Starting union.')
        combo = kwcoco.CocoDataset.union(
            *datasets,
            remember_parent=config.remember_parent)

        out_fpath = config.dst
        out_dpath = ub.Path(out_fpath).parent

        if not config.absolute:
            # Handle the case where the output is not in the same path as the
            # inputs.
            curr_bundle = ub.Path(combo.bundle_dpath)
            if curr_bundle != out_dpath:
                combo.reroot(out_dpath, check=False)

        if out_dpath:
            ub.ensuredir(out_dpath)
        print('Writing to out_fpath = {!r}'.format(out_fpath))
        combo.fpath = out_fpath
        dumpkw = {
            'newlines': True,
            'compress': config.compress,
        }
        combo.dump(combo.fpath, **dumpkw)


def _postprocess_absolute(dset):
    dset.reroot(absolute=True)

_CLI = CocoUnionCLI

if __name__ == '__main__':
    """
    CommandLine:
        python -m kwcoco.cli.coco_union
    """
    _CLI._main()
