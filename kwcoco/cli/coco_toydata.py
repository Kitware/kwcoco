import scriptconfig as scfg
import ubelt as ub


class CocoToyDataCLI(object):
    name = 'toydata'

    class CLIConfig(scfg.Config):
        """
        Create COCO toydata
        """
        default = {
            'key': scfg.Value('shapes8', help=ub.paragraph(
                '''
                Special demodata code. Available options are:
                photos, shapes8, vidshapes8, vidshapes8-multispectral

                Note that the number (e.g. 8) at the end of the "shapes"
                datasets can be replaced by any number to specify the number of
                images generated in the toy dataset.
                '''), position=1),

            'dst': scfg.Value(None, help=ub.paragraph(
                '''
                Output path for the final kwcoco json file.
                Note, that even when given, a data.kwcoco.json file
                will also be generated in a bundle_dpath.
                ''')),

            'bundle_dpath': scfg.Value(None, help=ub.paragraph(
                '''
                Creates a bundled dataset in the specified location.
                If unspecified, a bundle name is generated based on the
                toydata config.
                ''')),

            'use_cache': scfg.Value(True)
        }
        epilog = """
        Example Usage:
            kwcoco toydata --key=shapes8 --dst=toydata.kwcoco.json

            kwcoco toydata --key=shapes8 --bundle_dpath=my_test_bundle_v1
            kwcoco toydata --key=shapes8 --bundle_dpath=my_test_bundle_v1

            kwcoco toydata \
                --key=shapes8 \
                --dst=./shapes8.kwcoco/dataset.kwcoco.json

        TODO:
            - [ ] allow specification of images directory
        """

    @classmethod
    def main(cls, cmdline=True, **kw):
        """
        Example:
            >>> kw = {'key': 'shapes8', 'dst': 'test.json'}
            >>> cmdline = False
            >>> cls = CocoToyDataCLI
            >>> cls.main(cmdline, **kw)
        """
        import kwcoco
        config = cls.CLIConfig(kw, cmdline=cmdline)

        demo_kwargs = {
            'use_cache': config['use_cache'],
        }

        if config['bundle_dpath'] is not None:
            bundle_dpath = config['bundle_dpath']
            dset = kwcoco.CocoDataset.demo(config['key'],
                                           bundle_dpath=bundle_dpath,
                                           **demo_kwargs)
            # dset.reroot(absolute=True)
        else:
            if config['dst'] is not None:
                fpath = config['dst']
                from os.path import dirname
                dpath = dirname(fpath)
                dset = kwcoco.CocoDataset.demo(config['key'],
                                               dpath=dpath,
                                               **demo_kwargs)
                dset.fpath = fpath
            else:
                dset = kwcoco.CocoDataset.demo(config['key'],
                                               **demo_kwargs)
            dset.reroot(absolute=True)

        if config['dst'] is not None:
            print('dset.fpath = {!r}'.format(dset.fpath))
            print('Writing to dset.fpath = {!r}'.format(dset.fpath))
            dset.dump(dset.fpath, newlines=True)

_CLI = CocoToyDataCLI

if __name__ == '__main__':
    _CLI._main()
