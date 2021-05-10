import scriptconfig as scfg
import ubelt as ub


class CocoToyDataCLI(object):
    name = 'toydata'

    class CLIConfig(scfg.Config):
        """
        Create COCO toydata
        """
        default = {
            'key': scfg.Value('shapes8', help='special demodata code', position=1),

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

        if config['bundle_dpath'] is not None:
            bundle_dpath = config['bundle_dpath']
            dset = kwcoco.CocoDataset.demo(config['key'],
                                           bundle_dpath=bundle_dpath,
                                           use_cache=config['use_cache'])
            # dset.reroot(absolute=True)
        else:
            if config['dst'] is not None:
                fpath = config['dst']
                from os.path import dirname
                dpath = dirname(fpath)
                dset = kwcoco.CocoDataset.demo(config['key'],
                                               dpath=dpath,
                                               use_cache=config['use_cache'])
                dset.fpath = fpath
            else:
                dset = kwcoco.CocoDataset.demo(config['key'],
                                               use_cache=config['use_cache'])
            dset.reroot(absolute=True)

        if config['dst'] is not None:
            print('dset.fpath = {!r}'.format(dset.fpath))
            print('Writing to dset.fpath = {!r}'.format(dset.fpath))
            dset.dump(dset.fpath, newlines=True)

_CLI = CocoToyDataCLI

if __name__ == '__main__':
    _CLI._main()
