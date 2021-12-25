import scriptconfig as scfg
import ubelt as ub


class CocoToyDataCLI(object):
    name = 'toydata'

    class CLIConfig(scfg.Config):
        """
        Create COCO toydata for demo and testing purposes.
        """
        default = {
            'key': scfg.Value('shapes8', help=ub.paragraph(
                '''
                Special demodata code. Basic options that define which flavor
                of demodata to generate are: `photos`, `shapes`, and
                `vidshapes`. A numeric suffix e.g. `vidshapes8` can be
                specified to indicate the size of the generated demo dataset.
                There are other special suffixes that are available.
                See the code in :method:`kwcoco.CocoDataset.demo` for details
                on what is allowed.

                As a quick summary: the vidshapes key is the most robust and
                mature demodata set, and here are several useful variants of
                the vidshapes key.

                (1) vidshapes8 - the 8 suffix is the number of videos in this case.
                (2) vidshapes8-multispectral - generate 8 multispectral videos.
                (3) vidshapes8-msi - msi is an alias for multispectral.
                (4) vidshapes8-frames5 - generate 8 videos with 5 frames each.
                (4) vidshapes2-speed0.1-frames7 - generate 2 videos with 7
                    frames where the objects move with with a speed of 0.1.
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

            'use_cache': scfg.Value(True, help=ub.paragraph(
                '''
                if False, this will force the dataset to be regenerated.
                Otherwise, it will only regenerate the data if it doesn't
                already exist.
                ''')),

            'verbose': scfg.Value(False, help=ub.paragraph(
                ''' Verbosity ''')),
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
        print('config = {}'.format(ub.repr2(dict(config), nl=1)))

        demo_kwargs = {
            'use_cache': config['use_cache'],
            'verbose': config['verbose'],
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
