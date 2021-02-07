import scriptconfig as scfg


class CocoToyDataCLI(object):
    name = 'toydata'

    class CLIConfig(scfg.Config):
        """
        Create COCO toydata
        """
        default = {
            'key': scfg.Value('shapes8', help='special demodata code', position=1),

            'dst': scfg.Value('test.mscoco.json', help='output path', position=2),

            'image_root': scfg.Value(None, help='path to output the images to')

            'bundle_dpath': scfg.Value(None, help=ub.paragraph(
                '''
                If specified, overwrites the `dst` and `img_root` parameters
                and creates a bundled dataset.

                '''),
        }
        epilog = """
        Example Usage:
            kwcoco toydata --key=shapes8 --dst=toydata.mscoco.json

            kwcoco toydata \
                --key=shapes8 \
                --dst=./shapes8.kwcoco/dataset.kwcoco.json
                --image_root=./shapes8.kwcoco/.assets/images

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

        dset = kwcoco.CocoDataset.demo(config['key'])
        out_fpath = config['dst']
        print('Writing to out_fpath = {!r}'.format(out_fpath))
        dset.fpath = out_fpath
        dset.dump(dset.fpath, newlines=True)

_CLI = CocoToyDataCLI

if __name__ == '__main__':
    _CLI._main()
