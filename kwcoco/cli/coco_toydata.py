import scriptconfig as scfg


class CocoToyDataCLI(object):
    name = 'toydata'

    class CLIConfig(scfg.Config):
        """
        Create COCO toydata
        """
        default = {
            'key': scfg.Value('shapes8', help='special demodata code'),
            'dst': scfg.Value('test.mscoco.json', help='output path'),
        }

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
