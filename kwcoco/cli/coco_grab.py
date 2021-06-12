import scriptconfig as scfg


class CocoGrabCLI:
    """
    """
    name = 'grab'

    class CLIConfig(scfg.Config):
        """
        Grab standard datasets.

        Example:
            kwcoco grab --cifar10=True --cifar100=True --camvid=True

        TODO:
            - [ ] Make the cli arg setup less wonky
        """

        default = {
            # 'names': scfg.Value([], help='list of dataset names, cifar10, cifar100, or camvid'),
            'cifar10': scfg.Value(False, help='set to True to grab this dataset'),
            'cifar100': scfg.Value(False, help='set to True to grab this dataset'),
            # 'voc': scfg.Value(False, help='set to True to grab this dataset'),
            'camvid': scfg.Value(False, help='set to True to grab this dataset'),
        }

    @classmethod
    def main(cls, cmdline=True, **kw):
        import ubelt as ub
        from kwcoco.data import grab_cifar
        from kwcoco.data import grab_camvid
        config = cls.CLIConfig(kw, cmdline=cmdline)
        print('config = {}'.format(ub.repr2(dict(config), nl=1)))

        ensured = []
        if config['camvid']:
            dset = grab_camvid.grab_coco_camvid()
            ensured.append(dset)

        if config['cifar10']:
            dset = grab_cifar.convert_cifar10()
            ensured.append(dset)

        if config['cifar100']:
            dset = grab_cifar.convert_cifar100()
            ensured.append(dset)

        # if config['voc']:
        #     from kwcoco.data import grab_voc

        for dset in ensured:
            print('dset = {!r}'.format(dset))

        for dset in ensured:
            print('dset.fpath = {!r}'.format(dset.fpath))


_CLI = CocoGrabCLI

if __name__ == '__main__':
    _CLI.main()
