#!/usr/bin/env python
import ubelt as ub
import scriptconfig as scfg


class CocoGrabCLI:
    """
    """
    name = 'grab'

    class CLIConfig(scfg.Config):
        """
        Grab standard datasets.

        Example:
            kwcoco grab cifar10 camvid
        """

        __default__ = {
            'names': scfg.Value([], nargs='+', position=1, help=ub.paragraph(
                '''
                Dataset names to grab. Valid values are cifar10, cifar100,
                domainnet, spacenet7, and camvid.
                '''
            )),

            'dpath': scfg.Path(
                ub.Path.appdir('kwcoco', 'data', type='cache'), help=ub.paragraph(
                    '''
                    Download directory
                    '''))
        }

    @classmethod
    def main(cls, cmdline=True, **kw):
        from kwcoco.data import grab_cifar
        from kwcoco.data import grab_camvid
        from kwcoco.data import grab_domainnet
        config = cls.CLIConfig(kw, cmdline=cmdline)
        print('config = {}'.format(ub.urepr(dict(config), nl=1)))

        ensured = []
        names = config['names']
        print('names = {!r}'.format(names))

        # TODO: standardize this interface, allow specificaiton of dpath
        # everywhere

        if 'camvid' in names:
            dset = grab_camvid.grab_coco_camvid()
            ensured.append(dset)

        if 'cifar10' in names:
            dset = grab_cifar.convert_cifar10()
            ensured.append(dset)

        if 'cifar100' in names:
            dset = grab_cifar.convert_cifar100()
            ensured.append(dset)

        if 'domainnet' in names:
            dsets = grab_domainnet.grab_domain_net()
            for dset in dsets:
                ensured.append(dset)

        if 'spacenet7' in names:
            from kwcoco.data import grab_spacenet
            dsets = grab_spacenet.grab_spacenet7(config['dpath'])
            for dset in dsets:
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
