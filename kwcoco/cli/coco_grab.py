#!/usr/bin/env python
"""
TODO:
    - [ ] Access to data via multiple mirrors and a distributed option
    - [ ] Integrate ImageNet ISLVRC 2017
    - [ ] Integrate Tiny-ImageNet
    - [ ] Integrate Mini-ImageNet
    - [ ] Integrate TACO
    - [ ] Integrate ZeroWaste
    - [ ] Integrate ZeroWaste
    - [ ] Integrate MSCOCO 2017
    - [ ] Integrate MNIST
    - [ ] Integrate KITTI
    - [ ] Integrate Cityscapes
    - [ ] Integrate FashionMNIST
    - [ ] Integrate Oxford 102 Flower
    - [ ] Integrate VQA
    - [ ] Integrate LFW
    - [ ] Integrate Urban100
    - [ ] Integrate EruoSAT
    - [ ] Integrate TrackingNet
    - [ ] Integrate MOTChallenge
"""
import ubelt as ub
import scriptconfig as scfg


class CocoGrabCLI(scfg.DataConfig):
    """
    Grab standard datasets in kwcoco form.

    Example:
        kwcoco grab cifar10 camvid
    """
    __command__ = 'grab'

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
        config = cls.cli(data=kw, cmdline=cmdline, strict=True)
        print('config = {}'.format(ub.urepr(dict(config), nl=1)))

        ensured = []
        names = config['names']
        print('names = {!r}'.format(names))

        # TODO: standardize this interface, allow specification of dpath
        # everywhere

        for name in names:
            if 'camvid' == name:
                from kwcoco.data import grab_camvid
                dset = grab_camvid.grab_coco_camvid()
                ensured.append(dset)

            elif 'cifar10' == name:
                from kwcoco.data import grab_cifar
                dsets = grab_cifar.convert_cifar10(dpath=config.dpath)
                ensured.extend(dsets)

            elif 'cifar100' == name:
                from kwcoco.data import grab_cifar
                dsets = grab_cifar.convert_cifar100(dpath=config.dpath)
                ensured.extend(dsets)

            elif 'domainnet' == name:
                from kwcoco.data import grab_domainnet
                dsets = grab_domainnet.grab_domain_net()
                for dset in dsets:
                    ensured.append(dset)

            elif 'spacenet7' == name:
                from kwcoco.data import grab_spacenet
                dsets = grab_spacenet.grab_spacenet7(config['dpath'])
                for dset in dsets:
                    ensured.append(dset)
            else:
                raise Exception(f'Unknown name: {name}')

        # if config['voc']:
        #     from kwcoco.data import grab_voc

        for dset in ensured:
            print('dset = {!r}'.format(dset))

        for dset in ensured:
            print('dset.fpath = {!r}'.format(dset.fpath))


__cli__ = CocoGrabCLI

if __name__ == '__main__':
    __cli__.main()
