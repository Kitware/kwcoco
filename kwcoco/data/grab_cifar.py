"""
Downloads and converts CIFAR 10 and CIFAR 100 to kwcoco format
"""
import ubelt as ub
import os
import pickle
import torchvision
import kwimage


def _convert_cifar_x(dpath, cifar_dset, cifar_name, classes):
    import kwcoco

    bundle_dpath = ub.ensuredir((dpath, cifar_name))
    img_dpath = ub.ensuredir((bundle_dpath, 'images'))

    stamp = ub.CacheStamp('convert_cifar', dpath=dpath,
                          depends=[cifar_name], verbose=3)
    if stamp.expired():

        coco_dset = kwcoco.CocoDataset(bundle_dpath=bundle_dpath)
        coco_dset.fpath = os.path.join(
            bundle_dpath, '{}.kwcoco.json'.format(cifar_name))

        for cx, catname in enumerate(classes):
            cid = cifar_dset.class_to_idx[catname]
            coco_dset.add_category(id=cid, name=catname)

        data_label_iter = zip(
            cifar_dset.data,
            cifar_dset.targets)

        prog = ub.ProgIter(data_label_iter, total=len(cifar_dset.targets),
                           desc='convert {}'.format(cifar_name))

        for gx, (imdata, cidx) in enumerate(prog):
            catname = classes[cidx]
            name = 'img_{:08d}'.format(gx)
            subdir = ub.ensuredir((img_dpath, catname))
            fpath = os.path.join(subdir, '{}.png'.format(name))

            fname = os.path.relpath(fpath, bundle_dpath)

            if not os.path.exists(fpath):
                kwimage.imwrite(fpath, imdata)

            height, width = imdata.shape[0:2]

            gid = coco_dset.add_image(file_name=fname, id=gx, name=name,
                                      width=width, height=height)

            cid = coco_dset.index.name_to_cat[catname]['id']
            coco_dset.add_annotation(image_id=gid, bbox=[0, 0, width, height],
                                     category_id=cid)

        print('write coco_dset.fpath = {!r}'.format(coco_dset.fpath))
        stamp.renew()
        coco_dset.dump(coco_dset.fpath, newlines=True)
    else:
        fpath = os.path.join(bundle_dpath, '{}.kwcoco.json'.format(cifar_name))
        coco_dset = kwcoco.CocoDataset(fpath)

    coco_dset.tag = cifar_name
    return coco_dset


def convert_cifar10(dpath=None):
    if dpath is None:
        dpath = ub.ensure_app_cache_dir('kwcoco/data')
    # For some reason the torchvision objects dont have the label names
    # in the dataset. But the download directory will have them.
    # classes = [
    #     'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
    #     'horse', 'ship', 'truck',
    # ]
    DATASET = torchvision.datasets.CIFAR10
    cifar_dset = DATASET(
        root=ub.ensuredir((dpath, 'download')), download=True)
    meta_fpath = os.path.join(cifar_dset.root, cifar_dset.base_folder, 'batches.meta')
    meta_dict = pickle.load(open(meta_fpath, 'rb'))
    classes = meta_dict['label_names']
    cifar_name = 'cifar10'
    coco_dset = _convert_cifar_x(dpath, cifar_dset, cifar_name, classes)
    return coco_dset


def convert_cifar100(dpath=None):
    if dpath is None:
        dpath = ub.ensure_app_cache_dir('kwcoco/data')
    # classes = [
    #     'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
    #     'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
    #     'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    #     'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
    #     'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
    #     'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
    #     'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    #     'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    #     'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',
    #     'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
    #     'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
    #     'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
    #     'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
    #     'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
    #     'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    #     'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    #     'worm']
    cifar_name = 'cifar100'
    DATASET = torchvision.datasets.CIFAR100
    cifar_dset = DATASET(
        root=ub.ensuredir((dpath, 'download')), download=True)
    meta_fpath = os.path.join(cifar_dset.root, cifar_dset.base_folder, 'meta')
    meta_dict = pickle.load(open(meta_fpath, 'rb'))
    classes = meta_dict['fine_label_names']
    coco_dset = _convert_cifar_x(dpath, cifar_dset, cifar_name, classes)
    return coco_dset


def main():
    import scriptconfig as scfg
    class GrabCIFAR_Config(scfg.Config):
        """
        Ensure the CIFAR dataset exists in kwcoco format and prints its
        location and a bit of info.
        """
        default = {
            'dpath': scfg.Path(
                ub.get_app_cache_dir('kwcoco/data'),
                help='download location'),
            'with_10': scfg.Value(True, help='do cifar 10'),
            'with_100': scfg.Value(True, help='do cifar 100'),
        }
    config = GrabCIFAR_Config()
    dpath = config['dpath']

    items = {}
    if config['with_10']:
        coco_cifar10 = convert_cifar10(dpath)
        items['cifar10'] = coco_cifar10
    if config['with_100']:
        coco_cifar100 = convert_cifar100(dpath)
        items['cifar100'] = coco_cifar100

    for key, dset in items.items():
        print('dset = {!r}'.format(dset))

    for key, dset in items.items():
        print('{} dset.fpath = {!r}'.format(key, dset.fpath))

    return items


if __name__ == '__main__':
    """
    CommandLine:
        python -m kwcoco.data.grab_cifar
    """
    main()
