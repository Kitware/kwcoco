"""
Downloads and converts CIFAR 10 and CIFAR 100 to kwcoco format
"""
import ubelt as ub
import os
import pickle
import torchvision
import kwimage


def _convert_cifar_to_kwcoco(dpath, cifar_dset, cifar_name, classes):
    import kwcoco

    bundle_dpath = (ub.Path(dpath) / cifar_name).ensuredir()
    img_dpath = (bundle_dpath / 'images').ensuredir()

    CONVERSION_VERSION = 3

    stamp = ub.CacheStamp('convert_cifar', dpath=dpath,
                          depends=[cifar_name, CONVERSION_VERSION], verbose=3)
    if stamp.expired():

        coco_dset = kwcoco.CocoDataset(bundle_dpath=bundle_dpath)
        coco_dset.fpath = bundle_dpath / '{}.kwcoco.json'.format(cifar_name)

        for cx, catname in enumerate(classes):
            cid = cifar_dset.class_to_idx[catname]
            coco_dset.add_category(id=cid, name=catname)

        data_label_iter = zip(
            cifar_dset.data,
            cifar_dset.targets)

        prog = ub.ProgIter(data_label_iter, total=len(cifar_dset.targets),
                           desc=f'convert {cifar_name}')

        for gx, (imdata, cidx) in enumerate(prog):
            catname = classes[cidx]
            name = f'img_{gx:08d}'
            subdir = (img_dpath / catname).ensuredir()
            fpath = subdir / f'{name}.png'

            fname = fpath.relative_to(bundle_dpath)

            if not fpath.exists():
                kwimage.imwrite(fpath, imdata)

            height, width = imdata.shape[0:2]

            gid = coco_dset.add_image(file_name=fname, id=gx, name=name,
                                      channels='red|green|blue',
                                      num_overviews=0, width=width,
                                      height=height)

            cid = coco_dset.index.name_to_cat[catname]['id']
            coco_dset.add_annotation(image_id=gid, bbox=[0, 0, width, height],
                                     category_id=cid)

        print('write coco_dset.fpath = {!r}'.format(coco_dset.fpath))
        stamp.renew()
        coco_dset.dump(coco_dset.fpath, newlines=True)
    else:
        fpath = bundle_dpath / f'{cifar_name}.kwcoco.json'
        coco_dset = kwcoco.CocoDataset(fpath)

    coco_dset.tag = cifar_name
    return coco_dset


def convert_cifar10(dpath=None):
    if dpath is None:
        dpath = ub.Path.appdir('kwcoco/data').ensuredir()
    else:
        dpath = ub.Path(dpath).ensuredir()
    # For some reason the torchvision objects dont have the label names
    # in the dataset. But the download directory will have them.
    expected_classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
        'horse', 'ship', 'truck',
    ]
    DATASET = torchvision.datasets.CIFAR10

    download_dpath = (dpath / 'download').ensuredir()

    cifar_train_dset = DATASET(root=download_dpath, download=True, train=True)
    meta_fpath = ub.Path(cifar_train_dset.root) / cifar_train_dset.base_folder / 'batches.meta'
    with open(meta_fpath, 'rb') as file:
        meta_dict = pickle.load(file)
    classes = meta_dict['label_names']
    assert classes == expected_classes
    cifar_name = 'cifar10-train'
    train_coco_dset = _convert_cifar_to_kwcoco(dpath, cifar_train_dset,
                                               cifar_name, classes)

    cifar_test_dset = DATASET(root=download_dpath, download=True, train=False)
    cifar_name = 'cifar10-test'
    test_coco_dset = _convert_cifar_to_kwcoco(dpath, cifar_test_dset,
                                              cifar_name, classes)
    return train_coco_dset, test_coco_dset


def convert_cifar100(dpath=None):
    if dpath is None:
        dpath = ub.Path.appdir('kwcoco/data').ensuredir()
    expected_classes = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
        'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
        'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
        'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
        'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
        'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',
        'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
        'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
        'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
        'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
        'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
        'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
        'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm']
    cifar_name = 'cifar100-train'
    DATASET = torchvision.datasets.CIFAR100
    cifar_dset = DATASET(
        root=ub.ensuredir((dpath, 'download')), download=True, train=True)
    meta_fpath = os.path.join(cifar_dset.root, cifar_dset.base_folder, 'meta')
    meta_dict = pickle.load(open(meta_fpath, 'rb'))
    classes = meta_dict['fine_label_names']
    assert classes == expected_classes
    train_coco_dset = _convert_cifar_to_kwcoco(dpath, cifar_dset, cifar_name,
                                               classes)
    cifar_name = 'cifar100-test'
    DATASET = torchvision.datasets.CIFAR100
    cifar_dset = DATASET(
        root=ub.ensuredir((dpath, 'download')), download=True, train=False)
    meta_fpath = os.path.join(cifar_dset.root, cifar_dset.base_folder, 'meta')
    meta_dict = pickle.load(open(meta_fpath, 'rb'))
    classes = meta_dict['fine_label_names']
    test_coco_dset = _convert_cifar_to_kwcoco(dpath, cifar_dset, cifar_name,
                                              classes)
    return [train_coco_dset, test_coco_dset]


def main():
    import scriptconfig as scfg
    class GrabCIFAR_Config(scfg.Config):
        """
        Ensure the CIFAR dataset exists in kwcoco format and prints its
        location and a bit of info.
        """
        __default__ = {
            'dpath': scfg.Path(
                ub.Path.appdir('kwcoco/data'),
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

    for key, dsets in items.items():
        for dset in dsets:
            print('dset = {!r}'.format(dset))

    for key, dsets in items.items():
        for dset in dsets:
            print('{} dset.fpath = {!r}'.format(key, dset.fpath))

    return items


if __name__ == '__main__':
    """
    CommandLine:
        python -m kwcoco.data.grab_cifar
    """
    main()
