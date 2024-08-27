#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class ConvertImagenetCLI(scfg.DataConfig):
    """
    Converts the standard ILSVR 2017 dataset from PascalVOC to KWCoco.

    This folder should have the following structure:

    Note:
        The map from synsets to category names isn't obvious from the official
        data. Using the following map to compute it.

        https://gist.githubusercontent.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57/raw/aa66dd9dbf6b56649fa3fab83659b2acbf3cbfd1/map_clsloc.txt

    .. code::

        Annotations/CLS-LOC/test/*.xml
        Annotations/CLS-LOC/train/n*/*.xml
        Annotations/CLS-LOC/val/*.xml

        Data/CLS-LOC/test/*.JPEG
        Data/CLS-LOC/train/n*/*.JPEG
        Data/CLS-LOC/val/*.JPEG

    Example
    -------
    python ~/code/kwcoco/dev/poc/convert_imagenet.py /data/store/data/ImageNet/ILSVRC

    """
    ilsvrc_dpath = scfg.Value(None, help=ub.codeblock(
        '''
        Path to ILSVRC folder.
        '''), position=1)

    workers = scfg.Value('auto', help='number of IO workers')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        import rich
        from rich.markup import escape
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))

        ilsvrc_dpath = config.ilsvrc_dpath
        import kwutil
        workers = kwutil.util_parallel.coerce_num_workers(config.workers)
        print(f'workers={workers}')

        split = 'train'
        convert_imagenet(ilsvrc_dpath, split=split, workers=workers)

        split = 'val'
        convert_imagenet(ilsvrc_dpath, split=split, workers=workers)

__cli__ = ConvertImagenetCLI


def convert_imagenet(ilsvrc_dpath, split, workers=0):
    import kwcoco
    ilsvrc_dpath = ub.Path(ilsvrc_dpath)

    annot_dpath = ilsvrc_dpath / 'Annotations/CLS-LOC' / split
    data_dpath = ilsvrc_dpath / 'Data/CLS-LOC' / split

    # if split == 'train':
    #     split_fpath = ilsvrc_dpath / 'ImageSets/CLS-LOC' / (split + '_cls.txt')
    # else:
    #     split_fpath = ilsvrc_dpath / 'ImageSets/CLS-LOC' / (split + '.txt')

    assert annot_dpath.exists()
    assert data_dpath.exists()

    # assert split_fpath.exists()
    # print(split_fpath.read_text())
    image_xml_fpaths = sorted(annot_dpath.glob('**/*.xml'))
    print(f'Found {len(image_xml_fpaths)} annotation XMLs')

    dset = kwcoco.CocoDataset()

    # The map from synsets to category names isn't obvious from the official
    # data. Using the following map to compute it.
    cat_lut_fpath = ub.grabdata(
        'https://gist.githubusercontent.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57/raw/aa66dd9dbf6b56649fa3fab83659b2acbf3cbfd1/map_clsloc.txt',
        hash_prefix='794a3e693266268a9b4080df1d6eda52247621667f0912420dd1ffbcf39d3df662dc2b7b6a8146b7863975a540559400210d21684d723907bd701530fe3fde90',
        hasher='sha512'
    )
    synset_to_cid = {}
    munged_catnames = set()
    lines = ub.Path(cat_lut_fpath).read_text().strip().split('\n')
    for line in lines:
        synset_name, cat_id, catname = line.split(' ', 2)
        cat_id = int(cat_id)
        unique_catname = catname
        idx = 2
        while unique_catname in munged_catnames:
            unique_catname = catname + f'_{idx}'
            idx += 1
        dset.add_category(id=cat_id, name=unique_catname, synset=synset_name)
        synset_to_cid[synset_name] = cat_id
        munged_catnames.add(unique_catname)

    jobs = ub.JobPool(mode='process', max_workers=workers)
    with jobs:
        for xml_fpath in ub.ProgIter(image_xml_fpaths, desc=f'submit convert {split} jobs'):
            jobs.submit(_read_voc_image, data_dpath, xml_fpath)

        for job in ub.ProgIter(jobs, desc=f'collect {split} jobs', homogeneous=False):
            img, anns = job.result()
            img['channels'] = 'red|green|blue'  # hack in channels to make geowatch happy
            image_id = dset.add_image(**img)
            for ann in anns:
                ann['image_id'] = image_id
                ann['category_id'] = synset_to_cid[ann.pop('category_name')]
                dset.add_annotation(**ann)
    dset.reroot(ilsvrc_dpath)

    dset.fpath = ilsvrc_dpath / f'{split}.kwcoco.zip'
    print(f'Write dset.fpath={dset.fpath}')
    dset.dump()


def _read_voc_image(data_dpath, xml_fpath):
    # print(xml_fpath.read_text())
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_fpath)

    dname = tree.find('folder').text
    fname = tree.find('filename').text

    try:
        fpath = data_dpath / dname / (fname + '.JPEG')
        assert fpath.exists()
    except AssertionError:
        # hack for imagenet
        fpath = data_dpath / ('n' + dname) / (fname + '.JPEG')
        try:
            assert fpath.exists()
        except Exception:
            if dname == 'val':
                fpath = data_dpath / (fname + '.JPEG')
            else:
                raise

    img = {
        'file_name': fpath,
        'width': int(tree.find('size').find('width').text),
        'height': int(tree.find('size').find('height').text),
        'depth': int(tree.find('size').find('depth').text),
    }
    try:
        img['segmented'] = int(tree.find('segmented').text)
    except Exception:
        ...
    try:
        img['source'] = {
            elem.tag: elem.text
            for elem in list(tree.find('source'))
        }
    except Exception:
        ...

    assert img.pop('depth') == 3

    owner = tree.find('owner')
    if owner is not None:
        img['owner'] = {
            elem.tag: elem.text
            for elem in list(owner)
        }

    anns = []
    for obj in tree.findall('object'):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1

        diffc = obj.find('difficult')
        difficult = 0 if diffc is None else int(diffc.text)

        catname = obj.find('name').text.lower().strip()
        w = x2 - x1
        h = y2 - y1
        ann = {
            'bbox': [x1, y1, w, h],
            'category_name': catname,
            'difficult': difficult,
            'weight': 1.0 - difficult,
        }
        anns.append(ann)

    return img, anns


if __name__ == '__main__':
    __cli__.main()
