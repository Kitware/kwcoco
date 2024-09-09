"""
Helpers for the Pascal VOC format. (used by ImageNet)
"""


def read_voc_image(xml_fpath, data_dpath='.'):
    # print(xml_fpath.read_text())
    import ubelt as ub
    import os
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_fpath)

    data_dpath = ub.Path(data_dpath)
    dname = tree.find('folder').text
    fname = tree.find('filename').text

    # try:
    fpath = data_dpath / dname / fname
    # assert fpath.exists()
    # except AssertionError:
    #     # hack for imagenet (TODO: can we generalize)
    #     fpath = data_dpath / ('n' + dname) / (fname + '.JPEG')
    #     try:
    #         assert fpath.exists()
    #     except Exception:
    #         if dname == 'val':
    #             fpath = data_dpath / (fname + '.JPEG')
    #         else:
    #             raise

    try:
        img = {
            'file_name': fpath,
            'width': int(tree.find('size').find('width').text),
            'height': int(tree.find('size').find('height').text),
            'depth': int(tree.find('size').find('depth').text),
        }
        img['orig_xml_fpath'] = os.fspath(xml_fpath)
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
            if bbox is not None:
                # Make pixel indexes 0-based
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                w = x2 - x1
                h = y2 - y1
                bbox = [x1, y1, w, h]
            else:
                bbox is None

            diffc = obj.find('difficult')

            catname_xml = obj.find('name')
            if catname_xml is not None:
                catname = obj.find('name').text.lower().strip()
            else:
                catname = None
            ann = {}
            if bbox is not None:
                ann['bbox'] = bbox
            if catname is not None:
                ann['category_name'] = catname
            if diffc is not None:
                difficult = 0 if diffc is None else int(diffc.text)
                ann.update({
                    'difficult': difficult,
                    'weight': 1.0 - difficult,
                })
            anns.append(ann)
    except Exception:
        print(f'Error reading xml_fpath={xml_fpath}')
        raise

    return img, anns


def add_vocdata_to_coco(image_xml_fpaths, data_dpath, workers=0, dset=None):
    """
    Args,:
        dset (CocoDataset | None):
            existing dataset to add to. If not given, then make a new one.
    """
    import kwcoco
    import ubelt as ub
    if dset is None:
        dset = kwcoco.CocoDataset()

    jobs = ub.JobPool(mode='process', max_workers=workers)
    with jobs:
        for xml_fpath in ub.ProgIter(image_xml_fpaths, desc='submit VOC convert jobs'):
            jobs.submit(read_voc_image, xml_fpath, data_dpath)

        for job in ub.ProgIter(jobs, desc='collect VOC convert jobs', homogeneous=False):
            img, anns = job.result()
            image_id = dset.add_image(**img)
            for ann in anns:
                ann['image_id'] = image_id
                catname = ann.pop('category_name', None)
                if catname is not None:
                    ann['category_id'] = dset.ensure_category(catname)
                dset.add_annotation(**ann)
    return dset
