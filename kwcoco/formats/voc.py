"""
Helpers for the Pascal VOC format. (used by ImageNet)
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import kwcoco


def read_voc_image(
    xml_fpath: str | os.PathLike[str],
    data_dpath: str | os.PathLike[str] = '.',
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # print(xml_fpath.read_text())
    import ubelt as ub
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_fpath)

    data_dpath = ub.Path(data_dpath)
    dname = tree.find('folder').text  # type: ignore
    fname = tree.find('filename').text  # type: ignore

    # try:
    fpath = data_dpath / dname / fname  # type: ignore
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
            'width': int(tree.find('size').find('width').text),  # type: ignore
            'height': int(tree.find('size').find('height').text),  # type: ignore
            'depth': int(tree.find('size').find('depth').text),  # type: ignore
        }
        img['orig_xml_fpath'] = os.fspath(xml_fpath)
        try:
            img['segmented'] = int(tree.find('segmented').text)  # type: ignore
        except Exception:
            ...
        try:
            img['source'] = {elem.tag: elem.text for elem in list(tree.find('source'))}  # type: ignore
        except Exception:
            ...

        assert img.pop('depth') == 3

        owner = tree.find('owner')
        if owner is not None:
            img['owner'] = {elem.tag: elem.text for elem in list(owner)}

        anns = []
        for obj in tree.findall('object'):
            bbox = obj.find('bndbox')
            if bbox is not None:
                # Make pixel indexes 0-based
                x1 = float(bbox.find('xmin').text) - 1  # type: ignore
                y1 = float(bbox.find('ymin').text) - 1  # type: ignore
                x2 = float(bbox.find('xmax').text) - 1  # type: ignore
                y2 = float(bbox.find('ymax').text) - 1  # type: ignore
                w = x2 - x1
                h = y2 - y1
                bbox = [x1, y1, w, h]
            else:
                bbox is None

            diffc = obj.find('difficult')

            catname_xml = obj.find('name')
            if catname_xml is not None:
                catname = obj.find('name').text.lower().strip()  # type: ignore
            else:
                catname = None
            ann = {}
            if bbox is not None:
                ann['bbox'] = bbox
            if catname is not None:
                ann['category_name'] = catname
            if diffc is not None:
                difficult = 0 if diffc is None else int(diffc.text)  # type: ignore
                ann.update(
                    {
                        'difficult': difficult,
                        'weight': 1.0 - difficult,
                    }
                )
            anns.append(ann)
    except Exception:
        print(f'Error reading xml_fpath={xml_fpath}')
        raise

    return img, anns


def add_vocdata_to_coco(
    image_xml_fpaths: Iterable[str | os.PathLike[str]],
    data_dpath: str | os.PathLike[str],
    workers: int = 0,
    dset: kwcoco.CocoDataset | None = None,
) -> kwcoco.CocoDataset:
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

        for job in ub.ProgIter(
            jobs, desc='collect VOC convert jobs', homogeneous=False
        ):
            img, anns = job.result()
            image_id = dset.add_image(**img)
            for ann in anns:
                ann['image_id'] = image_id
                catname = ann.pop('category_name', None)
                if catname is not None:
                    ann['category_id'] = dset.ensure_category(catname)
                dset.add_annotation(**ann)
    return dset
