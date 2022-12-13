
def make_pycocotools_compliant(fpath):
    import kwcoco
    import kwimage
    import ubelt as ub

    print('Reading fpath = {!r}'.format(fpath))
    dset = kwcoco.CocoDataset(fpath)

    dset._ensure_imgsize(workers=8)

    for ann in ub.ProgIter(dset.dataset['annotations'], desc='update anns'):
        if 'iscrowd' not in ann:
            ann['iscrowd'] = False

        if 'ignore' not in ann:
            ann['ignore'] = ann.get('weight', 1.0) < .5

        if 'area' not in ann:
            # Use segmentation if available
            if 'segmentation' in ann:
                poly = kwimage.Polygon.from_coco(ann['segmentation'][0])
                ann['area'] = float(poly.to_shapely().area)
            else:
                x, y, w, h = ann['bbox']
                ann['area'] = w * h

    dset.dump(dset.fpath, newlines=True)


if __name__ == '__main__':
    """
    CommandLine:
        python make_pycocotools_compliant.py --help
    """
    import fire
    fire.Fire(make_pycocotools_compliant)
