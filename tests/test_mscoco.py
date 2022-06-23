def grab_mscoco_annotations():
    """
    TODO: move to a grabdata script
    """
    import ubelt as ub
    from kwcoco.util import util_archive
    dpath = ub.Path.appdir('kwcoco/data/mscoco').ensuredir()

    mscoco_urls = {
        'trainval2017': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        'trainval2014': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
        'testinfo2014': 'http://images.cocodataset.org/annotations/image_info_test2014.zip',
    }
    mscoco_fpaths = {}
    keys = ['trainval2017']
    for key in keys:
        url = mscoco_urls[key]
        zip_fpath = ub.grabdata(url, dpath=dpath)
        archive = util_archive.Archive.coerce(zip_fpath)
        _fpaths = archive.extractall(output_dpath=dpath)
        _fpaths = [ub.Path(p) for p in _fpaths]
        fpaths = {p.name: p for p in _fpaths}
        mscoco_fpaths[key] = fpaths
    return mscoco_fpaths


def test_standard_coco_dataset():
    import pytest
    pytest.skip('slow test')

    import kwcoco
    mscoco_fpaths = grab_mscoco_annotations()

    fpath = mscoco_fpaths['trainval2017']['instances_val2017.json']
    dset = kwcoco.CocoDataset(fpath)
    _test_dset_api(dset)


def _test_dset_api(dset):
    """
    Run various API calls on the dataset to validate everything works.
    """
    dset.validate(missing=False)

    stats = dset.stats()

    images = dset.images()
    videos = dset.videos()
    annots = dset.annots()
    categories = dset.categories()

    assert stats['basic']['n_anns'] == len(annots)
    assert stats['basic']['n_imgs'] == len(images)
    assert stats['basic']['n_videos'] == len(videos)
    assert stats['basic']['n_cats'] == len(categories)
