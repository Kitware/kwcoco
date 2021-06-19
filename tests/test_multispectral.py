"""
Test cases for multispectral data
"""
from os.path import dirname


def test_multispectral_name_to_img_index():
    import kwcoco
    dset = kwcoco.CocoDataset.demo('vidshapes1-multispectral', num_frames=5,
                                   verbose=0, rng=None)
    # This is the first use-case of image names
    assert len(dset.index.file_name_to_img) == 0, (
        'the multispectral demo case has no "base" image')
    assert len(dset.index.name_to_img) == len(dset.imgs) == 5
    dset.remove_images([1])
    assert len(dset.index.name_to_img) == len(dset.imgs) == 4
    dset.remove_videos([1])
    assert len(dset.index.name_to_img) == len(dset.imgs) == 0


def test_multispectral_union_absolute():
    import kwcoco
    dset1 = kwcoco.CocoDataset.demo('vidshapes1-multispectral', num_frames=2,
                                    verbose=0, rng=0)
    dset2 = kwcoco.CocoDataset.demo('vidshapes1-multispectral', num_frames=2,
                                    verbose=0, rng=1)
    # Ensure absolute rooted datasets
    dset1.reroot(absolute=True)
    dset2.reroot(absolute=True)

    others = [dset1, dset2]
    combo = kwcoco.CocoDataset.union(*others)
    combo.validate()

    stats1 = dset1.basic_stats()
    stats2 = dset2.basic_stats()
    stats3 = combo.basic_stats()

    assert len(combo.index.name_to_img.keys()) == 4
    assert dset1.get_image_fpath(1, 'B1') == combo.get_image_fpath(1, 'B1')
    assert dset1._get_img_auxiliary(1, 'B1')['file_name'] == combo._get_img_auxiliary(1, 'B1')['file_name'], (
        'union of absolute files should not change file names')
    assert stats1['n_anns'] + stats2['n_anns'] == stats3['n_anns']
    assert stats1['n_imgs'] + stats2['n_imgs'] == stats3['n_imgs']
    assert combo.bundle_dpath == dirname(dset1.bundle_dpath)


def test_multispectral_union_relative():
    import kwcoco
    dset1 = kwcoco.CocoDataset.demo('vidshapes1-multispectral', num_frames=2,
                                    verbose=0, rng=0)
    dset2 = kwcoco.CocoDataset.demo('vidshapes1-multispectral', num_frames=2,
                                    verbose=0, rng=1)
    # Ensure relative rooted datasets
    dset1.reroot(absolute=False)
    dset2.reroot(absolute=False)

    others = [dset1, dset2]
    combo = kwcoco.CocoDataset.union(*others)
    combo.validate()

    stats1 = dset1.basic_stats()
    stats2 = dset2.basic_stats()
    stats3 = combo.basic_stats()

    assert len(combo.index.name_to_img.keys()) == 4
    assert dset1.get_image_fpath(1, 'B1') == combo.get_image_fpath(1, 'B1')
    assert dset1._get_img_auxiliary(1, 'B1')['file_name'] != combo._get_img_auxiliary(1, 'B1')['file_name'], (
        'relative names should be different')
    assert stats1['n_anns'] + stats2['n_anns'] == stats3['n_anns']
    assert stats1['n_imgs'] + stats2['n_imgs'] == stats3['n_imgs']
    assert combo.bundle_dpath == dirname(dset1.bundle_dpath)


def test_multispectral_sql():
    try:
        import sqlalchemy  # NOQA
    except Exception:
        import pytest
        pytest.skip()

    import numpy as np
    import kwcoco
    import ubelt as ub
    dset1 = kwcoco.CocoDataset.demo('vidshapes1-multispectral')
    dset2 = dset1.view_sql(force_rewrite=True)

    dset2.basic_stats()

    name = ub.peek(dset1.index.name_to_img)
    img_dict = dset2.index.name_to_img[name]
    assert img_dict['name'] == name

    # file_name = ub.peek(dset1.index.file_name_to_img)
    # img_dict = dset2.index.name_to_img[name]
    # assert img_dict['name'] == name

    img1 = dset1.load_image(1, channels='B1')
    img2 = dset2.load_image(1, channels='B1')

    assert np.all(img1 == img2)
