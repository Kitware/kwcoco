

def test_union_with_aux():
    from os.path import join
    test_img1 = {
        'id': 1,
        'name': 'foo',
        'file_name': 'subdir/images/foo.png',
        'auxiliary': [
            {
                'channels': 'ir',
                'file_name': 'subdir/assets/foo.png',
            }
        ]
    }

    test_img2 = {
        'id': 1,
        'name': 'bar',
        'file_name': 'images/bar.png',
        'auxiliary': [
            {
                'channels': 'ir',
                'file_name': 'assets/foo.png',
            }
        ]
    }

    import kwcoco
    dset1 = kwcoco.CocoDataset()
    dset1.add_image(**test_img1)
    dset1.fpath = join('.', 'dset1', 'data.kwcoco.json')

    dset2 = kwcoco.CocoDataset()
    dset2.add_image(**test_img2)
    dset2.fpath = join('.', 'subdir/dset2', 'data.kwcoco.json')

    combo = kwcoco.CocoDataset.union(dset1, dset2)

    assert combo.get_image_fpath(1) == dset1.get_image_fpath(1)
    assert combo.get_image_fpath(1, channels='ir') == dset1.get_image_fpath(1, channels='ir')

    assert combo.get_image_fpath(2) == dset2.get_image_fpath(1)
    assert combo.get_image_fpath(2, channels='ir') == dset2.get_image_fpath(1, channels='ir')


def test_union_subdirs_to_root():
    """
    Test case where unions kwcoco files in the subdirs and then outputs the new
    file into a parent of the subdirs.
    """
    import kwcoco
    dset1 = kwcoco.CocoDataset.demo('vidshapes1')
    dset2 = kwcoco.CocoDataset.demo('vidshapes2')
    dset3 = kwcoco.CocoDataset.demo('vidshapes3')

    import ubelt as ub
    dpath = ub.Path.appdir('kwcoco', 'tests', 'union', 'subdirs1')
    dpath.ensuredir()

    multi_bundle_dpath = (dpath / 'multi_bundle').delete().ensuredir()

    ub.Path(dset1.fpath).parent.copy(multi_bundle_dpath / 'dset1')
    ub.Path(dset2.fpath).parent.copy(multi_bundle_dpath / 'dset2')
    ub.Path(dset3.fpath).parent.copy(multi_bundle_dpath / 'dset3')

    src_fpaths = [
        multi_bundle_dpath / 'dset1/data.kwcoco.json',
        multi_bundle_dpath / 'dset2/data.kwcoco.json',
        multi_bundle_dpath / 'dset3/data.kwcoco.json',
    ]
    dst_fpath = multi_bundle_dpath / 'combo.kwcoco.json'

    from kwcoco.cli import coco_union
    coco_union.__cli__.main(cmdline=0, src=src_fpaths, dst=dst_fpath)

    dst = kwcoco.CocoDataset(dst_fpath)
    assert not any(dst.missing_images())
    dst.validate()
    assert len(dst.videos()) == 6

    img_groups = dst.videos().images
    gids = [grp[0] for grp in img_groups]
    prefixes = {p.split('images')[0] for p in list(dst.images(gids).lookup('file_name'))}
    assert prefixes == {'dset1/_assets/', 'dset2/_assets/', 'dset3/_assets/'}


def test_union_subdirs_to_new_subdir():
    """
    Test case where unions kwcoco files in the subdirs and then outputs the new
    file into a different subdir in the root.
    """
    import kwcoco
    dset1 = kwcoco.CocoDataset.demo('vidshapes1')
    dset2 = kwcoco.CocoDataset.demo('vidshapes2')
    dset3 = kwcoco.CocoDataset.demo('vidshapes3')

    import ubelt as ub
    dpath = ub.Path.appdir('kwcoco', 'tests', 'union', 'subdirs2')
    dpath.ensuredir()

    multi_bundle_dpath = (dpath / 'multi_bundle').delete().ensuredir()

    ub.Path(dset1.fpath).parent.copy(multi_bundle_dpath / 'dset1')
    ub.Path(dset2.fpath).parent.copy(multi_bundle_dpath / 'dset2')
    ub.Path(dset3.fpath).parent.copy(multi_bundle_dpath / 'dset3')

    src_fpaths = [
        multi_bundle_dpath / 'dset1/data.kwcoco.json',
        multi_bundle_dpath / 'dset2/data.kwcoco.json',
        multi_bundle_dpath / 'dset3/data.kwcoco.json',
    ]
    dst_fpath = multi_bundle_dpath / 'new_subdir/combo.kwcoco.json'

    from kwcoco.cli import coco_union
    coco_union.__cli__.main(cmdline=0, src=src_fpaths, dst=dst_fpath)

    dst = kwcoco.CocoDataset(dst_fpath)
    dst.validate()
    assert not any(dst.missing_images())
    assert len(dst.videos()) == 6

    img_groups = dst.videos().images
    gids = [grp[0] for grp in img_groups]
    prefixes = {p.split('images')[0] for p in list(dst.images(gids).lookup('file_name'))}
    assert prefixes == {'../dset1/_assets/', '../dset2/_assets/', '../dset3/_assets/'}


def test_union_subdirs_to_existing_subdir():
    """
    Test case where unions kwcoco files in the subdirs and then outputs the new
    file into a different subdir in the root.
    """
    import kwcoco
    dset1 = kwcoco.CocoDataset.demo('vidshapes1')
    dset2 = kwcoco.CocoDataset.demo('vidshapes2')
    dset3 = kwcoco.CocoDataset.demo('vidshapes3')

    import ubelt as ub
    dpath = ub.Path.appdir('kwcoco', 'tests', 'union', 'subdirs3')
    dpath.ensuredir()

    multi_bundle_dpath = (dpath / 'multi_bundle').delete().ensuredir()

    ub.Path(dset1.fpath).parent.copy(multi_bundle_dpath / 'dset1')
    ub.Path(dset2.fpath).parent.copy(multi_bundle_dpath / 'dset2')
    ub.Path(dset3.fpath).parent.copy(multi_bundle_dpath / 'dset3')

    src_fpaths = [
        multi_bundle_dpath / 'dset1/data.kwcoco.json',
        multi_bundle_dpath / 'dset2/data.kwcoco.json',
        multi_bundle_dpath / 'dset3/data.kwcoco.json',
    ]
    dst_fpath = multi_bundle_dpath / 'dset2/combo.kwcoco.json'

    from kwcoco.cli import coco_union
    coco_union.__cli__.main(cmdline=0, src=src_fpaths, dst=dst_fpath)

    dst = kwcoco.CocoDataset(dst_fpath)
    dst.validate()
    assert not any(dst.missing_images())
    assert len(dst.videos()) == 6

    img_groups = dst.videos().images
    gids = [grp[0] for grp in img_groups]
    prefixes = {p.split('images')[0] for p in list(dst.images(gids).lookup('file_name'))}

    # NOTE: it would be nice if we had just "_assets" instead of
    # '../dset2/_assets/, but it is technically correct, and is what we
    # currently get, so leaving this check in.
    assert prefixes == {'../dset1/_assets/', '../dset2/_assets/', '../dset3/_assets/'}


def test_duplicate_union_with_tracks():
    import kwcoco
    import ubelt as ub

    # Test with disjoint_tracks=True
    dset1 = kwcoco.CocoDataset.demo('vidshapes1')
    combo1 = kwcoco.CocoDataset.union(dset1, dset1, disjoint_tracks=True)
    track_names1 = combo1.tracks().lookup('name')
    annot_trackids1 = combo1.annots().lookup('track_id')
    print(f'track_names1 = {ub.urepr(track_names1, nl=1)}')
    print(f'annot_trackids1 = {ub.urepr(annot_trackids1, nl=1)}')
    assert track_names1 == ['track_001', 'track_002', 'track_001_v001', 'track_002_v001']
    assert annot_trackids1 == [1, 1, 2, 2, 3, 3, 4, 4]

    # Test with disjoint_tracks=False
    combo2 = kwcoco.CocoDataset.union(dset1, dset1, disjoint_tracks=False)
    track_names2 = combo2.tracks().lookup('name')
    annot_trackids2 = combo2.annots().lookup('track_id')
    print(f'track_names2 = {ub.urepr(track_names2, nl=1)}')
    print(f'annot_trackids2 = {ub.urepr(annot_trackids2, nl=1)}')
    assert track_names2 == ['track_001', 'track_002']
    assert annot_trackids2 == [1, 1, 2, 2, 1, 1, 2, 2]


def test_duplicate_union_with_tracks_no_tracktable():
    """
    Test the old case where annots only had track ids and there was no table.
    This test case can be removed if we require that tracks exist in the track
    table.
    """
    import kwcoco
    import ubelt as ub

    # Test with disjoint_tracks=True
    dset1 = kwcoco.CocoDataset.demo('vidshapes1')
    for ann in dset1.dataset['annotations']:
        ann.pop('segmentation')
        ann.pop('keypoints')
    del dset1.dataset['tracks']
    dset1._build_index()

    combo1 = kwcoco.CocoDataset.union(dset1, dset1, disjoint_tracks=True)
    track_names1 = combo1.tracks().lookup('name')
    annot_trackids1 = combo1.annots().lookup('track_id')
    print(f'track_names1 = {ub.urepr(track_names1, nl=0)}')
    print(f'annot_trackids1 = {ub.urepr(annot_trackids1, nl=0)}')
    assert track_names1 == []
    assert annot_trackids1 == [0, 0, 1, 1, 2, 2, 3, 3]

    # Test with disjoint_tracks=False
    combo2 = kwcoco.CocoDataset.union(dset1, dset1, disjoint_tracks=False)
    track_names2 = combo2.tracks().lookup('name')
    annot_trackids2 = combo2.annots().lookup('track_id')
    print(f'track_names2 = {ub.urepr(track_names2, nl=0)}')
    print(f'annot_trackids2 = {ub.urepr(annot_trackids2, nl=0)}')
    assert track_names2 == []
    assert annot_trackids2 == [1, 1, 2, 2, 1, 1, 2, 2]


def test_union_keeps_category_ids_consistent():
    """
    If two kwcoco files have the same exact categories the union should
    preserve their structure.
    """
    import kwcoco
    categories = [
        {'id': 0, 'name': 'cat0'},
        {'id': 1, 'name': 'cat1'},
        {'id': 2, 'name': 'cat2'},
        {'id': 3, 'name': 'cat3'},
        {'id': 4, 'name': 'cat4'},
        {'id': 5, 'name': 'cat5'}
    ]
    dset1 = kwcoco.CocoDataset()
    dset2 = kwcoco.CocoDataset()
    [dset1.ensure_category(**cat) for cat in categories]
    [dset2.ensure_category(**cat) for cat in categories]

    combo_dset = dset1.union(dset2)
    assert combo_dset.dataset['categories'] == categories, (
        "Categories were exactly the same, so they should be presereved"
    )


def test_union_keeps_keypoint_category_ids_consistent():
    import kwcoco
    keypoint_categories = [
        {'id': 8, 'name': 'left_hand', 'reflection_id': 6},
        {'id': 6, 'name': 'right_hand', 'reflection_id': 8},
        {'id': 7, 'name': 'left_foot', 'reflection_id': 5},
        {'id': 5, 'name': 'right_foot', 'reflection_id': 7},
        {'id': 3, 'name': 'head'},
    ]
    dset1 = kwcoco.CocoDataset()
    dset2 = kwcoco.CocoDataset()
    dset1.add_keypoint_categories(keypoint_categories)
    dset2.add_keypoint_categories(keypoint_categories)

    combo_dset = dset1.union(dset2)
    assert combo_dset.dataset['keypoint_categories'] == keypoint_categories, (
        "Keypoint categories were exactly the same, so they should be presereved"
    )


def random_unique_integers(low, high, size, rng=None):
    import kwarray
    import numpy as np
    rng = kwarray.ensure_rng(rng, 'numpy')

    result = set()
    range_size = high - low

    # Generate numbers in chunks until we have enough unique values
    while len(result) < size:
        remaining = size - len(result)
        # Generate a chunk of random numbers (at most remaining or range_size)
        chunk = rng.randint(low, high, size=min(remaining * 2, range_size))
        result.update(chunk)

    # Convert the result to a numpy array
    result = np.array(list(result))
    # Permute the result so numbers are not in order
    result = rng.permutation(result)
    # select exactly `size` elements
    result = result[:size]
    return result


def test_union_with_inconsistent_category_ids():
    import kwcoco
    import kwarray
    rng = kwarray.ensure_rng(0)
    for max_id in [5, 25, 100000]:
        # Create datasets with the same keypoint category names, but different ids
        dsets = []
        num_dsets = 3
        for _ in range(num_dsets):
            ids = random_unique_integers(0, max_id, size=5, rng=rng)
            a, b, c, d, e = rng.permutation(ids)
            categories = [
                {'id': a, 'name': 'cat1'},
                {'id': b, 'name': 'cat2'},
                {'id': c, 'name': 'cat3'},
                {'id': d, 'name': 'cat4'},
                {'id': e, 'name': 'cat5'},
            ]
            dset = kwcoco.CocoDataset()
            dset.add_categories(categories)
            dsets.append(dset)

        combo_dset = kwcoco.CocoDataset.union(*dsets)
        assert combo_dset.dataset['categories'] == dsets[0].dataset['categories'], (
            "categories should agree with the first dataset"
        )


def test_union_with_inconsistent_keypoint_category_ids():
    import kwcoco
    import kwarray
    rng = kwarray.ensure_rng(0)
    for max_id in [5, 25, 100000]:
        # Create datasets with the same keypoint category names, but different ids
        dsets = []
        num_dsets = 3
        for _ in range(num_dsets):
            ids = random_unique_integers(0, max_id, size=5, rng=rng)
            a, b, c, d, e = rng.permutation(ids)
            keypoint_categories = [
                {'id': a, 'name': 'left_hand', 'reflection_id': b},
                {'id': b, 'name': 'right_hand', 'reflection_id': a},
                {'id': c, 'name': 'left_foot', 'reflection_id': d},
                {'id': d, 'name': 'right_foot', 'reflection_id': c},
                {'id': e, 'name': 'head'},
            ]
            dset = kwcoco.CocoDataset()
            dset.add_keypoint_categories(keypoint_categories)
            dsets.append(dset)

        combo_dset = kwcoco.CocoDataset.union(*dsets)
        assert combo_dset.dataset['keypoint_categories'] == dsets[0].dataset['keypoint_categories'], (
            "Keypoint categories should agree with the first dataset"
        )


def test_union_with_disjoint_inconsistent_kp_category_ids():
    import kwcoco
    import kwarray
    rng = kwarray.ensure_rng(132)
    for _ in range(100):
        for max_id in [5, 6, 25, 100000]:
            dsets = []
            num_dsets = 3
            for dataset_idx in range(num_dsets):
                ids = random_unique_integers(0, max_id, size=5, rng=rng)
                a, b, c, d, e = rng.permutation(ids)
                if dataset_idx % 2 == 0:
                    keypoint_categories = [
                        {'id': a, 'name': 'left_hand', 'reflection_id': b},
                        {'id': b, 'name': 'right_hand', 'reflection_id': a},
                    ]
                else:
                    keypoint_categories = [
                        {'id': c, 'name': 'left_foot', 'reflection_id': d},
                        {'id': d, 'name': 'right_foot', 'reflection_id': c},
                        {'id': e, 'name': 'head'},
                    ]
                dset = kwcoco.CocoDataset()
                dset.add_keypoint_categories(keypoint_categories)
                dsets.append(dset)

            combo_dset = kwcoco.CocoDataset.union(*dsets)
            combo_categories = combo_dset.dataset['keypoint_categories']
            assert len(set(c['id'] for c in combo_categories)) == 5, 'new ids must be disjoint'


def test_union_with_disjoint_inconsistent_category_ids():
    import kwcoco
    import kwarray
    rng = kwarray.ensure_rng(13212)
    for _ in range(100):
        for max_id in [5, 6, 25, 100000]:
            dsets = []
            num_dsets = 3
            for dataset_idx in range(num_dsets):
                ids = random_unique_integers(0, max_id, size=5, rng=rng)
                a, b, c, d, e = rng.permutation(ids)
                if dataset_idx % 2 == 0:
                    categories = [
                        {'id': a, 'name': 'cat1'},
                        {'id': b, 'name': 'cat2'},
                        {'id': e, 'name': 'cat5'},
                    ]
                else:
                    categories = [
                        {'id': c, 'name': 'cat3'},
                        {'id': d, 'name': 'cat4'},
                        {'id': e, 'name': 'cat5'},
                    ]
                dset = kwcoco.CocoDataset()
                dset.add_categories(categories)
                dsets.append(dset)

            combo_dset = kwcoco.CocoDataset.union(*dsets)
            combo_categories = combo_dset.dataset['categories']
            assert len(set(c['id'] for c in combo_categories)) == 5, 'new ids must be disjoint'


def test_union_with_disjoint_inconsistent_category_ids_explicit():
    import kwcoco
    categories1 = [
        {'id': 0, 'name': 'cat1'},
        {'id': 3, 'name': 'cat2'},
        {'id': 2, 'name': 'cat5'},
    ]
    categories2 = [
        {'id': 2, 'name': 'cat3'},
        {'id': 1, 'name': 'cat4'},
        {'id': 3, 'name': 'cat5'},
    ]
    dset1 = kwcoco.CocoDataset()
    dset2 = kwcoco.CocoDataset()
    dset1.add_categories(categories1)
    dset2.add_categories(categories2)
    combo_dset = dset1.union(dset2)
    combo_categories = combo_dset.dataset['categories']
    assert combo_dset.index.name_to_cat['cat3']['id'] == 4, (
        'cat3 should be given a new unseen id because its id was already taken')
    assert combo_categories == [
        {'id': 0, 'name': 'cat1'},
        {'id': 3, 'name': 'cat2'},
        {'id': 2, 'name': 'cat5'},
        {'id': 4, 'name': 'cat3'},
        {'id': 1, 'name': 'cat4'}]
