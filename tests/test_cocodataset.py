import ubelt as ub


def test_category_rename_merge_policy():
    """
    Make sure the same category rename produces the same new hashid
    """

    import kwcoco
    orig_dset = kwcoco.CocoDataset()
    orig_dset.add_category('a', a=1)
    orig_dset.add_category('b', b=2)
    orig_dset.add_category('c', c=3)
    print(ub.repr2(orig_dset.cats))

    # Test multiple cats map on to one existing unchanged cat
    mapping = {
        'a': 'b',
        'c': 'b',
    }

    dset = orig_dset.copy()
    dset.rename_categories(mapping, merge_policy='update')
    print(ub.repr2(dset.cats, nl=1))
    assert dset.name_to_cat['b'] == {'id': 2, 'name': 'b', 'b': 2, 'a': 1, 'c': 3}

    dset = orig_dset.copy()
    dset.rename_categories(mapping, merge_policy='ignore')
    print(ub.repr2(dset.cats, nl=1))
    assert dset.name_to_cat['b'] == {'id': 2, 'name': 'b', 'b': 2}

    # Test multiple cats map on to one new cat
    mapping = {
        'a': 'e',
        'c': 'e',
        'b': 'd',
    }
    orig_dset._next_ids
    dset = orig_dset.copy()
    dset.rename_categories(mapping, merge_policy='ignore')
    print(ub.repr2(dset.cats, nl=1))
    assert dset.cats == {
        1: { 'id': 1, 'name': 'd', 'b': 2, },
        2: { 'id': 2, 'name': 'e', }}

    dset = orig_dset.copy()
    dset.rename_categories(mapping, merge_policy='update')
    print(ub.repr2(dset.cats, nl=1))
    assert dset.cats == {
        1: {'id': 1, 'name': 'd', 'b': 2},
        2: {'id': 2, 'name': 'e', 'a': 1, 'c': 3},
    }


def test_copy_nextids():
    import kwcoco
    orig_dset = kwcoco.CocoDataset()
    orig_dset.add_category('a')

    # Make sure we aren't shallow copying the next id object
    dset = orig_dset.copy()
    dset.add_category('b')
    new_cid = dset.name_to_cat['b']['id']
    assert new_cid == 2

    dset = orig_dset.copy()
    dset.add_category('b')
    new_cid = dset.name_to_cat['b']['id']
    assert new_cid == 2


def test_only_null_category_id():
    import kwcoco
    orig_dset = kwcoco.CocoDataset()

    gid = orig_dset.add_image(file_name='foo')

    orig_dset.add_annotation(image_id=gid, bbox=[0, 0, 10, 10])
    orig_dset.add_annotation(image_id=gid, bbox=[0, 2, 10, 10])

    new_dset = orig_dset.union(orig_dset)
    orig_dset._check_pointers()

    cat_hist = ub.dict_hist([ann['category_id'] for ann in new_dset.anns.values()])
    assert cat_hist == {None: 4}
    assert new_dset.category_annotation_frequency() == ub.odict([(None, 4)])


def test_partial_null_category_id():
    import kwcoco
    orig_dset = kwcoco.CocoDataset()
    orig_dset.add_category('a')

    gid = orig_dset.add_image(file_name='foo')

    orig_dset.add_annotation(image_id=gid, bbox=[0, 0, 10, 10])
    orig_dset.add_annotation(image_id=gid, bbox=[0, 2, 10, 10])
    orig_dset.add_annotation(image_id=gid, category_id=1, bbox=[0, 2, 10, 10])

    new_dset = orig_dset.union(orig_dset)
    orig_dset._check_pointers()

    cat_hist = ub.dict_hist([ann['category_id'] for ann in new_dset.anns.values()])
    assert cat_hist == {None: 4, 1: 2}
    assert new_dset.category_annotation_frequency() == ub.odict([('a', 2), (None, 4)])
