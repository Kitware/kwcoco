import ubelt as ub


def test_category_rename_merge_policy():
    """
    Make sure the same category rename produces the same new hashid

    xdoctest ~/code/kwcoco/tests/test_cocodataset.py test_category_rename_merge_policy


    """

    import kwcoco
    from kwcoco.util import util_json
    orig_dset = kwcoco.CocoDataset()
    orig_dset.add_category('a', a=1)
    orig_dset.add_category('b', b=2)
    orig_dset.add_category('c', c=3)
    print(ub.urepr(orig_dset.cats))

    # Test multiple cats map on to one existing unchanged cat
    mapping = {
        'a': 'b',
        'c': 'b',
    }

    if 0:
        print('-- test 1 --')
        dset = orig_dset.copy()
        dset.rename_categories(mapping, merge_policy='update')
        print(ub.urepr(dset.cats, nl=1))
        got = dset.name_to_cat['b']
        want = {'id': 2, 'name': 'b', 'b': 2, 'a': 1, 'c': 3}
        print('got = {}'.format(ub.urepr(got, nl=1)))
        print('want = {}'.format(ub.urepr(want, nl=1)))
        # assert got == want
        flag, info = util_json.indexable_allclose(got, want, return_info=True)
        print('info = {}'.format(ub.urepr(info, nl=1)))
        assert flag
        # assert sorted(got.items()) == sorted(want.items())

        print('-- test 2 --')
        dset = orig_dset.copy()
        dset.rename_categories(mapping, merge_policy='ignore')
        print(ub.urepr(dset.cats, nl=1))
        got = dset.name_to_cat['b']
        want = {'id': 2, 'name': 'b', 'b': 2}
        flag, info = util_json.indexable_allclose(got, want, return_info=True)
        print('info = {}'.format(ub.urepr(info, nl=1)))
        assert flag

    # Test multiple cats map on to one new cat
    print('-- test 3 --')
    mapping = {
        'a': 'e',
        'c': 'e',
        'b': 'd',
    }
    orig_dset._next_ids
    dset = orig_dset.copy()
    dset.rename_categories(mapping, merge_policy='ignore')
    got = dset.cats
    want = {
        1: { 'id': 1, 'name': 'd', 'b': 2, },
        2: { 'id': 2, 'name': 'e', }}
    print('got = {}'.format(ub.urepr(got, nl=1)))
    print('want = {}'.format(ub.urepr(want, nl=1)))
    flag, info = util_json.indexable_allclose(got, want, return_info=True)
    print('info = {}'.format(ub.urepr(info, nl=1)))
    assert flag

    print('-- test 4 --')
    dset = orig_dset.copy()
    dset.rename_categories(mapping, merge_policy='update')
    print(ub.urepr(dset.cats, nl=1))
    got = dset.cats
    want = {
        1: {'id': 1, 'name': 'd', 'b': 2},
        2: {'id': 2, 'name': 'e', 'a': 1, 'c': 3},
    }
    flag, info = util_json.indexable_allclose(got, want, return_info=True)
    print('info = {}'.format(ub.urepr(info, nl=1)))
    assert flag


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


def test_dump():
    import tempfile
    import json
    import kwcoco
    self = kwcoco.CocoDataset.demo()
    file = tempfile.NamedTemporaryFile('w', delete=False)
    tmp_path = ub.Path(file.name)
    with file:
        self.dump(file, newlines=True)
    text = tmp_path.read_text()
    print(text)
    dataset = json.loads(text)
    self2 = kwcoco.CocoDataset(dataset, tag='demo2')
    assert self2.dataset == self.dataset
    assert self2.dataset is not self.dataset
    tmp_path.delete()


def test_dump_causes_saved_status():
    import kwcoco
    import ubelt as ub
    dpath = ub.Path.appdir('kwcoco/test/test_dump_causes_saved_status').ensuredir()
    dset = kwcoco.CocoDataset.demo()
    fpath = dpath / 'mydata.kwcoco.json'
    dset.dump(fpath)
    dpath.delete()
    assert dset._state['was_saved']


def test_ensure_video_does_not_duplicate():
    import kwcoco
    dset = kwcoco.CocoDataset()
    dset.ensure_video(name='foo')
    dset.ensure_video(name='foo')
    print(dset.dataset['videos'])
    assert len(dset.dataset['videos']) == 1
