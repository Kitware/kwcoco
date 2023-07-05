

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
    coco_union._CLI.main(src=src_fpaths, dst=dst_fpath)

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
    coco_union._CLI.main(src=src_fpaths, dst=dst_fpath)

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
    coco_union._CLI.main(src=src_fpaths, dst=dst_fpath)

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
