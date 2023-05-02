

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
