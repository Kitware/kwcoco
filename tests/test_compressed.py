def test_read_zipped_kwcoco():
    import kwcoco
    # import zipfile
    import ubelt as ub
    dset = kwcoco.CocoDataset.demo('vidshapes8')

    # Make a zipfile with a kwcoco file in it
    src_fpath = ub.Path(dset.fpath)
    zip_fpath = src_fpath.augment(ext='.zip')

    # Test that we can dump to a zipfile
    dset.dump(zip_fpath)
    import zipfile
    assert zipfile.is_zipfile(zip_fpath)

    # Test that the zipfile can be read
    dset2 = kwcoco.CocoDataset(zip_fpath)
    assert dset2.dataset == dset.dataset
    assert dset2.dataset is not dset.dataset


def test_compress_dump():
    import tempfile
    import kwcoco
    self = kwcoco.CocoDataset.demo()
    file = tempfile.NamedTemporaryFile('wb')
    self.dump(file, compress=True)
    import zipfile
    assert zipfile.is_zipfile(file.name)
    self2 = kwcoco.CocoDataset(file.name)
    assert self2.dataset == self.dataset
    assert self2.dataset is not self.dataset
