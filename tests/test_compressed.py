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
    import ubelt as ub
    self = kwcoco.CocoDataset.demo()
    file = tempfile.NamedTemporaryFile('wb', delete=False)
    fpath = ub.Path(file.name)
    with file:
        self.dump(file, compress=True)
    import zipfile
    assert zipfile.is_zipfile(fpath)
    self2 = kwcoco.CocoDataset(fpath)
    assert self2.dataset == self.dataset
    assert self2.dataset is not self.dataset
    fpath.delete()


def test_internal_archive_name():
    import kwcoco
    # import zipfile
    import ubelt as ub
    dset = kwcoco.CocoDataset.demo('vidshapes8')

    dpath = ub.Path.appdir('kwcoco/tests/test_compress').ensuredir()
    dset.fpath = dpath / 'mydataset.kwcoco.zip'
    dset.dump()

    import zipfile
    zfile = zipfile.ZipFile(dset.fpath)
    assert zfile.namelist() == ['mydataset.kwcoco.json']

    fpath2 = dpath / 'newname2.kwcoco.zip'
    with open(fpath2, 'wb') as file:
        dset.dump(file)
    # Did not update fpath, so the internal name wont change
    assert zipfile.ZipFile(fpath2).namelist() == ['mydataset.kwcoco.json']

    # Fallback name
    dset.fpath = ''
    fpath3 = dpath / 'newname3.kwcoco.zip'
    with open(fpath3, 'wb') as file:
        dset.dump(file)
    assert zipfile.ZipFile(fpath3).namelist() == ['_data.kwcoco.json']

    dpath.delete()  # cleanup
