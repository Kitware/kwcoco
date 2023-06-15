
def test_cached_hashid_with_permissions():
    """
    Check to make sure requesting a cached hash id doesnt fail if we don't have
    write permissions.
    """
    import kwcoco
    import ubelt as ub
    import os
    import stat
    dpath = ub.Path.appdir('kwcoco/tests/test_hashid').ensuredir()
    parent_mode = dpath.parent.stat().st_mode
    dpath.chmod(parent_mode)

    dset = kwcoco.CocoDataset.demo()
    dset.fpath = os.fspath(dpath / 'test.kwcoco.json')
    dset.dump()

    fpath = ub.Path(dset.fpath)
    dset = kwcoco.CocoDataset(fpath)

    ro = (
        stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH |
        stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )
    dpath.chmod(ro)

    # import pytest
    # with pytest.warns(match='Cannot write a cached hashid'):
    dset._cached_hashid()

    # Set permission back to normal, writing the cache id should work
    dpath.chmod(parent_mode)
    dset._cached_hashid()

    # Cleanup
    dpath.delete()
