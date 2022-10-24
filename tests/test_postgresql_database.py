

def have_sqlalchemy():
    try:
        import sqlalchemy  # NOQA
    except ImportError:
        return False
    return True


def have_postgresql():
    try:
        import psycopg2  # NOQA
    except ImportError:
        return False
    return True  # todo: make robust


def test_coerce_as_postgresql():
    import pytest
    if not have_postgresql() or not have_sqlalchemy():
        pytest.skip()

    import kwcoco
    dct_dset = kwcoco.CocoDataset.coerce('special:shapes8')
    psql_dset = kwcoco.CocoDataset.coerce(dct_dset.fpath, sqlview='postgresql')
    psql_dset.engine
    print(f'psql_dset.engine={psql_dset.engine}')
    assert str(psql_dset.engine.url).startswith('postgresql')


def test_coerce_as_sqlite():
    import pytest
    if not have_sqlalchemy():
        pytest.skip()
    import kwcoco
    dct_dset = kwcoco.CocoDataset.coerce('special:shapes8')
    psql_dset = kwcoco.CocoDataset.coerce(dct_dset.fpath, sqlview='sqlite')
    assert str(psql_dset.engine.url).startswith('sqlite')
