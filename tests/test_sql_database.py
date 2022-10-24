import ubelt as ub
import kwcoco


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
    dct_dset = kwcoco.CocoDataset.coerce('special:shapes8')
    psql_dset = kwcoco.CocoDataset.coerce(dct_dset.fpath, sqlview='postgresql')
    psql_dset.engine
    print(f'psql_dset.engine={psql_dset.engine}')
    assert str(psql_dset.engine.url).startswith('postgresql')


def test_coerce_as_sqlite():
    import pytest
    if not have_sqlalchemy():
        pytest.skip()
    dct_dset = kwcoco.CocoDataset.coerce('special:shapes8')
    psql_dset = kwcoco.CocoDataset.coerce(dct_dset.fpath, sqlview='sqlite')
    assert str(psql_dset.engine.url).startswith('sqlite')


def available_sql_backends():
    pass


def test_api_compatability_msi():
    """
    Use API paths in each backend and make sure they are the same up to
    known differences
    """
    dct_dset = kwcoco.CocoDataset.demo('vidshapes8-multisensor-msi')
    _api_compatability_tests(dct_dset)


def test_api_compatability_rgb():
    dct_dset = kwcoco.CocoDataset.demo('shapes8')
    _api_compatability_tests(dct_dset)


def test_api_compatability_photos():
    dct_dset = kwcoco.CocoDataset.demo('photos')
    _api_compatability_tests(dct_dset)


def _api_compatability_tests(dct_dset):
    dset_variants = {}
    dset_variants['dictionary'] = dct_dset

    if have_postgresql():
        # dset_variants['postgresql'] = kwcoco.CocoDataset.coerce(dct_dset.fpath, sqlview='postgresql')
        dset_variants['postgresql'] = dct_dset.view_sql(backend='postgresql')

    if have_sqlalchemy():
        dset_variants['sqlite'] = dct_dset.view_sql(backend='sqlite')

    results = {}
    for key, dset in dset_variants.items():
        results[key] = result = {}
        all_gids = sorted(dset.images())
        all_num_assets = []
        for gid in all_gids:
            coco_img = dset.coco_image(gid)
            assets = list(coco_img.iter_asset_objs())
            all_num_assets.append(len(assets))
        result['all_gids'] = all_gids
        result['all_num_assets'] = all_num_assets

    print('results = {}'.format(ub.repr2(results, nl=2)))
    for a, b in ub.iter_window(results.values(), 2):
        assert ub.indexable_allclose(a, b)
