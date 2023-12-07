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
    sql_dset = kwcoco.CocoDataset.coerce(dct_dset.fpath, sqlview='sqlite')
    assert str(sql_dset.engine.url).startswith('sqlite')


def available_sql_backends():
    pass


def test_api_compatability_msi():
    """
    Use API paths in each backend and make sure they are the same up to
    known differences
    """
    dct_dset = kwcoco.CocoDataset.demo('vidshapes8-multisensor-msi')
    _api_compatability_tests(dct_dset)


def test_api_compatability_msi_ooo_tracks():
    dct_dset = kwcoco.CocoDataset.demo('vidshapes8-multisensor-msi')
    video_id = dct_dset.add_video(name='ooo_video')

    import kwarray
    rng = kwarray.ensure_rng(0)
    frame_order = list(range(9))
    rng.shuffle(frame_order)

    # Add images to the video out of order
    for frame_index in frame_order:
        frame_name = ub.hash_data(rng.rand())[0:8]
        dct_dset.add_image(video_id=video_id, name=f'frame_{frame_name}', frame_index=frame_index)

    image_ids = list(dct_dset.images(video_id=video_id))
    rng.shuffle(image_ids)

    # Add a track to the image out of order
    for image_id in image_ids:
        dct_dset.add_annotation(**{'image_id': image_id, 'track_id': 9001, 'bbox': [0, 0, 10, 10]})

    dct_dset = kwcoco.CocoDataset.demo('vidshapes8-multisensor-msi')
    dct_dset.fpath = ub.Path(dct_dset.fpath).augment(stemsuffix='_with_ooo_tracks', multidot=True)
    dct_dset.dump()
    dct_dset = kwcoco.CocoDataset(dct_dset.fpath)
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

    print('results = {}'.format(ub.urepr(results, nl=2)))
    for a, b in ub.iter_window(results.values(), 2):
        assert ub.IndexableWalker(a).allclose(b)

    # Test track / image ordering
    results = {}
    for key, dset in dset_variants.items():
        try:
            track_ids = dset.annots().lookup('track_id')
        except KeyError:
            continue
        unique_track_ids = sorted(set(track_ids))
        tid_to_aids = {}
        for tid in unique_track_ids:
            annots = dset.annots(track_id=tid)
            annot_ids = list(annots)
            annot_frame_idxs = annots.images.lookup('frame_index')
            annot_frame_idxs = [x for x in annot_frame_idxs if x is not None]
            assert sorted(annot_frame_idxs) == annot_frame_idxs
            tid_to_aids[tid] = annot_ids
        results[key] = {'tid_to_aids': tid_to_aids}
    print('results = {}'.format(ub.urepr(results, nl=3)))

    # Test image in video ordering
    results = {}
    for key, dset in dset_variants.items():
        videos = dset.videos()
        video_images = videos.images
        vidid_to_gids = {}
        vidid_to_frame_idxs = {}
        for video_id, images in zip(videos, video_images):
            frame_idxs = images.lookup('frame_index')
            assert list(images) == list(dset.images(video_id=video_id))
            assert sorted(frame_idxs) == frame_idxs
            vidid_to_frame_idxs[video_id] = frame_idxs
            vidid_to_gids[video_id] = list(images)
        results[key] = {
            'vidid_to_frame_idxs': vidid_to_frame_idxs,
            'vidid_to_gids': vidid_to_gids,
        }
    print('results = {}'.format(ub.urepr(results, nl=2)))
    for a, b in ub.iter_window(results.values(), 2):
        assert ub.IndexableWalker(a).allclose(b)


def test_coerce_sql_from_zipfile():
    """
    Check that kwcoco.CocoDataset.coerce(., sqlview='sqlite') correctly
    converts zip files as well as json files.
    """
    import pytest
    if not have_sqlalchemy():
        pytest.skip('requires sqlalchemy')
    import kwcoco
    import ubelt as ub
    dpath = ub.Path.appdir('kwcoco/tests/test_coerce_sql_from_zipfile')
    dpath.delete().ensuredir()
    dct_dset = kwcoco.CocoDataset.demo('vidshapes8-multisensor-msi')
    dct_dset.fpath = dpath / 'data.kwcoco.zip'
    dct_dset.dump()
    import zipfile
    assert zipfile.is_zipfile(dct_dset.fpath)

    # Initial coerce should do conversion
    sql_dset1 = kwcoco.CocoDataset.coerce(dct_dset.fpath, sqlview='sqlite')
    assert isinstance(sql_dset1, kwcoco.CocoSqlDatabase)

    # Subsequenct coerce should read from cache
    sql_dset2 = kwcoco.CocoDataset.coerce(dct_dset.fpath, sqlview='sqlite')
    assert isinstance(sql_dset2, kwcoco.CocoSqlDatabase)

    assert sql_dset2 is not sql_dset1

    for sql_dset in [sql_dset1, sql_dset2]:
        orig_coco_fpath = sql_dset._orig_coco_fpath()
        assert orig_coco_fpath.exists()
        assert orig_coco_fpath == dct_dset.fpath

        hashid1 = sql_dset._cached_hashid()
        hashid2 = dct_dset._cached_hashid()
        assert hashid1 == hashid2


def test_python_index_maps():
    """
    Check that the indexes exposed in dictionary mode
    match the ones in sql mode
    """
    import kwcoco
    import pytest
    if not have_sqlalchemy():
        pytest.skip('requires sqlalchemy')
    dct_dset = kwcoco.CocoDataset.coerce('special:vidshapes8')
    sql_dset = kwcoco.CocoDataset.coerce(dct_dset.fpath, sqlview='sqlite')

    cand_attrs = [k for k in dir(dct_dset.index) if not k.startswith('_')]

    # Attribute with dictionary values in the index object are the indexes
    index_attrs = []
    for key in cand_attrs:
        value = getattr(dct_dset.index, key)
        if isinstance(value, dict):
            index_attrs.append(key)

    failures = []
    for key in index_attrs:
        print(f'Checking key = {key}')
        dct_index = getattr(dct_dset.index, key)
        sql_index = getattr(sql_dset.index, key)
        assert isinstance(dct_index, dict)
        assert sql_index is not None

        dct_keys = list(dct_index.keys())
        sql_keys = list(sql_index.keys())

        fudgable = None
        exact_same = sorted(dct_keys) == sorted(sql_keys)
        if not exact_same:
            fudgable = sorted(set(dct_keys)) == sorted(set(sql_keys))
            if fudgable:
                print(f'[yellow]WARNING index check key={key} had duplicate SQL keys, weird, FIXME')
            else:
                failures.append(key)
                print(f'[red]FAILED index check key={key}')
                raise

        for k in dct_keys:
            sql_v = sql_index[k]
            dct_v = dct_index[k]
            if isinstance(sql_v, dict):
                for k in dct_v.keys():
                    assert sql_v[k] == dct_v[k]
            else:
                assert sql_v == dct_v
