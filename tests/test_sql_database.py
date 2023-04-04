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
