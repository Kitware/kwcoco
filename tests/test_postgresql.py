"""
SeeAlso:
    ~/code/kwcoco/tests/test_sql_database.py
"""


def test_postgresql_cases():
    """
    pytest  ~/code/kwcoco/tests/test_postgresql.py -s
    """
    import pytest
    import ubelt as ub
    try:
        import psycopg2  # NOQA
        import sqlalchemy  # NOQA
    except ImportError:
        pytest.skip()
    import kwcoco

    from kwcoco.coco_sql_dataset import Image, Annotation
    from kwcoco.coco_sql_dataset import text, IS_GE_SQLALCH_2x
    dct_dset = kwcoco.CocoDataset.coerce('special:vidshapes8')
    # Add annots so there is an out of order track
    ooo_track_id = 9001
    dct_dset.add_track(name="ooo_track", id=ooo_track_id)
    dct_dset.add_annotation(**{'image_id': 6, 'track_id': ooo_track_id, 'bbox': [0, 0, 10, 10], 'category_id': 1})
    dct_dset.add_annotation(**{'image_id': 3, 'track_id': ooo_track_id, 'bbox': [0, 0, 10, 10], 'category_id': 1})
    dct_dset.add_annotation(**{'image_id': 5, 'track_id': ooo_track_id, 'bbox': [0, 0, 10, 10], 'category_id': 1})
    dct_dset.add_annotation(**{'image_id': 4, 'track_id': ooo_track_id, 'bbox': [0, 0, 10, 10], 'category_id': 1})

    new_fpath = ub.Path(dct_dset.fpath).augment(stemsuffix='_with_ooo_track', multidot=True)
    dct_dset.fpath = new_fpath
    dct_dset.dump()

    # psql_dset = dct_dset.view_sql(backend='postgresql')
    psql_dset = kwcoco.CocoDataset.coerce(dct_dset.fpath, sqlview='postgresql')
    psql_dset.engine
    print(f'psql_dset.engine={psql_dset.engine}')
    assert str(psql_dset.engine.url).startswith('postgresql')

    annots = psql_dset.annots()
    track_ids = annots.lookup('track_id')
    main_track_id = track_ids[0]
    assert main_track_id in set(track_ids)

    ann1 = annots.objs[0]
    ann2 = psql_dset.index.anns._uncached_getitem(1)
    assert ann1['track_id'] == ann2.track_id
    assert ann1['track_id'] == track_ids[0]

    # Target for aids with main_track_id
    aids1 = dct_dset.annots(track_id=main_track_id)._ids
    aids2 = psql_dset.annots(track_id=main_track_id)._ids
    assert list(aids2) == list(aids1)

    table = psql_dset.tabular_targets().pandas()
    table['track_ids'] = track_ids
    print(table)

    # if 0:
    #     import pandas as pd
    #     raw_tables = psql_dset._raw_tables()
    #     raw_annot_table = pd.DataFrame(raw_tables['annotations'])

    # Check that raw lookups work where we explicitly cast the key (v1)
    result = psql_dset.session.execute(text(ub.codeblock(
        f'''
        SELECT annotations.id FROM annotations
        WHERE CAST(annotations.track_id as int) = {main_track_id}
        '''
    )))
    aids2 = [f[0] for f in result]
    assert list(aids2) == list(aids1)

    # Check that raw lookups work where we explicitly cast the key (v2)
    result = psql_dset.session.execute(text(ub.codeblock(
        F'''
        SELECT annotations.id FROM annotations
        WHERE annotations.track_id::int = {main_track_id}
        '''
    )))
    aids2 = [f[0] for f in result]
    assert list(aids2) == list(aids1)

    # Check that raw lookups work where we explicitly cast the query
    result = psql_dset.session.execute(text(ub.codeblock(
        f'''
        SELECT annotations.id FROM annotations
        WHERE annotations.track_id = to_jsonb({main_track_id})
        '''
    )))
    psql_dset.session.rollback()
    aids2 = [f[0] for f in result]
    assert list(aids2) == list(aids1)

    id_group_dict_proxy = psql_dset.index.trackid_to_aids
    # This works ok  (before and after fix)
    track_id_int = main_track_id
    track_id_jsonb = sqlalchemy.func.to_jsonb(track_id_int)
    aids2 = id_group_dict_proxy._uncached_getitem(track_id_jsonb)
    assert list(aids2) == list(aids1)

    # SQLAlchemy Query
    proxy = psql_dset.index.trackid_to_aids
    session = proxy.session
    keyattr = proxy.keyattr
    valattr = proxy.valattr
    key = ooo_track_id
    from sqlalchemy.dialects.postgresql import JSONB
    dialect_name = session.get_bind().dialect.name

    if IS_GE_SQLALCH_2x:
        dialect_type = keyattr.expression.type._variant_mapping[dialect_name]
    else:
        dialect_type = keyattr.expression.type.mapping[dialect_name]
    if dialect_type.__class__ is JSONB:
        # Hack for columns with JSONB indexes (e.g. track_id)
        key = sqlalchemy.func.to_jsonb(key)

    query = session.query(valattr)
    query = query.filter(keyattr == key)
    query = query.join(Image, Image.id == Annotation.image_id)
    query = query.order_by(Image.frame_index)
    # print(query)
    item = [row[0] for row in query.all()]
    print(f'item={item}')

    HAS_FIX = 1
    if HAS_FIX:
        # Does not work before fix, but now does
        track_id_int = main_track_id
        aids2 = id_group_dict_proxy._uncached_getitem(track_id_int)
        assert list(aids2) == list(aids1)

        aids2 = psql_dset.annots(track_id=main_track_id)._ids
        assert list(aids2) == list(aids1)

        # Does not work before fix, but now does
        track_id_int = ooo_track_id
        ooo_aids2 = list(psql_dset.annots(track_id=track_id_int))
        ooo_aids1 = list(dct_dset.annots(track_id=track_id_int))
        assert list(ooo_aids2) == list(ooo_aids1)

    # session = id_group_dict_proxy.session
    # engine = session.get_bind()  # NOQA
    # id_group_dict_proxy.keyattr.expression.type.adapt(engine)
    # kwcoco.coco_sql_dataset.CocoBase.metadata.tables['annotations']
    # valattr = id_group_dict_proxy.valattr
    # keyattr = id_group_dict_proxy.keyattr
    # query1 = session.query(valattr)
    # query2.filter(keyattr == key)

    # # keyattr.expression.type.dialect_impl(engine.dialect)
    # dialect = engine.dialect
    # dialect_type = keyattr.expression.type.mapping[engine.dialect.name]
    # impl = dialect_type.dialect_impl(dialect)


def _process_worker(sql_dset, image_id):
    # if hasattr(sql_dset, 'connect'):
    #     # Reconnect to the backend if we are using SQL
    #     sql_dset.connect(readonly=True)

    img = sql_dset.index.imgs[image_id]
    anns = []
    for annot_id in sql_dset.index.gid_to_aids[image_id]:
        ann = sql_dset.index.anns[annot_id]
        anns.append(ann)

    return (img, anns)


def test_postgresql_in_process_worker():
    """
    A PostgreSQL view should be able to remember the filename it was populated
    from. This is important for interoperability with sidecar-file based hashes
    as well as relative pathing.

    CommandLine:
        xdoctest ~/code/kwcoco/tests/test_postgresql.py test_postgresql_in_process_worker
    """
    import pytest
    import ubelt as ub
    try:
        import psycopg2  # NOQA
        import sqlalchemy  # NOQA
    except ImportError:
        pytest.skip()

    import kwcoco
    dct_dset = kwcoco.CocoDataset.coerce('special:vidshapes8')
    # sql_dset = dct_dset.view_sql(backend='postgresql')
    sql_dset = dct_dset.view_sql(backend='sqlite')
    executor = ub.Executor(mode='process', max_workers=8)
    print('Build executor')
    with executor:
        jobs = []
        images = sql_dset.images()
        print('Submit jobs')
        for image_id in list(images):
            job = executor.submit(_process_worker, sql_dset, image_id)
            jobs.append(job)
        print('Collect jobs')
        sql_results = [job.result() for job in jobs]
        print('Build expected results')
        dct_results = [_process_worker(dct_dset, image_id) for image_id in images]
    print('Compare results')
    assert len(dct_results) == len(sql_results)
    for tup1, tup2 in zip(dct_results, sql_results):
        img1, anns1 = tup1
        img2, anns2 = tup1
        assert img1 == img2
        assert len(anns1) == len(anns2)
        assert [a['id'] for a in anns1] == [a['id'] for a in anns2]
