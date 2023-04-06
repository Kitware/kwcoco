"""
SeeAlso:
    ~/code/kwcoco/tests/test_sql_database.py
"""


def test_postgresql_cases():
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
    dct_dset.add_annotation(**{'image_id': 6, 'track_id': 9001, 'bbox': [0, 0, 10, 10], 'category_id': 1})
    dct_dset.add_annotation(**{'image_id': 3, 'track_id': 9001, 'bbox': [0, 0, 10, 10], 'category_id': 1})
    dct_dset.add_annotation(**{'image_id': 5, 'track_id': 9001, 'bbox': [0, 0, 10, 10], 'category_id': 1})
    dct_dset.add_annotation(**{'image_id': 4, 'track_id': 9001, 'bbox': [0, 0, 10, 10], 'category_id': 1})

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
    assert 0 in set(track_ids)

    ann1 = annots.objs[0]
    ann2 = psql_dset.index.anns._uncached_getitem(1)
    assert ann1['track_id'] == ann2.track_id
    assert ann1['track_id'] == track_ids[0]

    # Target for aids with trackid=0
    aids1 = dct_dset.annots(track_id=0)._ids

    table = psql_dset.tabular_targets().pandas()
    table['track_ids'] = track_ids
    print(table)

    # Check that raw lookups work where we explicitly cast the key (v1)
    result = psql_dset.session.execute(text(ub.codeblock(
        '''
        SELECT annotations.id FROM annotations
        WHERE CAST(annotations.track_id as int) = 0
        '''
    )))
    aids2 = [f[0] for f in result]
    assert list(aids2) == list(aids1)

    # Check that raw lookups work where we explicitly cast the key (v2)
    result = psql_dset.session.execute(text(ub.codeblock(
        '''
        SELECT annotations.id FROM annotations
        WHERE annotations.track_id::int = 0
        '''
    )))
    aids2 = [f[0] for f in result]
    assert list(aids2) == list(aids1)

    # Check that raw lookups work where we explicitly cast the query
    result = psql_dset.session.execute(text(ub.codeblock(
        '''
        SELECT annotations.id FROM annotations
        WHERE annotations.track_id = to_jsonb(0)
        '''
    )))
    psql_dset.session.rollback()
    aids2 = [f[0] for f in result]
    assert list(aids2) == list(aids1)

    id_group_dict_proxy = psql_dset.index.trackid_to_aids
    # This works ok  (before and after fix)
    track_id_int = 0
    track_id_jsonb = sqlalchemy.func.to_jsonb(track_id_int)
    aids2 = id_group_dict_proxy._uncached_getitem(track_id_jsonb)
    assert list(aids2) == list(aids1)

    # SQLAlchemy Query
    proxy = psql_dset.index.trackid_to_aids
    session = proxy.session
    keyattr = proxy.keyattr
    valattr = proxy.valattr
    key = 9001
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
        track_id_int = 0
        aids2 = id_group_dict_proxy._uncached_getitem(track_id_int)
        assert list(aids2) == list(aids1)

        aids2 = psql_dset.annots(track_id=0)._ids
        assert list(aids2) == list(aids1)

        # Does not work before fix, but now does
        track_id_int = 9001
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
