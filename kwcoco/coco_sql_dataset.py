"""

TODO:
    - [ ] We get better speeds with raw SQL over alchemy. Can we mitigate the
        speed difference so we can take advantage of alchemy's expressiveness?


Finally got a baseline implementation of an SQLite backend for COCO
datasets. This mostly plugs into my existing tools (as long as only read
operations are used; haven't impelmented writing yet) by duck-typing the
dict API.

This solves the issue of forking and then accessing nested dictionaries in
the JSON-style COCO objects. (When you access the dictionary Python will
increment a reference count which triggers copy-on-write for whatever
memory page that data happened to live in. Non-contiguous access had the
effect of excessive memory copies).

For "medium sized" datasets its quite a bit slower. Running through a torch
DataLoader with 4 workers for 10,000 images executes at a rate of 100Hz but
takes 850MB of RAM. Using the duck-typed SQL backend only uses 500MB (which
includes the cost of caching), but runs at 45Hz (which includes the benefit
of caching).

However, once I scale up to 100,000 images I start seeing benefits.  The
in-memory dictionary interface chugs at 1.05HZ, and is taking more than 4GB
of memory at the time I killed the process (eta was over an hour). The SQL
backend ran at 45Hz and took about 3 minutes and used about 2.45GB of memory.

Without a cache, SQL runs at 30HZ and takes 400MB for 10,000 images, and
for 100,000 images it gets 30Hz with 1.1GB. There is also a much larger startup
time. I'm not exactly sure what it is yet, but its probably some preprocessing
I'm doing.

Using a LRU cache we get 45Hz and 1.05GB of memory, so that's a clear win.  We
do need to be sure to disable the cache if we ever implement write mode.

I'd like to be a bit faster on the medium sized datasets (I'd really like
to avoid caching rows, which is why the speed is currently
semi-reasonable), but I don't think I can do any better than this because
single-row lookup time is `O(log(N))` for sqlite, whereas its `O(1)` for
dictionaries. (I wish sqlite had an option to create a hash-table index for
a table, but I dont think it does). I optimized as many of the dictionary
operations as possible (for instance, iterating through keys, values, and
items should be O(N) instead of `O(N log(N))`), but the majority of the
runtime cost is in the single-row lookup time.


There are a few questions I still have if anyone has insight:

    * Say I want to select a subset of K rows from a table with N entries,
      and I have a list of all of the rowids that I want. Is there any
      way to do this better than `O(K log(N))`? I tried using a
      `SELECT col FROM table WHERE id IN (?, ?, ?, ?, ...)` filling in
      enough `?` as there are rows in my subset. I'm not sure what the
      complexity of using a query like this is. I'm not sure what the `IN`
      implementation looks like. Can this be done more efficiently by
      with a temporary table and a `JOIN`?

    * There really is no way to do `O(1)` row lookup in sqlite right?
      Is there a way in PostgreSQL or some other backend sqlalchemy
      supports?


I found that PostgreSQL does support hash indexes:
https://www.postgresql.org/docs/13/indexes-types.html I'm really not
interested in setting up a global service though 😞. I also found a 10-year
old thread with a hash-index feature request for SQLite, which I
unabashedly resurrected
http://sqlite.1065341.n5.nabble.com/Feature-request-hash-index-td23367.html
"""
import json
import numpy as np
import ubelt as ub
from os.path import exists

from kwcoco.util.dict_like import DictLike  # NOQA
from kwcoco.coco_dataset import (  # NOQA
    MixinCocoJSONAccessors, MixinCocoAccessors, MixinCocoAttrs,
    MixinCocoStats, MixinCocoDraw
)

try:
    from sqlalchemy.sql.schema import Column
    from sqlalchemy.types import Float, Integer, String, JSON
    from sqlalchemy.ext.declarative import declarative_base
    import sqlalchemy
    import sqlite3
    sqlite3.register_adapter(np.int64, int)
    sqlite3.register_adapter(np.int32, int)
    CocoBase = declarative_base()
except ImportError:
    # Hack: xdoctest should have been able to figure out that
    # all of these tests were diabled due to the absense of sqlalchemy
    # but apparently it has a bug. We can remove this hack once that is fixed
    sqlalchemy = None
    Float = ub.identity
    String = ub.identity
    JSON = ub.identity
    Integer = ub.identity
    Column = ub.identity
    class CocoBase:
        _decl_class_registry = {}


class Category(CocoBase):
    __tablename__ = 'categories'
    id = Column(Integer, primary_key=True, doc='unique internal id')
    name = Column(String(256), doc='unique external name or identifier', index=True, unique=True)
    alias = Column(JSON, doc='list of alter egos')
    supercategory = Column(String(256), doc='coarser category name')

    foreign = Column(JSON, default=dict())


class KeypointCategory(CocoBase):
    __tablename__ = 'keypoint_categories'
    id = Column(Integer, primary_key=True, doc='unique internal id')
    name = Column(String(256), doc='unique external name or identifier', index=True, unique=True)
    alias = Column(JSON, doc='list of alter egos')
    supercategory = Column(String(256), doc='coarser category name')
    reflection_id = Column(Integer, doc='if augmentation reflects the image, change keypoint id to this')

    foreign = Column(JSON, default=dict())


class Video(CocoBase):
    __tablename__ = 'videos'
    id = Column(Integer, primary_key=True, doc='unique internal id')
    name = Column(String(256), nullable=False, index=True, unique=True)
    caption = Column(String(256), nullable=True)

    foreign = Column(JSON, default=dict())


class Image(CocoBase):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True, doc='unique internal id')
    file_name = Column(String(512), nullable=False, index=True, unique=True)

    width = Column(Integer)
    height = Column(Integer)

    video_id = Column(Integer, index=True, unique=False)
    timestamp = Column(Float)
    frame_index = Column(Integer)

    channels = Column(JSON)
    auxiliary = Column(JSON)

    foreign = Column(JSON, default=dict())


class Annotation(CocoBase):
    __tablename__ = 'annotations'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, doc='', index=True, unique=False)
    category_id = Column(Integer, doc='', index=True, unique=False)
    track_id = Column(JSON, index=True, unique=False)

    segmentation = Column(JSON)
    keypoints = Column(JSON)

    bbox = Column(JSON)
    _bbox_x = Column(Float)
    _bbox_y = Column(Float)
    _bbox_w = Column(Float)
    _bbox_h = Column(Float)
    weight = Column(Float)

    score = Column(Float)
    weight = Column(Float)
    prob = Column(JSON)

    iscrowd = Column(Integer)
    caption = Column(JSON)

    foreign = Column(JSON, default=dict())


# Global book keeping (It would be nice to find a way to avoid this)
CocoBase.TBLNAME_TO_CLASS = {}
for classname, cls in CocoBase._decl_class_registry.items():
    if not classname.startswith('_'):
        tblname = cls.__tablename__
        CocoBase.TBLNAME_TO_CLASS[tblname] = cls


def orm_to_dict(obj):
    item = obj.__dict__.copy()
    item.pop('_sa_instance_state', None)
    return item


def _orm_yielder(query, size=300):
    """
    TODO: figure out the best way to yield, in batches or otherwise
    """
    if 1:
        yield from query.all()
    else:
        yield from query.yield_per(size)


def _raw_yielder(result, size=300):
    """
    TODO: figure out the best way to yield, in batches or otherwise
    """
    if 1:
        chunk = result.fetchall()
        for item in chunk:
            yield item
    else:
        chunk = result.fetchmany(size)
        while chunk:
            for item in chunk:
                yield item
            chunk = result.fetchmany(size)


def _new_proxy_cache():
    """
    By returning None, we wont use item caching
    """
    # return None
    try:
        from ndsampler.utils import util_lru
        return util_lru.LRUDict.new(max_size=100, impl='auto')
    except Exception:
        return {}


class SqlListProxy(ub.NiceRepr):
    """
    A view of an SQL table that behaves like a Python list
    """
    def __init__(proxy, session, cls):
        proxy.cls = cls
        proxy.session = session
        proxy._colnames = None
        proxy.ALCHEMY_MODE = 0

    def __len__(proxy):
        query = proxy.session.query(proxy.cls)
        return query.count()

    def __nice__(proxy):
        return '{}: {}'.format(proxy.cls.__tablename__, len(proxy))

    def __iter__(proxy):
        if proxy.ALCHEMY_MODE:
            query = proxy.session.query(proxy.cls).order_by(proxy.cls.id)
            for obj in _orm_yielder(query):
                item = orm_to_dict(obj)
                yield item
        else:
            if proxy._colnames is None:
                from sqlalchemy import inspect
                import json
                inspector = inspect(proxy.session.get_bind())
                colinfo = inspector.get_columns(proxy.cls.__tablename__)
                # Huge hack to fixup json columns.
                # the session.execute seems to do this for
                # some columns, but not all, hense the isinstance
                def _json_caster(x):
                    if isinstance(x, str):
                        return json.loads(x)
                    else:
                        return x
                casters = []
                for c in colinfo:
                    t = c['type']
                    caster = ub.identity
                    if t.__class__.__name__ == 'JSON':
                        caster = _json_caster
                    # caster = t.result_processor(dialect, t)
                    casters.append(caster)
                proxy._casters = casters
                proxy._colnames = [c['name'] for c in colinfo]
            colnames = proxy._colnames
            casters = proxy._casters

            # Using raw SQL seems much faster
            result = proxy.session.execute(
                'SELECT * FROM {} ORDER BY id'.format(proxy.cls.__tablename__))

            for row in _raw_yielder(result):
                cast_row = [f(x) for f, x in zip(proxy._casters, row)]
                # Note: assert colnames == list(result.keys())
                item = dict(zip(colnames, cast_row))
                yield item

    def __getitem__(proxy, index):
        query = proxy.session.query(proxy.cls)
        if isinstance(index, slice):
            assert index.step in {None, 1}
            objs = query.slice(index.start, index.stop).all()
            items = [orm_to_dict(obj) for obj in objs]
            return items
        else:
            obj = query.slice(index, index + 1).all()[0]
            item = orm_to_dict(obj)
            return item

    def __contains__(proxy, item):
        raise Exception('cannot perform contains operations on SqlListProxy')

    def __setitem__(proxy, index, value):
        raise Exception('SqlListProxy is immutable')

    def __delitem__(proxy, index):
        raise Exception('SqlListProxy is immutable')


class SqlDictProxy(DictLike):
    """
    Duck-types an SQL table as a dictionary of dictionaries.

    The key is specified by an indexed column (by default it is the `id`
    column). The values are dictionaries containing all data for that row.

    Notes:
        With SQLite indexes are B-Trees so lookup is O(log(N)) and not O(1) as
        will regular dictionaries. Iteration should still be O(N), but
        databases have much more overhead than Python dictionaries.

    Args:
        session (Session): the sqlalchemy session
        cls (Type): the declarative sqlalchemy table class
        keyattr : the indexed column to use as the keys

    Example:
        >>> # xdoctest: +REQUIRES(module:sqlalchemy)
        >>> from kwcoco.coco_sql_dataset import *  # NOQA
        >>> import pytest
        >>> sql_dset, dct_dset = demo(num=10)
        >>> proxy = sql_dset.index.anns

        >>> keys = list(proxy.keys())
        >>> values = list(proxy.values())
        >>> items = list(proxy.items())
        >>> item_keys = [t[0] for t in items]
        >>> item_vals = [t[1] for t in items]
        >>> lut_vals = [proxy[key] for key in keys]
        >>> assert item_vals == lut_vals == values
        >>> assert item_keys == keys
        >>> assert len(proxy) == len(keys)

        >>> goodkey1 = keys[1]
        >>> badkey1 = -100000000000
        >>> badkey2 = 'foobarbazbiz'
        >>> badkey3 = object()
        >>> assert goodkey1 in proxy
        >>> assert badkey1 not in proxy
        >>> assert badkey2 not in proxy
        >>> assert badkey3 not in proxy
        >>> with pytest.raises(KeyError):
        >>>     proxy[badkey1]
        >>> with pytest.raises(KeyError):
        >>>     proxy[badkey2]
        >>> with pytest.raises(KeyError):
        >>>     proxy[badkey3]

        >>> # xdoctest: +SKIP
        >>> from kwcoco.coco_sql_dataset import _benchmark_dict_proxy_ops
        >>> ti = _benchmark_dict_proxy_ops(proxy)
        >>> print('ti.measures = {}'.format(ub.repr2(ti.measures, nl=2, align=':', precision=6)))

    Example:
        >>> # xdoctest: +REQUIRES(module:sqlalchemy)
        >>> from kwcoco.coco_sql_dataset import *  # NOQA
        >>> import pytest
        >>> sql_dset, dct_dset = demo(num=10)
        >>> proxy = sql_dset.index.name_to_cat

        >>> keys = list(proxy.keys())
        >>> values = list(proxy.values())
        >>> items = list(proxy.items())
        >>> item_keys = [t[0] for t in items]
        >>> item_vals = [t[1] for t in items]
        >>> lut_vals = [proxy[key] for key in keys]
        >>> assert item_vals == lut_vals == values
        >>> assert item_keys == keys
        >>> assert len(proxy) == len(keys)

        >>> goodkey1 = keys[1]
        >>> badkey1 = -100000000000
        >>> badkey2 = 'foobarbazbiz'
        >>> badkey3 = object()
        >>> assert goodkey1 in proxy
        >>> assert badkey1 not in proxy
        >>> assert badkey2 not in proxy
        >>> assert badkey3 not in proxy
        >>> with pytest.raises(KeyError):
        >>>     proxy[badkey1]
        >>> with pytest.raises(KeyError):
        >>>     proxy[badkey2]
        >>> with pytest.raises(KeyError):
        >>>     proxy[badkey3]

        >>> # xdoctest: +SKIP
        >>> from kwcoco.coco_sql_dataset import _benchmark_dict_proxy_ops
        >>> ti = _benchmark_dict_proxy_ops(proxy)
        >>> print('ti.measures = {}'.format(ub.repr2(ti.measures, nl=2, align=':', precision=6)))
    """
    def __init__(proxy, session, cls, keyattr=None):
        proxy.cls = cls
        proxy.session = session
        proxy.keyattr = keyattr
        proxy._colnames = None
        proxy._casters = None

        # It seems like writing the raw sql ourselves is fater than
        # using the ORM in most cases.
        proxy.ALCHEMY_MODE = 0

        # ONLY DO THIS IN READONLInterfaceError:Y MODE
        proxy._cache = _new_proxy_cache()

    def __len__(proxy):
        query = proxy.session.query(proxy.cls)
        return query.count()

    def __nice__(proxy):
        if proxy.keyattr is None:
            return 'id -> {}: {}'.format(proxy.cls.__tablename__, len(proxy))
        else:
            return '{} -> {}: {}'.format(proxy.keyattr.name, proxy.cls.__tablename__, len(proxy))

    def __contains__(proxy, key):
        if proxy._cache is not None:
            if key in proxy._cache:
                return True
        keyattr = proxy.keyattr
        if keyattr is None:
            keyattr = proxy.cls.id
        try:
            query = proxy.session.query(proxy.cls.id).filter(keyattr == key)
            flag = query.count() > 0
        except sqlalchemy.exc.InterfaceError as ex:
            if 'unsupported type' in str(ex):
                return False
            else:
                raise
        return flag

    def __getitem__(proxy, key):
        if proxy._cache is not None:
            if key in proxy._cache:
                return proxy._cache[key]
        try:
            session = proxy.session
            cls = proxy.cls
            if proxy.keyattr is None:
                query = session.query(cls)
                obj = query.get(key)
                if obj is None:
                    raise KeyError(key)
            else:
                keyattr = proxy.keyattr
                query = session.query(cls)
                results = query.filter(keyattr == key).all()
                if len(results) == 0:
                    raise KeyError(key)
                elif len(results) > 1:
                    raise AssertionError('Should only have 1 result')
                else:
                    obj = results[0]
        except sqlalchemy.exc.InterfaceError as ex:
            if 'unsupported type' in str(ex):
                raise KeyError(key)
            else:
                raise
        item = orm_to_dict(obj)
        if proxy._cache is not None:
            proxy._cache[key] = item
        return item

    def keys(proxy):
        if proxy.ALCHEMY_MODE:
            if proxy.keyattr is None:
                query = proxy.session.query(proxy.cls.id).order_by(proxy.cls.id)
            else:
                query = proxy.session.query(proxy.keyattr).order_by(proxy.cls.id)

            for item in _orm_yielder(query):
                key = item[0]
                yield key
        else:
            # Using raw SQL seems much faster
            if proxy.keyattr is None:
                result = proxy.session.execute(
                    'SELECT id FROM {} ORDER BY id'.format(proxy.cls.__tablename__))
            else:
                # raise NotImplementedError
                result = proxy.session.execute(
                    'SELECT {} FROM {} ORDER BY id'.format(
                        proxy.keyattr.key, proxy.cls.__tablename__))

            for item in _raw_yielder(result):
                yield item[0]

    def itervalues(proxy):
        if proxy.ALCHEMY_MODE:
            query = proxy.session.query(proxy.cls).order_by(proxy.cls.id)
            for obj in _orm_yielder(query):
                item = orm_to_dict(obj)
                yield item
        else:
            if proxy._colnames is None:
                from sqlalchemy import inspect
                import json
                inspector = inspect(proxy.session.get_bind())
                colinfo = inspector.get_columns(proxy.cls.__tablename__)
                # HACK: to fixup json columns, session.execute seems to fix
                # the type for some columns, but not all, hense the isinstance
                def _json_caster(x):
                    if isinstance(x, str):
                        return json.loads(x)
                    else:
                        return x
                casters = []
                for c in colinfo:
                    t = c['type']
                    caster = ub.identity
                    if t.__class__.__name__ == 'JSON':
                        caster = _json_caster
                    # caster = t.result_processor(dialect, t)
                    casters.append(caster)
                proxy._casters = casters
                proxy._colnames = [c['name'] for c in colinfo]
            colnames = proxy._colnames
            casters = proxy._casters

            # Using raw SQL seems much faster
            result = proxy.session.execute(
                'SELECT * FROM {} ORDER BY id'.format(proxy.cls.__tablename__))

            for row in _raw_yielder(result):
                cast_row = [f(x) for f, x in zip(proxy._casters, row)]
                # Note: assert colnames == list(result.keys())
                item = dict(zip(colnames, cast_row))
                yield item

    def iteritems(proxy):
        if proxy.keyattr is None:
            keyattr_name = 'id'
        else:
            keyattr_name = proxy.keyattr.name
        for value in proxy.itervalues():
            yield (value[keyattr_name], value)

    items = iteritems
    values = itervalues


class SqlIdGroupDictProxy(DictLike):
    """
    Similar to :class:`SqlDictProxy`, but maps ids to groups of other ids.

    Simulates a dictionary that maps ids of a parent table to all ids of
    another table corresponding to rows where a specific column has that parent
    id.

    For example, imagine two tables: images with one column (id) and
    annotations with two columns (id, image_id). This class can help provide a
    mpaping from each `image.id` to a `Set[annotation.id]` where those
    annotation rows have `annotation.image_id = image.id`.

    Example:
        >>> # xdoctest: +REQUIRES(module:sqlalchemy)
        >>> from kwcoco.coco_sql_dataset import *  # NOQA
        >>> sql_dset, dct_dset = demo(num=10)
        >>> proxy = sql_dset.index.gid_to_aids

        >>> keys = list(proxy.keys())
        >>> values = list(proxy.values())
        >>> items = list(proxy.items())
        >>> item_keys = [t[0] for t in items]
        >>> item_vals = [t[1] for t in items]
        >>> lut_vals = [proxy[key] for key in keys]
        >>> assert item_vals == lut_vals == values
        >>> assert item_keys == keys
        >>> assert len(proxy) == len(keys)

        >>> # xdoctest: +SKIP
        >>> from kwcoco.coco_sql_dataset import _benchmark_dict_proxy_ops
        >>> ti = _benchmark_dict_proxy_ops(proxy)
        >>> print('ti.measures = {}'.format(ub.repr2(ti.measures, nl=2, align=':', precision=6)))
    """
    def __init__(proxy, session, valattr, keyattr, parent_keyattr):
        proxy.valattr = valattr
        proxy.keyattr = keyattr
        proxy.session = session
        proxy.parent_keyattr = parent_keyattr
        proxy.ALCHEMY_MODE = 0
        proxy._cache = _new_proxy_cache()

    def __nice__(self):
        return str(len(self))

    def __len__(proxy):
        query = proxy.session.query(proxy.parent_keyattr)
        return query.count()

    def __getitem__(proxy, key):
        if proxy._cache is not None:
            if key in proxy._cache:
                return proxy._cache[key]

        session = proxy.session
        keyattr = proxy.keyattr
        valattr = proxy.valattr
        if proxy.ALCHEMY_MODE:
            query = session.query(valattr).filter(keyattr == key)
            item = [row[0] for row in query.all()]
        else:
            sql_expr = 'SELECT {} FROM {} WHERE {}=:key'.format(
                proxy.valattr.name,
                proxy.keyattr.class_.__tablename__,
                proxy.keyattr.name,
            )
            result = proxy.session.execute(sql_expr, params={'key': key})
            item = [row[0] for row in result.fetchall()]
        item = set(item)
        if proxy._cache is not None:
            proxy._cache[key] = item
        return item

    def __contains__(proxy, key):
        if proxy._cache is not None:
            if key in proxy._cache:
                return True
        try:
            query = (proxy.session.query(proxy.parent_keyattr)
                     .filter(proxy.parent_keyattr == key))
            flag = query.count() > 0
        except sqlalchemy.exc.InterfaceError as ex:
            if 'unsupported type' in str(ex):
                return False
            else:
                raise
        return flag

    def keys(proxy):
        if proxy.ALCHEMY_MODE:
            query = proxy.session.query(proxy.parent_keyattr)
            for item in _orm_yielder(query):
                key = item[0]
                yield key
        else:
            result = proxy.session.execute(
                'SELECT {} FROM {}'.format(
                    proxy.parent_keyattr.name,
                    proxy.parent_keyattr.class_.__tablename__))
            for item in _raw_yielder(result):
                yield item[0]

    def iteritems(proxy):
        if proxy.ALCHEMY_MODE:
            parent_keyattr = proxy.parent_keyattr
            keyattr = proxy.keyattr
            valattr = proxy.valattr
            session = proxy.session

            parent_table = parent_keyattr.class_.__table__
            table = keyattr.class_.__table__

            grouped_vals = sqlalchemy.func.json_group_array(valattr, type_=JSON)
            # Hack: have to cast to str because I don't know how to make
            # the json type work
            query = (
                session.query(parent_keyattr, str(grouped_vals))
                .outerjoin(table, parent_keyattr == keyattr)
                .group_by(parent_keyattr)
                .order_by(parent_keyattr)
            )

            for row in query.all():
                key = row[0]
                group = json.loads(row[1])
                if group[0] is None:
                    group = set()
                else:
                    group = set(group)
                tup = (key, group)
                yield tup

        else:
            parent_table = proxy.parent_keyattr.class_.__tablename__
            table = proxy.keyattr.class_.__tablename__
            parent_keycol = parent_table + '.' + proxy.parent_keyattr.name
            keycol = table + '.' + proxy.keyattr.name
            valcol = table + '.' + proxy.valattr.name
            expr = (
                'SELECT {parent_keycol}, json_group_array({valcol}) '
                'FROM {parent_table} '
                'LEFT OUTER JOIN {table} ON {keycol} = {parent_keycol} '
                'GROUP BY {parent_keycol} ORDER BY {parent_keycol}').format(
                    parent_table=parent_table,
                    table=table,
                    parent_keycol=parent_keycol,
                    keycol=keycol,
                    valcol=valcol,
                )
            print(expr)
            result = proxy.session.execute(expr)
            for row in result.fetchall():
                key = row[0]
                group = json.loads(row[1])
                if group[0] is None:
                    group = set()
                else:
                    group = set(group)
                tup = (key, group)
                yield tup

    def itervalues(proxy):
        if proxy.ALCHEMY_MODE:
            # Hack:
            for key, val in proxy.items():
                yield val
            # parent_keyattr = proxy.parent_keyattr
            # keyattr = proxy.keyattr
            # valattr = proxy.valattr
            # session = proxy.session

            # parent_table = parent_keyattr.class_.__table__
            # table = proxy.keyattr.class_.__tablename__

            # # TODO: This might have to be different for PostgreSQL
            # grouped_vals = sqlalchemy.func.json_group_array(valattr, type_=JSON)
            # query = (
            #     session.query(grouped_vals)
            #     .outerjoin(table, parent_keyattr == keyattr)
            #     .group_by(parent_keyattr)
            #     .order_by(parent_keyattr)
            # )
            # for row in _orm_yielder(query):
            #     group = row[0]
            #     if group[0] is None:
            #         group = set()
            #     else:
            #         group = set(group)
            #     yield group
        else:
            parent_table = proxy.parent_keyattr.class_.__tablename__
            table = proxy.keyattr.class_.__tablename__
            parent_keycol = parent_table + '.' + proxy.parent_keyattr.name
            keycol = table + '.' + proxy.keyattr.name
            valcol = table + '.' + proxy.valattr.name
            expr = (
                'SELECT {parent_keycol}, json_group_array({valcol}) '
                'FROM {parent_table} '
                'LEFT OUTER JOIN {table} ON {keycol} = {parent_keycol} '
                'GROUP BY {parent_keycol} ORDER BY {parent_keycol}').format(
                    parent_table=parent_table,
                    table=table,
                    parent_keycol=parent_keycol,
                    keycol=keycol,
                    valcol=valcol,
                )
            # print(expr)
            result = proxy.session.execute(expr)
            for row in result.fetchall():
                group = json.loads(row[1])
                if group[0] is None:
                    group = set()
                else:
                    group = set(group)
                yield group


class CocoSqlIndex(object):
    """
    Simulates the dictionary provided by CocoIndex
    """
    def __init__(index):
        index.anns = None
        index.imgs = None
        index.videos = None
        index.cats = None
        index.file_name_to_img = None

        index.dataset = None
        index._id_lookup = None

    def build(index, parent):
        session = parent.session
        index.anns = SqlDictProxy(session, Annotation)
        index.imgs = SqlDictProxy(session, Image)
        index.cats = SqlDictProxy(session, Category)
        index.kpcats = SqlDictProxy(session, KeypointCategory)
        index.videos = SqlDictProxy(session, Video)
        index.name_to_cat = SqlDictProxy(session, Category, Category.name)
        index.file_name_to_img = SqlDictProxy(session, Image, Image.file_name)

        index.gid_to_aids = SqlIdGroupDictProxy(
            session, Annotation.id, Annotation.image_id, Image.id)
        index.cid_to_aids = SqlIdGroupDictProxy(
            session, Annotation.id, Annotation.category_id, Category.id)
        index.vidid_to_gids = SqlIdGroupDictProxy(
            session, Image.id, Image.video_id, Video.id)

        # Make a list like view for algorithms
        index.dataset = {
            'annotations': SqlListProxy(session, Annotation),
            'videos': SqlListProxy(session, Video),
            'categories': SqlListProxy(session, Category),
            'keypoint_categories': SqlListProxy(session, KeypointCategory),
            'images': SqlListProxy(session, Image),
        }

        index._id_lookup = {
            'categories': index.cats,
            'images': index.imgs,
            'annotations': index.anns,
            'videos': index.videos,
        }


class CocoSqlDatabase(MixinCocoJSONAccessors, MixinCocoAccessors,
                      MixinCocoAttrs, MixinCocoStats, MixinCocoDraw,
                      ub.NiceRepr):
    """
    Provides an API nearly identical to :class:`kwcoco.CocoDatabase`, but uses
    an SQL backend data store. This makes it robust to copy-on-write memory
    issues that arise when forking, as discussed in [1]_.

    References:
        .. [1] https://github.com/pytorch/pytorch/issues/13246

    Example:
        >>> # xdoctest: +REQUIRES(module:sqlalchemy)
        >>> from kwcoco.coco_sql_dataset import *  # NOQA
        >>> sql_dset, dct_dset = demo()
        >>> assert_dsets_allclose(sql_dset, dct_dset)
    """

    MEMORY_URI = 'sqlite:///:memory:'

    def __init__(self, uri=None, tag=None, img_root=None):
        if uri is None:
            uri = self.MEMORY_URI
        self.uri = uri
        self.img_root = img_root
        self.session = None
        self.engine = None
        self.index = CocoSqlIndex()
        self.tag = tag

    def __nice__(self):
        if self.dataset is None:
            return 'not connected'

        parts = []
        parts.append('tag={}'.format(self.tag))
        if self.dataset is not None:
            info = ub.repr2(self.basic_stats(), kvsep='=', si=1, nobr=1, nl=0)
            parts.append(info)
        return ', '.join(parts)

    def __getstate__(self):
        """
        Return only the minimal info when pickling this object.

        Note:
            This object IS pickling when the multiprocessing context is
            "spawn".

            This object is NOT pickled when the multiprocessing context is
            "fork".  In this case the user needs to be careful to create new
            connections in the forked subprocesses.

        Example:
            >>> # xdoctest: +REQUIRES(module:sqlalchemy)
            >>> from kwcoco.coco_sql_dataset import *  # NOQA
            >>> sql_dset, dct_dset = demo()
            >>> # Test pickling works correctly
            >>> import pickle
            >>> serialized = pickle.dumps(sql_dset)
            >>> assert len(serialized) < 3e4, 'should be very small'
            >>> copy = pickle.loads(serialized)
            >>> dset1, dset2, tag1, tag2 = sql_dset, copy, 'orig', 'copy'
            >>> assert_dsets_allclose(dset1, dset2, tag1, tag2)
            >>> # --- other methods of copying ---
            >>> rw_copy = CocoSqlDatabase(
            >>>     sql_dset.uri, img_root=sql_dset.img_root, tag=sql_dset.tag)
            >>> rw_copy.connect()
            >>> ro_copy = CocoSqlDatabase(
            >>>     sql_dset.uri, img_root=sql_dset.img_root, tag=sql_dset.tag)
            >>> ro_copy.connect(readonly=True)
            >>> assert_dsets_allclose(dset1, ro_copy, tag1, 'ro-copy')
            >>> assert_dsets_allclose(dset1, rw_copy, tag1, 'rw-copy')
        """
        if self.uri == self.MEMORY_URI:
            raise Exception('Cannot Pickle Anonymous In-Memory Databases')
        return {
            'uri': self.uri,
            'img_root': self.img_root,
            'tag': self.tag,
        }

    def __setstate__(self, state):
        """
        Reopen new readonly connnections when unpickling the object.
        """
        self.__dict__.update(state)
        self.session = None
        self.engine = None
        # Make unpickled objects readonly
        self.index = CocoSqlIndex()
        self.connect(readonly=True)

    def connect(self, readonly=False):
        """
        References:
            # details on read only mode, some of these didnt seem to work
            https://github.com/sqlalchemy/sqlalchemy/blob/master/lib/sqlalchemy/dialects/sqlite/pysqlite.py#L71
            https://github.com/pudo/dataset/issues/136
            https://writeonly.wordpress.com/2009/07/16/simple-read-only-sqlalchemy-sessions/
        """
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy import create_engine
        # Create an engine that stores data at a specific uri location
        uri = self.uri
        if readonly:
            uri = uri + '?mode=ro&uri=true'
        elif uri.startswith('sqlite:///file:'):
            uri = uri + '?uri=true'
        self.engine = create_engine(uri)
        if len(self.engine.table_names()) == 0:
            # Opened an empty database, need to create the tables
            # Create all tables in the engine.
            # This is equivalent to "Create Table" statements in raw SQL.
            # if readonly:
            #     raise AssertionError('must open existing table in readonly mode')
            CocoBase.metadata.create_all(self.engine)
        DBSession = sessionmaker(bind=self.engine)
        self.session = DBSession()
        self._build_index()

    def delete(self):
        fpath = self.uri.split('///file:')[-1]
        if self.uri != self.MEMORY_URI and exists(fpath):
            ub.delete(fpath)

    def populate_from(self, dset):
        """
        Copy the information in a :class:`CocoDataset` into this SQL database.

        Example:
            >>> # xdoctest: +REQUIRES(module:sqlalchemy)
            >>> import kwcoco
            >>> from kwcoco.coco_sql_dataset import *
            >>> dset2 = dset = kwcoco.CocoDataset.demo()
            >>> dset1 = self = CocoSqlDatabase('sqlite:///:memory:')
            >>> self.connect()
            >>> self.populate_from(dset)
            >>> assert_dsets_allclose(dset1, dset2, tag1='sql', tag2='dct')
            >>> ti_sql = _benchmark_dset_readtime(dset1, 'sql')
            >>> ti_dct = _benchmark_dset_readtime(dset2, 'dct')
            >>> print('ti_sql.rankings = {}'.format(ub.repr2(ti_sql.rankings, nl=2, precision=6, align=':')))
            >>> print('ti_dct.rankings = {}'.format(ub.repr2(ti_dct.rankings, nl=2, precision=6, align=':')))
        """
        from sqlalchemy import inspect
        session = self.session
        inspector = inspect(self.engine)
        for key in self.engine.table_names():
            colinfo = inspector.get_columns(key)
            colnames = {c['name'] for c in colinfo}
            # TODO: is there a better way to grab this information?
            cls = CocoBase.TBLNAME_TO_CLASS[key]
            for item in dset.dataset.get(key, []):
                item_ = ub.dict_isect(item, colnames)
                # Everything else is a foreign key
                item_['foreign'] = ub.dict_diff(item, item_)
                if key == 'annotations':
                    # Need custom code to translate list-based properties
                    x, y, w, h = item_.get('bbox', [None, None, None, None])
                    item_['_bbox_x'] = x
                    item_['_bbox_y'] = y
                    item_['_bbox_w'] = w
                    item_['_bbox_h'] = h
                row = cls(**item_)
                session.add(row)
        session.commit()

    def _build_index(self):
        self.index.build(self)

    @property
    def dataset(self):
        return self.index.dataset

    @property
    def anns(self):
        return self.index.anns

    @property
    def cats(self):
        return self.index.cats

    @property
    def imgs(self):
        return self.index.imgs

    @property
    def name_to_cat(self):
        return self.index.name_to_cat

    def raw_table(self, table_name):
        """
        Loads an entire SQL table as a pandas DataFrame

        Args:
            table_name (str): name of the table

        Returns:
            DataFrame

        Example:
            >>> # xdoctest: +REQUIRES(module:sqlalchemy)
            >>> from kwcoco.coco_sql_dataset import *  # NOQA
            >>> self, dset = demo()
            >>> table_df = self.raw_table('annotations')
            >>> print(table_df)
        """
        import pandas as pd
        table_df = pd.read_sql_table(table_name, con=self.engine)
        return table_df

    def tabular_targets(self):
        """
        Convinience method to create an in-memory summary of basic annotation
        properties with minimal SQL overhead.

        Example:
            >>> # xdoctest: +REQUIRES(module:sqlalchemy)
            >>> from kwcoco.coco_sql_dataset import *  # NOQA
            >>> self, dset = demo()
            >>> targets = self.tabular_targets()
            >>> print(targets.pandas())
        """
        import kwarray
        stmt = ub.paragraph(
            '''
            SELECT
                annotations.id, image_id, category_id,
                _bbox_x + (_bbox_w / 2), _bbox_y + (_bbox_h / 2),
                _bbox_w, _bbox_h, images.width, images.height
            FROM annotations
            JOIN images on images.id = annotations.image_id
            ''')
        result = self.session.execute(stmt)
        rows = result.fetchall()
        aids, gids, cids, cxs, cys, ws, hs, img_ws, img_hs = list(zip(*rows))

        table = {
            # Annotation / Image / Category ids
            'aid': np.array(aids, dtype=np.int32),
            'gid': np.array(gids, dtype=np.int32),
            'category_id': np.array(cids, dtype=np.int32),
            # Subpixel box localizations wrt parent image
            'cx': np.array(cxs, dtype=np.float32),
            'cy': np.array(cys, dtype=np.float32),
            'width': np.array(ws, dtype=np.float32),
            'height': np.array(hs, dtype=np.float32),
        }

        # Parent image id and width / height
        table['img_width'] = np.array(img_ws, dtype=np.int32)
        table['img_height'] = np.array(img_hs, dtype=np.int32)
        targets = kwarray.DataFrameArray(table)
        return targets


def ensure_sql_coco_view(dset, db_fpath=None):
    """
    Create a cached on-disk SQL view of an on-disk COCO dataset.

    Note:
        This function is fragile. It depends on looking at file modified
        timestamps to determine if it needs to write the dataset.
    """
    if db_fpath is None:
        db_fpath = ub.augpath(dset.fpath, prefix='.', ext='.view.v002.sqlite')

    db_uri = 'sqlite:///file:' + db_fpath
    # dpath = dirname(dset.fpath)

    self = CocoSqlDatabase(db_uri, img_root=dset.img_root, tag=dset.tag)

    import os
    needs_rewrite = True
    if exists(db_fpath):
        needs_rewrite = (
            os.stat(dset.fpath).st_mtime >
            os.stat(db_fpath).st_mtime
        )
    if needs_rewrite:
        # Write to the SQL instance
        self.delete()
        self.connect()
        # Convert a coco file to an sql database
        self.populate_from(dset)
    else:
        self.connect()
    return self


def demo(num=10):
    import kwcoco
    dset = kwcoco.CocoDataset.demo(
        'vidshapes', num_videos=1, num_frames=num, gsize=(64, 64))
    HACK = 1
    if HACK:
        gids = list(dset.imgs.keys())
        aids1 = dset.gid_to_aids[gids[-2]]
        aids2 = dset.gid_to_aids[gids[-4]]
        # print('aids1 = {!r}'.format(aids1))
        # print('aids2 = {!r}'.format(aids2))
        dset.remove_annotations(aids1 | aids2)
        dset.fpath = ub.augpath(dset.fpath, suffix='_hack', multidot=True)
        if not exists(dset.fpath):
            dset.dump(dset.fpath, newlines=True)
    self = ensure_sql_coco_view(dset)
    return self, dset


def indexable_allclose(dct1, dct2, return_info=False):
    """
    Walks through two nested data structures and ensures that everything is
    roughly the same.

    Args:
        dct1: a nested indexable item
        dct2: a nested indexable item

    Example:
        >>> # xdoctest: +REQUIRES(module:sqlalchemy)
        >>> dct1 = {
        >>>     'foo': [1.222222, 1.333],
        >>>     'bar': 1,
        >>>     'baz': [],
        >>> }
        >>> dct2 = {
        >>>     'foo': [1.22222, 1.333],
        >>>     'bar': 1,
        >>>     'baz': [],
        >>> }
        >>> assert indexable_allclose(dct1, dct2)
    """
    from kwcoco.util.util_json import IndexableWalker
    walker1 = IndexableWalker(dct1)
    walker2 = IndexableWalker(dct2)
    flat_items1 = [
        (path, value) for path, value in walker1
        if not isinstance(value, walker1.indexable_cls) or len(value) == 0]
    flat_items2 = [
        (path, value) for path, value in walker2
        if not isinstance(value, walker1.indexable_cls) or len(value) == 0]

    flat_items1 = sorted(flat_items1)
    flat_items2 = sorted(flat_items2)

    if len(flat_items1) != len(flat_items2):
        info = {
            'faillist': ['length mismatch']
        }
        final_flag = False
    else:
        passlist = []
        faillist = []

        for t1, t2 in zip(flat_items1, flat_items2):
            p1, v1 = t1
            p2, v2 = t2
            assert p1 == p2

            flag = (v1 == v2)
            if not flag:
                if isinstance(v1, float) and isinstance(v2, float) and np.isclose(v1, v2):
                    flag = True
            if flag:
                passlist.append(p1)
            else:
                faillist.append((p1, v1, v2))

        final_flag = len(faillist) == 0
        info = {
            'passlist': passlist,
            'faillist': faillist,
        }

    if return_info:
        return final_flag, info
    else:
        return final_flag


def assert_dsets_allclose(dset1, dset2, tag1='sql', tag2='dct'):
    # Test that the duck types are working
    compare = {}
    compare['gid_to_aids'] = {
        tag1: dict(dset1.index.gid_to_aids),
        tag2: dict(dset2.index.gid_to_aids)}
    compare['cid_to_aids'] = {
        tag1: dict(dset1.index.cid_to_aids),
        tag2: dict(dset2.index.cid_to_aids)}
    compare['vidid_to_gids'] = {
        tag1: dict(dset1.index.vidid_to_gids),
        tag2: dict(dset2.index.vidid_to_gids)}
    for key, pair in compare.items():
        lut1 = pair[tag1]
        lut2 = pair[tag2]
        assert lut1 == lut2, (
            'Failed {} on lut1={!r}, lut2={!r}'.format(key, lut1, lut2))
    # ------
    # The row dictionaries may have extra Nones on the SQL side
    # So the comparison logic is slightly more involved here
    compare = {}
    compare['imgs'] = {
        tag1: dict(dset1.index.imgs),
        tag2: dict(dset2.index.imgs)}
    compare['anns'] = {
        tag1: dict(dset1.index.anns),
        tag2: dict(dset2.index.anns)}
    compare['cats'] = {
        tag1: dict(dset1.index.cats),
        tag2: dict(dset2.index.cats)}
    compare['file_name_to_img'] = {
        tag1: dict(dset1.index.file_name_to_img),
        tag2: dict(dset2.index.file_name_to_img)}
    compare['name_to_cat'] = {
        tag1: dict(dset1.index.name_to_cat),
        tag2: dict(dset2.index.name_to_cat)}
    special_cols = {'_bbox_x', '_bbox_y', '_bbox_w', '_bbox_h'}
    for key, pair in compare.items():
        lut1 = pair[tag1]
        lut2 = pair[tag2]
        keys = set(lut1.keys()) & set(lut2.keys())
        assert len(keys) == len(lut2) == len(lut1)
        for key in keys:
            item1 = ub.dict_diff(lut1[key], special_cols)
            item2 = ub.dict_diff(lut2[key], special_cols)
            item1.update(item1.pop('foreign', {}))
            item2.update(item2.pop('foreign', {}))
            common1 = ub.dict_isect(item2, item1)
            common2 = ub.dict_isect(item1, item2)
            diff1 = ub.dict_diff(item1, common2)
            diff2 = ub.dict_diff(item2, common1)
            assert indexable_allclose(common2, common1)
            assert all(v is None for v in diff2.values())
            assert all(v is None for v in diff1.values())


def _benchmark_dset_readtime(dset, tag='?'):
    """
    Helper for understanding the time differences between backends
    """

    import timerit
    ti = timerit.Timerit(4, bestof=2, verbose=2)

    for timer in ti.reset('{} dict(gid_to_aids)'.format(tag)):
        with timer:
            dict(dset.index.gid_to_aids)

    for timer in ti.reset('{} dict(cid_to_aids)'.format(tag)):
        with timer:
            dict(dset.index.cid_to_aids)

    for timer in ti.reset('{} dict(imgs)'.format(tag)):
        with timer:
            dict(dset.index.imgs)

    for timer in ti.reset('{} dict(cats)'.format(tag)):
        with timer:
            dict(dset.index.cats)

    for timer in ti.reset('{} dict(anns)'.format(tag)):
        with timer:
            dict(dset.index.anns)

    for timer in ti.reset('{} dict(vidid_to_gids)'.format(tag)):
        with timer:
            dict(dset.index.vidid_to_gids)

    for timer in ti.reset('{} ann list iteration'.format(tag)):
        with timer:
            list(dset.dataset['annotations'])

    for timer in ti.reset('{} ann dict iteration'.format(tag)):
        with timer:
            list(dset.index.anns.items())

    for timer in ti.reset('{} ann random lookup'.format(tag)):
        aids = list(dset.index.anns.keys())[0:10]
        with timer:
            for aid in aids:
                dset.index.anns[aid]

    return ti


def _benchmark_dict_proxy_ops(proxy):
    """
    Get insight on the efficiency of operations
    """
    import timerit
    orig_mode = proxy.ALCHEMY_MODE

    ti = timerit.Timerit(1, bestof=1, verbose=2)

    results = ub.ddict(dict)

    proxy.ALCHEMY_MODE = 1
    for timer in ti.reset('keys alc sql'):
        with timer:
            results['keys']['alc'] = list(proxy.keys())

    proxy.ALCHEMY_MODE = 0
    for timer in ti.reset('keys raw sql'):
        with timer:
            results['keys']['raw'] = list(proxy.keys())

    proxy.ALCHEMY_MODE = 1
    for timer in ti.reset('values alc sql'):
        with timer:
            results['vals']['alc'] = list(proxy.values())

    proxy.ALCHEMY_MODE = 0
    for timer in ti.reset('values raw sql'):
        with timer:
            results['vals']['raw'] = list(proxy.values())

    proxy.ALCHEMY_MODE = 1
    for timer in ti.reset('naive values'):
        with timer:
            results['vals']['alc-naive'] = [proxy[key] for key in proxy.keys()]

    proxy.ALCHEMY_MODE = 0
    for timer in ti.reset('naive values'):
        with timer:
            results['vals']['raw-naive'] = [proxy[key] for key in proxy.keys()]

    proxy.ALCHEMY_MODE = 1
    for timer in ti.reset('items alc sql'):
        with timer:
            results['items']['alc'] = list(proxy.items())

    proxy.ALCHEMY_MODE = 0
    for timer in ti.reset('items raw sql'):
        with timer:
            results['items']['raw'] = list(proxy.items())

    for key, modes in results.items():
        if not ub.allsame(modes.values()):
            raise AssertionError('Inconsistency in {!r}'.format(key))

    proxy.ALCHEMY_MODE = orig_mode
    return ti


def devcheck():
    """
    Scratch work for things that should eventually become unit or doc tests

    from kwcoco.coco_sql_dataset import *  # NOQA
    self, dset = demo()
    """
    # self = ensure_sql_coco_view(dset, db_fpath=':memory:')
    self, dset = demo()

    ti_sql = _benchmark_dset_readtime(self, 'sql')
    ti_dct = _benchmark_dset_readtime(dset, 'dct')
    print('ti_sql.rankings = {}'.format(ub.repr2(ti_sql.rankings, nl=2, precision=6, align=':')))
    print('ti_dct.rankings = {}'.format(ub.repr2(ti_dct.rankings, nl=2, precision=6, align=':')))

    # Read the sql tables into pandas
    for key in self.engine.table_names():
        print('\n----')
        print('key = {!r}'.format(key))
        table_df = self.raw_table(key)
        print(table_df)

    import ndsampler
    self.hashid = 'foobarjunk'
    sampler = ndsampler.CocoSampler(self, backend=None)

    regions = sampler.regions
    regions.isect_index
    regions.get_segmentations([1, 3])
    regions.get_positive(1)
    regions.get_negative()

    import timerit
    with timerit.Timer('annots'):
        self.annots()

    proxy = self.index.imgs
    gids = list(proxy.keys())

    chosen_gids = gids[100:1000:4]

    query = proxy.session.query(proxy.cls).order_by(proxy.cls.id)
    print(query.statement)

    query = proxy.session.query(proxy.cls).filter(proxy.cls.id.in_(chosen_gids)).order_by(proxy.cls.id)
    stmt = query.statement.compile(compile_kwargs={"literal_binds": True})
    print(stmt)

    with timerit.Timer('query with in hardcode'):
        query = proxy.session.query(proxy.cls).filter(proxy.cls.id.in_(chosen_gids)).order_by(proxy.cls.id)
        stmt = query.statement.compile(compile_kwargs={"literal_binds": True})
        proxy.session.execute(str(stmt)).fetchall()

    with timerit.Timer('query with in'):
        query = proxy.session.query(proxy.cls).filter(proxy.cls.id.in_(chosen_gids)).order_by(proxy.cls.id)
        items1 = query.all()

    with timerit.Timer('naive'):
        items2 = [proxy[gid] for gid in chosen_gids]

    print(query.statement)