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
      way to do this better than ``O(K log(N))``? I tried using a
      ``SELECT col FROM table WHERE id IN (?, ?, ?, ?, ...)`` filling in
      enough `?` as there are rows in my subset. I'm not sure what the
      complexity of using a query like this is. I'm not sure what the `IN`
      implementation looks like. Can this be done more efficiently by
      with a temporary table and a ``JOIN``?

    * There really is no way to do ``O(1)`` row lookup in sqlite right?
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
from os.path import exists, dirname

from kwcoco.util.dict_like import DictLike  # NOQA
from kwcoco.abstract_coco_dataset import AbstractCocoDataset
from kwcoco.coco_dataset import (  # NOQA
    MixinCocoAccessors, MixinCocoObjects,
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

    extra = Column(JSON, default=dict())


class KeypointCategory(CocoBase):
    __tablename__ = 'keypoint_categories'
    id = Column(Integer, primary_key=True, doc='unique internal id')
    name = Column(String(256), doc='unique external name or identifier', index=True, unique=True)
    alias = Column(JSON, doc='list of alter egos')
    supercategory = Column(String(256), doc='coarser category name')
    reflection_id = Column(Integer, doc='if augmentation reflects the image, change keypoint id to this')

    extra = Column(JSON, default=dict())


class Video(CocoBase):
    __tablename__ = 'videos'
    id = Column(Integer, primary_key=True, doc='unique internal id')
    name = Column(String(256), nullable=False, index=True, unique=True)
    caption = Column(String(256), nullable=True)

    width = Column(Integer)
    height = Column(Integer)

    extra = Column(JSON, default=dict())


class Image(CocoBase):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True, doc='unique internal id')

    name = Column(String(512), nullable=True, index=True, unique=True)
    file_name = Column(String(512), nullable=True, index=True, unique=True)

    width = Column(Integer)
    height = Column(Integer)

    video_id = Column(Integer, index=True, unique=False)
    timestamp = Column(Float)  # FIXME, timestamp does not have to be a float
    frame_index = Column(Integer)

    channels = Column(JSON)
    auxiliary = Column(JSON)

    extra = Column(JSON, default=dict())


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
    # weight = Column(Float)

    score = Column(Float)
    weight = Column(Float)
    prob = Column(JSON)

    iscrowd = Column(Integer)
    caption = Column(JSON)

    extra = Column(JSON, default=dict())

# As long as the flavor of sql conforms to our raw sql commands set
# this to 0, which uses the faster raw variant.
ALCHEMY_MODE_DEFAULT = 0

# Global book keeping (It would be nice to find a way to avoid this)
CocoBase.TBLNAME_TO_CLASS = {}
# sqlalchemy v 1.3.23 is the last to have _decl_class_registry
# v1.4 does not have it
if hasattr(CocoBase, '_decl_class_registry'):
    for classname, cls in CocoBase._decl_class_registry.items():
        if not classname.startswith('_'):
            tblname = cls.__tablename__
            CocoBase.TBLNAME_TO_CLASS[tblname] = cls
else:
    for mapper in CocoBase.registry.mappers:
        cls = mapper.class_
        classname = cls.__name__
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
        return util_lru.LRUDict.new(max_size=1000, impl='auto')
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
        proxy.ALCHEMY_MODE = ALCHEMY_MODE_DEFAULT

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

    Note:
        With SQLite indexes are B-Trees so lookup is O(log(N)) and not O(1) as
        will regular dictionaries. Iteration should still be O(N), but
        databases have much more overhead than Python dictionaries.

    Args:
        session (Session): the sqlalchemy session
        cls (Type): the declarative sqlalchemy table class
        keyattr : the indexed column to use as the keys
        ignore_null (bool): if True, ignores any keys set to NULL, otherwise
            NULL keys are allowed.

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
        >>> import kwcoco
        >>> # Test the variant of the SqlDictProxy where we ignore None keys
        >>> # This is the case for name_to_img and file_name_to_img
        >>> dct_dset = kwcoco.CocoDataset.demo('shapes1')
        >>> dct_dset.add_image(name='no_file_image1')
        >>> dct_dset.add_image(name='no_file_image2')
        >>> dct_dset.add_image(name='no_file_image3')
        >>> sql_dset = dct_dset.view_sql(memory=True)
        >>> assert len(dct_dset.index.imgs) == 4
        >>> assert len(dct_dset.index.file_name_to_img) == 1
        >>> assert len(dct_dset.index.name_to_img) == 3
        >>> assert len(sql_dset.index.imgs) == 4
        >>> assert len(sql_dset.index.file_name_to_img) == 1
        >>> assert len(sql_dset.index.name_to_img) == 3

        >>> proxy = sql_dset.index.file_name_to_img
        >>> assert len(list(proxy.keys())) == 1
        >>> assert len(list(proxy.values())) == 1

        >>> proxy = sql_dset.index.name_to_img
        >>> assert len(list(proxy.keys())) == 3
        >>> assert len(list(proxy.values())) == 3

        >>> proxy = sql_dset.index.imgs
        >>> assert len(list(proxy.keys())) == 4
        >>> assert len(list(proxy.values())) == 4
    """
    def __init__(proxy, session, cls, keyattr=None, ignore_null=False):
        proxy.cls = cls
        proxy.session = session
        proxy.keyattr = keyattr
        proxy._colnames = None
        proxy._casters = None
        proxy.ignore_null = ignore_null

        # It seems like writing the raw sql ourselves is fater than
        # using the ORM in most cases.
        proxy.ALCHEMY_MODE = ALCHEMY_MODE_DEFAULT

        # ONLY USE IN READONLY MODE
        proxy._cache = _new_proxy_cache()

    def __len__(proxy):
        if proxy.keyattr is None:
            query = proxy.session.query(proxy.cls)
        else:
            query = proxy.session.query(proxy.keyattr)
            if proxy.ignore_null:
                query = query.filter(proxy.keyattr != None)  # NOQA
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
        if proxy.ignore_null and key is None:
            return False
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
        if proxy.ignore_null and key is None:
            raise KeyError(key)
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

            if proxy.ignore_null:
                query = query.filter(proxy.keyattr != None)  # NOQA

            for item in _orm_yielder(query):
                key = item[0]
                yield key
        else:
            # Using raw SQL seems much faster
            keyattr = 'id' if proxy.keyattr is None else proxy.keyattr.key
            if proxy.ignore_null:
                expr = 'SELECT {} FROM {} WHERE {} IS NOT NULL ORDER BY id'.format(
                    keyattr, proxy.cls.__tablename__, keyattr)
            else:
                expr = 'SELECT {} FROM {} ORDER BY id'.format(
                    keyattr, proxy.cls.__tablename__)
            result = proxy.session.execute(expr)
            for item in _raw_yielder(result):
                yield item[0]

    def values(proxy):
        if proxy.ALCHEMY_MODE:
            query = proxy.session.query(proxy.cls)
            if proxy.ignore_null:
                # I think this is only needed in the autoreload session
                # otherwise it might be possible to jsut use proxy.keyattr
                cls_keyattr = getattr(proxy.cls, proxy.keyattr.key)
                query = query.filter(cls_keyattr != None)  # NOQA
            query = query.order_by(proxy.cls.id)
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
            if proxy.ignore_null:
                keyattr = 'id' if proxy.keyattr is None else proxy.keyattr.key
                expr = 'SELECT {} FROM {} WHERE {} IS NOT NULL ORDER BY id'.format(
                    keyattr, proxy.cls.__tablename__, keyattr)
            else:
                expr = (
                    'SELECT * FROM {} ORDER BY id'.format(proxy.cls.__tablename__))
            result = proxy.session.execute(expr)

            for row in _raw_yielder(result):
                cast_row = [f(x) for f, x in zip(proxy._casters, row)]
                # Note: assert colnames == list(result.keys())
                item = dict(zip(colnames, cast_row))
                yield item

    def items(proxy):
        if proxy.keyattr is None:
            keyattr_name = 'id'
        else:
            keyattr_name = proxy.keyattr.name
        for value in proxy.values():
            yield (value[keyattr_name], value)


class SqlIdGroupDictProxy(DictLike):
    """
    Similar to :class:`SqlDictProxy`, but maps ids to groups of other ids.

    Simulates a dictionary that maps ids of a parent table to all ids of
    another table corresponding to rows where a specific column has that parent
    id.

    The items in the group can be sorted by the ``group_order_attr`` if
    specified.

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

    Example:
        >>> # xdoctest: +REQUIRES(module:sqlalchemy)
        >>> from kwcoco.coco_sql_dataset import *  # NOQA
        >>> import kwcoco
        >>> # Test the group sorted variant of this by using vidid_to_gids
        >>> # where the "gids" must be sorted by the image frame indexes
        >>> dct_dset = kwcoco.CocoDataset.demo('vidshapes1')
        >>> dct_dset.add_image(name='frame-index-order-demo1', frame_index=-30, video_id=1)
        >>> dct_dset.add_image(name='frame-index-order-demo2', frame_index=10, video_id=1)
        >>> dct_dset.add_image(name='frame-index-order-demo3', frame_index=3, video_id=1)
        >>> dct_dset.add_video(name='empty-video1')
        >>> dct_dset.add_video(name='empty-video2')
        >>> dct_dset.add_video(name='empty-video3')
        >>> sql_dset = dct_dset.view_sql(memory=True)
        >>> orig = dct_dset.index.vidid_to_gids
        >>> proxy = sql_dset.index.vidid_to_gids
        >>> from kwcoco.util.util_json import indexable_allclose
        >>> assert indexable_allclose(orig, dict(proxy))
        >>> items = list(proxy.items())
        >>> vals = list(proxy.values())
        >>> keys = list(proxy.keys())
        >>> assert len(keys) == len(vals)
        >>> assert dict(zip(keys, vals)) == dict(items)
    """
    def __init__(proxy, session, valattr, keyattr, parent_keyattr,
                 group_order_attr=None):
        proxy.valattr = valattr
        proxy.keyattr = keyattr
        proxy.session = session
        proxy.parent_keyattr = parent_keyattr
        proxy.ALCHEMY_MODE = 0
        proxy._cache = _new_proxy_cache()

        # if specified, the items within groups are ordered by this attr
        proxy.group_order_attr = group_order_attr

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
            if proxy.group_order_attr is not None:
                query = query.order_by(proxy.group_order_attr)
            item = [row[0] for row in query.all()]
        else:
            if proxy.group_order_attr is None:
                sql_expr = 'SELECT {} FROM {} WHERE {}=:key'.format(
                    proxy.valattr.name,
                    proxy.keyattr.class_.__tablename__,
                    proxy.keyattr.name,
                )
            else:
                sql_expr = 'SELECT {} FROM {} WHERE {}=:key ORDER BY {}'.format(
                    proxy.valattr.name,
                    proxy.keyattr.class_.__tablename__,
                    proxy.keyattr.name,
                    proxy.group_order_attr.name,
                )
            result = proxy.session.execute(sql_expr, params={'key': key})
            item = [row[0] for row in result.fetchall()]
        _set = set if proxy.group_order_attr is None else ub.oset
        item = _set(item)
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
            query = query.order_by(proxy.parent_keyattr)
            for item in _orm_yielder(query):
                key = item[0]
                yield key
        else:
            result = proxy.session.execute(
                'SELECT {} FROM {} ORDER BY {}'.format(
                    proxy.parent_keyattr.name,
                    proxy.parent_keyattr.class_.__tablename__,
                    proxy.parent_keyattr.name
                ))
            for item in _raw_yielder(result):
                yield item[0]

    def items(proxy):
        _set = set if proxy.group_order_attr is None else ub.oset
        if proxy.group_order_attr is not None:
            print('proxy.group_order_attr = {!r}'.format(proxy.group_order_attr))
            # ALTERNATIVE MODE, EXPERIMENTAL MODE
            # groups based on post processing. this might be faster or more
            # robust than the group json array? Requires a hack to yield empty
            # groups. Should consolidate to use one method or the other.
            from collections import deque
            import itertools as it

            # hack to yield, empty groups
            all_keys = deque(proxy.keys())

            if proxy.ALCHEMY_MODE:
                keyattr = proxy.keyattr
                valattr = proxy.valattr
                session = proxy.session
                table = keyattr.class_.__table__
                query = session.query(keyattr, valattr).order_by(
                    keyattr, proxy.group_order_attr)
                result = session.execute(query)
                yielder = _raw_yielder(result)
            else:
                table = proxy.keyattr.class_.__tablename__
                keycol = table + '.' + proxy.keyattr.name
                valcol = table + '.' + proxy.valattr.name
                groupcol = table + '.' + proxy.group_order_attr.name

                expr = (
                    'SELECT {keycol}, {valcol} '
                    'FROM {table} '
                    'ORDER BY {keycol}, {group_order_attr}').format(
                        table=table,
                        keycol=keycol,
                        valcol=valcol,
                        group_order_attr=groupcol,
                    )
                result = proxy.session.execute(expr)
                yielder = _raw_yielder(result)

            for key, grouper in it.groupby(yielder, key=lambda x: x[0]):
                next_key = all_keys.popleft()
                while next_key != key:
                    # Hack to yield empty groups
                    next_group = _set()
                    next_tup = (next_key, next_group)
                    next_key = all_keys.popleft()
                    yield next_tup
                group = _set([g[1] for g in grouper])
                tup = (key, group)
                yield tup

            # any remaining groups are empty
            for key in all_keys:
                group = _set()
                tup = (key, group)
                yield tup

        else:
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
                        group = _set()
                    else:
                        group = _set(group)
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
                result = proxy.session.execute(expr)
                for row in result.fetchall():
                    key = row[0]
                    group = json.loads(row[1])
                    if group[0] is None:
                        group = _set()
                    else:
                        group = _set(group)
                    tup = (key, group)
                    yield tup

    def values(proxy):
        # Not sure if there if iterating over just the valuse can be more
        # efficient than iterating over the items and discarding the values.
        for key, val in proxy.items():
            yield val
        # _set = set if proxy.group_order_attr is None else ub.oset
        # if proxy.ALCHEMY_MODE:
        # else:
        #     parent_table = proxy.parent_keyattr.class_.__tablename__
        #     table = proxy.keyattr.class_.__tablename__
        #     parent_keycol = parent_table + '.' + proxy.parent_keyattr.name
        #     keycol = table + '.' + proxy.keyattr.name
        #     valcol = table + '.' + proxy.valattr.name
        #     expr = (
        #         'SELECT {parent_keycol}, json_group_array({valcol}) '
        #         'FROM {parent_table} '
        #         'LEFT OUTER JOIN {table} ON {keycol} = {parent_keycol} '
        #         'GROUP BY {parent_keycol} ORDER BY {parent_keycol}').format(
        #             parent_table=parent_table,
        #             table=table,
        #             parent_keycol=parent_keycol,
        #             keycol=keycol,
        #             valcol=valcol,
        #         )
        #     # print(expr)
        #     result = proxy.session.execute(expr)
        #     for row in result.fetchall():
        #         group = json.loads(row[1])
        #         if group[0] is None:
        #             group = _set()
        #         else:
        #             group = _set(group)
        #         yield group


class CocoSqlIndex(object):
    """
    Simulates the dictionary provided by :class:`kwcoco.coco_dataset.CocoIndex`
    """
    def __init__(index):
        index.anns = None
        index.imgs = None
        index.videos = None
        index.cats = None
        index.file_name_to_img = None

        index.name_to_cat = None
        index.name_to_img = None

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

        # These indexes are allowed to have null keys
        index.name_to_img = SqlDictProxy(session, Image, Image.name,
                                         ignore_null=True)
        index.file_name_to_img = SqlDictProxy(session, Image, Image.file_name,
                                              ignore_null=True)

        index.gid_to_aids = SqlIdGroupDictProxy(
            session, Annotation.id, Annotation.image_id, Image.id)
        index.cid_to_aids = SqlIdGroupDictProxy(
            session, Annotation.id, Annotation.category_id, Category.id)

        index.cid_to_gids = SqlIdGroupDictProxy(
            session, Annotation.image_id, Annotation.category_id, Category.id)

        # TODO:
        # index.trackid_to_aids = SqlIdGroupDictProxy(
        #     session, Annotation.id, Annotation.track_id)

        index.vidid_to_gids = SqlIdGroupDictProxy(
            session, Image.id, Image.video_id, Video.id,
            group_order_attr=Image.frame_index)

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


def _handle_sql_uri(uri):
    """
    Temporary function to deal with URI. Modern tools seem to use RFC 3968
    URIs, but sqlalchemy uses RFC 1738. Attempt to gracefully handle special
    cases. With a better understanding of the above specs, this function may be
    able to be written more eloquently.

    Ignore:
        _handle_sql_uri(':memory:')
        _handle_sql_uri('special:foobar')
        _handle_sql_uri('sqlite:///:memory:')
        _handle_sql_uri('/foo/bar')
        _handle_sql_uri('foo/bar')
    """
    import uritools
    uri_parsed = uritools.urisplit(uri)
    normalized = None
    local_path = None
    parent = None
    file_prefix = '/file:'

    scheme, authority, path, query, fragment = uri_parsed

    if scheme == 'special':
        pass
    elif scheme is None:
        # Add in the SQLite scheme
        scheme = 'sqlite'
        if path == ':memory:':
            path = '/:memory:'
        elif not path.startswith(file_prefix):
            path = file_prefix + path
        if authority is None:
            authority = ''
    elif scheme != 'sqlite':
        raise NotImplementedError(scheme)

    if path == '/:memory:':
        local_path = None
    elif path.startswith(file_prefix):
        local_path = path[len(file_prefix):]
    else:
        local_path = None

    if local_path is not None:
        parent = dirname(local_path)

    normalized = uritools.uricompose(scheme, authority, path, query, fragment)

    uri_info = {
        'uri': uri,
        'normalized': normalized,
        'local_path': local_path,
        'parsed': uri_parsed,
        'parent': parent,
    }
    return uri_info


class CocoSqlDatabase(AbstractCocoDataset,
                      MixinCocoAccessors, MixinCocoObjects, MixinCocoStats,
                      MixinCocoDraw, ub.NiceRepr):
    """
    Provides an API nearly identical to :class:`kwcoco.CocoDatabase`, but uses
    an SQL backend data store. This makes it robust to copy-on-write memory
    issues that arise when forking, as discussed in [1]_.

    Note:
        By default constructing an instance of the CocoSqlDatabase does not
        create a connection to the databse. Use the :func:`connect` method to
        open a connection.

    References:
        .. [1] https://github.com/pytorch/pytorch/issues/13246

    Example:
        >>> # xdoctest: +REQUIRES(module:sqlalchemy)
        >>> from kwcoco.coco_sql_dataset import *  # NOQA
        >>> sql_dset, dct_dset = demo()
        >>> dset1, dset2 = sql_dset, dct_dset
        >>> tag1, tag2 = 'dset1', 'dset2'
        >>> assert_dsets_allclose(sql_dset, dct_dset)
    """

    MEMORY_URI = 'sqlite:///:memory:'

    def __init__(self, uri=None, tag=None, img_root=None):
        if uri is None:
            uri = self.MEMORY_URI

        # TODO: make this more robust
        if img_root is None:
            img_root = _handle_sql_uri(uri)['parent']

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
        self.connect(readonly=True)

    def disconnect(self):
        """
        Drop references to any SQL or cache objects
        """
        self.session = None
        self.engine = None
        self.index = None

    def connect(self, readonly=False):
        """
        Connects this instance to the underlying database.

        References:
            # details on read only mode, some of these didnt seem to work
            https://github.com/sqlalchemy/sqlalchemy/blob/master/lib/sqlalchemy/dialects/sqlite/pysqlite.py#L71
            https://github.com/pudo/dataset/issues/136
            https://writeonly.wordpress.com/2009/07/16/simple-read-only-sqlalchemy-sessions/
        """
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy import create_engine
        # Create an engine that stores data at a specific uri location

        _uri_info = _handle_sql_uri(self.uri)
        uri = _uri_info['normalized']
        if readonly:
            uri = uri + '?mode=ro&uri=true'
        else:
            uri = uri + '?uri=true'
        self.engine = create_engine(uri)
        # table_names = self.engine.table_names()
        table_names = sqlalchemy.inspect(self.engine).get_table_names()
        if len(table_names) == 0:
            # Opened an empty database, need to create the tables
            # Create all tables in the engine.
            # This is equivalent to "Create Table" statements in raw SQL.
            # if readonly:
            #     raise AssertionError('must open existing table in readonly mode')
            CocoBase.metadata.create_all(self.engine)
        DBSession = sessionmaker(bind=self.engine)
        self.session = DBSession()

        self.session.execute('PRAGMA cache_size=-{}'.format(128 * 1000))

        self.index = CocoSqlIndex()
        self.index.build(self)
        return self

    @property
    def fpath(self):
        return self.uri

    def delete(self):
        fpath = self.uri.split('///file:')[-1]
        if self.uri != self.MEMORY_URI and exists(fpath):
            ub.delete(fpath)

    def populate_from(self, dset, verbose=1):
        """
        Copy the information in a :class:`CocoDataset` into this SQL database.

        Example:
            >>> # xdoctest: +REQUIRES(module:sqlalchemy)
            >>> from kwcoco.coco_sql_dataset import _benchmark_dset_readtime  # NOQA
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
        import itertools as it
        counter = it.count()
        session = self.session
        inspector = inspect(self.engine)
        # table_names = self.engine.table_names()
        table_names = sqlalchemy.inspect(self.engine).get_table_names()

        batch_size = 100000

        for key in table_names:
            colinfo = inspector.get_columns(key)
            colnames = {c['name'] for c in colinfo}
            # TODO: is there a better way to grab this information?
            cls = CocoBase.TBLNAME_TO_CLASS[key]
            for item in ub.ProgIter(dset.dataset.get(key, []),
                                    desc='Populate {}'.format(key),
                                    verbose=verbose):
                item_ = ub.dict_isect(item, colnames)
                # Everything else is a extra i.e. additional property
                item_['extra'] = ub.dict_diff(item, item_)
                if key == 'annotations':
                    # Need custom code to translate list-based properties
                    x, y, w, h = item_.get('bbox', [None, None, None, None])
                    item_['_bbox_x'] = x
                    item_['_bbox_y'] = y
                    item_['_bbox_w'] = w
                    item_['_bbox_h'] = h
                row = cls(**item_)
                session.add(row)

                if next(counter) % batch_size == 0:
                    # Commit data in batches
                    session.commit()
        if verbose:
            print('finish populate')
        session.commit()

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

    def _column_lookup(self, tablename, key, rowids, default=ub.NoParam,
                       keepid=False):
        """
        Convinience method to lookup only a single column of information

        Example:
            >>> # xdoctest: +REQUIRES(module:sqlalchemy)
            >>> from kwcoco.coco_sql_dataset import *  # NOQA
            >>> self, dset = demo(10)
            >>> tablename = 'annotations'
            >>> key = 'category_id'
            >>> rowids = list(self.anns.keys())[::3]
            >>> cids1 = self._column_lookup(tablename, key, rowids)
            >>> cids2 = self.annots(rowids).get(key)
            >>> cids3 = dset.annots(rowids).get(key)
            >>> assert cids3 == cids2 == cids1

        Ignore:
            import timerit
            ti = timerit.Timerit(10, bestof=3, verbose=2)

            for timer in ti.reset('time'):
                with timer:
                    self._column_lookup(tablename, key, rowids)

            for timer in ti.reset('time'):
                self.anns._cache.clear()
                with timer:
                    annots = self.annots(rowids)
                    annots.get(key)

            for timer in ti.reset('time'):
                self.anns._cache.clear()
                with timer:
                    anns = [self.anns[aid] for aid in rowids]
                    cids = [ann[key] for ann in anns]
        """
        # FIXME: Make this work for columns that need json decoding
        stmt = ub.paragraph(
            '''
            SELECT
                {tablename}.{key}
            FROM {tablename}
            WHERE {tablename}.id = :rowid
            ''').format(tablename=tablename, key=key)

        values = [
            # self.anns[aid][key]
            # if aid in self.anns._cache else
            self.session.execute(stmt, {'rowid': rowid}).fetchone()[0]
            for rowid in rowids
        ]
        if keepid:
            if default is ub.NoParam:
                attr_list = ub.dzip(rowids, values)
            else:
                raise NotImplementedError('cannot use default')
        else:
            if default is ub.NoParam:
                attr_list = values
            else:
                raise NotImplementedError('cannot use default')
        return attr_list

    def _all_rows_column_lookup(self, tablename, keys):
        """
        Convinience method to look up all rows from a table and only a few
        columns.

        Example:
            >>> # xdoctest: +REQUIRES(module:sqlalchemy)
            >>> from kwcoco.coco_sql_dataset import *  # NOQA
            >>> self, dset = demo(10)
            >>> tablename = 'annotations'
            >>> keys = ['id', 'category_id']
            >>> rows = self._all_rows_column_lookup(tablename, keys)
        """
        colnames_list = ['{}.{}'.format(tablename, key) for key in keys]
        colnames = ', '.join(colnames_list)
        stmt = ub.paragraph(
            '''
            SELECT
                {colnames}
            FROM {tablename} ORDER BY {tablename}.id
            ''').format(colnames=colnames, tablename=tablename)
        result = self.session.execute(stmt)
        rows = result.fetchall()
        return rows

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

    @property
    def bundle_dpath(self):
        return self.img_root

    @bundle_dpath.setter
    def bundle_dpath(self, value):
        self.img_root = value

    @property
    def data_fpath(self):
        """ data_fpath is an alias of fpath """
        return self.fpath

    @data_fpath.setter
    def data_fpath(self, value):
        self.fpath = value

    @classmethod
    def coerce(self, data):
        """
        Create an SQL CocoDataset from the input pointer.

        Example:
            import kwcoco
            dset = kwcoco.CocoDataset.demo('shapes8')
            data = dset.fpath
            self = CocoSqlDatabase.coerce(data)

            from kwcoco.coco_sql_dataset import CocoSqlDatabase
            import kwcoco
            dset = kwcoco.CocoDataset.coerce('spacenet7.kwcoco.json')

            self = CocoSqlDatabase.coerce(dset)

            from kwcoco.coco_sql_dataset import CocoSqlDatabase
            sql_dset = CocoSqlDatabase.coerce('spacenet7.kwcoco.json')

            # from kwcoco.coco_sql_dataset import CocoSqlDatabase
            import kwcoco
            sql_dset = kwcoco.CocoDataset.coerce('_spacenet7.kwcoco.view.v006.sqlite')

        """
        import kwcoco
        if isinstance(data, str):
            if data.endswith('.json'):
                dct_db_fpath = data
                self = cached_sql_coco_view(dct_db_fpath=dct_db_fpath)
            else:
                raise NotImplementedError
        elif isinstance(data, kwcoco.CocoDataset):
            self = cached_sql_coco_view(dset=data)
        else:
            raise NotImplementedError
        return self


def cached_sql_coco_view(dct_db_fpath=None, sql_db_fpath=None, dset=None,
                         force_rewrite=False):
    """
    Attempts to load a cached SQL-View dataset, only loading and converting the
    json dataset if necessary.
    """
    import os
    import kwcoco

    if dct_db_fpath is not None:
        if dset is not None:
            raise ValueError('must specify dset xor dct_db_fpath')
        bundle_dpath = dirname(dct_db_fpath)
        tag = None
    elif dset is not None:
        if dct_db_fpath is not None:
            raise ValueError('must specify dset xor dct_db_fpath')
        dct_db_fpath = dset.fpath
        bundle_dpath = dset.bundle_dpath
        tag = dset.tag
    else:
        raise AssertionError

    if sql_db_fpath is None:
        sql_db_fpath = ub.augpath(dct_db_fpath, prefix='_',
                                  ext='.view.v007.sqlite')

    self = CocoSqlDatabase(sql_db_fpath, img_root=bundle_dpath, tag=tag)

    needs_rewrite = True
    if not force_rewrite and exists(sql_db_fpath):
        needs_rewrite = (
            os.stat(dct_db_fpath).st_mtime > os.stat(sql_db_fpath).st_mtime
        )
    if needs_rewrite:
        # Write to the SQL instance

        # TODO: use a CacheStamp instead
        self.delete()
        self.connect()
        if dset is None:
            print('Loading json dataset for SQL conversion')
            dset = kwcoco.CocoDataset(dct_db_fpath)
        # Convert a coco file to an sql database
        print('Start SQL conversion')
        self.populate_from(dset, verbose=1)
    else:
        self.connect()
    return self


def ensure_sql_coco_view(dset, db_fpath=None, force_rewrite=False):
    """
    Create a cached on-disk SQL view of an on-disk COCO dataset.

    Note:
        This function is fragile. It depends on looking at file modified
        timestamps to determine if it needs to write the dataset.
    """
    return cached_sql_coco_view(dset=dset, sql_db_fpath=db_fpath,
                                force_rewrite=force_rewrite)


def demo(num=10):
    import kwcoco
    dset = kwcoco.CocoDataset.demo(
        'vidshapes', num_videos=1, num_frames=num, image_size=(64, 64))
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


def assert_dsets_allclose(dset1, dset2, tag1='dset1', tag2='dset2'):
    from kwcoco.util.util_json import indexable_allclose
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
            item1.update(item1.pop('extra', {}))
            item2.update(item2.pop('extra', {}))
            common1 = ub.dict_isect(item2, item1)
            common2 = ub.dict_isect(item1, item2)
            diff1 = ub.dict_diff(item1, common2)
            diff2 = ub.dict_diff(item2, common1)
            assert indexable_allclose(common2, common1)
            assert all(v is None for v in diff2.values())
            assert all(v is None for v in diff1.values())
    return True


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
    # table_names = self.engine.table_names()  # deprecated
    table_names = sqlalchemy.inspect(self.engine).get_table_names()
    for key in table_names:
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
        items1 = query.all()  # NOQA

    with timerit.Timer('naive'):
        items2 = [proxy[gid] for gid in chosen_gids]  # NOQA

    print(query.statement)
