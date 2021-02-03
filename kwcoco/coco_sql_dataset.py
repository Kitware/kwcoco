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

Without a cache, SQL runs at 30HZ and takes 400MB for 10,000 images, and
for 100,000 images it gets 30Hz with 1.1GB. There is also a much larger startup
time. I'm not exactly sure what it is yet, but its probably some preprocessing
I'm doing.

However, once I scale up to 100,000 images I start seeing benefits.  The
in-memory dictionary interface chugs at 1.05HZ, and is taking more than 4GB
of memory at the time I killed the process (eta was over an hour). The SQL
backend ran at 45Hz and took about 3 minutes and used about 2.45GB of memory.

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

"""
from sqlalchemy.sql.schema import Column
from sqlalchemy.types import Float, Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import relationship
import sqlalchemy
import ubelt as ub
from os.path import exists

import sqlite3
import numpy as np
sqlite3.register_adapter(np.int64, int)
sqlite3.register_adapter(np.int32, int)


CocoBase = declarative_base()


def column_names(cls):
    for key, value in cls.__dict__.items():
        if isinstance(value, sqlalchemy.orm.attributes.InstrumentedAttribute):
            yield key


class Category(CocoBase):
    __tablename__ = 'categories'
    id = Column(Integer, primary_key=True, doc='unique internal id')
    name = Column(String(256), doc='unique external name or identifier', index=True, unique=True)
    alias = Column(JSON, doc='list of alter egos')
    supercategory = Column(String(256), doc='coarser category name')

    foreign = Column(JSON)


class KeypointCategory(CocoBase):
    __tablename__ = 'keypoint_categories'
    id = Column(Integer, primary_key=True, doc='unique internal id')
    name = Column(String(256), doc='unique external name or identifier', index=True, unique=True)
    alias = Column(JSON, doc='list of alter egos')
    supercategory = Column(String(256), doc='coarser category name')
    reflection_id = Column(Integer, doc='if augmentation reflects the image, change keypoint id to this')

    foreign = Column(JSON)


class Video(CocoBase):
    __tablename__ = 'videos'
    id = Column(Integer, primary_key=True, doc='unique internal id')
    name = Column(String(256), nullable=False, index=True, unique=True)
    caption = Column(String(256), nullable=True)
    foreign = Column(JSON)


class Image(CocoBase):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True, doc='unique internal id')
    file_name = Column(String(512), nullable=False, index=True, unique=True)

    width = Column(Integer)
    height = Column(Integer)

    video_id = Column(Integer, index=True, unique=False)
    timestamp = Column(Float)
    frame_index = Column(Integer)

    foreign = Column(JSON)

    channels = Column(JSON)
    auxiliary = Column(JSON)


class Annotation(CocoBase):
    __tablename__ = 'annotations'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, doc='', index=True, unique=False)
    category_id = Column(Integer, doc='', index=True, unique=False)
    track_id = Column(String, index=True, unique=False)

    segmentation = Column(JSON)
    keypoints = Column(JSON)

    foreign = Column(JSON)

    bbox = Column(JSON)
    bbox_x = Column(Float)
    bbox_y = Column(Float)
    bbox_w = Column(Float)
    bbox_h = Column(Float)
    weight = Column(Float)

    score = Column(Float)
    weight = Column(Float)
    prob = Column(JSON)

    iscrowd = Column(Integer)
    caption = Column(JSON)

    # @property
    # def bbox(self):
    #     return [self.bbox_x, self.bbox_y, self.bbox_w, self.bbox_h]


# Global book keeping
TBLNAME_TO_CLASS = {}
for classname, cls in CocoBase._decl_class_registry.items():
    if not classname.startswith('_'):
        tblname = cls.__tablename__
        TBLNAME_TO_CLASS[tblname] = cls


class CocoSqlRuntime:
    """
    Singleton class
    """
    def __init__(self):
        self



from scriptconfig.dict_like import DictLike  # NOQA


def orm_to_dict(obj):
    item = obj.__dict__.copy()
    item.pop('_sa_instance_state', None)
    return item


class SqlListProxy(ub.NiceRepr):
    """
    A view of an SQL table that behaves like a Python list

    Ignore:
        from kwcoco.coco_sql_dataset import *  # NOQA
        self, dset = demo()
        proxy = self.dataset['images']

        proxy.ALCHEMY_MODE = 1
        ti = timerit.Timerit(4, bestof=2, verbose=2)
        for timer in ti.reset('iter alc sql'):
            with timer:
                list(proxy)

        proxy.ALCHEMY_MODE = 0
        for timer in ti.reset('iter raw sql'):
            with timer:
                list(proxy)

        for timer in ti.reset('iter naive'):
            with timer:
                [proxy[idx] for idx in range(len(proxy))]

    """
    def __init__(proxy, session, cls):
        proxy.cls = cls
        proxy.session = session
        proxy._colnames = None
        proxy.ALCHEMY_MODE = 0

    def __len__(proxy):
        query = proxy.session.query(proxy.cls)
        return query.count()

    def __iter__(proxy):
        # TODO: our non-alchemy implementation doesn't handle json
        if 1 or proxy.ALCHEMY_MODE:
            query = proxy.session.query(proxy.cls).order_by(proxy.cls.id)
            for obj in query.yield_per(300):
                item = orm_to_dict(obj)
                yield item
        else:
            if proxy._colnames is None:
                from sqlalchemy import inspect
                inspector = inspect(proxy.session.get_bind())
                colinfo = inspector.get_columns(proxy.cls.__tablename__)
                proxy._colnames = [c['name'] for c in colinfo]
            colnames = proxy._colnames

            # Using raw SQL seems much faster
            result = proxy.session.execute(
                'SELECT * FROM {} ORDER BY id'.format(proxy.cls.__tablename__))

            for row in _yield_per(result):
                item = dict(zip(colnames, row))
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


class SqlDictProxy(ub.NiceRepr, DictLike):
    """
    Args:
        session (Session): the sqlalchemy session
        cls (Type): the declarative sqlalchemy table class
    """
    def __init__(proxy, session, cls, keyattr=None):
        proxy.cls = cls
        proxy.session = session
        proxy.keyattr = keyattr
        proxy._colnames = None

        # It seems like writing the raw sql ourselves is fater than
        # using the ORM in most cases.
        proxy.ALCHEMY_MODE = 0

        # ONLY DO THIS IN READONLY MODE
        # proxy._cache = {}
        proxy._cache = None

    def __len__(proxy):
        query = proxy.session.query(proxy.cls)
        return query.count()

    def __contains__(proxy, key):
        keyattr = proxy.keyattr
        if keyattr is None:
            keyattr = proxy.cls.id
        query = proxy.session.query(proxy.cls.id).filter(keyattr == key)
        flag = query.count() > 0
        return flag

    def __getitem__(proxy, key):
        if proxy._cache is not None:
            if key in proxy._cache:
                return proxy._cache[key]
        session = proxy.session
        cls = proxy.cls
        if proxy.keyattr is None:
            query = session.query(cls)
            obj = query.get(key)
            if obj is None:
                raise KeyError(key)
        else:
            # keyattr = proxy.keyattr
            # str(sqlalchemy.select([cls]).compile())
            # print(str(sqlalchemy.select([cls]).where(keyattr == 3).compile()))
            # stmt = sqlalchemy.select([cls]).where(keyattr == key)
            # session.execute(stmt)
            keyattr = proxy.keyattr
            query = session.query(cls)
            results = query.filter(keyattr == key).all()
            if len(results) == 0:
                raise KeyError(key)
            elif len(results) > 1:
                raise AssertionError('Should only have 1 result')
            obj = results[0]
        item = orm_to_dict(obj)
        if proxy._cache is not None:
            proxy._cache[key] = item
        return item

    def keys(proxy):
        """
        Ignore:
            from kwcoco.coco_sql_dataset import *  # NOQA
            import pytest
            self, dset = demo()

            proxy = self.imgs
            proxy[2]
            with pytest.raises(KeyError):
                proxy['efffdsf']
            with pytest.raises(KeyError):
                proxy[100000000000]
            assert 'efffdsf' not in proxy
            assert 2 in proxy
            assert 300000000 not in proxy

            proxy = self.index.name_to_cat

            proxy['eff']
            with pytest.raises(KeyError):
                proxy['efffdsf']
            with pytest.raises(KeyError):
                proxy[3]
            assert 'efffdsf' not in proxy
            assert 'eff' in proxy
            assert 3 not in proxy

            proxy = self.imgs
            key = list(proxy.keys())[-1]
            proxy[key]
            list(proxy.values())

            import timerit
            ti = timerit.Timerit(4, bestof=2, verbose=2)

            proxy.ALCHEMY_MODE = 1
            for timer in ti.reset('keys alc sql'):
                with timer:
                    list(proxy.keys())

            proxy.ALCHEMY_MODE = 0
            for timer in ti.reset('keys raw sql'):
                with timer:
                    list(proxy.keys())

            proxy.ALCHEMY_MODE = 1
            for timer in ti.reset('values alc sql'):
                with timer:
                    list(proxy.values())

            proxy.ALCHEMY_MODE = 0
            for timer in ti.reset('values raw sql'):
                with timer:
                    list(proxy.values())

            proxy.ALCHEMY_MODE = 1
            for timer in ti.reset('items alc sql'):
                with timer:
                    list(proxy.items())

            proxy.ALCHEMY_MODE = 0
            for timer in ti.reset('items raw sql'):
                with timer:
                    list(proxy.items())

            for timer in ti.reset('naive items'):
                with timer:
                    [proxy[key] for key in proxy.keys()]

            print('ti.measures = {}'.format(ub.repr2(ti.measures, nl=2, align=':', precision=6)))

            # sub = proxy.session.query(Annotation.id).group_by(Annotation.image_id).subquery()
        """
        if proxy.ALCHEMY_MODE:
            if proxy.keyattr is None:
                query = proxy.session.query(proxy.cls.id)
            else:
                query = proxy.session.query(proxy.keyattr)

            for item in query.yield_per(300):
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

            for item in _yield_per(result):
                yield item[0]

    def itervalues(proxy):
        # TODO: our non-alchemy implementation doesn't handle json
        if 1 or proxy.ALCHEMY_MODE:
            query = proxy.session.query(proxy.cls).order_by(proxy.cls.id)
            for obj in query.yield_per(300):
                item = orm_to_dict(obj)
                yield item
        else:
            if proxy._colnames is None:
                from sqlalchemy import inspect
                inspector = inspect(proxy.session.get_bind())
                colinfo = inspector.get_columns(proxy.cls.__tablename__)
                proxy._colnames = [c['name'] for c in colinfo]
            colnames = proxy._colnames

            # Using raw SQL seems much faster
            result = proxy.session.execute(
                'SELECT * FROM {} ORDER BY id'.format(proxy.cls.__tablename__))
            for row in _yield_per(result):
                item = dict(zip(colnames, row))
                yield item

    def iteritems(proxy):
        # if proxy.ALCHEMY_MODE:
        #     return ((key, proxy[key]) for key in proxy.keys())
        # else:
        if proxy.keyattr is None:
            keyattr_name = 'id'
        else:
            keyattr_name = proxy.keyattr.name
        for value in proxy.itervalues():
            yield (value[keyattr_name], value)

    items = iteritems
    values = itervalues


def _yield_per(result, size=300):
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


class SqlIdGroupDictProxy(DictLike, ub.NiceRepr):
    def __init__(proxy, session, valattr, keyattr, parent_keyattr):
        proxy.valattr = valattr
        proxy.keyattr = keyattr
        proxy.session = session
        proxy.parent_keyattr = parent_keyattr
        proxy.ALCHEMY_MODE = 0
        # proxy._cache = {}
        proxy._cache = None

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
        if proxy._cache is not None:
            proxy._cache[key] = item
        return item

    def __contains__(proxy, key):
        query = proxy.session.query(
            proxy.parent_keyattr).filter(proxy.parent_keyattr == key)
        flag = query.count() > 0
        return flag

    def keys(proxy):
        """
        Ignore:
            from kwcoco.coco_sql_dataset import *  # NOQA
            self, dset = demo()

            proxy = self.index.gid_to_aids

            print(list(proxy.keys())[-4:])
            proxy[10000]
            proxy[9999]

            proxy.ALCHEMY_MODE = 1
            with ub.Timer('1'):
                print(list(proxy.values())[-4:])
            proxy.ALCHEMY_MODE = 0
            with ub.Timer('0'):
                print(list(proxy.values())[-4:])

            proxy.ALCHEMY_MODE = 1
            import timerit
            ti = timerit.Timerit(1, bestof=1, verbose=2)
            for timer in ti.reset('keys sql alchemy'):
                with timer:
                    list(proxy.keys())

            proxy.ALCHEMY_MODE = 0
            for timer in ti.reset('keys sql raw'):
                with timer:
                    list(proxy.keys())

            proxy.ALCHEMY_MODE = 1
            for timer in ti.reset('items sql alchemy'):
                with timer:
                    list(proxy.items())

            proxy.ALCHEMY_MODE = 0
            for timer in ti.reset('items sql raw'):
                with timer:
                    list(proxy.items())

            proxy.ALCHEMY_MODE = 1
            for timer in ti.reset('values sql alchemy'):
                with timer:
                    list(proxy.values())

            proxy.ALCHEMY_MODE = 0
            for timer in ti.reset('keys sql raw'):
                with timer:
                    list(proxy.values())

            sql_expr = 'EXPLAIN QUERY PLAN SELECT {} FROM {} WHERE {}={}'.format(
                proxy.valattr.name,
                proxy.keyattr.class_.__tablename__,
                proxy.keyattr.name,
                key
            )
            result = proxy.session.execute(sql_expr)
            result.fetchall()
            item = [row[0] for row in result.fetchall()]

        """
        if proxy.ALCHEMY_MODE:
            query = proxy.session.query(proxy.parent_keyattr)
            for item in query.yield_per(300):
                key = item[0]
                yield key
        else:
            result = proxy.session.execute(
                'SELECT {} FROM {}'.format(
                    proxy.parent_keyattr.name,
                    proxy.parent_keyattr.class_.__tablename__))
            for item in _yield_per(result):
                yield item[0]

    def iteritems(proxy):
        if proxy.ALCHEMY_MODE:
            # Not Implemented
            # return DictLike.iteritems(proxy)
            return super().iteritems()
        else:
            import json
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
                key = row[0]
                group = json.loads(row[1])
                if group[0] is None:
                    group = []
                tup = (key, group)
                yield tup

    def itervalues(proxy):
        if proxy.ALCHEMY_MODE:
            return super().itervalues()
        else:
            import json
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
                    group = []
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



from kwcoco.coco_dataset import (  # NOQA
    MixinCocoJSONAccessors, MixinCocoAccessors, MixinCocoAttrs, MixinCocoStats,
    MixinCocoDraw
)


class CocoSqlDatabase(MixinCocoJSONAccessors, MixinCocoAccessors,
                      MixinCocoAttrs, MixinCocoStats, MixinCocoDraw,
                      ub.NiceRepr):
    """
    Attempts to provide the CocoDatabase API but with an SQL backend
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
        print('\n\nRETURNING STATE FOR SQL DATABASE')
        return {
            'uri': self.uri,
            'img_root': self.img_root,
            'tag': self.tag,
        }

    def __setstate__(self, state):
        print('\n\nSETTING STATE FOR SQL DATABASE')
        self.__dict__.update(state)
        self.session = None
        self.engine = None
        # Make unpickled objects readonly
        self.index = CocoSqlIndex()
        self.connect(readonly=True)

    def connect(self, readonly=False):
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy import create_engine
        # Create an engine that stores data at a specific uri location

        uri = self.uri
        if readonly:
            # https://github.com/sqlalchemy/sqlalchemy/blob/master/lib/sqlalchemy/dialects/sqlite/pysqlite.py#L71
            uri = uri + '?mode=ro&uri=true'
        elif uri.startswith('sqlite:///file:'):
            uri = uri + '?uri=true'

        # if readonly:
        #     # https://github.com/pudo/dataset/issues/136
        #     connect_args = {'uri': True}
        # else:
        #     connect_args = {}
        # self.engine = create_engine(uri, connect_args=connect_args)
        print('\n\nMAKING SQLITE CONNECTION TO uri = {!r}'.format(uri))

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
        # if readonly:
        #     # https://writeonly.wordpress.com/2009/07/16/simple-read-only-sqlalchemy-sessions/
        #     def abort_ro(*args, **kwargs):
        #         pass
        #     self.session.flush = abort_ro

        self._build_index()

    def delete(self):
        fpath = self.uri.split('///file:')[-1]
        if self.uri != self.MEMORY_URI and exists(fpath):
            ub.delete(fpath)

    def populate_from(self, dset):
        from sqlalchemy import inspect
        session = self.session
        inspector = inspect(self.engine)
        for key in self.engine.table_names():
            colinfo = inspector.get_columns(key)
            colnames = {c['name'] for c in colinfo}
            # TODO: is there a better way to grab this information?
            cls = TBLNAME_TO_CLASS[key]
            for item in dset.dataset.get(key, []):
                item_ = ub.dict_isect(item, colnames)
                # Everything else is a foreign key
                item['foreign'] = ub.dict_diff(item, item_)
                if key == 'annotations':
                    # Need custom code to translate list-based properties
                    x, y, w, h = item['bbox']
                    item_['bbox_x'] = x
                    item_['bbox_y'] = y
                    item_['bbox_w'] = w
                    item_['bbox_h'] = h
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
        import pandas as pd
        table_df = pd.read_sql_table(table_name, con=self.engine)
        return table_df


def ensure_sql_coco_view(dset, db_fpath=None):
    """
    Create an SQL view of the COCO dataset
    """
    if db_fpath is None:
        db_fpath = ub.augpath(dset.fpath, prefix='.', ext='.sqlite')

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


def demo():
    import kwcoco
    dset = kwcoco.CocoDataset.demo(
        'vidshapes', num_videos=1, num_frames=1000, gsize=(64, 64))

    HACK = 1
    if HACK:
        gids = list(dset.imgs.keys())
        aids1 = dset.gid_to_aids[gids[-2]]
        aids2 = dset.gid_to_aids[gids[-4]]
        print('aids1 = {!r}'.format(aids1))
        print('aids2 = {!r}'.format(aids2))
        dset.remove_annotations(aids1 | aids2)
        dset.fpath = ub.augpath(dset.fpath, suffix='_mod', multidot=True)
        if not exists(dset.fpath):
            dset.dump(dset.fpath, newlines=True)

    self = ensure_sql_coco_view(dset)
    return self, dset


def devcheck():
    """
    from kwcoco.coco_sql_dataset import *  # NOQA
    """
    # self = ensure_sql_coco_view(dset, db_fpath=':memory:')
    self, dset = demo()

    import timerit

    with timerit.Timer('gid_to_aids'):
        self.index.gid_to_aids.to_dict()

    with timerit.Timer('cid_to_aids'):
        self.index.cid_to_aids.to_dict()

    with timerit.Timer('imgs'):
        self.index.imgs.to_dict()

    with timerit.Timer('cats'):
        self.index.cats.to_dict()

    with timerit.Timer('anns'):
        self.index.anns.to_dict()

    with timerit.Timer('vidid_to_gids'):
        self.index.vidid_to_gids.to_dict()

    # Read the sql tables into pandas
    for key in self.engine.table_names():
        print('\n----')
        print('key = {!r}'.format(key))
        table_df = self.raw_table(key)
        print(table_df)

    # Check the speed difference in data access

    ti = timerit.Timerit(4, bestof=2, verbose=2)

    for ref in [dset, self]:

        for timer in ti.reset('{} dataset iteration'.format(ref)):
            with timer:
                list(ref.dataset['annotations'])

        for timer in ti.reset('{} index ann iteration'.format(ref)):
            with timer:
                list(ref.index.anns.items())

        for timer in ti.reset('{} ann id lookup'.format(ref)):
            aids = list(ref.index.anns.keys())[0:10]
            with timer:
                for aid in aids:
                    ref.index.anns[aid]

    print('ti.rankings = {}'.format(ub.repr2(ti.rankings, nl=2, precision=8)))

    import pickle
    serialized = pickle.dumps(self)
    copy = pickle.loads(serialized)

    rw_copy = CocoSqlDatabase(self.uri, img_root=self.img_root, tag=self.tag)
    rw_copy.connect()

    ro_copy = CocoSqlDatabase(self.uri, img_root=self.img_root, tag=self.tag)
    ro_copy.connect(readonly=True)

    import ndsampler
    self.hashid = 'foobarjunk'
    sampler = ndsampler.CocoSampler(self, backend=None)

    regions = sampler.regions
    regions.isect_index
    regions.get_segmentations([1, 3])
    regions.get_positive(1)
    regions.get_negative()

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
