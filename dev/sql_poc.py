from sqlalchemy.sql.schema import Column
from sqlalchemy.types import Float, Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import relationship
import sqlalchemy
import ubelt as ub
from os.path import exists

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

    def __len__(proxy):
        query = proxy.session.query(proxy.cls)
        return query.count()

    def __getitem__(proxy, key):
        session = proxy.session
        cls = proxy.cls
        if proxy.keyattr is None:
            query = session.query(cls)
            obj = query.get(key)
        else:
            keyattr = proxy.keyattr
            query = session.query(cls)
            obj = query.filter(keyattr == key).all()[0]

        item = obj.__dict__.copy()
        item.pop('_sa_instance_state', None)
        return item

    def keys(proxy):
        if proxy.keyattr is None:
            query = proxy.session.query(proxy.cls.id)
        else:
            query = proxy.session.query(proxy.keyattr)
        for item in query.yield_per(10):
            key = item[0]
            yield key


class SqlIdGroupDictProxy(ub.NiceRepr, DictLike):
    def __init__(proxy, session, valattr, keyattr):
        proxy.valattr = valattr
        proxy.keyattr = keyattr
        proxy.session = session

    def __len__(proxy):
        query = proxy.session.query(proxy.keyattr).distinct()
        return query.count()

    def __getitem__(proxy, key):
        session = proxy.session
        keyattr = proxy.keyattr
        valattr = proxy.valattr
        query = session.query(valattr).filter(keyattr == key)
        item = [row[0] for row in query.all()]
        return item

    def keys(proxy):
        query = proxy.session.query(proxy.keyattr).distinct()
        for item in query.yield_per(10):
            key = item[0]
            yield key


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
        index.name_to_cat = None

    def build(index, parent):
        session = parent.session
        index.anns = SqlDictProxy(session, Annotation)
        index.imgs = SqlDictProxy(session, Image)
        index.cats = SqlDictProxy(session, Category)
        index.videos = SqlDictProxy(session, Video)
        index.name_to_cat = SqlDictProxy(session, Category, Category.name)
        index.file_name_to_img = SqlDictProxy(session, Image, Image.file_name)

        index.gid_to_aids = SqlIdGroupDictProxy(session, Annotation.id, Annotation.image_id)
        index.cid_to_aids = SqlIdGroupDictProxy(session, Annotation.id, Annotation.category_id)
        index.vidid_to_gids = SqlIdGroupDictProxy(session, Image.id, Image.video_id)


class CocoSqlDatabase(object):
    """
    Attempts to provide the CocoDatabase API but with an SQL backend
    """

    MEMORY_URI = 'sqlite:///:memory:'

    def __init__(self, uri=None):
        if uri is None:
            uri = self.MEMORY_URI
        self.uri = uri
        self.session = None
        self.engine = None
        self.index = CocoSqlIndex()

    def __getstate__(self):
        return {
            'uri': self.uri,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.session = None
        self.engine = None
        self.index = CocoSqlIndex()
        self.connect()

    def connect(self):
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy import create_engine
        # Create an engine that stores data at a specific uri location
        self.engine = create_engine(self.uri)
        if len(self.engine.table_names()) == 0:
            # Opened an empty database, need to create the tables
            # Create all tables in the engine.
            # This is equivalent to "Create Table" statements in raw SQL.
            CocoBase.metadata.create_all(self.engine)

        DBSession = sessionmaker(bind=self.engine)
        self.session = DBSession()
        self._build_index()

    def delete(self):
        fpath = self.uri.split('///')[-1]
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


def ensure_sql_coco_view(dset):
    """
    Create an SQL view of the COCO dataset
    """
    db_fpath = ub.augpath(dset.fpath, prefix='.', ext='.sqlite')
    db_uri = 'sqlite:///' + db_fpath
    # dpath = dirname(dset.fpath)

    self = CocoSqlDatabase(db_uri)

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


def main():
    """
    from sql_poc import CocoBase
    """
    import kwcoco
    dset = kwcoco.CocoDataset.demo('vidshapes8')
    self = ensure_sql_coco_view(dset)

    self.index.gid_to_aids.to_dict()
    self.index.cid_to_aids.to_dict()
    self.index.imgs.to_dict()
    self.index.cats.to_dict()
    self.index.anns.to_dict()

    self.index.vidid_to_gids.to_dict()

    # Read the sql tables into pandas
    for key in self.engine.table_names():
        print('\n----')
        print('key = {!r}'.format(key))
        table_df = self.raw_table(key)
        print(table_df)

    pool = ub.JobPool(mode='process', max_workers=2)

    job1 = pool.submit(test_sql_worker, self)
    job2 = pool.submit(test_sql_worker, self)

    job1.result()
    job2.result()


def test_sql_worker(self):
    print('self = {!r}'.format(self))
    print('self.anns = {!r}'.format(self.anns))
    print(self.anns[1])

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/dev/sql_poc.py
    """
    main()
