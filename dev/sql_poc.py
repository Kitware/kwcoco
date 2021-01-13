from sqlalchemy import Column, Integer, String
from sqlalchemy import Numeric, ARRAY
from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
import sqlalchemy

Base = declarative_base()


def column_names(cls):
    for key, value in cls.__dict__.items():
        if isinstance(value, sqlalchemy.orm.attributes.InstrumentedAttribute):
            yield key


class Video(Base):
    __tablename__ = 'videos'
    id = Column(Integer, primary_key=True)
    name = Column(String(250), nullable=False)
    caption = Column(String(250), nullable=False)


class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True)
    file_name = Column(String(512), nullable=False)

    width = Column(Integer)
    height = Column(Integer)

    video_id = Column(Integer)
    frame_index = Column(Integer)


class Category(Base):
    __tablename__ = 'categories'
    id = Column(Integer, primary_key=True)
    name = Column(String(250))
    alias = Column(String(250))
    supercategory = Column(String(250))


class Annotation(Base):
    __tablename__ = 'annotations'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer)
    category_id = Column(Integer)

    track_id = Column(String)

    # segmentation = Column(BINARY)
    # keypoints = Column(BINARY)

    # bbox = Column(ARRAY(Numeric, dimensions=4))  # only postgres
    bbox_x = Column(Numeric)
    bbox_y = Column(Numeric)
    bbox_w = Column(Numeric)
    bbox_h = Column(Numeric)

    weight = Column(Numeric)


def main():
    from sqlalchemy.orm import sessionmaker
    import ubelt as ub
    import kwcoco
    from os.path import exists
    # Create an engine that stores data in the local directory's
    # sqlalchemy_example.db file.
    if exists('demo_coco_database.db'):
        ub.delete('demo_coco_database.db')
    engine = create_engine('sqlite:///demo_coco_database.db')
    # Create all tables in the engine. This is equivalent to "Create Table"
    # statements in raw SQL.
    Base.metadata.create_all(engine)

    tblname_to_class = {}
    for classname, cls in Base._decl_class_registry.items():
        if not classname.startswith('_'):
            tblname = cls.__tablename__
            tblname_to_class[tblname] = cls

    # Bind the engine to the metadata of the Base class so that the
    # declaratives can be accessed through a DBSession instance
    Base.metadata.bind = engine

    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    # Base.metadata.tables.values()
    # Clear everything
    # Base.metadata.drop_all(engine)
    # for tbl in reversed(meta.sorted_tables):
    #     engine.execute(tbl.delete())

    # Convert a coco file to an sql database
    dset = kwcoco.CocoDataset.demo('shapes8')

    root_keys = ['categories', 'videos', 'images', 'annotations']
    for key in root_keys:
        cls = tblname_to_class[key]
        colnames = {c.name for c in cls.__table__.columns}
        for item in dset.dataset.get(key, []):
            item_ = ub.dict_isect(item, colnames)
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

    # Read the sql tables into pandas
    import pandas as pd
    for key in root_keys:
        cls = tblname_to_class[key]
        print('\n----')
        print('key = {!r}'.format(key))
        print('cls = {!r}'.format(cls))
        print('cls.__tablename__ = {!r}'.format(cls.__tablename__))
        table_df = pd.read_sql_table(
            cls.__tablename__,
            con=engine
        )
        print(table_df)

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/dev/sql_poc.py
    """
    main()
