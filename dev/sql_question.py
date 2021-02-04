"""
I've been able to figure out how to do this in raw SQLite. My question is how to do this in SQLAlchemy.

I have two tables: annotations:

```
   id  image_id              bbox
0   1         1  [13, 13, 28, 15]
1   2         2  [13, 13, 28, 15]
2   3         2  [18, 10, 25, 17]
3   4         4  [13, 10, 25, 17]
```

and images:

```
   id file_name
0   1  img1.jpg
1   2  img2.jpg
2   3  img3.jpg
3   4  img4.jpg
4   5  img5.jpg
```
"""


from sqlalchemy.sql.schema import Column
from sqlalchemy.types import Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy

Base = declarative_base()


class Annotation(Base):
    __tablename__ = 'annotations'
    id = Column(Integer, primary_key=True, doc='unique internal id')
    image_id = Column(Integer, doc='', index=True, unique=False)
    bbox = Column(JSON)


class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True, doc='unique internal id')
    file_name = Column(String(512), nullable=False, index=True, unique=True)


def main():
    from sqlalchemy.orm import sessionmaker
    import ubelt as ub
    from sqlalchemy import create_engine

    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    session.add(Annotation(id=1, image_id=1, bbox=[13, 13, 28, 15]))
    session.add(Annotation(id=2, image_id=2, bbox=[13, 13, 28, 15]))
    session.add(Annotation(id=3, image_id=2, bbox=[18, 10, 25, 17]))
    session.add(Annotation(id=4, image_id=4, bbox=[13, 10, 25, 17]))

    session.add(Image(id=1, file_name='img1.jpg'))
    session.add(Image(id=2, file_name='img2.jpg'))
    session.add(Image(id=3, file_name='img3.jpg'))
    session.add(Image(id=4, file_name='img4.jpg'))
    session.add(Image(id=5, file_name='img5.jpg'))
    session.commit()

    import pandas as pd
    print(pd.read_sql_table('annotations', con=engine))
    print(pd.read_sql_table('images', con=engine))

    # Args:
    parent_keyattr = Image.id
    keyattr = Annotation.image_id
    valattr = Annotation.id

    ###
    # Raw SQLite: Does exactly what I want
    ###
    parent_table = parent_keyattr.class_.__tablename__
    table = keyattr.class_.__tablename__
    parent_keycol = parent_table + '.' + parent_keyattr.name
    keycol = table + '.' + keyattr.name
    valcol = table + '.' + valattr.name
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
    import json
    result = session.execute(expr)
    final = []
    for row in result.fetchall():
        key = row[0]
        group = json.loads(row[1])
        if group[0] is None:
            group = set()
        else:
            group = set(group)
        tup = (key, group)
        final.append(tup)

    print('final = {}'.format(ub.repr2(final, nl=1)))

    """
    This returns:

    ```
    final = [
        (1, {1}),
        (2, {2, 3}),
        (3, {}),
        (4, {4}),
        (5, {}),
    ]
    ```

    The images 3 and 5 without annotations are correctly accounted for.

    But I'm having a very hard time figuring out how to do the equivalent
    behavior with SQLAlchemy. I've tried several variation:
    """

    # SQLite Alchemy
    ###
    # VERSION 1: Does not correctly return null for images without annotations
    ###

    grouped_vals = sqlalchemy.func.json_group_array(valattr, type_=JSON)
    parent_table = parent_keyattr.class_.__table__
    table = keyattr.class_.__table__
    # TODO: This might have to be different for PostgreSQL
    grouped_vals = sqlalchemy.func.json_group_array(valattr, type_=JSON)
    query = (
        session.query(keyattr, grouped_vals)
        .outerjoin(parent_table, parent_keyattr == keyattr)
        .group_by(parent_keyattr)
        .order_by(parent_keyattr)
    )
    print(query.statement)

    final = []
    for row in query.all():
        key = row[0]
        group = row[1]
        if group[0] is None:
            group = set()
        else:
            group = set(group)
        tup = (key, group)
        final.append(tup)
    print('final = {}'.format(ub.repr2(final, nl=1)))

    """
    This returns:

    ```
    final = [
        (1, {1}),
        (2, {2, 3}),
        (4, {4}),
    ]
    ```

    which is missing the values for images 3 and 5. This is because I queried
    on keyattr instead of parent_keyattr.


    But if I try to use parent_keyattr I get an error when I try the outer join
    """

    query = (
        session.query(parent_keyattr, grouped_vals)
        .outerjoin(parent_table, parent_keyattr == keyattr)
    )

    """

    Looking at:

    `print(session.query(parent_keyattr, grouped_vals))`

    this makes sense because I get:

    ```
    SELECT images.id AS images_id, json_group_array(annotations.id) AS json_group_array_1
    FROM images, annotations
    ```

    I'm not sure if there is a way to force `grouped_vals` to think its FROM
    statement targets the annotations table. I've tried several variants but have
    had little luck sofar.


    The best luck I've had was by wrapping `grouped_vals` in a `str`. Which does
    let me get exactly what I want, but I lose the nice `type_=JSON` that
    automatically took care of converting the result to json for me.

    """

    query = (
        session.query(parent_keyattr, str(grouped_vals))
        .outerjoin(table, parent_keyattr == keyattr)
        .group_by(parent_keyattr)
        .order_by(parent_keyattr)
    )
    print(query.statement)

    final = []
    for row in query.all():
        key = row[0]
        group = json.loads(row[1])
        if group[0] is None:
            group = set()
        else:
            group = set(group)
        tup = (key, group)
        final.append(tup)
    print('final = {}'.format(ub.repr2(final, nl=1)))

    """

    I would like to know if there is a way to force `grouped_vals` to target the
    "images" table instead of "annotations", so I don't have to wrap it in a
    string, and I don't have to manually convert to JSON.

    """

    print(session.query(parent_keyattr.expression, grouped_vals).select_from(parent_table))

    subq = session.query(parent_keyattr.expression, grouped_vals).subquery()
    y = subq.outerjoin(table, parent_keyattr == keyattr).select()
    z = y.group_by(parent_keyattr).order_by(parent_keyattr)
    print(z)

    z.all()
    print(subq)
    print(subq.outerjoin(table, parent_keyattr == keyattr))

    x = session.query(parent_keyattr.expression, grouped_vals.select().select_from(parent_table)).subquery()
    x.outerjoin(table, parent_keyattr == keyattr)
    .group_by(parent_keyattr).order_by(parent_keyattr)
    print(x)

    z = session.query(parent_keyattr).outerjoin(table, parent_keyattr == keyattr)
    z = session.query(parent_keyattr).outerjoin(table, parent_keyattr == keyattr)
    z.all()

    query = (
        session.query(parent_keyattr, str(grouped_vals))
        .outerjoin(table, parent_keyattr == keyattr)
        .group_by(parent_keyattr)
        .order_by(parent_keyattr)
    )
    print(query.statement)

    ojoin = parent_table.outerjoin(table, parent_keyattr == keyattr)
    z = ojoin.select()
    sub = session.query(z).subquery()
    print(sub)

    print(session.query(z))
    .all()

    z = ojoin.select()
    session.execute(z).fetchall()

    sel = sqlalchemy.select([parent_keyattr, grouped_vals]).select_from(
    )
    print(sel)
    session.execute(sel)

    """
    WANT:
        SELECT images.id, json_group_array(annotations.id)
        FROM images
        LEFT OUTER JOIN annotations
        ON annotations.image_id = images.id
        GROUP BY images.id ORDER BY images.id

    GOT:
        SELECT images.id, json_group_array(annotations.id) AS json_group_array_1
        FROM images, annotations, images
        LEFT OUTER JOIN annotations
        ON images.id = annotations.image_id
        GROUP BY images.id ORDER BY images.id

    """
