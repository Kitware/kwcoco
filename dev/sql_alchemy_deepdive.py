r"""
Notes about a deep dive into sqlalchemy picking appart some internals


Initialize the Global PostgreSQL database and server:

    # https://www.digitalocean.com/community/tutorials/how-to-install-and-use-postgresql-on-ubuntu-20-04

    # https://www.liquidweb.com/kb/how-to-install-sqlalchemy/

    pip install psycopg2-binary
    pip install sqlalchemy_utils
    pip install sqlalchemy
    sudo apt install postgresql postgresql-contrib
    sudo systemctl start postgresql.service
    sudo systemctl start postgresql.service
    sudo systemctl status postgresql.service

    # TODO: all these commands need to run as the postgress user
    # sudo -i -u postgres
    # psql
    # psql -U postgres
    # sudo -u postgres dropuser joncrall
    # sudo -u postgres psql -c "\du"
    # psql
    # sudo -u postgres createdb $USERNAME

    sudo -u postgres createuser --interactive

    sudo -u postgres createuser --superuser --no-password Admin
    sudo -u postgres createuser  --no-password --no-superuser --createdb --createrole --login --replication Maintainer
    sudo -u postgres createuser  --no-password --no-superuser --login Developer
    sudo -u postgres createuser --role=Maintainer $USERNAME

    sudo -u postgres createuser --role=Admin admin
    sudo -u postgres psql -c "ALTER USER admin WITH PASSWORD 'admin';"
    sudo -u postgres psql -c "ALTER USER admin WITH CREATEDB;"
    sudo -u postgres psql -c "ALTER USER admin WITH LOGIN;"
    sudo -u postgres psql -c "ALTER USER admin WITH SUPERUSER;"
    sudo -u postgres psql -c "ALTER USER admin WITH REPLICATION;"
    sudo -u postgres psql -c "ALTER USER admin WITH BYPASSRLS;"


    # This will be the kwcoco default user
    sudo -u postgres createuser --role=Maintainer kwcoco
    sudo -u postgres psql -c "ALTER USER kwcoco WITH PASSWORD 'kwcoco_pw';"
    sudo -u postgres psql -c "ALTER USER kwcoco WITH CREATEDB;"

    # https://dba.stackexchange.com/questions/37351/postgresql-exclude-using-error-data-type-integer-has-no-default-operator-class
    sudo -u postgres psql -c "CREATE EXTENSION btree_gist;"



    sudo -u postgres createuser --role=Developer kwcoco_rw
    sudo -u postgres createuser --role=Developer kwcoco_ro
    sudo -u postgres psql -c "ALTER USER kwcoco_rw WITH PASSWORD 'kwcoco_pw';"
    sudo -u postgres psql -c "ALTER USER kwcoco_ro WITH PASSWORD 'kwcoco_pw';"

    sudo -u postgres createdb kwcocodb
    python -c "from sqlalchemy import create_engine; create_engine('postgresql+psycopg2://kwcoco:kwcoco_pw@localhost:5432/kwcocodb').connect()"

    python -c "from sqlalchemy import create_engine; create_engine('postgresql+psycopg2://admin:admin@localhost:5432/joncrall').connect()"


Install
    pip install psycopg2-binary
    pip install sqlalchemy_utils


Ignore:
    from sqlalchemy import create_engine

    postgresql://scott:tiger@localhost/mydatabase

    from sqlalchemy import create_engine
    engine = create_engine('postgresql+psycopg2://admin:admin@localhost/joncrall')
    engine.connect()


References:
    https://stackoverflow.com/questions/9353822/connecting-postgresql-with-sqlalchemy
"""
from kwcoco.coco_sql_dataset import *  # NOQA


def testit():
    from sqlalchemy import create_engine
    from sqlalchemy_utils import database_exists, create_database
    engine = create_engine("postgresql+psycopg2://kwcoco:kwcoco_pw@localhost:5432/mydb")
    did_exist = database_exists(engine.url)
    if not did_exist:
        create_database(engine.url)


def values(proxy):
    """
    from kwcoco.coco_sql_dataset import *  # NOQA
    import pytest
    self, dset = demo()

    proxy = self.imgs

    """
    if proxy._colnames is None:
        from sqlalchemy import inspect
        inspector = inspect(proxy.session.get_bind())
        colinfo = inspector.get_columns(proxy.cls.__tablename__)

        if 0:
            casters = []
            dialect = proxy.session.bind.dialect
            for c in colinfo:
                t = c['type']
                caster = t.result_processor(dialect, t)
                # caster = t.bind_processor(dialect)
                if caster is None:
                    caster = ub.identity
                casters.append(caster)
            proxy._casters = casters

        proxy._colnames = [c['name'] for c in colinfo]
    colnames = proxy._colnames

    if 0:
        casters = proxy._casters

    if 0:
        tablename = 'annotations'
        result = proxy.session.execute("PRAGMA TABLE_INFO('" + tablename + "')")
        result.fetchall()
        # Using raw SQL seems much faster
        result = proxy.session.execute(
            'SELECT * FROM {} ORDER BY id'.format(proxy.cls.__tablename__))
        rows = result._fetchall_impl()
        row = rows[0]
        # result.process_rows(rows)

        for idx, (f, x) in enumerate(zip(proxy._casters, row)):
            print(repr(f(x)))
            pass

        _colnames = list(result.keys())

        query = proxy.session.query(proxy.cls).order_by(proxy.cls.id)
        context = query._compile_context()
        result = proxy.session.execute(
            'SELECT * FROM {} ORDER BY id'.format(proxy.cls.__tablename__))
        from sqlalchemy.orm import loading
        cursor = result
        context.runid = loading._new_runid()
        context.post_load_paths = {}
        process = []
        labels = []
        query_entity = query._entities[0]
        # adapter = query_entity._get_entity_clauses(query, context)

        adapter = None

        only_load_props = query._only_load_props
        refresh_state = context.refresh_state

        self = query_entity
        mapper = query_entity.mapper
        identity_class = mapper._identity_class
        props = mapper._prop_set
        path = self.path
        from sqlalchemy.orm.util import _none_set
        quick_populators = path.get(
            context.attributes, "memoized_setups", _none_set
        )
        for prop in props:
            if prop in quick_populators:
                # this is an inlined path just for column-based attributes.
                col = quick_populators[prop]
            else:
                x = prop.create_row_processor(
                    context, path, mapper, result, adapter, populators
                )

        _instance = loading._instance_processor(
            self.mapper,
            context,
            result,
            path,
            adapter,
            only_load_props=only_load_props,
            refresh_state=refresh_state,
            polymorphic_discriminator=self._polymorphic_discriminator,
        )


        for query_entity in query._entities:
            _instance, label_name = query_entity.row_processor(query, context, cursor)
            labels.append(label_name)
            process.append(_instance)
            # _instance(row)
            # print('x = {!r}'.format(x))
        # proc = query.bundle.create_row_processor(query, process, labels)

        list(loading.instances(query, result, context))

        assert colnames == _colnames

    # Using raw SQL seems much faster
    result = proxy.session.execute(
        'SELECT * FROM {} ORDER BY id'.format(proxy.cls.__tablename__))

    for row in _yield_per(result):
        # cast_row = [f(x) for f, x in zip(proxy._casters, row)]
        # item = dict(zip(colnames, cast_row))
        item = dict(zip(colnames, row))
        yield item
