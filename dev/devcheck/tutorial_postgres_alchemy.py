"""
Unlike other tutorials, this one is going to start with how to install the damn
things.


We need to configure a global service, which means we need sudo 😞. These
instructions are for Ubuntu. We assume you have a python virtual environment
already setup and enabled.



# Install Python tools
pip install psycopg2-binary sqlalchemy_utils sqlalchemy



"""


def install_postgres_instructions():
    import ubelt as ub
    # import ustd as u
    parts = {}
    parts['install'] = ub.codeblock(
        '''
        # Install PostgreSQL
        sudo apt install postgresql postgresql-contrib -y
        ''')

    parts['start_service'] = ub.codeblock(
        '''
        # Ensure it is started as a service
        sudo systemctl start postgresql.service
        sudo systemctl status postgresql.service
        ''')

    # FIXME: I'm not sure what best practice is here.
    parts['create_users'] = ub.codeblock(
        '''
        # Create roles and users
        sudo -u postgres createuser --superuser --no-password Admin
        sudo -u postgres createuser --role=Admin admin
        sudo -u postgres psql -c "ALTER USER admin WITH PASSWORD 'admin';"
        sudo -u postgres psql -c "ALTER USER admin WITH CREATEDB;"
        sudo -u postgres psql -c "ALTER USER admin WITH LOGIN;"
        sudo -u postgres psql -c "ALTER USER admin WITH SUPERUSER;"
        sudo -u postgres psql -c "ALTER USER admin WITH REPLICATION;"
        sudo -u postgres psql -c "ALTER USER admin WITH BYPASSRLS;"

        sudo -u postgres createuser --no-password --replication Maintainer
        sudo -u postgres psql -c "ALTER USER Maintainer WITH CREATEDB;"
        sudo -u postgres psql -c "ALTER USER Maintainer WITH SUPERUSER;"
        sudo -u postgres psql -c "ALTER USER Maintainer WITH REPLICATION;"
        sudo -u postgres psql -c "ALTER USER Maintainer WITH BYPASSRLS;"

        # This will be the kwcoco default user
        sudo -u postgres createuser --role=Maintainer kwcoco
        sudo -u postgres psql -c "ALTER USER kwcoco WITH PASSWORD 'kwcoco_pw';"
        sudo -u postgres psql -c "ALTER USER kwcoco WITH CREATEDB;"
        ''')

    # Move the database to some other directory
    # TODO
    # https://cloud.google.com/community/tutorials/setting-up-postgres-data-disk
    # parts['move_datadir'] = ub.codeblock(
    #     '''
    #     sudo -u postgres psql -c "SHOW config_file;"
    #     sudo -u postgres psql -c "SHOW data_directory;"
    #     sudo
    #     ''')


def use_postgresql_with_python():
    """
    Checks to see if postgres is setup as expected

    # Install Python tools
    pip install psycopg2-binary sqlalchemy_utils sqlalchemy

    xdoctest ~/code/kwcoco/dev/devcheck/tutorial_postgres_alchemy.py use_postgresql_with_python
    """
    from sqlalchemy import create_engine
    from sqlalchemy_utils import database_exists, create_database
    engine = create_engine("postgresql+psycopg2://kwcoco:kwcoco_pw@localhost:5432/mydb")
    print(f'engine={engine}')
    did_exist = database_exists(engine.url)
    print(f'did_exist={did_exist}')
    if not did_exist:
        print('creating')
        create_database(engine.url)
    does_exist = database_exists(engine.url)
    print(f'does_exist={does_exist}')


def simple_declarative_schema():
    from sqlalchemy import create_engine
    from sqlalchemy import inspect
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.sql.schema import Column
    from sqlalchemy.types import Integer, JSON
    from sqlalchemy_utils import database_exists, create_database

    CustomBase = declarative_base()

    from sqlalchemy.sql.schema import Index  # NOQA
    from sqlalchemy.dialects.postgresql import JSONB

    class User(CustomBase):
        __tablename__ = 'users'
        id = Column(Integer, primary_key=True, doc='unique internal id')
        name = Column(JSON)
        loose_identifer = Column(JSON, index=True, unique=False)
        # loose_identifer = Column(JSONB, index=True, unique=False)
        # loose_identifer = Column(JSON, unique=False)
        # __table_args__ =  (
        #     # https://stackoverflow.com/questions/30885846/how-to-create-jsonb-index-using-gin-on-sqlalchemy
        #     Index(
        #         "ix_users_loose_identifer", loose_identifer,
        #         postgresql_using="gist",
        #     ),
        # )

    from sqlalchemy.schema import CreateTable
    from sqlalchemy.schema import CreateIndex
    print(CreateTable(User.__table__))
    # print(CreateIndex(User))

    # uri = 'sqlite:///test_sqlite_v7.sqlite'
    uri = 'postgresql+psycopg2://admin:admin@localhost:5432/test_postgresql_v10.postgres'

    engine = create_engine(uri)
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    if 'postgresql' in uri:
        if not database_exists(uri):
            create_database(uri)

    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if len(table_names) == 0:
        CustomBase.metadata.create_all(engine)

    user_infos = [
        {'name': 'user1', 'loose_identifer': "AA" },
        {'name': 'user2', 'loose_identifer': "33" },
        {'name': 'user3', 'loose_identifer': 33 },
        {'name': 'user4', 'loose_identifer': 33 },
        {'name': 'user5', 'loose_identifer': "AA" },
        {'name': 'user6', 'loose_identifer': None},
        {'name': 'user7', 'loose_identifer': [1, 'weird']},
    ]
    for row in user_infos:
        user = User(**row)
        session.add(user)

    session.commit()

    import pandas as pd
    import json
    table_df = pd.read_sql_table('users', con=engine)
    table_df['loose_identifer'] = table_df['loose_identifer'].apply(repr)
    print(table_df)

    query = session.query(User.name, User.loose_identifer).filter(User.loose_identifer == json.dumps(33))
    results = list(query.all())
    print(f'results={results}')

    query = session.query(User.name, User.loose_identifer).filter(User.loose_identifer == json.dumps('33'))
    results = list(query.all())
    print(f'results={results}')
