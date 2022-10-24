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

    xdoctest ~/code/kwcoco/dev/tutorial_postgres_alchemy.py use_postgresql_with_python
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
