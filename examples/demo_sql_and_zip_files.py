"""
Demo how coco files can be written as json, zip, or with the sqlite/postgresql
backends.
"""


def main():
    import kwcoco
    import ubelt as ub
    # Make demo data
    coco_dset = kwcoco.CocoDataset.demo('vidshapes', num_videos=20, num_frames=20, image_size=(32, 32))

    json_path = ub.Path(coco_dset.fpath)
    zip_path = json_path.augment(ext='.zip')

    # Write to the compressed path
    coco_dset.dump(zip_path)

    # By reading a json file with CocoSqlDatabase it will cache it as a SQL
    # database. The first time will be slow because it actually reads the json
    # file, but the second time (as long as the json data hasn't changed) will be
    # fast.
    kwcoco.CocoSqlDatabase.coerce(json_path, backend='sqlite')
    kwcoco.CocoSqlDatabase.coerce(json_path, backend='postgresql')

    with ub.Timer(label='Read json file') as t1:
        kwcoco.CocoDataset(json_path)

    with ub.Timer(label='Read zip file') as t2:
        kwcoco.CocoDataset(zip_path)

    with ub.Timer(label='Read sqlite') as t3:
        sqlite_dset = kwcoco.CocoSqlDatabase.coerce(json_path, backend='sqlite')

    with ub.Timer(label='Read postgresql') as t4:
        kwcoco.CocoSqlDatabase.coerce(json_path, backend='postgresql')

    print('t1.elapsed = {}'.format(ub.urepr(t1.elapsed, nl=1)))
    print('t2.elapsed = {}'.format(ub.urepr(t2.elapsed, nl=1)))
    print('t3.elapsed = {}'.format(ub.urepr(t3.elapsed, nl=1)))
    print('t4.elapsed = {}'.format(ub.urepr(t4.elapsed, nl=1)))

    sqlite_bytes = ub.Path(sqlite_dset.fpath).stat().st_size
    zip_bytes = zip_path.stat().st_size
    json_bytes = json_path.stat().st_size
    print('zip_bytes  = {}'.format(ub.urepr(zip_bytes, nl=1)))
    print('json_bytes = {}'.format(ub.urepr(json_bytes, nl=1)))
    print('sql_bytes  = {}'.format(ub.urepr(sqlite_bytes, nl=1)))


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/kwcoco/examples/demo_sql_and_zip_files.py
    """
    main()
