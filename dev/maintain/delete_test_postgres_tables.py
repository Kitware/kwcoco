"""
Delete all kwcoco views in the global postgres datastore

NOTE: This does not delete the view cache stamps!
"""


def main():
    import ubelt as ub
    out = ub.cmd(r'sudo -u postgres psql -c "\list"')
    print(out.stdout)

    to_remove = []
    for line in out.stdout.split('\n')[2:]:
        if line.strip() and '|' in line:
            name = line.split('|')[0].strip()
            if name:
                if '.kwcoco.view' in name:
                    to_remove.append(name)

    print(f'to_remove = {ub.urepr(to_remove, nl=1)}')
    for name in ub.ProgIter(to_remove, desc='cleanup databases', verbose=3):
        ub.cmd(fr'sudo -u postgres psql -c "DROP DATABASE \"{name}\";"', verbose=3, system=1)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/dev/maintain/delete_test_postgres_tables.py
    """
    main()
