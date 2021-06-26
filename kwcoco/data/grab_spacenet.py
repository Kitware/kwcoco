"""


# Helper to extract the bit of utool that we need

# This is a "lightning" liberator tutorial.

# Problem: I have an old codebase that has a function that I want, but I don't
# want to bring over that entire old codebase.

# Solution: Liberate the functionality you need with "liberator"

# Example: I want to get the "unarchive_file" function from an old codebase so
# I can just import the codebase (once), create a liberator object, then add
# the function I care about to the liberator object. Liberator will do its best
# to extract that relevant bit of code.

# We can also remove dependencies on external modules using the "expand"
# function. In this example, we "expand" out utool itself, because we dont want
# to depend on it. The liberator object will then crawl through the code and
# try to bring in the source for the expanded dependencies. Note, that the
# expansion code works well, but not perfectly. It cant handle `import *` or
# other corner cases that can break the static analysis.


# Requires:
#     pip install liberator)

import liberator
import utool as ut

lib = liberator.Liberator()
lib.add_dynamic(ut.unarchive_file)
# lib.expand(['utool'])
print(lib.current_sourcecode())


# doesnt work
lib.expand(['utool.util_arg'])

"""
from os.path import dirname
from os.path import exists
from os.path import join
import ubelt as ub


def unarchive_file(archive_fpath, output_dpath='.', verbose=1, overwrite=True):
    import tarfile
    import zipfile
    if verbose:
        print('Unarchive archive_fpath = {!r} in {}'.format(archive_fpath, output_dpath))
    archive_file = None

    try:
        if tarfile.is_tarfile(archive_fpath):
            archive_file = tarfile.open(archive_fpath, 'r:gz')
            archive_namelist = [
                mem.path for mem in ub.ProgIter(
                    iter(archive_file), desc='enumerate members')
            ]
        elif zipfile.is_zipfile(archive_fpath):
            zip_file = zipfile.ZipFile(archive_fpath)
            if verbose:
                print('Enumerate members')
            archive_namelist = zip_file.namelist()
        else:
            raise NotImplementedError

        unarchived_paths = []
        for member in ub.ProgIter(archive_namelist, desc='extracting',
                                  verbose=verbose):
            fpath = join(output_dpath, member)
            unarchived_paths.append(fpath)
            if not overwrite and exists(fpath):
                continue
            ub.ensuredir(dirname(fpath))
            archive_file.extract(member, path=output_dpath)
    finally:
        if archive_file is not None:
            archive_file.close()

    return unarchived_paths


def grab_spacenet(data_dpath):
    """
    References:
        https://spacenet.ai/sn7-challenge/

    Requires:
        awscli

    Ignore:
        mkdir -p $HOME/.cache/kwcoco/data/spacenet/archives
        cd $HOME/.cache/kwcoco/data/spacenet/archives

        # Requires an AWS account
        export AWS_PROFILE=joncrall_at_kitware

        aws s3 cp s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz .
        aws s3 cp s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train_csvs.tar.gz .
        aws s3 cp s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_test_public.tar.gz .
    """
    import ubelt as ub
    import pathlib
    aws_exe = ub.find_exe('aws')

    dpath = ub.ensuredir((data_dpath, 'spacenet'))
    archive_dpath = pathlib.Path(ub.ensuredir((dpath, 'archives')))
    extract_dpath = pathlib.Path(ub.ensuredir((dpath, 'extracted')))

    if not aws_exe:
        raise Exception('requires aws exe')

    items = [
        # {
        #     'uri': 's3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz',
        #     'sha512': '5f810682825859951e55f6a3bf8e96eb6eb85864a90d75349',
        # },
        {
            'uri': 's3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train_csvs.tar.gz',
            'sha512': 'e4314ac129dd76e7984556c243b7b5c0c238085110ed7f7f619cb0',
        },
        {
            'uri': 's3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_test_public.tar.gz',
            'sha512': '0677a20f972cc463828bbff8d2fae08e17fdade3cf17ce213dc978',
        },
    ]

    for item in items:
        fname = pathlib.Path(item['uri']).name
        item['fpath'] = archive_dpath / fname
    print('archive_dpath = {!r}'.format(archive_dpath))

    for item in items:
        if not item['fpath'].exists():
            command = '{aws_exe} s3 cp {uri} {archive_dpath}'.format(
                aws_exe=aws_exe, uri=items['uri'], archive_dpath=archive_dpath)
            info = ub.cmd(command, verbose=3)
            assert info['ret'] == 0
            got_hash = ub.hash_file(item['fpath'], hasher='sha512')
            assert got_hash.startswith(item['sha512'])

    for item in ub.ProgIter(items, desc='extract spacenet', verbose=3):
        print('item = {}'.format(ub.repr2(item, nl=1)))
        archive_fpath = item['fpath']
        unarchive_file(archive_fpath, extract_dpath, overwrite=0, verbose=2)


def main():
    data_dpath = ub.ensure_app_cache_dir('kwcoco', 'data')
    grab_spacenet(data_dpath)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/kwcoco/data/grab_spacenet.py
    """
    main()
