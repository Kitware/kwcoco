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

import os
import tarfile
import zipfile
from os.path import relpath, dirname

class Archive(object):
    """
    Abstraction over zipfile and tarfile

    Example:
        >>> from kwcoco.data.grab_spacenet import *  # NOQA
        >>> from os.path import join
        >>> dpath = ub.ensure_app_cache_dir('ubelt', 'tests', 'archive')
        >>> ub.delete(dpath)
        >>> dpath = ub.ensure_app_cache_dir(dpath)
        >>> import pathlib
        >>> dpath = pathlib.Path(dpath)
        >>> #
        >>> #
        >>> mode = 'w'
        >>> self1 = Archive(str(dpath / 'demo.zip'), mode=mode)
        >>> self2 = Archive(str(dpath / 'demo.tar.gz'), mode=mode)
        >>> #
        >>> open(dpath / 'data_1only.txt', 'w').write('bazbzzz')
        >>> open(dpath / 'data_2only.txt', 'w').write('buzzz')
        >>> open(dpath / 'data_both.txt', 'w').write('foobar')
        >>> #
        >>> self1.add(dpath / 'data_both.txt')
        >>> self1.add(dpath / 'data_1only.txt')
        >>> #
        >>> self2.add(dpath / 'data_both.txt')
        >>> self2.add(dpath / 'data_2only.txt')
        >>> #
        >>> self1.close()
        >>> self2.close()
        >>> #
        >>> self1 = Archive(str(dpath / 'demo.zip'), mode='r')
        >>> self2 = Archive(str(dpath / 'demo.tar.gz'), mode='r')
        >>> #
        >>> extract_dpath = ub.ensuredir(str(dpath / 'extracted'))
        >>> extracted1 = self1.extractall(extract_dpath)
        >>> extracted2 = self2.extractall(extract_dpath)
        >>> for fpath in extracted2:
        >>>     print(open(fpath, 'r').read())
        >>> for fpath in extracted1:
        >>>     print(open(fpath, 'r').read())
    """
    def __init__(self, fpath, mode='r'):

        self.fpath = fpath
        self.mode = mode
        self.file = None
        self.backend = None

        exist_flag = os.path.exists(fpath)
        if fpath.endswith('.tar.gz'):
            self.backend = tarfile
        elif fpath.endswith('.zip'):
            self.backend = zipfile
        else:
            raise NotImplementedError('no-exist case')

        if self.backend is zipfile:
            if exist_flag and not zipfile.is_zipfile(fpath):
                raise Exception('corrupted zip?')
            self.file = zipfile.ZipFile(fpath, mode=mode)
        elif self.backend is tarfile:
            if exist_flag and not tarfile.is_tarfile(fpath):
                raise Exception('corrupted tar.gz?')
            self.file = tarfile.open(fpath, mode + ':gz')
        else:
            raise NotImplementedError

    def __iter__(self):
        if self.backend is tarfile:
            return (mem.name for mem in self.file)
        elif self.backend is zipfile:
            # does zip have an iterable structure?
            return iter(self.file.namelist())

    def add(self, fpath, arcname=None):
        if arcname is None:
            arcname = relpath(fpath, dirname(self.fpath))
        if self.backend is tarfile:
            self.file.add(fpath, arcname)
        if self.backend is zipfile:
            self.file.write(fpath, arcname)

    def close(self):
        return self.file.close()

    def __enter__(self):
        self.__file__.__enter__()
        return self

    def __exit__(self, *args):
        self.__file__.__exit__(*args)

    def extractall(self, output_dpath='.', verbose=1, overwrite=True):
        if verbose:
            print('Enumerate members')
        archive_namelist = list(ub.ProgIter(iter(self), desc='enumerate members'))
        unarchived_paths = []
        for member in ub.ProgIter(archive_namelist, desc='extracting',
                                  verbose=verbose):
            fpath = join(output_dpath, member)
            unarchived_paths.append(fpath)
            if not overwrite and exists(fpath):
                continue
            ub.ensuredir(dirname(fpath))
            self.file.extract(member, path=output_dpath)
        return unarchived_paths


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
    import kwcoco

    dpath = ub.ensuredir((data_dpath, 'spacenet'))

    coco_fpath = os.path.join(dpath, 'spacenet7.kwcoco.json')
    archive_dpath = pathlib.Path(ub.ensuredir((dpath, 'archives')))
    extract_dpath = pathlib.Path(ub.ensuredir((dpath, 'extracted')))

    stamp = ub.CacheStamp('convert_spacenet', dpath=dpath, depends=['v001'])
    if stamp.expired():
        items = [
            {
                'uri': 's3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz',
                'sha512': '5f810682825859951e55f6a3bf8e96eb6eb85864a90d75349',
            },
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

        need_download_archive = 1
        if need_download_archive:
            aws_exe = ub.find_exe('aws')
            if not aws_exe:
                raise Exception('requires aws exe')

            for item in items:
                if not item['fpath'].exists():
                    command = '{aws_exe} s3 cp {uri} {archive_dpath}'.format(
                        aws_exe=aws_exe, uri=items['uri'], archive_dpath=archive_dpath)
                    info = ub.cmd(command, verbose=3)
                    assert info['ret'] == 0
                    got_hash = ub.hash_file(item['fpath'], hasher='sha512')
                    assert got_hash.startswith(item['sha512'])

        need_unarchive = 1
        if need_unarchive:
            for item in ub.ProgIter(items, desc='extract spacenet', verbose=3):
                print('item = {}'.format(ub.repr2(item, nl=1)))
                archive_fpath = item['fpath']
                unarchive_file(archive_fpath, extract_dpath, overwrite=0, verbose=2)

        print('convert spacenet')
        rooot_contents = list(extract_dpath.glob('*'))
        print('rooot_contents = {!r}'.format(rooot_contents))

        csv_fpath = extract_dpath / 'csvs/sn7_train_ground_truth_pix.csv'

        contents = list(extract_dpath.glob('csvs/sn7_train_ground_truth_pix.csv'))

        tile_dpaths = list(extract_dpath.glob('train/*'))

        coco_dset = kwcoco.CocoDataset()
        # FIXME: CocoDataset broke when I passed fpath in the constructor
        coco_dset.fpath = coco_fpath

        for tile_dpath in tile_dpaths:
            tile_name = tile_dpath.name
            # subdirs = list(tile_dpath.glob('*'))
            vidid = coco_dset.add_video(name=tile_name)

            image_gpaths = sorted(tile_dpath.glob('images/*'))
            # sorted(tile_dpath.glob('labels/*'))
            # sorted(tile_dpath.glob('images_masked/*'))
            # sorted(tile_dpath.glob('labels_match/*'))
            sorted(tile_dpath.glob('labels_match_piz/*'))
            # sorted(tile_dpath.glob('UDM_masks/*'))

            for frame_index, gpath in enumerate(image_gpaths):
                coco_dset.add_image(
                    file_name=str(gpath.relative_to(coco_dset.bundle_dpath)),
                    name=str(gpath.name),
                    video_id=vidid,
                    frame_index=frame_index,
                )

        # Postprocess images
        import parse
        import datetime
        s7_fname_fmt = parse.Parser('global_monthly_{year:d}_{month:d}_mosaic_{}')
        for gid, img in coco_dset.index.imgs.items():
            gname = img['name']
            nameinfo = s7_fname_fmt.parse(gname)
            timestamp = datetime.datetime(year=nameinfo['year'], month=nameinfo['month'], day=1)
            img['timestamp'] = timestamp.isoformat()
            # TODO (in postprocesing?):
            # warp_img_to_vid=...
            # auxiliary=...
            # height
            # width
            # channels

        print('coco_dset.fpath = {!r}'.format(coco_dset.fpath))
        print('coco_dset = {!r}'.format(coco_dset))

        # TODO: add annotations


        coco_dset.dump(coco_dset.fpath, newlines=True)
        stamp.renew()

    coco_dset = kwcoco.CocoDataset(coco_fpath)
    return coco_dset



# def convert_spacenet_csv(data_dpath):
#     pas


def main():
    data_dpath = ub.ensure_app_cache_dir('kwcoco', 'data')
    grab_spacenet(data_dpath)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/kwcoco/data/grab_spacenet.py
    """
    main()
