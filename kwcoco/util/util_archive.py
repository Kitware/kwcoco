from os.path import dirname
from os.path import exists
from os.path import join
import ubelt as ub

import os
import tarfile
import zipfile
from os.path import relpath


class Archive(object):
    """
    Abstraction over zipfile and tarfile

    TODO:
        see if we can use one of these other tools instead

    SeeAlso:
        https://github.com/RKrahl/archive-tools
        https://pypi.org/project/arlib/

    Example:
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
    def __init__(self, fpath=None, mode='r', backend=None, file=None):

        self.fpath = fpath
        self.mode = mode
        self.file = file
        self.backend = backend

        if file is None:
            file, backend = self._open(fpath, mode)
            self.file = file
            self.backend = backend

    @classmethod
    def _open(cls, fpath, mode):
        if isinstance(fpath, os.PathLike):
            fpath = str(fpath)
        exist_flag = os.path.exists(fpath)
        if fpath.endswith('.tar.gz'):
            backend = tarfile
        elif fpath.endswith('.zip'):
            backend = zipfile
        else:
            if exist_flag and zipfile.is_zipfile(fpath):
                backend = zipfile
            elif exist_flag and tarfile.is_tarfile(fpath):
                backend = tarfile
            else:
                raise NotImplementedError('no-exist case')

        if backend is zipfile:
            if exist_flag and not zipfile.is_zipfile(fpath):
                raise Exception('corrupted zip?')
            file = zipfile.ZipFile(fpath, mode=mode)
        elif backend is tarfile:
            if exist_flag and not tarfile.is_tarfile(fpath):
                raise Exception('corrupted tar.gz?')
            file = tarfile.open(fpath, mode + ':gz')
        else:
            raise NotImplementedError
        return file, backend

    def __iter__(self):
        if self.backend is tarfile:
            return (mem.name for mem in self.file)
        elif self.backend is zipfile:
            # does zip have an iterable structure?
            return iter(self.file.namelist())

    @classmethod
    def coerce(cls, data):
        """
        Either open an archive file path or coerce an existing
        ZipFile or tarfile structure into this wrapper class
        """
        if isinstance(data, str):
            return cls(data)
        if isinstance(data, zipfile.ZipFile):
            fpath = data.fp.name
            return cls(fpath, file=data, backend=zipfile)
        else:
            pass

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

    # def move_internal(self, src, dst):
    #     """
    #     Move a file in the archive to a new location
    #     """
    #     # Seems to be tricky
    #     if self.backend is zipfile:
    #         raise
    #     else:
    #         raise NotImplementedError


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
