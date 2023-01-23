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
        >>> from kwcoco.util.util_archive import Archive
        >>> import ubelt as ub
        >>> dpath = ub.Path.appdir('kwcoco', 'tests', 'util', 'archive')
        >>> dpath.delete().ensuredir()
        >>> # Test write mode
        >>> mode = 'w'
        >>> arc_zip = Archive(str(dpath / 'demo.zip'), mode=mode)
        >>> arc_tar = Archive(str(dpath / 'demo.tar.gz'), mode=mode)
        >>> open(dpath / 'data_1only.txt', 'w').write('bazbzzz')
        >>> open(dpath / 'data_2only.txt', 'w').write('buzzz')
        >>> open(dpath / 'data_both.txt', 'w').write('foobar')
        >>> #
        >>> arc_zip.add(dpath / 'data_both.txt')
        >>> arc_zip.add(dpath / 'data_1only.txt')
        >>> #
        >>> arc_tar.add(dpath / 'data_both.txt')
        >>> arc_tar.add(dpath / 'data_2only.txt')
        >>> #
        >>> arc_zip.close()
        >>> arc_tar.close()
        >>> #
        >>> # Test read mode
        >>> arc_zip = Archive(str(dpath / 'demo.zip'), mode='r')
        >>> arc_tar = Archive(str(dpath / 'demo.tar.gz'), mode='r')
        >>> # Test names
        >>> name = 'data_both.txt'
        >>> assert name in arc_zip.names()
        >>> assert name in arc_tar.names()
        >>> # Test read
        >>> assert arc_zip.read(name, mode='r') == 'foobar'
        >>> assert arc_tar.read(name, mode='r') == 'foobar'
        >>> #
        >>> # Test extractall
        >>> extract_dpath = ub.ensuredir(str(dpath / 'extracted'))
        >>> extracted1 = arc_zip.extractall(extract_dpath)
        >>> extracted2 = arc_tar.extractall(extract_dpath)
        >>> for fpath in extracted2:
        >>>     print(open(fpath, 'r').read())
        >>> for fpath in extracted1:
        >>>     print(open(fpath, 'r').read())
    """
    _available_backends = {
        'tarfile': tarfile,
        'zipfile': zipfile,
    }

    def __init__(self, fpath=None, mode='r', backend=None, file=None):
        """
        Args:
            fpath (str | None): path to open

            mode (str): either r or w

            backend (str | ModuleType | None):
                either tarfile, zipfile string or module.

            file (tarfile.TarFile | zipfile.ZipFile | None):
                the open backend file if it already exists.
                If not set, than fpath will open it.
        """
        self.fpath = fpath
        self.mode = mode
        self.file = file
        self.backend = self._available_backends.get(backend, backend)

        if file is None:
            file, backend = self._open(fpath, mode, backend)
            self.file = file
            self.backend = backend

    @classmethod
    def _open(cls, fpath, mode, backend=None):
        fpath = os.fspath(fpath)
        exist_flag = os.path.exists(fpath)
        backend = cls._available_backends.get(backend, backend)
        if backend is None:
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
        return self.names()

    def names(self):
        if self.backend is tarfile:
            return (mem.name for mem in self.file)
        elif self.backend is zipfile:
            # does zip have an iterable structure?
            return iter(self.file.namelist())

    def read(self, name, mode='rb'):
        """
        Read data directly out of the archive.

        Args:
            name (str):
                the name of the archive member to read

            mode (str):
                This is a conceptual parameter that emulates the usual
                open mode. Defaults to "rb", which returns data as raw bytes.
                If "r" will decode the bytes into utf8-text.
        """
        if self.backend is tarfile:
            # a rework of makefile in tarfile.
            import io
            from tarfile import copyfileobj, ReadError
            self.file._check("r")
            tarinfo = self.file.getmember(name)
            source = self.file.fileobj
            source.seek(tarinfo.offset_data)
            bufsize = self.file.copybufsize
            target = io.BytesIO()
            if tarinfo.sparse is not None:
                for offset, size in tarinfo.sparse:
                    target.seek(offset)
                    copyfileobj(source, target, size, ReadError, bufsize)
                target.seek(tarinfo.size)
                target.truncate()
            else:
                copyfileobj(source, target, tarinfo.size, ReadError, bufsize)
            target.seek(0)
            data = target.read()
        elif self.backend is zipfile:
            # does zip have an iterable structure?
            data = self.file.read(name)
        else:
            raise NotImplementedError

        if mode == 'rb':
            return data
        elif mode == 'r':
            return data.decode('utf8')
        else:
            raise KeyError(mode)

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
            raise NotImplementedError

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


@ub.memoize
def _available_zipfile_compressions():
    available = set(['ZIP_STORED'])
    try:
        import zlib  # NOQA
    except ImportError:
        ...
    else:
        available.add('ZIP_DEFLATED')
    try:
        import bz2  # NOQA
    except ImportError:
        ...
    else:
        available.add('ZIP_BZIP2')
    try:
        import lzma  # NOQA
    except ImportError:
        ...
    else:
        available.add('ZIP_LZMA')
    return available


def _coerce_zipfile_compression(compression):
    if isinstance(compression, str):
        if compression == 'auto':
            priority = ['ZIP_LZMA', 'ZIP_DEFLATED', 'ZIP_BZIP2', 'ZIP_STORED']
            available = _available_zipfile_compressions()
            found = None
            for cand in priority:
                if cand in available:
                    found = cand
                    break
            compression = found
        compression = getattr(zipfile, compression)
    return compression
