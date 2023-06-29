#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class CocoFileHelper:
    """
    Transparent opening of either a regular json file, or a json file inside of
    a zipfile. TODO: relate to ub.zopen?
    """
    def __init__(self, fpath, mode='r'):
        self.fpath = fpath
        self.file = None
        self.zfile = None
        self.mode = mode

    def _open(self):
        import zipfile
        fpath = self.fpath
        if zipfile.is_zipfile(fpath):
            self.zfile = zfile = zipfile.ZipFile(fpath, 'r')
            members = zfile.namelist()
            if len(members) != 1:
                raise Exception(
                    'Currently only zipfiles with exactly 1 '
                    'kwcoco member are supported')
            self.file = zfile.open(members[0], mode=self.mode)
        else:
            self.file = open(fpath, mode=self.mode)
        return self.file

    def _close(self):
        if self.file:
            self.file.close()
        if self.zfile:
            self.zfile.close()

    def __enter__(self):
        self._open()
        return self.file

    def __exit__(self, ex_type, ex_value, ex_traceback):
        """
        Args:
            ex_type (Type[BaseException] | None):
            ex_value (BaseException | None):
            ex_traceback (TracebackType | None):

        Returns:
            bool | None
        """
        self._close()
        return False


class CocoInfoCLI(scfg.DataConfig):
    """
    Parse the "info" section of the coco json and print it.

    This is done using ijson, so it doesn't have to read the entire file.
    """
    __command__ = 'info'

    src = scfg.Value(None, help='input kwcoco path', position=1)

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> import sys, ubelt
            >>> sys.path.append(ubelt.expandpath('~/code/kwcoco'))
            >>> from kwcoco.cli.coco_info import *  # NOQA
            >>> from kwcoco.cli.coco_info import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict()
            >>> cls = CocoInfoCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        import rich
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + ub.urepr(config, nl=1))

        # import kwcoco
        from kwcoco.util import ijson_ext
        fpath = ub.Path(config.src)

        _cocofile = CocoFileHelper(fpath)
        try:
            file = _cocofile._open()
            info_section_iter = ijson_ext.items(file, prefix='info')
            try:
                info = next(info_section_iter)
            except StopIteration:
                info = None
        finally:
            _cocofile._close()
        rich.print('info = {}'.format(ub.urepr(info, nl=4)))


__cli__ = CocoInfoCLI
main = __cli__.main

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/kwcoco/kwcoco/cli/coco_info.py
        python -m kwcoco.cli.coco_info
    """
    main()
