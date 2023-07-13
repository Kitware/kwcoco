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
    This is useful when you quickly want to take a peek at a larger kwcoco
    file.
    """
    __command__ = 'info'

    src = scfg.Value(None, help='input kwcoco path', position=1)

    show_info = scfg.Value(False, isflag=True, help='if True, show the entire info section')
    first_video = scfg.Value(False, isflag=True, help='if True, show the first video dictionary')
    first_image = scfg.Value(False, isflag=True, help='if True, show the first image dictionary')
    first_annot = scfg.Value(False, isflag=True, help='if True, show the first annotation dictionary')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:ijson)
            >>> from kwcoco.cli.coco_info import *  # NOQA
            >>> import kwcoco
            >>> cls = CocoInfoCLI
            >>> cmdline = 0
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8')
            >>> # Add some info to the data section
            >>> dset.dataset['info'] = [{'type': 'demo', 'data': 'hi mom'}]
            >>> dset.fpath = ub.Path(dset.fpath).augment(prefix='infotest_')
            >>> dset.dump()
            >>> # test normal json
            >>> kwargs = dict(src=dset.fpath, first_image=True, first_video=True, first_annot=True)
            >>> cls.main(cmdline=cmdline, **kwargs)
            >>> # test zipped json
            >>> dset_zip = dset.copy()
            >>> dset_zip.fpath = dset_zip.fpath + '.zip'
            >>> dset_zip.dump()
            >>> kwargs = dict(src=dset_zip.fpath, first_image=True, first_video=True, first_annot=True)
            >>> cls.main(cmdline=cmdline, **kwargs)
            >>> # test bad-order json
            >>> dset_bad_order = dset.copy()
            >>> dset_bad_order.dataset['images'] = dset_bad_order.dataset.pop('images')
            >>> dset_bad_order.dataset['info'] = dset_bad_order.dataset.pop('info')
            >>> dset_bad_order.fpath = ub.Path(dset_bad_order.fpath).augment(prefix='bad_order')
            >>> dset_bad_order.dump()
            >>> kwargs = dict(src=dset_bad_order.fpath, first_image=True, first_video=True, first_annot=True)
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        print('config = ' + ub.urepr(config, nl=1))

        # TODO:
        # This assumes json files are in a standard order, but we can probably
        # use ijson's events to be robust to different orders.
        # Seems like it needs some custom code:
        # https://github.com/ICRAR/ijson/issues/85
        # from ijson.utils import coroutine, coros2gen  # NOQA
        # from ijson.common import ObjectBuilder  # NOQA
        # def multiprefix_items_basecoro(target, prefix, map_type=None):
        #     '''
        #     An couroutine dispatching native Python objects constructed from the events
        #     under a given prefix.
        #     '''
        #     while True:
        #         current, event, value = (yield)
        #         if current in prefix:
        #             if event in ('start_map', 'start_array'):
        #                 object_depth = 1
        #                 builder = ObjectBuilder(map_type=map_type)
        #                 while object_depth:
        #                     builder.event(event, value)
        #                     current, event, value = (yield)
        #                     if event in ('start_map', 'start_array'):
        #                         object_depth += 1
        #                     elif event in ('end_map', 'end_array'):
        #                         object_depth -= 1
        #                 del builder.containers[:]
        #                 target.send(builder.value)
        #             else:
        #                 target.send(value)

        # import kwcoco
        from kwcoco.util import ijson_ext
        fpath = ub.Path(config.src)

        _cocofile = CocoFileHelper(fpath)
        try:
            file = _cocofile._open()
            ijson_parser = ijson_ext.parse(file)

            if config.show_info:
                info_section_iter = ijson_ext.items(ijson_parser, prefix='info')
                try:
                    info = next(info_section_iter)
                except StopIteration:
                    info = None
                else:
                    print('info = {}'.format(ub.urepr(info, nl=4)))

            if config.first_video:
                video_section_iter = ijson_ext.items(ijson_parser, prefix='videos.item')
                try:
                    video = next(video_section_iter)
                except StopIteration:
                    video = None
                else:
                    print('video = {}'.format(ub.urepr(video, nl=4)))

            if config.first_image:
                image_section_iter = ijson_ext.items(ijson_parser, prefix='images.item')
                try:
                    image = next(image_section_iter)
                except StopIteration:
                    image = None
                else:
                    print('image = {}'.format(ub.urepr(image, nl=4)))

            if config.first_annot:
                image_section_iter = ijson_ext.items(ijson_parser, prefix='annotations.item')
                try:
                    annot = next(image_section_iter)
                except StopIteration:
                    annot = None
                else:
                    print('annot = {}'.format(ub.urepr(annot, nl=4)))

        finally:
            _cocofile._close()


__cli__ = CocoInfoCLI
main = __cli__.main

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/kwcoco/kwcoco/cli/coco_info.py
        python -m kwcoco.cli.coco_info

        kwcoco info --help
    """
    main()
