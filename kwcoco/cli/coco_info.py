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

    Note: there are issues with this tool when the sections are not in the
    expected order, or if the requested sections are empty. Help wanted.
    """
    __command__ = 'info'

    src = scfg.Value(None, help='input kwcoco path', position=1)

    show_info = scfg.Value('auto', isflag=True, help='The number of info dictionaries to show. if True, show all of them. The default of "auto" works around an issue. It is set to True if no other table is shown and 0 otherwise', short_alias=['i'])
    show_licenses = scfg.Value(0, isflag=True, help='The number of licenses dictionaries to show. if True, show all of them', short_alias=['l'])
    show_categories = scfg.Value(0, isflag=True, help='The number of category dictionaries to show. if True, show all of them', short_alias=['c'])
    show_videos = scfg.Value(0, isflag=True, help='The number of video dictionaries to show. if True, show all of them', short_alias=['v'])
    show_images = scfg.Value(0, isflag=True, help='The number of image dictionaries to show. if True, show all of them', short_alias=['g'])
    # TODO:
    show_tracks = scfg.Value(0, isflag=True, help='The number of track dictionaries to show. if True, show all of them', short_alias=['t'])
    show_annotations = scfg.Value(0, isflag=True, help='The number of annotation dictionaries to show. if True, show all of them', short_alias=['a'])

    rich = scfg.Value(True, isflag=True, help='if True, try to use rich')
    verbose = scfg.Value(0, isflag=True, help='if True, print extra information (i.e. the configuration). If false, then stdout should be redirectable as a regular json object')

    # TODO add more ways to query what parts we want to show
    image_name = scfg.Value(None, help='If specified, lookup and show the image with this name')

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
            >>> kwargs = dict(src=dset.fpath, show_images=True, show_videos=True, show_annotations=True)
            >>> cls.main(cmdline=cmdline, **kwargs)
            >>> # test zipped json
            >>> dset_zip = dset.copy()
            >>> dset_zip.fpath = dset_zip.fpath + '.zip'
            >>> dset_zip.dump()
            >>> kwargs = dict(src=dset_zip.fpath, show_images=True, show_videos=True, show_annotations=True)
            >>> cls.main(cmdline=cmdline, **kwargs)
            >>> # test bad-order json
            >>> dset_bad_order = dset.copy()
            >>> dset_bad_order.dataset['images'] = dset_bad_order.dataset.pop('images')
            >>> dset_bad_order.dataset['info'] = dset_bad_order.dataset.pop('info')
            >>> dset_bad_order.fpath = ub.Path(dset_bad_order.fpath).augment(prefix='bad_order')
            >>> dset_bad_order.dump()
            >>> kwargs = dict(src=dset_bad_order.fpath, show_images=True, show_videos=True, show_annotations=True)
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        try:
            if not config.rich:
                raise ImportError
            from rich.markup import escape as _rich_escape
            from rich import print as _raw_rich_print
            def rich_print(msg):
                _raw_rich_print(_rich_escape(msg))
        except ImportError:
            rich_print = print

        if config.verbose:
            rich_print('config = ' + ub.urepr(config, nl=1))

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
        if config.src is None:
            raise ValueError('A source kwcoco file is required')

        fpath = ub.Path(config.src)

        sentinel = object()
        _cocofile = CocoFileHelper(fpath)
        try:
            file = _cocofile._open()
            ijson_parser = ijson_ext.parse(file, use_float=True)

            config_table_keys = {
                'show_info': 'info',
                'show_licenses': 'licenses',
                'show_categories': 'categories',
                'show_videos': 'videos',
                'show_images': 'images',
                'show_tracks': 'tracks',
                'show_annotations': 'annotations',
            }
            parent_to_num = {}
            max_value = 0
            auto_keys = []
            for config_key, table_name in config_table_keys.items():
                value = config[config_key]
                if value is True:
                    value = float('inf')
                elif value == 'auto':
                    auto_keys.append(config_key)
                    continue
                else:
                    try:
                        value = int(value)
                    except Exception:
                        value = float(value)

                max_value = max(max_value, value)
                parent_to_num[table_name] = value

            # If any table is set to a non-zero number of rows to show, then
            # set all auto values to zero, otherwise set them to inf.
            for config_key in auto_keys:
                table_name = config_table_keys[config_key]
                if max_value > 0:
                    parent_to_num[table_name] = 0
                else:
                    parent_to_num[table_name] = float('inf')

            parent_to_request = {}

            if config.image_name is not None:
                parent_to_request['images'] = {
                    'name': config.image_name,
                }

            if config.verbose:
                print('parent_to_num = {}'.format(ub.urepr(parent_to_num, nl=1)))
                print('parent_to_request = {}'.format(ub.urepr(parent_to_request, nl=1)))

            print('{')

            # TODO: we want a function outside the CLI that
            # does the core of the work here.

            parent = 'info'
            if parent_to_num[parent] > 0:
                # Hack for info, which might not exist and is enabled by default
                info_iter = ijson_ext.items(ijson_parser, prefix=parent)
                print(f'"{parent}": ')
                try:
                    info_section = next(info_iter)
                except StopIteration:
                    print('{}')
                else:
                    rich_print('{}'.format(ub.urepr(info_section, nl=4, trailsep=False).replace('\'', '"')))

            parent_order = [
                'categories',
                'videos',
                'images',
                'tracks',
                'annotations',
            ]
            # TODO: need to be able to either:
            # check that a parent item was seen and then use that OR skip the
            # .item part if we detect an end_map right after the start map for
            # the parent prefix. Currently this will fail if any requested
            # section is empty, or if the sections are not in the same order
            # specified in parent-order.
            prev_parent = sentinel
            for parent in parent_order:

                num_to_show = parent_to_num[parent]
                request = parent_to_request.get(parent, {})

                satisfied = num_to_show == 0 and not bool(request)
                if not satisfied:

                    if prev_parent is not sentinel:
                        print(',')

                    obj_iter = ijson_ext.items(ijson_parser, prefix=f'{parent}.item')

                    print(f'"{parent}": [')

                    prev_obj = sentinel
                    num_shown = 0

                    for obj in obj_iter:

                        want_name = request.get('name')

                        show_this_one = False

                        if num_shown < num_to_show:
                            show_this_one = True

                        if want_name is not None and obj['name'] == want_name:
                            request.pop('name')  # Mark as satisfied? Probably a better way?
                            show_this_one = True

                        if show_this_one:
                            if prev_obj is not sentinel:
                                print(',')
                            rich_print('{}'.format(ub.urepr(obj, nl=4, trailsep=False).replace('\'', '"')))
                            num_shown += 1
                            prev_obj = obj

                        satisfied = num_shown >= num_to_show and not bool(request)
                        if satisfied:
                            break

                    print(']')
                    prev_parent = parent

            print('}')

        finally:
            _cocofile._close()


# This was a start for a fix for the different order problem, but I wasnt able
# to finish it. I need to learn how the ijson coroutines work better.
# def main2(config):
#     num_infos = float('inf') if config.show_info is True else int(config.show_info)
#     num_videos = float('inf') if config.show_videos is True else int(config.show_videos)
#     num_images = float('inf') if config.show_images is True else int(config.show_images)
#     num_annotations = float('inf') if config.show_annotations is True else int(config.show_annotations)
#     num_categories = float('inf') if config.show_annotations is True else int(config.show_annotations)

#     entry_iter = iterative_kwcoco_parse(fpath, num_infos, num_categories,
#                                         num_videos, num_images,
#                                         num_annotations)

#     # import sys
#     # write = sys.stdout.write
#     # from rich import get_console
#     # write_console = get_console()

#     print('{')
#     prev_parent = None
#     for parent, item in entry_iter:
#         if parent != prev_parent:
#             if prev_parent is not None:
#                 print('],')
#             print(f'"{parent}": [')
#         else:
#             if prev_parent is not None:
#                 print(',')
#         rich_print('{}'.format(ub.urepr(item, nl=4).replace('\'', '"')))

#     if prev_parent is not None:
#         print(']')
#     print('}')

# def iterative_kwcoco_parse(fpath, num_infos, num_categories, num_videos,
#                            num_images, num_annotations):
#     """
#     Iteravely generate parts of the kwcoco file for very fast response time.
#     """
#     from kwcoco.util import ijson_ext
#     parent_to_num = ub.odict([
#         ("info", num_infos),
#         ("categories", num_categories),
#         ("videos", num_videos),
#         ("images", num_images),
#         ("annotations", num_annotations),
#     ])
#     _cocofile = CocoFileHelper(fpath)
#     file = _cocofile._open()
#     try:
#         ijson_parser = ijson_ext.parse(file, use_float=True)

#         def make_backtracker(next_tuple, ijson_parser):
#             yield next_tuple
#             # ijson_parser.send(next_tuple)
#             yield from ijson_parser

#         trace = print

#         def _section_finder(ijson_parser):
#             # Handle items being missing or empty
#             for prefix, event, value in ijson_parser:
#                 # Find the start of any of the sections of interest
#                 if prefix in parent_to_num:

#                     trace(f'WE FOUND: {prefix}')

#                     # Check to see if that section ends immediately
#                     next_tuple = next(ijson_parser)
#                     next_prefix, next_event, next_value = next_tuple
#                     if next_event == 'end_array':
#                         # If it does, continue on
#                         trace(f'BUT {prefix} IT ENDED IMMEDIATELY')
#                         continue
#                     else:
#                         # Otherwise we need to munge the iterateor to reverse
#                         # itself one step, and then we parse some number of
#                         # items from the section.
#                         trace('AND THERE ARE ITEMS')
#                         parent = prefix
#                         num_objs = parent_to_num[parent]
#                         if num_objs > 0:
#                             backtracked = make_backtracker(next_tuple, ijson_parser)
#                             prefix = parent + '.item'
#                             print(f'ENUMERATING SOME OF prefix={prefix}')
#                             obj_iter = ijson_ext.items(backtracked, prefix=prefix)
#                             for idx, obj in enumerate(obj_iter, start=1):
#                                 yield parent, obj
#                                 if num_infos >= idx:
#                                     break

#         yield from list(_section_finder(ijson_parser))

#         # for prefix in _section_finder(ijson_parser):
#         #     break
#         #     ...

#         # file.seek(0)
#         # ijson_parser = ijson_ext.parse(file, use_float=True)
#         # obj_iter = ijson_ext.items(ijson_parser, prefix='info.item')
#         # list(obj_iter)

#         # for parent, num_objs in parent_to_num.items():
#         #     if num_objs > 0:
#         #         prefix = parent + '.item'
#         #         print(f'prefix={prefix}')
#         #         obj_iter = ijson_ext.items(ijson_parser, prefix=prefix)
#         #         print(list(obj_iter))
#         #         for idx, obj in enumerate(obj_iter, start=1):
#         #             print(f'obj={obj}')
#         #             yield parent, obj
#         #             if num_infos >= idx:
#         #                 break

#     finally:
#         _cocofile._close()


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
