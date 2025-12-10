#!/usr/bin/env python
import ubelt as ub
import scriptconfig as scfg


class CocoStatsCLI(scfg.DataConfig):
    """
    Compute summary statistics about a COCO dataset.

    Basic stats are the number of images, annotations, categories, videos, and
    tracks. Extended stats are also available.

    SeeAlso:
        kwcoco visual_stats --help
    """
    __command__ = 'stats'

    src = scfg.Value(['special:shapes8'], position=1, help='path to dataset', nargs='+')
    basic = scfg.Value(True, isflag=True, help='show basic stats')
    extended = scfg.Value(True, isflag=True, help='show extended stats')
    catfreq = scfg.Value(True, isflag=True, help='show category frequency stats')
    boxes = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            show bounding box stats in width-height format.
            '''))
    image_size = scfg.Value(False, isflag=True, help='show image size stats')
    annot_attrs = scfg.Value(False, isflag=True, help='show annotation attribute information')
    image_attrs = scfg.Value(False, isflag=True, help='show image attribute information')
    video_attrs = scfg.Value(False, isflag=True, help='show video attribute information')
    io_workers = scfg.Value(0, help=ub.paragraph(
            '''
            number of workers when reading multiple kwcoco files
            '''))

    disk_usage = scfg.Value(False, isflag=True, help='measure disk usage of assets')

    embed = scfg.Value(False, isflag=True, help='embed into interactive shell for debugging')
    format = scfg.Value('human', help='output format. Can be "human", "json", or "yaml"')

    __epilog__ = """
    Example Usage:
        kwcoco stats --src=special:shapes8
        kwcoco stats --src=special:shapes8 --boxes=True
    """

    @classmethod
    def main(cls, cmdline=True, **kw):
        """
        CommandLine:
            xdoctest -m kwcoco.cli.coco_stats CocoStatsCLI.main:0
            xdoctest -m kwcoco.cli.coco_stats CocoStatsCLI.main:1

        Example:
            >>> kw = {'src': 'special:shapes8'}
            >>> cmdline = False
            >>> cls = CocoStatsCLI
            >>> cls.main(cmdline, **kw)

        Example:
            >>> # xdoctest: +REQUIRES(module:pyyaml)
            >>> from kwcoco.cli.coco_stats import *  # NOQA
            >>> kw = {
            >>>     'src': ['special:shapes8', 'special:vidshapes8', 'special:vidshapes2'],
            >>>     'basic': True,
            >>>     'extended': True,
            >>>     'catfreq': True,
            >>>     'image_size': True,
            >>>     'annot_attrs': True,
            >>>     'image_attrs': True,
            >>>     'video_attrs': True,
            >>>     'disk_usage': True,
            >>>     'boxes': True,
            >>> }
            >>> cmdline = False
            >>> cls = CocoStatsCLI
            >>> print('-- Test YAML format --')
            >>> kw['format'] = 'yaml'
            >>> cls.main(cmdline, **kw)
            >>> print('-- Test Human format --')
            >>> kw['format'] = 'human'
            >>> cls.main(cmdline, **kw)
        """
        import kwcoco
        import numpy as np
        config = cls.cli(data=kw, cmdline=cmdline, strict=True)
        try:
            from rich import print as rich_print
        except ImportError:
            rich_print = print

        human_readable = config.format == 'human'

        if human_readable:
            rich_print('config = {}'.format(ub.urepr(config, nl=1)))

        if config['src'] is None:
            raise Exception('must specify source: {}'.format(config['src']))

        if isinstance(config['src'], str):
            fpaths = [config['src']]
        else:
            fpaths = config['src']

        datasets = list(kwcoco.CocoDataset.coerce_multiple(fpaths, workers=config.io_workers, verbose=human_readable))
        if human_readable:
            print('Finished reading datasets')

        # hack dataset tags
        dset_tags = [dset.tag for dset in datasets]
        if len(set(dset_tags)) < len(dset_tags):
            from os.path import commonprefix
            dset_fpaths = [dset.fpath for dset in datasets]
            toremove = commonprefix(dset_fpaths)
            for dset in datasets:
                dset.tag = dset.fpath.replace(toremove, '')

        if human_readable:
            try:
                import networkx as nx
                for dset in datasets:
                    print('dset = {!r}'.format(dset))
                    print('Category Hierarchy: ')
                    print(nx.write_network_text(dset.object_categories().graph))
            except Exception:
                pass

        # FIXME: don't clobber global options
        import pandas as pd
        pd.set_option('max_colwidth', 256)

        stat_types = {}

        if config['basic']:
            stat_types['basic'] = tag_to_stats = {}
            for dset in datasets:
                tag_to_stats[dset.tag] = dset.basic_stats()
            if human_readable:
                df = pd.DataFrame.from_dict(tag_to_stats)
                if human_readable:
                    rich_print(df.T.to_string(float_format=lambda x: '%0.3f' % x))

        if config['extended']:
            stat_types['extended'] = tag_to_ext_stats = {}
            for dset in datasets:
                tag_to_ext_stats[dset.tag] = dset.extended_stats()

            allkeys = sorted(set(ub.flatten(s.keys() for s in tag_to_ext_stats.values())))
            for key in allkeys:
                if human_readable:
                    print('\n--{!r}'.format(key))
                df = pd.DataFrame.from_dict(
                    {k: v[key] for k, v in tag_to_ext_stats.items()})
                if human_readable:
                    rich_print(df.T.to_string(float_format=lambda x: '%0.3f' % x))

        if config['catfreq']:
            stat_types['catfreq'] = tag_to_freq = {}
            for dset in datasets:
                tag_to_freq[dset.tag] = dset.category_annotation_frequency()
            df = pd.DataFrame.from_dict(tag_to_freq)
            if human_readable:
                rich_print(df.to_string(float_format=lambda x: '%0.3f' % x))

        if config['video_attrs']:
            if human_readable:
                print('Video Attribute Histogram')
            stat_types['video_attrs'] = {}
            for dset in datasets:
                attrs = dset.videos().attribute_frequency()
                stat_types['video_attrs'][dset.tag] = attrs
                if human_readable:
                    print('hist(video_attrs) = {}'.format(ub.urepr(attrs, nl=1)))

        if config['image_attrs']:
            if human_readable:
                print('Image Attribute Histogram')
            stat_types['image_attrs'] = {}
            for dset in datasets:
                if human_readable:
                    print('dset.tag = {!r}'.format(dset.tag))
                attrs = dset.images().attribute_frequency()
                stat_types['image_attrs'][dset.tag] = attrs
                if human_readable:
                    print('hist(image_attrs) = {}'.format(ub.urepr(attrs, nl=1)))

        if config['annot_attrs']:
            if human_readable:
                print('Annot Attribute Histogram')
            stat_types['annot_attrs'] = {}
            for dset in datasets:
                if human_readable:
                    print('dset.tag = {!r}'.format(dset.tag))
                attrs = dset.annots().attribute_frequency()
                stat_types['annot_attrs'][dset.tag] = attrs
                if human_readable:
                    print('hist(annot_attrs) = {}'.format(ub.urepr(attrs, nl=1)))

        if config['boxes']:
            if human_readable:
                print('Box stats')
            stat_types['boxes'] = {}
            for dset in datasets:
                box_stats = dset.boxsize_stats()
                if human_readable:
                    print('dset.tag = {!r}'.format(dset.tag))
                    print(ub.urepr(box_stats, nl=-1, precision=2))
                stat_types['boxes'][dset.tag] = box_stats

        if config['image_size']:
            if human_readable:
                print('Image size stats')
            stat_types['image_size'] = {}
            for dset in datasets:
                if human_readable:
                    print('dset.tag = {!r}'.format(dset.tag))
                images = dset.images()
                heights = np.array(images.lookup('height', np.nan))
                widths = np.array(images.lookup('width', np.nan))
                rt_areas = np.sqrt(heights * widths)
                imgsize_df = pd.DataFrame({
                    'height': heights,
                    'widths': widths,
                    'rt_areas': rt_areas,
                })
                stat_types['image_size'][dset.tag] = image_size_info = {}
                size_stats = imgsize_df.describe()
                image_size_info['size_stats'] = size_stats.to_dict()
                if human_readable:
                    print(size_stats)
                idx = np.argmax(rt_areas)

                try:
                    biggest_image = images.take([idx]).coco_images[0]
                    max_area_h = biggest_image.img['height']
                    max_area_w = biggest_image.img['width']
                    if human_readable:
                        print('Max image: {} x {}'.format(max_area_w, max_area_h))
                    image_size_info['max_image_wh'] = (max_area_w, max_area_h)
                    pixels = max_area_w * max_area_h
                    total_disk_bytes = 0
                    for fpath in list(biggest_image.iter_image_filepaths()):
                        fpath = ub.Path(fpath)
                        num_bytes = fpath.stat().st_size
                        total_disk_bytes += num_bytes
                    total_disk_gb = total_disk_bytes / 2 ** 30
                    pixel_gb_per_bit = (pixels / 8) / 2 ** 30
                    if human_readable:
                        print('total_disk_gb = {!r}'.format(total_disk_gb))
                        print('pixel_gb_per_bit = {!r}'.format(pixel_gb_per_bit))
                    image_size_info['total_disk_gb'] = (total_disk_gb)
                    image_size_info['pixel_gb_per_bit'] = (pixel_gb_per_bit)
                except Exception:
                    if human_readable:
                        print('error getting max size')
                    image_size_info['errors'] = 'error getting max size'

                # print('dset.tag = {!r}'.format(dset.tag))
                # print(ub.urepr(dset.boxsize_stats(), nl=-1, precision=2))

        if config['disk_usage']:
            if human_readable:
                print('Disk usage stats')
            stat_types['disk_usage'] = {}
            for dset in datasets:
                if human_readable:
                    print('dset.tag = {!r}'.format(dset.tag))
                disk_info = _dataset_disk_usage(dset)
                stat_types['disk_usage'][dset.tag] = disk_info
                if human_readable:
                    disk_size = byte_str(disk_info['total_bytes'])
                    if human_readable:
                        print(f'Disk Usage: {disk_size}')

        if not human_readable:
            import kwutil
            stat_types = kwutil.util_json.ensure_json_serializable(stat_types)
            # Rotate dictionaries so the dataset is the top-level key
            rotated_stat_type = {
                dset.tag: {
                    'fpath': dset.fpath,
                    'tag': dset.tag
                }
                for dset in datasets
            }
            for type_key1, value1 in stat_types.items():
                for tag_key2, value2 in value1.items():
                    rotated_stat_type[tag_key2][type_key1] = value2

            # output stats as a List[dict]
            stat_lists = list(rotated_stat_type.values())

            if config.format == 'json':
                import json
                print(json.dumps(stat_lists, indent=' '))
            elif config.format == 'yaml':
                import kwutil
                print(kwutil.Yaml.dumps(stat_lists, backend='pyyaml'))
            elif config.format == 'urepr':
                print(ub.urepr(stat_lists, nl=-1))
            else:
                raise KeyError(config.format)

        if config['embed']:
            # Hidden hack
            import xdev
            xdev.embed()

        # for dset in datasets:
        #     # dset = datasets[0]
        #     # kwcoco.CocoDataset.coerce(config['src'])
        #     print('dset.fpath = {!r}'.format(dset.fpath))

        #     if config['basic']:
        #         basic = dset.basic_stats()
        #         print('basic = {}'.format(ub.urepr(basic, nl=1)))

        #     if config['extended']:
        #         extended = dset.extended_stats()
        #         print('extended = {}'.format(ub.urepr(extended, nl=1, precision=2)))

        #     if config['catfreq']:
        #         print('Category frequency')
        #         freq = dset.category_annotation_frequency()
        #         import pandas as pd
        #         df = pd.DataFrame.from_dict({str(dset.tag): freq})
        #         pd.set_option('max_colwidth', 256)
        #         print(df.to_string(float_format=lambda x: '%0.3f' % x))

        #     if config['boxes']:
        #         print('Box stats')
        #         print(ub.urepr(dset.boxsize_stats(), nl=-1, precision=2))


def _dataset_disk_usage(dset):
    """
    Compute disk usage of all image assets referenced by this dataset.

    Returns:
        dict:
            {
                'num_files': int,
                'total_bytes': int,
                'total_gb': float,
                'missing_files': List[str],
            }
    """
    # Collect all filepaths from images. iter_image_filepaths() typically
    # includes primary + auxiliary assets.
    filepaths = []
    for coco_img in dset.images().coco_images_iter():
        for fpath in coco_img.iter_image_filepaths():
            if fpath is not None:
                filepaths.append(ub.Path(fpath))

    # Also measure the size of this file if it exists.
    fpath = dset.fpath
    if fpath is not None:
        filepaths.append(ub.Path(fpath))

    # Deduplicate paths (use resolved paths to avoid double-counting symlinks)
    unique_paths = []
    seen = set()
    for p in filepaths:
        try:
            r = p.resolve()
        except Exception:
            r = p
        if r not in seen:
            seen.add(r)
            unique_paths.append(r)

    total_bytes = 0
    missing = []

    for p in unique_paths:
        try:
            total_bytes += p.stat().st_size
        except FileNotFoundError:
            missing.append(str(p))
        except OSError:
            # Permissions or other oddities – just record as missing-ish
            missing.append(str(p))

    info = {
        'num_files': len(unique_paths),
        'total_bytes': int(total_bytes),
    }
    if missing:
        info['missing_files'] = missing
        info['num_missing_files'] = len(missing)
    return info


def byte_str(num, unit='auto', precision=2):
    """
    Automatically chooses relevant unit (KB, MB, or GB) for displaying some
    number of bytes.

    Args:
        num (int): number of bytes
        unit (str): which unit to use, can be auto, B, KB, MB, GB, TB, PB, EB,
            ZB, or YB.
        precision (int): number of decimals of precision

    References:
        https://en.wikipedia.org/wiki/Orders_of_magnitude_(data)

    Returns:
        str: string representing the number of bytes with appropriate units

    Example:
        >>> num_list = [1, 100, 1024,  1048576, 1073741824, 1099511627776]
        >>> result = ub.urepr(list(map(byte_str, num_list)), nl=0)
        >>> print(result)
        ['0.00 KB', '0.10 KB', '1.00 KB', '1.00 MB', '1.00 GB', '1.00 TB']
    """
    abs_num = abs(num)
    if unit == 'auto':
        if abs_num < 2.0 ** 10:
            unit = 'KB'
        elif abs_num < 2.0 ** 20:
            unit = 'KB'
        elif abs_num < 2.0 ** 30:
            unit = 'MB'
        elif abs_num < 2.0 ** 40:
            unit = 'GB'
        elif abs_num < 2.0 ** 50:
            unit = 'TB'
        elif abs_num < 2.0 ** 60:
            unit = 'PB'
        elif abs_num < 2.0 ** 70:
            unit = 'EB'
        elif abs_num < 2.0 ** 80:
            unit = 'ZB'
        else:
            unit = 'YB'
    if unit.lower().startswith('b'):
        num_unit = num
    elif unit.lower().startswith('k'):
        num_unit =  num / (2.0 ** 10)
    elif unit.lower().startswith('m'):
        num_unit =  num / (2.0 ** 20)
    elif unit.lower().startswith('g'):
        num_unit = num / (2.0 ** 30)
    elif unit.lower().startswith('t'):
        num_unit = num / (2.0 ** 40)
    elif unit.lower().startswith('p'):
        num_unit = num / (2.0 ** 50)
    elif unit.lower().startswith('e'):
        num_unit = num / (2.0 ** 60)
    elif unit.lower().startswith('z'):
        num_unit = num / (2.0 ** 70)
    elif unit.lower().startswith('y'):
        num_unit = num / (2.0 ** 80)
    else:
        raise ValueError('unknown num={!r} unit={!r}'.format(num, unit))
    return ub.urepr(num_unit, precision=precision) + ' ' + unit


__cli__ = CocoStatsCLI

if __name__ == '__main__':
    """
    CommandLine:
        python -m kwcoco.cli.coco_stats --src=special:shapes8
    """
    __cli__.main()
