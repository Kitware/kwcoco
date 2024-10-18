#!/usr/bin/env python3
"""
CommandLine:
    xdoctest -m kwcoco.cli.coco_plot_stats __doc__:0
    xdoctest -m kwcoco.cli.coco_plot_stats __doc__:1

Example:
    >>> # xdoctest: +REQUIRES(module:kwutil)
    >>> # Stats on a simple dataset
    >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
    >>> import kwcoco
    >>> dpath = ub.Path.appdir('kwcoco/tests/vis_stats').ensuredir()
    >>> coco_fpath = kwcoco.CocoDataset.demo('vidshapes8').fpath
    >>> cmdline = 0
    >>> kwargs = dict(src=coco_fpath, dst_dpath=dpath)
    >>> cls = PlotStatsCLI
    >>> cls.main(cmdline=cmdline, **kwargs)

Example:
    >>> # xdoctest: +REQUIRES(module:kwutil)
    >>> # Stats on a more complex dataset
    >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
    >>> import kwcoco
    >>> import kwarray.distributions
    >>> import kwarray
    >>> rng = kwarray.ensure_rng(0)
    >>> dpath = ub.Path.appdir('kwcoco/tests/vis_stats2').ensuredir()
    >>> dset = kwcoco.CocoDataset.demo('vidshapes8', image_size='random',
    >>>                                timestamps=True, rng=rng)
    >>> coco_fpath = dset.fpath
    >>> cmdline = 0
    >>> kwargs = dict(src=coco_fpath, dst_dpath=dpath)
    >>> cls = PlotStatsCLI
    >>> cls.main(cmdline=cmdline, **kwargs)
"""
import scriptconfig as scfg
import ubelt as ub
import os

try:
    from line_profiler import profile
except ImportError:
    profile = ub.identity


class PlotStatsCLI(scfg.DataConfig):
    """
    Inspect properties of dataset and write raw data tables and visual plots.
    """
    __command__ = 'plot_stats'
    __alias__ = ['visual_stats']

    src = scfg.Value(None, help='path to kwcoco file', position=1)
    plots = scfg.Value(None, help='names of specific plots to create', nargs='+', position=2)

    dst_fpath = scfg.Value('auto', help='manifest of results. If unspecfied defaults to dst_dpath / "stats.json"')
    dst_dpath = scfg.Value('./coco_annot_stats', help='directory to dump results')

    with_process_context = scfg.Value(True, help='set to false to disable process contxt')

    dpi = scfg.Value(300, help='dpi for figures')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        from kwcoco.util.util_rich import rich_print
        try:
            from rich.markup import escape
        except ImportError:
            from ubelt import identity as escape
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich_print('config = ' + escape(ub.urepr(config, nl=1)))
        run(config)

__cli__ = PlotStatsCLI


def prep_plots(src, plots_dpath=None, dpi=300):
    import kwcoco
    import json
    from kwcoco.util.util_rich import rich_print

    with ub.Timer(label='Loading kwcoco file') as load_timer:
        dset = kwcoco.CocoDataset.coerce(src)

    scalar_stats, tables_data, nonsaved_data, dataframes = build_stats_data(dset)

    cache_dpath = ub.Path(dset.fpath).absolute().parent / '_cache'
    cacher = ub.Cacher('stats', depends=dset._cached_hashid(), dpath=cache_dpath)
    loaded = cacher.tryload()
    if loaded is None:
        info = build_stats_data(dset)
        scalar_stats, tables_data, nonsaved_data, dataframes = info
        cacher.save((scalar_stats, tables_data, nonsaved_data))
    else:
        scalar_stats, tables_data, nonsaved_data = loaded
        dataframes = None
    scalar_stats['kwcoco_loadtime_seconds'] = load_timer.elapsed

    if dataframes is not None:
        perimage_data = dataframes['perimage_data']
        perannot_data = dataframes['perannot_data']
        perannot_summary = json.loads(perannot_data.describe().to_json())
        perimage_summary = json.loads(perimage_data.describe().to_json())
        rich_print('perannot_summary:')
        rich_print(ub.urepr(perannot_summary, nl=-1, align=':', precision=2))
        rich_print('perimage_summary:')
        rich_print(ub.urepr(perimage_summary, nl=-1, align=':', precision=2))

    print('Preparing plots')
    plots = Plots(plots_dpath, tables_data, nonsaved_data, dpi=dpi)
    plots.scalar_stats = scalar_stats
    return plots


@profile
def run(config):
    import json
    import safer
    import kwutil

    dst_dpath = ub.Path(config['dst_dpath'])
    if config['dst_fpath'] == 'auto':
        dst_fpath = dst_dpath / 'stats.json'
    else:
        dst_fpath = ub.Path(config['dst_fpath'])
    dst_dpath = dst_dpath.absolute()
    dst_dpath.ensuredir()
    plots_dpath = dst_dpath / 'annot_stat_plots'
    tables_fpath = dst_dpath / 'stats_tables.json'
    tables_fpath = dst_dpath / 'stats_tables.json'

    from kwcoco.util.util_rich import rich_print
    rich_print(f'Destination Path: [link={dst_dpath}]{dst_dpath}[/link]')

    if config.with_process_context:
        from kwutil.process_context import ProcessContext
        proc_context = ProcessContext(
            name='kwcoco.cli.coco_plot_stats',
            config=kwutil.Json.ensure_serializable(config.to_dict())
        )
        proc_context.start()
    else:
        proc_context = None

    src = config['src']
    dpi = config.dpi
    plots = prep_plots(src, plots_dpath, dpi=dpi)
    tables_data = plots.tables_data
    rich_print(f'Will write plots to: [link={plots_dpath}]{plots_dpath}[/link]')
    scalar_stats = plots.scalar_stats

    print(f'scalar_stats = {ub.urepr(scalar_stats, nl=1)}')
    rich_print(f'Write {tables_fpath}')
    with safer.open(tables_fpath, 'w', temp_file=not ub.WIN32) as fp:
        json.dump(tables_data, fp, indent='  ')

    available_plots = list(plots.plot_functions.keys())
    if config.plots is not None:
        user_requested_plots = ub.oset(config.plots)
        requested_plots = list(user_requested_plots & set(available_plots))
        unknown = user_requested_plots - set(available_plots)
        if unknown:
            print(f'WARNING: ignoring unknown plots unknown={unknown}')
    else:
        requested_plots = available_plots
    print(f'requested_plots = {ub.urepr(requested_plots, nl=1)}')

    rich_print(f'Will write plots to: [link={plots_dpath}]{plots_dpath}[/link]')
    with ub.Timer(label='Plotting') as plot_timer:
        pman = kwutil.util_progress.ProgressManager()
        with pman:
            # plot_func_keys = [
            #     # 'images_over_time',
            #     'images_timeofday_distribution',
            # ]
            for key in pman.ProgIter(requested_plots, desc='plot'):
                pman.update_info(f'Plotting: {key}')
                func = plots.plot_functions[key]
                try:
                    func(plots)
                except Exception as ex:
                    rich_print(f'[red] ERROR: in {func}')
                    rich_print(f'ex = {ub.urepr(ex, nl=1)}')
                    import traceback
                    traceback.print_exc()
                    if 0:
                        raise
    scalar_stats['plottime_seconds'] = plot_timer.elapsed
    rich_print(f'Finished writing plots to: [link={plots_dpath}]{plots_dpath}[/link]')

    # Write manifest of all data written to disk
    summary_data = {}
    if proc_context is not None:
        proc_context.stop()
        obj = proc_context.stop()
        obj = kwutil.Json.ensure_serializable(obj)
        summary_data['info'] = [obj]
    summary_data['scalar_stats'] = kwutil.Json.ensure_serializable(scalar_stats)

    print('Finalizing manifest')
    summary_data['src'] = str(config['src'])
    summary_data['plots_dpath'] = os.fspath(plots_dpath)
    summary_data['tables_fpath'] = os.fspath(tables_fpath)
    # summary_data['perannot_summary'] = perannot_summary
    # summary_data['perimage_summary'] = perimage_summary
    # Write file to indicate the process has completed correctly
    # TODO: Use safer
    rich_print(f'Write {dst_fpath}')
    with safer.open(dst_fpath, 'w', temp_file=not ub.WIN32) as fp:
        json.dump(summary_data, fp, indent='    ')


def rerun_plots(tables_fpath):
    """
    TODO:
        - [ ] Easy CLI / IPython mechanism to rerun plots with precompiled stat tables

    from kwcoco.cli.coco_plot_stats import *  # NOQA
    tables_fpath = './coco_annot_stats2/stats_tables.json'
    import kwplot
    import kwplot
    kwplot.autosns()
    """
    import kwutil
    tables_data = kwutil.Json.load(tables_fpath)
    plots_dpath = None
    nonsaved_data = None
    dpi = None
    plots = Plots(plots_dpath, tables_data, nonsaved_data, dpi=dpi)
    self = plots  # NOQA
    BuiltinPlots.polygon_area_vs_num_verts(self)
    # plots.plot_functions['polygon_area_vs_num_verts'](self)


def geospatial_stats(dset, images, perimage_data):
    import math
    import warnings
    import numpy as np
    ESTIMATE_SUNLIGHT = 1
    if ESTIMATE_SUNLIGHT:
        # This might be more of a domain-specific plugin feature, than
        # something that kwcoco should be concerned with.

        def coco_estimate_sunlight(dset, image_ids=None):
            try:
                from kwgis.utils.util_sunlight import estimate_sunlight
                import suntime  # NOQA
                import timezonefinder  # NOQA
                import pytz  # NOQA
            except ImportError as ex:
                from kwutil.util_exception import add_exception_note
                msg = f'''
                    Missing requirements, please:
                    pip install suntime timezonefinder pytz kwgis
                    {ex}
                    '''
                if 0:
                    raise add_exception_note(ex, ub.codeblock(msg))
                else:
                    warnings.warn(msg)
            else:
                from kwutil.util_math import Rational
                sunlight_values = []
                images = dset.images(image_ids)
                for img in images.objs_iter():
                    if 'geos_point' not in img:
                        sunlight = math.nan
                    else:
                        geos_point = img['geos_point']
                        if isinstance(geos_point, float) and math.isnan(geos_point):
                            sunlight = math.nan
                        elif not isinstance(geos_point, dict):
                            warnings.warn(f'Warning: unknown geos_point format {geos_point!r} in {img!r}')
                            sunlight = np.nan
                        else:
                            coords = geos_point['coordinates']
                            point = [Rational.coerce(x) for x in coords]
                            lon, lat = point
                            datetime = img['datetime']
                            sunlight = estimate_sunlight(lat, lon, datetime)
                    sunlight_values.append(sunlight)
                sunlight_values = np.array(sunlight_values)
                return sunlight_values

        try:
            sunlight_values = coco_estimate_sunlight(dset, image_ids=images)
            # print(f'sunlight_values={sunlight_values}')
            if not np.isnan(sunlight_values).all():
                perimage_data['sunlight'] = sunlight_values
        except ImportError:
            print('Unable to estimate sunlight')
            raise
        except Exception:
            print('Unable to estimate sunlight')


@profile
def build_stats_data(dset):
    """
    Build a table of perimage and perannotation as well as a summary table of
    higher level results.
    """
    import kwimage
    import numpy as np
    import pandas as pd
    import json

    with ub.Timer(label='Building tables for stats') as stats_timer:
        images = dset.images()
        image_widths = images.get('width')
        image_heights = images.get('height')
        max_width = max(image_widths)  # NOQA
        max_height = max(image_heights)  # NOQA

        anns_per_image = np.array(images.n_annots)
        images_with_eq0_anns = (anns_per_image == 0).sum()
        images_with_ge1_anns = (anns_per_image >= 1).sum()
        scalar_stats = {
            **dset.basic_stats(),
            'images_with_eq0_anns': images_with_eq0_anns,
            'images_with_ge1_anns': images_with_ge1_anns,
            'frac_images_with_ge1_anns': images_with_ge1_anns / len(images),
            'frac_images_with_eq0_anns': images_with_eq0_anns / len(images),
        }

        # Fixme, standardize timestamp field
        datetime = [
            a or b for a, b in zip(images.get('timestamp', None),
                                   images.get('datetime', None))
        ]

        perimage_data = pd.DataFrame({
            'anns_per_image': anns_per_image,
            'width': image_widths,
            'height': image_heights,
            'datetime': datetime,
        })

        area = (perimage_data['width'] * perimage_data['height'])
        area_sorted_idxs = area.sort_values().index
        img_dsizes = perimage_data.loc[area_sorted_idxs, ['width', 'height']]
        img_dsize_tuples = list(map(tuple, img_dsizes.values.tolist()))
        dsize_hist = ub.udict(ub.dict_hist(img_dsize_tuples))
        median_area_image_size = img_dsize_tuples[len(img_dsize_tuples) // 2]
        most_frequent_image_size = '{} x {}'.format(*list(dsize_hist.items())[-1][0])
        scalar_stats['median_area_image_size'] = median_area_image_size
        scalar_stats['most_frequent_image_size'] = most_frequent_image_size
        print(f'scalar_stats = {ub.urepr(scalar_stats, nl=1)}')
        # scalar_stats['table_buildtime_seconds'] = stats_timer.elapsed

        try:
            geospatial_stats(dset, images, perimage_data)
        except Exception as ex:
            raise
            print(f'Failed to build domain (geospatial) stats: ex={ex}')
        else:
            print('Built domain (goespatial) stats')
            # TODO: medical stats / other domain stats

        # try:
        #     # We dont want to require geopandas
        #     import geopandas as gpd
        #     _DataFrame = gpd.GeoDataFrame
        # except Exception:
        _DataFrame = pd.DataFrame

        annots = dset.annots()
        print('Building stats')
        # detections : kwimage.Detections = annots.detections
        # boxes : kwimage.Boxes = detections.boxes
        # polys : kwimage.PolygonList = detections.data['segmentations']

        alt_boxes = []
        alt_polys = []
        alt_geoms = []
        for ann in ub.ProgIter(annots.objs, desc='gather annotation polygons'):
            box = kwimage.Box.coerce(ann['bbox'], 'xywh')
            alt_boxes.append(box)
            try:
                poly = kwimage.MultiPolygon.coerce(ann['segmentation'])
                geom = poly.to_shapely()
            except Exception:
                # Masks with linestrings can get misinterpreted, but we can fix
                # them by considering pixels as areas instead of points
                sseg = ann.get('segmentation', None)
                if sseg is not None:
                    mask = kwimage.Mask.coerce(sseg)
                    poly = mask.to_multi_polygon(pixels_are='areas')
                    geom = poly.to_shapely()
                else:
                    # Fallback onto the box if the polygon broken
                    poly = box.to_polygon()
                    geom = poly.to_shapely()
            alt_polys.append(poly)
            alt_geoms.append(geom)

        boxes = kwimage.Boxes.concatenate(alt_boxes)
        polys = kwimage.PolygonList(alt_polys)

        box_width =  boxes.width.ravel()
        box_height = boxes.height.ravel()

        box_canvas_width = np.array(annots.images.get('width'))
        box_canvas_height = np.array(annots.images.get('height'))

        # geoms = [p.to_shapely() for p in polys]
        geoms = alt_geoms

        perannot_data = _DataFrame({
            'geometry': geoms,
            'annot_id': annots.ids,
            'image_id': annots.image_id,
            'box_rt_area': np.sqrt(boxes.area.ravel()),
            'box_width': box_height,
            'box_height': box_height,
            'rel_box_width': box_width / box_canvas_width,
            'rel_box_height': box_height / box_canvas_height,
        })
        perannot_data['num_vertices'] = perannot_data.geometry.apply(geometry_length)

        sorted_box_rt_area = perannot_data['box_rt_area'].sort_values()
        mean_box_rt_area_idx = sorted_box_rt_area.index[len(sorted_box_rt_area) // 2]
        scalar_stats['median_box_rt_area'] = perannot_data.loc[mean_box_rt_area_idx, 'box_rt_area']
        scalar_stats['median_box_dsize'] = tuple(map(float, perannot_data.loc[mean_box_rt_area_idx, ['box_width', 'box_height']]))

        try:
            perannot_data = polygon_shape_stats(perannot_data)
        except Exception as ex:
            print(f'ERROR: ex={ex}')
            raise
        else:
            sorted_sseg_rt_area = perannot_data['sseg_rt_area'].sort_values()
            mean_sseg_rt_area_idx = sorted_sseg_rt_area.index[len(sorted_sseg_rt_area) // 2]
            scalar_stats['median_sseg_rt_area'] = perannot_data.loc[mean_sseg_rt_area_idx, 'sseg_rt_area']
            scalar_stats['median_sseg_box_dsize'] = tuple(map(float, perannot_data.loc[mean_box_rt_area_idx, ['box_width', 'box_height']]))
            scalar_stats['median_sseg_obox_dsize'] = tuple(map(float, perannot_data.loc[mean_box_rt_area_idx, ['obox_major', 'obox_minor']]))

        geometry = perannot_data['geometry']
        perannot_data['centroid_x'] = geometry.apply(lambda s: s.centroid.x)
        perannot_data['centroid_y'] = geometry.apply(lambda s: s.centroid.y)
        perannot_data['rel_centroid_x'] = perannot_data['centroid_x'] / box_canvas_width
        perannot_data['rel_centroid_y'] = perannot_data['centroid_y'] / box_canvas_height

        _summary_data = ub.udict(perannot_data.to_dict()) - {'geometry'}
        _summary_df = pd.DataFrame(_summary_data)
        tables_data = {}
        tables_data['perannot_data'] = json.loads(_summary_df.to_json(orient='table'))
        tables_data['perimage_data'] = json.loads(perimage_data.to_json(orient='table'))

        # Data that we don't serialize, but some plots depend on.
        nonsaved_data = {
            'boxes': boxes,
            'polys': polys,
        }

        dataframes = {
            'perannot_data': perannot_data,
            'perimage_data': perimage_data,
        }

    scalar_stats['table_buildtime_seconds'] = stats_timer.elapsed
    # if 0:
    #     import io
    #     import pandas as pd
    #     perimage_data = pd.read_json(io.StringIO(json.dumps(tables_data['perimage_data'])), orient='table')
    #     perannot_data = pd.read_json(io.StringIO(json.dumps(tables_data['perannot_data'])), orient='table')
    return scalar_stats, tables_data, nonsaved_data, dataframes


class Plots:
    """
    Defines plot functions as a class, to make it easier to share common
    data and write the functions in a more concise manner. This also
    makes it easier to enable / disable plots.
    """

    _plot_function_registery = {}

    @classmethod
    def demo(cls, **kwargs):
        """
        Helper for tweaking visualizations

        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> self = Plots.demo(timestamps=True)
            >>> self.perimage_data
            >>> self.perannot_data
        """
        import kwcoco
        dpath = ub.Path.appdir('kwcoco/demo/vis_stats').ensuredir()
        src = kwcoco.CocoDataset.demo('vidshapes8', **kwargs)
        plots = prep_plots(src, plots_dpath=dpath)
        return plots

    @profile
    def __init__(self, plots_dpath, tables_data, nonsaved_data, dpi=300):
        import kwplot
        import pandas as pd
        import json
        import inspect
        import io

        # kwplot 0.5.3 doesnt provide util_seaborn access via lazy loading
        # work around this
        import kwplot.util_seaborn

        self.plots_dpath = plots_dpath
        self.tables_data = tables_data
        self.nonsaved_data = nonsaved_data
        self.dpi = dpi

        sns = kwplot.autosns(verbose=3)

        if nonsaved_data is not None:
            self.polys = nonsaved_data['polys']
            boxes = nonsaved_data['boxes']
            self.annot_max_x = boxes.br_x.max()
            self.annot_max_y = boxes.br_y.max()

        self.perannot_data = pd.read_json(io.StringIO(json.dumps(tables_data['perannot_data'])), orient='table')
        self.perimage_data = pd.read_json(io.StringIO(json.dumps(tables_data['perimage_data'])), orient='table')
        self.max_anns_per_image = self.perimage_data['anns_per_image'].max()
        self.plot_functions = {}

        figman = kwplot.FigureManager(
            dpath=plots_dpath,
            dpi=dpi,
            verbose=1
        )
        # define label mappings for humans
        figman.labels.add_mapping({
            'pd_datetime': 'Datetime',
            # 'collection_size': '',
            'num_vertices': 'Num Polygon Vertices',
            'centroid_x': 'Polygon Centroid X',
            'centroid_y': 'Polygon Centroid Y',
            'obox_major': 'OBox Major Axes Length',
            'obox_minor': 'OBox Minor Axes Length',
            'sseg_rt_area': 'Polygon sqrt(Area)'
        })
        self.figman = figman
        self.sns = sns
        self.kwplot = kwplot
        self.perimage_data = self.perimage_data

        # Might want to modify to make this nicer for interactive / reloading
        # use cases
        unbound_methods = inspect.getmembers(BuiltinPlots, predicate=inspect.isfunction)
        for name, func in unbound_methods:
            self.register(func)

    def __getitem__(self, key):
        return self.plot_functions[key]

    def register(self, func):
        key = func.__name__
        if key in self.plot_functions:
            raise AssertionError('duplicate plot name')
        self.plot_functions[key] = func

    def run(self, plot_keys):
        from kwcoco.util.util_rich import rich_print
        import kwutil
        pman = kwutil.util_progress.ProgressManager()
        with pman:
            # plot_func_keys = [
            #     # 'images_over_time',
            #     'images_timeofday_distribution',
            # ]
            if plot_keys is None:
                plot_keys = list(self.plot_functions.keys())
            for key in pman.ProgIter(plot_keys, desc='plot'):
                func = self.plot_functions[key]
                try:
                    func(self)
                except Exception as ex:
                    rich_print(f'[red] ERROR: in {func}')
                    rich_print(f'ex = {ub.urepr(ex, nl=1)}')
                    import traceback
                    traceback.print_exc()
                    if 0:
                        raise


class BuiltinPlots:
    """
    A class that ONLY contains methods that will produce a plot.
    This is used to register them with :class:`Plots`.
    """

    def polygon_centroid_absolute_distribution(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
            >>> self = Plots.demo()
            >>> self['polygon_centroid_absolute_distribution'](self)
        """
        ax = self.figman.figure(fnum=1, doclf=True).gca()
        self.sns.kdeplot(data=self.perannot_data, x='centroid_x', y='centroid_y', ax=ax)
        self.sns.scatterplot(data=self.perannot_data, x='centroid_x', y='centroid_y', ax=ax, hue='sseg_rt_area', alpha=0.8)
        ax.set_aspect('equal')
        ax.set_title('Polygon Absolute Centroid Positions')
        #ax.set_xlim(0, max_width)
        #ax.set_ylim(0, max_height)
        ax.set_xlim(0, ax.get_xlim()[1])
        ax.set_ylim(0, ax.get_ylim()[1])
        self.figman.labels.relabel(ax)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        self.figman.finalize('polygon_centroid_absolute_distribution.png')

    def polygon_centroid_relative_distribution(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
            >>> self = Plots.demo()
            >>> self['polygon_centroid_relative_distribution'](self)
        """
        ax = self.figman.figure(fnum=1, doclf=True).gca()
        self.sns.kdeplot(data=self.perannot_data, x='rel_centroid_x', y='rel_centroid_y', ax=ax)
        self.sns.scatterplot(data=self.perannot_data, x='rel_centroid_x', y='rel_centroid_y', ax=ax, hue='sseg_rt_area', alpha=0.8)
        ax.set_aspect('equal')
        ax.set_title('Polygon Relative Centroid Positions')
        #ax.set_xlim(0, max_width)
        #ax.set_ylim(0, max_height)
        ax.set_xlim(0, ax.get_xlim()[1])
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_xlabel('Polygon X Centroid')
        ax.set_ylabel('Polygon Y Centroid')
        self.figman.labels.relabel(ax)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        self.figman.finalize('polygon_centroid_relative_distribution.png')

    def polygon_centroid_absolute_distribution_jointplot(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
            >>> self = Plots.demo(timestamps=True)
            >>> self['polygon_centroid_absolute_distribution_jointplot'](self)
        """
        ax = self.figman.figure(fnum=1, doclf=True).gca()
        with self.kwplot.util_seaborn.MonkeyPatchPyPlotFigureContext(self.figman.fig):
            self.figman.fig.clf()
            marginal_kws = dict()
            joint_kws = dict()
            # marginal_kws['log_scale'] = True
            # joint_kws['log_scale'] = True
            jointplot_kws = dict(
                joint_kws=joint_kws,
                marginal_kws=marginal_kws,
                kind='hist',
            )
            self.sns.jointplot(data=self.perannot_data, x='centroid_x', y='centroid_y', **jointplot_kws)
            ax = self.figman.fig.gca()
        ax.set_title('Polygon Absolute Centroid Positions')
        #ax.set_xlim(0, max_width)
        #ax.set_ylim(0, max_height)
        ax.set_xlim(0, ax.get_xlim()[1])
        ax.set_ylim(0, ax.get_ylim()[1])
        self.figman.labels.relabel(ax)
        # ax.set_aspect('equal')
        ax.invert_yaxis()
        self.figman.finalize('polygon_centroid_absolute_distribution_jointplot.png')

    def polygon_centroid_relative_distribution_jointplot(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
            >>> self = Plots.demo(timestamps=True)
            >>> self['polygon_centroid_relative_distribution_jointplot'](self)
        """
        ax = self.figman.figure(fnum=1, doclf=True).gca()

        # self.sns.kdeplot(data=self.perannot_data, x='rel_centroid_x', y='rel_centroid_y', ax=ax)
        # self.sns.scatterplot(data=self.perannot_data, x='rel_centroid_x', y='rel_centroid_y', ax=ax, hue='sseg_rt_area', alpha=0.8)

        ax = self.figman.figure(fnum=1, doclf=True).gca()
        with self.kwplot.util_seaborn.MonkeyPatchPyPlotFigureContext(self.figman.fig):
            self.figman.fig.clf()
            marginal_kws = dict()
            joint_kws = dict()
            # marginal_kws['log_scale'] = True
            # joint_kws['log_scale'] = True
            jointplot_kws = dict(
                joint_kws=joint_kws,
                marginal_kws=marginal_kws,
                kind='hist',
            )
            self.sns.jointplot(data=self.perannot_data, x='rel_centroid_x', y='rel_centroid_y', **jointplot_kws)
            ax = self.figman.fig.gca()
            # self.sns.kdeplot(data=self.perannot_data, x='obox_major', y='obox_minor', ax=ax)
        ax.set_title('Polygon Relative Centroid Positions')
        ax.set_xlim(0, ax.get_xlim()[1])
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_xlabel('Polygon X Centroid')
        ax.set_ylabel('Polygon Y Centroid')
        self.figman.labels.relabel(ax)
        # ax.set_aspect('equal')
        ax.invert_yaxis()
        self.figman.finalize('polygon_centroid_relative_distribution_jointplot.png')

    def image_size_histogram(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
            >>> self = Plots.demo(timestamps=True, image_size='random', num_frames=100)
            >>> self['image_size_histogram'](self)
        """
        import kwplot
        # self.figman.figure(fnum=1, doclf=True).gca()
        img_dsizes = [f'{w}✕{h}' for w, h in zip(self.perimage_data['width'], self.perimage_data['height'])]
        self.perimage_data['img_dsizes'] = img_dsizes
        # self.sns.histplot(data=perimage_data, x='img_dsizes', ax=ax)
        data = self.perimage_data
        x = 'img_dsizes'
        snskw = dict(binwidth=1, discrete=True)
        ax_top, ax_bottom, split_y = kwplot.util_seaborn.histplot_splity(
            data=data, x=x, **snskw)
        ax_bottom.set_xlabel('Image Width ✕ Height')
        ax_bottom.set_ylabel('Number of Images')
        ax_top.set_title('Image Size Histogram')
        self.figman.finalize('image_size_histogram.png', fig=ax_top.figure)

    def image_size_scatter(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
            >>> self = Plots.demo(timestamps=True, image_size='random', num_frames=100)
            >>> self['image_size_scatter'](self)
        """
        ax = self.figman.figure(fnum=1, doclf=True).gca()
        # self.sns.kdeplot(data=self.perimage_data, x='width', y='height', ax=ax)
        # self.sns.stripplot(data=self.perimage_data, x='width', y='height', ax=ax)
        self.sns.kdeplot(data=self.perimage_data, x='width', y='height', ax=ax)
        self.sns.scatterplot(data=self.perimage_data, x='width', y='height', ax=ax)
        # self.sns.swarmplot(data=self.perimage_data, x='width', y='height', ax=ax)
        ax.set_title('Image Size Distribution')
        # ax.set_aspect('equal')
        ax.set_ylabel('Image Height')
        ax.set_xlabel('Image Width')
        # ax.set_xlim(0, ax.get_xlim()[1])
        # ax.set_ylim(0, ax.get_ylim()[1])
        self.figman.labels.relabel(ax)
        self.figman.finalize('image_size_scatter.png')

    def obox_size_distribution(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
            >>> self = Plots.demo()
            >>> self['obox_size_distribution'](self)
        """
        ax = self.figman.figure(fnum=1, doclf=True).gca()
        self.sns.kdeplot(data=self.perannot_data, x='obox_major', y='obox_minor', ax=ax)
        self.sns.scatterplot(data=self.perannot_data, x='obox_major', y='obox_minor', ax=ax)
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        ax.set_title('Oriented Bounding Box Sizes')
        ax.set_aspect('equal')
        ax.set_xlim(0, ax.get_xlim()[1])
        ax.set_ylim(0, ax.get_ylim()[1])
        self.figman.labels.relabel(ax)
        self.figman.finalize('obox_size_distribution.png')

    def obox_size_distribution_jointplot(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
            >>> self = Plots.demo()
            >>> self['obox_size_distribution_jointplot'](self)
        """
        ax = self.figman.figure(fnum=1, doclf=True).gca()
        with self.kwplot.util_seaborn.MonkeyPatchPyPlotFigureContext(self.figman.fig):
            self.figman.fig.clf()
            marginal_kws = dict()
            joint_kws = dict()
            # marginal_kws['log_scale'] = True
            # joint_kws['log_scale'] = True
            jointplot_kws = dict(
                joint_kws=joint_kws,
                marginal_kws=marginal_kws,
                kind='hist',
            )
            self.sns.jointplot(data=self.perannot_data, x='obox_major', y='obox_minor', **jointplot_kws)
            ax = self.figman.fig.gca()
            # self.sns.kdeplot(data=self.perannot_data, x='obox_major', y='obox_minor', ax=ax)
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        ax.set_title('Oriented Bounding Box Sizes')
        # ax.set_aspect('equal')
        ax.set_xlim(0, ax.get_xlim()[1])
        ax.set_ylim(0, ax.get_ylim()[1])
        self.figman.labels.relabel(ax)
        self.figman.finalize('obox_size_distribution_jointplot.png')

    def obox_size_distribution_logscale(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
            >>> self = Plots.demo()
            >>> self['obox_size_distribution_logscale'](self)
        """
        ax = self.figman.figure(fnum=1, doclf=True).gca()
        # self.sns.kdeplot(data=self.perannot_data, x='obox_major', y='obox_minor', ax=ax)
        # self.sns.scatterplot(data=self.perannot_data, x='obox_major', y='obox_minor', ax=ax)

        with self.kwplot.util_seaborn.MonkeyPatchPyPlotFigureContext(self.figman.fig):
            # TODO: logscale on boxes
            self.figman.fig.clf()
            marginal_kws = dict()
            joint_kws = dict()
            marginal_kws['log_scale'] = True
            joint_kws['log_scale'] = True
            jointplot_kws = dict(
                joint_kws=joint_kws,
                marginal_kws=marginal_kws,
                kind='hist',
                # kind='hex',
                # kind='scatter',
                # hue='num_vertices'
            )
            self.sns.jointplot(data=self.perannot_data, x='obox_major', y='obox_minor', **jointplot_kws)
            ax = self.figman.fig.gca()

        ax.set_xscale('symlog')
        ax.set_yscale('symlog')
        ax.set_title('Oriented Bounding Box Sizes')
        # ax.set_aspect('equal')
        # TODO: set better min
        minx = self.perannot_data['obox_major'].min()
        miny = self.perannot_data['obox_minor'].min()
        ax.set_xlim(minx, ax.get_xlim()[1])
        ax.set_ylim(miny, ax.get_ylim()[1])
        self.figman.labels.relabel(ax)
        self.figman.finalize('obox_size_distribution_logscale.png')

    def polygon_area_vs_num_verts(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
            >>> self = Plots.demo()
            >>> self['polygon_area_vs_num_verts'](self)
        """
        ax = self.figman.figure(fnum=1, doclf=True).gca()
        self.sns.kdeplot(data=self.perannot_data, x='sseg_rt_area', y='num_vertices', ax=ax)
        self.sns.scatterplot(data=self.perannot_data, x='sseg_rt_area', y='num_vertices', ax=ax)
        # self.sns.jointplot(data=self.perannot_data, x='sseg_rt_area', y='num_vertices')
        self.figman.labels.relabel(ax)
        ax.set_xlim(0, ax.get_xlim()[1])
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_title('Polygon Area vs Num Vertices')
        self.figman.finalize('polygon_area_vs_num_verts.png')

    def polygon_area_vs_num_verts_jointplot(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
            >>> self = Plots.demo()
            >>> self['polygon_area_vs_num_verts_jointplot'](self)
        """
        ax = self.figman.figure(fnum=1, doclf=True).gca()
        with self.kwplot.util_seaborn.MonkeyPatchPyPlotFigureContext(self.figman.fig):
            self.figman.fig.clf()
            marginal_kws = dict()
            joint_kws = dict()
            # marginal_kws['log_scale'] = True
            # joint_kws['log_scale'] = True
            jointplot_kws = dict(
                joint_kws=joint_kws,
                marginal_kws=marginal_kws,
                kind='hist',
                # kind='scatter',
            )
            self.sns.jointplot(data=self.perannot_data, x='sseg_rt_area', y='num_vertices', **jointplot_kws)
            ax = self.figman.fig.gca()
        # self.sns.kdeplot(data=self.perannot_data, x='sseg_rt_area', y='num_vertices', ax=ax)
        self.figman.labels.relabel(ax)

        minx = self.perannot_data['sseg_rt_area'].min()
        miny = self.perannot_data['num_vertices'].min()
        ax.set_xlim(minx, ax.get_xlim()[1])
        ax.set_ylim(miny, ax.get_ylim()[1])

        # ax.set_xlim(0, ax.get_xlim()[1])
        # ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_title('Polygon Area vs Num Vertices')
        self.figman.finalize('polygon_area_vs_num_verts_jointplot.png')

    def polygon_area_vs_num_verts_jointplot_logscale(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
            >>> self = Plots.demo(timestamps=True)
            >>> self['polygon_area_vs_num_verts_jointplot_logscale'](self)
        """
        ax = self.figman.figure(fnum=1, doclf=True).gca()

        x = 'sseg_rt_area'
        y = 'num_vertices'
        # Hack for logscale to work
        perannot_data = self.perannot_data.copy()
        perannot_data[perannot_data[y] == 0] = float('nan')
        perannot_data[perannot_data[x] == 0] = float('nan')
        maxx = perannot_data[x].max()
        maxy = perannot_data[y].max()
        minx = perannot_data[x].min()
        miny = perannot_data[y].min()

        with self.kwplot.util_seaborn.MonkeyPatchPyPlotFigureContext(self.figman.fig):
            self.figman.fig.clf()
            marginal_kws = dict()
            joint_kws = dict()
            marginal_kws['log_scale'] = True
            joint_kws['log_scale'] = True
            jointplot_kws = dict(
                joint_kws=joint_kws,
                marginal_kws=marginal_kws,
                kind='hist',
                # kind='scatter',
            )
            self.sns.jointplot(data=perannot_data, x=x, y=y, **jointplot_kws)
            ax = self.figman.fig.gca()
        self.figman.labels.relabel(ax)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        ax.set_title('Polygon Area vs Num Vertices')
        self.figman.finalize('polygon_area_vs_num_verts_jointplot_logscale.png')

    def polygon_area_histogram_logscale(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
            >>> self = Plots.demo()
            >>> self['polygon_area_histogram_logscale'](self)
        """
        ax = self.figman.figure(fnum=1, doclf=True).gca()
        self.sns.histplot(data=self.perannot_data, x='sseg_rt_area', ax=ax, kde=True, log_scale=True)
        # self.sns.histplot(data=self.perannot_data, x='sseg_rt_area', ax=ax, kde=True)
        self.figman.labels.relabel(ax)
        ax.set_title('Polygon sqrt(Area) Histogram')
        # TODO: better min. Figman needs a good way of helping with this.
        ax.set_xlim(10, ax.get_xlim()[1])
        ax.set_ylabel('Number of Annotations')
        # ax.set_yscale('symlog')
        # ax.set_xscale('symlog')
        self.figman.finalize('polygon_area_histogram_logscale.png')

    def polygon_area_histogram(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
            >>> self = Plots.demo()
            >>> self['polygon_area_histogram'](self)
        """
        ax = self.figman.figure(fnum=1, doclf=True).gca()
        # self.sns.histplot(data=self.perannot_data, x='sseg_rt_area', ax=ax, kde=True, binwidth=32, log_scale=True)
        self.sns.histplot(data=self.perannot_data, x='sseg_rt_area', ax=ax, kde=True)
        self.figman.labels.relabel(ax)
        ax.set_title('Polygon sqrt(Area) Histogram')
        ax.set_xlim(10, ax.get_xlim()[1])
        ax.set_ylabel('Number of Annotations')
        # ax.set_yscale('symlog')
        # ax.set_xscale('symlog')
        self.figman.finalize('polygon_area_histogram.png')

    def polygon_area_histogram_splity(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
            >>> self = Plots.demo()
            >>> self['polygon_area_histogram_splity'](self)
        """
        # ax = self.figman.figure(fnum=1, doclf=True).gca()
        # self.sns.histplot(data=self.perannot_data, x='sseg_rt_area', ax=ax, kde=True)
        # self.figman.labels.relabel(ax)
        # ax.set_title('Polygon sqrt(Area) Histogram')
        # ax.set_xlim(0, ax.get_xlim()[1])
        # ax.set_ylabel('Number of Annotations')
        # ax.set_yscale('symlog')
        # self.figman.finalize('polygon_area_histogram.png')

        split_y = 'auto'
        snskw = dict(binwidth=50, discrete=False, kde=True)
        ax_top, ax_bottom, split_y = self.kwplot.util_seaborn.histplot_splity(
            data=self.perannot_data, x='sseg_rt_area', split_y=split_y,
            **snskw)
        ax = ax_top
        fig = ax.figure

        ax.set_yscale('linear')
        ax_bottom.set_ylabel('Number of Annotations')
        ax_top.set_ylabel('')
        ax_top.set_title('Polygon sqrt(Area) Histogram')
        # ax_bottom.set_xlim(0 - 0.5, self.max_anns_per_image + 1.5)
        ax.set_xlim(0, ax.get_xlim()[1])

        ax_top.set_ylim(bottom=split_y)   # those limits are fake
        ax_bottom.set_ylim(0, split_y)

        # self.figman.labels.force_integer_ticks(axis='x', method='ticker', ax=ax_bottom)
        # self.figman.labels.force_integer_ticks(axis='x', method='maxn', ax=ax_bottom)
        self.figman.finalize('polygon_area_histogram_splity.png', fig=fig)

    def polygon_num_vertices_histogram(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
            >>> self = Plots.demo()
            >>> self['polygon_num_vertices_histogram'](self)
        """
        ax = self.figman.figure(fnum=1, doclf=True).gca()
        self.sns.histplot(data=self.perannot_data, x='num_vertices', ax=ax)
        ax.set_title('Polygon Number of Vertices Histogram')
        ax.set_ylabel('Number of Annotations')
        ax.set_xlim(0, ax.get_xlim()[1])
        ax.set_yscale('linear')
        self.figman.labels.relabel(ax)
        self.figman.finalize('polygon_num_vertices_histogram.png')

    def anns_per_image_histogram(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
            >>> self = Plots.demo()
            >>> self['anns_per_image_histogram'](self)
        """
        ax = self.figman.figure(fnum=1, doclf=True).gca()
        self.sns.histplot(data=self.perimage_data, x='anns_per_image', ax=ax, binwidth=1, discrete=True)
        ax.set_yscale('linear')
        ax.set_xlabel('Number of Annotations')
        ax.set_ylabel('Number of Images')
        ax.set_title('Number of Annotations per Image')
        ax.set_xlim(0 - 0.5, self.max_anns_per_image + 1.5)
        self.figman.labels.relabel(ax)
        # ax.set_yscale('symlog', linthresh=10)
        self.figman.labels.force_integer_xticks()
        self.figman.finalize('anns_per_image_histogram.png')

    def anns_per_image_histogram_splity(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
            >>> self = Plots.demo()
            >>> self['anns_per_image_histogram_splity'](self)
        """
        split_y = 'auto'
        snskw = dict(binwidth=1, discrete=True)
        ax_top, ax_bottom, split_y = self.kwplot.util_seaborn.histplot_splity(
            data=self.perimage_data, x='anns_per_image', split_y=split_y,
            **snskw
        )
        ax = ax_top
        fig = ax.figure

        ax.set_yscale('linear')
        ax_bottom.set_xlabel('Number of Annotations')
        ax_bottom.set_ylabel('Number of Images')
        ax_top.set_ylabel('')
        ax_top.set_title('Number of Annotations per Image')
        ax_bottom.set_xlim(0 - 0.5, self.max_anns_per_image + 1.5)

        ax_top.set_ylim(bottom=split_y)   # those limits are fake
        ax_bottom.set_ylim(0, split_y)

        # self.figman.labels.force_integer_ticks(axis='x', method='ticker', ax=ax_bottom)
        self.figman.labels.force_integer_ticks(axis='x', method='maxn', ax=ax_bottom)
        self.figman.finalize('anns_per_image_histogram_splity.png', fig=fig)

    def anns_per_image_histogram_ge1(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
            >>> self = Plots.demo()
            >>> self['anns_per_image_histogram_ge1'](self)
        """
        ax = self.figman.figure(fnum=1, doclf=True).gca()
        perimage_data = self.perimage_data
        perimage_ge1_data = perimage_data[perimage_data['anns_per_image'] >= 1]
        self.sns.histplot(data=perimage_ge1_data, x='anns_per_image', ax=ax, binwidth=1, discrete=True)
        ax.set_yscale('linear')
        ax.set_xlabel('Number of Annotations')
        ax.set_ylabel('Number of Images')
        ax.set_title('Number of Annotations per Image\n(with at least 1 annotation)')
        self.figman.labels.relabel(ax)
        ax.set_xlim(1 - 0.5, self.max_anns_per_image + 0.5)
        # self.figman.labels.force_integer_ticks(axis='x', method='maxn', ax=ax)
        self.figman.labels.force_integer_ticks(axis='x', method='ticker', ax=ax, hack_labels=0)
        # ax.set_xticks(ax.get_xticks().astype(int))

        # ax.set_yscale('symlog', linthresh=10)
        self.figman.finalize('anns_per_image_histogram_ge1.png')

    def images_over_time(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
            >>> self = Plots.demo(timestamps=True)
            >>> self['images_over_time'](self)
        """
        import pandas as pd
        import numpy as np
        img_df = self.perimage_data.sort_values('datetime')
        img_df['pd_datetime'] = pd.to_datetime(img_df.datetime)
        img_df['collection_size'] = np.arange(1, len(img_df) + 1)
        ax = self.figman.figure(fnum=1, doclf=True).gca()
        self.sns.histplot(data=img_df, x='pd_datetime', ax=ax, cumulative=True)
        # self.sns.lineplot(data=img_df, x='pd_datetime', y='collection_size')
        ax.set_title('Images collected over time')
        ax.set_xlabel('Datetime')
        ax.set_ylabel('Number of Images Collected')
        self.figman.finalize('images_over_time.png')

    def images_timeofday_distribution(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
            >>> self = Plots.demo(timestamps=True)
            >>> self['images_timeofday_distribution'](self)
        """
        import pandas as pd
        import numpy as np
        import kwutil
        import kwimage
        import kwplot
        img_df = self.perimage_data.sort_values('datetime')
        img_df['pd_datetime'] = pd.to_datetime(img_df.datetime)
        img_df['collection_size'] = np.arange(1, len(img_df) + 1)
        datetimes = [kwutil.datetime.coerce(x) for x in img_df['datetime']]
        # img_df['timestamp'] = [x.timestamp() for x in datetimes]
        # img_df['date'] = [x.date() for x in datetimes]
        # img_df['year_month'] = [x.strftime('%Y-%m') for x in datetimes]
        # img_df['month'] = [x.strftime('%m') for x in datetimes]
        img_df['time'] = [x.time() if not pd.isnull(x) else None for x in datetimes]
        img_df['day_of_year'] = [x.timetuple().tm_yday if not pd.isnull(x) else None for x in datetimes]
        img_df['hour_of_day'] = [None if z is None else z.hour + z.minute / 60 + z.second / 3600 for z in img_df['time']]

        self.snskw = {}
        has_sunlight = 'sunlight' in img_df.columns
        if has_sunlight and not img_df['sunlight'].isna().all():
            palette = self.sns.color_palette("flare", n_colors=4, as_cmap=True).reversed()
            self.snskw['hue'] = 'sunlight'
            self.snskw['palette'] = palette

        ax = self.figman.figure(fnum=1, doclf=True).gca()
        # self.sns.histplot(data=img_df, x='month', ax=ax)
        # self.sns.kdeplot(data=img_df, x='day_of_year', y='hour_of_day')
        # self.sns.scatterplot(data=img_df, x='day_of_year', y='hour_of_day', hue='sunlight_values')
        self.sns.scatterplot(data=img_df, x='day_of_year', y='hour_of_day')
        self.sns.scatterplot(data=img_df, x='day_of_year', y='hour_of_day', **self.snskw, legend=False)
        # self.sns.kdeplot(data=img_df, x='hour_of_day', y='day_of_year')
        if has_sunlight:
            kwplot.phantom_legend({
                'Night': kwimage.Color.coerce(palette.colors[0]).as255(),
                'Day': kwimage.Color.coerce(palette.colors[-1]).as255(),
                'nan': 'blue',
            }, mode='circle')
        ax.set_title('Time Captured')
        ax.set_xlabel('Day of Year')
        ax.set_ylabel('Hour of Day')
        self.figman.finalize('images_timeofday_distribution.png')

    def all_polygons(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwutil)
            >>> from kwcoco.cli.coco_plot_stats import *  # NOQA
            >>> self = Plots.demo(timestamps=True)
            >>> self['all_polygons'](self)

        Ignore:
            https://drive.google.com/file/d/1mUzJw4QrDfxWqqCsPZ_C7QWcQbfR_IBb/view
            ~/code/kwimage/kwimage/_im_color_data.py
        """
        import kwimage
        ax = self.figman.figure(fnum=1, doclf=True).gca()
        # edgecolor = kwimage.Color.coerce('kitware_darkblue').as01()
        # facecolor = kwimage.Color.coerce('kitware_green').as01()
        # edgecolor = kwimage.Color.coerce('white').as01()
        # facecolor = kwimage.Color.coerce('kitware_darkblue').as01()
        # edgecolor = kwimage.Color.coerce('almost black').as01()
        edgecolor = kwimage.Color.coerce('kitware_darkgray').as01()
        # facecolor = kwimage.Color.coerce('kitware_green').as01()
        facecolor = kwimage.Color.coerce('#b1dfaa').as01()
        self.polys.draw(alpha=0.5, edgecolor=edgecolor, facecolor=facecolor)
        ax.set_xlabel('Image X Coordinate')
        ax.set_ylabel('Image Y Coordinate')
        ax.set_title(f'All {len(self.polys)} Polygons')
        ax.set_aspect('equal')
        ax.set_xlim(0, self.annot_max_x)
        ax.set_ylim(0, self.annot_max_y)
        self.figman.labels.relabel(ax)
        ax.set_ylim(0, self.annot_max_y)  # not sure why this needs to be after the relabel, should ideally fix that.
        ax.invert_yaxis()
        self.figman.finalize('all_polygons.png', tight_layout=0)  # tight layout seems to cause issues here


def polygon_shape_stats(df):
    """
    Compute shape statistics about a geopandas dataframe (assume UTM CRS)

    TODO:
        - [ ] Use geopandas if available, but fallback to non-geopandas logic
    """
    import numpy as np
    import kwimage
    geometry = df['geometry']
    def hull_area(s):
        try:
            return s.convex_hull.area
        except Exception:
            return np.nan
    def sseg_area(s):
        try:
            return s.area
        except Exception:
            return np.nan
    df['hull_rt_area'] = np.sqrt(geometry.apply(hull_area))
    df['sseg_rt_area'] = np.sqrt(geometry.apply(sseg_area))

    obox_majors = []
    obox_minors = []
    for s in df.geometry:
        try:
            obox = kwimage.MultiPolygon.from_shapely(s).oriented_bounding_box()
            wh = obox.extent
            obox_majors.append(max(wh))
            obox_minors.append(min(wh))
        except Exception:
            obox_majors.append(np.nan)
            obox_minors.append(np.nan)
    df['obox_major'] = np.array(obox_majors)
    df['obox_minor'] = np.array(obox_minors)
    df['major_obox_ratio'] = df['obox_major'] / df['obox_minor']

    # df['ch_aspect_ratio'] =
    # df['isoperimetric_quotient'] = df.geometry.apply(shapestats.ipq)
    # df['boundary_amplitude'] = df.geometry.apply(shapestats.compactness.boundary_amplitude)
    # df['eig_seitzinger'] = df.geometry.apply(shapestats.compactness.eig_seitzinger)
    return df


def geometry_flatten(geom):
    """
    References:
        https://gis.stackexchange.com/questions/119453/count-the-number-of-points-in-a-multipolygon-in-shapely
    """
    if hasattr(geom, 'geoms'):  # Multi<Type> / GeometryCollection
        for g in geom.geoms:
            yield from geometry_flatten(g)
    elif hasattr(geom, 'interiors'):  # Polygon
        yield geom.exterior
        yield from geom.interiors
    else:  # Point / LineString
        yield geom


def geometry_length(geom):
    return sum(len(g.coords) for g in geometry_flatten(geom))


if __name__ == '__main__':
    r"""

    CommandLine:
        LINE_PROFILE=1 python -m kwcoco.cli.coco_plot_stats $HOME/data/dvc-repos/kwcoco/data.kwcoco.json \
            --dst_fpath $HOME/code/kwcoco/coco_annot_stats/stats.json \
            --dst_dpath $HOME/code/kwcoco/coco_annot_stats
    """
    __cli__.main()
