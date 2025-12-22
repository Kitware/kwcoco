"""
Compute intensity histograms of the underlying images in this dataset.

TODO:

    - [ ] Migrate to kwcoco proper

    - [ ] Fix accumulation of floating point values

    - [ ] Handle nodata

    - [x] Rename to coco_spectra

CommandLine:
    geowatch spectra geowatch-msi --dst foo.png
    geowatch spectra --src special:geowatch-msi --show=True --stat=density
    geowatch spectra --src special:photos --show=True --fill=False
    geowatch spectra --src special:shapes8 --show=True --stat=count --cumulative=True --multiple=stack

    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
    geowatch spectra --src $DVC_DATA_DPATH/Drop6/data.kwcoco.zip --channels='red|green|blue|nir' --workers=11

    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    KWCOCO_FPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01/combo_L_nowv_vali.kwcoco.json
    geowatch spectra --src $KWCOCO_FPATH --show=True --show=True --include_channels="forest|water|bare_ground"
"""
import ubelt as ub
import scriptconfig as scfg
import math


class CocoSpectraConfig(scfg.DataConfig):
    """
    Plot the spectrum of band intensities in a kwcoco file.
    """
    __default__ = {
        'src': scfg.Value('data.kwcoco.json', help='input coco dataset', position=1),

        'dst': scfg.Value(None, help='if specified dump the figure to disk at this file path (e.g. with a jpg or png suffix)'),

        'cache_dpath': scfg.Value(None, help='if specified, read/write cached stats files from here instead of as sidecar files'),

        'show': scfg.Value(False, isflag=True, help='if True, do a plt.show()'),
        'draw': scfg.Value(True, isflag=True, help='if False disables all visualization and just print tables'),

        'workers': scfg.Value(0, help='number of io workers'),
        'mode': scfg.Value('process', help='type of parallelism'),

        'include_channels': scfg.Value(None, help='if specified can be | separated valid channels', alias=['channels']),
        'exclude_channels': scfg.Value(None, help='if specified can be | separated invalid channels'),

        'include_sensors': scfg.Value(None, help='if specified can be comma separated valid sensors'),
        'exclude_sensors': scfg.Value(None, help='if specified can be comma separated invalid sensors'),

        'valid_range': scfg.Value(None, help='Only include values within this range; specified as <min_val>:<max_val> e.g. (0:10000)'),

        'title': scfg.Value(None, help='Provide a title for the histogram figure'),

        'max_images': scfg.Value(None, help='if given only sample this many images when computing statistics', group='subset'),

        'select_images': scfg.Value(
            None, type=str, help=ub.paragraph(
                '''
                A json query (via the jq spec) that specifies which images
                belong in the subset. Note, this is a passed as the body of
                the following jq query format string to filter valid ids
                '.images[] | select({select_images}) | .id'.

                Examples for this argument are as follows:
                '.id < 3' will select all image ids less than 3.
                '.file_name | test(".*png")' will select only images with
                file names that end with png.
                '.file_name | test(".*png") | not' will select only images
                with file names that do not end with png.
                '.myattr == "foo"' will select only image dictionaries
                where the value of myattr is "foo".
                '.id < 3 and (.file_name | test(".*png"))' will select only
                images with id less than 3 that are also pngs.
                .myattr | in({"val1": 1, "val4": 1}) will take images
                where myattr is either val1 or val4.

                Requries the "jq" python library is installed.
                '''), group='subset'),

        'select_videos': scfg.Value(
            None, help=ub.paragraph(
                '''
                A json query (via the jq spec) that specifies which videos
                belong in the subset. Note, this is a passed as the body of
                the following jq query format string to filter valid ids
                '.videos[] | select({select_images}) | .id'.

                Examples for this argument are as follows:
                '.name | startswith("foo")' will select only videos
                where the name starts with foo.

                Only applicable for dataset that contain videos.

                Requries the "jq" python library is installed.
                '''), group='subset'),


        # Histogram modifiers
        'kde': scfg.Value(True, group='histplot', help='if True compute a kernel density estimate to smooth the distribution'),
        'cumulative': scfg.Value(False, group='histplot', help='If True, plot the cumulative counts as bins increase.'),

        # 'bins': scfg.Value(256, help='Generic bin parameter that can be the name of a reference rule or the number of bins.'),
        'bins': scfg.Value('auto', group='histplot', help='Generic bin parameter that can be the name of a reference rule or the number of bins.'),

        'fill': scfg.Value(True, isflag=True, group='histplot', help='If True, fill in the space under the histogram.'),

        'element': scfg.Value('step', help='Visual representation of the histogram statistic.', group='histplot', choices=['bars', 'step', 'poly']),

        'multiple': scfg.Value('layer',
                               choices=['layer', 'dodge', 'stack', 'fill'],
                               group='histplot',
                               help='Approach to resolving multiple elements when semantic mapping creates subsets.'),

        'stat': scfg.Value('probability', choices={'count', 'frequency', 'density', 'probability'}, group='histplot', help=ub.paragraph(
            '''
            Aggregate statistic to compute in each bin.

            - ``count`` shows the number of observations
            - ``frequency`` shows the number of observations divided by the bin width
            - ``density`` normalizes counts so that the area of the histogram is 1
            - ``probability`` normalizes counts so that the sum of the bar heights is 1
            ''')),
    }


class HistAccum:
    """
    Helper to accumulate histograms
    """

    def __init__(self):
        self.accum = {}
        self.n = 0

    def update(self, data, sensor, channel):
        self.n += 1
        if sensor not in self.accum:
            self.accum[sensor] = {}
        # TODO: video / month
        sensor_accum = self.accum[sensor]
        if channel not in sensor_accum:
            sensor_accum[channel] = ub.ddict(lambda: 0)
        final_accum = sensor_accum[channel]
        for k, v in data.items():
            final_accum[k] += v

    def finalize(self):
        import pandas as pd
        import numpy as np
        # Stack all accuulated histograms into a longform dataframe
        to_stack = {}
        for sensor, sub in self.accum.items():
            for channel, hist in sub.items():
                hist = ub.sorted_keys(hist)
                # hist.pop(0)
                df = pd.DataFrame({
                    # 'intensity_bin': np.array(list(hist.keys()), dtype=int),
                    'intensity_bin': np.array(list(hist.keys())),
                    'value': np.array(list(hist.values())),
                    'channel': [channel] * len(hist),
                    'sensor': [sensor] * len(hist),
                })
                to_stack[(channel, sensor)] = df

        full_df = pd.concat(list(to_stack.values()))
        is_finite = np.isfinite(full_df.intensity_bin)
        num_nonfinite = (~is_finite).sum()
        if num_nonfinite > 0:
            import warnings
            warnings.warn('There were {} non-finite values'.format(num_nonfinite))
            full_df = full_df[is_finite]
        full_df = full_df.reset_index()
        return full_df


def main(cmdline=True, **kwargs):
    r"""
    CommandLine:
        XDEV_PROFILE=1 xdoctest -m geowatch.cli.coco_spectra main

    Example:
        >>> from geowatch.cli.coco_spectra import *  # NOQA
        >>> import kwcoco
        >>> test_dpath = ub.Path.appdir('geowatch/tests/cli/spectra').ensuredir()
        >>> image_fpath = test_dpath + '/intensityhist_demo.jpg'
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes-msi-multisensor-videos1-frames2-gsize8')
        >>> kwargs = {'src': coco_dset, 'dst': image_fpath, 'mode': 'thread'}
        >>> kwargs['multiple'] = 'layer'
        >>> kwargs['element'] = 'step'
        >>> kwargs['workers'] = 'avail'
        >>> kwargs['show'] = False
        >>> kwargs['draw'] = False
        >>> main(cmdline=False, **kwargs)

    Example:
        >>> # xdoctest: +REQUIRES(--slow)
        >>> from geowatch.cli.coco_spectra import *  # NOQA
        >>> import kwcoco
        >>> import geowatch
        >>> test_dpath = ub.Path.appdir('geowatch/tests/cli/spectra').ensuredir()
        >>> image_fpath = test_dpath + '/intensityhist_demo2.jpg'
        >>> coco_dset = geowatch.coerce_kwcoco('geowatch-msi')
        >>> kwargs = {
        >>>     'src': coco_dset,
        >>>     'dst': image_fpath,
        >>>     'mode': 'thread',
        >>>     'valid_range': '10:2000',
        >>>     'workers': 'avail',
        >>> }
        >>> kwargs['multiple'] = 'layer'
        >>> kwargs['element'] = 'step'
        >>> main(cmdline=False, **kwargs)
    """
    from geowatch.utils import kwcoco_extensions
    from kwutil import util_parallel
    import geowatch
    import kwcoco
    import numpy as np

    config = CocoSpectraConfig.cli(data=kwargs, cmdline=cmdline)
    print('config = {}'.format(ub.urepr(config.to_dict(), nl=1)))

    # coco_dset = kwcoco.CocoDataset.coerce(config['src'])
    coco_dset = geowatch.coerce_kwcoco(config['src'])

    valid_gids = kwcoco_extensions.filter_image_ids(
        coco_dset,
        include_sensors=config['include_sensors'],
        exclude_sensors=config['exclude_sensors'],
        select_images=config['select_images'],
        select_videos=config['select_videos'],
    )

    images = coco_dset.images(valid_gids)
    workers = util_parallel.coerce_num_workers(config['workers'])
    print('workers = {!r}'.format(workers))

    if config['max_images'] is not None:
        print('images = {!r}'.format(images))
        images = coco_dset.images(list(images)[:int(config['max_images'])])
        print('filter images = {!r}'.format(images))

    include_channels = config['include_channels']
    exclude_channels = config['exclude_channels']
    include_channels = None if include_channels is None else kwcoco.FusedChannelSpec.coerce(include_channels)
    exclude_channels = None if exclude_channels is None else kwcoco.FusedChannelSpec.coerce(exclude_channels)

    if config['valid_range'] is not None:
        valid_min, valid_max = map(float, config['valid_range'].split(':'))
    else:
        valid_min = -math.inf
        valid_max = math.inf

    class PeriodicCondition:
        """
        Helper that does something based on some periodic condition (e.g.
        number of seconds passed, or every n-th iteration)

        TODO:
            rectify with PeriodicMemoryMonitor in fusion.predict and port
            to kwutil
        """
        def __init__(self, seconds=10, do_first=True):
            self.timer = ub.Timer().tic()
            self.num_checks = 0
            self.do_first = do_first
            self.seconds = seconds

        def _satisfied(self):
            return (
                (self.timer.toc() > self.seconds) or
                (self.do_first and self.num_checks == 0)
            )

        def check(self):
            if self._satisfied():
                self.timer.tic()
                self.num_checks += 1
                return True
            return False

    jobs = ub.JobPool(mode=config['mode'], max_workers=workers, transient=True)
    from kwutil import util_progress
    pman = util_progress.ProgressManager()
    with pman:
        for coco_img in pman.progiter(images.coco_images, desc='submit stats jobs'):
            coco_img.detach()
            job = jobs.submit(ensure_intensity_stats, coco_img,
                              include_channels=include_channels,
                              exclude_channels=exclude_channels,
                              cache_dpath=config.cache_dpath,
                              bundle_dpath=coco_dset.bundle_dpath,
                              valid_min=valid_min,
                              valid_max=valid_max)
            job.coco_img = coco_img

        accum = HistAccum()

        show_seen_props = False
        if show_seen_props:
            seen_props = ub.ddict(set)

        # Update the report once every 10 seconds
        report_condition = PeriodicCondition(
            seconds=3,
            do_first=True,
        )

        for job in pman.progiter(jobs.as_completed(), total=len(jobs), desc='accumulate stats'):
            intensity_stats = job.result()
            sensor = job.coco_img.get('sensor_coarse', job.coco_img.get('sensor', 'unknown_sensor'))
            for band_stats in intensity_stats['bands']:
                intensity_hist = band_stats['intensity_hist']
                band_props = band_stats['properties']
                band_name = band_props['band_name']
                if show_seen_props:
                    for k, v in band_props.items():
                        seen_props[k].add(v)
                accum.update(intensity_hist, sensor, band_name)

            # Need to truncate repr...
            # should probably try to use geowatch.utils.util_kwutil.distributed_subitems
            # to reduce urepr overhead
            if show_seen_props:
                from kwutil.slugify_ext import smart_truncate
                seen_props_text = ub.urepr(seen_props, nl=1)
                seen_props_text = smart_truncate(seen_props_text, max_length=1600, head='\n~TRUNCATED...', tail='\n...~')
                pman.update_info(
                    ('seen_props = {}'.format(seen_props_text))
                )
            if report_condition.check():
                # TODO: can we compute a running average for efficiency
                # instead? As a workaround, only compute once every few seconds
                current_ave = single_persensor_table(accum.finalize())
                pman.update_info(
                    current_ave.to_string()
                    # ('seen_props = {}'.format(seen_props_text))
                )

        full_df = accum.finalize()
        sensor_chan_stats, distance_metrics = sensor_stats_tables(full_df)
        pman.update_info(
            sensor_chan_stats.to_string()
            # ('seen_props = {}'.format(seen_props_text))
        )

        COMPARSE_SENSORS = True
        if COMPARSE_SENSORS:
            request_columns = ['emd', 'energy_dist', 'mean_diff', 'std_diff']
            have_columns = list(ub.oset(request_columns) & ub.oset(distance_metrics.columns))
            harmony_scores = distance_metrics[have_columns].mean()
            harmony_scores = harmony_scores.to_dict()
            if harmony_scores:
                extra_text = ub.urepr(harmony_scores, precision=4, compact=1)
                print('extra_text = {!r}'.format(extra_text))
            else:
                extra_text = None
        else:
            extra_text = None

    if config['draw']:
        import kwplot
        kwplot.autosns()
        fig = plot_intensity_histograms(full_df, config)

        title_lines = []
        title = config.get('title', None)
        if title is not None:
            title_lines.append(title)

        title_lines.append(ub.Path(coco_dset.fpath).name)
        if extra_text is not None:
            title_lines.append(extra_text)

        final_title = '\n'.join(title_lines)
        fig.suptitle(final_title)

        dst_fpath = None if config['dst'] is None else ub.Path(config['dst'])
        if dst_fpath is not None:
            print('Write spectra viz to dst_fpath = {!r}'.format(dst_fpath))
            dst_fpath.parent.ensuredir()
            fig.set_size_inches(np.array([6.4, 4.8]) * 1.68)
            fig.tight_layout()
            fig.savefig(dst_fpath)

        if config['show']:
            from matplotlib import pyplot as plt
            plt.show()

    results = {
        'sensor_chan_stats': sensor_chan_stats,
        'distance_metrics': distance_metrics,
    }
    return results


def single_persensor_table(full_df):
    """
    Like sensor_stats_tables, but only used for intermediate stats

    Args:
        full_df (pd.DataFrame):
            with columns: index  intensity_bin  value channel   sensor
    """
    import pandas as pd
    import numpy as np
    sensor_channel_to_vwf = {}
    for _key, chan_df in full_df.groupby(['sensor', 'channel']):
        _sensor, channel = _key
        _values = chan_df['intensity_bin']
        _weights = chan_df['value']
        norm_weights = _weights / _weights.sum()
        sensor_channel_to_vwf[(_sensor, channel)] = {
            'raw_values': _values,
            'raw_weights': _weights,
            'norm_weights': norm_weights,
        }

    single_rows = []
    for sensor, channel in sensor_channel_to_vwf:
        sensor_data = sensor_channel_to_vwf[(sensor, channel)]
        values = sensor_data['raw_values']
        weights = sensor_data['raw_weights']

        # Note: the calculation of the variance depends on the type of
        # weighting we choose
        average = np.average(values, weights=weights)
        variance = np.average((values - average) ** 2, weights=weights)
        variance = variance * sum(weights) / (sum(weights) - 1)
        stddev = np.sqrt(variance)

        pytype = float if values.values.dtype.kind == 'f' else int

        info = {
            'min': pytype(values.min()),
            'max': pytype(values.max()),
            'mean': average,
            'std': stddev,
            'total_weight': weights.sum(),
            'channel': channel,
            'sensor': sensor,
        }
        assert info['max'] >= info['min']
        single_rows.append(info)

    sensor_chan_stats = pd.DataFrame(single_rows)
    sensor_chan_stats = sensor_chan_stats.set_index(['sensor', 'channel'])
    return sensor_chan_stats


def sensor_stats_tables(full_df):
    """
    Args:
        full_df (pd.DataFrame):
            with columns: index  intensity_bin  value channel   sensor
    """
    import itertools as it
    import scipy
    import kwarray
    import scipy.stats
    import pandas as pd
    import numpy as np
    sensor_channel_to_vwf = {}
    for _key, chan_df in full_df.groupby(['sensor', 'channel']):
        _sensor, channel = _key
        _values = chan_df['intensity_bin']
        _weights = chan_df['value']
        norm_weights = _weights / _weights.sum()
        sensor_channel_to_vwf[(_sensor, channel)] = {
            'raw_values': _values,
            'raw_weights': _weights,
            'norm_weights': norm_weights,
            'sensorchan_df': chan_df,
        }

    print('compute per-sensor stats')
    print(ub.urepr(list(sensor_channel_to_vwf.keys()), nl=1))
    single_rows = []
    for sensor, channel in ub.ProgIter(sensor_channel_to_vwf, desc='compute stats'):
        sensor_data = sensor_channel_to_vwf[(sensor, channel)]
        values = sensor_data['raw_values']
        weights = sensor_data['raw_weights']
        sensorchan_df = sensor_data['sensorchan_df']

        # Note: the calculation of the variance depends on the type of
        # weighting we choose
        average = np.average(values, weights=weights)
        variance = np.average((values - average) ** 2, weights=weights)
        variance = variance * sum(weights) / (sum(weights) - 1)
        stddev = np.sqrt(variance)

        pytype = float if values.values.dtype.kind == 'f' else int
        auto_bins = _weighted_auto_bins(
            sensorchan_df, 'intensity_bin', 'value')

        info = {
            'min': pytype(values.min()),
            'max': pytype(values.max()),
            'mean': average,
            'std': stddev,
            'total_weight': weights.sum(),
            'channel': channel,
            'sensor': sensor,
            'auto_bins': auto_bins,
        }
        assert info['max'] >= info['min']
        # print('info = {}'.format(ub.urepr(info, nl=1, sort=False)))
        single_rows.append(info)

    sensor_chan_stats = pd.DataFrame(single_rows)
    sensor_chan_stats = sensor_chan_stats.set_index(['sensor', 'channel'])
    print(sensor_chan_stats.to_string())

    print('compare channels between sensors')
    chan_to_group = ub.group_items(
        sensor_channel_to_vwf.keys(),
        [t[1] for t in sensor_channel_to_vwf.keys()]
    )
    chan_to_combos = {
        chan: list(it.combinations(group, 2)) for chan, group in chan_to_group.items()
    }
    to_compare = list(ub.flatten(chan_to_combos.values()))
    pairwise_rows = []
    for item1, item2 in ub.ProgIter(to_compare, desc='comparse_sensors', verbose=1):
        sensor1, channel1 = item1
        sensor2, channel2 = item2
        assert channel1 == channel2
        channel = channel1

        row = {
            'sensor1': sensor1,
            'sensor2': sensor2,
            'channel': channel,
        }

        u_sensor_data = sensor_channel_to_vwf[(sensor1, channel1)]
        v_sensor_data = sensor_channel_to_vwf[(sensor2, channel2)]

        u_values, u_weights = ub.take(u_sensor_data, ['raw_values', 'raw_weights'])
        v_values, v_weights = ub.take(v_sensor_data, ['raw_values', 'raw_weights'])

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html#scipy.stats.wasserstein_distance
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.energy_distance.html#scipy.stats.energy_distance
        dist_inputs = dict(
            u_values=u_values, v_values=v_values, u_weights=u_weights,
            v_weights=v_weights)

        if 1:
            # TODO: normalize data such that density (or probability?) is 1.0
            row['emd'] = scipy.stats.wasserstein_distance(**dist_inputs)

        if 1:
            row['energy_dist'] = scipy.stats.energy_distance(**dist_inputs)

        u_stats = sensor_chan_stats.loc[(sensor1, channel1)]
        v_stats = sensor_chan_stats.loc[(sensor2, channel2)]
        row['mean_diff'] = abs(u_stats['mean'] - v_stats['mean'])
        row['std_diff'] = abs(u_stats['std'] - v_stats['std'])

        # TODO: robust alignment of pdfs
        if 1:
            cuv_values = np.union1d(u_values, v_values)
            cuv_values.sort()
            cu_weights = np.zeros(cuv_values.shape, dtype=np.float64)
            cv_weights = np.zeros(cuv_values.shape, dtype=np.float64)
            cu_indexes = np.where(kwarray.isect_flags(cuv_values, u_values))[0]
            cv_indexes = np.where(kwarray.isect_flags(cuv_values, v_values))[0]
            cu_weights[cu_indexes] = u_weights
            cv_weights[cv_indexes] = v_weights

            cu_weights * cv_weights

            row['kld'] = scipy.stats.entropy(cu_weights, cv_weights)

        pairwise_rows.append(row)

    distance_metrics = pd.DataFrame(pairwise_rows)
    print(distance_metrics.to_string())
    return sensor_chan_stats, distance_metrics


def ensure_intensity_sidecar(fpath, cache_dpath=None, bundle_dpath=None,
                             recompute=False):
    """
    Write statistics next to the image

    Args:
        fpath (str | PathLike): the path to the image to compute spectra for

        cache_dpath (str | PathLike | None):
            if specified, write cache files here instead of next to the input
            file.

        bundle_dpath (str | PathLike | None):
            if cache_dpath is specified and this is specified, this is
            considered the "root" of the image, which allows the cache to use
            specific parent folders, but not the entire filesystem.
            Specifying this allows portablity of the cache.

        recompute (bool): if True, force recomputation.

    Example:
        >>> from geowatch.cli.coco_spectra import *  # NOQA
        >>> import kwimage
        >>> import ubelt as ub
        >>> dpath = ub.Path.appdir('geowatch/tests/intensity_sidecar').ensuredir()
        >>> dpath.delete().ensuredir()
        >>> img = kwimage.grab_test_image(dsize=(16, 16))
        >>> img01 = kwimage.ensure_float01(img)
        >>> img255 = kwimage.ensure_uint255(img)
        >>> fpath1 = dpath / 'img01.tif'
        >>> fpath2 = dpath / 'img255.tif'
        >>> kwimage.imwrite(fpath1, img01)
        >>> kwimage.imwrite(fpath2, img255)
        >>> fpath = fpath1
        >>> stats_fpath1 = ensure_intensity_sidecar(fpath1)
        >>> fpath = fpath2
        >>> stats_fpath2 = ensure_intensity_sidecar(fpath1)
        >>> import pickle
        >>> pickle.loads(stats_fpath1.read_bytes())
        >>> pickle.loads(stats_fpath2.read_bytes())

    Example:
        >>> from geowatch.cli.coco_spectra import *  # NOQA
        >>> import kwimage
        >>> import ubelt as ub
        >>> dpath = ub.Path.appdir('geowatch/tests/intensity_sidecar2').ensuredir()
        >>> dpath.delete().ensuredir()
        >>> cache_dpath = dpath / 'cache'
        >>> bundle_dpath = None
        >>> img = kwimage.grab_test_image(dsize=(16, 16))
        >>> img255 = kwimage.ensure_uint255(img)
        >>> fpath = dpath / 'img255.tif'
        >>> kwimage.imwrite(fpath, img255)
        >>> stats_fpath = ensure_intensity_sidecar(fpath, cache_dpath=cache_dpath)
        >>> import pickle
        >>> pickle.loads(stats_fpath.read_bytes())
    """
    import os
    import kwarray
    import kwimage
    import pickle

    if cache_dpath is None:
        stats_fpath = ub.Path(os.fspath(fpath) + '.stats_v1.pkl')
    else:
        fpath = ub.Path(fpath)
        if bundle_dpath is not None:
            relpath = fpath.relative_to(bundle_dpath)
        else:
            proxy_parent = ub.hash_data(fpath.parent, hasher='sha256')
            relpath = ub.Path(proxy_parent) / fpath.name
        stats_fpath = ub.Path(cache_dpath) / (relpath + '.stats_v1.pkl')

    if recompute or not stats_fpath.exists():
        import safer
        imdata = kwimage.imread(fpath, backend='gdal', nodata_method='ma')
        imdata = kwarray.atleast_nd(imdata, 3)
        # TODO: even better float handling
        if imdata.dtype.kind == 'f':
            precision = 2
            # Round floats to p decimals to keep things semi-managable
            # binning floats is necessary due to fundamental limitations
            # We may need to decrease our resolution even further.
            imdata = imdata.round(precision)
        stats_info = {'bands': []}
        for imband in imdata.transpose(2, 0, 1):
            masked_data = imband.ravel()
            # Remove masked data
            data = masked_data.data[~masked_data.mask]
            intensity_hist = ub.dict_hist(data)
            intensity_hist = ub.sorted_keys(intensity_hist)
            stats_info['bands'].append({
                'intensity_hist': intensity_hist,
            })
        stats_fpath.parent.ensuredir()
        with safer.open(stats_fpath, 'wb') as file:
            pickle.dump(stats_info, file)
    return stats_fpath


def ensure_intensity_stats(coco_img,
                           recompute=False,
                           cache_dpath=None,
                           bundle_dpath=None,
                           include_channels=None,
                           exclude_channels=None,
                           valid_min=-math.inf,
                           valid_max=math.inf):
    """
    Ensures a sidecar file exists for the kwcoco image
    """
    from os.path import join
    import numpy as np
    import kwcoco
    import kwimage
    import pickle
    intensity_stats = {'bands': []}
    for obj in coco_img.iter_asset_objs():
        fpath = join(coco_img.bundle_dpath, obj['file_name'])
        channels = obj.get('channels', None)
        if channels is None:
            shape = kwimage.load_image_shape(fpath)
            num_channels = shape[2]
            if num_channels == 3:
                channels = 'r|g|b'
            elif num_channels == 4:
                channels = 'r|g|b|a'
            elif num_channels == 1:
                channels = 'r|g|b'
            else:
                channels = kwcoco.FusedChannelSpec.coerce(num_channels)

        channels = kwcoco.FusedChannelSpec.coerce(channels)
        declared_channel_list = channels.as_list()

        quantization = obj.get('quantization', None)

        requested_channels = channels
        if include_channels:
            requested_channels = requested_channels & include_channels
        if exclude_channels:
            requested_channels = requested_channels - exclude_channels

        if requested_channels.numel() > 0:
            stats_fpath = ensure_intensity_sidecar(fpath,
                                                   cache_dpath=cache_dpath,
                                                   bundle_dpath=bundle_dpath,
                                                   recompute=recompute)
            try:
                with open(stats_fpath, 'rb') as file:
                    stat_info = pickle.load(file)
            except Exception as ex:
                print('ex = {!r}'.format(ex))
                print('error loading stats_fpath = {!r}'.format(stats_fpath))
                raise

            alwaysappend = requested_channels.numel() == channels.numel()

            for band_idx, band_stat in enumerate(stat_info['bands']):
                try:
                    band_name = declared_channel_list[band_idx]
                except IndexError:
                    print('bad channel declaration fpath = {!r}'.format(fpath))
                    if 0:
                        print('obj = {}'.format(ub.urepr(obj, nl=1)))
                        print('coco_img = {!r}'.format(coco_img))
                        print('fpath = {!r}'.format(fpath))
                        print('stats_fpath = {!r}'.format(stats_fpath))
                        print(len(stat_info['bands']))
                        print('band_idx = {!r}'.format(band_idx))
                        print('channels = {!r}'.format(channels))
                    # raise
                    band_name = 'unknown'
                if quantization is not None:
                    # Handle dequantization
                    from delayed_image.helpers import dequantize
                    quant_values = np.array(list(band_stat['intensity_hist'].keys()))
                    counts = list(band_stat['intensity_hist'].values())
                    dequant_values = dequantize(quant_values, quantization)
                    band_stat['intensity_hist'] = ub.odict(zip(dequant_values, counts))
                if alwaysappend or band_name in requested_channels:
                    band_stat['properties'] = props = {
                        'video_name': None,
                        'band_name': band_name,
                        'month': None,
                    }
                    try:
                        props['video_name'] = coco_img.video['name']
                    except Exception:
                        ...
                    try:
                        from kwutil import util_time
                        props['month'] = util_time.coerce_datetime(coco_img.img['date_captured']).month
                    except Exception:
                        ...
                    band_stat['band_name'] = band_name
                    intensity_stats['bands'].append(band_stat)

    if math.isfinite(valid_max) or math.isfinite(valid_min):
        # Filter on the fly, but in the worker process
        for band_stats in intensity_stats['bands']:
            intensity_hist = band_stats['intensity_hist']
            band_stats['intensity_hist'] = {
                k: v for k, v in intensity_hist.items()
                if k >= valid_min and k <= valid_max}

    return intensity_stats


def plot_intensity_histograms(full_df, config, ax=None):
    """
    Args:
        full_df (pd.DataFrame):
            with columns: index  intensity_bin  value channel   sensor
        ax (Axes | None):
            if specified, we assume only 1 plot is made

    Example:
        >>> # xdoctest: +REQUIRES(--show)
        >>> from geowatch.cli.coco_spectra import *  # NOQA
        >>> import pandas as pd
        >>> n = 256
        >>> data = {
        >>>     'intensity_bin': np.arange(0, n),
        >>>     'value': np.random.randint(0, 256, n),
        >>>     'channel': ['red'] * n,
        >>>     'sensor': ['unknown'] * n,
        >>> }
        >>> full_df = pd.DataFrame(data)
        >>> config = CocoSpectraConfig()
        >>> kwplot.autompl()
        >>> ax = plot_intensity_histograms(full_df, config)
    """
    unique_channels = full_df['channel'].unique()
    unique_sensors = full_df['sensor'].unique()

    import kwplot
    sns = kwplot.autosns()

    palette = {
        'red': 'red',
        'blue': 'blue',
        'green': 'green',
        'cirrus': 'skyblue',
        'coastal': 'purple',
        'nir': 'orange',
        'swir16': 'pink',
        'swir22': 'hotpink',
    }
    if 'red' not in unique_channels and 'r' in unique_channels:
        # hack
        palette['r'] = 'red'
        palette['g'] = 'green'
        palette['b'] = 'blue'
    for channel in unique_channels:
        if channel not in palette:
            palette[channel] = None
    palette = _fill_missing_colors(palette)

    hist_data_kw = dict(
        x='intensity_bin',
        weights='value',
        bins=config['bins'],
        stat=config['stat'],
        hue='channel',
    )
    hist_style_kw = dict(
        palette=palette,
        fill=config['fill'],
        element=config['element'],
        multiple=config['multiple'],
        kde=config['kde'],
        cumulative=config['cumulative'],
    )

    if 0:
        # __hisplot_notes__
        import inspect
        sig = inspect.signature(sns.histplot)
        # Print params we might not have looked at in detail
        exposed_params = {
            'cumulative', 'kde', 'multiple', 'element', 'fill', 'hue', 'stat',
            'bins', 'weights', 'x', 'palette',
        }
        probably_ignorable_params = {
            'pmax', 'hue_order', 'hue_norm', 'cbar', 'cbar_kws', 'cbar_ax', 'ax',
            'legend', 'thresh' 'y',
        }
        maybe_expose = (set(sig.parameters) - exposed_params) - probably_ignorable_params
        print('maybe_expose = {}'.format(ub.urepr(maybe_expose, nl=1)))

    # For S2 that is supposed to be divide by 10000.
    # For L8 it is multiply by 2.75e-5 and subtract 0.2.
    # 1 / 2.75e-5
    print('start plot')

    print(f'unique_sensors={unique_sensors}')
    if ax is None:
        fig = kwplot.figure(fnum=1, doclf=True)
        fig.clf()
        print('fig = {!r}'.format(fig))
        pnum_ = kwplot.PlotNums(nSubplots=len(unique_sensors))
        print('pnum_ = {!r}'.format(pnum_))
    else:
        fig = None
        assert len(unique_sensors) == 1

    for sensor_name, sensor_df in full_df.groupby('sensor'):
        print('plot sensor_name = {!r}'.format(sensor_name))

        hist_data_kw_ = hist_data_kw.copy()
        if hist_data_kw_['bins'] == 'auto':
            xvar = hist_data_kw['x']
            weightvar = hist_data_kw['weights']
            hist_data_kw_['bins'] = _weighted_auto_bins(sensor_df, xvar, weightvar)

        ax = kwplot.figure(fnum=1, pnum=pnum_()).gca()

        # z = [tuple(a.values()) for a in sensor_df[['intensity_bin', 'channel', 'sensor']].to_dict('records')]
        # ub.find_duplicates(z)
        try:
            # We have already computed the histogram, but we can get seaborn to
            # show it using a simple trick: use weight to represent the
            # frequency that should be used for every bin.

            # https://github.com/mwaskom/seaborn/issues/2709
            sns.histplot(ax=ax, data=sensor_df.reset_index(), **hist_data_kw_, **hist_style_kw)
        except Exception:
            print('hist_data_kw_ = {}'.format(ub.urepr(hist_data_kw_, nl=1)))
            print('hist_style_kw = {}'.format(ub.urepr(hist_style_kw, nl=1)))
            print('ERROR')
            print(sensor_df)
            raise
            pass
        if config['valid_range'] is not None:
            valid_min, valid_max = map(float, config['valid_range'].split(':'))
            ax.set_xlim(valid_min, valid_max)

        ax.set_title(sensor_name)
        # maxx = sensor_df.intensity_bin.max()
        # maxx = sensor_maxes[sensor_name]
        # ax.set_xlim(0, maxx)

    return fig


def _weighted_auto_bins(data, xvar, weightvar):
    """
    Generalized histogram bandwidth estimators for weighted univariate data

    References:
        https://github.com/mwaskom/seaborn/issues/2710

    TODO:
        add to util_kwarray

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> n = 100
        >>> to_stack = []
        >>> rng = np.random.RandomState(432)
        >>> for group_idx in range(3):
        >>>     part_data = pd.DataFrame({
        >>>         'x': np.arange(n),
        >>>         'weights': rng.randint(0, 100, size=n),
        >>>         'hue': [f'group_{group_idx}'] * n,
        >>>     })
        >>>     to_stack.append(part_data)
        >>> data = pd.concat(to_stack).reset_index()
        >>> xvar = 'x'
        >>> weightvar = 'weights'
        >>> n_equal_bins = _weighted_auto_bins(data, xvar, weightvar)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import seaborn as sns
        >>> sns.histplot(data=data, bins=n_equal_bins, x='x', weights='weights', hue='hue')
    """
    import numpy as np
    sort_df = data.sort_values(xvar)
    values = sort_df[xvar]
    weights = sort_df[weightvar]
    minval = values.iloc[0]
    maxval = values.iloc[-1]

    total = weights.sum()
    ptp = maxval - minval

    # _hist_bin_sqrt = ptp / np.sqrt(total)
    _hist_bin_sturges = ptp / (np.log2(total) + 1.0)

    cumtotal = weights.cumsum().values
    quantiles = cumtotal / cumtotal[-1]
    idx2, idx1 = np.searchsorted(quantiles, [0.75, 0.25])
    # idx2, idx1 = _weighted_quantile(weights, [0.75, 0.25])
    iqr = values.iloc[idx2] - values.iloc[idx1]
    _hist_bin_fd = 2.0 * iqr * total ** (-1.0 / 3.0)

    fd_bw = _hist_bin_fd  # Freedman-Diaconis
    sturges_bw = _hist_bin_sturges

    if fd_bw:
        bw_est = min(fd_bw, sturges_bw)
    else:
        # limited variance, so we return a len dependent bw estimator
        bw_est = sturges_bw

    # from numpy.lib.histograms import _get_outer_edges, _unsigned_subtract
    first_edge, last_edge = _get_outer_edges(values, None)
    if bw_est:
        n_equal_bins = int(np.ceil(_unsigned_subtract(last_edge, first_edge) / bw_est))
    else:
        # Width can be zero for some estimators, e.g. FD when
        # the IQR of the data is zero.
        n_equal_bins = 1

    # Take the minimum of this and the number of actual bins
    n_equal_bins = min(n_equal_bins, len(values))
    return n_equal_bins


def _get_outer_edges(a, range):
    """
    Determine the outer bin edges to use, from either the data or the range
    argument

    Note: vendored from numpy.lib._histograms_impl
    """
    import numpy as np
    if range is not None:
        first_edge, last_edge = range
        if first_edge > last_edge:
            raise ValueError(
                'max must be larger than min in range parameter.')
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                "supplied range of [{}, {}] is not finite".format(first_edge, last_edge))
    elif a.size == 0:
        # handle empty arrays. Can't determine range, so use 0-1.
        first_edge, last_edge = 0, 1
    else:
        first_edge, last_edge = a.min(), a.max()
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                "autodetected range of [{}, {}] is not finite".format(first_edge, last_edge))

    # expand empty range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    return first_edge, last_edge


def _unsigned_subtract(a, b):
    """
    Subtract two values where a >= b, and produce an unsigned result

    This is needed when finding the difference between the upper and lower
    bound of an int16 histogram

    Note: vendored from numpy.lib._histograms_impl
    """
    import numpy as np
    # coerce to a single type
    signed_to_unsigned = {
        np.byte: np.ubyte,
        np.short: np.ushort,
        np.intc: np.uintc,
        np.int_: np.uint,
        np.longlong: np.ulonglong
    }
    dt = np.result_type(a, b)
    try:
        unsigned_dt = signed_to_unsigned[dt.type]
    except KeyError:
        return np.subtract(a, b, dtype=dt)
    else:
        # we know the inputs are integers, and we are deliberately casting
        # signed to unsigned.  The input may be negative python integers so
        # ensure we pass in arrays with the initial dtype (related to NEP 50).
        return np.subtract(np.asarray(a, dtype=dt), np.asarray(b, dtype=dt),
                           casting='unsafe', dtype=unsigned_dt)


def _fill_missing_colors(label_to_color):
    """
    label_to_color = {'foo': kwimage.Color('red').as01(), 'bar': None}
    """
    from distinctipy import distinctipy
    import kwarray
    import numpy as np
    import kwimage
    given = {k: kwimage.Color(v).as01() for k, v in label_to_color.items() if v is not None}
    needs_color = sorted(set(label_to_color) - set(given))

    seed = 6777939437
    # hack in our code

    def _patched_get_random_color(pastel_factor=0, rng=None):
        rng = kwarray.ensure_rng(seed, api='python')
        color = [(rng.random() + pastel_factor) / (1.0 + pastel_factor) for _ in range(3)]
        return tuple(color)
    distinctipy.get_random_color = _patched_get_random_color

    exclude_colors = [
        tuple(map(float, (d, d, d)))
        for d in np.linspace(0, 1, 5)
    ] + list(given.values())

    final = given.copy()
    new_colors = distinctipy.get_colors(len(needs_color), exclude_colors=exclude_colors)
    for key, new_color in zip(needs_color, new_colors):
        final[key] = tuple(map(float, new_color))
    return final


__cli__ = CocoSpectraConfig

if __name__ == '__main__':
    main()
