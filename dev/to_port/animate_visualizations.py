#!/usr/bin/env python3
__notes__ = r"""
.. :code: bash

    # Make an animated gif for specified bands (use "," to separate)
    # Requires a CD
    CHANNELS="red|green|blue"
    mapfile -td \, _BANDS < <(printf "%s\0" "$CHANNELS")
    items=$(jq -r '.videos[] | .name' $OUTPUT_COCO_FPATH)
    for item in ${items[@]}; do
        echo "item = $item"
        for bandname in ${_BANDS[@]}; do
            echo "_BANDS = $_BANDS"
            BAND_DPATH="$VIZ_DPATH/${item}/_anns/${bandname}/"
            GIF_FPATH="$VIZ_DPATH/${item}_anns_${bandname}.gif"
            python -m kwplot.cli.gifify --frames_per_second .7 \
                --input "$BAND_DPATH" --output "$GIF_FPATH"
        done
    done
"""


def animate_visualizations(viz_dpath, channels=None, video_names=None,
                           frames_per_second=0.7, draw_anns=True,
                           draw_imgs=True, workers=0, zoom_to_tracks=False,
                           verbose=0):
    r"""
    Helper that roughly does the same thing as this bash script:

    Args:
        viz_dpath (str): the path where visualizations were dumped with the
            coco_visualize_videos script.

        zoom_to_tracks (bool):
            if specified uses "track" based-logic find paths to animate

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(--ffmpeg-test')
        >>> import ubelt as ub
        >>> dpath = ub.Path.appdir('geowatch/tests/ani_video').delete().ensuredir()
        >>> import kwcoco
        >>> from geowatch.utils import kwcoco_extensions
        >>> dset = kwcoco.CocoDataset.demo('vidshapes2-msi', num_frames=5)
        >>> img = dset.dataset['images'][0]
        >>> coco_img = dset.coco_image(img['id'])
        >>> channel_chunks = list(ub.chunks(coco_img.channels.fuse().parsed, chunksize=3))
        >>> channels = ','.join(['|'.join(p) for p in channel_chunks])
        >>> kwargs = {
        >>>     'src': dset.fpath,
        >>>     'viz_dpath': dpath,
        >>>     'space': 'video',
        >>>     'channels': channels,
        >>>     'zoom_to_tracks': False,
        >>> }
        >>> from geowatch.cli.coco_visualize_videos import main
        >>> cmdline = False
        >>> main(cmdline=cmdline, **kwargs)
        >>> viz_dpath = dpath
        >>> channels = None
        >>> video_names = None
        >>> frame_per_second = 0.7
        >>> from geowatch.cli.animate_visualizations import *  # NOQA
        >>> animate_visualizations(viz_dpath, verbose=1, workers=0)
    """
    from kwplot.cli import gifify
    import ubelt as ub
    import kwcoco
    from kwutil import util_parallel

    if channels is not None:
        channels = kwcoco.ChannelSpec.coerce(channels)

    workers = util_parallel.coerce_num_workers(workers)
    viz_dpath = ub.Path(viz_dpath)

    ffmpeg_exe = ub.find_exe('ffmpeg')
    if ffmpeg_exe is None:
        raise Exception('Cannot find ffmpeg, which is required to run animation')

    if video_names is None:
        video_dpaths = [p for p in viz_dpath.glob('*') if p.is_dir()]
    else:
        if len(video_names) == 1:
            workers = 0
        video_dpaths = [viz_dpath / n for n in video_names]

    pool = ub.JobPool(mode='thread', max_workers=workers)

    types = []
    if draw_imgs:
        types.append('_imgs')
    if draw_anns:
        types.append('_anns')

    if workers == 1:
        workers = 0
    verbose_worker = verbose and workers <= 1

    # We make heavy reliance on a known directory structure here.
    # In general I don't like this, but this is not a system-critical part
    # so we can leave refactoring as a todo.

    from kwutil import util_progress
    pman = util_progress.ProgressManager()
    pman.__enter__()

    # prog = ub.ProgIter(desc='submit video jobs', verbose=3)
    prog = pman.progiter(desc='submit video jobs')
    prog.begin()

    with_gif = 'auto'
    with_mp4 = True

    outputs = []

    for type_ in types:
        for video_dpath in video_dpaths:
            prog.set_extra('type_={!r} video_dpath={!r}'.format(type_, video_dpath))
            prog.step()
            video_name = video_dpath.name

            if zoom_to_tracks:
                track_subdpath = video_dpath / '_tracks'
                track_dpaths = list(track_subdpath.glob('*'))
                for track_dpath in track_dpaths:
                    track_name = track_dpath.name
                    type_dpath = track_dpath / type_

                    if channels is None:
                        channel_dpaths = [p for p in type_dpath.glob('*') if p.is_dir()]
                    else:
                        channel_dpaths = [type_dpath / c.path_sanitize()
                                          for c in channels.streams()]

                    for chan_dpath in channel_dpaths:
                        frame_fpaths = sorted(chan_dpath.glob('*'))
                        if len(frame_fpaths):
                            if len(frame_fpaths) < 300:
                                gif_fname = '{}{}_{}.gif'.format(track_name, type_, chan_dpath.name)
                                gif_fpath = track_subdpath / gif_fname
                                pool.submit(
                                    gifify.ffmpeg_animate_frames, frame_fpaths,
                                    gif_fpath, in_framerate=frames_per_second,
                                    verbose=verbose_worker)
                                outputs.append({
                                    'fpath': gif_fpath,
                                    'type': 'gif',
                                })
                            ani_fname = '{}{}_{}.mp4'.format(track_name, type_, chan_dpath.name)
                            ani_fpath = track_subdpath / ani_fname
                            pool.submit(
                                gifify.ffmpeg_animate_frames, frame_fpaths,
                                ani_fpath, in_framerate=frames_per_second,
                                verbose=verbose_worker)
                            outputs.append({
                                'fpath': ani_fpath,
                                'type': 'mp4',
                            })

            else:
                type_dpath = video_dpath / type_
                if channels is None:
                    channel_dpaths = [p for p in type_dpath.glob('*') if p.is_dir()]
                else:
                    channel_dpaths = [type_dpath / c.path_sanitize()
                                      for c in channels.streams()]
                for chan_dpath in channel_dpaths:
                    frame_fpaths = sorted(chan_dpath.glob('*'))
                    if len(frame_fpaths):
                        with_gif_resolved = with_gif
                        if with_gif == 'auto':
                            with_gif_resolved = len(frame_fpaths) < 300

                        if with_gif_resolved:
                            gif_fname = '{}{}_{}.gif'.format(video_name, type_, chan_dpath.name)
                            gif_fpath = video_dpath / gif_fname
                            pool.submit(
                                gifify.ffmpeg_animate_frames, frame_fpaths, gif_fpath,
                                in_framerate=frames_per_second, verbose=verbose_worker)
                            outputs.append({
                                'fpath': gif_fpath,
                                'type': 'gif',
                            })

                        if with_mp4:
                            ani_fname = '{}{}_{}.mp4'.format(video_name, type_, chan_dpath.name)
                            ani_fpath = video_dpath / ani_fname
                            pool.submit(
                                gifify.ffmpeg_animate_frames, frame_fpaths, ani_fpath,
                                in_framerate=frames_per_second, verbose=verbose_worker)
                            outputs.append({
                                'fpath': ani_fpath,
                                'type': 'mp4',
                            })
    prog.end()

    failed = []
    # for job in ub.ProgIter(pool.as_completed(), total=len(pool), desc='collect animate jobs'):
    for job in pman.progiter(pool.as_completed(), total=len(pool), desc='collect animate jobs'):
        try:
            job.result()
        except Exception as ex:
            failed.append(ex)
            pass

    if failed:
        print('Animation jobs failed with the following errors:')
        print('failed = {}'.format(ub.urepr(failed, nl=1)))
        raise Exception(f'{len(failed)} / {len(pool)} animations failed')

    pman.__exit__(None, None, None)

    print('Wrote animations to viz_dpath = {!r}'.format(viz_dpath))
    # The animation jobs can do something weird to the tty, so we should try
    # and fix it.
    ub.cmd('stty sane')
    return outputs


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/geowatch/cli/animate_visualizations.py
    """
    import fire
    fire.Fire(animate_visualizations)
