#!/usr/bin/env python
"""
Visualize annotations on kwcoco video frames.
"""
import math
import ubelt as ub
import scriptconfig as scfg


class CocoVisualizeVideosCLI(scfg.DataConfig):
    """
    Visualize one or more videos from a kwcoco dataset.

    Example:
        >>> # xdoctest: +REQUIRES(module:kwimage)
        >>> from kwcoco.cli.coco_visualize_videos import *  # NOQA
        >>> import kwcoco
        >>> import ubelt as ub
        >>> dpath = ub.Path.appdir('kwcoco/tests/visualize_videos').delete().ensuredir()
        >>> dset = kwcoco.CocoDataset.demo('vidshapes2', num_frames=2, num_videos=1)
        >>> kw = {
        >>>     'src': dset.fpath,
        >>>     'viz_dpath': dpath,
        >>>     'workers': 0,
        >>>     'draw_labels': False,
        >>> }
        >>> CocoVisualizeVideosCLI.main(cmdline=False, **kw)
        >>> assert len(list(dpath.glob('**/*.jpg'))) > 0

    Example:
        >>> # Minimal example without annotations (no cv2 required)
        >>> # xdoctest: +REQUIRES(module:kwimage)
        >>> from kwcoco.cli.coco_visualize_videos import *  # NOQA
        >>> import kwcoco
        >>> import kwimage
        >>> import numpy as np
        >>> import ubelt as ub
        >>> dpath = ub.Path.appdir('kwcoco/tests/visualize_videos_noanns').delete().ensuredir()
        >>> dset = kwcoco.CocoDataset()
        >>> vidid = dset.add_video(name='video1', width=64, height=64)
        >>> for frame in range(2):
        >>>     canvas = np.zeros((64, 64, 3), dtype=np.uint8)
        >>>     canvas[..., 2] = 255
        >>>     fpath = dpath / f'frame_{frame}.png'
        >>>     kwimage.imwrite(fpath, canvas)
        >>>     dset.add_image(
        >>>         file_name=str(fpath),
        >>>         width=64,
        >>>         height=64,
        >>>         video_id=vidid,
        >>>         frame_index=frame,
        >>>     )
        >>> dset.fpath = dpath / 'data.kwcoco.json'
        >>> dset.dump(dset.fpath)
        >>> kw = {
        >>>     'src': dset.fpath,
        >>>     'viz_dpath': dpath / 'viz',
        >>>     'workers': 0,
        >>>     'draw_anns': False,
        >>>     'draw_header': False,
        >>>     'draw_chancode': False,
        >>> }
        >>> CocoVisualizeVideosCLI.main(cmdline=False, **kw)
        >>> assert len(list((dpath / 'viz').glob('**/*.jpg'))) > 0

    Example:
        >>> # Example with annotations (requires cv2 for drawing)
        >>> # xdoctest: +REQUIRES(module:kwimage)
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> from kwcoco.cli.coco_visualize_videos import *  # NOQA
        >>> import kwcoco
        >>> import kwimage
        >>> import numpy as np
        >>> import ubelt as ub
        >>> dpath = ub.Path.appdir('kwcoco/tests/visualize_videos_anns').delete().ensuredir()
        >>> dset = kwcoco.CocoDataset()
        >>> vidid = dset.add_video(name='video1', width=64, height=64)
        >>> catid = dset.add_category('thing', color='lime')
        >>> for frame in range(2):
        >>>     canvas = np.zeros((64, 64, 3), dtype=np.uint8)
        >>>     canvas[..., 1] = 255
        >>>     fpath = dpath / f'frame_{frame}.png'
        >>>     kwimage.imwrite(fpath, canvas)
        >>>     gid = dset.add_image(
        >>>         file_name=str(fpath),
        >>>         width=64,
        >>>         height=64,
        >>>         video_id=vidid,
        >>>         frame_index=frame,
        >>>     )
        >>>     if frame == 0:
        >>>         dset.add_annotation(
        >>>             image_id=gid,
        >>>             category_id=catid,
        >>>             bbox=[10, 10, 20, 20],
        >>>         )
        >>> dset.fpath = dpath / 'data.kwcoco.json'
        >>> dset.dump(dset.fpath)
        >>> kw = {
        >>>     'src': dset.fpath,
        >>>     'viz_dpath': dpath / 'viz',
        >>>     'workers': 0,
        >>>     'draw_labels': False,
        >>> }
        >>> CocoVisualizeVideosCLI.main(cmdline=False, **kw)
        >>> assert len(list((dpath / 'viz').glob('**/*.jpg'))) > 0
    """
    __command__ = 'visualize'
    __alias__ = ['visualize_videos']

    src = scfg.Value(None, help='Input kwcoco dataset', position=1)

    viz_dpath = scfg.Value(None, help=ub.paragraph(
        '''
        Where to save the visualizations. If unspecified, writes them
        adjacent to the input kwcoco file.
        '''))

    video = scfg.Value(None, help=ub.paragraph(
        '''
        Video id/name selection. Accepts a YAML list or a comma-separated
        list of names/ids. If unspecified, all videos are rendered.
        '''))

    gids = scfg.Value(None, help=ub.paragraph(
        '''
        Optional list of image ids to render. Accepts YAML list or
        comma-separated ids. Applied after video selection.
        '''))

    select_images = scfg.Value(None, type=str, help=ub.paragraph(
        '''
        A jq-style query for image dictionaries. Requires the "jq" python
        library. See kwcoco._helpers._query_image_ids for details.
        '''))

    select_videos = scfg.Value(None, type=str, help=ub.paragraph(
        '''
        A jq-style query for video dictionaries. Requires the "jq" python
        library. See kwcoco._helpers._query_image_ids for details.
        '''))

    include_loose = scfg.Value(True, isflag=True, help=ub.paragraph(
        '''
        If True, render images without a video in a "loose-images" folder.
        '''))

    workers = scfg.Value('auto', help='Number of parallel workers')

    space = scfg.Value('video', help='Render in image or video space')

    max_dim = scfg.Value(1024, help='Resize if larger than this dimension')
    min_dim = scfg.Value(256, help='Resize if smaller than this dimension')
    resolution = scfg.Value(None, help='Output resolution (if supported)')

    channels = scfg.Value(None, type=str, help='Only visualize these channels')
    any3 = scfg.Value(False, help=ub.paragraph(
        '''
        If True, ensure any 3 channels are drawn as a false-color view. If
        set to "only", other per-channel visualizations are suppressed.
        '''))

    draw_imgs = scfg.Value(True, isflag=True, help='Draw images')
    draw_anns = scfg.Value('auto', help=ub.paragraph(
        '''
        Draw annotations. When "auto", only draw if annotations exist.
        '''))
    draw_boxes = scfg.Value(True, help='Draw bounding boxes')
    draw_segmentations = scfg.Value(True, help='Draw segmentation polygons')
    draw_labels = scfg.Value(True, help='Draw annotation labels')
    ann_thickness = scfg.Value(2, help='Annotation line thickness')
    alpha = scfg.Value(None, help='Annotation transparency')
    ann_score_thresh = scfg.Value(0, help='Drop annotations below this score')

    draw_valid_region = scfg.Value(False, help='Draw valid image regions')
    draw_header = scfg.Value(True, help='Draw header text')
    draw_chancode = scfg.Value(True, help='Draw channel code text')

    cmap = scfg.Value('viridis', type=str, help=ub.paragraph(
        '''
        Name of a colormap for single channel data. Can also be a YAML
        mapping from channel name to colormap name.
        '''))

    skip_missing = scfg.Value(True, isflag=True, help=ub.paragraph(
        '''
        If true, skip any image that does not have the requested channels.
        '''))
    skip_aggressive = scfg.Value(False, isflag=True, help=ub.paragraph(
        '''
        Aggressively skip frames that appear to be invalid.
        '''))

    draw_track_trails = scfg.Value(False, isflag=True, help=ub.paragraph(
        '''
        Draw history of track locations.
        '''))

    verbose = scfg.Value(0, isflag=True, help='verbosity level')

    @classmethod
    def main(cls, cmdline=True, **kwargs):
        """
        CommandLine:
            xdoctest -m kwcoco.cli.coco_visualize_videos CocoVisualizeVideosCLI.main
        """
        import kwcoco
        import kwimage
        import numpy as np
        from kwutil import util_parallel
        from kwcoco._helpers import _query_image_ids

        config = cls.cli(data=kwargs, cmdline=cmdline, strict=True)
        print('config = {}'.format(ub.urepr(dict(config), nl=1)))

        if config['src'] is None:
            raise ValueError('Must specify a kwcoco dataset path via --src')

        coco_dset = kwcoco.CocoDataset.coerce(config['src'])

        if config['draw_anns'] == 'auto':
            config['draw_anns'] = coco_dset.n_annots > 0

        dset_idstr = coco_dset._dataset_id()
        bundle_dpath = ub.Path(coco_dset.bundle_dpath)
        if config['viz_dpath'] is not None:
            viz_dpath = ub.Path(config['viz_dpath'])
        else:
            viz_dpath = bundle_dpath / '_viz_{}'.format(dset_idstr)
        viz_dpath.ensuredir()

        max_workers = util_parallel.coerce_num_workers(config['workers'])

        selected_gids = _query_image_ids(
            coco_dset,
            select_images=config['select_images'],
            select_videos=config['select_videos'],
        )

        gids_filter = _coerce_id_list(config['gids'])
        if gids_filter is not None:
            selected_gids = sorted(set(selected_gids) & set(gids_filter))

        if config['skip_missing'] and config['channels'] is not None:
            requested_channels = kwcoco.ChannelSpec.coerce(
                config['channels']).fuse().as_set()
            coco_images = coco_dset.images(selected_gids).coco_images
            keep = []
            for coco_img in coco_images:
                img_channels = coco_img.channels
                if img_channels is None:
                    if not config['skip_aggressive']:
                        keep.append(coco_img.img['id'])
                else:
                    code = img_channels.fuse().as_set()
                    if config['skip_aggressive']:
                        if len(requested_channels & code) == len(requested_channels):
                            keep.append(coco_img.img['id'])
                    else:
                        if requested_channels & code:
                            keep.append(coco_img.img['id'])
            if config['verbose']:
                print(f'Filtered {len(coco_images) - len(keep)} images without requested channels')
            selected_gids = keep

        vidids, vidname_lut = _select_videos(coco_dset, config['video'])
        video_items = []
        if vidids is None:
            video_items.extend(list(coco_dset.index.videos.items()))
        else:
            for vidid in vidids:
                video_items.append((vidid, coco_dset.index.videos[vidid]))

        if config['include_loose']:
            video_items.append((None, {'name': 'loose-images'}))

        pool = ub.JobPool(mode='thread', max_workers=max_workers)

        for vidid, video in ub.ProgIter(video_items, desc='visualize videos', verbose=3):
            video_name = str(video.get('name', 'unknown'))
            if vidid is None:
                loose_gids = [
                    gid for gid, v in coco_dset.images().lookup('video_id', None, keepid=1).items()
                    if v is None
                ]
                gids = loose_gids
            else:
                gids = coco_dset.index.vidid_to_gids.get(vidid, [])

            if selected_gids is not None:
                gids = list(ub.oset(gids) & set(selected_gids))

            if not gids:
                if config['verbose']:
                    print(f'Skip video {video_name!r} with no selected images')
                continue

            sub_dpath = viz_dpath / video_name
            sub_dpath.ensuredir()

            for local_frame_index, gid in enumerate(gids):
                img = coco_dset.index.imgs[gid]
                anns = coco_dset.annots(gid=gid).objs
                pool.submit(
                    _render_frame,
                    coco_dset=coco_dset,
                    img=img,
                    anns=anns,
                    sub_dpath=sub_dpath,
                    space=config['space'],
                    channels=config['channels'],
                    draw_imgs=config['draw_imgs'],
                    draw_anns=config['draw_anns'],
                    draw_boxes=config['draw_boxes'],
                    draw_segmentations=config['draw_segmentations'],
                    draw_labels=config['draw_labels'],
                    ann_thickness=config['ann_thickness'],
                    alpha=config['alpha'],
                    ann_score_thresh=config['ann_score_thresh'],
                    draw_valid_region=config['draw_valid_region'],
                    draw_header=config['draw_header'],
                    draw_chancode=config['draw_chancode'],
                    cmap=config['cmap'],
                    any3=config['any3'],
                    max_dim=config['max_dim'],
                    min_dim=config['min_dim'],
                    resolution=config['resolution'],
                    skip_missing=config['skip_missing'],
                    skip_aggressive=config['skip_aggressive'],
                    draw_track_trails=config['draw_track_trails'],
                    local_frame_index=local_frame_index,
                    local_max_frame=len(gids),
                    video_name=video_name,
                    dset_idstr=dset_idstr,
                    verbose=config['verbose'],
                )

            for job in ub.ProgIter(pool.as_completed(), total=len(pool), desc='write frames', verbose=3):
                try:
                    job.result()
                except SkipFrame:
                    continue

            pool.jobs.clear()

        print(f'Wrote visualizations to: {viz_dpath}')


def _coerce_id_list(value):
    if value is None:
        return None
    try:
        import kwutil
        coerced = kwutil.Yaml.coerce(value)
        if isinstance(coerced, list):
            return [int(v) for v in coerced]
    except Exception:
        coerced = None
    if coerced is None:
        parts = [p.strip() for p in str(value).split(',') if p.strip()]
        return [int(p) for p in parts]
    return None


def _select_videos(coco_dset, video_spec):
    if video_spec is None:
        return None, {}
    try:
        import kwutil
        parsed = kwutil.Yaml.coerce(video_spec)
    except Exception:
        parsed = None
    if parsed is None:
        parsed = [p.strip() for p in str(video_spec).split(',') if p.strip()]
    if not isinstance(parsed, list):
        parsed = [parsed]

    vidids = set()
    vidname_lut = {v['name']: vidid for vidid, v in coco_dset.index.videos.items()}
    for item in parsed:
        if isinstance(item, int) or (isinstance(item, str) and item.isdigit()):
            vidid = int(item)
            if vidid not in coco_dset.index.videos:
                raise KeyError(f'Video id {vidid} not found in dataset')
            vidids.add(vidid)
        else:
            if item not in vidname_lut:
                raise KeyError(f'Video name {item!r} not found in dataset')
            vidids.add(vidname_lut[item])
    return sorted(vidids), vidname_lut


def _build_header_lines(coco_dset, img, video_name, dset_idstr, extra=None):
    header_lines = []
    header_lines.append(f'dataset={dset_idstr}')
    if video_name is not None:
        header_lines.append(f'video={video_name}')
    header_lines.append(f'gid={img.get("id", "?")}')
    if 'frame_index' in img:
        header_lines.append(f'frame={img.get("frame_index")}')
    if 'sensor_coarse' in img:
        header_lines.append(f'sensor={img.get("sensor_coarse")}')
    fname = img.get('file_name', None)
    if fname:
        header_lines.append(f'file={fname}')
    if extra:
        header_lines.append(extra)
    return header_lines


class SkipFrame(Exception):
    pass


class SkipChanGroup(Exception):
    pass


class TrackInfoLookup:
    """
    Helper to look up track trails in video space.
    """
    def __init__(self, dset):
        self.dset = dset

    def get_track_trail_by_video_id(self, video_id, image_id=None):
        import kwimage
        vid_images = self.dset.images(video_id=video_id)
        if image_id is not None:
            vid_gids = list(vid_images)
            max_index = vid_gids.index(image_id)
            vid_gids = vid_gids[:max_index + 1]
            vid_images = self.dset.images(vid_gids)
            max_frame_index = self.dset.imgs[image_id]['frame_index']
        else:
            max_frame_index = None

        vid_annots = vid_images.annots
        track_ids = set(ub.flatten(vid_annots.lookup('track_id', None)))
        track_ids -= {None}
        trails = []
        for track_id in track_ids:
            track_aids = self.dset.index.trackid_to_aids[track_id]
            vidspace_boxes = []
            track_colors = []
            track = self.dset.index.tracks[track_id]
            track_color = track.get('color', None)
            track_gids = []
            default_color = kwimage.Color.random(rng=track_id)
            for aid in track_aids:
                ann = self.dset.index.anns[aid]
                gid = ann['image_id']
                img = self.dset.index.imgs[gid]
                if max_frame_index is not None:
                    if img['frame_index'] >= max_frame_index:
                        continue
                bbox = ann['bbox']
                vid_from_img = kwimage.Affine.coerce(img.get('warp_img_to_vid', None))
                imgspace_box = kwimage.Boxes([bbox], 'xywh')
                vidspace_box = imgspace_box.warp(vid_from_img)
                vidspace_boxes.append(vidspace_box)
                if track_color is None:
                    color = ann.get('color', default_color)
                else:
                    color = track_color
                track_colors.append(color)
                track_gids.append(gid)
            if len(vidspace_boxes):
                vidspace_boxes = kwimage.Boxes.concatenate(vidspace_boxes)
                motion_path = {
                    'vidspace_boxes': vidspace_boxes,
                    'track_gids': track_gids,
                    'track_aids': track_aids,
                    'track_colors': track_colors,
                }
                trail = {
                    'track_id': track_id,
                    'motion_path': motion_path,
                }
                for key in ['role', 'color', 'thickness']:
                    if key in track:
                        trail[key] = track[key]
                trails.append(trail)
        return trails


def _resolve_channel_groups(coco_img, channels, verbose, request_grouped_bands,
                            any3):
    from delayed_image import channel_spec
    import kwcoco
    if channels is not None:
        if isinstance(channels, list):
            channels = ','.join(channels)
        channels = channel_spec.ChannelSpec.coerce(channels)
        chan_groups = [{'chan': chan_obj} for chan_obj in channels.streams()]
    else:
        channels = coco_img.channels
        if channels is None:
            chan_groups = [{'pname': 'null', 'chan': None}]
            return chan_groups

        if request_grouped_bands == 'default':
            request_grouped_bands = ['red|green|blue', 'r|g|b', 'ir']

        for cand in request_grouped_bands:
            cand = kwcoco.FusedChannelSpec.coerce(cand)
            has_cand = (channels & cand).numel() == cand.numel()
            if has_cand:
                channels = channels - cand
                channels = channels + cand

        initial_groups = channels.streams()
        chan_groups = []
        for group in initial_groups:
            if group.numel() > 3:
                if group.numel() > 8:
                    group = group.normalize()[0:3]
                    chan_groups.append({'chan': group})
                else:
                    for part in group:
                        chan_groups.append({'chan': kwcoco.FusedChannelSpec.coerce(part)})
            else:
                chan_groups.append({'chan': group})

    for row in chan_groups:
        row['pname'] = row['chan'].path_sanitize() if row['chan'] is not None else 'null'

    if any3:
        if any3 == 'only':
            chan_groups = []
        avail_channels = channels.fuse() if channels is not None else None
        if avail_channels is not None:
            common_visualizers = list(map(kwcoco.FusedChannelSpec.coerce, [
                'red|green|blue', 'r|g|b', 'pan', 'panchromatic']))
            found = None
            for cand in common_visualizers:
                flag = (cand & avail_channels).spec == cand.spec
                if flag:
                    found = cand
                    break
            if found is None:
                first3 = avail_channels.as_list()[0:3]
                found = kwcoco.FusedChannelSpec.coerce('|'.join(first3))
            chan_groups.append({'pname': 'any3', 'chan': found})

    return chan_groups


def _render_frame(coco_dset,
                  img,
                  anns,
                  sub_dpath,
                  space,
                  channels,
                  draw_imgs,
                  draw_anns,
                  draw_boxes,
                  draw_segmentations,
                  draw_labels,
                  ann_thickness,
                  alpha,
                  ann_score_thresh,
                  draw_valid_region,
                  draw_header,
                  draw_chancode,
                  cmap,
                  any3,
                  max_dim,
                  min_dim,
                  resolution,
                  skip_missing,
                  skip_aggressive,
                  draw_track_trails,
                  local_frame_index,
                  local_max_frame,
                  video_name,
                  dset_idstr,
                  verbose):
    import kwimage

    if local_max_frame is None:
        num_digits = 8
    else:
        num_digits = int(math.log10(max(local_max_frame, 1))) + 1
    frame_id = f'{local_frame_index:0{num_digits}d}'

    coco_img = coco_dset.coco_image(img['id'])
    finalize_opts = {
        'interpolation': 'linear',
        'nodata_method': 'float',
    }

    if resolution is None:
        factor = 1
    else:
        factor = coco_img._scalefactor_for_resolution(space=space, resolution=resolution)
    warp_viz_from_space = kwimage.Affine.scale(factor)

    delayed = coco_img.imdelay(space=space, resolution=resolution, **finalize_opts)
    warp_vid_from_img = coco_img.warp_vid_from_img
    if space == 'video':
        warp_viz_from_img = warp_viz_from_space @ warp_vid_from_img
    else:
        warp_viz_from_img = warp_viz_from_space

    if verbose > 2:
        print(f'Render frame {frame_id} ({img.get("name", "unnamed")})')

    header_lines = _build_header_lines(coco_dset, img, video_name, dset_idstr)

    chan_groups = _resolve_channel_groups(
        coco_img, channels, verbose, request_grouped_bands='default', any3=any3)

    img_view_dpath = sub_dpath / '_imgs'
    ann_view_dpath = sub_dpath / '_anns'

    dets = None
    trails = None
    if draw_anns and anns:
        dets = kwimage.Detections.from_coco_annots(anns, dset=coco_dset)
        if ann_score_thresh:
            flags = [float(ann.get('score', 1)) > ann_score_thresh for ann in anns]
            dets = dets.compress(flags)
        dets = dets.warp(warp_viz_from_img)

    if draw_track_trails and coco_img.img.get('video_id', None) is not None:
        if space != 'video':
            raise ValueError('draw_track_trails is only supported in video space')
        tilut = TrackInfoLookup(coco_dset)
        trails = tilut.get_track_trail_by_video_id(video_id=coco_img.img['video_id'])

    valid_image_poly = None
    if draw_valid_region:
        valid_region = img.get('valid_region', None)
        if valid_region:
            valid_image_poly = kwimage.MultiPolygon.coerce(valid_region)
            valid_image_poly = valid_image_poly.warp(warp_viz_from_img)

    viz_scale_factor = _compute_viz_scale(delayed.dsize, min_dim, max_dim)
    if viz_scale_factor != 1:
        viz_warp = kwimage.Affine.scale(viz_scale_factor)
        delayed = delayed.warp(viz_warp)
        if dets is not None:
            dets.warp(viz_warp, inplace=True)
        if valid_image_poly is not None:
            valid_image_poly = valid_image_poly.warp(viz_warp)
        if trails is not None:
            for trail in trails:
                trail['motion_path']['vidspace_boxes'] = trail['motion_path']['vidspace_boxes'].warp(viz_warp)

    for chan_row in chan_groups:
        try:
            _draw_channel_group(
                coco_dset=coco_dset,
                frame_id=frame_id,
                img=img,
                dets=dets,
                trails=trails,
                ann_view_dpath=ann_view_dpath,
                img_view_dpath=img_view_dpath,
                delayed=delayed,
                chan_row=chan_row,
                finalize_opts=finalize_opts,
                skip_missing=skip_missing,
                skip_aggressive=skip_aggressive,
                cmap=cmap,
                header_lines=header_lines,
                valid_image_poly=valid_image_poly,
                draw_imgs=draw_imgs,
                draw_anns=draw_anns,
                draw_boxes=draw_boxes,
                draw_labels=draw_labels,
                draw_segmentations=draw_segmentations,
                draw_header=draw_header,
                draw_chancode=draw_chancode,
                ann_thickness=ann_thickness,
                alpha=alpha,
                verbose=verbose,
            )
        except SkipChanGroup:
            if verbose > 1:
                print(f'Skip channel group {chan_row.get("pname")}')


def _compute_viz_scale(dsize, min_dim, max_dim):
    viz_scale_factor = 1.0
    if min_dim is not None:
        try:
            chan_min_dim = min(dsize) * viz_scale_factor
            if chan_min_dim < min_dim:
                viz_scale_factor *= min_dim / chan_min_dim
        except TypeError:
            viz_scale_factor = 1.0
    if max_dim is not None:
        try:
            chan_max_dim = max(dsize) * viz_scale_factor
            if chan_max_dim > max_dim:
                viz_scale_factor *= max_dim / chan_max_dim
        except TypeError:
            viz_scale_factor = 1.0
    return viz_scale_factor


def _draw_channel_group(coco_dset,
                        frame_id,
                        img,
                        dets,
                        trails,
                        ann_view_dpath,
                        img_view_dpath,
                        delayed,
                        chan_row,
                        finalize_opts,
                        skip_missing,
                        skip_aggressive,
                        cmap,
                        header_lines,
                        valid_image_poly,
                        draw_imgs,
                        draw_anns,
                        draw_boxes,
                        draw_labels,
                        draw_segmentations,
                        draw_header,
                        draw_chancode,
                        ann_thickness,
                        alpha,
                        verbose):
    import kwimage
    import numpy as np
    import kwcoco

    chan_pname = chan_row['pname']
    chan_group_obj = chan_row['chan']

    img_chan_dpath = img_view_dpath / chan_pname
    ann_chan_dpath = ann_view_dpath / chan_pname

    if chan_group_obj is not None:
        chan_list = chan_group_obj.parsed
        chan_group = chan_group_obj.spec
        chan_pname2 = kwcoco.FusedChannelSpec.coerce(chan_group).path_sanitize(maxlen=10)
        prefix = '_'.join([frame_id, chan_pname2])
    else:
        chan_group = None
        chan_list = None
        prefix = '_'.join([frame_id, 'null'])

    img_name = img.get('name', None)
    if img_name is None:
        fname = img.get('file_name', 'image')
        img_name = ub.Path(fname).stem
    view_img_fpath = img_chan_dpath / (prefix + '_' + img_name + '.view_img.jpg')
    view_ann_fpath = ann_chan_dpath / (prefix + '_' + img_name + '.view_ann.jpg')

    if chan_group_obj is not None and delayed.channels is not None:
        chan = delayed.take_channels(chan_group)
    else:
        chan = delayed
    chan = chan.prepare().optimize()
    raw_canvas = canvas = chan.finalize(**finalize_opts)

    if verbose > 1:
        print(f'raw_canvas.shape = {raw_canvas.shape}')

    if skip_missing and np.all(np.isnan(raw_canvas)):
        if skip_aggressive:
            raise SkipFrame
        raise SkipChanGroup

    dmax = np.nanmax(raw_canvas)
    if dmax > 1.0:
        mask = ~np.isnan(raw_canvas)
        canvas = kwimage.normalize_intensity(raw_canvas, mask=mask, params={
            'high': 0.90,
            'mid': 0.5,
            'low': 0.01,
            'mode': 'linear',
        })
        canvas = np.clip(canvas, 0, None)

    canvas = kwimage.nodata_checkerboard(canvas, on_value=0.3)

    if chan_group_obj is not None:
        chan_names = chan_row['chan'].to_list()
    else:
        chan_names = []

    channel_colors = []
    for cname in chan_names:
        if cname in coco_dset.index.name_to_cat:
            cat = coco_dset.index.name_to_cat[cname]
            channel_colors.append(cat.get('color', None))
        else:
            channel_colors.append(None)

    if any(c is not None for c in channel_colors):
        if kwimage.num_channels(canvas) != 1:
            canvas = perchannel_colorize(canvas, channel_colors=channel_colors)
            canvas = canvas[..., 0:3]

    if cmap is not None:
        if kwimage.num_channels(canvas) == 1:
            import matplotlib as mpl
            import kwutil

            chan_to_cmap = kwutil.Yaml.coerce(cmap)
            if isinstance(chan_to_cmap, str):
                chan_to_cmap = {'__default__': chan_to_cmap}
            elif not isinstance(chan_to_cmap, dict):
                raise TypeError(f'Did not coerce chan_to_cmap: {type(chan_to_cmap)} correctly')

            default_cmap_name = chan_to_cmap.get('__default__', 'viridis')
            cmap_name = chan_to_cmap.get(chan_group, default_cmap_name)

            try:
                import matplotlib.cm  # NOQA
                cmap_ = mpl.cm.get_cmap(cmap_name)
            except AttributeError:
                cmap_ = mpl.colormaps[cmap_name]

            canvas = np.nan_to_num(canvas)
            if len(canvas.shape) == 3:
                canvas = canvas[..., 0]
            canvas = cmap_(canvas)[..., 0:3].astype(np.float32)

    canvas = ensure_false_color(canvas)
    canvas = kwimage.ensure_uint255(canvas)

    if len(canvas.shape) > 2 and canvas.shape[2] > 4:
        canvas = canvas[..., 0]

    chan_header_lines = header_lines.copy()
    chan_header_lines.append(str(chan_group))
    header_text = '\n'.join(chan_header_lines)

    if valid_image_poly is not None:
        if any([p.data['exterior'].data.size for p in valid_image_poly.data]):
            canvas = valid_image_poly.draw_on(
                canvas, color='kitware_green', fill=False, border=True, alpha=alpha)

    if draw_imgs:
        img_chan_dpath.ensuredir()
        img_canvas = kwimage.ensure_uint255(canvas, copy=True)
        if draw_chancode and chan_group is not None:
            img_canvas = kwimage.draw_text_on_image(
                img_canvas, chan_group, (1, 2), valign='top', color='lime', border=3)
        if draw_header:
            img_header = kwimage.draw_header_text(image=img_canvas, text=header_text,
                                                  stack=False, fit='shrink')
            img_canvas = kwimage.stack_images([img_header, img_canvas], axis=0)
        kwimage.imwrite(view_img_fpath, img_canvas)

    if draw_anns:
        ann_chan_dpath.ensuredir()
        ann_canvas = kwimage.ensure_float01(canvas, copy=True)
        if dets is not None:
            ann_canvas = dets.draw_on(
                ann_canvas,
                boxes=bool(draw_boxes),
                sseg=bool(draw_segmentations),
                labels=bool(draw_labels),
                alpha=alpha,
                thickness=ann_thickness,
            )
        if trails:
            for trail in trails:
                thickness = trail.get('thickness', ann_thickness)
                trail_cxy = trail['motion_path']['vidspace_boxes'].xy_center
                trail_colors = trail['motion_path']['track_colors'][1:]
                ann_canvas = draw_polyline_on_image(
                    ann_canvas, trail_cxy, color=trail_colors, thickness=thickness)
        ann_canvas = kwimage.ensure_uint255(ann_canvas)
        if draw_chancode and chan_group is not None:
            ann_canvas = kwimage.draw_text_on_image(
                ann_canvas, chan_group, (1, 2), valign='top', color='lime', border=3)
        if draw_header:
            ann_header = kwimage.draw_header_text(image=ann_canvas, text=header_text,
                                                  stack=False, fit='shrink')
            ann_canvas = kwimage.stack_images([ann_header, ann_canvas], axis=0)
        kwimage.imwrite(view_ann_fpath, ann_canvas)


def draw_polyline_on_image(image, xy_pts, color=None, thickness=1):
    """
    Draw a polyline on an image.
    """
    import kwimage
    if len(xy_pts) > 1:
        pts1 = xy_pts[0:-1]
        pts2 = xy_pts[1:]
        image = kwimage.draw_line_segments_on_image(
            image, pts1, pts2, color=color, thickness=thickness)
    return image


# Adapted from dev/to_port/util_kwimage.py::perchannel_colorize
# Originally ported from geowatch.utils.util_kwimage.
def perchannel_colorize(data, channel_colors=None):
    import kwimage
    import numpy as np

    num_channels = data.shape[2]

    if len(data.shape) == 2:
        data = data[None, :, :]

    existing_colors = [
        kwimage.Color.coerce(c).as01()
        for c in channel_colors if c is not None
    ]

    fill_colors = kwimage.Color.distinct(
        num_channels - len(existing_colors),
        existing=existing_colors)
    fill_color_iter = iter(fill_colors)

    resolved_channel_colors = []
    for c in channel_colors:
        if c is None:
            c = next(fill_color_iter)
        else:
            c = kwimage.Color.coerce(c).as01()
        resolved_channel_colors.append(c)

    sumtotal = np.nansum(data, axis=2)
    sumtotal[np.isnan(sumtotal)] = 1
    sumtotal[sumtotal == 0] = 1
    sumtotal = np.maximum(sumtotal, 1)
    layers = []
    for cidx in range(num_channels):
        chan = data[:, :, cidx]
        alpha = chan / sumtotal
        color = resolved_channel_colors[cidx]
        layer = np.empty(tuple(chan.shape) + (4,))
        layer[..., 3] = alpha
        layer[..., 0:3] = color
        layers.append(layer)

    background = np.zeros_like(layer)
    background[..., 3] = 1
    layers.append(background)
    colormask = kwimage.overlay_alpha_layers(layers, keepalpha=False)
    return colormask


# Adapted from dev/to_port/util_kwimage.py::ensure_false_color
# Originally ported from geowatch.utils.util_kwimage.
def ensure_false_color(canvas, method='ortho'):
    import kwarray
    import numpy as np
    import kwimage
    canvas = kwarray.atleast_nd(canvas, 3)

    if canvas.shape[2] in {1, 3}:
        rgb_canvas = canvas
    else:
        if method == 'ortho':
            rng = kwarray.ensure_rng(canvas.shape[2])
            seedmat = rng.rand(canvas.shape[2], 3).T
            h, tau = np.linalg.qr(seedmat, mode='raw')
            false_colored = (canvas @ h)
            rgb_canvas = kwarray.normalize(false_colored)
        elif method.lower() == 'pca':
            import sklearn
            dims = canvas.shape[0:2]
            in_channels = canvas.shape[2]

            if in_channels > 1:
                model = sklearn.decomposition.PCA(1)
                X = canvas.reshape(-1, in_channels)
                X_ = model.fit_transform(X)
                gray = X_.reshape(dims)
                rgb_canvas = kwimage.make_heatmask(gray, with_alpha=1)[:, :, 0:3]
            else:
                gray = canvas.reshape(dims)
                rgb_canvas = gray
        else:
            raise KeyError(method)
    return rgb_canvas


__cli__ = CocoVisualizeVideosCLI

if __name__ == '__main__':
    __cli__.main()
