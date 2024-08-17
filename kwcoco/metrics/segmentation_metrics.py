#!/usr/bin/env python3
"""
Experimental support for kwcoco-only.

Compute semantic segmentation evaluation metrics

TODO::
- RRMSE (relative root mean squared error) RMSE normalized by root mean sqare value where each residual is scaled against the actual value
  sqrt((1 / n) * sum((y - y_hat) ** 2) / sum(y ** 2))
"""
import json
import kwarray
import kwcoco
import kwimage
import numpy as np
import os
import pandas as pd
import sklearn.metrics as skm
import ubelt as ub
import warnings
from kwcoco.coco_evaluator import CocoSingleResult
from kwcoco.metrics.confusion_vectors import BinaryConfusionVectors
from kwcoco.metrics.confusion_measures import OneVersusRestMeasureCombiner
from kwcoco.metrics.confusion_vectors import OneVsRestConfusionVectors
from kwcoco.metrics.confusion_measures import MeasureCombiner
# from kwcoco.metrics.confusion_measures import PerClass_Measures
from kwcoco.metrics.confusion_measures import Measures
from typing import Dict
import scriptconfig as scfg
from shapely.ops import unary_union

# from geowatch.utils import kwcoco_extensions
# from geowatch import heuristics

try:
    from line_profiler import profile
except Exception:
    profile = ub.identity


# The colors I traditionally use for truth and predictions
# TRUE_GREEN = 'limegreen'
# PRED_BLUE = 'dodgerblue'

# If we have a recent kwimage we can use kitware colors, which look pretty good
# in these roles too.
TRUE_GREEN = 'kitware_green'
PRED_BLUE = 'kitware_blue'

CONFUSION_COLOR_SCHEME = {
    'TN': 'black',
    # 'TP': 'white',
    # 'TP': 'snow',  # off white
    'TP': 'whitesmoke',  # off white
    'FN': 'teal',
    'FP': 'red',
}


# TODO: parameterize these class categories
# TODO: remove and generalize before porting to kwcoco
IGNORE_CLASSNAMES = {'ignore', 'Unknown'}
BACKGROUND_CLASSES = {'background'}
NEGATIVE_CLASSES = {'negative'}
UNDISTINGUISHED_CLASSES =  {'positive'}
CONTEXT_CLASSES = {}


class SegmentationEvalConfig(scfg.DataConfig):
    """
    Evaluation script for change/segmentation task
    """
    true_dataset = scfg.Value(None, help='path to the groundtruth dataset')
    pred_dataset = scfg.Value(None, help='path to the predicted dataset')
    eval_dpath = scfg.Value(None, help='directory to dump results')
    eval_fpath = scfg.Value(None, help='path to dump result summary')
    # options
    draw_curves = scfg.Value('auto', help='flag to draw curves or not')
    draw_heatmaps = scfg.Value('auto', help='flag to draw heatmaps or not')

    draw_legend = scfg.Value(True, help='enable/disable the class legend')
    draw_weights = scfg.Value(False, help='enable/disable pixel weight visualization')

    score_space = scfg.Value('auto', help='can score in image or video space. If auto, chooses video if there are any, otherwise image')
    resolution = scfg.Value(None, help='if specified, override the default resolution to score at')

    workers = scfg.Value('auto', help='number of parallel scoring workers')
    draw_workers = scfg.Value('auto', help='number of parallel drawing workers')
    viz_thresh = scfg.Value('auto', help='visualization threshold')
    balance_area = scfg.Value(False, isflag=True, help='upweight small instances, downweight large instances')
    # thresh_bins = scfg.Value(128 * 128, help='threshold resolution, default is high, generally ok to lower')
    thresh_bins = scfg.Value(32 * 32, help='threshold resolution.')
    salient_channel = scfg.Value('salient', help='channel that is the positive class')


def main(cmdline=True, **kwargs):
    """
    Entry point: todo: doctest and CLI structure

    todo: ProcessContext to track resource usage
    """
    full_config = SegmentationEvalConfig.cli(
        cmdline=cmdline, data=kwargs, strict=True)
    import rich
    rich.print('full_config = {}'.format(ub.urepr(full_config, nl=1)))

    full_config = ub.udict(full_config)
    true_coco = kwcoco.CocoDataset.coerce(full_config['true_dataset'])
    pred_coco = kwcoco.CocoDataset.coerce(full_config['pred_dataset'])
    eval_fpath = full_config['eval_fpath']
    eval_dpath = full_config['eval_dpath']

    config = full_config - {
        'true_dataset', 'pred_dataset', 'eval_dpath', 'eval_fpath'}
    evaluate_segmentations(true_coco, pred_coco, eval_dpath, eval_fpath,
                           config)


@profile
def single_image_segmentation_metrics(pred_coco_img, true_coco_img,
                                      true_classes, true_dets, video1=None,
                                      thresh_bins=None, config=None):
    """
    Args:
        true_coco_img (kwcoco.CocoImage): detatched true coco image

        pred_coco_img (kwcoco.CocoImage): detatched predicted coco image

        thresh_bins (int): if specified rounds scores into this many bins
            to make calculating metrics more efficient

        config (None | dict): see usage

    CommandLine:
        xdoctest -m geowatch.tasks.fusion.evaluate single_image_segmentation_metrics

    Example:
        >>> from kwcoco.metrics.segmentation_metrics import *  # NOQA
        >>> from kwcoco.coco_evaluator import CocoEvaluator
        >>> from kwcoco.demo.perterb import perterb_coco
        >>> import kwcoco
        >>> # TODO: kwcoco demodata with easy dummy heatmap channels
        >>> true_coco = kwcoco.CocoDataset.demo('vidshapes2', image_size=(64, 64))
        >>> # Score an image against itself
        >>> true_coco_img = true_coco.images()[0:1].coco_images[0]
        >>> pred_coco_img = true_coco.images()[0:1].coco_images[0]
        >>> config = {}
        >>> true_dets = true_coco_img.annots().detections
        >>> video1 = true_coco_img.video
        >>> true_classes = true_coco.object_categories()
        >>> config['salient_channel'] = 'r'  # pretend red is the salient channel
        >>> thresh_bins = np.linspace(0, 255, 1024)
        >>> info = single_image_segmentation_metrics(
        >>>    pred_coco_img, true_coco_img, true_classes, true_dets,
        >>>    thresh_bins=thresh_bins, config=config, video1=video1)
    """
    if config is None:
        config = {}

    viz_thresh = config.get('viz_thresh', 'auto')
    score_space = config.get('score_space', 'auto')
    resolution = config.get('resolution', None)
    balance_area = config.get('balance_area', False)

    if score_space == 'auto':
        pred_vidid = pred_coco_img.img.get('video_id', None)
        true_vidid = true_coco_img.img.get('video_id', None)
        if true_vidid is not None or pred_vidid is not None:
            score_space = 'video'
        else:
            score_space = 'image'

    true_gid = true_coco_img.img['id']
    pred_gid = pred_coco_img.img['id']

    if thresh_bins is not None:
        if isinstance(thresh_bins, int):
            left_bin_edges = np.linspace(0, 1, thresh_bins)
        else:
            left_bin_edges = thresh_bins
    else:
        left_bin_edges = None

    img1 = true_coco_img.img

    if score_space == 'image':
        dsize = np.array((img1['width'], img1['height']))
    elif score_space == 'video':
        dsize = np.array((video1['width'], video1['height']))
    else:
        raise KeyError(score_space)

    if resolution is None:
        scale = None
    else:
        try:
            scale = true_coco_img._scalefactor_for_resolution(resolution=resolution, space=score_space)
        except Exception as ex:
            print(f'warning: ex={ex}')
            scale = None

    if scale is not None:
        dsize = np.ceil(np.array(dsize) * np.array(scale)).astype(int)

    row = {
        'true_gid': true_gid,
        'pred_gid': pred_gid,
    }
    if video1 is not None:
        row['video'] = video1['name']

    shape = dsize[::-1]
    info = {
        'row': row,
        'shape': shape,
    }

    # TODO: parameterize these class categories
    # TODO: remove and generalize before porting to kwcoco
    ignore_classes = IGNORE_CLASSNAMES
    background_classes = BACKGROUND_CLASSES
    undistinguished_classes = UNDISTINGUISHED_CLASSES
    context_classes = CONTEXT_CLASSES
    negative_classes = NEGATIVE_CLASSES
    # HACK! FIXME: There needs to be a clear definition of what classes are
    # scored and which are not.
    background_classes = background_classes | negative_classes
    """
    The above heuristics should roughly be:

        * ignore_classes - ignore, Unknown
        * background_classes - background, negative
        * undistinguished_classes - positive
        * context_classes - unused
    """

    # Determine what true/predicted categories are in common
    predicted_classes = []
    for stream in pred_coco_img.channels.streams():
        have = stream.intersection(true_classes)
        predicted_classes.extend(have.parsed)

    classes_of_interest = ub.oset(predicted_classes) - (
        negative_classes | background_classes | ignore_classes |
        undistinguished_classes)

    # Determine if saliency has been predicted
    salient_class = config.get('salient_channel', 'salient')
    has_saliency = salient_class in pred_coco_img.channels

    # Load ground truth annotations
    if score_space == 'video':
        warp_img_to_vid = kwimage.Affine.coerce(
            true_coco_img.img.get('warp_img_to_vid', {'type': 'affine'}))
        true_dets = true_dets.warp(warp_img_to_vid)
    if scale is not None:
        true_dets = true_dets.scale(scale)
    info['true_dets'] = true_dets
    true_cidxs = true_dets.data['class_idxs']
    true_ssegs = true_dets.data['segmentations']
    true_catnames = list(ub.take(true_dets.classes.idx_to_node, true_cidxs))

    # NOTE: The exact definition of how we build the "truth" segmentation mask
    # is up for debate. I think this is a reasonable definition, but this needs
    # to be reviewed. It also likely needs updating to become general and
    # remove the need for heuristics.

    # We might need to:
    #     * add in a per-category weight canvas. This lets us say we can ignore
    #     clas A when scoring class B. Is there an example where this is
    #     relevant?

    # Does negative get moved to the background or scored?
    # Currently I'm just moving it to the background

    # How do we distinguish that

    # TODO:
    # Use the "valid_polygon" to zero out evaluations in invalid regions
    # Also use nan values in the predictions to do the same.
    # Combine these two measures.

    # Create a truth "panoptic segmentation" style mask for each task
    if has_saliency:
        # Truth for saliency-task
        true_saliency = np.zeros(shape, dtype=np.uint8)
        saliency_weights = np.ones(shape, dtype=np.float32)

        sseg_groups = {
            'ignore': [],
            'context': [],
            'foreground': [],
            'background': [],
        }
        for true_sseg, true_catname in zip(true_ssegs, true_catnames):
            if true_catname in background_classes:
                key = 'background'
            elif true_catname in ignore_classes:
                key = 'ignore'
            elif true_catname in context_classes:
                key = 'context'
            else:
                key = 'foreground'
            sseg_groups[key].append(true_sseg)

        if balance_area:
            if len(sseg_groups['foreground']):
                fg_poly = unary_union([p.to_shapely() for p in sseg_groups['foreground']])
                unit_sseg_share = fg_poly.area / len(sseg_groups['foreground'])
            else:
                unit_sseg_share = 1

        # background should be background, do nothing with it
        sseg_groups['background']
        # Ignore context classes in saliency
        # Ignore no-activity and post-construction, ignore, and Unknown
        for true_sseg in sseg_groups['ignore']:
            saliency_weights = true_sseg.fill(saliency_weights, value=0)
        for true_sseg in sseg_groups['context']:
            # saliency_weights = true_sseg.fill(saliency_weights, value=0)
            ...
        # Score positive, site prep, and active construction.
        for true_sseg in sseg_groups['foreground']:
            true_saliency = true_sseg.fill(true_saliency, value=1)
            if balance_area:
                # Fill in the weights to upweight smaller areas.
                instance_weight = unit_sseg_share / true_sseg.area
                saliency_weights = true_sseg.fill(saliency_weights, value=instance_weight)
        # saliency_weights = saliency_weights / saliency_weights.max()

    if classes_of_interest:
        # Truth for class-task
        catname_to_true: Dict[str, np.ndarray] = {
            catname: np.zeros(shape, dtype=np.float32)
            for catname in classes_of_interest
        }
        class_weights = np.ones(shape, dtype=np.float32)
        initial_total_weight = class_weights.size

        sseg_groups = {
            'background': [],
            'ignore': [],
            'undistinguished': [],
            'foreground': [],
        }
        for true_sseg, true_catname in zip(true_ssegs, true_catnames):
            if true_catname in background_classes:
                key = 'background'
            elif true_catname in ignore_classes:
                key = 'ignore'
            elif true_catname in undistinguished_classes:
                key = 'undistinguished'
            else:
                key = 'foreground'
                true_sseg.meta['true_catname'] = true_catname
            sseg_groups[key].append(true_sseg)

        if balance_area:
            if len(sseg_groups['foreground']):
                fg_poly = unary_union([p.to_shapely() for p in sseg_groups['foreground']])
                unit_sseg_share = fg_poly.area / len(sseg_groups['foreground'])
            else:
                unit_sseg_share = 1

        true_sseg.area / initial_total_weight

        # background should be background, do nothing with it
        sseg_groups['background']
        # Ignore context classes in saliency
        # Ignore no-activity and post-construction, ignore, and Unknown
        for true_sseg in sseg_groups['ignore']:
            class_weights = true_sseg.fill(class_weights, value=0)
        for true_sseg in sseg_groups['undistinguished']:
            class_weights = true_sseg.fill(class_weights, value=0)
        # Score positive, site prep, and active construction.
        for true_sseg in sseg_groups['foreground']:
            true_catname = true_sseg.meta['true_catname']
            if balance_area:
                # Fill in the weights to upweight smaller areas.
                instance_weight = unit_sseg_share / true_sseg.area
                class_weights = true_sseg.fill(class_weights, value=instance_weight)
            catname_to_true[true_catname] = true_sseg.fill(catname_to_true[true_catname], value=1)

        # Hack:
        # normalize to 0-1, this downweights the background too much, but
        # I think fixes a upstream issue. Remove (or justify?) if possible.
        # class_weights = class_weights / class_weights.max()

    if classes_of_interest:
        # handle multiclass case
        pred_chan_of_interest = '|'.join(classes_of_interest)
        delayed_probs = pred_coco_img.imdelay(
            pred_chan_of_interest, space=score_space,
            resolution=resolution, nodata_method='float').as_xarray()
        # Do we need xarray anymore?

        class_probs = delayed_probs.finalize()
        invalid_mask = np.isnan(class_probs).all(axis=2)

        # import xdev
        # with xdev.embed_on_exception_context(before_embed=util_progress.ProgressManager.stopall):
        class_weights[invalid_mask] = 0

        catname_to_prob = {}
        cx_to_binvecs = {}
        for cx, cname in enumerate(classes_of_interest):
            is_true = catname_to_true[cname]
            score = class_probs.loc[:, :, cname].data.copy()
            invalid_mask = np.isnan(score)
            weights = class_weights.copy()
            weights[invalid_mask] = 0
            score[invalid_mask] = 0

            pred_score = score.ravel()
            if left_bin_edges is not None:
                # round scores down to the nearest bin
                rounded_idx = np.searchsorted(left_bin_edges, pred_score)
                pred_score = left_bin_edges[rounded_idx]

            catname_to_prob[cname] = score
            bin_data = {
                # is_true denotes if the true class of the item is the
                # category of interest.
                'is_true': is_true.ravel(),
                'pred_score': pred_score,
                'weight': weights.ravel(),
            }
            bin_data = kwarray.DataFrameArray(bin_data)
            bin_cfsn = BinaryConfusionVectors(bin_data, cx, classes_of_interest)
            # TODO: use me?
            # bin_measures = bin_cfsn.measures()
            # bin_measures.summary()
            cx_to_binvecs[cname] = bin_cfsn
        ovr_cfns = OneVsRestConfusionVectors(cx_to_binvecs, classes_of_interest)
        class_measures = ovr_cfns.measures()
        row['mAP'] = class_measures['mAP']
        row['mAUC'] = class_measures['mAUC']
        info.update({
            'class_weights': class_weights,
            'class_measures': class_measures,
            'catname_to_true': catname_to_true,
            'catname_to_prob': catname_to_prob,
        })

    if has_saliency:
        # TODO: consolidate this with above class-specific code
        salient_delay = pred_coco_img.imdelay(salient_class, space=score_space,
                                              resolution=resolution,
                                              nodata_method='float')
        salient_prob = salient_delay.finalize(nodata_method='float')[..., 0]
        salient_prob_orig = salient_prob.copy()
        invalid_mask = np.isnan(salient_prob)

        salient_prob[invalid_mask] = 0
        try:
            saliency_weights[invalid_mask] = 0
        except Exception:
            print(f'invalid_mask.shape={invalid_mask.shape}')
            print(f'saliency_weights.shape={saliency_weights.shape}')
            raise

        pred_score = salient_prob.ravel()
        if left_bin_edges is not None:
            rounded_idx = np.searchsorted(left_bin_edges, pred_score)
            pred_score = left_bin_edges[rounded_idx]

        bin_cfns = BinaryConfusionVectors(kwarray.DataFrameArray({
            'is_true': true_saliency.ravel(),
            'pred_score': pred_score,
            'weight': saliency_weights.ravel().astype(np.float32),
        }))
        salient_measures = bin_cfns.measures()
        salient_summary = salient_measures.summary()

        salient_metrics = {
            'salient_' + k: v
            for k, v in ub.dict_isect(salient_summary, {
                'ap', 'auc', 'max_f1'}).items()
        }
        try:
            # Requires kwcoco 0.8.3
            salient_metrics['realpos_total'] = salient_measures['realpos_total']
            salient_metrics['realneg_total'] = salient_measures['realneg_total']
            submeasures = salient_measures['max_f1_submeasures']
            salient_metrics['salient_max_f1_thresh'] = submeasures['thresh']
            salient_metrics['salient_max_f1_ppv'] = submeasures['ppv']
            salient_metrics['salient_max_f1_tpr'] = submeasures['tpr']
            salient_metrics['salient_max_f1_fpr'] = submeasures['fpr']
            salient_metrics['salient_max_f1_tnr'] = submeasures['tnr']
        except Exception:
            ...
        row.update(salient_metrics)

        info.update({
            'salient_measures': salient_measures,
            'salient_prob': salient_prob_orig,
            'true_saliency': true_saliency,
        })

        if 1:
            maximized_info = salient_measures.maximized_thresholds()

            # This cherry-picks a threshold per image!
            if viz_thresh == 'auto':
                cherry_picked_thresh = maximized_info['f1']['thresh']
                saliency_thresh = cherry_picked_thresh
            else:
                saliency_thresh = viz_thresh
            pred_saliency = salient_prob > saliency_thresh

            y_true = true_saliency.ravel()
            y_pred = pred_saliency.ravel()
            sample_weight = saliency_weights.ravel()
            mat = skm.confusion_matrix(y_true, y_pred, labels=np.array([0, 1]),
                                       sample_weight=sample_weight)
            info.update({
                'mat': mat,
                'pred_saliency': pred_saliency,
                'saliency_thresh': saliency_thresh,
                'saliency_weights': saliency_weights,
            })

    # TODO: look at the category ranking at each pixel by score.
    # Is there a generalization of a confusion matrix to a ranking tensor?
    # if 0:
    #     # TODO: Reintroduce hard-polygon segmentation scoring?
    #     # Score hard-threshold predicted annotations
    #     # SCORE PREDICTED ANNOTATIONS
    #     # Create a pred "panoptic segmentation" style mask
    #     pred_saliency = np.zeros(shape, dtype=np.uint8)
    #     pred_dets = pred_coco.annots(gid=gid2).detections
    #     for pred_sseg in pred_dets.data['segmentations']:
    #         pred_saliency = pred_sseg.fill(pred_saliency, value=1)
    return info


@ub.memoize
def _memo_legend(label_to_color):
    import kwplot
    legend_img = kwplot.make_legend_img(label_to_color)
    return legend_img


def draw_confusion_image(pred, target):
    canvas = np.zeros_like(pred)
    np.putmask(canvas, (target == 0) & (pred == 0), 0)  # true-neg
    np.putmask(canvas, (target == 1) & (pred == 1), 1)  # true-pos
    np.putmask(canvas, (target == 1) & (pred == 0), 2)  # false-neg
    np.putmask(canvas, (target == 0) & (pred == 1), 3)  # false-pos
    return canvas


@profile
def colorize_class_probs(probs, classes):
    """
    probs = pred_cat_ohe
    classes = pred_classes
    """
    # color = classes.graph.nodes[node].get('color', None)

    # Define default colors
    # default_cidx_to_color = kwimage.Color.distinct(len(data))

    # try and read colors from classes CategoryTree
    # try:
    #     cidx_to_color = []

    cidx_to_color = []
    for cidx in range(len(probs)):
        node = classes[cidx]
        color = classes.graph.nodes[node].get('color', None)
        if color is not None:
            color = kwimage.Color(color).as01()
        cidx_to_color.append(color)

    import distinctipy
    have_colors = [c for c in cidx_to_color if c is not None]
    num_need = sum(c is None for c in cidx_to_color)
    if num_need:
        new_colors = distinctipy.get_colors(
            num_need, exclude_colors=have_colors, rng=569944)
        new_color_iter = iter(new_colors)
        cidx_to_color = [next(new_color_iter) if c is None else c for c in cidx_to_color]

    canvas_dtype = np.float32

    # Each class gets its own color, and modulates the alpha
    h, w = probs.shape[-2:]
    layer_shape = (h, w, 4)
    background = np.zeros(layer_shape, dtype=canvas_dtype)
    background[..., 3] = 1.0
    layers = []
    for cidx, chan in enumerate(probs):
        color = cidx_to_color[cidx]
        layer = np.empty(layer_shape, dtype=canvas_dtype)
        layer[..., 3] = chan
        layer[..., 0:3] = color
        layers.append(layer)
    layers.append(background)

    colormask = kwimage.overlay_alpha_layers(
        layers, keepalpha=False, dtype=canvas_dtype)

    return colormask


@profile
def draw_truth_borders(true_dets, canvas, alpha=1.0, color=None):
    true_sseg = true_dets.data['segmentations']
    true_cidxs = true_dets.data['class_idxs']
    _classes = true_dets.data['classes']

    if color is None:
        _nodes = ub.take(_classes.idx_to_node, true_cidxs)
        _node_data = ub.take(_classes.graph.nodes, _nodes)
        _node_colors = [d['color'] for d in _node_data]
        color = _node_colors

    canvas = kwimage.ensure_float01(canvas)
    if alpha < 1.0:
        # remove this condition when kwimage 0.8.3 is released always take else
        empty_canvas = np.zeros_like(canvas, shape=(canvas.shape[0:2] + (4,)))
        overlay_canvas = true_sseg.draw_on(empty_canvas, fill=False,
                                           border=True, color=color, alpha=1.0)
        overlay_canvas[..., 3] *= alpha
        canvas = kwimage.overlay_alpha_images(overlay_canvas, canvas)
    else:
        canvas = true_sseg.draw_on(canvas, fill=False, border=True,
                                   color=color, alpha=alpha)
    return canvas


@profile
def dump_chunked_confusion(full_classes, true_coco_imgs, chunk_info,
                           heatmap_dpath, title=None, config=None):
    """
    Draw a a sequence of true/pred image predictions
    """
    color_labels = ['TN', 'TP', 'FN', 'FP']

    score_space = config.get('score_space', 'video')

    colors = list(ub.take(CONFUSION_COLOR_SCHEME, color_labels))
    # colors = ['blue', 'green', 'yellow', 'red']
    # colors = ['black', 'white', 'yellow', 'red']
    color_lut = np.array([kwimage.Color(c).as255() for c in colors])
    # full_classes: kwcoco.CategoryTree = true_coco.object_categories()

    if config is None:
        config = {}

    resolution = config.get('resolution', None)
    draw_legend = config.get('draw_legend', True)

    # Make a legend
    color01_lut = color_lut / 255.0
    legend_images = []

    if 'catname_to_prob' in chunk_info[0]:
        # Class Legend
        label_to_color = {
            node: kwimage.Color(data['color']).as01()
            for node, data in full_classes.graph.nodes.items()}
        label_to_color = ub.sorted_keys(label_to_color)
        legend_img_class = _memo_legend(label_to_color)
        legend_images.append(legend_img_class)

    if 'pred_saliency' in chunk_info[0]:
        # Confusion Legend
        label_to_color = ub.dzip(color_labels, color01_lut)
        if draw_legend:
            legend_img_saliency_cfsn = _memo_legend(label_to_color)
            legend_img_saliency_cfsn = kwimage.ensure_uint255(legend_img_saliency_cfsn)
            legend_images.append(legend_img_saliency_cfsn)

    if len(legend_images):
        legend_img = kwimage.stack_images(legend_images, axis=0, pad=5)
    else:
        legend_img = None

    # Draw predictions on each frame
    parts = []
    frame_nums = []
    true_gids = []
    unique_vidnames = set()
    for info, true_coco_img in zip(chunk_info, true_coco_imgs):
        row = info['row']
        if row.get('video', ''):
            unique_vidnames.add(row['video'])

        # true_gid = row['true_gid']
        # true_coco_img = true_coco.coco_image(true_gid)
        true_gid = true_coco_img.img['id']

        true_img = true_coco_img.img
        frame_index = true_img.get('frame_index', None)
        if frame_index is not None:
            frame_nums.append(frame_index)
        true_gids.append(true_gid)

        # image_header_text = f'{frame_index} - gid = {true_gid}'

        header_lines = build_image_header_text(
            img=true_img,
            name=None,
            _header_extra=None,
        )
        # date_captured = true_img.get('date_captured', '')
        # frame_index = true_img.get('frame_index', None)
        # gid = true_img.get('id', None)
        # sensor_coarse = true_img.get('sensor_coarse', 'unknown')
        # _header_extra = None
        # header_line_infos = [
        #     [f'gid={gid}, frame={frame_index}', _header_extra],
        #     [sensor_coarse, date_captured],
        # ]
        # header_lines = []
        # for line_info in header_line_infos:
        #     header_line = ' '.join([p for p in line_info if p])
        #     if header_line:
        #         header_lines.append(header_line)
        image_header_text = '\n'.join(header_lines)

        imgw = info['shape'][1]
        # SC_smt_it_stm_p8_newanns_weighted_raw_v39_epoch=52-step=2269088
        header = kwimage.draw_header_text(
            {'width': imgw},
            # image=confusion_image,
            # image=None,
            text=image_header_text, color='red', stack=False)

        vert_parts = [
            header,
        ]
        DRAW_WEIGHTS = config.get('draw_weights', False)

        if 'catname_to_prob' in info:
            true_dets = info['true_dets']
            true_dets.data['classes'] = full_classes

            pred_classes = kwcoco.CategoryTree.coerce(list(info['catname_to_prob'].keys()))
            true_classes = kwcoco.CategoryTree.coerce(list(info['catname_to_true'].keys()))
            # todo: ensure colors are robust and consistent
            for node in pred_classes.graph.nodes():
                pred_classes.graph.nodes[node]['color'] = full_classes.graph.nodes[node]['color']
            for node in true_classes.graph.nodes():
                true_classes.graph.nodes[node]['color'] = full_classes.graph.nodes[node]['color']

            # pred_classes = kwcoco.CategoryTree
            pred_cat_ohe = np.stack(list(info['catname_to_prob'].values()))
            true_cat_ohe = np.stack(list(info['catname_to_true'].values()))
            # class_pred_idx = pred_cat_ohe.argmax(axis=0)
            # class_true_idx = true_cat_ohe.argmax(axis=0)

            true_overlay = colorize_class_probs(true_cat_ohe, true_classes)[..., 0:3]
            # true_heatmap = kwimage.Heatmap(class_probs=true_cat_ohe, classes=true_classes)
            # true_overlay = true_heatmap.colorize('class_probs')[..., 0:3]
            true_overlay = draw_truth_borders(true_dets, true_overlay, alpha=1.0)
            true_overlay = kwimage.ensure_uint255(true_overlay)
            true_overlay = kwimage.draw_text_on_image(
                true_overlay, 'true class', org=(1, 1), valign='top',
                color=TRUE_GREEN, border=1)
            vert_parts.append(true_overlay)

            if DRAW_WEIGHTS:
                class_weights = info['class_weights']
                if class_weights.max() > 1:
                    weight_image = kwarray.normalize(class_weights, min_val=0)
                    weight_title = 'weights (normed)'
                else:
                    weight_image = class_weights
                    weight_title = 'weights'
                weight_image = kwimage.ensure_uint255(weight_image)
                weight_image = kwimage.draw_text_on_image(
                    weight_image,
                    weight_title,
                    org=(1, 1), valign='top',
                    color='pink', border=1)
                vert_parts.append(weight_image)

            pred_overlay = colorize_class_probs(pred_cat_ohe, pred_classes)[..., 0:3]
            # pred_heatmap = kwimage.Heatmap(class_probs=pred_cat_ohe, classes=pred_classes)
            # pred_overlay = pred_heatmap.colorize('class_probs')[..., 0:3]
            pred_overlay = draw_truth_borders(true_dets, pred_overlay, alpha=0.05, color='white')
            # pred_overlay = draw_truth_borders(true_dets, pred_overlay, alpha=0.05)
            pred_overlay = kwimage.ensure_uint255(pred_overlay)
            pred_overlay = kwimage.draw_text_on_image(
                pred_overlay, 'pred class', org=(1, 1), valign='top',
                color=PRED_BLUE, border=1)
            vert_parts.append(pred_overlay)

        if 'pred_saliency' in info:
            pred_saliency = info['pred_saliency'].astype(np.uint8)
            true_saliency = info['true_saliency']
            saliency_thresh = info['saliency_thresh']
            confusion_idxs = draw_confusion_image(pred_saliency, true_saliency)
            confusion_image = color_lut[confusion_idxs]
            confusion_image = kwimage.ensure_uint255(confusion_image)
            confusion_image = kwimage.draw_text_on_image(
                confusion_image,
                f'confusion saliency: thresh={saliency_thresh:0.3f}',
                org=(1, 1), valign='top',
                color='white', border=1)
            vert_parts.append(
                confusion_image
            )

            if DRAW_WEIGHTS:
                saliency_weights = info['saliency_weights']
                if saliency_weights.max() > 1:
                    weight_image = kwarray.normalize(saliency_weights, min_val=0)
                    weight_title = 'weights (normed)'
                else:
                    weight_image = saliency_weights
                    weight_title = 'weights'
                weight_image = kwimage.ensure_uint255(weight_image)
                weight_image = kwimage.draw_text_on_image(
                    weight_image,
                    weight_title,
                    org=(1, 1), valign='top',
                    color='pink', border=1)
                vert_parts.append(weight_image)

        elif 'true_saliency' in info:
            true_saliency = info['true_saliency']
            true_saliency = true_saliency.astype(np.float32)
            heatmap = kwimage.make_heatmask(
                true_saliency, with_alpha=0.5, cmap='plasma')
            # heatmap[invalid_mask] = 0
            heatmap_int = kwimage.ensure_uint255(heatmap[..., 0:3])
            heatmap_int = kwimage.draw_text_on_image(
                heatmap_int, 'true saliency', org=(1, 1), valign='top',
                color=TRUE_GREEN, border=1)
            vert_parts.append(heatmap_int)
        # confusion_image = kwimage.draw_text_on_image(
        #     confusion_image, image_text, org=(1, 1), valign='top',
        #     color='white', border={'color': 'black'})

        # TODO:
        # Can we show the reference image?
        # TODO:
        # Show the datetime on the top of the image (and the display band?)
        real_image_norm = None
        real_image_int = None

        TRY_IMREAD = 1
        if TRY_IMREAD:
            avali_chans = {p2 for p1 in true_coco_img.channels.spec.split(',') for p2 in p1.split('|')}
            chosen_viz_channs = None
            if len(avali_chans & {'red', 'green', 'blue'}) == 3:
                chosen_viz_channs = 'red|green|blue'
            elif len(avali_chans & {'r', 'g', 'b'}) == 3:
                chosen_viz_channs = 'r|g|b'
            elif len(avali_chans & {'pan'}) == 3:
                chosen_viz_channs = 'pan'
            else:
                chosen_viz_channs = true_coco_img.primary_asset()['channels']
            try:
                real_image = true_coco_img.imdelay(chosen_viz_channs,
                                                   space=score_space,
                                                   nodata_method='float',
                                                   resolution=resolution).finalize()[:]
                real_image_norm = kwimage.normalize_intensity(real_image)
                real_image_norm = kwimage.fill_nans_with_checkers(real_image_norm)
                real_image_int = kwimage.ensure_uint255(real_image_norm)
            except Exception as ex:
                print('ex = {!r}'.format(ex))

        TRY_SOFT = 1
        salient_prob = None
        if TRY_SOFT:
            salient_prob = info.get('salient_prob', None)
            # invalid_mask = info.get('invalid_mask', None)
            if salient_prob is not None:
                invalid_mask = np.isnan(salient_prob)
                heatmap = kwimage.make_heatmask(
                    salient_prob, with_alpha=0.5, cmap='plasma')
                heatmap[invalid_mask] = np.nan
                heatmap = kwimage.fill_nans_with_checkers(heatmap)
                # heatmap[invalid_mask] = 0
                heatmap_int = kwimage.ensure_uint255(heatmap[..., 0:3])
                heatmap_int = kwimage.draw_text_on_image(
                    heatmap_int, 'pred saliency', org=(1, 1), valign='top',
                    color=PRED_BLUE, border=1)
                vert_parts.append(heatmap_int)
                # if real_image_norm is not None:
                #     overlaid = kwimage.overlay_alpha_layers([heatmap, real_image_norm.mean(axis=2)])
                #     overlaid = kwimage.ensure_uint255(overlaid[..., 0:3])
                #     vert_parts.append(overlaid)

        if real_image_int is not None:
            vert_parts.append(real_image_int)

        vert_parts = [kwimage.ensure_uint255(c) for c in vert_parts]
        vert_stack = kwimage.stack_images(vert_parts, axis=0)
        parts.append(vert_stack)

    max_frame = None if len(frame_nums) == 0 else max(frame_nums)
    min_frame = None if len(frame_nums) == 0 else min(frame_nums)
    max_gid = max(true_gids)
    min_gid = min(true_gids)

    try:
        # num_digits = _max_digits(max_num) # TODO
        if max_frame == min_frame:
            frame_part = f'{min_frame:04d}'
        else:
            frame_part = f'{min_frame:04d}-{max_frame:04d}'
    except TypeError:
        frame_part = f'{min_frame}'

    try:
        if max_gid == min_gid:
            gid_part = f'{min_gid:04d}'
        else:
            gid_part = f'{min_gid:04d}-{max_gid:04d}'
    except TypeError:
        gid_part = f'{min_gid}'

    vidname_part = '_'.join(list(unique_vidnames))
    if not vidname_part:
        vidname_part = '_loose_images'

    plot_fstem = f'{vidname_part}-{frame_part}-{gid_part}'

    canvas_title_parts = []
    if title:
        canvas_title_parts.append(title)
    canvas_title_parts.append(plot_fstem)
    canvas_title = '\n'.join(canvas_title_parts)

    plot_canvas = kwimage.stack_images(parts, axis=1, overlap=-10)

    if draw_legend:
        if legend_img is not None:
            plot_canvas = kwimage.stack_images(
                [plot_canvas, legend_img], axis=1, overlap=-10)

    header = kwimage.draw_header_text(
        {'width': plot_canvas.shape[1]}, canvas_title)
    plot_canvas = kwimage.stack_images([header, plot_canvas], axis=0)

    heatmap_dpath = ub.Path(str(heatmap_dpath))
    vid_plot_dpath = (heatmap_dpath / vidname_part).ensuredir()
    plot_fpath = vid_plot_dpath / (plot_fstem + '.jpg')
    kwimage.imwrite(str(plot_fpath), plot_canvas)


@profile
def evaluate_segmentations(true_coco, pred_coco, eval_dpath=None,
                           eval_fpath=None, config=None):
    """
    TODO:
        - [ ] Fold non-critical options into the config

    CommandLine:
        XDEV_PROFILE=1 xdoctest -m geowatch.tasks.fusion.evaluate evaluate_segmentations

    Example:
        >>> # xdoctest: +REQUIRES(module:kwutil)
        >>> from kwcoco.coco_evaluator import CocoEvaluator
        >>> from kwcoco.demo.perterb import perterb_coco
        >>> import kwcoco
        >>> true_coco1 = kwcoco.CocoDataset.demo('vidshapes2', image_size=(64, 64))
        >>> true_coco2 = kwcoco.CocoDataset.demo('shapes2', image_size=(64, 64))
        >>> #true_coco1 = kwcoco.CocoDataset.demo('vidshapes9')
        >>> #true_coco2 = kwcoco.CocoDataset.demo('shapes128')
        >>> true_coco = kwcoco.CocoDataset.union(true_coco1, true_coco2)
        >>> kwargs = {
        >>>     'box_noise': 0.5,
        >>>     'n_fp': (0, 10),
        >>>     'n_fn': (0, 10),
        >>>     'with_probs': True,
        >>>     'with_heatmaps': True,
        >>>     'verbose': 1,
        >>> }
        >>> # TODO: it would be nice to demo the soft metrics
        >>> # functionality by adding "salient_prob" or "class_prob"
        >>> # auxiliary channels to this demodata.
        >>> print('perterbing')
        >>> pred_coco = perterb_coco(true_coco, **kwargs)
        >>> eval_dpath = ub.Path.appdir('kwcoco/tests/fusion_eval').ensuredir()
        >>> print('eval_dpath = {!r}'.format(eval_dpath))
        >>> config = {}
        >>> config['score_space'] = 'image'
        >>> draw_curves = 'auto'
        >>> draw_heatmaps = 'auto'
        >>> #draw_heatmaps = False
        >>> config['workers'] = 'min(avail-2,6)'
        >>> #workers = 0
        >>> evaluate_segmentations(true_coco, pred_coco, eval_dpath, config=config)

    Example:
        >>> # xdoctest: +REQUIRES(env:SLOW_DOCTEST)
        >>> # xdoctest: +REQUIRES(module:kwutil)
        >>> from kwcoco.coco_evaluator import CocoEvaluator
        >>> from kwcoco.demo.perterb import perterb_coco
        >>> import kwcoco
        >>> true_coco = kwcoco.CocoDataset.demo('vidshapes2', image_size=(64, 64))
        >>> kwargs = {
        >>>     'box_noise': 0.5,
        >>>     'n_fp': (0, 10),
        >>>     'n_fn': (0, 10),
        >>>     'with_probs': True,
        >>>     'with_heatmaps': True,
        >>>     'verbose': 1,
        >>> }
        >>> # TODO: it would be nice to demo the soft metrics
        >>> # functionality by adding "salient_prob" or "class_prob"
        >>> # auxiliary channels to this demodata.
        >>> print('perterbing')
        >>> pred_coco = perterb_coco(true_coco, **kwargs)
        >>> eval_dpath = ub.Path.appdir('kwcoco/tests/fusion_eval-video').ensuredir()
        >>> print('eval_dpath = {!r}'.format(eval_dpath))
        >>> config = {}
        >>> config['score_space'] = 'video'
        >>> config['balance_area'] = True
        >>> draw_curves = 'auto'
        >>> draw_heatmaps = 'auto'
        >>> #draw_heatmaps = False
        >>> config['workers'] = 'min(avail-2,6)'
        >>> #workers = 0
        >>> evaluate_segmentations(true_coco, pred_coco, eval_dpath, config=config)
    """
    import rich
    from kwutil import process_context
    from kwutil import util_progress
    from kwutil import util_parallel

    if config is None:
        config = {}

    draw_curves = config.get('draw_curves', 'auto')
    draw_heatmaps = config.get('draw_heatmaps', 'auto')
    score_space = config.get('score_space', 'auto')
    draw_workers = config.get('draw_workers', 'auto')

    if score_space == 'auto':
        if true_coco.n_videos:
            score_space = 'video'
        else:
            score_space = 'image'
        config['score_space'] = score_space

    # Ensure each class has colors.
    ensure_heuristic_coco_colors(true_coco)
    true_classes = list(true_coco.object_categories())
    full_classes: kwcoco.CategoryTree = true_coco.object_categories()

    # Sometimes supercategories dont get colors, this fixes that.
    ensure_heuristic_category_tree_colors(full_classes)

    workers = util_parallel.coerce_num_workers(config.get('workers', 0))
    if draw_workers == 'auto':
        draw_workers = min(2, workers)
    else:
        draw_workers = util_parallel.coerce_num_workers(draw_workers)

    # Extract metadata about the predictions to persist
    meta = {}
    meta['info'] = info = []

    if pred_coco.fpath is not None:
        pred_fpath = ub.Path(pred_coco.fpath)
        meta['pred_name'] = '_'.join((list(pred_fpath.parts[-2:-1]) + [pred_fpath.stem]))

    predicted_info = pred_coco.dataset.get('info', [])
    for item in predicted_info:
        if item.get('type', None) == 'measure':
            info.append(item)
        if item.get('type', None) == 'process':
            proc_name = item.get('properties', {}).get('name', None)
            if proc_name == 'geowatch.tasks.fusion.predict':
                package_fpath = item['properties']['config'].get('package_fpath')
                if 'title' not in item:
                    item['title'] = ub.Path(package_fpath).stem
                if 'package_name' not in item:
                    item['package_name'] = ub.Path(package_fpath).stem

                # FIXME: title should also include pred-config info
                meta['title'] = item['title']
                meta['package_name'] = item['package_name']
                info.append(item)

    # Title contains the model package name if we can infer it
    package_name = meta.get('package_name', '')
    pred_name = meta.get('pred_name', '')
    title_parts = [p for p in [package_name, pred_name] if p]

    resolution = config.get('resolution', None)
    balance_area = config.get('balance_area', False)
    if resolution is not None:
        title_parts.append(f'space={score_space} @ {resolution}, balance_area={balance_area}')
    else:
        title_parts.append(f'space={score_space} balance_area={balance_area}')

    meta['title_parts'] = title_parts
    title = meta['title'] = ' - '.join(title_parts)

    required_marked = 'auto'  # parametarize
    if required_marked == 'auto':
        # In "auto" mode dont require marks if all images are unmarked,
        # otherwise assume that we should restirct to marked images
        required_marked = any(pred_coco.images().lookup('has_predictions', False))

    matches  = associate_images(
        true_coco, pred_coco, key_fallback='id')

    video_matches = matches['video']
    image_matches = matches['image']

    n_vid_matches = len(video_matches)
    n_img_per_vid_matches = [len(d['match_gids1']) for d in video_matches]
    n_img_matches = len(image_matches['match_gids1'])
    print('n_img_per_vid_matches = {}'.format(ub.urepr(n_img_per_vid_matches, nl=1)))
    print('n_vid_matches = {}'.format(ub.urepr(n_vid_matches, nl=1)))
    print('n_img_matches = {!r}'.format(n_img_matches))
    rich.print(f'Eval Dpath: [link={eval_dpath}]{eval_dpath}[/link]')

    chunk_size = 5
    num_thresh_bins = config.get('thresh_bins', 32 * 32)
    thresh_bins = np.linspace(0, 1, num_thresh_bins)  # this is more stable using an ndarray

    if draw_curves == 'auto':
        draw_curves = bool(eval_dpath is not None)

    if draw_heatmaps == 'auto':
        draw_heatmaps = bool(eval_dpath is not None)

    pcontext = process_context.ProcessContext(
        name='geowatch.tasks.fusion.evaluate',
        config=config,
    )
    pcontext.start()

    if eval_dpath is None:
        heatmap_dpath = None
    else:
        eval_dpath = ub.Path(eval_dpath)
        curve_dpath = (eval_dpath / 'curves').ensuredir()
        pcontext.write_invocation(curve_dpath / 'invocation.sh')

    # Objects that will aggregate confusion across multiple images
    salient_measure_combiner = MeasureCombiner(thresh_bins=thresh_bins)
    class_measure_combiner = OneVersusRestMeasureCombiner(thresh_bins=thresh_bins)

    # Gather the true and predicted image pairs to be scored
    total_images = 0
    if required_marked:
        for video_match in video_matches:
            gids1 = video_match['match_gids1']
            gids2 = video_match['match_gids2']
            flags = pred_coco.images(gids2).lookup('has_predictions', False)
            video_match['match_gids1'] = list(ub.compress(gids1, flags))
            video_match['match_gids2'] = list(ub.compress(gids2, flags))
            total_images += len(gids1)
        gids1 = image_matches['match_gids1']
        gids2 = image_matches['match_gids2']
        flags = pred_coco.images(gids2).lookup('has_predictions', False)
        image_matches['match_gids1'] = list(ub.compress(gids1, flags))
        image_matches['match_gids2'] = list(ub.compress(gids2, flags))
        total_images += len(gids1)
    else:
        total_images = None

    # Prepare job pools
    print('workers = {!r}'.format(workers))
    print('draw_workers = {!r}'.format(draw_workers))
    # draw_executor = ub.Executor(mode='process', max_workers=draw_workers)
    # metrics_executor = ub.Executor(mode='process', max_workers=workers)

    # We want to prevent too many evaluate jobs from piling up results to draw,
    # as it takes longer to draw than it does to score. For this reason, block
    # if the draw queue gets too big.
    metrics_executor = _DelayedBlockingJobQueue(max_unhandled_jobs=workers, mode='process', max_workers=workers)
    draw_executor = _MaxQueuePool(mode='process', max_workers=draw_workers, max_queue_size=draw_workers * 4)

    prog = ub.ProgIter(total=total_images, desc='submit scoring jobs', adjust=False, freq=1)
    prog.begin()

    job_chunks = []
    draw_jobs = []

    # Submit scoring jobs over pairs of true-predicted images in videos
    for video_match in video_matches:
        prog.set_postfix_str('comparing ' + video_match['vidname'])
        gids1 = video_match['match_gids1']
        gids2 = video_match['match_gids2']
        if required_marked:
            flags = pred_coco.images(gids2).lookup('has_predictions', False)
            gids1 = list(ub.compress(gids1, flags))
            gids2 = list(ub.compress(gids2, flags))

        current_chunk = []
        for gid1, gid2 in zip(gids1, gids2):
            pred_coco_img = pred_coco.coco_image(gid1).detach()
            true_coco_img = true_coco.coco_image(gid2).detach()
            true_dets = true_coco.annots(gid=gid1).detections

            vidid1 = true_coco.imgs[gid1]['video_id']
            video1 = true_coco.index.videos[vidid1]

            job = metrics_executor.submit(
                single_image_segmentation_metrics, pred_coco_img,
                true_coco_img, true_classes, true_dets, video1,
                thresh_bins=thresh_bins, config=config)

            if len(current_chunk) >= chunk_size:
                job_chunks.append(current_chunk)
                current_chunk = []
            current_chunk.append(job)
            prog.update()

        if len(current_chunk) > 0:
            job_chunks.append(current_chunk)

    # Submit scoring jobs over pairs of true-predicted images without videos
    if score_space == 'image':
        gids1 = image_matches['match_gids1']
        gids2 = image_matches['match_gids2']
        gid_pairs = list(zip(gids1, gids2))
        # Might want to vary the order (or shuffle) depending on user input
        gid_pairs = sorted(gid_pairs, key=lambda x: x[0])

        # TODO: modify to prevent to many unhandled jobs from building up and
        # causing memory issues. Maybe with kwutil.BlockingJobQueue
        for gid1, gid2 in gid_pairs:
            pred_coco_img = pred_coco.coco_image(gid1).detach()
            true_coco_img = true_coco.coco_image(gid2).detach()
            true_dets = true_coco.annots(gid=gid1).detections
            video1 = None
            job = metrics_executor.submit(
                single_image_segmentation_metrics, pred_coco_img,
                true_coco_img, true_classes, true_dets, video1,
                thresh_bins=thresh_bins, config=config)
            prog.update()
            job_chunks.append([job])
    else:
        if len(image_matches['match_gids1']) > 0:
            warnings.warn(ub.paragraph(
                f'''
                Scoring was requested in video mode, but there are
                {len(image_matches['match_gids1'])} true/pred image pairs that
                are unassociated with a video. These pairs will not be included
                in video space scoring.
                '''))
    prog.end()

    num_jobs = sum(map(len, job_chunks))

    RICH_PROG = 'auto'
    if RICH_PROG == 'auto':
        # Use rich outside of slurm
        RICH_PROG = not os.environ.get('SLURM_JOBID', '')

    pman = util_progress.ProgressManager(backend='rich' if RICH_PROG else 'progiter')

    DEBUG = 0
    if DEBUG:
        orig_infos = []

    VERBOSE_DEBUG = 0

    rows = []
    with pman:
        score_prog = pman.progiter(desc="[cyan] Scoring...", total=num_jobs)
        score_prog.start()
        if draw_heatmaps:
            draw_prog = pman.progiter(desc="[green] Drawing...", total=len(job_chunks))
            draw_prog.start()

        for job_chunk in job_chunks:
            chunk_info = []
            for job in job_chunk:
                info = job.result()
                if VERBOSE_DEBUG:
                    print('Gather job result')
                if DEBUG:
                    orig_infos.append(info)
                score_prog.update(1)
                rows.append(info['row'])
                if VERBOSE_DEBUG:
                    print(f'Add new row: {info["row"]}')
                    print(f'Table size: {len(rows)}')

                class_measures = info.get('class_measures', None)
                salient_measures = info.get('salient_measures', None)
                if salient_measures is not None:
                    salient_measure_combiner.submit(salient_measures)
                if class_measures is not None:
                    class_measure_combiner.submit(class_measures)
                if draw_heatmaps:
                    chunk_info.append(info)

            # Once a job chunk is done, clear its memory
            if VERBOSE_DEBUG:
                print(f'Clear job chunk of len {len(job_chunk)}')
            job = None
            job_chunk.clear()

            # Reduce measures over the chunk
            if salient_measure_combiner.queue_size > chunk_size:
                salient_measure_combiner.combine()
            if class_measure_combiner.queue_size > chunk_size:
                class_measure_combiner.combine()

            if draw_heatmaps:
                heatmap_dpath = (ub.Path(eval_dpath) / 'heatmaps').ensuredir()
                # Let the draw executor release any memory it can
                remaining_draw_jobs = []
                if VERBOSE_DEBUG:
                    print(f'Handle {len(draw_jobs)} draw jobs')
                for draw_job in draw_jobs:
                    if draw_job.done():
                        draw_job.result()
                        draw_prog.update(1)
                    else:
                        remaining_draw_jobs.append(draw_job)
                draw_job = None
                draw_jobs = remaining_draw_jobs
                if VERBOSE_DEBUG:
                    print(f'Remaining draw jobs: {len(draw_jobs)}')

                # As chunks of evaluation jobs complete, submit background jobs to
                # draw results to disk if requested.
                true_gids = [info['row']['true_gid'] for info in chunk_info]
                true_coco_imgs = true_coco.images(true_gids).coco_images
                true_coco_imgs = [g.detach() for g in true_coco_imgs]
                if VERBOSE_DEBUG:
                    print(f'Submit {len(true_gids)} new draw jobs')
                draw_job = draw_executor.submit(
                    dump_chunked_confusion, full_classes, true_coco_imgs,
                    chunk_info, heatmap_dpath, title=title, config=config)
                draw_jobs.append(draw_job)

        if VERBOSE_DEBUG:
            print('Finished metric jobs')
        metrics_executor.shutdown()

        if draw_heatmaps:
            # Allow all drawing jobs to finalize
            if VERBOSE_DEBUG:
                print(f'Finalize {len(draw_jobs)} draw jobs')
            while draw_jobs:
                job = draw_jobs.pop()
                job.result()
                draw_prog.update(1)
            draw_executor.shutdown()

    df = pd.DataFrame(rows)
    df_summary = df.describe().T
    print('Per Image Pixel Measures')
    rich.print(df)
    rich.print(df_summary.to_string())

    if eval_dpath is not None:
        perimage_table_fpath = eval_dpath / 'perimage_table.json'
        perimage_summary_fpath = eval_dpath / 'perimage_summary.json'
        perimage_table_fpath.write_text(df.to_json(orient='table', indent=4))
        perimage_summary_fpath.write_text(df_summary.to_json(orient='table', indent=4))

    # Finalize all of the aggregated measures
    print('Finalize salient measures')
    # Note: this will return False if there are no salient measures
    salient_combo_measures = salient_measure_combiner.finalize()
    if salient_combo_measures is False or salient_combo_measures is None:
        # Use nan measures from empty binary confusion vectors
        salient_combo_measures = BinaryConfusionVectors(None).measures()
    # print('salient_combo_measures = {!r}'.format(salient_combo_measures))

    if DEBUG:
        # Redo salient combine
        tocombine = []

        for p in tocombine:
            z = ub.dict_isect(p, {'fp_count', 'tp_count', 'fn_count', 'tn_count', 'thresholds', 'nsupport'})
            print(ub.urepr(ub.map_vals(list, z), nl=0))

        salient_measure_combiner = MeasureCombiner(thresh_bins=thresh_bins)
        print('salient_combo_measures.__dict__ = {!r}'.format(salient_combo_measures.__dict__))
        # precision = None
        # growth = None
        from kwcoco.metrics.confusion_measures import Measures
        for info in orig_infos:
            class_measures = info.get('class_measures', None)
            salient_measures = info.get('salient_measures', None)
            if salient_measures is not None:
                tocombine.append(salient_measures)
                salient_measure_combiner.submit(salient_measures)

        combo = Measures.combine(tocombine, thresh_bins=thresh_bins).reconstruct()
        print('combo = {!r}'.format(combo))

        combo = Measures.combine(tocombine, precision=2)
        combo.reconstruct()
        print('combo = {!r}'.format(combo))

        combo = Measures.combine(tocombine, growth='max')
        combo.reconstruct()
        print('combo = {!r}'.format(combo))

        salient_combo_measures = salient_measure_combiner.finalize()
        print('salient_combo_measures = {!r}'.format(salient_combo_measures))

    print('Finalize class measures')
    class_combo_measure_dict = class_measure_combiner.finalize()
    ovr_combo_measures = class_combo_measure_dict['perclass']

    # Combine class + salient measures using the "SingleResult" container
    # (TODO: better API)
    result = CocoSingleResult(
        salient_combo_measures, ovr_combo_measures, None, meta)
    rich.print('result = {}'.format(result))

    meta['info'].append(pcontext.stop())

    if salient_combo_measures is not None:
        if eval_dpath is not None:
            if isinstance(salient_combo_measures, dict):
                salient_combo_measures['meta'] = meta

            title = '\n'.join(meta.get('title_parts', [meta.get('title', '')]))

            if eval_fpath is None:
                eval_fpath = curve_dpath / 'measures2.json'
            print('Dump eval_fpath={}'.format(eval_fpath))
            result.dump(os.fspath(eval_fpath))

            if draw_curves:
                import kwplot
                # kwplot.autompl()
                with kwplot.BackendContext('agg'):
                    fig = kwplot.figure(doclf=True)

                    print('Dump salient figures')
                    salient_combo_measures.summary_plot(fnum=1, title=title)
                    fig = kwplot.autoplt().gcf()
                    fig.savefig(str(curve_dpath / 'salient_summary.png'))

                    print('Dump class figures')
                    result.dump_figures(curve_dpath, expt_title=title)

    summary = {}
    if class_combo_measure_dict is not None:
        summary['class_mAP'] = class_combo_measure_dict['mAP']
        summary['class_mAUC'] = class_combo_measure_dict['mAUC']

    if salient_combo_measures is not None:
        summary['salient_ap'] = salient_combo_measures['ap']
        summary['salient_auc'] = salient_combo_measures['auc']
        summary['salient_max_f1'] = salient_combo_measures['max_f1']

    rich.print('summary = {}'.format(ub.urepr(
        summary, nl=1, precision=4, align=':', sort=0)))

    rich.print(f'Eval Dpath: [link={eval_dpath}]{eval_dpath}[/link]')
    print(f'eval_fpath={eval_fpath}')
    return df


class _DelayedFuture:
    """
    todo: move to kwutil

    Wraps a future object so we can execute logic when its result has been
    accessed.
    """
    def __init__(self, func, args, kwargs, parent):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.task = (func, args, kwargs)
        self.parent = parent
        self.future = None

    def result(self, timeout=None):
        if self.future is None:
            raise Exception('The task has not been submitted yet')
        result = self.future.result(timeout)
        self.parent._job_result_accessed_callback(self)
        return result


class _DelayedBlockingJobQueue:
    """
    todo: move to kwutil

    References:
        .. [GISTnoxdafoxMaxQueuePool] https://gist.github.com/noxdafox/4150eff0059ea43f6adbdd66e5d5e87e

    Ignore:
        >>> self = _DelayedBlockingJobQueue(max_unhandled_jobs=5)
        >>> futures = [
        >>>     self.submit(print, i)
        >>>     for i in range(10)
        >>> ][::-1]
        >>> import time
        >>> time.sleep(0.5)
        >>> print(self._num_submitted_jobs)
        >>> print(self._num_handled_results)
        >>> print('--- First 5 should have printed ---')
        >>> for _ in range(3):
        >>>     f = futures.pop()
        >>>     f.result()
        >>> time.sleep(0.5)
        >>> print(self._num_submitted_jobs)
        >>> print(self._num_handled_results)
        >>> print('--- 3 Results were haneld, so 3 more can join the queue')
        >>> for _ in range(3):
        >>>     f = futures.pop()
        >>>     f.result()
        >>> time.sleep(0.5)
        >>> print(self._num_submitted_jobs)
        >>> print(self._num_handled_results)
        >>> print('--- Handling the rest, but everything should have already been submitted')
        >>> for _ in range(4):
        >>>     f = futures.pop()
        >>>     f.result()
    """
    def __init__(self, max_unhandled_jobs, mode='thread', max_workers=None):
        from collections import deque
        self._unsubmitted = deque()
        self.pool = ub.Executor(mode=mode, max_workers=max_workers)
        self.max_unhandled_jobs = max_unhandled_jobs
        self._num_handled_results = 0
        self._num_submitted_jobs = 0
        self._num_unhandled = 0

    def submit(self, func, *args, **kwargs):
        """
        Queues a new job, but wont execute until
        some conditions are met
        """
        delayed = _DelayedFuture(func, args, kwargs, parent=self)
        self._unsubmitted.append(delayed)
        self._submit_if_room()
        return delayed

    def _submit_if_room(self):
        while self._num_unhandled < self.max_unhandled_jobs and self._unsubmitted:
            delayed = self._unsubmitted.popleft()
            self._num_submitted_jobs += 1
            self._num_unhandled += 1
            delayed.future = self.pool.submit(delayed.func, *delayed.args, **delayed.kwargs)

    def _job_result_accessed_callback(self, _):
        """Called when the user handles a result """
        self._num_handled_results += 1
        self._num_unhandled -= 1
        self._submit_if_room()

    def shutdown(self):
        """
        Calls the shutdown function of the underlying backend.
        """
        return self.pool.shutdown()


class _MaxQueuePool:
    """

    todo: move to kwutil

    This Class wraps a concurrent.futures.Executor
    limiting the size of its task queue.
    If `max_queue_size` tasks are submitted, the next call to submit will block
    until a previously submitted one is completed.

    References:
        .. [GISTnoxdafoxMaxQueuePool] https://gist.github.com/noxdafox/4150eff0059ea43f6adbdd66e5d5e87e

    Ignore:
        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/geowatch'))
        from geowatch.tasks.fusion.evaluate import *  # NOQA
        from geowatch.tasks.fusion.evaluate import _memo_legend, _redraw_measures
        self = _MaxQueuePool(max_queue_size=0)

        dpath = ub.Path.appdir('kwutil/doctests/maxpoolqueue')
        dpath.delete().ensuredir()
        signal_fpath = dpath / 'signal'

        def waiting_worker():
            counter = 0
            while not signal_fpath.exists():
                counter += 1
            return counter

        future = self.submit(waiting_worker)

        try:
            future.result(timeout=0.001)
        except TimeoutError:
            ...
        signal_fpath.touch()
        result = future.result()

    """
    def __init__(self, max_queue_size=None, mode='thread', max_workers=0):
        if max_queue_size is None:
            max_queue_size = max_workers
        self.pool = ub.Executor(mode=mode, max_workers=max_workers)
        if 'serial' in self.pool.backend.__class__.__name__.lower():
            self.pool_queue = None
        else:
            from threading import BoundedSemaphore  # NOQA
            self.pool_queue = BoundedSemaphore(max_queue_size)

    def submit(self, function, *args, **kwargs):
        """Submits a new task to the pool, blocks if Pool queue is full."""
        if self.pool_queue is not None:
            self.pool_queue.acquire()

        future = self.pool.submit(function, *args, **kwargs)
        future.add_done_callback(self.pool_queue_callback)

        return future

    def pool_queue_callback(self, _):
        """Called once task is done, releases one queue slot."""
        if self.pool_queue is not None:
            self.pool_queue.release()

    def shutdown(self):
        """
        Calls the shutdown function of the underlying backend.
        """
        return self.pool.shutdown()


def _redraw_measures(eval_dpath):
    """
    hack helper for developer, not critical
    """
    curve_dpath = ub.Path(eval_dpath) / 'curves'
    measures_fpath = curve_dpath / 'measures.json'
    with open(measures_fpath, 'r') as file:
        state = json.load(file)
        salient_combo_measures = Measures.from_json(state)
        meta = salient_combo_measures.get('meta', [])
        title = ''
        if meta is not None:
            if isinstance(meta, list):
                # Old
                for item in meta:
                    title = item.get('title', title)
            else:
                # title = meta.get('title', title)
                title = '\n'.join(meta.get('title_parts', [meta.get('title', '')]))
        import kwplot
        with kwplot.BackendContext('agg'):
            salient_combo_measures.summary_plot(fnum=1, title=title)
            fig = kwplot.autoplt().gcf()
            fig.savefig(str(curve_dpath / 'summary_redo.png'))


def _max_digits(max_num):
    """
    Use like this:
        your_var = 231
        max_num = 9180
        num_digits = _max_digits(max_num)
        f'{your_var:0{num_digits}d}'
        # or
        f'{your_var:0{_max_digits(max_num)}d}'
    """
    import math
    if max_num is None:
        num_digits = 8
    else:
        num_digits = int(math.log10(max(max_num, 1))) + 1
    return num_digits


@profile
def associate_images(dset1, dset2, key_fallback=None):
    """
    Builds an association between image-ids in two datasets.

    One use for this is if ``dset1`` is a truth dataset and ``dset2`` is a
    prediction dataset, and you need the to know which images are in common so
    they can be scored.

    Args:
        dset1 (kwcoco.CocoDataset): a kwcoco datset.

        dset2 (kwcoco.CocoDataset): another kwcoco dataset

        key_fallback (str):
            The fallback key to use if the image "name" is not specified.
            This can either be "file_name" or "id" or None.

    TODO:
        - [ ] port to kwcoco proper
        - [ ] use in kwcoco eval as a robust image/video association method

    Example:
        >>> import kwcoco
        >>> from kwcoco.demo.perterb import perterb_coco
        >>> dset1 = kwcoco.CocoDataset.demo('shapes2')
        >>> kwargs = {
        >>>     'box_noise': 0.5,
        >>>     'n_fp': (0, 10),
        >>>     'n_fn': (0, 10),
        >>> }
        >>> dset2 = perterb_coco(dset1, **kwargs)
        >>> matches = associate_images(dset1, dset2, key_fallback='file_name')
        >>> assert len(matches['image']['match_gids1'])
        >>> assert len(matches['image']['match_gids2'])
        >>> assert not len(matches['video'])

    Example:
        >>> import kwcoco
        >>> from kwcoco.demo.perterb import perterb_coco
        >>> dset1 = kwcoco.CocoDataset.demo('vidshapes2')
        >>> kwargs = {
        >>>     'box_noise': 0.5,
        >>>     'n_fp': (0, 10),
        >>>     'n_fn': (0, 10),
        >>> }
        >>> dset2 = perterb_coco(dset1, **kwargs)
        >>> matches = associate_images(dset1, dset2, key_fallback='file_name')
        >>> assert not len(matches['image']['match_gids1'])
        >>> assert not len(matches['image']['match_gids2'])
        >>> assert len(matches['video'])
    """
    common_vidnames = (set(dset1.index.name_to_video) &
                       set(dset2.index.name_to_video))

    def image_keys(dset, gids):
        # Generate image "keys" that should be compatible between datasets
        for gid in gids:
            img = dset.imgs[gid]
            if img.get('name', None) is not None:
                yield img['name']
            else:
                if key_fallback is None:
                    raise Exception('images require names to associate')
                elif key_fallback == 'id':
                    yield img['id']
                elif key_fallback == 'file_name':
                    yield img['file_name']
                else:
                    raise KeyError(key_fallback)

    all_gids1 = list(dset1.imgs.keys())
    all_gids2 = list(dset2.imgs.keys())
    all_keys1 = list(image_keys(dset1, all_gids1))
    all_keys2 = list(image_keys(dset2, all_gids2))
    key_to_gid1 = ub.dzip(all_keys1, all_gids1)
    key_to_gid2 = ub.dzip(all_keys2, all_gids2)
    gid_to_key1 = ub.invert_dict(key_to_gid1)
    gid_to_key2 = ub.invert_dict(key_to_gid2)

    video_matches = []

    all_match_gids1 = set()
    all_match_gids2 = set()

    for vidname in common_vidnames:
        video1 = dset1.index.name_to_video[vidname]
        video2 = dset2.index.name_to_video[vidname]
        vidid1 = video1['id']
        vidid2 = video2['id']
        gids1 = dset1.index.vidid_to_gids[vidid1]
        gids2 = dset2.index.vidid_to_gids[vidid2]
        keys1 = ub.oset(ub.take(gid_to_key1, gids1))
        keys2 = ub.oset(ub.take(gid_to_key2, gids2))
        match_keys = ub.oset(keys1) & ub.oset(keys2)
        match_gids1 = list(ub.take(key_to_gid1, match_keys))
        match_gids2 = list(ub.take(key_to_gid2, match_keys))
        all_match_gids1.update(match_gids1)
        all_match_gids2.update(match_gids2)
        video_matches.append({
            'vidname': vidname,
            'match_gids1': match_gids1,
            'match_gids2': match_gids2,
        })

    # Associate loose images not belonging to any video
    unmatched_gid_to_key1 = ub.dict_diff(gid_to_key1, all_match_gids1)
    unmatched_gid_to_key2 = ub.dict_diff(gid_to_key2, all_match_gids2)

    remain_keys = (set(unmatched_gid_to_key1.values()) &
                   set(unmatched_gid_to_key2.values()))
    remain_gids1 = [key_to_gid1[key] for key in remain_keys]
    remain_gids2 = [key_to_gid2[key] for key in remain_keys]

    image_matches = {
        'match_gids1': remain_gids1,
        'match_gids2': remain_gids2,
    }

    matches = {
        'image': image_matches,
        'video': video_matches,
    }
    return matches


def build_image_header_text(**kwargs):
    """
    A heuristic for what sort of info is useful to plot on the header of an
    image.

    Kwargs:
        img
        coco_dset
        vidname,
        _header_extra

        gid,
        frame_index,
        dset_idstr,
        name,
        sensor_coarse,
        date_captured

    Example:
        >>> img = {
        >>>     'id': 1,
        >>>     'frame_index': 0,
        >>>     'date_captured': '2020-01-01',
        >>>     'name': 'BLARG',
        >>>     'sensor_coarse': 'Sensor1',
        >>> }
        >>> kwargs = {
        >>>     'img': img,
        >>>     'dset_idstr': '',
        >>>     'name': '',
        >>>     '_header_extra': None,
        >>> }
        >>> header_lines = build_image_header_text(**kwargs)
        >>> print('header_lines = {}'.format(ub.urepr(header_lines, nl=1)))
    """
    img = kwargs.get('img', {})
    _header_extra = kwargs.get('_header_extra', None)
    dset_idstr = kwargs.get('dset_idstr', '')

    def _multi_get(key, default=ub.NoParam, *dicts):
        # try to lookup from multiple dictionaries
        found = default
        for d in dicts:
            if key in d:
                found = d[key]
                break
        if found is ub.NoParam:
            raise Exception
        return found

    sensor_coarse = _multi_get('sensor_coarse', 'unknown', kwargs, img)
    # name = _multi_get('name', 'unknown', kwargs, img)

    date_captured = _multi_get('date_captured', '', kwargs, img)
    frame_index = _multi_get('frame_index', None, kwargs, img)
    gid = _multi_get('id', None, kwargs, img)
    image_name = _multi_get('name', '', kwargs, img)

    vidname = None
    if 'vidname' in kwargs:
        vidname = kwargs['vidname']
    else:
        coco_dset = kwargs.get('coco_dset', None)
        if coco_dset is not None:
            video_id = img.get('video_id', None)
            if video_id is not None:
                vidname = coco_dset.index.videos[video_id]['name']
            else:
                vidname = 'loose-images'

    image_id_parts = []
    image_id_parts.append(f'gid={gid}')
    image_id_parts.append(f'frame_index={frame_index}')
    image_id_part = ', '.join(image_id_parts)

    header_line_infos = []
    header_line_infos.append([vidname, image_id_part, _header_extra])
    header_line_infos.append([dset_idstr])
    header_line_infos.append([image_name])
    header_line_infos.append([sensor_coarse, date_captured])
    header_lines = []
    for line_info in header_line_infos:
        header_line = ' '.join([p for p in line_info if p])
        header_line = header_line.replace('\\n', '\n')  # hack
        if header_line:
            header_lines.append(header_line)
    return header_lines


def ensure_heuristic_coco_colors(coco_dset, force=False):
    """
    Args:
        coco_dset (kwcoco.CocoDataset): object to modify
        force (bool): if True, overwrites existing colors if needed

    TODO:
        - [ ] Move this non-heuristic functionality to
            :func:`kwcoco.CocoDataset.ensure_class_colors`

    Example:
        >>> import kwcoco
        >>> coco_dset = kwcoco.CocoDataset.demo()
        >>> ensure_heuristic_coco_colors(coco_dset)
        >>> assert all(c['color'] for c in coco_dset.cats.values())
    """
    CATEGORIES = []
    for hcat in CATEGORIES:
        cat = coco_dset.index.name_to_cat.get(hcat['name'], None)
        if cat is not None:
            if force or cat.get('color', None) is None:
                cat['color'] = hcat['color']
    data_dicts = coco_dset.dataset['categories']
    _ensure_distinct_dict_colors(data_dicts)


def ensure_heuristic_category_tree_colors(classes, force=False):
    """
    Args:
        classes (kwcoco.CategoryTree): object to modify
        force (bool): if True, overwrites existing colors if needed

    TODO:
        - [ ] Move this non-heuristic functionality to
            :func:`kwcoco.CategoryTree.ensure_colors`
        - [ ] Consolidate with ~/code/watch/geowatch/tasks/fusion/utils :: category_tree_ensure_color
        - [ ] Consolidate with ~/code/watch/geowatch/utils/kwcoco_extensions :: category_category_colors
        - [ ] Consolidate with ~/code/watch/geowatch/heuristics.py :: ensure_heuristic_category_tree_colors
        - [ ] Consolidate with ~/code/watch/geowatch/heuristics.py :: ensure_heuristic_coco_colors

    Example:
        >>> # xdoctest: +REQUIRES(module:kwutil)
        >>> import kwcoco
        >>> classes = kwcoco.CategoryTree.coerce(['ignore', 'positive', 'Active Construction', 'foobar', 'Unknown', 'baz'])
        >>> ensure_heuristic_category_tree_colors(classes)
        >>> assert all(d['color'] for n, d in classes.graph.nodes(data=True))
    """
    # Set any missing class color with the heuristic category
    CATEGORIES = []
    for hcat in CATEGORIES:
        node_data = classes.graph.nodes.get(hcat['name'], None)
        if node_data is not None:
            if force or node_data.get('color', None) is None:
                node_data['color'] = hcat['color']
    data_dicts = [data for node, data in classes.graph.nodes(data=True)]
    _ensure_distinct_dict_colors(data_dicts)


def _ensure_distinct_dict_colors(data_dicts, force=False):
    # Generalized part that could move to kwcoco
    have_dicts = [d for d in data_dicts if d.get('color', None) is not None]
    miss_dicts = [d for d in data_dicts if d.get('color', None) is None]
    num_uncolored = len(miss_dicts)
    if num_uncolored:
        import kwimage
        existing_colors = [kwimage.Color(d['color']).as01() for d in have_dicts]
        new_colors = kwimage.Color.distinct(
            num_uncolored, existing=existing_colors, legacy=False)
        for d, c in zip(miss_dicts, new_colors):
            d['color'] = c


if __name__ == '__main__':
    # import xdev
    # xdev.make_warnings_print_tracebacks()
    main()
