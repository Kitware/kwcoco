"""
Evaluates a predicted coco dataset against a truth coco dataset.

The components in this module work programatically or as a command line script.
"""
import glob
import numpy as np
import six
import ubelt as ub
import kwimage
import kwarray
from os.path import exists
from os.path import isdir
from os.path import isfile
from os.path import join
import scriptconfig as scfg
# from kwcoco.metrics.util import DictProxy


from kwcoco.category_tree import CategoryTree
try:
    import ndsampler
    CATEGORY_TREE_CLS = (CategoryTree, ndsampler.CategoryTree)
    COCO_SAMPLER_CLS = ndsampler.CocoSampler
except Exception:
    COCO_SAMPLER_CLS = None
    CATEGORY_TREE_CLS = (CategoryTree,)


class CocoEvalConfig(scfg.Config):
    """
    Evaluate and score predicted versus truth detections / classifications in a COCO dataset
    """
    default = {
        'true_dataset': scfg.Value(None, type=str, help='coercable true detections'),
        'pred_dataset': scfg.Value(None, type=str, help='coercable predicted detections'),

        'classes_of_interest': scfg.Value(None, type=list, help='if specified only these classes are given weight'),
        'ignore_classes': scfg.Value(None, type=list, help='classes to ignore (give them zero weight)'),

        # 'TODO: OPTION FOR CLASSLESS-LOCALIZATION SCOREING ONLY'

        # 'discard_classes': scfg.Value(None, type=list, help='classes to completely remove'),  # TODO

        'fp_cutoff': scfg.Value(float('inf'), help='false positive cutoff for ROC'),

        'implicit_negative_classes': scfg.Value(['background']),

        'implicit_ignore_classes': scfg.Value(['ignore']),

        'use_image_names': scfg.Value(False, help='if True use image file_name to associate images instead of ids'),

        # These should go into the CLI args, not the class config args
        'expt_title': scfg.Value('', type=str, help='title for plots'),
        'draw': scfg.Value(True, help='draw metric plots'),
        'out_dpath': scfg.Value('./coco_metrics', type=str),
    }


class CocoEvaluator(object):
    """
    Abstracts the evaluation process to execute on two coco datasets.

    This can be run as a standalone script where the user specifies the paths
    to the true and predited dataset explicitly, or this can be used by a
    higher level script that produces the predictions and then sends them to
    this evaluator.

    Ignore:
        >>> pred_fpath1 = ub.expandpath("$HOME/remote/viame/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-fine-coi-v43/eval/may_priority_habcam_cfarm_v7_test.mscoc/bioharn-det-mc-cascade-rgb-fine-coi-v43__epoch_00000007/c=0.1,i=window,n=0.8,window_d=512,512,window_o=0.0/all_pred.mscoco.json")
        >>> pred_fpath2 = ub.expandpath('$HOME/tmp/cached_clf_out_cli/reclassified.mscoco.json')
        >>> true_fpath = ub.expandpath('$HOME/remote/namek/data/noaa_habcam/combos/habcam_cfarm_v8_test.mscoco.json')
        >>> config = {
        >>>     'true_dataset': true_fpath,
        >>>     'pred_dataset': pred_fpath2,
        >>>     'out_dpath': ub.expandpath('$HOME/remote/namek/tmp/reclassified_eval'),
        >>>     'classes_of_interest': [],
        >>> }
        >>> coco_eval = CocoEvaluator(config)
        >>> config = coco_eval.config
        >>> coco_eval._init()
        >>> coco_eval.evaluate()

    Example:
        >>> from kwcoco.coco_evaluator import CocoEvaluator
        >>> import kwcoco
        >>> dpath = ub.ensure_app_cache_dir('kwcoco/tests/test_out_dpath')
        >>> true_dset = kwcoco.CocoDataset.demo('shapes8')
        >>> from kwcoco.demo.perterb import perterb_coco
        >>> kwargs = {
        >>>     'box_noise': 0.5,
        >>>     'n_fp': (0, 10),
        >>>     'n_fn': (0, 10),
        >>>     'with_probs': True,
        >>> }
        >>> pred_dset = perterb_coco(true_dset, **kwargs)
        >>> config = {
        >>>     'true_dataset': true_dset,
        >>>     'pred_dataset': pred_dset,
        >>>     'out_dpath': dpath,
        >>>     'classes_of_interest': [],
        >>> }
        >>> coco_eval = CocoEvaluator(config)
        >>> results = coco_eval.evaluate()
    """

    def __init__(coco_eval, config):
        coco_eval.config = CocoEvalConfig(config)
        coco_eval._is_init = False
        coco_eval._logs = []
        coco_eval._verbose = 1

    def log(coco_eval, msg, level='INFO'):
        if coco_eval._verbose:
            print(msg)
        coco_eval._logs.append((level, msg))

    def _init(coco_eval):
        # TODO: coerce into a cocodataset form if possible
        coco_eval.log('init truth dset')

        # FIXME: What is the image names line up correctly, but the image ids
        # do not? This will be common if an external detector is used.
        gid_to_true, true_extra = CocoEvaluator._coerce_dets(
            coco_eval.config['true_dataset'])

        coco_eval.log('init pred dset')
        gid_to_pred, pred_extra = CocoEvaluator._coerce_dets(
            coco_eval.config['pred_dataset'])

        if coco_eval.config['use_image_names']:
            # TODO: currently this is a hacky implementation that modifies the
            # pred dset, we should not do that, just store a gid mapping.
            pred_to_true_gid = {}
            true_coco = true_extra['coco_dset']
            pred_coco = pred_extra['coco_dset']
            for gid, true_img in true_coco.imgs.items():
                fname = true_img['file_name']
                if fname not in pred_coco.index.file_name_to_img:
                    continue
                pred_img = pred_coco.index.file_name_to_img[fname]
                pred_to_true_gid[pred_img['id']] = true_img['id']

            if not pred_to_true_gid:
                raise Exception('FAILED TO MAP IMAGE NAMES')

            unused_pred_gids = set(pred_coco.imgs.keys()) - set(pred_to_true_gid.keys())
            pred_coco.remove_images(unused_pred_gids)

            new_gid_to_pred = {}
            for pred_img in pred_coco.imgs.values():
                old_gid = pred_img['id']
                new_gid = pred_to_true_gid[old_gid]
                pred = gid_to_pred[old_gid]
                pred_img['id'] = new_gid
                for ann in pred_coco.annots(gid=old_gid).objs:
                    ann['image_id'] = new_gid
                new_gid_to_pred[new_gid] = pred

            gid_to_pred = new_gid_to_pred
            pred_coco._build_index()

        pred_gids = sorted(gid_to_pred.keys())
        true_gids = sorted(gid_to_true.keys())
        gids = list(set(pred_gids) & set(true_gids))

        true_classes = ub.peek(gid_to_true.values()).classes
        pred_classes = ub.peek(gid_to_pred.values()).classes

        if 0:
            import xdev
            xdev.set_overlaps(set(true_gids), set(pred_gids), 'true', 'pred')

        classes, unified_cid_maps = CocoEvaluator._rectify_classes(
            true_classes, pred_classes)

        true_to_unified_cid = unified_cid_maps['true']
        pred_to_unified_cid = unified_cid_maps['pred']

        # Helper infor for mapping predicted probabilities
        pred_new_idxs = []
        pred_old_idxs = []
        for old_idx, old_node in enumerate(pred_classes.idx_to_node):
            old_cid = pred_classes.node_to_id[old_node]
            new_cid = pred_to_unified_cid[old_cid]
            new_idx = classes.id_to_idx[new_cid]
            pred_old_idxs.append(old_idx)
            pred_new_idxs.append(new_idx)

        needs_prob_remap = (
            (pred_new_idxs == pred_old_idxs) or
            (len(classes) != len(pred_classes))
        ) or True

        # Move truth to the unified class indices
        for gid in ub.ProgIter(gids, desc='Rectify truth class idxs'):
            det = gid_to_true[gid]
            new_classes = classes
            old_classes = det.meta['classes']
            old_cidxs = det.data['class_idxs']
            old_cids = [None if cx is None else old_classes.idx_to_id[cx] for cx in old_cidxs]
            new_cids = [None if cid is None else true_to_unified_cid.get(cid, cid) for cid in old_cids]
            new_cidxs = np.array([None if c is None else new_classes.id_to_idx[c] for c in new_cids])
            det.meta['classes'] = new_classes
            det.data['class_idxs'] = new_cidxs

        # Move predictions to the unified class indices
        for gid in ub.ProgIter(gids, desc='Rectify pred class idxs'):
            det = gid_to_pred[gid]
            new_classes = classes
            old_classes = det.meta['classes']
            old_cidxs = det.data['class_idxs']
            old_cids = [None if cx is None else old_classes.idx_to_id[cx] for cx in old_cidxs]
            new_cids = [None if cid is None else pred_to_unified_cid.get(cid, cid) for cid in old_cids]
            new_cidxs = np.array([None if c is None else new_classes.id_to_idx[c] for c in new_cids])
            det.meta['classes'] = new_classes
            det.data['class_idxs'] = new_cidxs

            if needs_prob_remap and 'probs' in det.data:
                # Ensure predicted probabilities are in the unified class space
                old_probs = det.data['probs']
                new_probs = np.zeros_like(old_probs, shape=(len(old_probs), len(classes)))
                if len(new_probs):
                    new_probs[:, pred_new_idxs] = old_probs[:, pred_old_idxs]
                det.data['probs'] = new_probs

        coco_eval.gids = gids
        coco_eval.classes = classes
        coco_eval.gid_to_true = gid_to_true
        coco_eval.gid_to_pred = gid_to_pred
        coco_eval.true_extra = true_extra
        coco_eval.pred_extra = pred_extra
        coco_eval._is_init = True

    def _ensure_init(coco_eval):
        if not coco_eval._is_init:
            coco_eval._init()

    @classmethod
    def _rectify_classes(coco_eval, true_classes, pred_classes):
        import kwcoco
        # Determine if truth and model classes are compatible, attempt to remap
        # if possible.
        errors = []
        for node1, id1 in true_classes.node_to_id.items():
            if id1 in pred_classes.id_to_node:
                node2 = pred_classes.id_to_node[id1]
                if node1 != node2:
                    errors.append(
                        'id={} exists in pred and true but have '
                        'different names, {}, {}'.format(id1, node1, node2))
            if node1 in pred_classes.node_to_id:
                id2 = pred_classes.node_to_id[node1]
                if id1 != id2:
                    errors.append(
                        'node={} exists in pred and true but have '
                        'different ids, {}, {}'.format(node1, id1, id2))

        # TODO: determine if the class ids are the same, in which case we dont
        # need to do unification.

        # mappings to unified cids
        unified_cid_maps = {
            'true': {},
            'pred': {},
        }
        def _normalize_name(name):
            return name.lower().replace(' ', '_')
        pred_norm = {_normalize_name(name): name for name in pred_classes}
        true_norm = {_normalize_name(name): name for name in true_classes}
        # TODO: remove background hack, use "implicit_negative_classes" or "negative_classes"
        unified_names = list(ub.unique(['background'] + list(pred_norm) + list(true_norm)))
        classes = kwcoco.CategoryTree.coerce(unified_names)

        # raise Exception('\n'.join(errors))
        for true_name, true_cid in true_classes.node_to_id.items():
            true_norm_name = _normalize_name(true_name)
            cid = classes.node_to_id[true_norm_name]
            unified_cid_maps['true'][true_cid] = cid

        for pred_name, pred_cid in pred_classes.node_to_id.items():
            pred_norm_name = _normalize_name(pred_name)
            cid = classes.node_to_id[pred_norm_name]
            unified_cid_maps['pred'][pred_cid] = cid

        # if errors:
        #     graph2 = pred_classes.graph.copy()
        #     for node1, id1 in true_classes.node_to_id.items():
        #         if node1 not in pred_classes.node_to_id:
        #             graph2.add_node(node1, id=id1)
        #     classes = ndsampler.CategoryTree(graph2)

        return classes, unified_cid_maps

    @classmethod
    def _coerce_dets(CocoEvaluator, dataset, verbose=0):
        """
        Coerce the input to a mapping from image-id to kwimage.Detection

        Also capture a CocoDataset if possible.

        Returns:
            Tuple[Dict[int, Detections], Dict]:
                gid_to_det: mapping from gid to dets
                extra: any extra information we gathered via coercion

        Example:
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('shapes8')
            >>> gid_to_det, extras = CocoEvaluator._coerce_dets(coco_dset)
        """
        # coerce the input into dictionary of detection objects.
        import kwcoco
        # if 0:
        #     # hack
        #     isinstance = kwimage.structs._generic._isinstance2

        # We only need the box locations, but if we can coerce extra
        # information we will maintain that as well.
        extra = {}

        if isinstance(dataset, dict):
            if len(dataset):
                first = ub.peek(dataset.values())
                if isinstance(first, kwimage.Detections):
                    # We got what we wanted
                    gid_to_det = dataset
                else:
                    raise NotImplementedError
            else:
                gid_to_det = {}
        elif isinstance(dataset, kwcoco.CocoDataset):
            extra['coco_dset'] = coco_dset = dataset
            gid_to_det = {}
            gids = sorted(coco_dset.imgs.keys())
            classes = coco_dset.object_categories()
            for gid in ub.ProgIter(gids, desc='convert coco to dets'):
                aids = list(coco_dset.index.gid_to_aids[gid])
                anns = [coco_dset.anns[aid] for aid in aids]
                cids = [a['category_id'] for a in anns]
                # remap truth cids to be consistent with "classes"
                # cids = [cid_true_to_pred.get(cid, cid) for cid in cids]

                cxs = np.array([None if c is None else classes.id_to_idx[c] for c in cids])
                ssegs = [a.get('segmentation') for a in anns]
                weights = [a.get('weight', 1) for a in anns]
                scores = [a.get('score', np.nan) for a in anns]

                kw = {}
                if all('prob' in a for a in anns):
                    # TODO: can we ensure the probs are always in the proper
                    # order here? I think they are, but I'm not 100% sure.
                    kw['probs'] = [a['prob'] for a in anns]

                dets = kwimage.Detections(
                    boxes=kwimage.Boxes([a['bbox'] for a in anns], 'xywh'),
                    segmentations=ssegs,
                    class_idxs=cxs,
                    classes=classes,
                    weights=np.array(weights),
                    scores=np.array(scores),
                    aids=np.array(aids),
                    datakeys=['aids'],
                    **kw,
                ).numpy()
                gid_to_det[gid] = dets
        elif COCO_SAMPLER_CLS and isinstance(dataset, COCO_SAMPLER_CLS):
            # Input is an ndsampler.CocoSampler object
            extra['sampler'] = sampler = dataset
            coco_dset = sampler.dset
            gid_to_det, _extra = CocoEvaluator._coerce_dets(coco_dset, verbose)
            extra.update(_extra)
        elif isinstance(dataset, six.string_types):
            if exists(dataset):
                # on-disk detections
                if isdir(dataset):
                    if verbose:
                        print('Loading mscoco directory')
                    # directory of predictions
                    extra['coco_dpath'] = coco_dpath = dataset
                    pat = join(coco_dpath, '**/*.json')
                    coco_fpaths = sorted(glob.glob(pat, recursive=True))
                    dets, coco_dset = _load_dets(coco_fpaths)
                    extra['coco_dset'] = coco_dset
                    # coco_dset = kwcoco.CocoDataset.from_coco_paths(
                    #     coco_fpaths, max_workers=6, verbose=1, mode='process')
                    gid_to_det = {d.meta['gid']: d for d in dets}
                elif isfile(dataset):
                    # mscoco file
                    if verbose:
                        print('Loading mscoco file')
                    extra['dataset_fpath'] = coco_fpath = dataset
                    coco_dset = kwcoco.CocoDataset(coco_fpath)
                    gid_to_det, _extra = CocoEvaluator._coerce_dets(coco_dset, verbose)
                    extra.update(_extra)
                else:
                    raise NotImplementedError
            else:
                raise Exception('{!r} does not exist'.format(dataset))
        else:
            raise TypeError('Unknown dataset type: {!r}'.format(type(dataset)))

        return gid_to_det, extra

    def evaluate(coco_eval):
        """

        Example:
            >>> from kwcoco.coco_evaluator import *  # NOQA
            >>> from kwcoco.coco_evaluator import CocoEvaluator
            >>> import kwcoco
            >>> dpath = ub.ensure_app_cache_dir('kwcoco/tests/test_out_dpath')
            >>> true_dset = kwcoco.CocoDataset.demo('shapes8')
            >>> from kwcoco.demo.perterb import perterb_coco
            >>> kwargs = {
            >>>     'box_noise': 0.5,
            >>>     'n_fp': (0, 10),
            >>>     'n_fn': (0, 10),
            >>>     'with_probs': True,
            >>> }
            >>> pred_dset = perterb_coco(true_dset, **kwargs)
            >>> config = {
            >>>     'true_dataset': true_dset,
            >>>     'pred_dataset': pred_dset,
            >>>     'out_dpath': dpath,
            >>> }
            >>> coco_eval = CocoEvaluator(config)
            >>> results = coco_eval.evaluate()
        """
        coco_eval.log('evaluating')
        coco_eval._ensure_init()
        classes = coco_eval.classes

        # Ignore any categories with too few tests instances
        # print('coco_eval.config = {}'.format(ub.repr2(dict(coco_eval.config), nl=1)))

        negative_classes = coco_eval.config['implicit_negative_classes']
        classes_of_interest = coco_eval.config['classes_of_interest']
        ignore_classes = set(coco_eval.config['implicit_ignore_classes'])
        if coco_eval.config['ignore_classes']:
            ignore_classes.update(coco_eval.config['ignore_classes'])
        if classes_of_interest:
            ignore_classes.update(set(classes) - set(classes_of_interest))
        coco_eval.log('negative_classes = {!r}'.format(negative_classes))
        coco_eval.log('classes_of_interest = {!r}'.format(classes_of_interest))
        coco_eval.log('ignore_classes = {!r}'.format(ignore_classes))

        gid_to_true = coco_eval.gid_to_true
        gid_to_pred = coco_eval.gid_to_pred

        if 0:
            true_names = []
            for det in coco_eval.gid_to_true.values():
                class_idxs = det.data['class_idxs']
                cnames = list(ub.take(det.meta['classes'], class_idxs))
                true_names += cnames
            ub.dict_hist(true_names)

            pred_names = []
            for det in coco_eval.gid_to_pred.values():
                class_idxs = det.data['class_idxs']
                cnames = list(ub.take(det.meta['classes'], class_idxs))
                pred_names += cnames
            ub.dict_hist(pred_names)

        fp_cutoff = coco_eval.config['fp_cutoff']

        from kwcoco.metrics import DetectionMetrics
        dmet = DetectionMetrics(classes=classes)
        for gid in ub.ProgIter(coco_eval.gids):
            pred_dets = gid_to_pred[gid]
            true_dets = gid_to_true[gid]
            dmet.add_predictions(pred_dets, gid=gid)
            dmet.add_truth(true_dets, gid=gid)

        if 0:
            voc_info = dmet.score_voc(ignore_classes='ignore')
            print('voc_info = {!r}'.format(voc_info))

        # Detection only scoring
        print('Building confusion vectors')
        cfsn_vecs = dmet.confusion_vectors(ignore_classes=ignore_classes,
                                           ovthresh=0.5,  # compute AP@0.5iou
                                           workers=8)

        # Get pure per-item detection results
        binvecs = cfsn_vecs.binarize_peritem(negative_classes=negative_classes)

        measures = binvecs.measures(fp_cutoff=fp_cutoff)
        print('measures = {}'.format(ub.repr2(measures, nl=1)))

        # Get per-class detection results
        ovr_binvecs = cfsn_vecs.binarize_ovr(ignore_classes=ignore_classes)
        ovr_measures = ovr_binvecs.measures(fp_cutoff=fp_cutoff)['perclass']
        print('ovr_measures = {!r}'.format(ovr_measures))

        # Remove large datasets values in configs that are not file references
        meta = dict(coco_eval.config)
        if not isinstance(meta['true_dataset'], str):
            meta['true_dataset'] = '<not-a-file-ref>'
        if not isinstance(meta['pred_dataset'], str):
            meta['pred_dataset'] = '<not-a-file-ref>'

        results = CocoResults(
            measures,
            ovr_measures,
            cfsn_vecs,
            meta,
        )

        # TODO: when making the ovr localization curves, it might be a good
        # idea to include a second version where any COI prediction assigned
        # to a non-COI truth is given a weight of zero, so we can focus on
        # our TPR and FPR with respect to the COI itself and the background.
        # This metric is useful when we assume we have a subsequent classifier.
        if classes_of_interest:
            ovr_binvecs2 = cfsn_vecs.binarize_ovr(ignore_classes=ignore_classes)
            for key, vecs in ovr_binvecs2.cx_to_binvecs.items():
                cx = cfsn_vecs.classes.index(key)
                vecs.data['weight'] = vecs.data['weight'].copy()

                assert not np.may_share_memory(ovr_binvecs[key].data['weight'], vecs.data['weight'])

                # Find locations where the predictions or truth was COI
                pred_coi = cfsn_vecs.data['pred'] == cx
                # Find truth locations that are either background or this COI
                true_coi_or_bg = kwarray.isect_flags(
                        cfsn_vecs.data['true'], {cx, -1})

                # Find locations where we predicted this COI, but truth was a
                # valid classes, but not this non-COI
                ignore_flags = (pred_coi & (~true_coi_or_bg))
                vecs.data['weight'][ignore_flags] = 0

            ovr_measures2 = ovr_binvecs2.measures(fp_cutoff=fp_cutoff)['perclass']
            print('ovr_measures2 = {!r}'.format(ovr_measures2))
            results.ovr_measures2 = ovr_measures2

        if 0:
            # TODO: The evaluate method itself probably shouldn't do the drawing it
            # should be the responsibility of the caller.
            if coco_eval.config['draw']:
                results.dump_figures(
                    coco_eval.config['out_dpath'],
                    expt_title=coco_eval.config['expt_title']
                )
        return results


class CocoResults(ub.NiceRepr):
    """
    Container class to store, draw, summarize, and serialize results from
    CocoEvaluator.
    """

    def __init__(results, measures, ovr_measures, cfsn_vecs, meta=None):
        results.measures = measures
        results.ovr_measures = ovr_measures
        results.cfsn_vecs = cfsn_vecs
        results.meta = meta

    def __nice__(results):
        text = ub.repr2({
            'measures': results.measures,
            'ovr_measures': results.ovr_measures,
        }, sv=1)
        return text

    def __json__(results):
        state = {
            'measures': results.measures.__json__(),
            'ovr_measures': results.ovr_measures.__json__(),
            'cfsn_vecs': results.cfsn_vecs.__json__(),
            'meta': results.meta,
        }
        from kwcoco.util.util_json import ensure_json_serializable
        ensure_json_serializable(state, normalize_containers=True, verbose=0)
        return state

    def dump(results, file, indent='    '):
        """
        Serialize to json file
        """
        if isinstance(file, str):
            with open(file, 'w') as fp:
                return results.dump(fp, indent=indent)
        else:
            import json
            state = results.__json__()
            json.dump(state, file, indent=indent)

    def dump_figures(results, out_dpath, expt_title=None):
        # classes_of_interest=[], ignore_classes=None,
        # if 0:
        #     cname = 'flatfish'
        #     cx = cfsn_vecs.classes.index(cname)
        #     is_true = (cfsn_vecs.data['true'] == cx)
        #     num_localized = (cfsn_vecs.data['pred'][is_true] != -1).sum()
        #     num_missed = is_true.sum() - num_localized
        # metrics_dpath = ub.ensuredir(coco_eval.config['out_dpath'])
        if expt_title is None:
            expt_title = results.meta.get('expt_title', '')
        metrics_dpath = ub.ensuredir(out_dpath)
        # coco_eval.config['out_dpath'])

        measures = results.measures
        ovr_measures = results.ovr_measures

        # TODO: separate into standalone method that is able to run on
        # serialized / cached metrics on disk.
        print('drawing evaluation metrics')
        import kwplot
        # TODO: kwcoco matplotlib backend context
        kwplot.autompl()

        import seaborn
        seaborn.set()

        figsize = (9, 7)

        fig = kwplot.figure(fnum=1, pnum=(1, 2, 1), doclf=True,
                            figtitle=expt_title)
        fig.set_size_inches(figsize)
        measures.draw(key='pr')
        kwplot.figure(fnum=1, pnum=(1, 2, 2))
        measures.draw(key='roc')
        fig_fpath = join(metrics_dpath, 'loc_pr_roc.png')
        print('write fig_fpath = {!r}'.format(fig_fpath))
        fig.savefig(fig_fpath)

        fig = kwplot.figure(fnum=1, pnum=(1, 1, 1), doclf=True,
                            figtitle=expt_title)
        fig.set_size_inches(figsize)
        measures.draw(key='thresh')
        fig_fpath = join(metrics_dpath, 'loc_thresh.png')
        print('write fig_fpath = {!r}'.format(fig_fpath))
        fig.savefig(fig_fpath)

        fig = kwplot.figure(fnum=2, pnum=(1, 1, 1), doclf=True,
                            figtitle=expt_title)
        fig.set_size_inches(figsize)
        ovr_measures.draw(key='roc', fnum=2)

        fig_fpath = join(metrics_dpath, 'perclass_roc.png')
        print('write fig_fpath = {!r}'.format(fig_fpath))
        fig.savefig(fig_fpath)

        fig = kwplot.figure(fnum=2, pnum=(1, 1, 1), doclf=True,
                            figtitle=expt_title)
        fig.set_size_inches(figsize)
        ovr_measures.draw(key='pr', fnum=2)
        fig_fpath = join(metrics_dpath, 'perclass_pr.png')
        print('write fig_fpath = {!r}'.format(fig_fpath))
        fig.savefig(fig_fpath)

        if hasattr(results, 'ovr_measures2'):
            ovr_measures2 = results.ovr_measures2
            fig = kwplot.figure(fnum=2, pnum=(1, 1, 1), doclf=True,
                                figtitle=expt_title)
            fig.set_size_inches(figsize)
            ovr_measures2.draw(key='pr', fnum=2, prefix='coi-vs-bg-only')
            fig_fpath = join(metrics_dpath, 'perclass_pr_coi_vs_bg.png')
            print('write fig_fpath = {!r}'.format(fig_fpath))
            fig.savefig(fig_fpath)

            fig = kwplot.figure(fnum=2, pnum=(1, 1, 1), doclf=True,
                                figtitle=expt_title)
            fig.set_size_inches(figsize)
            ovr_measures2.draw(key='roc', fnum=2, prefix='coi-vs-bg-only')
            fig_fpath = join(metrics_dpath, 'perclass_roc_coi_vs_bg.png')
            print('write fig_fpath = {!r}'.format(fig_fpath))
            fig.savefig(fig_fpath)

        # keys = ['mcc', 'g1', 'f1', 'acc', 'ppv', 'tpr', 'mk', 'bm']
        keys = ['mcc', 'f1', 'ppv', 'tpr']
        for key in keys:
            fig = kwplot.figure(fnum=2, pnum=(1, 1, 1), doclf=True,
                                figtitle=expt_title)
            fig.set_size_inches(figsize)
            ovr_measures.draw(fnum=2, key=key)
            fig_fpath = join(metrics_dpath, 'perclass_{}.png'.format(key))
            print('write fig_fpath = {!r}'.format(fig_fpath))
            fig.savefig(fig_fpath)

        # NOTE: The threshold on these confusion matrices is VERY low.
        # FIXME: robustly skip in cases where predictions have no class information
        try:
            cfsn_vecs = results.cfsn_vecs
            fig = kwplot.figure(fnum=3, doclf=True)
            confusion = cfsn_vecs.confusion_matrix()
            import kwplot
            ax = kwplot.plot_matrix(confusion, fnum=3, showvals=0, logscale=True)
            fig_fpath = join(metrics_dpath, 'confusion.png')
            print('write fig_fpath = {!r}'.format(fig_fpath))
            ax.figure.savefig(fig_fpath)

            # classes_of_interest = coco_eval.config['classes_of_interest']
            # if classes_of_interest:
            #     # coco_eval.config['implicit_negative_classes']
            #     subkeys = ['background'] + classes_of_interest
            #     coi_confusion = confusion[subkeys].loc[subkeys]
            #     ax = kwplot.plot_matrix(coi_confusion, fnum=3, showvals=0, logscale=True)
            #     fig_fpath = join(metrics_dpath, 'confusion_coi.png')
            #     print('write fig_fpath = {!r}'.format(fig_fpath))
            #     ax.figure.savefig(fig_fpath)

            fig = kwplot.figure(fnum=3, doclf=True)
            row_norm_cfsn = confusion / confusion.values.sum(axis=1, keepdims=True)
            row_norm_cfsn = row_norm_cfsn.fillna(0)
            ax = kwplot.plot_matrix(row_norm_cfsn, fnum=3, showvals=0, logscale=0)
            ax.set_title('Row (truth) normalized confusions')
            fig_fpath = join(metrics_dpath, 'row_confusion.png')
            print('write fig_fpath = {!r}'.format(fig_fpath))
            ax.figure.savefig(fig_fpath)

            fig = kwplot.figure(fnum=3, doclf=True)
            col_norm_cfsn = confusion / confusion.values.sum(axis=0, keepdims=True)
            col_norm_cfsn = col_norm_cfsn.fillna(0)
            ax = kwplot.plot_matrix(col_norm_cfsn, fnum=3, showvals=0, logscale=0)
            ax.set_title('Column (pred) normalized confusions')
            fig_fpath = join(metrics_dpath, 'col_confusion.png')
            print('write fig_fpath = {!r}'.format(fig_fpath))
            ax.figure.savefig(fig_fpath)
        except Exception:
            pass


def _load_dets(pred_fpaths, workers=6):
    """
    Example:
        >>> from kwcoco.coco_evaluator import _load_dets, _load_dets_worker
        >>> import ubelt as ub
        >>> import kwcoco
        >>> from os.path import join
        >>> dpath = ub.ensure_app_cache_dir('kwcoco/tests/load_dets')
        >>> N = 4
        >>> pred_fpaths = []
        >>> for i in range(1, N + 1):
        >>>     dset = kwcoco.CocoDataset.demo('shapes{}'.format(i))
        >>>     dset.fpath = join(dpath, 'shapes_{}.mscoco.json'.format(i))
        >>>     dset.dump(dset.fpath)
        >>>     pred_fpaths.append(dset.fpath)
        >>> dets, coco_dset = _load_dets(pred_fpaths)
        >>> print('dets = {!r}'.format(dets))
        >>> print('coco_dset = {!r}'.format(coco_dset))
    """
    # Process mode is much faster than thread.
    import kwcoco
    from kwcoco.util import util_futures
    jobs = util_futures.JobPool(mode='process', max_workers=workers)
    for single_pred_fpath in ub.ProgIter(pred_fpaths, desc='submit load dets jobs'):
        job = jobs.submit(_load_dets_worker, single_pred_fpath, with_coco=True)
    results = []
    for job in ub.ProgIter(jobs.jobs, total=len(jobs), desc='loading cached dets'):
        results.append(job.result())
    dets = [r[0] for r in results]
    pred_cocos = [r[1] for r in results]
    coco_dset = kwcoco.CocoDataset.union(*pred_cocos)
    return dets, coco_dset


def _load_dets_worker(single_pred_fpath, with_coco=True):
    """
    Ignore:
        >>> from kwcoco.coco_evaluator import _load_dets, _load_dets_worker
        >>> import ubelt as ub
        >>> import kwcoco
        >>> from os.path import join
        >>> dpath = ub.ensure_app_cache_dir('kwcoco/tests/load_dets')
        >>> dset = kwcoco.CocoDataset.demo('shapes8')
        >>> dset.fpath = join(dpath, 'shapes8.mscoco.json')
        >>> dset.dump(dset.fpath)
        >>> single_pred_fpath = dset.fpath
        >>> dets, coco = _load_dets_worker(single_pred_fpath)
        >>> print('dets = {!r}'.format(dets))
        >>> print('coco = {!r}'.format(coco))
    """
    import kwcoco
    single_img_coco = kwcoco.CocoDataset(single_pred_fpath, autobuild=False)
    dets = kwimage.Detections.from_coco_annots(single_img_coco.dataset['annotations'],
                                               dset=single_img_coco)
    if len(single_img_coco.dataset['images']) == 1:
        # raise Exception('Expected predictions for a single image only')
        gid = single_img_coco.dataset['images'][0]['id']
        dets.meta['gid'] = gid
    else:
        import warnings
        warnings.warn('Loading dets with muliple images, must track gids carefully')

    if with_coco:
        return dets, single_img_coco
    else:
        return dets


class CocoEvalCLIConfig(scfg.Config):
    default = ub.dict_union(CocoEvalConfig.default, {
        # These should go into the CLI args, not the class config args
        'expt_title': scfg.Value('', type=str, help='title for plots'),
        'draw': scfg.Value(True, help='draw metric plots'),
        'out_dpath': scfg.Value('./coco_metrics', type=str),
    })


def main(cmdline=True, **kw):
    config = CocoEvalCLIConfig(cmdline=cmdline, default=kw)
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))

    coco_eval = CocoEvaluator(config)
    coco_eval._init()

    results = coco_eval.evaluate()

    ub.ensuredir(config['out_dpath'])

    if 1:
        metrics_fpath = join(config['out_dpath'], 'metrics.json')
        print('dumping metrics_fpath = {!r}'.format(metrics_fpath))
        results.dump(metrics_fpath, indent='    ')
    else:
        import json
        with open(join(config['out_dpath'], 'meta.json'), 'w') as file:
            state = results.meta
            json.dump(state, file, indent='    ')

        with open(join(config['out_dpath'], 'measures.json'), 'w') as file:
            state = results.measures.__json__()
            json.dump(state, file, indent='    ')

        with open(join(config['out_dpath'], 'ovr_measures.json'), 'w') as file:
            state = results.ovr_measures.__json__()
            json.dump(state, file, indent='    ')

        with open(join(config['out_dpath'], 'cfsn_vecs.json'), 'w') as file:
            state = results.cfsn_vecs.__json__()
            json.dump(state, file, indent='    ')

    if config['draw']:
        results.dump_figures(
            config['out_dpath'],
            expt_title=config['expt_title']
        )

    if 'coco_dset' in coco_eval.true_extra:
        truth_dset = coco_eval.true_extra['coco_dset']
    elif 'sampler' in coco_eval.true_extra:
        truth_dset = coco_eval.true_extra['sampler'].dset
    else:
        truth_dset = None

    if truth_dset is not None:
        print('Attempting to draw examples')
        gid_to_stats = {}
        import kwarray
        gids, groupxs = kwarray.group_indices(results.cfsn_vecs.data['gid'])
        for gid, groupx in zip(gids, groupxs):
            true_vec = results.cfsn_vecs.data['true'][groupx]
            pred_vec = results.cfsn_vecs.data['pred'][groupx]
            is_true = (true_vec > 0)
            is_pred = (pred_vec > 0)
            has_pred = is_true & is_pred

            stats = {
                'num_assigned_pred': has_pred.sum(),
                'num_true': is_true.sum(),
                'num_pred': is_pred.sum(),
            }
            stats['frac_assigned'] = stats['num_assigned_pred'] / stats['num_true']
            gid_to_stats[gid] = stats

        set([stats['frac_assigned'] for stast in gid_to_stats.values()])
        gid = ub.argmax(gid_to_stats, key=lambda x: x['num_pred'] * x['num_true'])
        stat_gids = [gid]

        rng = kwarray.ensure_rng(None)
        random_gids = rng.choice(gids, size=5).tolist()
        # import random
        # random_gids = random.choices(gids, k=5)
        found_gids = truth_dset.find_representative_images(gids)
        draw_gids = list(ub.unique(found_gids + stat_gids + random_gids))

        for gid in ub.ProgIter(draw_gids):
            truth_dets = coco_eval.gid_to_true[gid]
            pred_dets = coco_eval.gid_to_pred[gid]

            canvas = truth_dset.load_image(gid)
            canvas = truth_dets.draw_on(canvas, color='green', sseg=False)
            canvas = pred_dets.draw_on(canvas, color='blue', sseg=False)

            viz_dpath = ub.ensuredir((config['out_dpath'], 'viz'))
            fig_fpath = join(viz_dpath, 'eval-gid={}.jpg'.format(gid))
            kwimage.imwrite(fig_fpath, canvas)

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/kwcoco/coco_evaluator.py
    """
    main()
