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


class CocoEvalConfig(scfg.Config):
    default = {
        'true_dataset': scfg.Value(None, help='coercable true detections'),
        'pred_dataset': scfg.Value(None, help='coercable predicted detections'),

        'classes_of_interest': scfg.Value(None, type=list, help='if specified only these classes are given weight'),
        'ignore_classes': scfg.Value(None, type=list, help='classes to ignore'),

        'draw': scfg.Value(True, help='draw metric plots'),
        'out_dpath': scfg.Value('./coco_metrics'),

        'fp_cutoff': scfg.Value(float('inf'), help='false positive cutoff for ROC'),
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
    """

    def __init__(coco_eval, config):
        coco_eval.config = CocoEvalConfig(config)

    def _init(coco_eval):
        # TODO: coerce into a cocodataset form if possible

        print('init truth dset')
        gid_to_true, true_extra = CocoEvaluator._coerce_dets(
            coco_eval.config['true_dataset'])

        print('init pred dset')
        gid_to_pred, pred_extra = CocoEvaluator._coerce_dets(
            coco_eval.config['pred_dataset'])

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
            old_cids = [old_classes.idx_to_id[cx] for cx in old_cidxs]
            new_cids = [true_to_unified_cid.get(cid, cid) for cid in old_cids]
            new_cidxs = np.array([new_classes.id_to_idx[c] for c in new_cids])
            det.meta['classes'] = new_classes
            det.data['class_idxs'] = new_cidxs

        # Move predictions to the unified class indices
        for gid in ub.ProgIter(gids, desc='Rectify pred class idxs'):
            det = gid_to_pred[gid]
            new_classes = classes
            old_classes = det.meta['classes']
            old_cidxs = det.data['class_idxs']
            old_cids = [old_classes.idx_to_id[cx] for cx in old_cidxs]
            new_cids = [pred_to_unified_cid.get(cid, cid) for cid in old_cids]
            new_cidxs = np.array([new_classes.id_to_idx[c] for c in new_cids])
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

    def evaluate(coco_eval):

        classes_of_interest = coco_eval.config['classes_of_interest']
        ignore_classes = coco_eval.config['ignore_classes']

        classes = coco_eval.classes
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

        # n_true_annots = sum(map(len, gid_to_true.values()))
        # fp_cutoff = n_true_annots
        fp_cutoff = coco_eval.config['fp_cutoff']
        # fp_cutoff = None

        from netharn.metrics import DetectionMetrics
        dmet = DetectionMetrics(classes=classes)
        for gid in ub.ProgIter(coco_eval.gids):
            pred_dets = gid_to_pred[gid]
            true_dets = gid_to_true[gid]
            dmet.add_predictions(pred_dets, gid=gid)
            dmet.add_truth(true_dets, gid=gid)

        if 0:
            voc_info = dmet.score_voc(ignore_classes='ignore')
            print('voc_info = {!r}'.format(voc_info))

        # Ignore any categories with too few tests instances
        if ignore_classes is None:
            ignore_classes = {'ignore'}

        if classes_of_interest:
            ignore_classes.update(set(classes) - set(classes_of_interest))

        # Detection only scoring
        print('Building confusion vectors')
        cfsn_vecs = dmet.confusion_vectors(ignore_classes=ignore_classes,
                                           workers=8)

        negative_classes = ['background']

        print('negative_classes = {!r}'.format(negative_classes))
        print('classes_of_interest = {!r}'.format(classes_of_interest))
        print('ignore_classes = {!r}'.format(ignore_classes))

        # Get pure per-item detection results
        binvecs = cfsn_vecs.binarize_peritem(negative_classes=negative_classes)

        roc_result = binvecs.roc(fp_cutoff=fp_cutoff)
        pr_result = binvecs.precision_recall()
        thresh_result = binvecs.threshold_curves()

        print('roc_result = {!r}'.format(roc_result))
        print('pr_result = {!r}'.format(pr_result))
        print('thresh_result = {!r}'.format(thresh_result))

        # Get per-class detection results
        ovr_binvecs = cfsn_vecs.binarize_ovr(ignore_classes=ignore_classes)
        ovr_roc_result = ovr_binvecs.roc(fp_cutoff=fp_cutoff)['perclass']
        ovr_pr_result = ovr_binvecs.precision_recall()['perclass']
        ovr_thresh_result = ovr_binvecs.threshold_curves()['perclass']

        print('ovr_roc_result = {!r}'.format(ovr_roc_result))
        print('ovr_pr_result = {!r}'.format(ovr_pr_result))
        # print('ovr_thresh_result = {!r}'.format(ovr_thresh_result))

        results = {
            'cfsn_vecs': cfsn_vecs,

            'roc_result': roc_result,
            'pr_result': pr_result,
            'thresh_result': thresh_result,

            'ovr_roc_result': ovr_roc_result,
            'ovr_pr_result': ovr_pr_result,
            'ovr_thresh_result': ovr_thresh_result,
        }

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

            ovr_roc_result2 = ovr_binvecs2.roc(fp_cutoff=fp_cutoff)['perclass']
            ovr_pr_result2 = ovr_binvecs2.precision_recall()['perclass']
            # ovr_thresh_result2 = ovr_binvecs2.threshold_curves()['perclass']
            print('ovr_roc_result2 = {!r}'.format(ovr_roc_result2))
            print('ovr_pr_result2 = {!r}'.format(ovr_pr_result2))
            # print('ovr_thresh_result2 = {!r}'.format(ovr_thresh_result2))
            results.update({
                'ovr_roc_result2': ovr_roc_result2,
                'ovr_pr_result2': ovr_pr_result2,
            })

        # FIXME: there is a lot of redundant information here,
        # this needs to be consolidated both here and in netharn metrics
        # metrics_dpath = coco_eval.config['out_dpath']
        if coco_eval.config['draw']:
            coco_eval.plot_results(results)
        return results

    def plot_results(coco_eval, results, expt_title):
        # classes_of_interest=[], ignore_classes=None,
        # if 0:
        #     cname = 'flatfish'
        #     cx = cfsn_vecs.classes.index(cname)
        #     is_true = (cfsn_vecs.data['true'] == cx)
        #     num_localized = (cfsn_vecs.data['pred'][is_true] != -1).sum()
        #     num_missed = is_true.sum() - num_localized
        metrics_dpath = ub.ensuredir(coco_eval.config['out_dpath'])

        classes_of_interest = coco_eval.config['classes_of_interest']

        pr_result = results['pr_result']
        roc_result = results['roc_result']
        thresh_result = results['thresh_result']

        ovr_pr_result = results['ovr_pr_result']
        ovr_roc_result = results['ovr_roc_result']
        ovr_thresh_result = results['ovr_thresh_result']

        cfsn_vecs = results['cfsn_vecs']

        # TODO: separate into standalone method that is able to run on
        # serialized / cached metrics on disk.
        print('drawing evaluation metrics')
        import kwplot
        kwplot.autompl()

        import seaborn
        seaborn.set()

        figsize = (9, 7)

        fig = kwplot.figure(fnum=1, pnum=(1, 2, 1), doclf=True,
                            figtitle=expt_title)
        fig.set_size_inches(figsize)
        pr_result.draw()
        kwplot.figure(fnum=1, pnum=(1, 2, 2))
        roc_result.draw()
        fig_fpath = join(metrics_dpath, 'loc_pr_roc.png')
        print('write fig_fpath = {!r}'.format(fig_fpath))
        fig.savefig(fig_fpath)

        fig = kwplot.figure(fnum=1, pnum=(1, 1, 1), doclf=True,
                            figtitle=expt_title)
        fig.set_size_inches(figsize)
        thresh_result.draw()
        fig_fpath = join(metrics_dpath, 'loc_thresh.png')
        print('write fig_fpath = {!r}'.format(fig_fpath))
        fig.savefig(fig_fpath)

        fig = kwplot.figure(fnum=2, pnum=(1, 1, 1), doclf=True,
                            figtitle=expt_title)
        fig.set_size_inches(figsize)
        ovr_roc_result.draw(fnum=2)

        fig_fpath = join(metrics_dpath, 'perclass_roc.png')
        print('write fig_fpath = {!r}'.format(fig_fpath))
        fig.savefig(fig_fpath)

        fig = kwplot.figure(fnum=2, pnum=(1, 1, 1), doclf=True,
                            figtitle=expt_title)
        fig.set_size_inches(figsize)
        ovr_pr_result.draw(fnum=2)
        fig_fpath = join(metrics_dpath, 'perclass_pr.png')
        print('write fig_fpath = {!r}'.format(fig_fpath))
        fig.savefig(fig_fpath)

        if 'ovr_pr_result2' in results:
            ovr_pr_result2 = results['ovr_pr_result2']
            ovr_roc_result2 = results['ovr_roc_result2']
            fig = kwplot.figure(fnum=2, pnum=(1, 1, 1), doclf=True,
                                figtitle=expt_title)
            fig.set_size_inches(figsize)
            ovr_pr_result2.draw(fnum=2, prefix='coi-vs-bg-only')
            fig_fpath = join(metrics_dpath, 'perclass_pr_coi_vs_bg.png')
            print('write fig_fpath = {!r}'.format(fig_fpath))
            fig.savefig(fig_fpath)

            fig = kwplot.figure(fnum=2, pnum=(1, 1, 1), doclf=True,
                                figtitle=expt_title)
            fig.set_size_inches(figsize)
            ovr_roc_result2.draw(fnum=2, prefix='coi-vs-bg-only')
            fig_fpath = join(metrics_dpath, 'perclass_roc_coi_vs_bg.png')
            print('write fig_fpath = {!r}'.format(fig_fpath))
            fig.savefig(fig_fpath)

        # keys = ['mcc', 'g1', 'f1', 'acc', 'ppv', 'tpr', 'mk', 'bm']
        keys = ['mcc', 'f1', 'ppv', 'tpr']
        for key in keys:
            fig = kwplot.figure(fnum=2, pnum=(1, 1, 1), doclf=True,
                                figtitle=expt_title)
            fig.set_size_inches(figsize)
            ovr_thresh_result.draw(fnum=2, key=key)
            fig_fpath = join(metrics_dpath, 'perclass_{}.png'.format(key))
            print('write fig_fpath = {!r}'.format(fig_fpath))
            fig.savefig(fig_fpath)

        # NOTE: The threshold on these confusion matrices is VERY low.
        # FIXME: robustly skip in cases where predictions have no class information
        try:
            fig = kwplot.figure(fnum=3, doclf=True)
            confusion = cfsn_vecs.confusion_matrix()
            import kwplot
            ax = kwplot.plot_matrix(confusion, fnum=3, showvals=0, logscale=True)
            fig_fpath = join(metrics_dpath, 'confusion.png')
            print('write fig_fpath = {!r}'.format(fig_fpath))
            ax.figure.savefig(fig_fpath)

            if classes_of_interest:
                subkeys = ['background'] + classes_of_interest
                coi_confusion = confusion[subkeys].loc[subkeys]
                ax = kwplot.plot_matrix(coi_confusion, fnum=3, showvals=0, logscale=True)
                fig_fpath = join(metrics_dpath, 'confusion_coi.png')
                print('write fig_fpath = {!r}'.format(fig_fpath))
                ax.figure.savefig(fig_fpath)

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

    @classmethod
    def _rectify_classes(coco_eval, true_classes, pred_classes):
        import ndsampler
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
        unified_names = list(ub.unique(['background'] + list(pred_norm) + list(true_norm)))
        classes = ndsampler.CategoryTree.coerce(unified_names)

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

        Returns:
            Tuple[Dict[int, Detections], Dict]:
                gid_to_det: mapping from gid to dets
                extra: any extra information we gathered via coercion

        Example:
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('shapes8')
            >>> gid_to_det, extras = CocoEvaluator._coerce_dets(coco_dset)

        Ignore:
            >>> dataset = ub.expandpath('$HOME/remote/namek/tmp/cached_clf_out_cli/reclassified.mscoco.json')
            >>> gid_to_pred, extras = CocoEvaluator._coerce_dets(dataset)
        """
        # coerce the input into dictionary of detection objects.
        import kwcoco
        import ndsampler
        if 1:
            # hack
            isinstance = kwimage.structs._generic._isinstance2

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
                aids = coco_dset.index.gid_to_aids[gid]
                anns = [coco_dset.anns[aid] for aid in aids]
                cids = [a['category_id'] for a in anns]
                # remap truth cids to be consistent with "classes"
                # cids = [cid_true_to_pred.get(cid, cid) for cid in cids]

                cxs = np.array([classes.id_to_idx[c] for c in cids])
                ssegs = [a.get('segmentation') for a in anns]
                weights = [a.get('weight', 1) for a in anns]
                scores = [a.get('score', np.nan) for a in anns]

                kw = {}
                if all('prob' in a for a in anns):
                    kw['probs'] = [a['prob'] for a in anns]

                dets = kwimage.Detections(
                    boxes=kwimage.Boxes([a['bbox'] for a in anns], 'xywh'),
                    segmentations=ssegs,
                    class_idxs=cxs,
                    classes=classes,
                    weights=np.array(weights),
                    scores=np.array(scores),
                    **kw,
                ).numpy()
                gid_to_det[gid] = dets
        elif isinstance(dataset, ndsampler.CocoSampler):
            # Input is an ndsampler object
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
                    coco_fpaths = sorted(glob.glob(join(coco_dpath, '*.json')))
                    dets = _load_dets(coco_fpaths)
                    gid_to_det = {d.meta['gid']: d for d in dets}
                    pass
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
            raise NotImplementedError

        return gid_to_det, extra


def _load_dets(pred_fpaths, workers=6):
    # Process mode is much faster than thread.
    from kwcoco.utils import util_futures
    jobs = util_futures.JobPool(mode='process', max_workers=workers)
    for single_pred_fpath in ub.ProgIter(pred_fpaths, desc='submit load dets jobs'):
        job = jobs.submit(_load_dets_worker, single_pred_fpath)
    dets = []
    for job in ub.ProgIter(jobs.jobs, total=len(jobs), desc='loading cached dets'):
        dets.append(job.result())
    return dets


def _load_dets_worker(single_pred_fpath):
    """
    single_pred_fpath = ub.expandpath('$HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-v36/brekugqz/eval/habcam_cfarm_v6_test.mscoc/bioharn-det-mc-cascade-rgbd-v36__epoch_00000018/c=0.2,i=window,n=0.5,window_d=512,512,window_o=0.0/pred/dets_gid_00004070_v2.mscoco.json')
    """
    import kwcoco
    single_img_coco = kwcoco.CocoDataset(single_pred_fpath, autobuild=False)
    if len(single_img_coco.dataset['images']) != 1:
        raise Exception('Expected predictions for a single image only')
    gid = single_img_coco.dataset['images'][0]['id']
    dets = kwimage.Detections.from_coco_annots(single_img_coco.dataset['annotations'],
                                               dset=single_img_coco)
    dets.meta['gid'] = gid
    return dets
