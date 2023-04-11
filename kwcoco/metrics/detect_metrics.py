import numpy as np
import ubelt as ub
import networkx as nx
# from .assignment import _assign_confusion_vectors
from kwcoco.metrics.confusion_vectors import ConfusionVectors
from kwcoco.metrics.assignment import _assign_confusion_vectors


# Helper for xdev docstubs
__docstubs__ = """
from kwcoco.metrics.confusion_vectors import ConfusionVectors
"""


class DetectionMetrics(ub.NiceRepr):
    """
    Object that computes associations between detections and can convert them
    into sklearn-compatible representations for scoring.

    Attributes:
        gid_to_true_dets (Dict[int, kwimage.Detections]):
            maps image ids to truth

        gid_to_pred_dets (Dict[int, kwimage.Detections]):
            maps image ids to predictions

        classes (kwcoco.CategoryTree | None):
            the categories to be scored, if unspecified attempts to
            determine these from the truth detections

    Example:
        >>> # Demo how to use detection metrics directly given detections only
        >>> # (no kwcoco file required)
        >>> from kwcoco.metrics import detect_metrics
        >>> import kwimage
        >>> # Setup random true detections (these are just boxes and scores)
        >>> true_dets = kwimage.Detections.random(3)
        >>> # Peek at the simple internals of a detections object
        >>> print('true_dets.data = {}'.format(ub.urepr(true_dets.data, nl=1)))
        >>> # Create similar but different predictions
        >>> true_subset = true_dets.take([1, 2]).warp(kwimage.Affine.coerce({'scale': 1.1}))
        >>> false_positive = kwimage.Detections.random(3)
        >>> pred_dets = kwimage.Detections.concatenate([true_subset, false_positive])
        >>> dmet = DetectionMetrics()
        >>> dmet.add_predictions(pred_dets, imgname='image1')
        >>> dmet.add_truth(true_dets, imgname='image1')
        >>> # Raw confusion vectors
        >>> cfsn_vecs = dmet.confusion_vectors()
        >>> print(cfsn_vecs.data.pandas().to_string())
        >>> # Our scoring definition (derived from confusion vectors)
        >>> print(dmet.score_kwcoco())
        >>> # VOC scoring
        >>> print(dmet.score_voc(bias=0))
        >>> # Original pycocotools scoring
        >>> # xdoctest: +REQUIRES(module:pycocotools)
        >>> print(dmet.score_pycocotools())

    Example:
        >>> dmet = DetectionMetrics.demo(
        >>>     nimgs=100, nboxes=(0, 3), n_fp=(0, 1), classes=8, score_noise=0.9, hacked=False)
        >>> print(dmet.score_kwcoco(bias=0, compat='mutex', prioritize='iou')['mAP'])
        ...
        >>> # NOTE: IN GENERAL NETHARN AND VOC ARE NOT THE SAME
        >>> print(dmet.score_voc(bias=0)['mAP'])
        0.8582...
        >>> #print(dmet.score_coco()['mAP'])
    """
    def __init__(dmet, classes=None):
        dmet.classes = classes
        dmet.gid_to_true_dets = {}
        dmet.gid_to_pred_dets = {}
        dmet._imgname_to_gid = {}

    def clear(dmet):
        dmet.gid_to_true_dets = {}
        dmet.gid_to_pred_dets = {}
        dmet._imgname_to_gid = {}

    def __nice__(dmet):
        info = {
            'n_true_imgs': len(dmet.gid_to_true_dets),
            'n_pred_imgs': len(dmet.gid_to_pred_dets),
            'n_true_anns': sum(map(len, dmet.gid_to_true_dets.values())),
            'n_pred_anns': sum(map(len, dmet.gid_to_pred_dets.values())),
            'classes': dmet.classes,
        }
        return ub.urepr(info)

    def enrich_confusion_vectors(dmet, cfsn_vecs):
        """
        Adds annotation id information into confusion vectors computed
        via this detection metrics object.

        TODO: should likely use this at the end of the function that builds the
        confusion vectors.
        """
        import kwarray
        cfsn_data = cfsn_vecs.data
        all_gids = cfsn_data['gid']
        all_txs = cfsn_data['txs']
        all_pxs = cfsn_data['pxs']

        # Initialize new annotation-id columns to add to confusion vectors
        cfsn_pred_aids = np.full(len(all_gids), fill_value=-1, dtype=int)
        cfsn_true_aids = np.full(len(all_gids), fill_value=-1, dtype=int)

        # For each image
        unique_gids, grouped_indexes = kwarray.group_indices(all_gids)
        for gid, rowxs in zip(unique_gids, grouped_indexes):
            # Find the confusion rows that correspond to it
            txs = all_txs[rowxs]
            pxs = all_pxs[rowxs]
            # Filter to only valid indexes and their row index in the
            # confusion vectors.
            tx_flags = txs >= 0
            px_flags = pxs >= 0
            valid_txs = txs[tx_flags]
            valid_pxs = pxs[px_flags]
            valid_tx_rowxs = rowxs[tx_flags]
            valid_px_rowxs = rowxs[px_flags]
            # Get the true and predicted detection annotation ids
            # the txs and pxs index into these arrays.
            pred_aids = dmet.gid_to_pred_dets[gid].data['aids']
            true_aids = dmet.gid_to_true_dets[gid].data['aids']
            valid_true_aids = true_aids[valid_txs]
            valid_pred_aids = pred_aids[valid_pxs]
            # Assign the annotation ids to their appropriate confusion rows
            # in the confusion vectors.
            cfsn_true_aids[valid_tx_rowxs] = valid_true_aids
            cfsn_pred_aids[valid_px_rowxs] = valid_pred_aids

        cfsn_data['true_aid'] = cfsn_true_aids
        cfsn_data['pred_aid'] = cfsn_pred_aids

    @classmethod
    def from_coco(DetectionMetrics, true_coco, pred_coco, gids=None, verbose=0):
        """
        Create detection metrics from two coco files representing the truth and
        predictions.

        Args:
            true_coco (kwcoco.CocoDataset): coco dataset with ground truth
            pred_coco (kwcoco.CocoDataset): coco dataset with predictions

        Example:
            >>> import kwcoco
            >>> from kwcoco.demo.perterb import perterb_coco
            >>> true_coco = kwcoco.CocoDataset.demo('shapes')
            >>> perterbkw = dict(box_noise=0.5, cls_noise=0.5, score_noise=0.5)
            >>> pred_coco = perterb_coco(true_coco, **perterbkw)
            >>> self = DetectionMetrics.from_coco(true_coco, pred_coco)
            >>> self.score_voc()
        """
        # import kwimage
        classes = true_coco.object_categories()
        self = DetectionMetrics(classes)

        if gids is None:
            gids = sorted(set(true_coco.imgs.keys()) & set(pred_coco.imgs.keys()))

        def _coco_to_dets(coco_dset, desc=''):
            import kwimage
            for gid in ub.ProgIter(gids, desc=desc, verbose=verbose):
                img = coco_dset.imgs[gid]
                imgname = img['file_name']
                aids = coco_dset.gid_to_aids[gid]
                annots = [coco_dset.anns[aid] for aid in aids]
                # dets = true_coco.annots(gid=gid).detections
                dets = kwimage.Detections.from_coco_annots(
                    annots, dset=coco_dset, classes=classes)
                yield dets, imgname, gid

        for dets, imgname, gid in _coco_to_dets(true_coco, desc='add truth'):
            self.add_truth(dets, imgname, gid=gid)

        for dets, imgname, gid in _coco_to_dets(pred_coco, desc='add pred'):
            self.add_predictions(dets, imgname, gid=gid)

        return self

    def _register_imagename(dmet, imgname, gid=None):
        if gid is not None:
            if imgname is None:
                imgname = 'gid_{}'.format(str(gid))
            dmet._imgname_to_gid[imgname] = gid
        else:
            if imgname is None:
                raise ValueError('must specify imgname or gid')
            try:
                gid = dmet._imgname_to_gid[imgname]
            except KeyError:
                gid = len(dmet._imgname_to_gid) + 1
                dmet._imgname_to_gid[imgname] = gid
        return gid

    def add_predictions(dmet, pred_dets, imgname=None, gid=None):
        """
        Register/Add predicted detections for an image

        Args:
            pred_dets (kwimage.Detections): predicted detections
            imgname (str | None): a unique string to identify the image
            gid (int | None): the integer image id if known
        """
        gid = dmet._register_imagename(imgname, gid)
        dmet.gid_to_pred_dets[gid] = pred_dets

    def add_truth(dmet, true_dets, imgname=None, gid=None):
        """
        Register/Add groundtruth detections for an image

        Args:
            true_dets (kwimage.Detections): groundtruth
            imgname (str | None): a unique string to identify the image
            gid (int | None): the integer image id if known
        """
        gid = dmet._register_imagename(imgname, gid)
        dmet.gid_to_true_dets[gid] = true_dets

    def true_detections(dmet, gid):
        """ gets Detections representation for groundtruth in an image """
        return dmet.gid_to_true_dets[gid]

    def pred_detections(dmet, gid):
        """ gets Detections representation for predictions in an image """
        return dmet.gid_to_pred_dets[gid]

    @property
    def classes(dmet):
        if dmet._classes is not None:
            return dmet._classes
        # If the detection metrics object doest have a top-level class
        # list, then try to extract one from the ground truth.
        # Try to grab classes from the truth if they exist
        for dets in dmet.gid_to_true_dets.values():
            if dets.classes is not None:
                import kwcoco
                classes = kwcoco.CategoryTree.coerce(dets.classes)
                return classes

    @classes.setter
    def classes(dmet, classes):
        import kwcoco
        if classes is not None:
            classes = kwcoco.CategoryTree.coerce(classes)
        dmet._classes = classes

    def confusion_vectors(dmet, iou_thresh=0.5, bias=0, gids=None, compat='mutex',
                          prioritize='iou', ignore_classes='ignore',
                          background_class=ub.NoParam, verbose='auto',
                          workers=0, track_probs='try', max_dets=None):
        """
        Assigns predicted boxes to the true boxes so we can transform the
        detection problem into a classification problem for scoring.

        Args:

            iou_thresh (float | List[float]):
                bounding box overlap iou threshold required for assignment
                if a list, then return type is a dict. Defaults to 0.5

            bias (float):
                for computing bounding box overlap, either 1 or 0
                Defaults to 0.

            gids (List[int] | None):
                which subset of images ids to compute confusion metrics on. If
                not specified all images are used. Defaults to None.

            compat (str):
                can be ('ancestors' | 'mutex' | 'all').  determines which pred
                boxes are allowed to match which true boxes. If 'mutex', then
                pred boxes can only match true boxes of the same class. If
                'ancestors', then pred boxes can match true boxes that match or
                have a coarser label. If 'all', then any pred can match any
                true, regardless of its category label.  Defaults to all.

            prioritize (str):
                can be ('iou' | 'class' | 'correct') determines which box to
                assign to if mutiple true boxes overlap a predicted box.  if
                prioritize is iou, then the true box with maximum iou (above
                iou_thresh) will be chosen.  If prioritize is class, then it will
                prefer matching a compatible class above a higher iou. If
                prioritize is correct, then ancestors of the true class are
                preferred over descendents of the true class, over unreleated
                classes. Default to 'iou'

            ignore_classes (set | str):
                class names indicating ignore regions. Default={'ignore'}

            background_class (str | NoParamType):
                Name of the background class. If unspecified we try to
                determine it with heuristics. A value of None means there is no
                background class.

            verbose (int | str): verbosity flag. Default to 'auto'. In auto mode,
                verbose=1 if len(gids) > 1000.

            workers (int):
                number of parallel assignment processes. Defaults to 0

            track_probs (str):
                can be 'try', 'force', or False.  if truthy, we assume
                probabilities for multiple classes are available. default='try'

        Returns:
            ConfusionVectors | Dict[float, ConfusionVectors]

        Example:
            >>> dmet = DetectionMetrics.demo(nimgs=30, classes=3,
            >>>                              nboxes=10, n_fp=3, box_noise=10,
            >>>                              with_probs=False)
            >>> iou_to_cfsn = dmet.confusion_vectors(iou_thresh=[0.3, 0.5, 0.9])
            >>> for t, cfsn in iou_to_cfsn.items():
            >>>     print('t = {!r}'.format(t))
            ...     print(cfsn.binarize_ovr().measures())
            ...     print(cfsn.binarize_classless().measures())

        Ignore:
            globals().update(xdev.get_func_kwargs(dmet.confusion_vectors))
        """
        import kwarray
        _tracking_probs = bool(track_probs)
        iou_thresh_list = [iou_thresh] if not ub.iterable(iou_thresh) else iou_thresh

        iou_to_yaccum = {
            t: ub.ddict(list)
            for t in iou_thresh_list
        }

        if _tracking_probs:
            iou_to_probaccum = {
                t: []
                for t in iou_thresh_list
            }

        if gids is None:
            gids = sorted(dmet._imgname_to_gid.values())

        if verbose == 'auto':
            verbose = 1 if len(gids) > 10 else 0

        classes = dmet.classes

        if background_class is ub.NoParam:
            # Try to autodetermine background class name,
            # otherwise fallback to None
            background_class = None
            if classes is not None:
                lower_classes = [c.lower() for c in classes]
                try:
                    idx = lower_classes.index('background')
                    background_class = classes[idx]
                    # TODO: if we know the background class name should we
                    # change bg_cidx in assignment?
                except ValueError:
                    pass

        jobs = ub.JobPool(mode='process', max_workers=workers)
        for gid in ub.ProgIter(gids, desc='submit assign jobs',
                               verbose=verbose):
            true_dets = dmet.true_detections(gid)
            pred_dets = dmet.pred_detections(gid)
            job = jobs.submit(
                _assign_confusion_vectors, true_dets, pred_dets,
                bg_weight=1, iou_thresh=iou_thresh_list, bg_cidx=-1, bias=bias,
                classes=classes, compat=compat, prioritize=prioritize,
                ignore_classes=ignore_classes, max_dets=max_dets)
            job.gid = gid

        for job in ub.ProgIter(jobs.jobs, desc='assign detections',
                               verbose=verbose):
            iou_thresh_to_y = job.result()
            gid = job.gid

            for t, y in iou_thresh_to_y.items():
                y_accum = iou_to_yaccum[t]

                if _tracking_probs:
                    prob_accum = iou_to_probaccum[t]
                    # Keep track of per-class probs
                    pred_dets = dmet.pred_detections(gid)
                    try:
                        pred_probs = pred_dets.probs
                        if pred_probs is None:
                            raise KeyError
                    except KeyError:
                        _tracking_probs = False
                        if track_probs == 'force':
                            raise Exception('unable to track probs')
                        elif track_probs == 'try':
                            pass
                        else:
                            raise KeyError(track_probs)
                    else:
                        pxs = np.array(y['pxs'], dtype=int)

                        # For unassigned truths, we need to create dummy probs
                        # where a background class has probability 1.
                        flags = pxs > -1
                        probs = np.zeros((len(pxs), pred_probs.shape[1]),
                                         dtype=np.float32)
                        if background_class is not None:
                            bg_idx = classes.index(background_class)
                            probs[:, bg_idx] = 1
                        probs[flags] = pred_probs[pxs[flags]]
                        prob_accum.append(probs)

                y['gid'] = [gid] * len(y['pred'])
                for k, v in y.items():
                    y_accum[k].extend(v)

        iou_to_cfsn = {}

        for t, y_accum in iou_to_yaccum.items():
            _data = {}
            for k, v in ub.ProgIter(list(y_accum.items()), desc='ndarray convert', verbose=verbose):
                # Try to use 32 bit types for large evaluation problems
                kw = dict()
                if k in {'iou', 'score', 'weight'}:
                    kw['dtype'] = np.float32
                if k in {'pxs', 'txs', 'gid', 'pred', 'true'}:
                    kw['dtype'] = np.int32
                try:
                    _data[k] = np.asarray(v, **kw)
                except TypeError:
                    _data[k] = np.asarray(v)

            # Avoid pandas when possible
            cfsn_data = kwarray.DataFrameArray(_data)

            if 0:
                import xdev
                nbytes = 0
                for k, v in _data.items():
                    nbytes += v.size * v.dtype.itemsize
                print(xdev.byte_str(nbytes))

            if _tracking_probs:
                prob_accum = iou_to_probaccum[t]
                y_prob = np.vstack(prob_accum)
            else:
                y_prob = None
            cfsn_vecs = ConfusionVectors(cfsn_data, classes=classes,
                                         probs=y_prob)
            iou_to_cfsn[t] = cfsn_vecs

        if ub.iterable(iou_thresh):
            return iou_to_cfsn
        else:
            cfsn_vecs = iou_to_cfsn[t]
            return cfsn_vecs

    def score_kwant(dmet, iou_thresh=0.5):
        """
        Scores the detections using kwant
        """
        try:
            from kwil.misc import kwant
            if not kwant.is_available():
                raise ImportError
        except ImportError:
            raise RuntimeError('kwant is not available')

        import kw18
        gids = list(dmet.gid_to_true_dets.keys())
        true_kw18s = []
        pred_kw18s = []
        for gid in ub.ProgIter(gids, desc='convert to kw18'):
            true_dets = dmet.gid_to_true_dets[gid]
            pred_dets = dmet.gid_to_pred_dets[gid]

            if len(true_dets) == 0:
                print('foo')
            if len(pred_dets) == 0:
                # kwant breaks on 0 predictions, hack in a bad prediction
                import kwimage
                hack_ = kwimage.Detections.random(1)
                hack_.scores[:] = 0
                pred_dets = hack_

            true_kw18 = kw18.make_kw18_from_detections(true_dets,
                                                       frame_number=gid,
                                                       timestamp=gid)
            pred_kw18 = kw18.make_kw18_from_detections(pred_dets,
                                                       frame_number=gid,
                                                       timestamp=gid)
            true_kw18s.append(true_kw18)
            pred_kw18s.append(pred_kw18)

        true_kw18 = true_kw18s
        pred_kw18 = pred_kw18s

        roc_info = kwant.score_events(true_kw18s, pred_kw18s,
                                      iou_thresh=iou_thresh, prefiltered=True,
                                      verbose=3)

        fp = roc_info['fp'].values
        tp = roc_info['tp'].values

        ppv = tp / (tp + fp)
        ppv[np.isnan(ppv)] = 1

        tpr = roc_info['pd'].values
        fpr = fp / fp[0]
        import sklearn
        roc_auc = sklearn.metrics.auc(fpr, tpr)

        from kwcoco.metrics.functional import _average_precision
        ap = _average_precision(tpr, ppv)

        roc_info['fpr'] = fpr
        roc_info['ppv'] = ppv

        info = {
            'roc_info': roc_info,
            'ap': ap,
            'roc_auc': roc_auc,
        }

        if False:
            import kwil
            kwil.autompl()
            kwil.multi_plot(roc_info['fa'], roc_info['pd'],
                            xlabel='fa (fp count)',
                            ylabel='pd (tpr)', fnum=1,
                            title='kwant roc_auc={:.4f}'.format(roc_auc))

            kwil.multi_plot(tpr, ppv,
                            xlabel='recall (fpr)',
                            ylabel='precision (tpr)',
                            fnum=2,
                            title='kwant ap={:.4f}'.format(ap))

        return info

    def score_kwcoco(dmet, iou_thresh=0.5, bias=0, gids=None,
                      compat='all', prioritize='iou'):
        """ our scoring method """
        cfsn_vecs = dmet.confusion_vectors(iou_thresh=iou_thresh, bias=bias,
                                           gids=gids,
                                           compat=compat,
                                           prioritize=prioritize)
        info = {}
        try:
            cfsn_perclass = cfsn_vecs.binarize_ovr(mode=1)
            perclass = cfsn_perclass.measures()
        except Exception as ex:
            print('warning: ex = {!r}'.format(ex))
        else:
            info['perclass'] = perclass['perclass']
            info['mAP'] = perclass['mAP']
        return info

    def score_voc(dmet, iou_thresh=0.5, bias=1, method='voc2012', gids=None,
                  ignore_classes='ignore'):
        """
        score using voc method

        Example:
            >>> dmet = DetectionMetrics.demo(
            >>>     nimgs=100, nboxes=(0, 3), n_fp=(0, 1), classes=8,
            >>>     score_noise=.5)
            >>> print(dmet.score_voc()['mAP'])
            0.9399...
        """
        # from . import voc_metrics
        from kwcoco.metrics.assignment import _filter_ignore_regions
        from kwcoco.metrics import voc_metrics
        if gids is None:
            gids = sorted(dmet._imgname_to_gid.values())
        # Convert true/pred detections into VOC format
        classes = dmet.classes
        vmet = voc_metrics.VOC_Metrics(classes=classes)
        for gid in gids:
            true_dets = dmet.true_detections(gid)
            pred_dets = dmet.pred_detections(gid)

            if ignore_classes is not None:
                true_ignore_flags, pred_ignore_flags = _filter_ignore_regions(
                    true_dets, pred_dets, ioaa_thresh=iou_thresh,
                    ignore_classes=ignore_classes)
                true_dets = true_dets.compress(~true_ignore_flags)
                pred_dets = pred_dets.compress(~pred_ignore_flags)

            vmet.add_truth(true_dets, gid=gid)
            vmet.add_predictions(pred_dets, gid=gid)
        voc_scores = vmet.score(iou_thresh, bias=bias, method=method)
        return voc_scores

    def _to_coco(dmet):
        """
        Convert to a coco representation of truth and predictions

        with inverse aid mappings
        """
        import kwcoco
        true = kwcoco.CocoDataset()
        pred = kwcoco.CocoDataset()

        gt_aid_to_tx = {}
        dt_aid_to_px = {}
        classes = dmet.classes

        for node in classes:
            # cid = classes.graph.node[node]['id']
            cid = classes.index(node)
            supercategory = list(classes.graph.pred[node])
            if len(supercategory) == 0:
                supercategory = None
            else:
                assert len(supercategory) == 1
                supercategory = supercategory[0]
            true.add_category(node, id=cid, supercategory=supercategory)
            pred.add_category(node, id=cid, supercategory=supercategory)

        for imgname, gid in dmet._imgname_to_gid.items():
            true.add_image(imgname, id=gid)
            pred.add_image(imgname, id=gid)

        idx_to_id = {
            idx: classes.index(node)
            for idx, node in enumerate(classes.idx_to_node)
        }

        for gid, pred_dets in dmet.gid_to_pred_dets.items():
            pred_boxes = pred_dets.boxes
            if 'scores' in pred_dets.data:
                pred_scores = pred_dets.scores
            else:
                pred_scores = np.ones(len(pred_dets))
            pred_cids = list(ub.take(idx_to_id, pred_dets.class_idxs))
            pred_xywh = pred_boxes.to_xywh().data.tolist()

            dt_aids = []
            for bbox, cid, score in zip(pred_xywh, pred_cids, pred_scores):
                aid = pred.add_annotation(gid, cid, bbox=bbox, score=score)
                dt_aids.append(aid)

            dt_aid_to_px.update(dict(zip(dt_aids, range(len(dt_aids)))))

        for gid, true_dets in dmet.gid_to_true_dets.items():
            true_boxes = true_dets.boxes
            if 'weights' in true_dets.data:
                true_weights = true_dets.weights
            else:
                true_weights = np.ones(len(true_boxes))
            true_cids = list(ub.take(idx_to_id, true_dets.class_idxs))
            true_xywh = true_boxes.to_xywh().data.tolist()

            gt_aids = []
            for bbox, cid, weight in zip(true_xywh, true_cids, true_weights):
                aid = true.add_annotation(gid, cid, bbox=bbox, weight=weight)
                gt_aids.append(aid)

            gt_aid_to_tx.update(dict(zip(gt_aids, range(len(gt_aids)))))

        return pred, true, gt_aid_to_tx, dt_aid_to_px

    def score_pycocotools(dmet, with_evaler=False, with_confusion=False,
                          verbose=0, iou_thresholds=None):
        """
        score using ms-coco method

        Returns:
            Dict : dictionary with pct info

        Example:
            >>> # xdoctest: +REQUIRES(module:pycocotools)
            >>> from kwcoco.metrics.detect_metrics import *
            >>> dmet = DetectionMetrics.demo(
            >>>     nimgs=10, nboxes=(0, 3), n_fn=(0, 1), n_fp=(0, 1), classes=8, with_probs=False)
            >>> pct_info = dmet.score_pycocotools(verbose=1,
            >>>                                   with_evaler=True,
            >>>                                   with_confusion=True,
            >>>                                   iou_thresholds=[0.5, 0.9])
            >>> evaler = pct_info['evaler']
            >>> iou_to_cfsn_vecs = pct_info['iou_to_cfsn_vecs']
            >>> for iou_thresh in iou_to_cfsn_vecs.keys():
            >>>     print('iou_thresh = {!r}'.format(iou_thresh))
            >>>     cfsn_vecs = iou_to_cfsn_vecs[iou_thresh]
            >>>     ovr_measures = cfsn_vecs.binarize_ovr().measures()
            >>>     print('ovr_measures = {}'.format(ub.urepr(ovr_measures, nl=1, precision=4)))

        Note:
            by default pycocotools computes average precision as the literal
            average of computed precisions at 101 uniformly spaced recall
            thresholds.

            pycocoutils seems to only allow predictions with the same category
            as the truth to match those truth objects. This should be the
            same as calling dmet.confusion_vectors with compat = mutex

            pycocoutils does not take into account the fact that each box often
            has a score for each category.

            pycocoutils will be incorrect if any annotation has an id of 0

            a major difference in the way kwcoco scores versus pycocoutils is
            the calculation of AP. The assignment between truth and predicted
            detections produces similar enough results. Given our confusion
            vectors we use the scikit-learn definition of AP, whereas
            pycocoutils seems to compute precision and recall --- more or less
            correctly --- but then it resamples the precision at various
            specified recall thresholds (in the `accumulate` function,
            specifically how `pr` is resampled into the `q` array). This
            can lead to a large difference in reported scores.

            pycocoutils also smooths out the precision such that it is
            monotonic decreasing, which might not be the best idea.

            pycocotools area ranges are inclusive on both ends, that means the
            "small" and "medium" truth selections do overlap somewhat.

        """
        from pycocotools import coco  # NOQA
        from pycocotools import cocoeval
        from kwcoco.util.util_monkey import SupressPrint

        pred, true, gt_aid_to_tx, dt_aid_to_px = dmet._to_coco()

        # The original pycoco-api prints to much, supress it
        quiet = verbose == 0
        with SupressPrint(coco, cocoeval, enabled=quiet):
            cocoGt = true._aspycoco()
            cocoDt = pred._aspycoco()

            for ann in cocoGt.dataset['annotations']:
                w, h = ann['bbox'][-2:]
                ann['ignore'] = ann['weight'] < .5
                ann['area'] = w * h
                ann['iscrowd'] = False

            for ann in cocoDt.dataset['annotations']:
                w, h = ann['bbox'][-2:]
                ann['area'] = w * h

            evaler = cocoeval.COCOeval(cocoGt, cocoDt, iouType='bbox')

            modified_params = False
            if iou_thresholds is None:
                iou_thresholds = evaler.params.iouThrs
            else:
                iou_thresholds = np.array(iou_thresholds)

                if len(iou_thresholds) != len(evaler.params.iouThrs) or np.allclose(iou_thresholds, evaler.params.iouThrs):
                    evaler.params.iouThrs = iou_thresholds
                    modified_params = True

            print('evaler.params.iouThrs = {!r}'.format(evaler.params.iouThrs))

            evaler.evaluate()
            evaler.accumulate()

            # if 0:
            #     # Get curves at a specific pycocoutils param
            #     Tx = np.where(evaler.params.iouThrs == 0.5)[0]
            #     Rx = slice(0, len(evaler.params.recThrs))
            #     Kx = slice(0, len(evaler.params.catIds))
            #     Ax = evaler.params.areaRng.index([0, 10000000000.0])
            #     Mx = evaler.params.maxDets.index(100)
            #     perclass_prec = evaler.eval['precision'][Tx, Rx, Kx, Ax, Mx]
            #     perclass_rec = evaler.eval['recall'][Tx, Kx, Ax, Mx]
            #     perclass_score = evaler.eval['scores'][Tx, Rx, Kx, Ax, Mx]

            pct_info = {}

            if modified_params:
                print('modified params')
                stats = pct_summarize2(evaler)
                evaler.stats = stats
            else:
                print('standard pycocotools params')
                evaler.summarize()
                coco_ap = evaler.stats[1]
                pct_info['mAP'] = coco_ap

            pct_info['evalar_stats'] = evaler.stats

        # 'mAP': coco_ap,
        if with_evaler:
            pct_info['evaler'] = evaler

        if with_confusion:
            iou_to_cfsn_vecs = {}

            for iou_thresh in iou_thresholds:
                cfsn_vecs = pycocotools_confusion_vectors(
                    dmet, evaler, iou_thresh=iou_thresh)
                cfsn_vecs.data['pxs'] = np.array(list(ub.take(dt_aid_to_px, cfsn_vecs.data['dt_aid'], default=-1)))
                cfsn_vecs.data['txs'] = np.array(list(ub.take(gt_aid_to_tx, cfsn_vecs.data['gt_aid'], default=-1)))
                iou_to_cfsn_vecs[iou_thresh] = cfsn_vecs

            pct_info['iou_to_cfsn_vecs'] = iou_to_cfsn_vecs

        return pct_info

    score_coco = score_pycocotools

    @classmethod
    def demo(cls, **kwargs):
        """
        Creates random true boxes and predicted boxes that have some noisy
        offset from the truth.

        Kwargs:
            classes (int):
                class list or the number of foreground classes.
                Defaults to 1.

            nimgs (int): number of images in the coco datasts. Defaults to 1.

            nboxes (int): boxes per image. Defaults to 1.

            n_fp (int): number of false positives. Defaults to 0.

            n_fn (int):
                number of false negatives. Defaults to 0.

            box_noise (float):
                std of a normal distribution used to perterb both box location
                and box size. Defaults to 0.

            cls_noise (float):
                probability that a class label will change. Must be within 0
                and 1. Defaults to 0.

            anchors (ndarray):
                used to create random boxes. Defaults to None.

            null_pred (bool):
                if True, predicted classes are returned as null, which means
                only localization scoring is suitable. Defaults to 0.

            with_probs (bool):
                if True, includes per-class probabilities with predictions
                Defaults to 1.

            rng (int | None | RandomState): random seed / state

        CommandLine:
            xdoctest -m kwcoco.metrics.detect_metrics DetectionMetrics.demo:2 --show

        Example:
            >>> kwargs = {}
            >>> # Seed the RNG
            >>> kwargs['rng'] = 0
            >>> # Size parameters determine how big the data is
            >>> kwargs['nimgs'] = 5
            >>> kwargs['nboxes'] = 7
            >>> kwargs['classes'] = 11
            >>> # Noise parameters perterb predictions further from the truth
            >>> kwargs['n_fp'] = 3
            >>> kwargs['box_noise'] = 0.1
            >>> kwargs['cls_noise'] = 0.5
            >>> dmet = DetectionMetrics.demo(**kwargs)
            >>> print('dmet.classes = {}'.format(dmet.classes))
            dmet.classes = <CategoryTree(nNodes=12, maxDepth=3, maxBreadth=4...)>
            >>> # Can grab kwimage.Detection object for any image
            >>> print(dmet.true_detections(gid=0))
            <Detections(4)>
            >>> print(dmet.pred_detections(gid=0))
            <Detections(7)>

        Example:
            >>> # Test case with null predicted categories
            >>> dmet = DetectionMetrics.demo(nimgs=30, null_pred=1, classes=3,
            >>>                              nboxes=10, n_fp=3, box_noise=0.1,
            >>>                              with_probs=False)
            >>> dmet.gid_to_pred_dets[0].data
            >>> dmet.gid_to_true_dets[0].data
            >>> cfsn_vecs = dmet.confusion_vectors()
            >>> binvecs_ovr = cfsn_vecs.binarize_ovr()
            >>> binvecs_per = cfsn_vecs.binarize_classless()
            >>> measures_per = binvecs_per.measures()
            >>> measures_ovr = binvecs_ovr.measures()
            >>> print('measures_per = {!r}'.format(measures_per))
            >>> print('measures_ovr = {!r}'.format(measures_ovr))
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> measures_ovr['perclass'].draw(key='pr', fnum=2)

        Example:
            >>> from kwcoco.metrics.confusion_vectors import *  # NOQA
            >>> from kwcoco.metrics.detect_metrics import DetectionMetrics
            >>> dmet = DetectionMetrics.demo(
            >>>     n_fp=(0, 1), n_fn=(0, 1), nimgs=32, nboxes=(0, 16),
            >>>     classes=3, rng=0, newstyle=1, box_noise=0.5, cls_noise=0.0, score_noise=0.3, with_probs=False)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> summary = dmet.summarize(plot=True, title='DetectionMetrics summary demo', with_ovr=True, with_bin=False)
            >>> summary['bin_measures']
            >>> kwplot.show_if_requested()
        """
        import kwimage
        import kwarray
        import kwcoco
        # Parse kwargs
        rng = kwarray.ensure_rng(kwargs.get('rng', 0))

        # todo: accept and coerce classes instead of classes
        classes = kwargs.get('classes', None)

        if classes is None:
            classes = 1

        nimgs = kwargs.get('nimgs', 1)
        box_noise = kwargs.get('box_noise', 0)
        cls_noise = kwargs.get('cls_noise', 0)

        null_pred = kwargs.get('null_pred', False)
        with_probs = kwargs.get('with_probs', True)

        # specify an amount of overlap between true and false scores
        score_noise = kwargs.get('score_noise', 0.2)

        anchors = kwargs.get('anchors', None)
        scale = 100.0

        # TODO: make newstyle False
        newstyle = kwargs.get('newstyle', False)

        if newstyle:
            perterbkw = ub.dict_isect(kwargs, {
                'rng': 0,
                'box_noise': 0,
                'cls_noise': 0,
                'null_pred': False,
                'with_probs': False,
                'score_noise': 0.2,
                'n_fp': 0,
                'n_fn': 0,
                'hacked': 1})

            # TODO: use kwcoco.demo.perterb instead of rolling the logic here
            from kwcoco.demo import perterb
            # TODO: don't do any rendering
            # true_dset = kwcoco.CocoDataset.random()  # TODO
            true_dset = kwcoco.CocoDataset.demo('shapes{}'.format(nimgs))  # FIXME
            # true_dset = kwcoco.CocoDataset.demo(
            #     'vidshapes', num_frames=1, num_videos=nimgs, render=False)
            pred_dset = perterb.perterb_coco(true_dset, **perterbkw)
            dmet = cls.from_coco(true_dset, pred_dset)
        else:
            # Unfortunately this is not ready for deprecation, the above case
            # does not handle everything yet.
            # ub.schedule_deprecation(
            #     'kwcoco', 'newstyle=False', 'kwarg to DetectionMetrics.demo',
            #     migration='adapt to newstyle=True instead',
            #     deprecate='0.6.1', error='0.7.0', remove='0.7.1'
            # )

            # Build random variables
            from kwarray import distributions
            DiscreteUniform = distributions.DiscreteUniform.seeded(rng=rng)
            def _parse_arg(key, default):
                value = kwargs.get(key, default)
                try:
                    low, high = value
                    return (low, high + 1)
                except Exception:
                    return (0, value + 1)
            nboxes_RV = DiscreteUniform(*_parse_arg('nboxes', 1))
            n_fp_RV = DiscreteUniform(*_parse_arg('n_fp', 0))
            n_fn_RV = DiscreteUniform(*_parse_arg('n_fn', 0))

            box_noise_RV = distributions.Normal(0, box_noise, rng=rng)
            cls_noise_RV = distributions.Bernoulli(cls_noise, rng=rng)

            # the values of true and false scores starts off with no overlap and
            # the overlap increases as the score noise increases.
            def _interp(v1, v2, alpha):
                return v1 * alpha + (1 - alpha) * v2
            mid = 0.5
            # true_high = 2.0
            true_high = 1.0
            true_low   = _interp(0, mid, score_noise)
            false_high = _interp(true_high, mid - 1e-3, score_noise)
            true_mean  = _interp(0.5, .8, score_noise)
            false_mean = _interp(0.5, .2, score_noise)

            true_score_RV = distributions.TruncNormal(
                mean=true_mean, std=.5, low=true_low, high=true_high, rng=rng)
            false_score_RV = distributions.TruncNormal(
                mean=false_mean, std=.5, low=0, high=false_high, rng=rng)

            # Create the category hierarcy
            if isinstance(classes, int):
                graph = nx.DiGraph()
                graph.add_node('background', id=0)
                for cid in range(1, classes + 1):
                    # binary heap encoding of a tree
                    cx = cid - 1
                    parent_cx = (cx - 1) // 2
                    node = 'cat_{}'.format(cid)
                    graph.add_node(node, id=cid)
                    if parent_cx > 0:
                        supercategory = 'cat_{}'.format(parent_cx + 1)
                        graph.add_edge(supercategory, node)
                classes = kwcoco.CategoryTree(graph)
                frgnd_cx_RV = distributions.DiscreteUniform(1, len(classes), rng=rng)
            else:
                classes = kwcoco.CategoryTree.coerce(classes)
                # TODO: remove background classes via rejection sampling
                frgnd_cx_RV = distributions.DiscreteUniform(0, len(classes), rng=rng)

            dmet = cls()
            dmet.classes = classes

            for gid in range(nimgs):

                # Sample random variables
                nboxes_ = nboxes_RV()
                n_fp_ = n_fp_RV()
                n_fn_ = n_fn_RV()

                imgname = 'img_{}'.format(gid)
                dmet._register_imagename(imgname, gid)

                # Generate random ground truth detections
                true_boxes = kwimage.Boxes.random(num=nboxes_, scale=scale,
                                                  anchors=anchors, rng=rng,
                                                  format='cxywh')
                # Prevent 0 sized boxes: increase w/h by 1
                true_boxes.data[..., 2:4] += 1
                true_cxs = frgnd_cx_RV(len(true_boxes))
                true_weights = np.ones(len(true_boxes), dtype=np.int32)

                # Initialize predicted detections as a copy of truth
                pred_boxes = true_boxes.copy()
                pred_cxs = true_cxs.copy()

                # Perterb box coordinates
                pred_boxes.data = np.abs(pred_boxes.data.astype(float) +
                                         box_noise_RV())

                # Perterb class predictions
                change = cls_noise_RV(len(pred_cxs))
                pred_cxs_swap = frgnd_cx_RV(len(pred_cxs))
                pred_cxs[change] = pred_cxs_swap[change]

                # Drop true positive boxes
                if n_fn_:
                    pred_boxes.data = pred_boxes.data[n_fn_:]
                    pred_cxs = pred_cxs[n_fn_:]

                # pred_scores = np.linspace(true_min, true_max, len(pred_boxes))[::-1]
                n_tp_ = len(pred_boxes)
                pred_scores = true_score_RV(n_tp_)

                # Add false positive boxes
                if n_fp_:
                    false_boxes = kwimage.Boxes.random(num=n_fp_, scale=scale,
                                                       rng=rng, format='cxywh')
                    false_cxs = frgnd_cx_RV(n_fp_)
                    false_scores = false_score_RV(n_fp_)

                    pred_boxes.data = np.vstack([pred_boxes.data, false_boxes.data])
                    pred_cxs = np.hstack([pred_cxs, false_cxs])
                    pred_scores = np.hstack([pred_scores, false_scores])

                # Transform the scores for the assigned class into a predicted
                # probability for each class. (Currently a bit hacky).
                class_probs = _demo_construct_probs(
                    pred_cxs, pred_scores, classes, rng,
                    hacked=kwargs.get('hacked', 1))

                true_dets = kwimage.Detections(boxes=true_boxes,
                                               class_idxs=true_cxs,
                                               weights=true_weights)

                pred_dets = kwimage.Detections(boxes=pred_boxes,
                                               class_idxs=pred_cxs,
                                               scores=pred_scores)

                # Hack in the probs
                if with_probs:
                    pred_dets.data['probs'] = class_probs

                if null_pred:
                    pred_dets.data['class_idxs'] = np.array(
                        [None] * len(pred_dets), dtype=object)

                dmet.add_truth(true_dets, imgname=imgname)
                dmet.add_predictions(pred_dets, imgname=imgname)

        return dmet

    def summarize(dmet, out_dpath=None, plot=False, title='', with_bin='auto', with_ovr='auto'):
        """
        Example:
            >>> from kwcoco.metrics.confusion_vectors import *  # NOQA
            >>> from kwcoco.metrics.detect_metrics import DetectionMetrics
            >>> dmet = DetectionMetrics.demo(
            >>>     n_fp=(0, 128), n_fn=(0, 4), nimgs=512, nboxes=(0, 32),
            >>>     classes=3, rng=0)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> dmet.summarize(plot=True, title='DetectionMetrics summary demo')
            >>> kwplot.show_if_requested()
        """
        cfsn_vecs = dmet.confusion_vectors()

        summary = {}
        if with_ovr:
            ovr_cfsn = cfsn_vecs.binarize_ovr(keyby='name')
            ovr_measures = ovr_cfsn.measures()
            summary['ovr_measures'] = ovr_measures
        if with_bin:
            bin_cfsn = cfsn_vecs.binarize_classless()
            bin_measures = bin_cfsn.measures()
            summary['bin_measures'] = bin_measures
        if plot:
            print('summary = {}'.format(ub.urepr(summary, nl=1)))
            print('out_dpath = {!r}'.format(out_dpath))

            if with_bin:
                bin_measures.summary_plot(title=title, fnum=1, subplots=with_bin)

            if with_ovr:
                perclass = ovr_measures['perclass']
                perclass.summary_plot(title=title, fnum=2, subplots=with_ovr)
            # # Is this micro-versus-macro average?
            # bin_measures['ap']
            # bin_measures['auc']
        return summary


def _demo_construct_probs(pred_cxs, pred_scores, classes, rng, hacked=1):
    """
    Constructs random probabilities for demo data
    """
    # Setup probs such that the assigned class receives a probability
    # equal-(ish) to the assigned score.
    # Its a bit tricky to setup hierarchical probs such that we get the
    # scores in the right place. We punt and just make probs
    # conditional. The right thing to do would be to do this, and then
    # perterb ancestor categories such that the probability evenetually
    # converges on the right value at that specific classes depth.
    # import torch

    # Ensure probs
    pred_scores2 = pred_scores.clip(0, 1.0)

    class_energy = rng.rand(len(pred_scores2), len(classes)).astype(np.float32)
    for p, x, s in zip(class_energy, pred_cxs, pred_scores2):
        p[x] = s

    if hacked:
        # HACK! All that nice work we did is too slow for doctests
        return class_energy

    # class_energy = torch.Tensor(class_energy)
    # cond_logprobs = classes.conditional_log_softmax(class_energy, dim=1)
    # cond_probs = torch.exp(cond_logprobs).numpy()

    # # I was having a difficult time getting this right, so an
    # # inefficient per-item non-vectorized implementation it is.
    # # Note: that this implementation takes 70% of the time in this function
    # # and is a bottleneck for the doctests. A vectorized implementation would
    # # be nice.
    # idx_to_ancestor_idxs = classes.idx_to_ancestor_idxs()
    # idx_to_groups = {idx: group for group in classes.idx_groups for idx in group}

    # def set_conditional_score(row, cx, score, idx_to_groups):
    #     group_cxs = np.array(idx_to_groups[cx])
    #     flags = group_cxs == cx
    #     group_row = row[group_cxs]
    #     # Ensure that that heriarchical probs sum to 1
    #     current = group_row[~flags]
    #     other = current * (1 - score) / current.sum()
    #     other = np.nan_to_num(other)
    #     group_row[~flags] = other
    #     group_row[flags] = score
    #     row[group_cxs] = group_row

    # for row, cx, score in zip(cond_probs, pred_cxs, pred_scores2):
    #     set_conditional_score(row, cx, score, idx_to_groups)
    #     for ancestor_cx in idx_to_ancestor_idxs[cx]:
    #         if ancestor_cx != cx:
    #             # Hack all parent probs to 1.0 so conditional probs
    #             # turn into real probs.
    #             set_conditional_score(row, ancestor_cx, 1.0, idx_to_groups)
    #             # TODO: could add a fudge factor here so the
    #             # conditional prob is higher than score, but parent
    #             # probs are less than 1.0

    #             # TODO: could also maximize entropy of descendant nodes
    #             # so classes.decision2 would stop at this node

    # # For each level the conditional probs must sum to 1
    # if cond_probs.size > 0:
    #     for idxs in classes.idx_groups:
    #         level = cond_probs[:, idxs]
    #         totals = level.sum(axis=1)
    #         assert level.shape[1] == 1 or np.allclose(totals, 1.0), str(level) + ' : ' + str(totals)

    # cond_logprobs = torch.Tensor(cond_probs).log()
    # class_probs = classes._apply_logprob_chain_rule(cond_logprobs, dim=1).exp().numpy()
    # class_probs = class_probs.reshape(-1, len(classes))
    # # print([p[x] for p, x in zip(class_probs, pred_cxs)])
    # # print(pred_scores2)
    # return class_probs


def pycocotools_confusion_vectors(dmet, evaler, iou_thresh=0.5, verbose=0):
    """
    Example:
        >>> # xdoctest: +REQUIRES(module:pycocotools)
        >>> from kwcoco.metrics.detect_metrics import *
        >>> dmet = DetectionMetrics.demo(
        >>>     nimgs=10, nboxes=(0, 3), n_fn=(0, 1), n_fp=(0, 1), classes=8, with_probs=False)
        >>> coco_scores = dmet.score_pycocotools(with_evaler=True)
        >>> evaler = coco_scores['evaler']
        >>> cfsn_vecs = pycocotools_confusion_vectors(dmet, evaler, verbose=1)

    """
    import numpy as np
    import kwarray
    import ubelt as ub
    import copy

    target_areaRng = [0, 10000000000.0]
    target_iou = iou_thresh
    target_maxDet = 100

    # Get curves at a specific pycocoutils param
    Tx = np.where(evaler.params.iouThrs == target_iou)[0]
    # Rx = slice(0, len(evaler.params.recThrs))
    # Kx = slice(0, len(evaler.params.catIds))
    # Ax = evaler.params.areaRng.index(target_areaRng)
    # Mx = evaler.params.maxDets.index(target_maxDet)

    # prints match info for each category
    y_accum = ub.ddict(list)

    for info in evaler.evalImgs:
        if info is None:
            continue
        info = copy.deepcopy(info)
        if info['aRng'] == target_areaRng and info['maxDet'] == target_maxDet:
            info['dtMatches'] = info['dtMatches'][Tx]
            info['gtMatches'] = info['gtMatches'][Tx]
            info['dtIgnore'] = info['dtIgnore'][Tx]

            # Transform pycocotools assignments into a confusion vector
            y = dict(
                pred=[],
                true=[],
                score=[],
                weight=[],
                iou=[],
                gt_aid=[],
                dt_aid=[],
                gid=[],
            )
            cidx = info['category_id']
            gid = info['image_id']

            # cidx = dmet.classes.id_to_idx[cid]
            for dtid, gtid, score, ignore in zip(info['dtIds'], info['dtMatches'][0], info['dtScores'], info['dtIgnore'][0]):
                if gtid == 0:
                    y['pred'].append(cidx)
                    y['true'].append(-1)
                    y['score'].append(score)
                    y['weight'].append(1 - float(ignore))
                    y['iou'].append(-1)

                    # TODO: can we find the indexes instead?
                    # To better match dmet? Or are aids better?

                    y['gt_aid'].append(-1)
                    y['dt_aid'].append(dtid)
                    y['gid'].append(info['image_id'])
                else:
                    y['pred'].append(cidx)
                    y['true'].append(cidx)
                    y['score'].append(score)
                    y['weight'].append(1 - float(ignore))
                    y['iou'].append(np.nan)  # TODO, we should be able to find this
                    y['gt_aid'].append(gtid)
                    y['dt_aid'].append(dtid)
                    y['gid'].append(gid)

            flags = ~kwarray.isect_flags(info['gtMatches'][0], info['dtIds'])
            gt_aids = list(ub.compress(info['gtIds'], flags))
            gtignores = info['gtIgnore'][flags]

            for gtid, ignore in zip(gt_aids, gtignores):
                y['pred'].append(-1)
                y['true'].append(cidx)
                y['score'].append(-np.inf)
                y['weight'].append(1 - float(ignore))
                y['iou'].append(-1)
                y['gt_aid'].append(gtid)
                y['dt_aid'].append(-1)
                y['gid'].append(gid)
            # y
            # y = kwarray.DataFrameArray.from_pandas(y.pandas())
            for k, v in y.items():
                y_accum[k].extend(v)

        _data = {}
        for k, v in ub.ProgIter(list(y_accum.items()), desc='ndarray convert', verbose=verbose):
            # Try to use 32 bit types for large evaluation problems
            kw = dict()
            if k in {'iou', 'score', 'weight'}:
                kw['dtype'] = np.float32
            if k in {'pxs', 'txs', 'gid', 'pred', 'true'}:
                kw['dtype'] = np.int32
            try:
                _data[k] = np.asarray(v, **kw)
            except TypeError:
                _data[k] = np.asarray(v)

        # Avoid pandas when possible
        cfsn_data = kwarray.DataFrameArray(_data)

        # if _tracking_probs:
        #     y_prob = np.vstack(prob_accum)
        # else:
        y_prob = None
        cfsn_vecs = ConfusionVectors(
            cfsn_data, classes=dmet.classes, probs=y_prob)
    return cfsn_vecs


def eval_detections_cli(**kw):
    """
    DEPRECATED USE `kwcoco eval` instead

    CommandLine:
        xdoctest -m ~/code/kwcoco/kwcoco/metrics/detect_metrics.py eval_detections_cli
    """
    import scriptconfig as scfg
    import kwcoco

    ub.schedule_deprecation(
        'kwcoco', name='kwcoco.metrics.detect_metrics.eval_detections_cli',
        type='method',
        deprecate='0.3.4', error='1.0.0', remove='1.1.0',
        migration=(
            'Use `kwcoco eval` in kwcoco.cli.coco_eval instead. '
        )
    )

    class EvalDetectionCLI(scfg.Config):
        default = {
            'true': scfg.Path(None, help='true coco dataset'),
            'pred': scfg.Path(None, help='predicted coco dataset'),
            'out_dpath': scfg.Path('./out', help='output directory')
        }
        pass

    config = EvalDetectionCLI()
    cmdline = kw.pop('cmdline', True)
    config.load(kw, cmdline=cmdline)

    true_coco = kwcoco.CocoDataset(config['true'])
    pred_coco = kwcoco.CocoDataset(config['pred'])

    from kwcoco.metrics.detect_metrics import DetectionMetrics
    dmet = DetectionMetrics.from_coco(true_coco, pred_coco)

    voc_info = dmet.score_voc()

    cls_info = voc_info['perclass'][0]
    tp = cls_info['tp']
    fp = cls_info['fp']
    fn = cls_info['fn']

    tpr = cls_info['tpr']
    ppv = cls_info['ppv']
    fp = cls_info['fp']

    # Compute the MCC as TN->inf
    thresh = cls_info['thresholds']

    # https://erotemic.wordpress.com/2019/10/23/closed-form-of-the-mcc-when-tn-inf/
    mcc_lim = tp / (np.sqrt(fn + tp) * np.sqrt(fp + tp))
    f1 = 2 * (ppv * tpr) / (ppv + tpr)

    draw = False
    if draw:

        mcc_idx = mcc_lim.argmax()
        f1_idx = f1.argmax()

        import kwplot
        plt = kwplot.autoplt()

        kwplot.multi_plot(
            xdata=thresh,
            ydata=mcc_lim,
            xlabel='threshold',
            ylabel='mcc*',
            fnum=1, pnum=(1, 4, 1),
            title='MCC*',
            color=['blue'],
        )
        plt.plot(thresh[mcc_idx], mcc_lim[mcc_idx], 'r*', markersize=20)
        plt.plot(thresh[f1_idx], mcc_lim[f1_idx], 'k*', markersize=20)

        kwplot.multi_plot(
            xdata=fp,
            ydata=tpr,
            xlabel='fp (fa)',
            ylabel='tpr (pd)',
            fnum=1, pnum=(1, 4, 2),
            title='ROC',
            color=['blue'],
        )
        plt.plot(fp[mcc_idx], tpr[mcc_idx], 'r*', markersize=20)
        plt.plot(fp[f1_idx], tpr[f1_idx], 'k*', markersize=20)

        kwplot.multi_plot(
            xdata=tpr,
            ydata=ppv,
            xlabel='tpr (recall)',
            ylabel='ppv (precision)',
            fnum=1, pnum=(1, 4, 3),
            title='PR',
            color=['blue'],
        )
        plt.plot(tpr[mcc_idx], ppv[mcc_idx], 'r*', markersize=20)
        plt.plot(tpr[f1_idx], ppv[f1_idx], 'k*', markersize=20)

        kwplot.multi_plot(
            xdata=thresh,
            ydata=f1,
            xlabel='threshold',
            ylabel='f1',
            fnum=1, pnum=(1, 4, 4),
            title='F1',
            color=['blue'],
        )
        plt.plot(thresh[mcc_idx], f1[mcc_idx], 'r*', markersize=20)
        plt.plot(thresh[f1_idx], f1[f1_idx], 'k*', markersize=20)


def _summarize(self, ap=1, iouThr=None, areaRngLbl='all', maxDets=100):
    import numpy as np
    p = self.params
    iStr = '{:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'  # noqa: E501
    titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
    typeStr = '(AP)' if ap == 1 else '(AR)'
    iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
        if iouThr is None else '{:0.2f}'.format(iouThr)

    aind = [
        i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRngLbl
    ]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    if ap == 1:
        # dimension of precision: [TxRxKxAxM]
        s = self.eval['precision']
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            if len(t):
                Tx = t[0]
                Ax = aind[0]
                Mx = mind[0]
                pct_perclass_ap = s[Tx, :, :, Ax, Mx].mean(axis=0)
                catnames = [self.cocoGt.cats[cid]['name'] for cid in self.params.catIds]
                catname_to_ap = ub.dzip(catnames, pct_perclass_ap)
                pct_map = pct_perclass_ap.mean()
                print('catname_to_ap = {}'.format(ub.urepr(catname_to_ap, nl=1, precision=2)))
                # print('pct_perclass_ap = {}'.format(ub.urepr(pct_perclass_ap.tolist(), nl=1, precision=2)))
                print('pct_map = {}'.format(ub.urepr(pct_map.tolist(), nl=0, precision=2)))
            else:
                raise Exception('not known iou')
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, :, aind, mind]

    else:
        # dimension of recall: [TxKxAxM]
        s = self.eval['recall']
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, aind, mind]
    if len(s[s > -1]) == 0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s > -1])
    print(
        iStr.format(titleStr, typeStr, iouStr, areaRngLbl, maxDets,
                    mean_s))
    return mean_s


def pct_summarize2(self):
    stats = []
    for ap in [1, 0]:
        for areaRngLbl in self.params.areaRngLbl:
            stats.append(_summarize(self, ap=ap, iouThr=None, areaRngLbl=areaRngLbl))
            if areaRngLbl == 'all':
                if len(self.params.iouThrs) > 1:
                    for iouThr in self.params.iouThrs:
                        stats.append(_summarize(self, ap=ap, iouThr=iouThr,
                                                areaRngLbl=areaRngLbl))
    return stats


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/kwcoco/metrics/detect_metrics.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
