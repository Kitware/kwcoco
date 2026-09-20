"""
TODO:
    - [ ] Does this rely on the image ids being the same? In either case document it.
"""

from __future__ import annotations

import numpy as np
import ubelt as ub
import networkx as nx
from kwcoco.metrics.confusion_vectors import ConfusionVectors
from kwcoco.metrics.assignment import _assign_confusion_vectors, ASSIGN_COLUMNS
# from .assignment import _assign_confusion_vectors

import typing
from collections.abc import Iterable

if typing.TYPE_CHECKING:
    from typing import Dict, List
    import kwimage
    import kwcoco
    from ubelt.util_const import NoParamType

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

    gid_to_true_dets: Dict[int, kwimage.Detections]
    gid_to_pred_dets: Dict[int, kwimage.Detections]
    gid_to_image_dims: dict[int, tuple[int, int]]
    _imgname_to_gid: dict[str, int]

    def __init__(
        dmet,
        classes: (
            kwcoco.CategoryTree
            | list[str]
            | list[dict[str, typing.Any]]
            | int
            | nx.DiGraph
            | None
        ) = None,
    ) -> None:
        dmet.classes = classes
        dmet.gid_to_true_dets = {}
        dmet.gid_to_pred_dets = {}
        dmet.gid_to_image_dims = {}
        dmet._imgname_to_gid = {}

    def clear(dmet) -> None:
        dmet.gid_to_true_dets = {}
        dmet.gid_to_pred_dets = {}
        dmet.gid_to_image_dims = {}
        dmet._imgname_to_gid = {}

    def __nice__(dmet) -> str:
        info = {
            'n_true_imgs': len(dmet.gid_to_true_dets),
            'n_pred_imgs': len(dmet.gid_to_pred_dets),
            'n_true_anns': sum(map(len, dmet.gid_to_true_dets.values())),
            'n_pred_anns': sum(map(len, dmet.gid_to_pred_dets.values())),
            'classes': dmet.classes,
        }
        return ub.urepr(info)

    def enrich_confusion_vectors(
        dmet, cfsn_vecs: ConfusionVectors
    ) -> None:
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
        cfsn_pred_aids: np.ndarray = np.full(
            len(all_gids), fill_value=-1, dtype=int
        )
        cfsn_true_aids: np.ndarray = np.full(
            len(all_gids), fill_value=-1, dtype=int
        )

        # For each image
        unique_gids, grouped_indexes = kwarray.group_indices(all_gids)
        for gid, rowxs in zip(unique_gids, grouped_indexes):
            gid = typing.cast(int, gid)
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
    def from_coco(
        cls,
        true_coco: kwcoco.CocoDataset,
        pred_coco: kwcoco.CocoDataset,
        gids: Iterable[int] | None = None,
        verbose: int = 0,
    ) -> 'DetectionMetrics':
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
        self = cls(classes)

        if gids is None:
            gids = sorted(set(true_coco.imgs.keys()) & set(pred_coco.imgs.keys()))

        def _coco_to_dets(coco_dset, desc=''):
            import kwimage

            for gid in ub.ProgIter(gids, desc=desc, verbose=verbose):
                img = coco_dset.imgs[gid]
                imgname = img['file_name']
                height = img.get('height', None)
                width = img.get('width', None)
                img_dims = None
                if height is not None and width is not None:
                    img_dims = (int(height), int(width))
                aids = list(coco_dset.gid_to_aids[gid])
                annots = [coco_dset.anns[aid] for aid in aids]
                # dets = true_coco.annots(gid=gid).detections
                dets = kwimage.Detections.from_coco_annots(
                    annots, dset=coco_dset, classes=classes
                )
                # ``from_coco_annots`` historically normalizes segmentations
                # to polygons.  Keep the source segmentation backend here so
                # official mask evaluation does not polygonize exact RLE input.
                dets.data['segmentations'] = kwimage.SegmentationList.coerce(
                    [a.get('segmentation', None) for a in annots]
                )
                # Keep COCO-only metadata needed by the official evaluator.
                # Arbitrary Detections data columns survive indexing/warping.
                dets.data['aids'] = np.asarray(aids, dtype=int)
                dets.data['iscrowd'] = np.asarray(
                    [bool(a.get('iscrowd', False)) for a in annots],
                    dtype=bool,
                )
                dets.data['ignore'] = np.asarray(
                    [bool(a.get('ignore', False)) for a in annots],
                    dtype=bool,
                )
                yield dets, imgname, gid, img_dims

        for dets, imgname, gid, img_dims in _coco_to_dets(
            true_coco, desc='add truth'
        ):
            self.add_truth(dets, imgname, gid=gid, img_dims=img_dims)

        for dets, imgname, gid, _img_dims in _coco_to_dets(
            pred_coco, desc='add pred'
        ):
            # Truth image dimensions are authoritative.  Do not introduce a
            # new bbox-scoring failure mode if prediction metadata disagrees.
            self.add_predictions(dets, imgname, gid=gid)

        return self

    def _register_imagename(
        dmet, imgname: str | None, gid: int | None = None
    ) -> int:
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

    def add_predictions(
        dmet,
        pred_dets: kwimage.Detections,
        imgname: str | None = None,
        gid: int | None = None,
        img_dims: tuple[int, int] | None = None,
    ) -> None:
        """
        Register/Add predicted detections for an image

        Args:
            pred_dets (kwimage.Detections):
                Predicted detections. These should contain boxes, class_idxs.
                They can optionally include scores.

            imgname (str | None): a unique string to identify the image

            gid (int | None): the integer image id if known

            img_dims (Tuple[int, int] | None):
                ``(height, width)`` if known. Required for official or native
                COCO-compatible instance-segmentation scoring when masks must
                be represented in full-image coordinates.
        """
        gid = dmet._register_imagename(imgname, gid)
        dmet._register_image_dims(gid, img_dims)
        dmet.gid_to_pred_dets[gid] = pred_dets

    def add_truth(
        dmet,
        true_dets: kwimage.Detections,
        imgname: str | None = None,
        gid: int | None = None,
        img_dims: tuple[int, int] | None = None,
    ) -> None:
        """
        Register/Add groundtruth detections for an image

        Args:
            true_dets (kwimage.Detections): groundtruth detections.
            These should contain boxes, class_idxs. They can optionally include
            weights.

            imgname (str | None): a unique string to identify the image

            gid (int | None): the integer image id if known

            img_dims (Tuple[int, int] | None):
                ``(height, width)`` if known. Required for official or native
                COCO-compatible instance-segmentation scoring when masks must
                be represented in full-image coordinates.
        """
        gid = dmet._register_imagename(imgname, gid)
        dmet._register_image_dims(gid, img_dims)
        dmet.gid_to_true_dets[gid] = true_dets

    def _register_image_dims(
        dmet, gid: int, img_dims: tuple[int, int] | None
    ) -> None:
        if img_dims is None:
            return
        img_dims = (int(img_dims[0]), int(img_dims[1]))
        if min(img_dims) <= 0:
            raise ValueError(f'invalid image dimensions: {img_dims!r}')
        old_dims = dmet.gid_to_image_dims.get(gid, None)
        if old_dims is not None and old_dims != img_dims:
            raise ValueError(
                f'conflicting dimensions for gid={gid}: '
                f'{old_dims!r} != {img_dims!r}'
            )
        dmet.gid_to_image_dims[gid] = img_dims

    def true_detections(dmet, gid: int) -> kwimage.Detections:
        """gets Detections representation for groundtruth in an image"""
        return dmet.gid_to_true_dets[gid]

    def pred_detections(dmet, gid: int) -> kwimage.Detections:
        """gets Detections representation for predictions in an image"""
        return dmet.gid_to_pred_dets[gid]

    @property
    def classes(dmet) -> kwcoco.CategoryTree | None:
        if dmet._classes is not None:
            return dmet._classes
        # If the detection metrics object does not have a top-level class
        # list, then try to extract one from the ground truth.
        # Try to grab classes from the truth if they exist
        for dets in dmet.gid_to_true_dets.values():
            if dets.classes is not None:
                import kwcoco

                classes = kwcoco.CategoryTree.coerce(dets.classes)
                return classes

    @classes.setter
    def classes(
        dmet,
        classes: (
            kwcoco.CategoryTree
            | list[str]
            | list[dict[str, typing.Any]]
            | int
            | nx.DiGraph
            | None
        ),
    ) -> None:
        import kwcoco

        if classes is not None:
            classes = kwcoco.CategoryTree.coerce(classes)
        dmet._classes = classes

    @typing.overload
    def confusion_vectors(
        dmet,
        iou_thresh: float = 0.5,
        bias: float = 0,
        gids: Iterable[int] | None = None,
        compat: str = 'mutex',
        prioritize: str = 'iou',
        ignore_classes: str | Iterable[str] | None = 'ignore',
        background_class: str | None | NoParamType = ub.NoParam,
        verbose: int | typing.Literal['auto'] = 'auto',
        workers: int | str = 0,
        track_probs: str | bool = 'try',
        max_dets: int | None = None,
        truth_reuse_policy: bool = False,
    ) -> ConfusionVectors: ...

    @typing.overload
    def confusion_vectors(
        dmet,
        iou_thresh: Iterable[float],
        bias: float = 0,
        gids: Iterable[int] | None = None,
        compat: str = 'mutex',
        prioritize: str = 'iou',
        ignore_classes: str | Iterable[str] | None = 'ignore',
        background_class: str | None | NoParamType = ub.NoParam,
        verbose: int | typing.Literal['auto'] = 'auto',
        workers: int | str = 0,
        track_probs: str | bool = 'try',
        max_dets: int | None = None,
        truth_reuse_policy: bool = False,
    ) -> dict[float, ConfusionVectors]: ...

    def confusion_vectors(
        dmet,
        iou_thresh: float | Iterable[float] = 0.5,
        bias: float = 0,
        gids: Iterable[int] | None = None,
        compat: str = 'mutex',
        prioritize: str = 'iou',
        ignore_classes: str | Iterable[str] | None = 'ignore',
        background_class: str | None | NoParamType = ub.NoParam,
        verbose: int | typing.Literal['auto'] = 'auto',
        workers: int | str = 0,
        track_probs: str | bool = 'try',
        max_dets: int | None = None,
        truth_reuse_policy: bool = False,
    ) -> ConfusionVectors | dict[float, ConfusionVectors]:
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
                assign to if multiple true boxes overlap a predicted box.  if
                prioritize is iou, then the true box with maximum iou (above
                iou_thresh) will be chosen.  If prioritize is class, then it will
                prefer matching a compatible class above a higher iou. If
                prioritize is correct, then ancestors of the true class are
                preferred over descendents of the true class, over unrelated
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

            truth_reuse_policy (bool):
                EXPERIMENTAL

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

        Example:
            >>> from kwcoco.metrics.detect_metrics import *  # NOQA
            >>> dmet = DetectionMetrics.demo(nimgs=30, classes=3,
            >>>                              nboxes=10, n_fp=3, box_noise=10,
            >>>                              with_probs=True, rng=0)
            >>> iou_thresh = 0.3
            >>> # When matching boxes, we can specify compat='all' to allow
            >>> # predicted boxes of one class to match true boxes of a
            >>> # different class. Setting compat='mutex' forces hard category
            >>> # decisions.
            >>> # In this case you should also set track_probs='force' so
            >>> # subsequent analysis can give credit for cases where the
            >>> # correct answer has a high, but not top probability
            >>> # Also, setting prioritize=class instead of the default
            >>> # prioritize=iou will choose compatible classes instead of
            >>> # higher overlaps, as long as the iou is still above a threshold
            >>> iou_to_cfsn = dmet.confusion_vectors(iou_thresh=[iou_thresh],
            >>>                                      compat='all',
            >>>                                      prioritize='class',
            >>>                                      track_probs='force')
            >>> cfsn = iou_to_cfsn[iou_thresh]  # NOQA
            >>> # Print the confusion assignment
            >>> print(cfsn.data.pandas())
            >>> # Doing a OVR binarization will score each class such that
            >>> # a box gets some credit if its second highest score is the
            >>> # correct class.
            >>> print(cfsn.binarize_ovr().measures())

        Ignore:
            globals().update(xdev.get_func_kwargs(dmet.confusion_vectors))
        """
        import kwarray

        _tracking_probs = bool(track_probs)
        if isinstance(iou_thresh, Iterable):
            iou_thresh_list = [float(t) for t in iou_thresh]
            multi_iou = True
        else:
            iou_thresh_list = [float(iou_thresh)]
            multi_iou = False

        iou_to_yaccum = {t: ub.ddict(list) for t in iou_thresh_list}

        if _tracking_probs:
            iou_to_probaccum = {t: [] for t in iou_thresh_list}

        if gids is None:
            gids = sorted(dmet._imgname_to_gid.values())
        else:
            gids = list(gids)

        if verbose == 'auto':
            verbose = 1 if len(gids) > 10 else 0

        classes = dmet.classes

        if background_class is ub.NoParam:
            # Try to autodetermine background class name,
            # otherwise fallback to None
            background_class = None
            if classes is not None:
                lower_classes = [str(c).lower() for c in classes]
                try:
                    idx = lower_classes.index('background')
                    background_class = str(classes[idx])
                    # TODO: if we know the background class name should we
                    # change bg_cidx in assignment?
                except ValueError:
                    pass

        from kwcoco.util.util_parallel import coerce_num_workers

        resolved_workers = coerce_num_workers(workers)
        if resolved_workers is None:
            resolved_workers = 0
        jobs = ub.JobPool(mode='process', max_workers=resolved_workers)
        for gid in ub.ProgIter(gids, desc='submit assign jobs', verbose=verbose):
            true_dets = dmet.true_detections(gid)
            pred_dets = dmet.pred_detections(gid)
            job = jobs.submit(
                _assign_confusion_vectors,
                true_dets,
                pred_dets,
                bg_weight=1,
                iou_thresh=iou_thresh_list,
                bg_cidx=-1,
                bias=bias,
                classes=classes,
                compat=compat,
                prioritize=prioritize,
                ignore_classes=ignore_classes,
                max_dets=max_dets,
                truth_reuse_policy=truth_reuse_policy,
            )
            job.gid = gid  # type: ignore

        for job in ub.ProgIter(jobs.jobs, desc='assign detections', verbose=verbose):
            iou_thresh_to_y = job.result()
            gid = job.gid  # type: ignore

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
                        probs = np.zeros(
                            (len(pxs), pred_probs.shape[1]), dtype=np.float32
                        )
                        if background_class is not None:
                            if classes is None:
                                raise ValueError(
                                    'classes are required when background_class is specified'
                                )
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
            for k, v in ub.ProgIter(
                list(y_accum.items()), desc='ndarray convert', verbose=verbose
            ):
                # Try to use 32 bit types for large evaluation problems
                dtype = None
                if k in {'iou', 'score', 'weight'}:
                    dtype = np.float32
                if k in {'pxs', 'txs', 'gid', 'pred', 'true'}:
                    dtype = np.int32
                try:
                    _data[k] = np.asarray(v, dtype=dtype)
                except TypeError:
                    _data[k] = np.asarray(v)

            # Avoid pandas when possible
            if len(_data) == 0:
                # Handle empty input case - note: dtypes here are not correct
                # but it works well enough.
                _data = {k: np.empty((0,)) for k in ['gid'] + ASSIGN_COLUMNS}
            cfsn_data = kwarray.DataFrameArray(_data)

            if 0:
                import xdev

                nbytes = 0
                for k, v in _data.items():
                    nbytes += v.size * v.dtype.itemsize
                print(xdev.byte_str(nbytes))

            if _tracking_probs:
                prob_accum = iou_to_probaccum[t]
                try:
                    y_prob = np.vstack(prob_accum)
                except ValueError:
                    y_prob = None
            else:
                y_prob = None
            cfsn_vecs = ConfusionVectors(cfsn_data, classes=classes, probs=y_prob)
            iou_to_cfsn[t] = cfsn_vecs

        if multi_iou:
            return iou_to_cfsn
        else:
            cfsn_vecs = iou_to_cfsn[iou_thresh_list[0]]
            return cfsn_vecs

    def confusion_vectors_v2(
        dmet,
        iou_thresh: float | Iterable[float] = 0.5,
        *,
        iou_type: str = 'bbox',
        overlap_backend: str = 'auto',
        segmentation_rasterizer: str = 'auto',
        raster_backend: str = 'auto',
        threshold_policy: str = 'auto',
        annotation_ignore_policy: str = 'coco',
        gids: Iterable[int] | None = None,
        compat: str = 'mutex',
        prioritize: str = 'iou',
        ignore_classes: str | Iterable[str] | None = None,
        max_dets: int | None = None,
        max_dets_scope: str = 'image',
        area_range: tuple[float, float] | None = None,
        truth_reuse_policy: str = 'never',
        min_iou: float | None = None,
        spatial_threshold: int | None = 50_000,
        bbox_overlap_mode: str = 'legacy',
    ) -> ConfusionVectors | dict[float, ConfusionVectors]:
        """Experimental reusable-candidate confusion-vector path.

        This is intentionally separate from :meth:`confusion_vectors` while
        ``assignment_v2`` semantics are still being measured.  In addition to
        bbox IoU it supports experimental native instance-mask assignment via
        ``iou_type='segm'``.  Segmentation mode defaults to COCO-compatible
        polygon rasterization and inclusive IoU thresholds, implements COCO
        crowd reuse/ignore weighting, and keeps kwcoco's class-based ignore
        regions as a separate explicit mechanism. ``area_range`` applies
        inclusive assignment-time area semantics. ``max_dets_scope`` defaults
        to the original image-wide v2 cap; native COCO-shaped scoring uses the
        COCO category-scoped variant explicitly.
        """
        import kwarray
        from kwcoco.metrics.assignment_v2 import AssignmentCandidateIndex

        if isinstance(iou_thresh, Iterable):
            thresholds = [float(t) for t in iou_thresh]
            multi = True
        else:
            thresholds = [float(iou_thresh)]
            multi = False
        if min_iou is None:
            min_iou = min(thresholds) if thresholds else 0.0

        if gids is None:
            gids = sorted(dmet._imgname_to_gid.values())
        else:
            gids = list(gids)

        iou_to_yaccum = {t: ub.ddict(list) for t in thresholds}
        classes = dmet.classes
        for gid in gids:
            true_dets = dmet.true_detections(gid)
            pred_dets = dmet.pred_detections(gid)
            image_dims = dmet.gid_to_image_dims.get(gid, None)
            index = AssignmentCandidateIndex.build(
                true_dets,
                pred_dets,
                classes=classes,
                compat=compat,
                prioritize=prioritize,
                min_iou=float(min_iou),
                spatial_threshold=spatial_threshold,
                iou_type=iou_type,
                bbox_overlap_mode=bbox_overlap_mode,
                overlap_backend=overlap_backend,
                segmentation_rasterizer=segmentation_rasterizer,
                raster_backend=raster_backend,
                threshold_policy=threshold_policy,
                annotation_ignore_policy=annotation_ignore_policy,
                image_dims=image_dims,
            )
            iou_to_y = index.assign_many(
                thresholds,
                bg_weight=1.0,
                bg_cidx=-1,
                ignore_classes=ignore_classes,
                max_dets=max_dets,
                max_dets_scope=max_dets_scope,
                area_range=area_range,
                truth_reuse_policy=truth_reuse_policy,
            )
            for t, y in iou_to_y.items():
                y = dict(y)
                y['gid'] = [gid] * len(y['pred'])
                accum = iou_to_yaccum[t]
                for key, vals in y.items():
                    accum[key].extend(vals)

        iou_to_cfsn = {}
        for t, y_accum in iou_to_yaccum.items():
            data = {}
            for key, vals in y_accum.items():
                dtype = None
                if key in {'iou', 'score', 'weight'}:
                    dtype = np.float32
                elif key in {'pxs', 'txs', 'gid', 'pred', 'true', 'rank'}:
                    dtype = np.int32
                data[key] = np.asarray(vals, dtype=dtype)
            if not data:
                data = {k: np.empty((0,)) for k in ['gid'] + ASSIGN_COLUMNS}
            cfsn_data = kwarray.DataFrameArray(data)
            iou_to_cfsn[t] = ConfusionVectors(cfsn_data, classes=classes)

        if multi:
            return iou_to_cfsn
        return iou_to_cfsn[thresholds[0]]

    def score_native_coco(
        dmet,
        *,
        iou_type: str = 'segm',
        gids: Iterable[int] | None = None,
        overlap_backend: str = 'auto',
        raster_backend: str = 'auto',
        assignment_backend: str = 'auto',
        annotation_ignore_policy: str = 'coco',
        ignore_classes: str | Iterable[str] | None = None,
        iou_thresholds: Iterable[float] | None = None,
        recall_thresholds: Iterable[float] | None = None,
        area_ranges: typing.Mapping[str, Iterable[float]] | None = None,
        max_dets: Iterable[int] | None = None,
        category_idxs: Iterable[int] | None = None,
        with_details: bool = False,
        with_eval: bool = False,
    ) -> dict[str, typing.Any]:
        """Compute a native kwcoco COCO-compatible detection summary.

        The native path uses :mod:`assignment_v2` to build one reusable
        candidate graph per image, then reproduces COCO's area/maxDet matching
        semantics and 101-point accumulation. Normal scoring keeps assignment
        state compact through accumulation; explicit rows remain available for
        diagnostics and native extensions. It remains separate from the
        historical :meth:`CocoEvaluator.evaluate` path and never changes the
        default kwcoco bbox matcher.

        Args:
            iou_type:
                ``'segm'`` for instance masks or ``'bbox'`` for the explicit
                Phase-3 COCO bbox profile.  Bbox mode uses continuous-coordinate
                COCO IoU and crowd intersection-over-detection-area semantics.
            gids:
                Optional image-id subset. Repetitions are removed and ids are
                sorted to mirror COCOeval parameter normalization.
            overlap_backend, raster_backend:
                Optional mask backends used only by ``iou_type='segm'``.
            assignment_backend:
                ``'auto'``, ``'python'``, or ``'kwimage_ext'``. ``'auto'``
                uses the optional kwimage_ext area-by-threshold batched Rust
                greedy matcher when available and falls back to the compact
                Python reference kernel.
            annotation_ignore_policy:
                ``'coco'`` uses COCO crowd ignore semantics. ``'kwcoco'`` also
                honors explicit kwcoco truth ``ignore`` metadata.
            ignore_classes:
                Optional native kwcoco ignore-region extension. Official COCO
                comparison should leave this unset.
            iou_thresholds, recall_thresholds, area_ranges, max_dets:
                Optional COCO parameter overrides. Defaults are the standard
                detection settings. ``area_ranges`` is a mapping from stable
                labels to inclusive ``(min_area, max_area)`` pairs.
            category_idxs:
                Optional ordered subset of ``dmet.classes`` indexes to expose
                on the COCO category axis. Matching still uses the complete
                class space. This is primarily used by ``CocoEvaluator`` to
                omit its legacy synthetic background bookkeeping category.
            with_details:
                Include per-class scalar diagnostics for each computed slice.
            with_eval:
                Materialize the full COCO precision / recall / score tensors.
                When false, only the six slices needed by the standard 12-value
                summary are assigned, preserving the faster summary path.

        Returns:
            Dict:
                A versioned native record with the standard named summary,
                parameter/provenance metadata, timing, and optionally an
                ``eval`` tensor record compatible with COCOeval dimensions.

        Notes:
            COCO does not define kwcoco's arbitrary truth ``weight`` extension,
            so truth weights are normalized to one before native COCO matching.
            Crowd / area / configured ignore state is represented separately by
            assignment-time zero-weight rows.
        """
        import time
        from kwcoco.metrics.assignment_v2 import AssignmentCandidateIndex
        from kwcoco.metrics.assignment_v2 import resolve_assignment_backend
        from kwcoco.metrics.assignment_v2 import resolve_assignment_grid_backend
        from kwcoco.metrics.coco_metrics import COCO_AREA_RANGES
        from kwcoco.metrics.coco_metrics import COCO_DETECTION_STAT_KEYS
        from kwcoco.metrics.coco_metrics import COCO_EVAL_TENSOR_SCHEMA
        from kwcoco.metrics.coco_metrics import COCO_IOU_THRESHOLDS
        from kwcoco.metrics.coco_metrics import COCO_MAX_DETS
        from kwcoco.metrics.coco_metrics import COCO_RECALL_THRESHOLDS
        from kwcoco.metrics.coco_metrics import NATIVE_COCO_IMPLEMENTATION
        from kwcoco.metrics.coco_metrics import COCO_DETECTION_RESULT_SCHEMA
        from kwcoco.metrics.coco_metrics import OFFICIAL_COCO_IMPLEMENTATION
        from kwcoco.metrics.coco_metrics import accumulate_coco_assignment_rows
        from kwcoco.metrics.coco_metrics import accumulate_coco_compact_grid
        from kwcoco.metrics.coco_metrics import summarize_coco_eval_tensor

        if iou_type not in {'bbox', 'segm'}:
            raise KeyError(f'Unsupported native COCO iou_type={iou_type!r}')

        requested_assignment_backend = assignment_backend
        # The normal COCO path can remain entirely in compact NumPy/Rust match
        # state. Diagnostic rows and threshold-dependent kwcoco ignore regions
        # retain the explicit-row reference implementation.
        use_compact_matches = ignore_classes is None and not with_details
        if use_compact_matches:
            resolver = resolve_assignment_grid_backend
        else:
            resolver = resolve_assignment_backend
        resolved_assignment_backend, _assign_func, assignment_backend_meta = (
            resolver(assignment_backend)
        )
        if ignore_classes is not None and resolved_assignment_backend == 'kwimage_ext':
            if assignment_backend == 'kwimage_ext':
                raise NotImplementedError(
                    'assignment_backend="kwimage_ext" does not yet support '
                    'threshold-dependent kwcoco ignore_classes'
                )
            resolved_assignment_backend = 'python'
            assignment_backend_meta = {
                'name': 'kwcoco.assignment_v2.python',
                'kind': 'python',
                'module': 'kwcoco.metrics.assignment_v2',
                'fallback_reason': 'ignore_classes',
            }

        if gids is None:
            gids = sorted(dmet._imgname_to_gid.values())
        else:
            # COCOeval unique-sorts params.imgIds before evaluation. Match
            # that behavior because image concatenation order is observable
            # when predictions from different images have equal scores.
            gids = sorted(set(map(int, gids)))

        if iou_thresholds is None:
            thresholds = [float(t) for t in COCO_IOU_THRESHOLDS]
        else:
            thresholds = [float(t) for t in iou_thresholds]
        if not thresholds:
            raise ValueError('iou_thresholds must contain at least one value')
        if not all(np.isfinite(t) and 0 <= t <= 1 for t in thresholds):
            raise ValueError(f'invalid iou_thresholds={thresholds!r}')
        if len(set(thresholds)) != len(thresholds):
            raise ValueError('iou_thresholds must be unique')
        # COCOeval caps each matching threshold just below one before the
        # overlap comparison. This is observable for numerically near-perfect
        # floating-point bbox overlaps when a caller explicitly requests 1.0.
        assignment_thresholds = {
            t: min(t, 1.0 - 1e-10) for t in thresholds
        }

        if recall_thresholds is None:
            recalls = [float(t) for t in COCO_RECALL_THRESHOLDS]
        else:
            recalls = [float(t) for t in recall_thresholds]
        if not recalls:
            raise ValueError('recall_thresholds must contain at least one value')
        if not all(np.isfinite(t) and 0 <= t <= 1 for t in recalls):
            raise ValueError(f'invalid recall_thresholds={recalls!r}')
        if any(b < a for a, b in zip(recalls, recalls[1:])):
            raise ValueError('recall_thresholds must be nondecreasing')

        if area_ranges is None:
            area_ranges_ = {
                key: tuple(map(float, value))
                for key, value in COCO_AREA_RANGES.items()
            }
        else:
            area_ranges_ = {}
            for label, value in area_ranges.items():
                pair = tuple(map(float, value))
                if len(pair) != 2:
                    raise ValueError(
                        f'area range {label!r} must have two values, got {pair!r}'
                    )
                lo, hi = pair
                if not np.isfinite(lo) or lo < 0 or np.isnan(hi) or hi < lo:
                    raise ValueError(f'invalid area range {label!r}: {pair!r}')
                area_ranges_[str(label)] = pair
        if not area_ranges_:
            raise ValueError('area_ranges must contain at least one range')

        if max_dets is None:
            max_dets_ = list(map(int, COCO_MAX_DETS))
        else:
            max_dets_ = sorted(set(map(int, max_dets)))
        if not max_dets_ or any(n < 0 for n in max_dets_):
            raise ValueError(f'invalid max_dets={max_dets_!r}')

        classes = dmet.classes
        if classes is None:
            raise ValueError('classes are required for native COCO scoring')
        if category_idxs is None:
            category_idxs_ = list(range(len(classes)))
        else:
            category_idxs_ = sorted(set(map(int, category_idxs)))
            invalid = [
                cx for cx in category_idxs_
                if cx < 0 or cx >= len(classes)
            ]
            if invalid:
                raise ValueError(
                    f'category_idxs contains out-of-range indexes: {invalid!r}'
                )
        class_names = [str(classes[cx]) for cx in category_idxs_]

        total_start = time.perf_counter()
        build_start = time.perf_counter()
        gid_to_index = {}
        candidate_predictions_before_cap = 0
        candidate_predictions_after_cap = 0
        largest_requested_max_det = max(max_dets_)
        for gid in gids:
            true_dets = dmet.true_detections(gid)
            pred_dets = dmet.pred_detections(gid)
            image_dims = dmet.gid_to_image_dims.get(gid, None)

            # COCOeval computes image/category overlaps only for detections in
            # the largest requested maxDet prefix. Smaller maxDet tensor slices
            # are accumulated from that same prefix. Prune before candidate
            # geometry so low-score detections beyond the largest requested cap
            # do not consume bbox/mask overlap work at all.
            candidate_predictions_before_cap += len(pred_dets)
            if len(pred_dets):
                pred_scores = np.asarray(
                    pred_dets.data.get('scores', np.ones(len(pred_dets))),
                    dtype=float,
                )
                pred_cxs = np.asarray(pred_dets.class_idxs, dtype=int)
                score_order = np.argsort(-pred_scores, kind='mergesort')
                category_counts = ub.ddict(int)
                keep_flags = np.zeros(len(pred_dets), dtype=bool)
                for px in score_order:
                    px = int(px)
                    cx = int(pred_cxs[px])
                    if category_counts[cx] < largest_requested_max_det:
                        keep_flags[px] = True
                        category_counts[cx] += 1
                if not np.all(keep_flags):
                    pred_dets = pred_dets.take(np.where(keep_flags)[0])
            candidate_predictions_after_cap += len(pred_dets)

            # COCOeval has no kwcoco truth-weight extension. Preserve crowd /
            # annotation / area ignore state in assignment, but normalize any
            # user-authored truth weights for this compatibility profile.
            true_weights = true_dets.data.get('weights', None)
            if true_weights is not None:
                true_weights = np.asarray(true_weights)
                if not np.all(true_weights == 1):
                    true_dets = true_dets.copy()
                    true_dets.data['weights'] = np.ones(
                        len(true_dets), dtype=np.float32
                    )

            common_min_iou = min(assignment_thresholds.values())
            if iou_type == 'segm':
                index = AssignmentCandidateIndex.build(
                    true_dets,
                    pred_dets,
                    classes=classes,
                    compat='mutex',
                    prioritize='iou',
                    min_iou=common_min_iou,
                    spatial_threshold=None,
                    iou_type='segm',
                    threshold_policy='inclusive',
                    annotation_ignore_policy=annotation_ignore_policy,
                    image_dims=image_dims,
                    overlap_backend=overlap_backend,
                    segmentation_rasterizer='coco',
                    raster_backend=raster_backend,
                )
            else:
                index = AssignmentCandidateIndex.build(
                    true_dets,
                    pred_dets,
                    classes=classes,
                    compat='mutex',
                    prioritize='iou',
                    min_iou=common_min_iou,
                    spatial_threshold=50_000,
                    iou_type='bbox',
                    bbox_overlap_mode='coco',
                    threshold_policy='inclusive',
                    annotation_ignore_policy=annotation_ignore_policy,
                    image_dims=image_dims,
                )
            gid_to_index[gid] = index
        candidate_build_seconds = time.perf_counter() - build_start

        if ignore_classes is None:
            measure_ignore_classes = set()
        elif isinstance(ignore_classes, str):
            measure_ignore_classes = {ignore_classes}
        else:
            measure_ignore_classes = set(ignore_classes)

        area_labels = list(area_ranges_)
        if with_eval:
            requested_slices = [
                (area_label, max_det)
                for area_label in area_labels
                for max_det in max_dets_
            ]
        else:
            # Compute only slices required by the standard summary. If custom
            # parameters omit a standard slice, it will correctly summarize -1.
            requested_slices = []
            for area_label, max_det in [
                ('all', 1), ('all', 10), ('all', 100),
                ('small', 100), ('medium', 100), ('large', 100),
            ]:
                if area_label in area_ranges_ and max_det in max_dets_:
                    requested_slices.append((area_label, max_det))
            if not requested_slices:
                requested_slices = [
                    (area_label, max_det)
                    for area_label in area_labels
                    for max_det in max_dets_
                ]

        slice_metrics = {}
        assignment_seconds = 0.0
        accumulation_seconds = 0.0
        assignment_calls = 0
        assignment_batches = 0

        # COCO matching is score ordered. For a fixed image/category/area/IoU
        # slice, matches made inside the smaller maxDet prefixes cannot be
        # changed by detections that occur later in the score order. Match at
        # the largest requested maxDet and derive the smaller tensor slices.
        area_to_max_dets = ub.ddict(list)
        for area_label, max_det in requested_slices:
            area_to_max_dets[area_label].append(max_det)
        for area_label in list(area_to_max_dets):
            area_to_max_dets[area_label] = sorted(set(map(
                int, area_to_max_dets[area_label]
            )))

        compact_tensor = None
        if use_compact_matches:
            # Fast COCO path: keep matcher output compact all the way into the
            # accumulator. The Rust backend handles every active area range x
            # IoU threshold in one call per image; the Python backend mirrors
            # the same compact contract for differential testing.
            T = len(thresholds)
            R = len(recalls)
            K = len(category_idxs_)
            A = len(area_labels)
            M = len(max_dets_)
            precision = -np.ones((T, R, K, A, M), dtype=float)
            recall = -np.ones((T, K, A, M), dtype=float)
            scores = -np.ones((T, R, K, A, M), dtype=float)
            realpos = np.zeros((T, K, A, M), dtype=int)

            active_area_labels = list(area_to_max_dets)
            active_area_ranges = [
                area_ranges_[label] for label in active_area_labels
            ]
            largest_active_max_det = max(
                max(values) for values in area_to_max_dets.values()
            )
            effective_thresholds = [
                assignment_thresholds[t] for t in thresholds
            ]
            compact_batches = []

            assign_start = time.perf_counter()
            for gid, index in gid_to_index.items():
                compact = index.match_coco_grid(
                    effective_thresholds,
                    active_area_ranges,
                    assignment_backend=resolved_assignment_backend,
                    max_dets=largest_active_max_det,
                    max_dets_scope='category',
                )
                assignment_calls += (
                    len(effective_thresholds) * len(active_area_labels)
                )
                assignment_batches += 1
                compact_batches.append(compact)
            assignment_seconds += time.perf_counter() - assign_start

            accum_start = time.perf_counter()
            active_max_dets = sorted({
                max_det
                for requested in area_to_max_dets.values()
                for max_det in requested
            })
            compact_info = accumulate_coco_compact_grid(
                compact_batches,
                classes,
                thresholds=thresholds,
                max_dets=active_max_dets,
                num_areas=len(active_area_labels),
                category_idxs=category_idxs_,
                recall_thresholds=recalls,
            )
            accumulation_seconds += time.perf_counter() - accum_start
            for local_ai, area_label in enumerate(active_area_labels):
                ai = area_labels.index(area_label)
                for max_det in area_to_max_dets[area_label]:
                    local_mi = active_max_dets.index(max_det)
                    mi = max_dets_.index(max_det)
                    precision[:, :, :, ai, mi] = (
                        compact_info['precision'][:, :, :, local_ai, local_mi]
                    )
                    recall[:, :, ai, mi] = (
                        compact_info['recall'][:, :, local_ai, local_mi]
                    )
                    scores[:, :, :, ai, mi] = (
                        compact_info['scores'][:, :, :, local_ai, local_mi]
                    )
                    realpos[:, :, ai, mi] = (
                        compact_info['realpos_total'][:, :, local_ai, local_mi]
                    )

            compact_tensor = {
                'precision': precision,
                'recall': recall,
                'scores': scores,
                'realpos_total': realpos,
            }
        else:
            # Diagnostic/native-extension path. Keep explicit assignment rows
            # available when callers request per-class details or threshold-
            # dependent kwcoco ignore regions.
            for area_label, requested_max_dets in area_to_max_dets.items():
                area_range = area_ranges_[area_label]
                largest_max_det = requested_max_dets[-1]
                maxdet_to_y_accum = {
                    max_det: {t: ub.ddict(list) for t in thresholds}
                    for max_det in requested_max_dets
                }

                assign_start = time.perf_counter()
                effective_thresholds = [
                    assignment_thresholds[t] for t in thresholds
                ]
                for gid, index in gid_to_index.items():
                    y_many = index.assign_many(
                        effective_thresholds,
                        assignment_backend=resolved_assignment_backend,
                        bg_weight=1.0,
                        bg_cidx=-1,
                        ignore_classes=ignore_classes,
                        max_dets=largest_max_det,
                        max_dets_scope='category',
                        area_range=area_range,
                        truth_reuse_policy='never',
                    )
                    assignment_calls += len(effective_thresholds)
                    assignment_batches += 1
                    for threshold in thresholds:
                        effective_threshold = assignment_thresholds[threshold]
                        y_largest = y_many[effective_threshold]
                        for max_det in requested_max_dets:
                            if max_det == largest_max_det:
                                y = y_largest
                            else:
                                y = index.truncate_max_dets(
                                    y_largest,
                                    max_det,
                                    max_dets_scope='category',
                                    bg_cidx=-1,
                                )
                            accum = maxdet_to_y_accum[max_det][threshold]
                            nrows = len(y['pred'])
                            accum['gid'].extend([gid] * nrows)
                            for key in ('pred', 'true', 'score', 'weight'):
                                accum[key].extend(y[key])
                assignment_seconds += time.perf_counter() - assign_start

                for max_det in requested_max_dets:
                    accum_start = time.perf_counter()
                    threshold_metrics = {}
                    y_accum = maxdet_to_y_accum[max_det]
                    for threshold in thresholds:
                        threshold_metrics[threshold] = (
                            accumulate_coco_assignment_rows(
                                y_accum[threshold],
                                classes,
                                category_idxs=category_idxs_,
                                ignore_classes=ignore_classes,
                                recall_thresholds=recalls,
                                with_details=with_details,
                            )
                        )
                    accumulation_seconds += time.perf_counter() - accum_start
                    slice_metrics[(area_label, max_det)] = threshold_metrics

        eval_record = None
        if use_compact_matches:
            assert compact_tensor is not None
            T = len(thresholds)
            R = len(recalls)
            K = len(category_idxs_)
            A = len(area_labels)
            M = len(max_dets_)
            compact_eval_record = {
                'schema': COCO_EVAL_TENSOR_SCHEMA,
                'counts': [T, R, K, A, M],
                'precision': compact_tensor['precision'],
                'recall': compact_tensor['recall'],
                'scores': compact_tensor['scores'],
                'realpos_total': compact_tensor['realpos_total'],
                'iou_thresholds': thresholds,
                'recall_thresholds': recalls,
                'category_ids': category_idxs_,
                'category_names': class_names,
                'image_ids': sorted(map(int, gid_to_index)),
                'area_ranges': [list(area_ranges_[k]) for k in area_labels],
                'area_range_labels': area_labels,
                'max_dets': max_dets_,
            }
            summary = summarize_coco_eval_tensor(compact_eval_record)
            if with_eval:
                eval_record = compact_eval_record
        elif with_eval:
            T = len(thresholds)
            R = len(recalls)
            K = len(category_idxs_)
            A = len(area_labels)
            M = len(max_dets_)
            precision = -np.ones((T, R, K, A, M), dtype=float)
            recall = -np.ones((T, K, A, M), dtype=float)
            scores = -np.ones((T, R, K, A, M), dtype=float)
            realpos = np.zeros((T, K, A, M), dtype=int)
            for ai, area_label in enumerate(area_labels):
                for mi, max_det in enumerate(max_dets_):
                    threshold_info = slice_metrics[(area_label, max_det)]
                    for ti, threshold in enumerate(thresholds):
                        info = threshold_info[threshold]
                        precision[ti, :, :, ai, mi] = info['precision']
                        recall[ti, :, ai, mi] = info['recall']
                        scores[ti, :, :, ai, mi] = info['scores']
                        realpos[ti, :, ai, mi] = info['realpos_total']
            eval_record = {
                'schema': COCO_EVAL_TENSOR_SCHEMA,
                'counts': [T, R, K, A, M],
                'precision': precision,
                'recall': recall,
                'scores': scores,
                'realpos_total': realpos,
                'iou_thresholds': thresholds,
                'recall_thresholds': recalls,
                'category_ids': category_idxs_,
                'category_names': class_names,
                'image_ids': sorted(map(int, gid_to_index)),
                'area_ranges': [list(area_ranges_[k]) for k in area_labels],
                'area_range_labels': area_labels,
                'max_dets': max_dets_,
            }
            summary = summarize_coco_eval_tensor(eval_record)
        else:
            def mean_metric(area_label, max_det, metric, selected=None):
                threshold_info = slice_metrics.get((area_label, max_det), None)
                if threshold_info is None:
                    return -1.0
                if selected is None:
                    values = [threshold_info[t][metric] for t in thresholds]
                else:
                    candidates = [t for t in thresholds if t == selected]
                    values = [] if not candidates else [
                        threshold_info[candidates[0]][metric]
                    ]
                values = [v for v in values if v >= 0 and np.isfinite(v)]
                return float(np.mean(values)) if values else -1.0

            summary = {
                'AP': mean_metric('all', 100, 'mAP'),
                'AP50': mean_metric('all', 100, 'mAP', 0.50),
                'AP75': mean_metric('all', 100, 'mAP', 0.75),
                'AP_small': mean_metric('small', 100, 'mAP'),
                'AP_medium': mean_metric('medium', 100, 'mAP'),
                'AP_large': mean_metric('large', 100, 'mAP'),
                'AR1': mean_metric('all', 1, 'mAR'),
                'AR10': mean_metric('all', 10, 'mAR'),
                'AR100': mean_metric('all', 100, 'mAR'),
                'AR_small': mean_metric('small', 100, 'mAR'),
                'AR_medium': mean_metric('medium', 100, 'mAR'),
                'AR_large': mean_metric('large', 100, 'mAR'),
            }

        stats = [summary[key] for key in COCO_DETECTION_STAT_KEYS]
        indexes = list(gid_to_index.values())
        overlap_names = sorted({
            idx.overlap_backend.get('name', 'unknown')
            for idx in indexes
            if idx.overlap_backend is not None
        })
        raster_names = sorted({
            idx.raster_backend.get('name', 'unknown')
            for idx in indexes
            if idx.raster_backend is not None
        })
        area_names = sorted({
            idx.area_backend.get('name', 'unknown')
            for idx in indexes
            if idx.area_backend is not None
        })

        total_seconds = time.perf_counter() - total_start
        result = {
            'backend': 'kwcoco',
            'implementation': NATIVE_COCO_IMPLEMENTATION,
            'result_schema': COCO_DETECTION_RESULT_SCHEMA,
            'official': False,
            'reference_implementation': OFFICIAL_COCO_IMPLEMENTATION,
            'iou_type': iou_type,
            'geometry': iou_type,
            'assignment': 'assignment_v2',
            'accumulator': (
                'compact_matches->coco_tensor'
                if use_compact_matches else 'assignment_rows->coco_tensor'
            ),
            'summary': summary,
            'stats': stats,
            'params': {
                'iou_thresholds': thresholds,
                'recall_thresholds': recalls,
                'max_dets': max_dets_,
                'max_dets_scope': 'image-category',
                'area_ranges': [
                    list(area_ranges_[key]) for key in area_labels
                ],
                'area_range_labels': area_labels,
                'area_range_map': {
                    key: list(value) for key, value in area_ranges_.items()
                },
                'image_ids': sorted(map(int, gid_to_index)),
                'category_ids': category_idxs_,
                'threshold_policy': 'inclusive',
                'segmentation_rasterizer': (
                    'coco' if iou_type == 'segm' else None
                ),
                'bbox_overlap_mode': (
                    'coco' if iou_type == 'bbox' else None
                ),
                'annotation_ignore_policy': annotation_ignore_policy,
                'ignore_classes': (
                    None if ignore_classes is None
                    else sorted(map(str, measure_ignore_classes))
                ),
                'ap_method': 'coco101',
                'truth_weight_policy': 'coco-unweighted',
                'assignment_backend': resolved_assignment_backend,
            },
            'provenance': {
                'requested_assignment_backend': requested_assignment_backend,
                'resolved_assignment_backend': assignment_backend_meta,
                'requested_overlap_backend': (
                    overlap_backend if iou_type == 'segm' else None
                ),
                'requested_raster_backend': (
                    raster_backend if iou_type == 'segm' else None
                ),
                'overlap_backends': overlap_names,
                'raster_backends': raster_names,
                'area_backends': area_names,
            },
            'timing': {
                'candidate_build_seconds': candidate_build_seconds,
                'assignment_seconds': assignment_seconds,
                'accumulation_seconds': accumulation_seconds,
                'total_seconds': total_seconds,
                'candidate_index_builds': len(gid_to_index),
                'candidate_predictions_before_maxdet': candidate_predictions_before_cap,
                'candidate_predictions_after_maxdet': candidate_predictions_after_cap,
                'assignment_slices': len(requested_slices),
                'assignment_base_slices': len(area_to_max_dets),
                'assignment_calls': assignment_calls,
                'assignment_batches': assignment_batches,
                'compact_match_path': bool(use_compact_matches),
            },
        }
        if eval_record is not None:
            result['eval'] = eval_record
        if with_details:
            result['details'] = {
                f'{area_label}:maxDets={max_det}': {
                    str(t): {
                        key: value for key, value in info.items()
                        if key not in {'precision', 'recall', 'scores', 'realpos_total'}
                    }
                    for t, info in threshold_info.items()
                }
                for (area_label, max_det), threshold_info
                in slice_metrics.items()
            }
        return result

    def score_kwant(dmet, iou_thresh: float = 0.5) -> dict[str, typing.Any]:
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

            if len(pred_dets) == 0:
                # kwant breaks on 0 predictions, hack in a bad prediction
                import kwimage

                hack_ = kwimage.Detections.random(1)
                hack_.scores[:] = 0
                pred_dets = hack_

            true_kw18 = kw18.make_kw18_from_detections(
                true_dets, frame_number=gid, timestamp=gid
            )
            pred_kw18 = kw18.make_kw18_from_detections(
                pred_dets, frame_number=gid, timestamp=gid
            )
            true_kw18s.append(true_kw18)
            pred_kw18s.append(pred_kw18)

        true_kw18 = true_kw18s
        pred_kw18 = pred_kw18s

        roc_info = kwant.score_events(
            true_kw18s, pred_kw18s, iou_thresh=iou_thresh, prefiltered=True, verbose=3
        )

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
            import kwplot

            kwplot.autompl()
            kwplot.multi_plot(
                roc_info['fa'],
                roc_info['pd'],
                xlabel='fa (fp count)',
                ylabel='pd (tpr)',
                fnum=1,
                title='kwant roc_auc={:.4f}'.format(roc_auc),
            )

            kwplot.multi_plot(
                tpr,
                ppv,
                xlabel='recall (fpr)',
                ylabel='precision (tpr)',
                fnum=2,
                title='kwant ap={:.4f}'.format(ap),
            )

        return info

    def score_kwcoco(
        dmet,
        iou_thresh: float = 0.5,
        bias: float = 0,
        gids: Iterable[int] | None = None,
        compat: str = 'all',
        prioritize: str = 'iou',
        stabalize_thresh: int = 7,
        strict: bool = False,
    ) -> dict[str, typing.Any]:
        """
        our scoring method

        Args:
            iou_thresh (float | List[float]):
                See :func:`DetectionMetrics.confusion_vectors`.

            bias (float):
                See :func:`DetectionMetrics.confusion_vectors`.

            gids (List[int] | None):
                See :func:`DetectionMetrics.confusion_vectors`.

            compat (str):
                See :func:`DetectionMetrics.confusion_vectors`.

            prioritize (str):
                See :func:`DetectionMetrics.confusion_vectors`.

            stabalize_thresh (int):
                if fewer than this many data points inserts dummy stabilization
                data so curves can still be drawn. Defaults to 7.

        Returns:
            dict
        """
        cfsn_vecs = dmet.confusion_vectors(
            iou_thresh=iou_thresh,
            bias=bias,
            gids=gids,
            compat=compat,
            prioritize=prioritize,
        )
        info = {}
        try:
            cfsn_perclass = cfsn_vecs.binarize_ovr(mode=1)
            perclass = cfsn_perclass.measures(stabalize_thresh=stabalize_thresh)
        except Exception as ex:
            if strict:
                raise
            info['perclass_error'] = {
                'type': type(ex).__name__,
                'message': str(ex),
                'repr': repr(ex),
            }
        else:
            info['perclass'] = perclass['perclass']
            info['mAP'] = perclass['mAP']
        return info

    def score_voc(
        dmet,
        iou_thresh: float = 0.5,
        bias: float = 1,
        method: str = 'voc2012',
        gids: Iterable[int] | None = None,
        ignore_classes: str | Iterable[str] | None = 'ignore',
    ) -> dict[str, typing.Any]:
        """
        score using voc method

        Example:
            >>> dmet = DetectionMetrics.demo(
            >>>     nimgs=100, nboxes=(0, 3), n_fp=(0, 1), classes=8,
            >>>     score_noise=.5)
            >>> print(dmet.score_voc()['mAP'])
            0.9399...
        """
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
                    true_dets,
                    pred_dets,
                    ioaa_thresh=iou_thresh,
                    ignore_classes=ignore_classes,
                )
                true_dets = true_dets.compress(~true_ignore_flags)
                pred_dets = pred_dets.compress(~pred_ignore_flags)

            vmet.add_truth(true_dets, gid=gid)
            vmet.add_predictions(pred_dets, gid=gid)
        voc_scores = vmet.score(iou_thresh, bias=bias, method=method)
        return voc_scores

    def _to_coco(dmet, iou_type: str = 'bbox'):
        """
        Convert truth and predictions to a strict COCO representation.

        Args:
            iou_type (str):
                ``'bbox'`` preserves the historical box-only conversion.
                ``'segm'`` additionally emits strict MS-COCO segmentations
                and uses raster mask area for area-range evaluation.

        Returns:
            Tuple[CocoDataset, CocoDataset, dict, dict]:
                prediction dataset, truth dataset, and inverse annotation
                mappings used to reconstruct confusion vectors.
        """
        import kwcoco

        if iou_type not in {'bbox', 'segm'}:
            raise KeyError(f'Unsupported COCO iou_type={iou_type!r}')

        if iou_type == 'segm':
            from kwcoco.metrics.coco_metrics import (
                coerce_pycocotools_segmentation,
            )

        true = kwcoco.CocoDataset()
        pred = kwcoco.CocoDataset()

        gt_aid_to_tx = {}
        dt_aid_to_px = {}
        classes = dmet.classes
        if classes is None:
            raise ValueError('classes are required to convert metrics to COCO')

        for node in classes:
            cid = classes.index(node)
            supercategory = list(classes.graph.pred[node])
            if len(supercategory) == 0:
                supercategory = None
            else:
                assert len(supercategory) == 1
                supercategory = supercategory[0]
            node_name = str(node)
            supercategory_name = (
                None if supercategory is None else str(supercategory)
            )
            true.add_category(
                node_name, id=cid, supercategory=supercategory_name
            )
            pred.add_category(
                node_name, id=cid, supercategory=supercategory_name
            )

        for imgname, gid in dmet._imgname_to_gid.items():
            image_kw = {}
            img_dims = dmet.gid_to_image_dims.get(gid, None)
            if img_dims is not None:
                image_kw['height'], image_kw['width'] = img_dims
            true.add_image(imgname, id=gid, **image_kw)
            pred.add_image(imgname, id=gid, **image_kw)

        idx_to_id = {
            idx: classes.index(node) for idx, node in enumerate(classes.idx_to_node)
        }

        for gid, pred_dets in dmet.gid_to_pred_dets.items():
            pred_boxes = pred_dets.boxes
            if 'scores' in pred_dets.data:
                pred_scores = pred_dets.scores
            else:
                pred_scores = np.ones(len(pred_dets))
            pred_cids = list(ub.take(idx_to_id, pred_dets.class_idxs))
            pred_xywh = pred_boxes.to_xywh().data.tolist()
            pred_segs = pred_dets.data.get('segmentations', None)

            dt_aids = []
            for px, (bbox, cid, score) in enumerate(
                zip(pred_xywh, pred_cids, pred_scores)
            ):
                ann_kw = {
                    'bbox': bbox,
                    'score': float(score),
                }
                if iou_type == 'segm':
                    img_dims = dmet.gid_to_image_dims.get(gid, None)
                    if img_dims is None:
                        raise ValueError(
                            'Image dimensions are required for COCO '
                            f'instance-segmentation scoring (gid={gid})'
                        )
                    if pred_segs is None:
                        raise ValueError(
                            'Predicted segmentations are required for '
                            f'iou_type="segm" (gid={gid}, px={px})'
                        )
                    strict_seg, seg_area = coerce_pycocotools_segmentation(
                        pred_segs[px], img_dims
                    )
                    ann_kw['segmentation'] = strict_seg
                    ann_kw['area'] = seg_area
                aid = pred.add_annotation(gid, cid, **ann_kw)
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
            true_segs = true_dets.data.get('segmentations', None)
            true_crowd = true_dets.data.get('iscrowd', None)
            true_ignore = true_dets.data.get('ignore', None)

            gt_aids = []
            for tx, (bbox, cid, weight) in enumerate(
                zip(true_xywh, true_cids, true_weights)
            ):
                ann_kw = {
                    'bbox': bbox,
                    'weight': float(weight),
                }
                if true_crowd is not None:
                    ann_kw['iscrowd'] = bool(true_crowd[tx])
                if true_ignore is not None:
                    ann_kw['ignore'] = bool(true_ignore[tx])
                if iou_type == 'segm':
                    img_dims = dmet.gid_to_image_dims.get(gid, None)
                    if img_dims is None:
                        raise ValueError(
                            'Image dimensions are required for COCO '
                            f'instance-segmentation scoring (gid={gid})'
                        )
                    if true_segs is None:
                        raise ValueError(
                            'Truth segmentations are required for '
                            f'iou_type="segm" (gid={gid}, tx={tx})'
                        )
                    strict_seg, seg_area = coerce_pycocotools_segmentation(
                        true_segs[tx], img_dims
                    )
                    ann_kw['segmentation'] = strict_seg
                    ann_kw['area'] = seg_area
                aid = true.add_annotation(gid, cid, **ann_kw)
                gt_aids.append(aid)

            gt_aid_to_tx.update(dict(zip(gt_aids, range(len(gt_aids)))))

        return pred, true, gt_aid_to_tx, dt_aid_to_px

    def score_pycocotools(
        dmet,
        with_evaler: bool = False,
        with_confusion: bool = False,
        verbose: int = 0,
        iou_thresholds: Iterable[float] | np.ndarray | None = None,
        iou_type: str = 'bbox',
        *,
        with_eval: bool = False,
        recall_thresholds: Iterable[float] | np.ndarray | None = None,
        area_ranges: typing.Mapping[str, Iterable[float]] | None = None,
        max_dets: Iterable[int] | None = None,
        gids: Iterable[int] | None = None,
        category_idxs: Iterable[int] | None = None,
        bbox_annotation_policy: str = 'historical',
    ) -> dict[str, typing.Any]:
        """
        score using ms-coco method

        Args:
            iou_type (str):
                Official COCO geometry to score. ``'bbox'`` retains the
                historical behavior. ``'segm'`` evaluates instance masks.
            bbox_annotation_policy (str):
                ``'historical'`` preserves kwcoco's long-standing direct
                ``score_pycocotools`` bbox behavior, which disables crowd
                annotations. ``'coco'`` preserves ``iscrowd`` and is the
                strict official-reference profile used by the explicit
                ``CocoEvaluator.evaluate_pycocotools`` API and conformance
                harness.
            iou_thresholds, recall_thresholds, area_ranges, max_dets, gids:
                Optional COCO parameter overrides. Custom grids are accumulated
                by official COCOeval and returned through the same stable tensor
                schema as standard parameters.
            category_idxs:
                Optional subset of the generated COCO category ids. These ids
                correspond to indexes in ``dmet.classes``. ``CocoEvaluator``
                uses this to remove only its synthetic background class from
                strict COCO evaluation while retaining real declared categories.

        Returns:
            Dict : dictionary with pct info. With standard COCO parameters,
            the ``summary`` field contains named official AP / AR statistics.

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

        if iou_type not in {'bbox', 'segm'}:
            raise KeyError(f'Unsupported COCO iou_type={iou_type!r}')
        if bbox_annotation_policy not in {'historical', 'coco'}:
            raise KeyError(bbox_annotation_policy)

        pred, true, gt_aid_to_tx, dt_aid_to_px = dmet._to_coco(
            iou_type=iou_type
        )

        # The original pycoco-api prints to much, suppress it
        quiet = verbose == 0
        with SupressPrint(coco, cocoeval, enabled=quiet):
            cocoGt = true._aspycoco()
            cocoDt = pred._aspycoco()

            for ann in cocoGt.dataset['annotations']:
                if iou_type == 'bbox':
                    if bbox_annotation_policy == 'historical':
                        # Preserve the long-standing direct-method behavior.
                        ann['ignore'] = ann['weight'] < 0.5
                        ann['iscrowd'] = False
                    else:
                        # Strict COCO reference behavior: arbitrary kwcoco
                        # truth weights / explicit ignore are not part of the
                        # detection metric, while iscrowd is.
                        ann['ignore'] = False
                        ann['iscrowd'] = bool(ann.get('iscrowd', False))
                    w, h = ann['bbox'][-2:]
                    ann['area'] = w * h
                else:
                    # Official segmentation evaluation has no kwcoco truth
                    # weight / explicit-ignore extension. COCOeval derives its
                    # detection ignore state from iscrowd and area slicing.
                    ann['ignore'] = False
                    ann['iscrowd'] = bool(ann.get('iscrowd', False))

            for ann in cocoDt.dataset['annotations']:
                if iou_type == 'bbox':
                    w, h = ann['bbox'][-2:]
                    ann['area'] = w * h

            evaler = cocoeval.COCOeval(cocoGt, cocoDt, iouType=iou_type)

            modified_params = False
            if category_idxs is not None:
                requested = sorted(set(map(int, category_idxs)))
                available = set(map(int, evaler.params.catIds))
                invalid = sorted(set(requested) - available)
                if invalid:
                    raise ValueError(
                        'category_idxs contains ids outside the generated '
                        f'COCO category set: {invalid!r}'
                    )
                if requested != sorted(available):
                    modified_params = True
                evaler.params.catIds = requested

            if gids is not None:
                requested = sorted(set(map(int, gids)))
                if requested != sorted(set(map(int, evaler.params.imgIds))):
                    modified_params = True
                evaler.params.imgIds = requested

            if iou_thresholds is None:
                iou_thresholds = np.asarray(evaler.params.iouThrs, dtype=float)
            else:
                iou_thresholds = np.asarray(list(iou_thresholds), dtype=float)
                if (
                    len(iou_thresholds) == 0
                    or not np.all(np.isfinite(iou_thresholds))
                    or np.any(iou_thresholds < 0)
                    or np.any(iou_thresholds > 1)
                ):
                    raise ValueError(
                        f'invalid iou_thresholds={iou_thresholds!r}')
                if len(set(map(float, iou_thresholds))) != len(iou_thresholds):
                    raise ValueError('iou_thresholds must be unique')
                if not np.array_equal(iou_thresholds, evaler.params.iouThrs):
                    modified_params = True
                evaler.params.iouThrs = iou_thresholds

            if recall_thresholds is not None:
                recall_thresholds = np.asarray(
                    list(recall_thresholds), dtype=float)
                if len(recall_thresholds) == 0:
                    raise ValueError('recall_thresholds must not be empty')
                if (
                    not np.all(np.isfinite(recall_thresholds))
                    or np.any(recall_thresholds < 0)
                    or np.any(recall_thresholds > 1)
                    or np.any(np.diff(recall_thresholds) < 0)
                ):
                    raise ValueError(
                        f'invalid recall_thresholds={recall_thresholds!r}')
                if not np.array_equal(
                    recall_thresholds, evaler.params.recThrs
                ):
                    modified_params = True
                evaler.params.recThrs = recall_thresholds

            if area_ranges is not None:
                area_labels: list[str] = []
                area_values: list[list[float] | list[int]] = []
                for label, bounds in area_ranges.items():
                    pair = tuple(map(float, bounds))
                    if len(pair) != 2:
                        raise ValueError(
                            f'area range {label!r} must have two values')
                    lo, hi = pair
                    if not np.isfinite(lo) or lo < 0 or np.isnan(hi) or hi < lo:
                        raise ValueError(
                            f'invalid area range {label!r}: {pair!r}')
                    area_labels.append(str(label))
                    area_values.append(list(pair))
                if not area_values:
                    raise ValueError('area_ranges must not be empty')
                if (
                    area_labels != list(evaler.params.areaRngLbl)
                    or area_values != [
                        list(map(float, r)) for r in evaler.params.areaRng
                    ]
                ):
                    modified_params = True
                evaler.params.areaRngLbl = area_labels
                evaler.params.areaRng = area_values

            if max_dets is not None:
                max_det_values: list[int] = sorted(set(map(int, max_dets)))
                if not max_det_values or any(v < 0 for v in max_det_values):
                    raise ValueError(f'invalid max_dets={max_det_values!r}')
                if max_det_values != list(map(int, evaler.params.maxDets)):
                    modified_params = True
                evaler.params.maxDets = max_det_values

            print('evaler.params.iouThrs = {!r}'.format(evaler.params.iouThrs))

            evaler.evaluate()
            evaler.accumulate()

            pct_info: dict[str, typing.Any] = {}
            from kwcoco.metrics.coco_metrics import summarize_pycocotools_eval

            if modified_params:
                print('modified params')
                pct_info.update(summarize_pycocotools_eval(evaler))
                evaler.stats = np.asarray(pct_info['stats'], dtype=float)
            else:
                print('standard pycocotools params')
                evaler.summarize()
                pct_info.update(summarize_pycocotools_eval(evaler))

            # NOTE: ``mAP`` historically exposed COCO AP50 here. Keep that
            # legacy alias on both standard and custom parameter grids. A
            # custom grid without an IoU=0.50 slice reports ``-1`` via the
            # stable named summary rather than dropping the key entirely.
            pct_info['mAP'] = float(pct_info['summary']['AP50'])
            pct_info['params']['bbox_annotation_policy'] = (
                bbox_annotation_policy if iou_type == 'bbox' else None
            )
            pct_info['iou_type'] = iou_type
            pct_info['evalar_stats'] = evaler.stats

        # 'mAP': coco_ap,
        if with_evaler:
            pct_info['evaler'] = evaler

        if with_eval:
            from kwcoco.metrics.coco_metrics import snapshot_pycocotools_eval
            pct_info['eval'] = snapshot_pycocotools_eval(evaler)

        if with_confusion:
            iou_to_cfsn_vecs = {}

            for iou_thresh in iou_thresholds:
                cfsn_vecs = pycocotools_confusion_vectors(
                    dmet, evaler, iou_thresh=iou_thresh
                )
                cfsn_vecs.data['pxs'] = np.array(
                    list(ub.take(dt_aid_to_px, cfsn_vecs.data['dt_aid'], default=-1))
                )
                cfsn_vecs.data['txs'] = np.array(
                    list(ub.take(gt_aid_to_tx, cfsn_vecs.data['gt_aid'], default=-1))
                )
                iou_to_cfsn_vecs[iou_thresh] = cfsn_vecs

            pct_info['iou_to_cfsn_vecs'] = iou_to_cfsn_vecs

        return pct_info

    score_coco = score_pycocotools

    @classmethod
    def demo(cls, **kwargs: typing.Any) -> 'DetectionMetrics':
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
            >>> from kwcoco.metrics.detect_metrics import *  # NOQA
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
            perterbkw = ub.dict_isect(
                kwargs,
                {
                    'rng': 0,
                    'box_noise': 0,
                    'cls_noise': 0,
                    'null_pred': False,
                    'with_probs': False,
                    'score_noise': 0.2,
                    'n_fp': 0,
                    'n_fn': 0,
                    'hacked': 1,
                },
            )

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

            DiscreteUniform = distributions.DiscreteUniform.seeded(rng=rng)  # type: ignore

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
            true_low = _interp(0, mid, score_noise)
            false_high = _interp(true_high, mid - 1e-3, score_noise)
            true_mean = _interp(0.5, 0.8, score_noise)
            false_mean = _interp(0.5, 0.2, score_noise)

            true_score_RV = distributions.TruncNormal(
                mean=true_mean, std=0.5, low=true_low, high=true_high, rng=rng
            )
            false_score_RV = distributions.TruncNormal(
                mean=false_mean, std=0.5, low=0, high=false_high, rng=rng
            )

            # Create the category hierarchy
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
                nboxes_ = int(np.asarray(nboxes_RV()).item())
                n_fp_ = int(np.asarray(n_fp_RV()).item())
                n_fn_ = int(np.asarray(n_fn_RV()).item())

                imgname = 'img_{}'.format(gid)
                dmet._register_imagename(imgname, gid)

                # Generate random ground truth detections
                true_boxes = kwimage.Boxes.random(
                    num=nboxes_, scale=scale, anchors=anchors, rng=rng, format='cxywh'
                )
                # Prevent 0 sized boxes: increase w/h by 1
                true_boxes.data[..., 2:4] += 1
                true_cxs = np.asarray(frgnd_cx_RV(len(true_boxes)), dtype=int)
                true_weights = np.ones(len(true_boxes), dtype=np.int32)

                # Initialize predicted detections as a copy of truth
                pred_boxes = true_boxes.copy()
                pred_cxs = true_cxs.copy()

                # Perterb box coordinates
                pred_boxes.data = np.abs(pred_boxes.data.astype(float) + box_noise_RV())

                # Perterb class predictions
                change = np.asarray(cls_noise_RV(len(pred_cxs)), dtype=bool)
                pred_cxs_swap = np.asarray(frgnd_cx_RV(len(pred_cxs)), dtype=int)
                pred_cxs[change] = pred_cxs_swap[change]

                # Drop true positive boxes
                if n_fn_:
                    pred_boxes.data = pred_boxes.data[n_fn_:]
                    pred_cxs = pred_cxs[n_fn_:]

                # pred_scores = np.linspace(true_min, true_max, len(pred_boxes))[::-1]
                n_tp_ = len(pred_boxes)
                pred_scores = np.asarray(true_score_RV(n_tp_), dtype=float)

                # Add false positive boxes
                if n_fp_:
                    false_boxes = kwimage.Boxes.random(
                        num=n_fp_, scale=scale, rng=rng, format='cxywh'
                    )
                    false_cxs = np.asarray(frgnd_cx_RV(n_fp_), dtype=int)
                    false_scores = np.asarray(false_score_RV(n_fp_), dtype=float)

                    pred_boxes.data = np.vstack([pred_boxes.data, false_boxes.data])
                    pred_cxs = np.hstack([pred_cxs, false_cxs])
                    pred_scores = np.hstack([pred_scores, false_scores])

                # Transform the scores for the assigned class into a predicted
                # probability for each class. (Currently a bit hacky).
                class_probs = _demo_construct_probs(
                    pred_cxs, pred_scores, classes, rng, hacked=kwargs.get('hacked', 1)
                )

                true_dets = kwimage.Detections(
                    boxes=true_boxes, class_idxs=true_cxs, weights=true_weights
                )

                pred_dets = kwimage.Detections(
                    boxes=pred_boxes, class_idxs=pred_cxs, scores=pred_scores
                )

                # Hack in the probs
                if with_probs:
                    pred_dets.data['probs'] = class_probs

                if null_pred:
                    pred_dets.data['class_idxs'] = np.array(
                        [None] * len(pred_dets), dtype=object
                    )

                dmet.add_truth(true_dets, imgname=imgname)
                dmet.add_predictions(pred_dets, imgname=imgname)

        return dmet

    def summarize(
        dmet,
        out_dpath: str | ub.Path | None = None,
        plot: bool = False,
        title: str = '',
        with_bin: str | bool = 'auto',
        with_ovr: str | bool = 'auto',
    ) -> dict[str, typing.Any]:
        """
        Create summary one-versus-rest and binary metrics.

        Args:
            out_dpath (pathlib.Path | None):
                FIXME: not hooked up

            with_ovr (str | bool):
                include one-versus-rest metrics (wrt the classes). If 'auto' enables if possible.
                FIXME: auto is not working.

            with_bin (str | bool):
                include binary classless metrics (i.e. detected or not).
                If 'auto' enables if possible. FIXME: auto is not working.

            plot (bool):
                if true, also write plots. Defaults to False.

            title (str):
                passed if plot is given

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


def pycocotools_confusion_vectors(
    dmet: DetectionMetrics,
    evaler: typing.Any,
    iou_thresh: float = 0.5,
    verbose: int = 0,
) -> ConfusionVectors:
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
            for dtid, gtid, score, ignore in zip(
                info['dtIds'],
                info['dtMatches'][0],
                info['dtScores'],
                info['dtIgnore'][0],
            ):
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

    # Convert the accumulated rows once after all COCO eval-image records have
    # been visited.  Historically this conversion happened inside the loop,
    # repeatedly rebuilding larger arrays and making confusion extraction
    # unnecessarily superlinear in the number of eval records.
    _data = {}
    for k, v in ub.ProgIter(
        list(y_accum.items()), desc='ndarray convert', verbose=verbose
    ):
        # Try to use 32 bit types for large evaluation problems
        dtype = None
        if k in {'iou', 'score', 'weight'}:
            dtype = np.float32
        if k in {'pxs', 'txs', 'gid', 'pred', 'true'}:
            dtype = np.int32
        try:
            _data[k] = np.asarray(v, dtype=dtype)
        except TypeError:
            _data[k] = np.asarray(v)

    # Avoid pandas when possible
    cfsn_data = kwarray.DataFrameArray(_data)

    # if _tracking_probs:
    #     y_prob = np.vstack(prob_accum)
    # else:
    y_prob = None
    cfsn_vecs = ConfusionVectors(cfsn_data, classes=dmet.classes, probs=y_prob)
    return cfsn_vecs


def eval_detections_cli(**kw: typing.Any) -> None:
    """
    DEPRECATED USE `kwcoco eval` instead

    CommandLine:
        xdoctest -m ~/code/kwcoco/kwcoco/metrics/detect_metrics.py eval_detections_cli
    """
    import scriptconfig as scfg
    import kwcoco

    ub.schedule_deprecation(
        'kwcoco',
        name='kwcoco.metrics.detect_metrics.eval_detections_cli',
        type='method',
        deprecate='0.3.4',
        error='1.0.0',
        remove='1.1.0',
        migration=('Use `kwcoco eval` in kwcoco.cli.coco_eval instead. '),
    )

    class EvalDetectionCLI(scfg.Config):
        default = {
            'true': scfg.Path(None, help='true coco dataset'),
            'pred': scfg.Path(None, help='predicted coco dataset'),
            'out_dpath': scfg.Path('./out', help='output directory'),
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
            fnum=1,
            pnum=(1, 4, 1),
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
            fnum=1,
            pnum=(1, 4, 2),
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
            fnum=1,
            pnum=(1, 4, 3),
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
            fnum=1,
            pnum=(1, 4, 4),
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
    iouStr = (
        '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1])
        if iouThr is None
        else '{:0.2f}'.format(iouThr)
    )

    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRngLbl]
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
                print(
                    'catname_to_ap = {}'.format(
                        ub.urepr(catname_to_ap, nl=1, precision=2)
                    )
                )
                # print('pct_perclass_ap = {}'.format(ub.urepr(pct_perclass_ap.tolist(), nl=1, precision=2)))
                print(
                    'pct_map = {}'.format(ub.urepr(pct_map.tolist(), nl=0, precision=2))
                )
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
    print(iStr.format(titleStr, typeStr, iouStr, areaRngLbl, maxDets, mean_s))
    return mean_s


def pct_summarize2(self: typing.Any) -> list[typing.Any]:
    stats = []
    for ap in [1, 0]:
        for areaRngLbl in self.params.areaRngLbl:
            stats.append(_summarize(self, ap=ap, iouThr=None, areaRngLbl=areaRngLbl))
            if areaRngLbl == 'all':
                if len(self.params.iouThrs) > 1:
                    for iouThr in self.params.iouThrs:
                        stats.append(
                            _summarize(
                                self, ap=ap, iouThr=iouThr, areaRngLbl=areaRngLbl
                            )
                        )
    return stats


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/kwcoco/metrics/detect_metrics.py all
    """
    import xdoctest

    xdoctest.doctest_module(__file__)
