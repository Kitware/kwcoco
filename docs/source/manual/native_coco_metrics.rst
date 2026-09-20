Native and Official COCO Detection Metrics
==========================================

KWCoco exposes several metric families that should not be conflated.

``CocoEvaluator.evaluate()``
    The historical configurable kwcoco detection evaluator. Its established
    bbox matcher and output remain unchanged by the native COCO campaign.

``CocoEvaluator.evaluate_pycocotools()``
    The official COCO reference path. It delegates matching and accumulation
    to ``pycocotools.COCOeval`` and supports ``bbox`` and ``segm``. The
    dataset-level API uses strict COCO bbox crowd semantics by default.

``CocoEvaluator.evaluate_native_coco()``
    KWCoco's explicit native COCO-compatibility path for ``bbox`` and
    ``segm``. It uses reusable assignment-v2 candidate graphs plus a dedicated
    COCO tensor accumulator. This API is opt-in and does not replace the
    historical bbox evaluator.

``kwcoco.metrics.segmentation_metrics``
    Dense pixel / saliency metrics. These answer a different question from
    instance detection AP and remain independently useful.

Native COCO summary and tensor
------------------------------

The native COCO path retains inspectable kwcoco assignment semantics while
using a compact match representation for normal scoring. Explicit assignment
rows are materialized only for diagnostic/native-extension requests. With
standard parameters it uses ten IoU thresholds from 0.50 through 0.95, 101
recall samples, the standard inclusive area ranges, and per-image/per-category
``maxDets`` values of 1, 10, and 100.

``with_eval=True`` additionally returns the full COCO evaluation tensors:

* precision: ``[T, R, K, A, M]``;
* recall: ``[T, K, A, M]``;
* score thresholds: ``[T, R, K, A, M]``.

Those tensors are the primary differential target used by the Phase-2.4 / 3
conformance suite. Matching the 12 scalar summary values alone is not treated
as sufficient evidence.

Both native and official wrappers accept the same optional parameter overrides:
``iou_thresholds``, ``recall_thresholds``, ``area_ranges``, ``max_dets``, and
``gids``. Image ids are unique-sorted like COCOeval. This makes it possible to
validate nonstandard tensor shapes directly instead of assuming compatibility
from the standard grid. The standard 12 summary keys remain present; a named
standard slice that is absent from a custom grid is reported as ``-1``.

The official reference remains ``pycocotools.COCOeval``. Native and official
summary records share a versioned result-schema identifier. Native records
identify themselves explicitly with ``official=False`` and include both the
native implementation name and the reference implementation name; official
records use ``official=True``.

.. code:: python

   from kwcoco.coco_evaluator import CocoEvaluator

   evaluator = CocoEvaluator({
       'true_dataset': true_dataset,
       'pred_dataset': pred_dataset,
   })

   native = evaluator.evaluate_native_coco(
       ('bbox', 'segm'), with_eval=True)
   official = evaluator.evaluate_pycocotools(
       ('bbox', 'segm'), with_eval=True)

   native_mask_ap = native['segm']['summary']['AP']
   official_mask_ap = official['segm']['summary']['AP']
   native_bbox_precision = native['bbox']['eval']['precision']

BBox compatibility profile
--------------------------

Native bbox COCO scoring is deliberately an explicit compatibility profile.
It does **not** make assignment-v2 the default bbox engine and does not change
``CocoEvaluator.evaluate()``.

The COCO bbox profile differs from generic assignment-v2 bbox behavior in
several important ways:

* IoU thresholds are inclusive;
* ``iscrowd`` truths use intersection divided by detection area rather than
  ordinary union IoU;
* crowd truths are reusable and matched detections are ignored;
* equal-IoU truth ties follow COCOeval's scan semantics, which prefer the later
  truth among otherwise equivalent candidates;
* score ties preserve original prediction order via stable descending sort;
* area ranges participate during matching rather than as post-hoc weights;
* ``maxDets`` is per image and per category.

The lower-level assignment-v2 option is ``bbox_overlap_mode='coco'``. Its
normal default remains ``'legacy'`` specifically to avoid changing existing
experimental bbox users while the historical kwcoco matcher remains separate.

Official bbox reference policy
------------------------------

``DetectionMetrics.score_pycocotools()`` predates the explicit official
reference API and historically disabled bbox crowd annotations. That direct
method retains ``bbox_annotation_policy='historical'`` by default for
compatibility.

``CocoEvaluator.evaluate_pycocotools()`` defaults to
``bbox_annotation_policy='coco'`` and should be used when strict official bbox
semantics are required. The Phase-3 conformance runner also uses the strict
``'coco'`` policy.

Backend selection and provenance
--------------------------------

Native mask evaluation needs a COCO-compatible RLE / rasterization backend.
``overlap_backend='auto'`` and ``raster_backend='auto'`` prefer
``kwimage_ext`` when available and otherwise fall back to ``pycocotools``.
The requested and resolved backend names are recorded in the result's
``provenance`` dictionary. ``kwimage_ext`` is an optional acceleration backend.

Native bbox COCO evaluation has no mask dependency. Its overlap provenance is
``numpy_coco_bbox`` and its area provenance is ``numpy_coco_xywh_area``.

Greedy assignment has a separate optional acceleration switch.  The
``assignment_backend`` parameter accepts ``'auto'``, ``'python'``, or
``'kwimage_ext'``.  ``'auto'`` prefers the batched Rust
``kwimage_ext.coco_greedy_match_grid`` kernel when that capability is installed
and falls back to the Python assignment-v2 reference implementation otherwise.
The compiled kernel operates only on the already-built sparse candidate graph
and batches all active area ranges and IoU thresholds in one call per image.
Normal COCO accumulation consumes the resulting compact NumPy match arrays
directly; it does not expand them into Python confusion-row dictionaries.
Explicit rows remain available when callers request per-class details or native
kwcoco ignore-region semantics.  Geometry, area policy, class compatibility,
and COCO tensor semantics remain in kwcoco.  The requested and resolved
assignment backends are recorded in ``params`` / ``provenance``. Native kwcoco
ignore-class regions remain on the Python row backend because their suppression
mask depends on the IoU threshold.

For backend differential testing, force both implementations explicitly::

   python dev/pycocotools_metrics_compare/run_phase3_coco_conformance.py \
       --mode=full --assignment-backend=python --assert-parity \
       --output-dpath=_metric_compare_phase3_python

   python dev/pycocotools_metrics_compare/run_phase3_coco_conformance.py \
       --mode=full --assignment-backend=kwimage_ext --assert-parity \
       --output-dpath=_metric_compare_phase3_rust

If neither ``kwimage_ext`` nor ``pycocotools`` is importable, native mask
scoring raises an ``ImportError`` that names both backend choices rather than
falling back to different geometry semantics.

Semantics fixed for COCO compatibility
--------------------------------------

* IoU thresholds are inclusive. COCOeval's matching cap
  ``min(threshold, 1 - 1e-10)`` is also mirrored for explicit threshold 1.0.
* Polygon rasterization uses COCO-compatible rasterization semantics.
* COCO crowd annotations use intersection over detection area, can absorb
  multiple detections, and contribute zero metric weight.
* Out-of-range truths remain ignored match candidates; unmatched detections
  outside the requested area range are ignored for that slice.
* Segmentation area slices use mask area; bbox slices use bbox area.
* ``maxDets`` is applied per image and per category.
* Equal prediction scores preserve original order during stable descending
  sorting.
* Equal-IoU truth ties reproduce COCOeval's later-truth scan result.
* COCOeval image-id normalization is mirrored before dataset accumulation.
* Arbitrary kwcoco truth ``weight`` values are ignored because COCOeval has no
  corresponding weighting extension.
* A category literally named ``background`` is an ordinary COCO category name;
  it is not silently removed from native COCO aggregation.
* Strict ``CocoEvaluator`` COCO wrappers key category identity by the declared
  truth category ids and do not inherit the historical evaluator's lowercase /
  space-to-underscore class-name rectification. Distinct categories such as
  ``Foo Bar`` and ``foo_bar`` therefore remain distinct on the COCO category
  axis, even though the internal tensor axis is densely indexed.

KWCoco extensions
-----------------

``annotation_ignore_policy='kwcoco'`` can additionally honor explicit kwcoco
truth ``ignore`` metadata, and ``ignore_classes`` can apply kwcoco's spatial
ignore-region behavior. Those are useful native extensions but are not part of
strict official COCOeval semantics. Use the default
``annotation_ignore_policy='coco'`` and no ``ignore_classes`` when comparing
against the official reference.

Conformance evidence
--------------------

The authoritative differential harness is:

.. code:: bash

   python dev/pycocotools_metrics_compare/run_phase3_coco_conformance.py \
       --mode=full \
       --assert-parity \
       --output-dpath=_metric_compare_phase3_coco_conformance_full

It compares native and official bbox/segmentation results at three independent
layers: row assignment, the complete COCO accumulation tensor, and the standard
12 AP/AR summary fields. Selected cases also run a nonstandard IoU / recall /
area / maxDet grid and a normalized image subset. Adversarial cases cover crowd
IoA, exact threshold and area boundaries, stable score ties, equal-IoU truth
ties, ``maxDets`` cutoffs, duplicate detections, empty truth/prediction sets,
prediction-only categories, truth-weight and explicit-ignore extensions,
negative detection scores, fractional polygons, rich polygons with holes,
compressed RLE input, and multiple randomized perturbation seeds.

A passing summary-only run is not considered equivalent evidence to a passing
Phase-3 tensor conformance run.
