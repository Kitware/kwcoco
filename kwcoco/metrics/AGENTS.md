# AGENTS.md

Vectorized metrics for **detection / classification / segmentation**. Core pattern:

(1) build `ConfusionVectors` (per-prediction rows) → (2) aggregate to `Measures` (AP/AUC/F1/MCC/etc.). More flexible than `pycocotools`: keep assignments, change thresholds / AP methods without rematching.

## Map

* `confusion_vectors.py`: row-wise truth/pred/score (+ weight, IoU, txs/pxs); OVR binarize; matrix rendering.
* `confusion_measures.py`: thresholded counts → metrics (PR/ROC curves, AP variants, MCC, etc.); `ap_method` controls AP style.
* `assignment.py`: IoU matching → confusion vectors.
* `detect_metrics.py`: end-to-end detection eval (from `kwimage.Detections` or `CocoDataset`); can score via kwcoco, VOC, or `pycocotools`.
* `coco_metrics.py`: narrow strict-MS-COCO compatibility helpers and stable named summaries for official bbox / instance-mask `pycocotools` metrics.
* `segmentation_metrics.py`, `voc_metrics.py`, `clf_report.py`, `drawing.py`, `functional.py`: task helpers + plotting/utilities (mostly operate on vectors/measures).

## kwcoco vs `pycocotools` (mental model)

* **kwcoco**: store all matches/unmatches as explicit rows → later aggregate however you want.

  * Typical: `ConfusionVectors.binarize_ovr().measures()` / `BinaryConfusionVectors.measures()`
  * AP parity: `ap_method='pycocotools'` (COCO-style 101 recall sampling); other methods exist (`sklearn`, `outlier`, `sklish`).
* **pycocotools**: aggregates during matching (category-filtered, precision smoothing, recall resampling). Changing AP style typically means rerunning its pipeline.

## Interop

* `DetectionMetrics.score_pycocotools(iou_type=...)` wraps official COCO eval (after conversion). `iou_type='bbox'` is the historical default; `iou_type='segm'` computes official instance-mask metrics. Expect small deltas vs `score_kwcoco()` due to smoothing/AP details.
* `CocoEvaluator.evaluate_pycocotools()` is the dataset-level API for named official COCO box / mask AP and AR summaries. Keep this separate from the custom confusion-vector evaluator unless its assignment geometry is explicitly generalized.
* KWCoco polygons may contain interiors (holes), which strict MS-COCO polygons cannot represent. Preserve rich geometry internally and convert only incompatible cases to exact RLE at the pycocotools boundary.
* `with_confusion=True` returns confusion vectors from the pycocotools assignment for side-by-side debugging.
* COCO “mutex” matching (pred category must equal truth) corresponds to `compat='mutex'` in kwcoco builders.


## Metric backend comparison

The official `pycocotools` bridge is a reference backend, not a replacement for
kwcoco's confusion-vector architecture.  Before changing assignment or AP
semantics, use `dev/pycocotools_metrics_compare/compare_metrics.py` to identify
which layer causes a difference.

The comparison suite separates:

* overlap-kernel behavior (`kwimage_ext` versus current pycocotools masks);
* assignment behavior (kwcoco versus COCOeval);
* accumulation behavior (kwcoco `Measures` versus COCOeval accumulation).

For bbox evaluation it can measure all three combinations.  Instance-mask work
has progressed through native reusable candidate indexes, COCO-compatible
polygon rasterization, inclusive mask thresholds, crowd semantics, and kwcoco
ignore-region semantics.  Keep the official COCO path as an independent
reference so assignment differences and accumulation differences remain
separately measurable.

## Debug / Usage

* Inspect assignments: `ConfusionVectors.data._pandas()`.
* Per-class diagnostics: `binarize_ovr()` then `.measures()`.
* For dataset scoring: `DetectionMetrics.from_coco(true, pred)` → `score_kwcoco()` / `score_voc()` / `score_pycocotools()`.
* When comparing to external reports: set `ap_method` explicitly (COCO parity → `pycocotools`).

---


### Phase-1 assignment comparison gate

Before changing production assignment semantics or adding native instance-mask
assignment, run `dev/pycocotools_metrics_compare/run_phase1_suite.py`.  It
compares the legacy native matcher with `assignment_v2` over the full
COCO-style IoU threshold set and records row-level differences, timing, sparse
candidate-index behavior, official COCO reference metrics, mask-kernel backend
identity, and dense saliency accumulation behavior.

Treat `assignment_v2` as experimental until measured differences are
classified.  In particular, do not promote it solely because its reusable
candidate graph is a better architectural fit for mask overlaps.

### Phase-1.5 backend gate

When backend availability changes (notably a legacy-vs-Rust `kwimage_ext`
installation), run `dev/pycocotools_metrics_compare/run_phase15_suite.py` and
retain both result bundles.  Do not infer assignment-algorithm speedups from a
run where the underlying overlap backend also changed.

### Phase-2.3 native COCO-shaped mask gate

`DetectionMetrics.score_native_coco(iou_type='segm')` is the dataset-level
native mask summary surface.  It uses assignment-time COCO area semantics,
per-image/per-category `maxDets`, one reusable candidate index per image, and
native `ConfusionVectors -> Measures` accumulation.  It is not official
COCOeval and must stay labeled as kwcoco.

Run `dev/pycocotools_metrics_compare/run_phase23_native_coco.py` when changing
area, maxDet, mask geometry, or native summary semantics.  Historical bbox
`CocoEvaluator.evaluate()` remains on assignment-v1 unless a separate bbox
parity campaign explicitly changes that decision.
