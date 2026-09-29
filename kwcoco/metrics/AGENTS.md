# AGENTS.md

Vectorized metrics for **detection / classification / segmentation**. Core pattern:

(1) build `ConfusionVectors` (per-prediction rows) → (2) aggregate to `Measures` (AP/AUC/F1/MCC/etc.). More flexible than `pycocotools`: keep assignments, change thresholds / AP methods without rematching.

## Map

* `confusion_vectors.py`: row-wise truth/pred/score (+ weight, IoU, txs/pxs); OVR binarize; matrix rendering.
* `confusion_measures.py`: thresholded counts → metrics (PR/ROC curves, AP variants, MCC, etc.); `ap_method` controls AP style.
* `assignment.py`: IoU matching → confusion vectors.
* `detect_metrics.py`: end-to-end detection eval (from `kwimage.Detections` or `CocoDataset`); can score via kwcoco, VOC, or `pycocotools`.
* `coco_metrics.py`: strict-MS-COCO reference helpers plus the narrow native
  COCO assignment-row accumulator; keep official conversion and native
  accumulation conceptually separate.
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
a dedicated native COCO 101-recall-point accumulator over explicit assignment
rows.  Do not route this summary back through generic `Measures`: COCO requires
stable descending score ordering for ties, while historical generic curve code
has different tie behavior.  `ConfusionVectors -> Measures` remains the
general kwcoco analysis path and should not be globally changed merely for
COCO compatibility.  The native summary is not official COCOeval and must stay
labeled as kwcoco.

Run `dev/pycocotools_metrics_compare/run_phase23_native_coco.py` when changing
area, maxDet, mask geometry, tie ordering, or native summary semantics.  Use
`--assert-parity` for the COCO-compatibility regression corpus.  The corpus must
retain cases for arbitrary kwcoco truth weights (which COCO scoring ignores) and
a literal category named `background` (which COCO treats as an ordinary label).
Do not special-case category names in the native COCO accumulator.

Native result records keep `backend='kwcoco'` for the package-level backend and
also expose an explicit implementation id, `official=False`, the official
reference implementation name, requested backend choices, and resolved backend
provenance.  Official summary records expose the same versioned result-schema
metadata with `official=True`. Preserve this distinction when extending the API.

`overlap_backend='auto'` and `raster_backend='auto'` must prefer `kwimage_ext`
and fall back to `pycocotools`.  If neither optional backend is importable, the
error should name both installation choices rather than silently changing
geometry semantics.

Historical bbox `CocoEvaluator.evaluate()` remains on assignment-v1 unless a
separate bbox parity campaign explicitly changes that decision.

### Phase-2.4 / Phase-3 COCO conformance gate

The native COCO campaign now has a stronger target than the 12 scalar summary
values. `score_native_coco(..., with_eval=True)` and
`score_pycocotools(..., with_eval=True)` expose the accumulated COCO precision,
recall, and sampled-score tensors. Use
`dev/pycocotools_metrics_compare/run_phase3_coco_conformance.py` to compare
those tensors, their absent-category masks, row assignments, and summaries.

Phase 3 also adds an **explicit** native COCO bbox compatibility profile. This
does not promote assignment-v2 as the default bbox matcher. Preserve these
boundaries:

* `CocoEvaluator.evaluate()` remains the historical kwcoco bbox evaluator.
* Generic `assignment_v2` bbox keeps `bbox_overlap_mode='legacy'` by default.
* Native COCO bbox scoring explicitly uses `bbox_overlap_mode='coco'`.
* The COCO bbox profile uses continuous-coordinate IoU, crowd
  intersection/detection-area, reusable ignored crowd truths, inclusive
  thresholds, category-scoped maxDets, and COCO equal-IoU truth tie behavior.
* `DetectionMetrics.score_pycocotools()` keeps its historical direct-method
  bbox annotation policy by default. `CocoEvaluator.evaluate_pycocotools()`
  opts into strict `bbox_annotation_policy='coco'` for official reference use.

Do not call bbox-v2 parity with COCO proof of parity with historical kwcoco
bbox assignment. Those are different contracts and must remain separate.

Strict native COCO scoring may accelerate the greedy candidate-graph matcher
with ``assignment_backend='kwimage_ext'``.  The Rust capability is deliberately
a narrow kernel: kwcoco still owns candidate geometry, COCO area / ignore
policy, and tensor semantics.  The production compact path batches every active
area range and IoU threshold in one backend call per image and feeds compact
match arrays directly into the COCO accumulator instead of materializing Python
confusion-row dictionaries.  Keep ``assignment_backend='python'`` as the
readable compact semantic reference and fallback.  Explicit assignment rows are
still materialized for diagnostic ``with_details`` requests and native kwcoco
ignore-region semantics. ``auto`` may select Rust only when the grid capability
is present and the requested semantics are supported.  In particular,
threshold-dependent kwcoco ``ignore_classes`` remain Python-owned.  Every Rust
assignment change must pass the same full bbox+segm tensor conformance gate as
Python, preferably with both backends forced in separate runs.

Strict ``CocoEvaluator`` COCO wrappers must also bypass the historical class-name
rectification used by ``evaluate()``.  Declared truth category ids define
identity for the strict native/official wrappers, so labels such as ``Foo Bar``
and ``foo_bar`` remain distinct even if the internal tensor axis is densely
indexed.  Do not move that normalization into the strict
COCO path merely to share setup code with the historical evaluator.

For new COCO compatibility changes, the preferred acceptance command is:

```bash
python dev/pycocotools_metrics_compare/run_phase3_coco_conformance.py \
    --mode=full --assert-parity \
    --output-dpath=_metric_compare_phase3_coco_conformance_full
```

A mismatch should be localized in this order: overlap -> per-image assignment
-> accumulated tensor -> 12-value summary. Do not tune summary values directly
when a lower layer disagrees. The conformance gate also exercises a nonstandard
parameter grid and normalized image subset; avoid hard-coding standard COCO
tensor dimensions in either the native or official wrapper. COCO's explicit
IoU threshold 1.0 uses the reference comparison cap ``1 - 1e-10``.


### Historical release comparison gate

Use ``dev/pycocotools_metrics_compare/run_version_timeline.py`` when making
claims about upgrade compatibility or speed relative to an older KWCoco
release.  It is intentionally separate from native-versus-pycocotools release
benchmarking: freeze the input JSON once, change only the KWCoco source tree,
and compare the complete serialized historical evaluator metric state.  Do not
infer a v0.9.0-to-v0.9.1 speedup from the current native-versus-official
benchmark because the older release did not expose the same native COCO path.

### Release evidence and large-scale benchmarking

After semantic/tensor conformance is green, use
``dev/pycocotools_metrics_compare/run_release_benchmark.py`` for release and
publication evidence.  Keep timing and memory passes separate, run every backend
in an isolated process, and treat peak process RSS (not tracemalloc) as the
primary total-memory metric because native C/Rust allocations matter.  The
benchmark must retain complete precision/recall/sample-score tensor parity on
every scaling workload and record CPU/RAM/package/git/backend provenance.

For technical reports, publish raw timing distributions and workload counts,
not only a best-case speedup.  Report both core scorer and public
``CocoEvaluator`` scopes when making user-facing latency claims.  Synthetic
scaling results should be paired with at least one real truth/prediction corpus
before making application-specific claims.  Keep ``pycocotools.COCOeval`` named
as the official reference even when the accelerated native path is faster on
measured workloads.
