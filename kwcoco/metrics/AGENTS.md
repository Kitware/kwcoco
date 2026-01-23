# AGENTS.md

Vectorized metrics for **detection / classification / segmentation**. Core pattern:

(1) build `ConfusionVectors` (per-prediction rows) → (2) aggregate to `Measures` (AP/AUC/F1/MCC/etc.). More flexible than `pycocotools`: keep assignments, change thresholds / AP methods without rematching.

## Map

* `confusion_vectors.py`: row-wise truth/pred/score (+ weight, IoU, txs/pxs); OVR binarize; matrix rendering.
* `confusion_measures.py`: thresholded counts → metrics (PR/ROC curves, AP variants, MCC, etc.); `ap_method` controls AP style.
* `assignment.py`: IoU matching → confusion vectors.
* `detect_metrics.py`: end-to-end detection eval (from `kwimage.Detections` or `CocoDataset`); can score via kwcoco, VOC, or `pycocotools`.
* `segmentation_metrics.py`, `voc_metrics.py`, `clf_report.py`, `drawing.py`, `functional.py`: task helpers + plotting/utilities (mostly operate on vectors/measures).

## kwcoco vs `pycocotools` (mental model)

* **kwcoco**: store all matches/unmatches as explicit rows → later aggregate however you want.

  * Typical: `ConfusionVectors.binarize_ovr().measures()` / `BinaryConfusionVectors.measures()`
  * AP parity: `ap_method='pycocotools'` (COCO-style 101 recall sampling); other methods exist (`sklearn`, `outlier`, `sklish`).
* **pycocotools**: aggregates during matching (category-filtered, precision smoothing, recall resampling). Changing AP style typically means rerunning its pipeline.

## Interop

* `DetectionMetrics.score_pycocotools()` wraps official COCO eval (after conversion). Expect small deltas vs `score_kwcoco()` due to smoothing/AP details.
* `with_confusion=True` returns confusion vectors from the pycocotools assignment for side-by-side debugging.
* COCO “mutex” matching (pred category must equal truth) corresponds to `compat='mutex'` in kwcoco builders.

## Debug / Usage

* Inspect assignments: `ConfusionVectors.data._pandas()`.
* Per-class diagnostics: `binarize_ovr()` then `.measures()`.
* For dataset scoring: `DetectionMetrics.from_coco(true, pred)` → `score_kwcoco()` / `score_voc()` / `score_pycocotools()`.
* When comparing to external reports: set `ap_method` explicitly (COCO parity → `pycocotools`).

---

