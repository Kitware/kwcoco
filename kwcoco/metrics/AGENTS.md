# Agent Notes for `kwcoco.metrics`

This subpackage implements vectorized metric tooling for detection, classification, and segmentation tasks. The public APIs revolve around generating *confusion vectors* first, then aggregating them into *confusion measures* (AP, AUC, F1, MCC, etc.). The workflow is more general than the classic `pycocotools` evaluation and is compatible with, but not limited to, COCO-style data.

## Package map
- `confusion_vectors.py` – tabular containers for per-example truth/pred/score records plus helpers to binarize one-vs-rest and render matrices.
- `confusion_measures.py` – accumulates thresholded TP/FP/FN/TN counts into derived metrics (AP variants, ROC/PR curves, MCC, etc.).
- `detect_metrics.py` – end-to-end detection evaluator producing confusion vectors from `kwimage.Detections` or `CocoDataset`s; can score with kwcoco metrics, VOC, or delegate to `pycocotools`.
- `assignment.py` – IoU matching logic that creates confusion vectors from predicted/true detections.
- `segmentation_metrics.py`, `voc_metrics.py`, `clf_report.py`, `drawing.py`, `functional.py` – task-specific helpers and plotting/utilities; most operate on confusion vectors or measures.

## Confusion vectors vs. pycocotools accumulation
- Confusion vectors store every matched/unmatched prediction as a row with true label, predicted label, score, weight, IoU, and indices (`txs`/`pxs`). This makes later aggregation flexible (e.g., binarize per-class, recompute metrics with new thresholds) and debuggable because the underlying assignments are explicit.【F:kwcoco/metrics/confusion_vectors.py†L1-L135】
- Aggregation happens in `BinaryConfusionVectors.measures()` / `ConfusionVectors.binarize_ovr().measures()`, which compute multiple AP flavors. The default `ap_method='pycocotools'` reproduces the 101-point recall sampling used by the COCO API, while alternatives (`sklearn`, `outlier`, `sklish`) use different trapezoidal or outlier-robust schemes.【F:kwcoco/metrics/confusion_measures.py†L1300-L1414】
- In contrast, `pycocotools` aggregates during matching: predictions are filtered to the same category, precision is monotonically smoothed, and AP is averaged over resampled recall bins. kwcoco preserves the raw assignments and lets you switch accumulation strategies without recomputing matches.【F:kwcoco/metrics/detect_metrics.py†L730-L782】【F:kwcoco/metrics/confusion_measures.py†L1300-L1414】

## Interoperability tips
- `DetectionMetrics.score_pycocotools()` wraps the official evaluator after converting kwcoco detections to pycoco objects. Expect minor score deltas versus `score_kwcoco()` because AP calculation and precision smoothing differ. Use `with_confusion=True` to pull confusion vectors derived from the pycocotools assignments for side-by-side comparisons.【F:kwcoco/metrics/detect_metrics.py†L730-L810】
- COCO’s mutex matching (pred category must equal truth category) is mirrored via `compat='mutex'` on kwcoco confusion-vector builders; pycocotools ignores multi-label scores per box, so kwcoco’s per-class binarization can yield more nuanced diagnostics.【F:kwcoco/metrics/detect_metrics.py†L755-L782】【F:kwcoco/metrics/confusion_vectors.py†L62-L76】

## Practical guidance
- When debugging metrics, inspect `ConfusionVectors.data._pandas()` to view per-sample assignments, or call `binarize_ovr()` to get per-class binary vectors before computing measures.【F:kwcoco/metrics/confusion_vectors.py†L15-L76】
- Choose `ap_method` explicitly when comparing against external reports. For COCO parity use `pycocotools`; for sklearn parity or outlier robustness pick the corresponding option in `Measures.summary()`/`reconstruct()` inputs.【F:kwcoco/metrics/confusion_measures.py†L1300-L1414】
- `DetectionMetrics.from_coco(true, pred)` is the quickest path to score two datasets; you can then call `score_kwcoco`, `score_voc`, or `score_pycocotools` depending on the target benchmark.【F:kwcoco/metrics/detect_metrics.py†L15-L82】【F:kwcoco/metrics/detect_metrics.py†L730-L810】
