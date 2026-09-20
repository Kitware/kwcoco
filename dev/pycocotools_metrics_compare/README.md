# pycocotools / kwcoco metric comparison

This directory is a development comparison suite for understanding the
behavioral and performance differences between the native kwcoco metric stack,
`pycocotools.COCOeval`, and the mask-IoU C kernels available through kwimage.

The goal is not to declare one implementation universally better.  The goal is
to identify which layer is responsible for each difference:

1. geometry conversion / rasterization;
2. pairwise overlap kernels;
3. truth/prediction assignment;
4. confusion-vector construction;
5. AP / AR accumulation and interpolation.

This matters because kwcoco has capabilities that COCOeval does not expose
through the same abstraction: explicit assignments, `ConfusionVectors`,
`Measures`, alternate AP definitions, threshold diagnostics, weights, class
hierarchies, and reuse of the same assignments for multiple analyses.  At the
same time, official COCOeval is an important ecosystem reference and its mask
kernels are highly optimized.

## Current matrix

Phase 1 intentionally measures the cells that are already implemented without
changing native assignment semantics:

| geometry | assignment | accumulation | status |
| --- | --- | --- | --- |
| bbox | kwcoco | kwcoco `Measures` | available |
| bbox | COCOeval | kwcoco `Measures` | available |
| bbox | COCOeval | COCOeval | available |
| mask | COCOeval | kwcoco `Measures` | available |
| mask | COCOeval | COCOeval | available |
| mask | kwcoco | kwcoco `Measures` | **Phase 2 target** |

The official mask path is therefore a reference implementation, not a
replacement for the native confusion-vector architecture.

## Mask-kernel comparison

`kwimage.Mask` prefers the optional `kwimage_ext` C/Cython backend and falls
back to `pycocotools` when that backend is unavailable.  The kwimage extension
was historically derived from an older COCO mask implementation and stabilized
inside the Kitware stack.  The current pycocotools implementation may have
changed since that fork.

The suite sends the *same compressed RLE objects* to both low-level backends
and compares:

* pairwise IoU values;
* exact / near-exact agreement;
* wall-clock runtime.

Backend identity is explicit in the report.  `pycocotools` and `kwimage_ext`
each receive their own row with `status=measured` or `status=skipped`.  If
`kwimage_ext` is not installed, the suite records that fact and skips the
`kwimage_ext` measurement/comparison; it never silently calls pycocotools and
labels the result as a kwimage-extension timing.  The environment section also
records the expected kwimage mask-dispatch choice (`kwimage_ext`,
`pycocotools-fallback`, or unavailable).

This isolates the kernel implementation from polygon rasterization and metric
accumulation.


## Dense saliency / historical segmentation accumulation

The suite also exercises the confusion accumulation architecture used by
`segmentation_metrics.py`.  It converts the synthetic instance predictions into
a dense saliency score raster (maximum instance score at each covered pixel)
and compares:

* one direct global `BinaryConfusionVectors(...).measures()` reduction with
  unquantized scores;
* the same direct global reduction after 1024-bin score quantization; and
* per-image `BinaryConfusionVectors -> Measures` reductions combined through
  `MeasureCombiner` with the same 1024-bin strategy used by the scalable
  segmentation evaluator.

This separates error introduced by score quantization from error introduced by
the bounded-memory measure combination itself.

This is intentionally reported as a *different metric family* from COCO
instance-mask AP.  The comparison is there to protect and measure the historical
Kitware/ShitSpotter pixel-scoring path while instance metrics are added, not to
pretend pixel AP and instance AP should have the same value.

## Synthetic data

The suite deliberately reuses kwcoco's own test infrastructure:

* `CocoDataset.demo('shapesN', ...)` generates truth datasets at controlled
  sizes;
* `kwcoco.demo.perterb.perterb_coco()` introduces controlled box, class, score,
  false-positive, and false-negative perturbations;
* a local mask-only translation perturbation is included to construct cases
  where bbox and mask quality intentionally disagree without changing package
  behavior.

This makes it possible to scale the same behavioral cases from a smoke test to
larger timing experiments.

## Usage

Start with a short correctness-oriented run:

```bash
python dev/pycocotools_metrics_compare/compare_metrics.py \
    --profile=smoke \
    --output=metrics_compare_smoke.json
```

Then run larger timing profiles in a normal kwcoco development environment:

```bash
python dev/pycocotools_metrics_compare/compare_metrics.py \
    --profile=small \
    --repeat=3 \
    --output=metrics_compare_small.json

python dev/pycocotools_metrics_compare/compare_metrics.py \
    --profile=medium \
    --repeat=5 \
    --output=metrics_compare_medium.json
```

The script records package versions and backend availability in every report.
A missing optional `kwimage_ext` backend is a normal, explicit benchmark state:
the pycocotools rows still run and the extension row is marked skipped.  For a
complete kernel comparison, repeat the same profile in an environment where
`kwimage_ext` is installed and retain both reports.  Do not compare timings
from different environments without also comparing the environment metadata.

## How to interpret the report

For bbox AP50 the report exposes three values:

* `native_native_mAP50`: kwcoco assignment followed by kwcoco `Measures` with
  `ap_method='pycocotools'`;
* `official_native_mAP50`: assignments reconstructed from COCOeval, then fed
  through the same kwcoco `Measures` accumulator;
* `official_official_AP50`: the official COCOeval AP50 result.

This decomposition is important:

* native/native vs official/native isolates assignment differences;
* official/native vs official/official isolates accumulation/interpolation
  differences;
* native/native vs official/official is the end-to-end user-visible delta.

The same official/native vs official/official decomposition is available for
instance masks.  Native mask assignment is intentionally not implemented by
this Phase 1 suite.

The report also compares the explicit matched `(gid, prediction-index,
truth-index)` sets at IoU=0.5 for bbox evaluation.  This makes assignment
differences inspectable instead of inferring them from a final scalar AP.

## Planned Phase 2

Do not implement Phase 2 until Phase 1 has been run on representative scales.
The intended next step is to generalize kwcoco assignment so the pairwise
overlap provider can be bbox IoU or instance-mask IoU while preserving the
existing `ConfusionVectors -> Measures` pipeline.

Phase 1 should answer which mask overlap backend to use by default and whether
there are compatibility/performance reasons to retain more than one backend.

## Phase-1 orchestration bundle

`run_phase1_suite.py` is the next measurement gate.  It leaves production
metric semantics unchanged and adds the experiments needed before native mask
assignment is designed.

It compares the current native assignment implementation against the
experimental `assignment_v2.AssignmentCandidateIndex` over the complete
COCO-style IoU threshold set (`0.50:0.05:0.95`).  Both the dense v2 candidate
builder and its automatic sparse-grid path are measured.  The report preserves
row-level assignment differences as `(gid, tx, px, true-class, pred-class)`
multisets, not only scalar AP differences.

It also runs deterministic hand-built cases for:

* an IoU exactly equal to 0.50;
* equal-score duplicate detections;
* one prediction competing for two overlapping truths;
* empty prediction and empty truth inputs;
* a polygon-with-hole truth against a filled prediction.

Three scaling sweeps are included:

* image count with `CocoDataset.demo('shapesN')`;
* image resolution with the same shapes generator;
* object density using `CocoDataset.demo('vidshapes1', num_tracks=...)`.

The density sweep uses `render=False`; scoring depends only on KWCoco geometry,
not image bytes.  Prediction perturbations still use the package's existing
`perterb_coco()` machinery.

Every experiment is caught independently.  A failure is serialized with its
traceback and the suite continues, which makes a partial long run useful for
analysis instead of losing all later measurements.

### Quick measurement gate

Run the focused unit tests first, then the quick suite:

```bash
pytest \
    tests/test_coco_instance_metrics.py \
    tests/test_metrics_assignment_v2.py \
    tests/test_metrics_phase1_suite.py

python dev/pycocotools_metrics_compare/run_phase1_suite.py \
    --mode=quick \
    --output-dpath=_metric_compare_phase1_quick
```

The command writes one uploadable artifact:

```text
_metric_compare_phase1_quick/kwcoco_metrics_phase1_quick.tar.gz
```

Upload that tarball for analysis.  It contains the full JSON report, a compact
Markdown status/timing table, and Git/source-state metadata.

### Full scaling run

After the quick bundle is clean enough to interpret, run:

```bash
python dev/pycocotools_metrics_compare/run_phase1_suite.py \
    --mode=full \
    --output-dpath=_metric_compare_phase1_full
```

The output to upload is:

```text
_metric_compare_phase1_full/kwcoco_metrics_phase1_full.tar.gz
```

The full mode increases image-count, resolution, and object-density sweeps and
uses repeated timings.  Keep quick/full reports from a no-`kwimage_ext`
environment, and repeat the same command unchanged in an environment with
`kwimage_ext` when available.  Backend availability is part of the report, so
those runs remain distinguishable.

### Interpretation gate before Phase 2

Do not promote `assignment_v2` or implement native mask assignment solely
because it is newer or faster.  First inspect:

1. where v1 and v2 assignment row multisets disagree;
2. whether those disagreements are score-tie / hierarchy / ignore-policy
   semantics or defects;
3. how v1 and v2 scale with images and object density;
4. whether v2's sparse candidate builder changes any assignments;
5. where native-vs-official deltas enter: assignment or accumulation;
6. whether `kwimage_ext` and current pycocotools mask kernels agree numerically
   and which is faster on the same RLE inputs.

Only then should Phase 2 generalize the reusable candidate graph around a
bbox/mask overlap provider.

## Phase-1.5: acceleration-backend characterization

The initial Phase-1 runs showed that installing `kwimage_ext` can materially
change native assignment throughput, especially for dense box IoU matrices.
Before changing assignment architecture, run the low-level backend suite:

```bash
python dev/pycocotools_metrics_compare/run_phase15_suite.py \
    --mode=quick \
    --output-dpath=_metric_compare_phase15_quick
```

Upload:

```text
_metric_compare_phase15_quick/kwcoco_metrics_phase15_quick.tar.gz
```

Run the exact same command in each relevant environment: no extension, legacy
Cython extension, and Rust extension.  The report records which public backend
actually dispatched and separately probes `kwimage_ext._rust` when available.

Phase-1.5 measures:

* NumPy, kwimage automatic, public kwimage_ext, and direct Rust box-IoU paths;
* pycocotools, public kwimage_ext, and direct Rust COCO-RLE IoU paths;
* rectangle, fragmented, and sparse/noisy RLE complexity;
* adaptive repetitions so sub-millisecond wrapper overhead is not mistaken for
  kernel throughput;
* exact/numerical parity and threshold-decision parity at IoU 0.50/0.75/0.95;
* RLE compressed-byte, decode, area, and bbox parity;
* inherited edge cases: one-dimensional bboxes, `iscrowd` length validation,
  Nx2 quadrilaterals, and degenerate polygons.

Only after the same Phase-1.5 profile is available from the legacy and Rust
backends should native mask assignment choose its production overlap backend.

## Phase 2: native instance-mask assignment

`run_phase2_native_mask.py` exercises the experimental geometry-independent
`assignment_v2` path with `iou_type='segm'`.  It compares native mask
assignment + native `Measures`, COCOeval assignment + native `Measures`, and
full COCOeval accumulation at the same controlled perturbation cases.

The native segmentation path intentionally requires explicit image dimensions
and currently rejects crowd / ignore-region semantics.  This keeps the first
measurement surface honest while those semantics are designed separately.

```bash
python dev/pycocotools_metrics_compare/run_phase2_native_mask.py \
    --mode=quick \
    --output-dpath=_metric_compare_phase2_native_mask
```

## Phase 2.1: rasterization and threshold-boundary localization

The first native-mask run showed a small number of assignment differences even
when the RLE-IoU kernel itself was numerically identical to pycocotools.  The
next comparison therefore varies polygon rasterization and IoU threshold
boundary independently:

```bash
python dev/pycocotools_metrics_compare/run_phase21_rasterization.py \
    --mode=quick \
    --output-dpath=_metric_compare_phase21_rasterization
```

Upload:

```text
_metric_compare_phase21_rasterization/kwcoco_metrics_phase21_rasterization_quick.tar.gz
```

The native variants are:

* `kwimage_strict`: historical `Segmentation.to_mask()` rasterization and
  `IoU > threshold`;
* `coco_rust_strict`: COCO `frPoly` rasterization through `kwimage_ext` Rust,
  but the same strict threshold convention;
* `coco_rust_inclusive`: COCO/Rust rasterization and `IoU >= threshold`;
* `coco_pycoco_inclusive`: the same native matcher, but pycocotools polygon
  rasterization as an independent codec reference.

The report also compares Rust- and pycocotools-rasterized RLE bytes one
annotation at a time and records pairwise IoUs for every assignment mismatch.
Do not implement crowd/ignore semantics until this layer is localized.


## Phase 2.2: crowd and ignore semantics

Native instance-mask assignment now separates two concepts that are easy to
conflate:

* **COCO crowd semantics** are annotation-level behavior. Crowd truth masks use
  intersection-over-detection-area for overlap, may match multiple detections,
  and matched detections receive zero metric weight. COCO-compatible native
  mask assignment defaults to COCO polygon rasterization and an inclusive IoU
  threshold boundary.
* **kwcoco ignore classes** are region-suppression semantics. Truth annotations
  whose class is configured as an ignore class are unioned in mask space;
  predictions whose intersection-over-prediction-area is strictly greater than
  the ignore threshold are removed from scoring. This remains distinct from
  COCO crowd behavior.

Current official pycocotools bbox/segm evaluation derives annotation-level
ignore from ``iscrowd``. The experimental native path therefore exposes
``annotation_ignore_policy='coco'`` for reference parity and
``annotation_ignore_policy='kwcoco'`` when an explicit truth ``ignore`` column
should also carry zero metric weight.

Run the focused comparison with:

```bash
python dev/pycocotools_metrics_compare/run_phase22_crowd_ignore.py \
    --output-dpath=_metric_compare_phase22_crowd_ignore
```

## Phase 2.3: dataset-level native COCO-shaped mask summary

Phase 2.3 adds assignment-time area ranges, COCO-style per-image/per-category
`maxDets`, and a native dataset summary built from the reusable mask candidate
indexes. The native summary remains explicitly distinct from official
`COCOeval`: assignment is native `assignment_v2`, while the COCO-shaped
summary uses a narrow native 101-recall-point accumulator over the explicit
assignment rows.  Generic `ConfusionVectors -> Measures` remains available
for kwcoco's broader metric analyses and is not globally changed for this
compatibility target.

The dedicated summary accumulator is intentional.  COCO accumulation requires
a stable descending mergesort for equal detection scores.  Historical generic
binary-curve code has different tie ordering, so routing the COCO-shaped
summary through it can change AP even when assignment is identical.  Keeping
the rule local avoids changing established `Measures` behavior.

Run the focused tests first:

```bash
pytest -q \
    tests/test_coco_instance_metrics.py \
    tests/test_metrics_assignment_v2.py \
    tests/test_metrics_native_mask_v2.py \
    tests/test_metrics_native_mask_crowd_ignore.py \
    tests/test_metrics_native_coco_v2.py
```

Then run the deterministic comparison suite:

```bash
python dev/pycocotools_metrics_compare/run_phase23_native_coco.py \
    --mode=quick \
    --assert-parity \
    --output-dpath=_metric_compare_phase23_native_coco
```

Upload:

```text
_metric_compare_phase23_native_coco/kwcoco_metrics_phase23_native_coco_quick.tar.gz
```

The report includes the 12 named native and official AP/AR values, absolute
summary deltas, row-level assignment differences at IoU 0.50/0.75/0.95,
backend provenance, and candidate-build / assignment / accumulation timing.
The adversarial corpus includes area-priority matching, `maxDets`, a negative
image, crowd behavior, equal-score accumulation order, all three area bins,
100-detection truncation, prediction-only categories, and a nontrivial
precision/recall curve in addition to the existing perturbation cases.
