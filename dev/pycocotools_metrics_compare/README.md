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
Native and official records also expose a shared versioned summary schema plus
an explicit implementation / ``official`` distinction.  ``--assert-parity``
checks that contract as well as requiring resolved native overlap, raster, and
area backend provenance, so a passing evidence bundle validates the public
result identity in addition to AP/AR and row assignment parity.
The adversarial corpus includes area-priority matching, `maxDets`, a negative
image, crowd behavior, equal-score accumulation order, all three area bins,
100-detection truncation, prediction-only categories, arbitrary kwcoco truth
weights, a literal category named `background`, and a nontrivial precision/
recall curve in addition to the existing perturbation cases.  The truth-weight
case protects the COCO contract: native COCO scoring deliberately ignores
kwcoco's non-COCO weighting extension while preserving assignment-generated
zero weight for crowd / area / ignore semantics.

Before treating Phase 2.3 as production-ready, run both the focused pytest set
above and the full parity profile:

```bash
python dev/pycocotools_metrics_compare/run_phase23_native_coco.py \
    --mode=full \
    --assert-parity \
    --output-dpath=_metric_compare_phase23_native_coco_full
```

Generated ``_metric_compare_*`` directories and result tarballs are evidence,
not source files; do not commit them.

## Phase 2.4 / Phase 3: tensor conformance and explicit native bbox COCO

Phase 2.4 raises the differential target from the 12 standard summary scalars
to the full COCO accumulation state. Native and official evaluation can expose
precision ``[T,R,K,A,M]``, recall ``[T,K,A,M]``, and sampled score
``[T,R,K,A,M]`` tensors. The conformance runner requires both the populated
values and the ``-1`` absent-category masks to agree.

Phase 3 adds an explicit native COCO bbox profile without changing historical
``CocoEvaluator.evaluate()`` or assignment-v2's generic bbox default. The
profile implements COCO continuous-coordinate bbox overlap, crowd
intersection-over-detection-area, reusable/ignored crowd matches, inclusive
thresholds, area-priority matching, category-scoped ``maxDets``, stable score
ties, and COCO equal-IoU truth tie behavior.

Run the focused tests:

```bash
pytest -q \
    tests/test_coco_instance_metrics.py \
    tests/test_metrics_assignment_v2.py \
    tests/test_metrics_native_mask_v2.py \
    tests/test_metrics_native_mask_crowd_ignore.py \
    tests/test_metrics_native_coco_v2.py
```

Then run the new two-geometry differential suite:

```bash
python dev/pycocotools_metrics_compare/run_phase3_coco_conformance.py \
    --mode=quick \
    --assert-parity \
    --output-dpath=_metric_compare_phase3_coco_conformance_quick

python dev/pycocotools_metrics_compare/run_phase3_coco_conformance.py \
    --mode=full \
    --assert-parity \
    --output-dpath=_metric_compare_phase3_coco_conformance_full
```

A larger stress profile is also available when runtime / memory scaling is the
question:

```bash
python dev/pycocotools_metrics_compare/run_phase3_coco_conformance.py \
    --mode=stress \
    --assert-parity \
    --output-dpath=_metric_compare_phase3_coco_conformance_stress
```

The runner compares standard summary values, assignments at IoU
0.50/0.75/0.95, every populated tensor cell, tensor parameter metadata
(including normalized image ids), absent-slice masks, backend provenance, and
native/reference wall time for both
``bbox`` and ``segm``. Selected fixtures are additionally evaluated on a
nonstandard IoU / recall / area / maxDet grid and a repeated/reordered image
subset, so tensor parity is not inferred only from the familiar standard
``10 x 101 x K x 4 x 3`` shape. The adversarial corpus also includes valid
negative detection scores and kwcoco-only explicit-ignore metadata. Geometry
fixtures include fractional polygons, rich polygons with holes (exact RLE strict
boundary), and compressed RLE input.
Generated ``_metric_compare_*`` directories remain evidence artifacts and must
not be committed.

## Native COCO performance profiling

The Phase-3 conformance timing is intentionally part of a correctness runner:
it uses a single wall-time sample per case and its stress profile primarily
scales image count / canvas size.  Use `profile_native_coco.py` before making
performance changes.  It builds `DetectionMetrics` outside the timed region,
warms both scorers, repeats unprofiled timing, checks native/reference parity,
and can run separate cProfile, line-profiler, and tracemalloc diagnostics.

The current full conformance evidence suggests the first performance campaign
should concentrate on assignment and accumulation rather than mask geometry
construction.  In the recorded Phase-3 full run, the median native phase
fractions were approximately:

```text
                 candidate build     assignment     accumulation
bbox                  2.3%              52.7%           38.0%
segm                  3.6%              51.0%           37.3%
```

Treat these as directional because the conformance runner records one wall-time
sample per case.  The profiling runner uses warmups and repeats for decisions.

Start with the same generated workload used by conformance:

```bash
python dev/pycocotools_metrics_compare/profile_native_coco.py \
    --scenario perfect \
    --dataset-mode full \
    --iou-type bbox \
    --with-eval \
    --warmup 2 \
    --repeat 7 \
    --output-dpath=_metric_profile_native_coco

python dev/pycocotools_metrics_compare/profile_native_coco.py \
    --scenario perfect \
    --dataset-mode full \
    --iou-type segm \
    --with-eval \
    --warmup 2 \
    --repeat 7 \
    --output-dpath=_metric_profile_native_coco
```

Two synthetic pressure cases probe behavior the conformance corpus does not
stress strongly:

```bash
# Many predictions in one category. COCO only needs the top largest maxDet
# (normally 100). The optimized native path applies that cap before candidate
# geometry, making this a regression/performance pressure case.
python dev/pycocotools_metrics_compare/profile_native_coco.py \
    --scenario maxdet-pressure \
    --iou-type bbox \
    --with-eval \
    --num-objects 1000 \
    --warmup 2 \
    --repeat 7 \
    --profile cprofile \
    --output-dpath=_metric_profile_native_coco

# Long mutually-overlapping candidate lists. This emphasizes the Python greedy
# assignment loop rather than spatial rejection.
python dev/pycocotools_metrics_compare/profile_native_coco.py \
    --scenario candidate-heavy \
    --iou-type bbox \
    --with-eval \
    --num-objects 400 \
    --warmup 2 \
    --repeat 7 \
    --profile cprofile \
    --output-dpath=_metric_profile_native_coco
```

For line-level attribution, install the optional profiler only in the
benchmarking environment:

```bash
uv pip install line-profiler

python dev/pycocotools_metrics_compare/profile_native_coco.py \
    --scenario perfect \
    --dataset-mode full \
    --iou-type bbox \
    --with-eval \
    --warmup 1 \
    --repeat 5 \
    --profile line \
    --output-dpath=_metric_profile_native_coco
```

The line profile is focused on `score_native_coco`, candidate-index build /
assignment, COCO bbox overlap / sparse candidate generation, native COCO
accumulation, and mask raster/area conversion.  Timing repeats are always run
without profiler instrumentation; profiler and tracemalloc runs are separate,
so their overhead does not contaminate the reported benchmark median.

The first profile-driven optimization pass implemented the two strongest
hypotheses below: largest-maxDet reuse and scalar scanning of the already
overlap-sorted candidate lists. It also caps prediction geometry at the largest
requested per-category maxDet before candidate construction. Standard
``with_eval=True`` therefore performs four area-base assignments per IoU/image
instead of twelve area×maxDet assignments, reducing the standard 32-image,
10-IoU case from 3840 to 1280 calls to ``AssignmentCandidateIndex.assign()``.
The profiler reports both the logical 12 tensor slices and the actual base
assignment/call counts so this optimization remains observable.

After the pure-Python optimization pass, the recorded repeated-timing
baseline was approximately ``2.649x`` COCOeval for the 32-image perfect bbox
profile and ``1.750x`` for perfect segmentation, while the maxDet-pressure and
candidate-heavy bbox workloads were already faster than COCOeval at ``0.238x``
and ``0.310x`` respectively.  This is why the compiled pass targets the many
small threshold assignments in realistic multi-image evaluation rather than
moving overlap geometry into Rust.

A second profile-driven pass adds an optional **batched Rust greedy
assignment kernel** in ``kwimage_ext``.  ``AssignmentCandidateIndex.assign_many``
can flatten its already-sorted sparse candidate lists to a CSR representation
once and evaluate the complete IoU threshold family in one PyO3 call.  The
kernel deliberately does not own geometry, category compatibility, area policy,
row materialization, or accumulation.  The Python matcher remains the semantic
reference and fallback.

Select the backend explicitly when measuring:

```bash
python dev/pycocotools_metrics_compare/profile_native_coco.py \
    --scenario perfect --dataset-mode full --iou-type bbox --with-eval \
    --assignment-backend python --warmup 2 --repeat 7 \
    --output-dpath=_metric_profile_native_coco_python

python dev/pycocotools_metrics_compare/profile_native_coco.py \
    --scenario perfect --dataset-mode full --iou-type bbox --with-eval \
    --assignment-backend kwimage_ext --warmup 2 --repeat 7 \
    --output-dpath=_metric_profile_native_coco_rust
```

The profiler distinguishes logical threshold queries from compiled backend
batches.  For the standard full tensor, the Python path still represents 1280
threshold assignments on the 32-image perfect profile, while Rust receives ten
thresholds at once for each image/area base.  Force both backends through the
full conformance runner before accepting a speedup:

```bash
python dev/pycocotools_metrics_compare/run_phase3_coco_conformance.py \
    --mode=full --assignment-backend=python --assert-parity \
    --output-dpath=_metric_compare_phase3_assignment_python

python dev/pycocotools_metrics_compare/run_phase3_coco_conformance.py \
    --mode=full --assignment-backend=kwimage_ext --assert-parity \
    --output-dpath=_metric_compare_phase3_assignment_rust
```

The remaining performance hypotheses after the batched kernel are:

1. **row materialization:** COCO accumulation consumes only true class,
   predicted class, score, and ignore/weight state, while inspectable assignment
   rows retain IoU, truth/pred indexes, and rank as well.
2. **accumulator reuse:** class masks and stable score sorting are repeated for
   each IoU/area/maxDet slice.  The full-tensor path may be able to share score
   order and other static state across thresholds.
3. **Python/Rust boundary cost:** if small images do not benefit from Rust,
   measure CSR flattening, NumPy conversion, and row materialization separately
   before moving additional evaluator policy into compiled code.

Do not optimize candidate geometry first merely because it is conceptually
expensive.  The measured full corpus currently attributes a small median
fraction of native runtime to candidate construction; confirm a workload where
that changes before complicating the overlap backend.

## Historical release-to-release compatibility and speed

The release benchmark above answers whether the **current** strict native COCO
implementation matches and outperforms the current official pycocotools
reference.  It does not answer a different release-engineering question:
whether upgrading KWCoco changes the long-standing
``CocoEvaluator.evaluate()`` result or its runtime.

Use ``run_version_timeline.py`` for that question.  The default run compares
the released ``v0.9.0`` tag with the current worktree.  It generates one
version-independent bbox truth/prediction pair using only the Python standard
library, exports historical refs directly from git, and executes each KWCoco
source tree in a fresh subprocess with the **same Python executable and shared
dependency environment**.  Only the KWCoco source path changes.

```bash
# v0.9.0 imports scriptconfig in CocoEvaluator, while the current worktree uses
# kwconf.  Keep both available in the common benchmark environment.
python -m pip install scriptconfig

python dev/pycocotools_metrics_compare/run_version_timeline.py \
    --refs=v0.9.0,current \
    --baseline=v0.9.0 \
    --warmup=2 --repeat=7 \
    --output-dpath=_metric_version_timeline_090_to_091
```

The generated report includes:

* repeated prepared-evaluator and end-to-end wall time for the historical public evaluator;
* speedup / slowdown relative to the chosen baseline;
* a recursive comparison of the complete serialized ``nocls_measures`` and
  ``ovr_measures`` trees, including curves, thresholds, and counts rather than
  only one AP scalar;
* exact dependency versions and the imported KWCoco source path for every run;
* a capability matrix showing when named official COCO statistics, segmentation
  COCO scoring, full evaluation tensors, native COCO scoring, and compiled
  assignment backends became available.

More releases can be included in one run, for example
``--refs=v0.8.9,v0.9.0,current``.  A historical source that cannot run in the
common dependency environment is reported as an error instead of silently
switching environments.  This is intentional: dependency drift would confound
a source-code runtime comparison.  If a historical environment genuinely
requires different dependencies, retain that as a separate compatibility run
and do not present its timing ratio as a controlled source-only speedup.

The historical runner and the current release benchmark are complementary:

* ``run_version_timeline.py``: old KWCoco versus new KWCoco, historical public
  contract, frozen bbox workload, same dependencies;
* ``run_release_benchmark.py``: current official pycocotools versus current
  native Python/Rust, bbox + segmentation, full tensor parity, memory, and
  scaling / real-dataset evidence.

## Release-scale timing, memory, and publication evidence

Once Phase-3 tensor conformance is green, use `run_release_benchmark.py` for
release evidence rather than quoting the small conformance timings.  It runs
each backend in an isolated process, measures timing and peak RSS separately,
checks complete COCO tensor parity on the scaling workloads, records hardware /
package / git provenance, and generates JSON, CSV, Markdown, and an uploadable
tar bundle.

A short smoke run is:

```bash
python dev/pycocotools_metrics_compare/run_release_benchmark.py \
    --suite=quick \
    --backends=official,native-rust,native-python \
    --iou-types=bbox,segm \
    --output-dpath=_metric_release_quick
```

The publication-oriented campaign is intentionally longer:

```bash
python dev/pycocotools_metrics_compare/run_release_benchmark.py \
    --suite=publication \
    --backends=official,native-rust,native-python \
    --parity-backends=native-rust,native-python \
    --iou-types=bbox,segm \
    --scopes=core,evaluator \
    --eval-modes=tensor \
    --warmup=2 --repeat=7 --memory-repeat=3 \
    --conformance-mode=full \
    --output-dpath=_metric_release_publication
```

Use `--truth-coco` and `--pred-coco` to repeat the same methodology on a real
project validation set.  Peak process RSS is the primary memory measure because
it includes native C/Rust allocations; optional `--python-allocations` adds
tracemalloc diagnostics but is not a substitute for RSS.
