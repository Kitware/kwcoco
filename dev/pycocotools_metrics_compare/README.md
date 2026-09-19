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
