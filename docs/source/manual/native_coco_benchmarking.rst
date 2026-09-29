Native COCO Release Benchmarking
================================

The native COCO evaluator has two different evidence requirements:

* semantic conformance to the official ``pycocotools.COCOeval`` reference;
* practical performance and memory behavior on realistic dataset scales.

Do not reduce either requirement to one AP number or one microbenchmark.  The
release benchmark campaign records full tensor parity, repeated timing, peak
process RSS, backend provenance, environment metadata, and workload scale in
one reproducible artifact.

Recommended release campaign
----------------------------

Install the metric dependencies explicitly::

   pip install -e '.[metrics]'

The ``metrics`` extra installs the official reference package and the optional
``kwimage_ext`` acceleration package.  Native scoring remains capability-aware:
an older or unavailable extension does not masquerade as the accelerated grid
backend.

Before publishing performance numbers, run the adversarial conformance suite
with both assignment implementations::

   python dev/pycocotools_metrics_compare/run_phase3_coco_conformance.py \
       --mode=full --assignment-backend=python --assert-parity \
       --output-dpath=_metric_compare_release_python

   python dev/pycocotools_metrics_compare/run_phase3_coco_conformance.py \
       --mode=full --assignment-backend=kwimage_ext --assert-parity \
       --output-dpath=_metric_compare_release_rust

Then run the publication benchmark::

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

This is intentionally a long run.  For iteration, use ``--suite=quick`` and
omit the public evaluator scope or the conformance rerun.

What is measured
----------------

Each backend is benchmarked in a fresh process.  This isolates allocator state
between native and official implementations.  Timing and memory are separate
passes so the RSS sampling thread does not contaminate the headline timing
samples.

The benchmark records:

* warmup and repeated wall-clock samples;
* median/min/max/stdev plus p10/p90 timing;
* throughput in images and predictions per second;
* absolute peak process RSS and peak RSS above the pre-score baseline;
* optional Python-only ``tracemalloc`` peak allocation;
* AP/AP50/AR100 and implementation/backend provenance;
* complete precision, recall, sampled-score, and real-positive tensor parity;
* Python, package, git, CPU, RAM, Rust capability, and thread-environment data.

Peak RSS is the primary memory metric.  It includes Python, NumPy, C-extension,
and Rust allocations visible in the process resident set.  ``tracemalloc`` is
useful for debugging Python object pressure but does not observe all native
allocations and is therefore never presented as total memory usage.

Benchmark scopes
----------------

``core``
    Constructs ``DetectionMetrics`` outside the timed region, then times the
    native or official scorer.  This isolates metric implementation cost.

``evaluator``
    Times the public ``CocoEvaluator`` strict COCO wrapper, including its
    conversion/setup work.  These numbers are closer to user-facing end-to-end
    latency and should accompany core timings in a technical report.

``summary`` versus ``tensor`` output can also be selected independently.  The
``tensor`` mode is the stronger comparison because it validates and returns the
full COCO accumulated evaluation object.

Synthetic scaling workloads
---------------------------

The release runner uses deterministic polygonal kwcoco datasets.  Workloads
span ordinary sparse scenes, dense overlapping candidate graphs, and maxDet
pressure.  The ``publication`` suite combines ``standard`` and ``large``
profiles and reaches thousands of images / tens of thousands of instances.
The synthetic generator includes fractional coordinates, multiple COCO area
ranges, negative images, crowd instances, misses, class errors, and false
positives so results are not limited to perfect AP=1 fixtures.

Synthetic data is useful for controlled scaling but is not a substitute for a
real application corpus.  Run the same harness against project outputs with::

   python dev/pycocotools_metrics_compare/run_release_benchmark.py \
       --truth-coco=/path/to/truth.kwcoco.json \
       --pred-coco=/path/to/predictions.kwcoco.json \
       --scenario-name=my_real_validation_set \
       --backends=official,native-rust,native-python \
       --iou-types=bbox,segm \
       --scopes=core,evaluator \
       --repeat=7 \
       --output-dpath=_metric_release_real

Publication artifacts
---------------------

The runner writes:

``environment.json``
    CPU/RAM/platform, package versions, git state, Rust capabilities, and
    relevant thread environment.

``raw_records.jsonl``
    Per-backend timing and memory measurements including all raw timing samples.

``summary.csv`` / ``summary.json``
    Flat publication-friendly tables with speedup, runtime ratio, RSS ratio,
    throughput, and parity status.

``parity/`` and ``parity_results.json``
    Full accumulated tensor artifacts and native-vs-official numerical diffs
    for every large workload.

``release_benchmark_report.md``
    A generated technical-report starting point with methodology, tables,
    environment, parity status, memory interpretation, and claim guardrails.

``kwcoco_metrics_release_evidence.tar.gz``
    Uploadable bundle containing the complete evidence needed to audit the
    generated report.

Interpretation rules
--------------------

A benchmark should support narrow measured claims, for example:

* "On these workloads and this machine, accelerated kwcoco was 1.7x faster
  than pycocotools";
* "Peak scorer RSS delta was 0.8x the official reference";
* "All populated COCO tensor cells matched within 1e-12 on the benchmark
  workloads, and the adversarial conformance suite also passed."

Avoid turning those measurements into universal statements.  Always publish
package revisions, CPU/RAM, workload counts, output mode, and resolved backend
provenance beside performance results.  ``pycocotools.COCOeval`` remains the
official external reference implementation even when the native path matches
it and is faster on the measured workloads.
