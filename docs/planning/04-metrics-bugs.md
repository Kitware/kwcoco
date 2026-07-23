# 04 — Metrics & evaluator correctness bugs

Status: TODO
Risk: medium (numerical outputs change; every fix needs a pin-down test)
Prereqs: 01 (assignment_v2 must be committed first)

Files: `kwcoco/metrics/*`, `kwcoco/coco_evaluator.py`. Read
`kwcoco/metrics/AGENTS.md` before starting. Items `[verified]` were confirmed
by executing code (NumPy 2.4.4 in the audit environment).

## A. Environment-breaking / crashing bugs

### Task 4.1 [verified] — `np.trapz` removed in NumPy 2.0

`kwcoco/metrics/functional.py:197` (`_average_precision`) calls `np.trapz`,
which no longer exists (verified on NumPy 2.4.4). Called from
`detect_metrics.py:597-599` (`score_kwant`). Use the same shim already used in
`confusion_measures.py:1518-1521` (`scipy.integrate.trapezoid` /
`np.trapezoid` fallback). Add a smoke test that `score_kwant` runs on demo
data under NumPy 2.x.

### Task 4.2 [verified] — `OneVersusRestMeasureCombiner.__init__` typo

`confusion_measures.py:1037`: `self.growth = precision` (should be `growth`).
`growth='max'` is silently dropped; `precision=6` becomes growth and then
raises `ValueError('dont use precision with growth')` downstream. One-line
fix + a test asserting both attributes after construction and that
`combine()` honors `growth`.

### Task 4.3 — `score_pycocotools` inverted iou_thresholds condition

`detect_metrics.py:895-899`:

```python
if len(iou_thresholds) != len(evaler.params.iouThrs) or np.allclose(
    iou_thresholds, evaler.params.iouThrs):
```

should be `... or not np.allclose(...)`. Consequences: (a) custom thresholds
with the same length as the pycocotools default (10) are silently ignored and
`with_confusion` later IndexErrors at `:1435-1437`; (b) passing exactly the
default thresholds skips `summarize()` so `pct_info['mAP']` is never set.
Fix + tests for both trigger cases (a same-length-but-different list; the
exact default list).

### Task 4.4 — `pycocotools_confusion_vectors` builds output inside the loop

`detect_metrics.py:1501-1524`: the conversion block, `DataFrameArray`, and
`ConfusionVectors` construction are indented inside
`for info in evaler.evalImgs:` — O(n²) reconversion, and `cfsn_vecs` is
unbound (→ `UnboundLocalError`) if `evalImgs` is empty. De-indent by one
level; return an empty-but-valid `ConfusionVectors` for the empty case.

### Task 4.5 — Segmentation metrics swap true/pred image ids

`segmentation_metrics.py`: `associate_images` yields `match_gids1` (true-side
ids) and `match_gids2` (pred-side ids), but job submission does the opposite:
`pred_coco_img = pred_coco.coco_image(gid1)` and
`true_coco_img = true_coco.coco_image(gid2)` (`:1650-1651` videos,
`:1688-1689` loose images) while `true_dets = true_coco.annots(gid=gid1)`
(`:1652`, `:1690`). Works only when ids coincide; scores wrong pairs or
KeyErrors otherwise. Swap to `pred_coco.coco_image(gid2)` /
`true_coco.coco_image(gid1)`. Regression test: two datasets with the same
image *names* but different ids (offset one side's ids) → evaluation succeeds
and matches the same-ids result.

### Task 4.6 — Threshold binning IndexError and wrong rounding direction

`segmentation_metrics.py:695-697` and `:789-792`:
`np.searchsorted(self.left_bin_edges, pred_score)` is unclipped — any score
above the last edge (e.g. 1.0000001) IndexErrors. Also `side='left'` rounds
interior scores *up*, contradicting the "round scores down" comment. Use
`np.searchsorted(edges, score, side='right') - 1` clipped to `[0, n-1]`
(the proof-of-concept copy at `:2308-2311` already clips). Test with scores
at/above the top bin edge.

## B. Silent-corruption bugs

### Task 4.7 [verified] — `binarize_ovr` mutates `cfsn_vecs.probs` in place

`confusion_vectors.py:560-565`: `pred_score = cfsn_vecs.probs[:, cx]` is a
view; the subsequent `-inf` assignment writes into the stored probs matrix,
poisoning later `coarsen()`, serialization, and user analysis. Fix:
`.copy()`. Test: assert `probs` unchanged after `binarize_ovr()`.

### Task 4.8 [verified] — `_binary_clf_curve2` mutates the caller's scores

`sklearn_alts.py:166-169`: `y_score[nan_flags] = -np.inf` mutates the input
array (1-D `column_or_1d` doesn't copy). Copy when NaNs are present. Test:
NaN-containing scores unchanged after `measures()`.

### Task 4.9 — `Measures.combine` -inf handling clobbers real counts

`confusion_measures.py:601-604` sets `new_fp[-1] = np.inf; new_tn[-1] = -np.inf`
when any input had a `-inf` threshold (which every detection eval has); the
code comment admits "Not sure if this is right". **⚠️ DECISION** — this needs
a definitive design. Recommended approach: add the regression test *first*
(combined per-image measures ≈ measures of the concatenated confusion vectors
within binning tolerance, including the `-inf` rows), then adjust the combine
logic until it passes; keep `trunc_auc` consistent.

### Task 4.10 — Stabilization pad inflates `realpos_total`/`realneg_total`

`confusion_vectors.py:991-1010`: `_stabalize_data` pads ~7 synthetic rows
*before* `realpos_total`/`realneg_total` are computed, biasing AP/AUC for
classes with < `stabalize_thresh=7` rows (`nsupport` is corrected; these are
not). Compute the totals from the unpadded arrays. Pin-down test with a
3-sample class.

### Task 4.11 — `VOC_Metrics.score` omits truth-only classes, inflating mAP

`voc_metrics.py:94-123`: the loop covers only classes with ≥1 prediction; a
class with truth but zero predictions contributes nothing to mAP instead of
AP≈0. Iterate the union of truth and prediction classes, emitting an AP-0
record (with `realpos_total`) for prediction-less classes with truth. Test:
2-class dataset where the detector never fires on class B → mAP halves.
Also fix the degenerate-branch dead code at `:224-231` (`ap = np.nan`
unconditionally overwritten by `0.0`; no-truth/no-detection should stay NaN)
and guard the `npos == 0` divide at `:358`.

### Task 4.12 — `coco_evaluator` area-range and result-sharing bugs

Three related bugs in `coco_evaluator.py`:

1. `:885-886` — `dmet_area_weights(...)` is called without forwarding
   `use_area_attr`, so the documented pycocotools-compatible `'try'` behavior
   (config `:158-169`) never runs (`use_area_attr` defaults False at `:961`).
2. `:996-1004` — once (1) is fixed, `true_area` is unbound (`NameError`) when
   the `'try'` lookup fails; mirror the `pred_area` fallback at `:1009-1016`.
3. `:883-910` — every `CocoSingleResult` shares one `cfsn_vecs` whose
   `data['weight']` is overwritten per area range, so all but the last
   result's stored vectors have mismatched weights (breaks `dump_figures`
   confusion matrices at `:1315-1341`). Store a per-result copy of the
   weights (or of the vectors).

- [ ] Fix all three; test that small/medium/large area-range results differ
      when 'area' attrs disagree with bbox area, and that each stored
      result's weights match its own range.

### Task 4.13 — Smaller evaluator/metrics fixes

- [ ] `coco_evaluator.py:379-380` — `_init` does `true_extra['coco_dset']`
      unconditionally but `_coerce_dets` returns no such key for the
      documented `Dict[gid, Detections]` input (`:626-635`). Guard with
      `.get`; test the dict-of-Detections path.
- [ ] `coco_evaluator.py:951-952` — raw `pct_stats` list inserted into
      `resdata` breaks `CocoResults.from_json` round-trip (`:1111-1117`).
      Make it JSON-safe; add a `force_pycocoutils` round-trip test.
- [ ] `confusion_measures.py:1480-1488` — OUTLIER_AP block: dead assignment,
      bare `except Exception: pass` that later causes `KeyError` for
      `ap_method='outlier'`, and reliance on variables leaked from the SKLISH
      block. Restructure so each ap-method block is self-contained and the
      all-outlier case sets `outlier_ap = np.nan` explicitly.
- [ ] `sklearn_alts.py:63-70` — `class_accuracy_from_confusion` applies
      `nan_to_num` after `.mean()`, zeroing the whole result if any class has
      no support. No in-repo callers: fix (`np.nanmean`) or delete.
      **⚠️ DECISION** — recommended: delete (doc 09 policy), it is unused.
- [ ] `detect_metrics.py:1568-1582` — `eval_detections_cli` reads
      `cls_info['tp']` but `_voc_eval` emits `tp_count` etc., and indexes
      `perclass[0]` which is name-keyed. It is deprecated-scheduled for
      1.0.0: delete now (recommended) or fix.
- [ ] `drawing.py:679-697` — best-threshold star mixes compressed and
      original index spaces; correct mapping is
      `np.where(finite_flags)[0][max_idx]`. Currently benign (offset always
      0) but fix while touching the file.

## C. assignment_v2 (new module) semantic contract

Context: `assignment_v2.py` is unwired — nothing outside its own tests imports
it. It is an alternative candidate-index design, not necessarily a drop-in
semantic clone of `assignment.py`. Differences should be evaluated as API and
metric choices rather than automatically classified as correctness bugs.

The follow-up implementation aligned several important baseline behaviors:

- [x] The wrapper default is `compat='all'` in both implementations.
- [x] Both use strict `iou > threshold` matching.
- [x] `compat='ancestors'` relabels a finer prediction to its matched ancestor
      truth in both implementations.
- [x] Sparse candidate generation accounts for positive IoU bias.

### Task 4.14 — Preserve and document the known differences

The remaining differences are part of the current v2 contract:

1. **Hierarchy source when `classes=None`**: v2 consults
   `true_dets.classes` and then `pred_dets.classes`. V1 does not infer classes
   from the detections and downgrades `compat='ancestors'` to flat mutex
   matching unless `classes=` is passed explicitly. V2's inference is useful
   and should not be removed solely for parity.
2. **Relabeling under `compat='all'`**: with an explicit hierarchy, v1 may
   relabel a finer predicted class to an ancestor truth even though all classes
   are match-compatible. V2 preserves the original predicted class in
   `compat='all'` and performs hierarchy coercion only in
   `compat='ancestors'`. This is a semantic choice; the v2 behavior more
   literally separates unconstrained matching from hierarchy-aware scoring.
3. **Equal-score prediction order**: v2 uses a stable descending sort and
   preserves input order for tied scores. V1 reverses `argsort` output and does
   not provide the same tie-order contract. Greedy assignment can therefore
   choose a different truth when tied predictions compete for it.
4. **Supported feature subset**: v2 currently supports only
   `prioritize='iou'` and `truth_reuse_policy='never'`. V1 additionally
   supports class/correct prioritization and least-used truth reuse.
5. **Output and reuse model**: v2 adds a `rank` column and exposes a reusable
   candidate index for querying multiple IoU thresholds. V1 recomputes through
   its monolithic function and emits only `ASSIGN_COLUMNS`.

Keep these differences visible in the module documentation and changelog. If
v2 is wired into the evaluator, make the selected semantics explicit instead
of silently treating either implementation as the universal reference.

### Task 4.15 — Characterization and integration tests

Add seeded randomized tests over `DetectionMetrics.demo`-style data comparing
`_assign_confusion_vectors` and `assign_confusion_vectors_v2` across
`compat ∈ {mutex, all, ancestors}`, `bias ∈ {0, 1}`, weights on/off,
`max_dets`, iterable `iou_thresh`, `ignore_classes='ignore'`, and forced
sparse versus dense candidate builds.

Exact equality is required only where the two implementations have the same
semantic contract. Add explicit expected-difference tests for hierarchy
inference, `compat='all'` relabeling, and tied-score ordering. The integration
gate is that every observed difference is either documented and intentional or
fixed — not that v2 must reproduce every v1 edge behavior.

## D. Addendum — deep-dive verification round (2026-07-09)

A second, execution-based verification pass ran every claim above that could
be run (sklearn 1.9 / pycocotools installed for reference comparisons).
Verified-at-runtime: 4.1, 4.2, 4.3 (both trigger cases), 4.7, 4.8, 4.9
(tn tail clobbered AND combined AP shifts 0.816→0.845 with -inf rows present,
vs 2e-3 binning error without), 4.10, 4.11 (mAP=1.0 where truth says 0.5),
4.12 items 1+3 (spy shows `use_area_attr` never passed; all area-range
results share ONE cfsn_vecs object id), and 4.13's dict-of-Detections KeyError.

The original audit described all assignment-v2 differences as parity defects.
Subsequent work aligned the shared defaults, strict threshold behavior,
ancestor-mode relabeling, and sparse bias handling. The remaining differences
are now recorded in part C as explicit semantic or feature choices. The test
counts below describe the repository at the time of the historical audit, not
the coverage after the follow-up commits.

New findings from this pass (add as tasks):

### Task 4.16 [verified, HIGH] — probs presence silently switches mAP semantics

`confusion_vectors.py:552-565` (`binarize_ovr` mode=1): when `cfsn_vecs.probs`
is available, per-class scores become `probs[:, cx]` over ALL assignment rows
(classification-style OVR); otherwise detection-style scoring is used.
Measured effect on identical detections (`DetectionMetrics.demo(nimgs=10,
nboxes=(0,10), n_fp=(0,3), n_fn=(0,2), classes=3, rng=0)`):
`score_kwcoco` mAP = **0.389** with probs vs **0.8116** without — and the
without-probs value equals `score_pycocotools` and `score_voc` (0.8116 /
0.8154) almost exactly. The code's own TODO at `:554-556` says "perhaps we
shouldn't use these or at least allow for configuration".

Maintainer position (2026-07-09): exact score parity with pycocotools is NOT
a goal — divergence is fine as long as the underlying logic is correct. The
probs-based OVR score is a legitimate metric. The problem to address is only
that the semantics switch *silently based on data availability* and are
undocumented: the same call reports classification-style OVR AP when probs
exist and detection-style AP when they don't, and nothing tells the user
which one they got. Action: add an explicit `ovr_score_mode` (or similar)
config, document both modes and when each activates, and emit a one-time
info/warning when the probs path activates. No default-behavior change
required.

### Task 4.17 [verified, MEDIUM] — modified-params path never returns `mAP`

`detect_metrics.py:919-931`: when `modified_params` is True,
`pct_summarize2` runs but `pct_info['mAP']` is never assigned — so ANY custom
`iou_thresholds` (even after Task 4.3's fix) returns a dict without `mAP`.
Fix together with 4.3: populate `mAP` from the summarize stats in both
branches. Note `evaler.stats[1]` (standard branch) is AP@0.5, not AP@.5:.95
— decide which is meant by 'mAP' and document it.

### Task 4.18 [verified, MEDIUM] — v1 and v2 source hierarchy differently

`_assign_confusion_vectors` builds a flat compatibility model when
`classes=None` even if `true_dets.classes` or `pred_dets.classes` carries a
real hierarchy; `assignment_v2` reads those detection-level classes.
Verified: the same hierarchical inputs give v1 no cross-hierarchy match
without explicit `classes=`, while v2 can use the attached hierarchy. This is
now a documented semantic difference rather than an automatic parity bug.
Preserve v2's inference unless an evaluator integration decision explicitly
chooses v1 behavior.

Also verified: v1's hierarchical relabel fires for `compat='all'` when a
hierarchy is explicitly supplied, while v2 reserves relabeling for
`compat='ancestors'`. The characterization tests in 4.15 must cover and name
this difference.

### Task 4.19 [verified, LOW] — misc from this pass

- `detect_metrics.py:1671-1683` (local `_summarize`): unconditional debug
  prints on every call, and the printed `pct_map` averages per-class AP
  INCLUDING `-1` sentinel (background/no-truth) columns — the printed number
  is wrong (0.36 vs real 0.81) though the returned `mean_s` correctly masks
  `s > -1`. Remove or gate the prints; fix the printed mean.
- `measures(ap_method=...)` accepts only
  {pycocotools, sklearn, sklish, outlier} and raises bare `KeyError` for
  anything else (e.g. voc2007/voc2012 are NOT valid here) — document the
  choices in `CocoEvalConfig.ap_method` help (`coco_evaluator.py:147-155`)
  and raise a helpful error listing them.

## Verification (whole doc)

```
python -m pytest tests/test_metrics_assignment_v2.py tests/test_metrics_assignment.py \
    tests/test_detect_metrics.py tests/test_confusion_vectors.py tests/test_sseg_metrics.py -q
# Compare shared-contract cases exactly and assert documented differences
# separately; do not use unconditional row-for-row parity as the oracle.
python -m pytest --xdoctest kwcoco/metrics kwcoco/coco_evaluator.py -q
```

Numerical outputs WILL change for: mAP with truth-only classes (4.11),
AP/AUC of tiny-support classes (4.10), area-range results (4.12). Each such
change must be listed in `CHANGELOG.md` under `Fixed`.
