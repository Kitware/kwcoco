
Done. The full-repo audit is complete and the plan is written to docs/planning/ — a README.md plus 11 ordered, self-contained execution docs (~1,800 lines total) that an Opus-level agent can pick up and run in sequence.

How the audit ran: six parallel deep-audit agents covered core dataset + mixins, coco_image/SQL/schema, metrics + evaluator, CLI, util/demo/data/formats, and tests/CI/packaging. They produced ~150 findings; many were empirically verified by executing minimal repros (marked [verified] in the docs), and I spot-checked six of the highest-profile citations against source — all accurate.

Most significant verified bugs the plan fixes:

Core index corruption (doc 03): duplicate-id detection that can never fire (mixin_index.py:481), bulk add_images skipping vidid_to_gids, clear_images() leaving three stale index structures, broken unindexed fallbacks in remove_videos/tracks/categories, and union track-id collisions that silently merge unrelated tracks.
Metrics correctness (doc 04): np.trapz crashes under NumPy 2.x, binarize_ovr corrupts the caller's probability matrix in place, an inverted condition makes score_pycocotools silently ignore custom IoU thresholds, segmentation metrics swap true/pred image ids, and the new assignment_v2 has important semantic differences from v1 that must be documented before it can be wired in.
CLI (doc 06): kwcoco eval writes metrics to a file literally named auto — which explains the stray auto file sitting in your repo root — plus split --num_write writing identical splits N times, and move_assets never registered as a subcommand.
CocoImage/SQL (doc 05): assets vs auxiliary alias confusion splits asset lists so imdelay silently returns NaNs; the postgres _column_lookup has no ORDER BY and can return wrong per-annotation values; the SQL view cache never invalidates when the source JSON changes.
Plan structure: docs 01–02 are zero-risk hygiene/infra (delete 13 MB of stray tarballs, the ubelt-copy-pasted .coveragerc, Python-2.7-era CI configs; commit the untracked assignment_v2 that the changelog already claims exists). Docs 03–07 are bug fixes with test-first instructions per finding. Docs 08–11 harden the test suite (revives the permanently-skipped CLI smoke test), retire dead/vendored code, fix perf (SQL N+1, eager CLI imports), and polish API consistency. Behavior-changing choices are flagged ⚠️ DECISION with a recommended default so an agent can run autonomously and hand you a DECISIONS.md to review at the end.




The deep dive is complete and docs/planning/04-metrics-bugs.md now carries a verification addendum (tasks 4.16–4.19) recording everything below, including your note that pycocotools parity is explicitly not a goal.

What I did
I executed nearly every claim in the metrics stack rather than trusting the earlier audit: installed scikit-learn and pycocotools as reference implementations (user site, --break-system-packages, plus sortedcontainers/safer which the repo needed to even run), then ran targeted repro scripts against kwcoco/metrics/ and coco_evaluator.py. The bottom line: all 24 existing metrics tests and 68 doctests pass, yet 16 real logic defects are confirmed at runtime. The suite exercises happy paths only.

Confirmed logic bugs (these need fixing regardless of parity philosophy)
Silent data corruption — highest priority:

binarize_ovr writes -inf into the caller's probs matrix through a NumPy view (confusion_vectors.py:560-565); _binary_clf_curve2 similarly rewrites NaN scores in the caller's array in place.
Every per-area-range CocoSingleResult shares one cfsn_vecs object whose weights get overwritten each iteration — I verified all three results hold the same object id with identical (last-written) weights, so stored results for "all"/"small" are silently wrong.
Wrong numbers:

VOC_Metrics.score drops classes that have truth but no predictions: a 2-class case where the detector never fires on class B reports mAP = 1.0 instead of 0.5.
Stabilization padding inflates realpos_total 2.0→2.14 and realneg_total 1.0→1.86 for small classes (totals computed after padding instead of before).
Measures.combine with -inf thresholds (which every detection eval produces) clobbers the tn tail and shifts combined AP by ~3% (0.816→0.845) versus ~0.2% normal binning error.
Segmentation metrics swap true/pred image ids at job submission (segmentation_metrics.py:1650-1652) — internally inconsistent in the same function, only works when both datasets happen to share ids.
Crashes:

np.trapz is gone in NumPy 2.x → score_kwant crashes (confirmed on NumPy 2.4.4).
score_pycocotools(iou_thresholds=...) inverted condition: custom same-length thresholds are silently ignored then with_confusion IndexErrors; and — new finding — the modified-params branch never sets mAP at all, so even after the inversion fix, custom thresholds return no mAP.
OneVersusRestMeasureCombiner.__init__ typo (growth = precision) disables/breaks both options; dict-of-Detections evaluator input KeyErrors; use_area_attr is never forwarded (verified with a spy — documented behavior never runs); unclipped searchsorted binning IndexErrors on scores above the top edge.
assignment_v2 (historical comparison with v1)
The initial prototype differed at exact IoU thresholds, ancestor-mode relabeling, defaults, and sparse bias handling; follow-up commits aligned those shared baseline behaviors. Other differences are legitimate design choices rather than automatic defects: v2 infers hierarchy from detection objects, preserves the predicted class under compat='all', uses stable input ordering for tied scores, adds rank metadata and candidate-index reuse, and currently implements a narrower feature subset. These differences now need explicit characterization rather than forced parity.

Reframed per your note
The headline "kwcoco mAP is half of pycocotools" (0.389 vs 0.812 on identical detections) is not a computation bug: sklearn agrees with kwcoco's AP on kwcoco's own vectors, and with with_probs=False kwcoco matches pycocotools to 4 decimals. The issue is that binarize_ovr silently switches to classification-style OVR scoring whenever probability data is present — the same call reports a semantically different metric depending on data availability, undocumented. The plan now asks only for an explicit mode config + documentation + a one-time notice, no default change. Similarly, the wrong pct_map print I chased down is diagnostic-only (the returned value correctly masks -1 sentinels) — downgraded to "remove/fix the debug prints."

Everything is captured as executable tasks in the plan; the characterization test in task 4.15 is the gate before v2 can be wired in. It requires exact agreement only for shared semantics and explicit assertions for intentional differences.
