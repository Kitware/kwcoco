# 09 — Dead code removal & deprecation consistency

Status: TODO
Risk: medium (removals; each needs a grep for external-looking usage and a
deprecation cycle where the symbol was public)
Prereqs: 03–07 (don't delete code those docs fix; conflicts resolved in their favor)

Policy: private/broken/never-called code is deleted outright. Public symbols
get a `ub.schedule_deprecation` shim cycle (deprecate now, error at 1.0.0,
remove at 1.1.0) — and the shim must actually re-export the functionality
(see the kpf/kw18 cautionary tale, doc 07 task 7.1).

## Task 9.1 — Orphaned util modules (verified zero in-repo consumers)

- [ ] `kwcoco/util/util_futures.py` — pure re-export of
      `ubelt.Executor/JobPool`, self-described deprecated. Convert to a
      warning shim; schedule removal.
- [ ] `kwcoco/util/lazy_frame_backends.py` — re-export of
      `delayed_image.lazy_loaders`; same treatment.
- [ ] `kwcoco/util/util_reroot.py` — exported but unused (only a
      commented-out import at `_dataset_components/mixin_extras.py:796`).
      **⚠️ DECISION** — recommended: keep but mark experimental in the
      docstring (it contains real logic, not a re-export), OR wire it into
      `reroot` if doc 03 work showed a need. Do not silently delete.
- [ ] `kwcoco/util/delayed_ops/__init__.py` — compat re-export of
      delayed_image; add a `schedule_deprecation` call.

## Task 9.2 — Vendored code with upstream equivalents

- [ ] `kwcoco/util/util_networkx.py` (490 lines) — vendored networkx PR
      #5602; `nx.write_network_text` landed in networkx 3.0. Only consumer:
      `category_tree.py:747` (+ CLI fallbacks, `cli/coco_stats.py:155`).
      Use the upstream function when available (version check), keep the
      vendored copy only if the networkx floor in requirements is < 3.0 —
      check `requirements/runtime.txt` and prefer bumping the floor.
- [ ] `kwcoco/util/util_sklearn.py` `StratifiedGroupKFold` — sklearn ≥1.0
      ships it. Only consumer: `cli/coco_split.py:101`.
      **⚠️ DECISION** — delegating to upstream changes split outputs for
      seeded users. Recommended: switch to upstream, document in changelog
      as a behavior change; keep the vendored class one release behind a
      deprecation warning.
- [ ] `kwcoco/util/util_json.py` — header says deprecated;
      `ensure_json_serializable` still used by
      `coco_evaluator.py:1098,1210`. Migrate those callers to `kwutil.Json`
      equivalents, shrink the module to warning shims.
- [ ] `kwcoco/util/util_rich.py` — self-labeled deprecated but actively used
      (`cli/find_unregistered_images.py:99`, `cli/coco_plot_stats.py`).
      Either migrate callers to kwutil and shim, or remove the deprecated
      label. Coordinate with doc 11 task 11.2 (rich_print consolidation).
- [ ] Finish the half-done `util_parallel` migration: some callers already
      use `kwutil.util_parallel` (`cli/coco_visualize_videos.py:273`,
      `metrics/segmentation_metrics.py:1461`); move the rest
      (`cli/coco_union.py`, `_dataset_components/mixin_constructors.py:630`)
      and shim `kwcoco.util.util_parallel`.
- [ ] `kwcoco/util/dict_like.py` vs `dict_proxy2.py` — two generations of
      the same abstraction. Consolidate on dict_proxy2: migrate
      `coco_evaluator` and `coco_sql_dataset` off `dict_like.DictProxy`,
      then shim `dict_like`.

## Task 9.3 — Dead code blocks (delete outright)

Core/dataset: `coco_dataset.py:857-879` (`_check_warnables`, never called);
`mixin_extras.py:761-769` (`DO_OLD_CHECK=0` block); `mixin_index.py:551-568`
(`if 0:` alt implementation).
SQL: `coco_sql_dataset.py:1109-1141` (commented-out values()),
`:1556-1577` (`if 0:` postgres knobs), `:2058-2104` (duplicate
`ADDITIONAL_HACK` branches — the 1.x/2.x bodies are now identical),
`:345-368` (`if 1:` yielder scaffolding — resolve in concert with doc 10
task 10.2).
Metrics: `functional.py` `_truncated_roc` + `functional._pr_curves` (no
callers — verify with grep, then delete); `detect_metrics.py:515-521`
(`if 0: xdev`); `coco_evaluator.py:747-760,775-777` (`if 0:` blocks);
`drawing.py:295-341,497-510` (`if 0:` seaborn paths);
`segmentation_metrics.py:1727-1894` (DEBUG scaffolding);
`confusion_vectors.py` `binarize_ovr(mode=0)` — docstring says "PROBABLY
BROKEN"; remove the mode (raise on request).
CLI: `coco_modify_categories.py:169-174` (`if 0:`); `coco_stats.py:356-379`
(commented logic); `coco_info.py:397-512` (commented prototype);
`coco_validate.py:9-36` (`__autogen_cli_args__` blob referencing a
developer-local vimtk module).
Demo/util: `boids.py:261-269,513-537`; `perterb.py:341-393` (dead
`raise AssertionError('must be hacked')` + 50 commented lines);
`toydata_video.py:644-680` (unused inner `warp_within_bounds`),
`:1494-1556` (unreachable torch branch), `:1411-1416` (`if 0:` PCA);
`formats/kpf.py:85-86` (`KPFStream` stub), `:122-148` (no-op `demo()`);
`grab_voc.py:10-23` (`__torrent_voc` stub); `grab_camvid.py:387,446-465`;
`util/__init__.py:17-73` (Python ≤3.6 branch — the package requires ≥3.8);
`jsonschema_elements.py:411-421` (`if 0:` scratch);
`coco_image.py:1148-1163` (commented `elif` in imdelay).

- [ ] Delete each after a repo-wide grep confirms no live references.
      Run the full suite after each file-cluster.

## Task 9.4 — Deprecation consistency sweep

- [ ] `_build_index` (`coco_dataset.py:1019-1022`): docstring says "use
      rebuild_index" with a commented-out schedule_deprecation, yet 8+
      internal call sites still use it (`compat_dataset.py:48`,
      `coco_evaluator.py:421`, `cli/coco_union.py:135`, demo modules,
      `data/grab_camvid.py:481`). Migrate internal callers to
      `rebuild_index`, then enable the scheduled deprecation.
- [ ] `add_auxiliary_item = add_asset` alias: `mixin_add_remove.py:182-183`
      is a bare alias while `coco_image.py:1532` wraps the same name in
      `deprecated_function_alias` — make both warn.
- [ ] `img_root`/`data_root` properties (`mixin_extras.py:894-910`) and the
      `img_root=` ctor arg (`coco_dataset.py:279-280`): "will deprecate" in
      docs only — add real scheduled deprecations.
- [ ] `subset` writes the deprecated `img_root` key into every output
      (`coco_dataset.py:1643`) — stop propagating it (write `bundle_dpath`
      semantics only).
- [ ] `category_tree.py:938-945` `_calldict.__call__` ad-hoc
      `warnings.warn` → schedule_deprecation.
- [ ] `mixin_objects.py:77-80,171-172` (`aids=`/`gid=`/`gids=` arg aliases)
      and `Annots.cnames` (`coco_objects1d.py:891,902` TODO) — decide keep
      (documented alias) or deprecate; make it uniform.
- [ ] Core `union` doctest (`coco_dataset.py:1087-1091`) uses
      `keypoint_annotation_frequency`, which errors at 1.0.0 — rewrite the
      doctest with the non-deprecated API now.
- [ ] `mixin_deprecate.py:57-67` — the deprecated method's doctest calls a
      *different* method; fix so the deprecation path itself is exercised.

## Task 9.5 — Naming typos with public exposure

- [ ] `kwcoco/demo/perterb.py` → `perturb.py` with a re-export shim
      (`perterb_coco` → `perturb_coco` alias kept, deprecation-scheduled).
      ~20 import sites in-repo (coco_evaluator, metrics, compat_dataset,
      CLI docstrings) — update them all.
- [ ] `stabalize_thresh` (misspelled) in `confusion_vectors.py:703,876,950`,
      `detect_metrics.py:641` and JSON keys — add correctly-spelled
      `stabilize_thresh` parameter alias (accept both, warn on old) before
      it spreads further. JSON keys: write both, read either, for one
      release.
- [ ] `toydata_video.py:546` `senorchan` variable typo (internal — just
      rename).

## Task 9.6 — Remove the deprecated `validate --fix` pathway

`cli/coco_validate.py:123-149,184-213` duplicates `kwcoco fixup` and carries
its own bugs. After doc 06 task 6.8's guard has shipped in at least one
release: schedule deprecation, then remove (~40 lines). If executing this
whole plan in one release cycle, just add the scheduled deprecation now and
leave removal for 1.0.0.

## Verification

```
grep -rn 'if 0:' kwcoco/ | wc -l    # should approach zero (document survivors)
python run_tests.py
python -m pytest --xdoctest kwcoco -q
./run_linter.sh
```
