# 10 — Performance

Status: TODO
Risk: medium (hot paths; benchmark before/after)
Prereqs: 03, 05 (several perf items were rewritten there; don't duplicate)

Benchmark harness: use `kwcoco toydata vidshapes32-msi` scale data and
`ub.Timerit`; record before/after numbers in this doc when done. Some items
below were already fixed as bugs (noted) — verify rather than redo.

## Already handled elsewhere (verify only)

- SQL `_column_lookup` N+1 + ordering → doc 05 task 5.4.
- `coco_plot_stats` double-compute → doc 06 task 6.6.
- `pycocotools_confusion_vectors` O(n²) loop → doc 04 task 4.4.
- v1 assignment per-prediction Python loop → doc 04 part C wires
  `assignment_v2` as the replacement path once parity holds.

## Task 10.1 — Lazy CLI registration

`kwcoco/cli/__main__.py:36-38` eagerly imports all 19 CLI modules and their
heavy deps on every invocation, including `--help` and shell completion —
`cli/coco_eval.py:7` pulls the whole metrics stack at module level,
`cli/coco_visualize_videos.py:10` pulls delayed_image. Options:
scriptconfig ModalCLI lazy registration by module-path string, or defer heavy
imports into `main()` in the offending modules (most modules already follow
that convention). Measure: `time kwcoco --help` before/after (target: well
under 1s).

## Task 10.2 — SQL proxy iteration materializes whole tables

`coco_sql_dataset.py:345-352`: `_orm_yielder` uses `query.all()` (`yield_per`
disabled behind `if 1:`), so every proxy iteration loads the entire table
into memory — contradicting the module's purpose (constant-memory views of
large datasets). Enable `yield_per` (test with both sqlite and postgres;
SQLAlchemy 2.x supports `yield_per` with ORM well). Also:

- [ ] `:371-381` `_new_proxy_cache` falls back to an unbounded `{}` when
      ndsampler is absent — long-running dataloaders grow without eviction.
      Use a small internal LRU (an `OrderedDict`-based one is ~15 lines) so
      the bound doesn't depend on an optional heavy package.
- [ ] `:395-400,573-588,899-901` — `__nice__`/`basic_stats` issue COUNT
      queries per repr/log line; cache counts per session with a dirty flag.
- [ ] `:996-998` — ordered `items()` runs a full extra key query just to
      synthesize empty groups; reuse the OUTER JOIN approach of the
      unordered path (coordinated with doc 05 task 5.5's rewrite).

## Task 10.3 — CocoImage per-call re-parsing

`coco_image.py:303-318` (`channels`) re-coerces every asset's channel spec on
each access; `:1082-1112` (`imdelay`) re-runs `FusedChannelSpec.coerce().normalize()`
per asset per call. Samplers calling `imdelay` per batch item pay repeated
parsing. Cache parsed specs keyed on the raw spec string (module-level
`functools.lru_cache` on a pure `str -> FusedChannelSpec` helper is enough —
do NOT cache on the CocoImage instance, whose dict can mutate). Benchmark
with a loop of 1000 `imdelay` calls on an msi demo image.

## Task 10.4 — Removal cascade rescans

`mixin_add_remove.py:815-818` (acknowledged TODO): `remove_annotations`
rebuilds `aid_to_index` over the whole annotation table per call, and
`remove_categories`/`remove_tracks`/`remove_images`/`remove_videos` each call
it plus build their own id→index maps — `remove_videos` costs ~3 full
annotation-table scans. Accept for now unless profiling shows pain;
worthwhile cheap step: let the internal helpers share one prebuilt
`id_to_index` map per outer call. Skip micro-optimizing beyond that.

## Task 10.5 — Small wins batch

- [ ] `demo/toydata_image.py:531` — `boxes.ious(boxes)` computed and
      discarded before NMS; delete the line (O(n²) per generated image; this
      speeds up every `shapes*` test fixture).
- [ ] `demo/toydata_video.py:1204-1211` — fourier noise on every B* channel
      is self-documented as "up to 80% of" render time; add a cheap-noise
      escape hatch used by the test fixtures (`vidshapes*` speedup
      package-wide). **⚠️ DECISION** — recommended: keep default output
      identical, add `renderkw['cheap_noise']` used only where tests opt in.
- [ ] `util/util_archive.py:334` — `'auto'` zip compression prefers
      ZIP_LZMA; prefer ZIP_DEFLATED first (order-of-magnitude faster writes
      for JSON payloads).
- [ ] `json_backend.py:13-24` — the ujson read path requires BOTH the
      package and `KWCOCO_USE_UJSON=1`; AGENTS.md documents the env var, so
      keep the gate but document it in the `CocoDataset` docstring, or flip
      the default if the maintainer agrees. **⚠️ DECISION** — recommended:
      keep gated, just document.

## Verification

Record in this doc: `time kwcoco --help`, `Timerit` numbers for imdelay loop
and SQL `annots().lookup('bbox')` on a 100k-annotation view, before/after.
Full suite green; no doctest hash changes unless flagged.
