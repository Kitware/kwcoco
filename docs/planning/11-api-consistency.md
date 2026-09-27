# 11 — API consistency & polish

Status: TODO
Risk: medium (observable behavior/flag changes; every rename keeps a
deprecated alias for one cycle)
Prereqs: 03–09

## Task 11.1 — CLI flag and error consistency

- [ ] Unify worker flags: `io_workers` (stats/union/move_assets/
      find_unregistered_images) vs `workers` (validate/conform/fixup/
      visualize). Recommended: accept both everywhere via kwconf
      alias, document `workers` as canonical. Remove the accidental
      `isflag=True` on the *int* `workers` at `cli/coco_validate.py:113`.
- [ ] Unify missing-`src` handling: modules raise bare `Exception` or
      `ValueError` inconsistently (`coco_union.py:79`, `coco_split.py:107`,
      `coco_subset.py:166`, `coco_show.py:98`, `coco_validate.py:173`,
      `coco_modify_categories.py:177`, `coco_conform.py:110`,
      `coco_reroot.py:125` vs `coco_fixup.py:84`,
      `coco_visualize_videos.py:280`, `coco_info.py:241`,
      `coco_stats.py:123`). Add one shared `require_src(config)` helper in
      `kwcoco/cli/__init__.py` raising a clean, consistent error.
- [ ] Consolidate the try-import-rich `rich_print` shims duplicated in 6+
      CLI modules onto `kwcoco.util.util_rich.rich_print` (or its kwutil
      successor per doc 09 task 9.2).
- [ ] Unify dump kwargs: `dumpkw = {'newlines': True, ...}` duplicated in
      union/split/subset/conform/reroot/modify_categories while
      `coco_fixup.py:121` and `coco_move_assets.py:204` call bare
      `dset.dump()`. One helper, consistent output formatting.
- [ ] `cli/coco_stats.py:21` defaults `src` to `special:shapes8` — bare
      `kwcoco stats` silently generates demo data instead of erroring.
      Make `src` required (matching every other subcommand).
- [ ] Migrate the three legacy `__default__ = {...}` configs
      (`coco_split.py`, `coco_grab.py`, `coco_show.py`) to class-attribute
      style used by all other modules.
- [ ] Document `KWCOCO_LOOSE_CLI` (`cli/__main__.py:47,105`) in `--help`
      epilog or remove it.
- [ ] Surface hidden behaviors: `coco_show` `'both'` mode and the
      xdev InteractiveIter loop (`coco_show.py:125,136-149`), `coco_stats`
      `embed` hack (`:58-60`) — document or remove. Remove the
      class-level monkey-patch of `version` onto kwconf's ModalCLI
      (`cli/__main__.py:59-64`) in favor of an instance attribute or
      upstream support.

## Task 11.2 — Metrics `compat` default unification

Defaults currently disagree: `DetectionMetrics.confusion_vectors` = 'mutex'
(`detect_metrics.py:284`, but its docstring says 'all' — `:318`),
`score_kwcoco` = 'all' (`:639`), `CocoEvalConfig` = 'mutex'
(`coco_evaluator.py:122`), v1 assigner = 'all' (`assignment.py:48`).
`binarize_classless` is only correct with 'all'
(`confusion_vectors.py:374-379`; acknowledged at `coco_evaluator.py:875-877`).
**⚠️ DECISION** — recommended: standardize the *documented* default to the
current behavior of each public entry point (no silent behavior change), fix
the lying docstring, and add a loud note in `metrics/AGENTS.md` about which
default each entry point uses and why. A true unification (all → one value)
is a 1.0 decision for the maintainer.
Also fix `detect_metrics.py:537` returning via the leaked loop variable
(`iou_to_cfsn[t]` → `iou_to_cfsn[iou_thresh]`).

## Task 11.3 — Core API consistency

- [ ] Duplicate-add exception types: `add_annotation` raises `IndexError`
      (`mixin_add_remove.py:314-315`) where siblings raise
      `DuplicateAddError`. Raise `DuplicateAddError` (subclassing per doc 07
      task 7.11 keeps old `except ValueError` code working — verify; note
      IndexError callers in changelog).
- [ ] `_resolve_to_ann/_resolve_to_img` raise `IndexError` vs
      `_resolve_to_cat`'s `KeyError` (`mixin_accessors.py:311,330` vs cat).
      Standardize on KeyError with aliasing period.
- [ ] `remove_categories(keep_annots=True)` pops `category_id` while
      `add_annotation` stores explicit None (`mixin_add_remove.py:898-902`
      vs `:320`) — standardize on `None` (doc 03 task 3.6 chose the same
      for tracks).
- [ ] `annots()` filter semantics: `track_id` silently overrides `image_id`
      and returns SortedSet vs list (`mixin_objects.py:107-121`). Raise on
      conflicting filters; normalize return order type.
- [ ] `AnnotGroups.cnames` / `category_names` are copy-paste identical
      (`coco_objects1d.py:1129-1159`) — alias one to the other.
- [ ] CocoIndex one-to-zero handling for tracks: `mixin_index.py:617-628`
      fixes cats/imgs/videos but not tracks, forcing bespoke tolerance in
      `_check_index` (`coco_dataset.py:913-926`). Add the track fix; delete
      the special-case tolerance.
- [ ] Stale docstrings from the mixin split: `coco_dataset.py:150`
      (`MixinCocoExtras._build_hashid` → `MixinCocoHashing`), `dumps`
      documents `newlines` twice (`:596-603`), `subset` copy-default doc
      wrong (`:1523,1540-1542`), `ensure_video/ensure_track` document a
      nonexistent `file_name` arg (`mixin_add_remove.py:492,526`).
      Give `_dataset_components/__init__.py` a docstring naming the mixin
      composition order (MRO matters, `coco_dataset.py:109-122`).

## Task 11.4 — SQL/dict backend parity statement

Doc 05 fixed the outright bugs; what remains is honesty about the contract:

- [ ] Make mutating methods raise `NotImplementedError` ("SQL views are
      read-only") on `CocoSqlDatabase` instead of silently poisoning the
      proxy cache (`coco_sql_dataset.py:661-668,943-950` mutable cached rows
      — the audit's S6). A `_readonly` guard mixin overriding the
      known mutators is enough.
- [ ] `CocoSqlIndex.__init__` (`:1149-1163`) declares only a subset of the
      attributes `build` creates — declare them all as None so unbuilt
      access fails with a clear message like the dict CocoIndex
      (`mixin_index.py:101-121`).
- [ ] Document (module docstring table) which mixins/methods the SQL view
      supports vs not (`:1314-1321` inherits Accessors/Objects/Stats/Draw;
      no Hashing — and `_cached_hashid` raising on miss at `:2284-2330`
      should say why).
- [ ] Postgres credentials: defaults `admin:admin` embedded in URIs and a
      plaintext path-LUT at `~/.cache/kwcoco/postgres-fpath-lut`
      (`:2380-2413`); also the 60-char truncated db name can collide for two
      long dataset paths, silently pointing both at the same DB — add a
      short hash suffix to the db name (fixes collisions) and document the
      credential story.

## Task 11.5 — Docs alignment pass

- [ ] Move `docs/dataset_api.txt` and `docs/format_notes.md` under
      `docs/source/manual/` (or delete if superseded).
- [ ] Regenerate `docs/source/auto/` after all removals/renames
      (doc 09) so the API docs match.
- [ ] Update `kwcoco/coco_schema_informal.rst` for the schema keys added in
      doc 05 task 5.9.
- [ ] README pass: confirm examples still run (they exercise the union/eval
      surfaces changed above); AGENTS.md pass for anything this plan made
      stale.

## Verification

```
python run_tests.py
python -m pytest --xdoctest kwcoco -q
make -C docs html   # docs build clean (needs requirements/docs.txt)
```

Final step of the whole plan: write a summary of all **⚠️ DECISION** items
taken (there are ~15 across the docs) into `docs/planning/DECISIONS.md` for
maintainer review, and update each doc's `Status:` header.
