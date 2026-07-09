# 08 — Test-suite hardening

Status: TODO
Risk: low (additive)
Prereqs: 03–07 (per-bug regression tests already landed with their fixes;
this doc covers the systemic gaps)

Baseline at audit time: 33 test files, ~3800 lines, 90 collected tests, no
`conftest.py`, no markers, several permanently-skipped or assert-free tests.
CI runs xdoctest over the whole package, so doctests count as coverage — the
gaps below are places with neither.

## Task 8.1 — Revive the CLI smoke test

`tests/test_kwcoco_cli.py:8` — the file's only test (end-to-end
toydata/stats/split/union/show subprocess smoke) is `pytest.skip('disable for
now')` and has been dead indefinitely. Re-enable it; if a step is legitimately
broken, that's a bug to fix (doc 06), not a reason to skip. While here, add:

- [ ] Parametrized `python -m kwcoco.cli.<mod> --help` subprocess test over
      every module in `kwcoco/cli/` (guards doc 06 task 6.2's bug class).
- [ ] Registration-drift test: every `kwcoco/cli/*.py` exposing `__cli__`
      appears in `kwcoco/cli/__main__.py::modnames` (guards task 6.3).
- [ ] Exit-code test: a failing subcommand exits nonzero through both
      `kwcoco` and `python -m kwcoco`.

## Task 8.2 — Un-skip or fix the other dead tests

- [ ] `tests/test_tracks.py:8` — `test_track_order` skipped with no reason.
      Investigate: if it fails, that's a real track-ordering bug; fix or
      file it. `test_add_tracks` (`:39`) builds a tracked dataset and asserts
      nothing — add assertions (`_check_integrity`, track membership).
- [ ] `tests/test_mscoco.py:31` — skipped as 'slow test'; this is the only
      real-data MS-COCO compatibility check. Add a `slow` pytest marker
      (register in pyproject) and run it in a scheduled/manual CI job rather
      than never.
- [ ] `tests/test_cached_hashid.py` — zero asserts; assert cache-file
      existence/behavior at each phase.
- [ ] `tests/test_keypoints.py` — 2 of 3 tests assert-free; add assertions.
- [ ] `tests/test_sql_with_torch_datasets.py:75` — assert-free AND uses
      `special:vidshapes800` (800 videos!); shrink to `vidshapes8` and add
      asserts; mark `slow` if still heavy.
- [ ] Add reason strings to every bare `pytest.skip()`
      (`test_multispectral.py:95`, `test_sql_database.py:25,37`,
      `test_postgresql.py:18,234`).

## Task 8.3 — Coverage for zero-coverage modules

Modules with no tests AND no executed doctests (verified):

- [ ] `kwcoco/formats/voc.py` — round-trip test on a tiny synthetic VOC xml
      tree (would have caught 3 audit bugs).
- [ ] `kwcoco/util/util_kwutil.py` — the queue classes' examples are in
      `Ignore:` blocks; write real tests (blocking semantics, shutdown,
      exception propagation) — these are concurrency primitives used by
      `metrics/segmentation_metrics.py:32`.
- [ ] `kwcoco/util/ijson_ext.py` — tests for incomplete-JSON, NaN handling
      (it monkey-patches an ijson backend against internal upstream APIs;
      an upstream refactor would break `kwcoco info` silently).
- [ ] `kwcoco/util/util_reroot.py`, `dict_like.py`, `util_special_json.py`
      — minimal behavior tests or doctests (or delete under doc 09 if
      orphaned — check doc 09 first to avoid testing code slated for
      removal).
- [ ] `kwcoco/data/` pure converters (`convert_camvid_raw_to_coco`,
      `_convert_cifar_to_kwcoco`, `convert_voc_to_coco` helpers) — test
      against tiny synthetic trees; do NOT test the network downloads.

## Task 8.4 — Structural test-suite improvements

- [ ] Add `tests/conftest.py` providing common fixtures: `photos_dset`,
      `vidshapes_dset` (session-scoped where safe), and a tmp bundle dir
      helper. Migrate the heaviest repeated demo-construction calls.
- [ ] Register markers in `pyproject.toml`: `slow`, `network`, `postgres`,
      `gdal`. Apply to existing conditional-skip tests so
      `pytest -m 'not slow and not network'` is a fast local loop.
- [ ] Make module-scope imports of optional deps
      (`tests/test_subset.py:1-5`, `tests/test_metrics_assignment.py`)
      collection-safe with `pytest.importorskip` where the dep is genuinely
      optional (kwimage is a hard dep — leave those).
- [ ] Property-style index test: a randomized sequence of
      add/remove/clear/union operations on a demo dataset asserting
      `_check_integrity()` after every step (seeded; ~100 ops). This is the
      long-term guard for the whole doc-03 bug class.

## Task 8.5 — Doctest gating hygiene

Many CLI/module doctests are gated `+REQUIRES(module:kwplot)` even when they
don't plot (e.g. `find_unregistered_images` module doctest, gated on kwplot
"because the requires directive does not work on ruamel.yaml" per its own
comment). Sweep `grep -rn 'REQUIRES(module:' kwcoco/` and re-gate on the
actual dependency used. Doctests that exist only to demo plotting keep their
gates.

## Verification

```
python -m pytest tests/ -q                       # everything green
python -m pytest tests/ -q -m 'not slow'          # fast loop works
python -m pytest --collect-only -q tests/ | tail -3   # count grew, no errors
python run_tests.py                                # coverage report; compare % to baseline
```

Record the coverage percentage before and after in this doc when done.
