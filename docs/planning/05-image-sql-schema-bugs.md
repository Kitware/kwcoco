# 05 — CocoImage, SQL backend, and schema bugs

Status: TODO
Risk: medium (SQL fixes need postgres+sqlite testing; asset fixes touch the multispectral hot path)
Prereqs: 01

Files: `kwcoco/coco_image.py`, `kwcoco/coco_sql_dataset.py`,
`kwcoco/coco_schema.py`, `kwcoco/coco_schema.json`. SQL tests:
`tests/test_sql_database.py` (sqlite) and `tests/test_postgresql.py`
(skipped without a reachable postgres — run what you can, keep both in mind).

## A. CocoImage `assets` vs `auxiliary` (fix as one unit)

The root cause of the two high-severity bugs is scattered dual-key handling.
Fix by **centralizing**: give `CocoImage` one private helper, e.g.
`_raw_asset_lists(self) -> list[tuple[str, list]]`, returning the actually
present raw keys (inspect `self._proxy`/raw img dict, NOT the aliased proxy),
and route `_assets_key`, `iter_asset_objs`, `n_assets`, `primary_asset`,
`imdelay` through it.

### Task 5.1 [verified] — `_assets_key()` alias confusion splits assets across two keys

`coco_image.py:690-700` checks `'auxiliary' in self`, but
`CocoImage.__alias_to_primary__ = {'assets': 'auxiliary'}` makes the aliased
`__contains__` (`util/dict_proxy2.py:400-405`) return True when the raw dict
only has `'assets'`. `add_asset` (`coco_image.py:899-903`) then does a raw
`img.get('auxiliary')` → None → creates a **new** `auxiliary` list alongside
the existing `assets` list. Verified with a repro.

### Task 5.2 [verified-by-reading] — `imdelay` and `primary_asset` ignore `assets` when `auxiliary` exists

`coco_image.py:459` and `:1084` use
`img.get('auxiliary', img.get('assets', [])) or []` (either/or), whereas
`iter_asset_objs` (`:551-558`), `channels`, and `n_assets` iterate **both**
lists — the schema permits both (`coco_schema.json:139-174`). With both keys
present (which 5.1 actively creates), `imdelay` returns NaNs for channels
stored under `assets`.

- [ ] Implement the centralized helper; fix both call sites to concatenate.
- [ ] Regression tests (new `tests/test_coco_image_assets.py`):
      (a) image defined with `assets` only → `add_asset` appends to the same
      list, no `auxiliary` key appears;
      (b) image with channels split across both keys → `imdelay` loads all
      channels, `primary_asset` considers all, `n_assets == channels` count.

### Task 5.3 — CocoImage misc edge-case fixes

- [ ] `:170-178` `datetime`: second literal is not an f-string (braces emitted
      verbatim); docstring claims `| None` return but the code raises
      `KeyError`. Return None (matching docs) and fix the message.
- [ ] `:333-335` `num_channels`: crashes with `AttributeError` when
      `channels` is None; return None.
- [ ] `:1392-1393` `resolution(space='video')` on video-less images:
      `aliased_get(self.video, ...)` iterates `None`. Fall back to image
      space (`self.video or {}` like `imdelay:1117` does). Also replace the
      bare `assert`s at `:1397,:1415` with real exceptions.
- [ ] `:1438-1440` `resolution(space='asset')` with unmatched channel:
      `find_asset_obj` may return None → raise a clear KeyError naming the
      channel. Also note `imdelay(space='asset', resolution=...)` always
      fails (`_scalefactor_for_resolution` called with `channel=None` at
      `:1190-1194`) — make it raise `NotImplementedError` with guidance, or
      thread the channel through.
- [ ] `:505-509` `iter_image_filepaths`: skip assets with `file_name=None`
      (legitimately created by `add_asset(imdata=...)`).
- [ ] `:686` `find_asset_obj`: use `obj.get('channels')` so assets missing
      the key don't KeyError.
- [ ] `:818-826` `add_asset`: accept `np.integer` width/height
      (`isinstance(width, (int, np.integer))` — numbers from `arr.shape`
      arithmetic); move the `file_name is None and not imwrite` validation
      out of `if __debug__` (`:880-881`); restore the friendly error for
      detached images without bundle_dpath (`:882-892`, the intended block is
      right there, commented out).
- [ ] Tests for each in `tests/test_coco_image_assets.py` or extensions of
      `tests/test_coco_image.py`.

## B. SQL backend

### Task 5.4 — Order-preserving batched `_column_lookup` (correctness + perf)

Two defects, one rewrite:

1. `[HIGH]` `coco_sql_dataset.py:1987-2026` — the postgres "join optimize"
   path has **no ORDER BY**: rows can come back in any order (hash joins
   reorder) and missing rowids are silently dropped, then zipped positionally
   against `rowids` (`:2120-2121`) → silently wrong per-annotation values.
2. `[PERF-HIGH]` `:2051-2056` — the fallback is an N+1 loop: one
   `session.execute` per rowid (100k lookups for 100k annotations).

- [ ] Replace both with one batched, order-preserving implementation:
      postgres: `JOIN unnest(:ids) WITH ORDINALITY ... ORDER BY ordinality`
      (LEFT JOIN from the id list so missing ids yield None);
      sqlite: single `WHERE id IN (...)` then reorder in Python via a dict.
      Parameter-bind the ids (also closes the injection surface at `:1998`).
- [ ] Test with shuffled, duplicated, and missing rowids on both backends;
      on postgres use a table large enough (≥10k rows) to trigger a hash
      join, comparing against the dict backend.

### Task 5.5 — `SqlIdGroupDictProxy` is wrong for `trackid_to_aids`

`coco_sql_dataset.py:1194-1201, 860-1045`. Built without `parent_keyattr`, so:
`keys()` yields one key per annotation row (duplicates + None); `__len__`
counts annotations not tracks; `items()` with `order_attr` emits spurious
empty groups that clobber real values in `dict(proxy)`; the ALCHEMY ordered
branch (`:1000-1008`) orders by `Image.frame_index` without joining Image
(cartesian product); the raw branches reference a nonexistent
`annotations.frame_index` column (`:1011-1025`, `:931-936`).

- [ ] Give the proxy an explicit distinct-parent mode (DISTINCT keys, correct
      len) and make ordered `items()` perform the same join as
      `_uncached_getitem` (`:915-921`).
- [ ] Test: `dict(sql.index.trackid_to_aids)` equals the dict backend's, and
      add `trackid_to_aids` to `assert_dsets_allclose`'s compare set
      (`:2513-2534` currently omits it deliberately).

### Task 5.6 — Stale SQL view cache

`[verified]` `coco_sql_dataset.py:2436-2445`: `ub.CacheStamp(...,
depends=[dct_db_fpath], hasher=None)` keys on the path string only —
rewriting the source json (contents + mtime) does not expire the stamp, so
users get stale SQL views. The docstring claim at `coco_dataset.py:1678-1680`
("depends on the timestamp") is false. Also `cache_product=[]` for postgres
(`:2420-2421`) means no product check at all there.

- [ ] Include source file size+mtime (cheap) or content hash in `depends`.
- [ ] Test: build view → modify json → `view_sql` again → data reflects the
      change.

### Task 5.7 — Raw-SQL mode (`ALCHEMY_MODE=0`): fix or delete

Raw mode is broken under SQLAlchemy 2.x (plain strings passed to
`session.execute`, `:439-441,:694,:747,:937,:976-982,:1026,:1098` →
`ArgumentError`), returns garbage rows in `values()` with `ignore_null`
(`:740-744` selects only the key column then zips against all column names),
skips `dict_restructure` in `SqlListProxy.__iter__` (`:443-447` leaks
`_unstructured`), and has zero test coverage. **⚠️ DECISION** — recommended:
**delete raw mode** (the `ALCHEMY_MODE` flag, `_raw_yielder`, all raw
branches); it contradicts its purpose if it's slower to maintain than the
ORM path it duplicates. If keeping it, wrap statements in `sa.text()`, fix
the two data bugs, and parametrize the proxy tests over both modes.

### Task 5.8 — SQL lifecycle and safety fixes

- [ ] `:2223-2225` `data_fpath` setter assigns `self.fpath` but `fpath`
      (`:1588-1590`) is getter-only → always `AttributeError`. Add a setter
      or make `data_fpath` set the underlying attribute.
- [ ] `:1362-1364` `__nice__` after `disconnect()` crashes (`self.index` is
      None); make it print `'not connected'`. Also make `disconnect()` call
      `self.engine.dispose()`.
- [ ] `:952-966` `SqlIdGroupDictProxy.__contains__` lacks the `to_jsonb`
      handling `_uncached_getitem` has (`:911-913`) → unreliable on postgres
      JSONB keys; and swallowed psycopg2 errors leave the session in
      aborted-transaction state — issue `session.rollback()` in the except
      paths here and in `SqlDictProxy.__contains__` (`:599-611`)
      (see `tests/test_postgresql.py:136` manually working around this).
- [ ] `:1797-1801` `raw_table` / `:2139-2148` `_all_rows_column_lookup`:
      interpolate caller-supplied table/column names — validate against the
      known table/column allowlist (`CocoBase.metadata.tables`).
- [ ] `:1516-1522` `connect(must_exist=True)` is ignored for sqlite (missing
      file silently created empty); honor it.
- [ ] `:1697-1728` `populate_from`: wrap in a transaction so an
      `IntegrityError` (e.g. duplicate file_names, legal in the dict backend)
      doesn't leave a partially-populated DB; re-raise with a message naming
      the offending duplicate.

## C. Schema

### Task 5.9 — Regenerate and sync `coco_schema.json`; anchor patterns

- [ ] `[verified]` The checked-in JSON is stale vs `coco_schema.py`
      (missing the `KWCOCO_KEYPOINT` description added at
      `coco_schema.py:83-92`). The generator (`coco_schema.py:444`) uses
      `urepr(...).replace("'", '"')` which breaks on any apostrophe — replace
      with `json.dumps(..., indent=2)`.
- [ ] `[verified]` `NAME` (`coco_schema.py:149-150`) and `CHANNELS`
      (`:205-211`) patterns are unanchored (`re.search` semantics), so
      `'a/b'` passes `[^/]+`. Anchor as `^[^/]+$` etc.
- [ ] Add schema keys the code actually reads/writes: IMAGE
      `warp_img_to_vid`, `valid_region`, `datetime`, `date_captured`; ASSET
      `warp_aux_to_img`, `warp_to_wld`; VIDEO `width`, `height`. Give TRACK a
      `required` list and align `TRACK.id` type with `ANNOTATION.track_id`
      (currently INTEGER-only vs ANYOF(INTEGER, STRING)).
- [ ] Add `tests/test_schema.py::test_schema_json_in_sync`: regenerate the
      schema in-memory and compare to the checked-in JSON; and a test that
      `NAME` rejects `'a/b'`.
- [ ] Then do doc 02 task 2.5 (ship the JSON) if keeping it.

## Verification (whole doc)

```
python -m pytest tests/test_coco_image.py tests/test_coco_image_assets.py \
    tests/test_sql_database.py tests/test_schema.py -q
python -m pytest tests/test_postgresql.py -q   # if postgres available
python -m pytest --xdoctest kwcoco/coco_image.py kwcoco/coco_sql_dataset.py -q
```
