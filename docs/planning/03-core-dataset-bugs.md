# 03 — Core CocoDataset correctness bugs

Status: TODO
Risk: medium (touches the most-used code paths; every fix needs a regression test)
Prereqs: 01

All bugs below were found in `kwcoco/coco_dataset.py`, `kwcoco/_dataset_components/`,
`kwcoco/_helpers.py`, `kwcoco/coco_objects1d.py`, `kwcoco/compat_dataset.py`.
Items marked `[verified]` were reproduced empirically during the audit.
The oracle for index bugs is `dset._check_integrity()` /
`dset._check_index()` — use it as the assertion in every regression test here.
Add new tests to `tests/test_index_integrity.py` (new file) unless noted.

## A. Index-corruption bugs (highest priority)

### Task 3.1 [verified] — Duplicate-id detection never fires for cats/kpcats/tracks

`kwcoco/_dataset_components/mixin_index.py:481` tests `if cid in cat:`
(membership against the category *dict itself*, i.e. its string keys), `:491`
uses `if cid in obj:`, `:536` uses `if tid in track:`. Should be
`if cid in cats:`, `if cid in kpcats:`, `if tid in tracks:` (the
images/videos/annotations loops at `:501,:511,:521` are correct). Loading a
dataset with two categories sharing an id silently drops one from the index.
Bonus defect: if the checks fired, the warning messages would crash because
`cats[cid]` is read before assignment — fix the message construction too.

- [ ] Fix the three membership targets and the warning messages.
- [ ] Regression test: build a `CocoDataset` from a dict with duplicate
      category ids (and separately duplicate track ids) and assert
      `pytest.warns` fires per table.

### Task 3.2 [verified] — Bulk `add_images` fails to maintain `vidid_to_gids`

`kwcoco/_dataset_components/mixin_index.py:251`: `if index.vidid_to_gids:` is
falsy for an empty dict, so adding images with `video_id` to a dataset with no
pre-existing videos skips the video-index update entirely. Repro:
`CocoDataset().add_images([{'id':1,'file_name':'a.png','video_id':1,'frame_index':0}])`
then `_check_index()` fails with `['vidid_to_gids']`. Even when non-empty,
`:257` `index.vidid_to_gids[vidid].update(gids)` raises `KeyError` for a video
id absent from the index, whereas the single-image `_add_image`
(`:211-222`) warns and creates the entry.

- [ ] Change the truthiness check to `is not None` and mirror the
      single-image fallback for unknown vidids. Also rename the shadowed
      inner `gids` at `:256`.
- [ ] Regression test: bulk `add_images` with video_ids into (a) an empty
      dataset, (b) a dataset where the video table lacks the id →
      `_check_integrity()` passes in both.

### Task 3.3 [verified] — `clear_images()` leaves three stale index structures

`CocoIndex._remove_all_images` (`mixin_index.py:348-358`) clears `imgs`,
`anns`, `gid_to_aids`, `file_name_to_img` and the *values* of
`cid_to_aids`/`vidid_to_gids`, but not `name_to_img`, `trackid_to_aids`, nor
`cid_to_aids` keys. After `clear_images()` on a dataset with a named image and
a tracked annotation, `_check_index()` fails with
`['cid_to_aids', 'name_to_img', 'trackid_to_aids']`, and re-adding an image
with the same name raises `DuplicateAddError`.

- [ ] Clear `name_to_img` and reset `trackid_to_aids` values (annotations are
      all gone, so all track buckets must be emptied) in
      `_remove_all_images`. Audit `_remove_all_annotations` (same file) for
      the analogous gap (it does clear tracks — confirm).
- [ ] Regression test: `vidshapes`-style demo (has names + tracks) →
      `clear_images()` → `_check_integrity()`; then re-add an image with a
      previously-used name.

### Task 3.4 [verified] — `remove_videos` fallback path (no index) is broken

`mixin_add_remove.py:1125-1130`: when `self.index.vidid_to_gids` is falsy the
fallback iterates `self.dataset['videos']` looking up `ann['image_id']`
(→ `KeyError: 'image_id'`). It should iterate `self.dataset['images']`
selecting `img.get('video_id') in remove_vidids`. Sibling fallbacks have the
same class of bug: `remove_tracks` (`:987-991`) uses `ann['track_id']` without
`.get` (KeyErrors on untracked annotations); `remove_categories` (`:892-896`)
KeyErrors on annotations whose `category_id` was previously popped.

- [ ] Fix all three fallbacks to iterate the right table with `.get`.
- [ ] Regression tests: for each of the three methods, build a demo dataset,
      call `dset.index.clear()`, then call the remover and assert correct
      dataset contents (the index is rebuilt lazily afterwards).

### Task 3.5 [verified] — `add_image` / `add_annotation` half-commit on invalid input

Two ordering bugs leave partial state when validation fails:

1. `CocoIndex._add_image` mutates the index (`mixin_index.py:196-203`)
   *before* validating that `video_id` implies `frame_index` (`:205-209`).
   A failed add leaves phantom entries in `index.imgs`, `gid_to_aids`,
   `file_name_to_img` while `dataset['images']` stays empty.
2. `add_annotation` (`mixin_add_remove.py:352-353`) appends to
   `dataset['annotations']`, then `index._add_annotation` sets
   `index.anns[aid]` (`mixin_index.py:263`) before
   `index.gid_to_aids[gid].add(aid)` (`:265`) raises `KeyError` for a
   nonexistent `image_id`.

- [ ] Validate before mutating in both paths (move the frame_index check to
      the top of `_add_image`; check `image_id` existence at the top of
      `add_annotation` when an index is built, raising `InvalidAddError`).
- [ ] Regression tests: attempt both invalid adds inside `pytest.raises`,
      then assert `_check_integrity()` passes and that a subsequent valid
      add of the same file_name/aid succeeds.

### Task 3.6 [verified] — `remove_tracks(keep_annots=True)` leaves index inconsistent

`mixin_add_remove.py:993-997` pops `track_id` from surviving annotations, then
`_remove_tracks` deletes `trackid_to_aids[tid]` wholesale
(`mixin_index.py:388`). A rebuilt index would place those aids under
`trackid_to_aids[None]`; the live index does not.

- [ ] **⚠️ DECISION** — recommended: set `ann['track_id'] = None` instead of
      popping (matches `add_annotation` semantics, see also Task 3.14), and
      move the orphaned aids into the `None` bucket in `_remove_tracks`.
- [ ] Regression test asserts `_check_integrity()` after
      `remove_tracks(..., keep_annots=True)`.

### Task 3.7 [verified] — `remove_annotations` crashes on annotations lacking `category_id`

`CocoIndex._remove_annotations` (`mixin_index.py:376-379`):
`cid = ann.get('category_id', None)` then `index.cid_to_aids[cid].remove(aid)`
raises `KeyError` when the aid is in no bucket (happens after
`remove_categories(..., keep_annots=True)` popped the key and deleted the
bucket).

- [ ] Use `.discard(aid)` on a `.get(cid)`-guarded bucket (and same for the
      other grouped structures in that function — audit `trackid_to_aids`
      and `vidid_to_gids` handling for the same pattern).
- [ ] Regression test: `remove_categories([cid], keep_annots=True)` then
      `remove_annotations([aid])` → succeeds, `_check_integrity()` passes.

## B. `union` bugs

### Task 3.8 [verified] — Track-id remapping collides with real track-table ids

`coco_dataset.py:1234` creates `hacked_track_id_map = _ID_Remapper(reuse=False)`
whose new ids start at **0** (`_helpers.py:87`), while table-backed tracks get
`len(merged['tracks'])+1` or preserved ids (`coco_dataset.py:1296-1299`).
Unioning a dataset that has a tracks table with one that has bare `track_id`s
produces colliding ids → unrelated tracks silently merge (repro produced ann
track ids `[1, 0, 1]` where the last two should be distinct).

- [ ] Seed the hacked remapper past the max of all existing/planned track ids
      (or make it consult the set of ids reserved by table-backed tracks).
- [ ] Regression test in `tests/test_union.py`: union(table-tracked dset,
      bare-track_id dset) → the number of distinct track ids equals the sum
      of distinct input tracks.

### Task 3.9 [verified] — Union crashes on old-style flat keypoints when keypoint_categories exist

`coco_dataset.py:1453-1463`: when `kpcat_id_map` is non-empty every
annotation's `keypoints` is assumed to be new-style (list of dicts); a legacy
flat list `[x, y, v, ...]` raises `TypeError: 'int' object is not subscriptable`.

- [ ] Detect flat lists and pass them through unchanged (they don't reference
      kpcat ids so no remap is needed).
- [ ] Regression test in `tests/test_union.py` mixing both keypoint styles.

## C. Other correctness bugs

### Task 3.10 — `conform(mmlab=True)` never sets `instance_id`

`mixin_stats.py:172`: `for ann in enumerate(self.dataset['annotations']):`
iterates `(index, dict)` tuples so `'track_id' in ann` is always False.
Drop the `enumerate`. Extend `tests/test_conform.py::test_conform_mmlab` to
assert `instance_id` is set for tracked annotations.

### Task 3.11 — `CocoImageView.__getitem__` dead code breaks slicing

`coco_objects1d.py:1247-1252`: an early `return` on line 1248 makes the
slice-handling branch unreachable; `dset.images().coco_images[0:2]` raises
`TypeError`. Delete the early return; add a doctest slicing a view.

### Task 3.12 [verified] — `_cached_hashid_for` is dead *and* broken

`mixin_hash.py:269-304`: on a cache miss `cache_miss` is unbound
(`UnboundLocalError`); on a hit the hashid is computed but never returned.
No caller exists in the repo. **⚠️ DECISION** — recommended: delete the
method (doc 09 collects dead code, but since this one is broken, remove it
here).

### Task 3.13 — `reroot` (and friends) do not invalidate the hashid

`mixin_extras.py:587-892`: `reroot` rewrites every `file_name` but never calls
`self._invalidate_hashid()` (compare `rename_categories`, `mixin_extras.py:564`),
so `_cached_hashid()` can serve a pre-reroot hash. Same gap: `_ensure_imgsize`
(`mixin_extras.py:153-155`) and `conform` add width/height/iscrowd/area
without invalidation; `remove_keypoint_categories`
(`mixin_add_remove.py:1175-1225`) also never invalidates.

- [ ] Add `_invalidate_hashid()` calls (with the appropriate part keys, e.g.
      `['images']`) to `reroot`, `_ensure_imgsize`, `conform`,
      `remove_keypoint_categories`.
- [ ] Extend `tests/test_cached_hashid.py`: hash → reroot → hash differs.

### Task 3.14 — `validate` misses `assets`-keyed images; config handling drift

`mixin_stats.py:423,430`: `subtable_keys=['asset', 'auxiliary']` — the real
modern key is `'assets'` (plural; see `_get_img_auxiliary`,
`mixin_accessors.py:154-157`), so `assets`-style images are never checked.
Also `:415` `config.get('img_attrs', False)` defaults False while the
docstring (`:257-260`) claims `'warn'`; `:435-437` computes `required` for
annot_attrs then hardcodes `required=False`.

- [ ] Fix the key list, align the defaults with the docstring, use the
      computed `required`.
- [ ] Test: validate a dataset whose image uses `assets` with a missing
      width → warning fires.

### Task 3.15 — `safe=True` semantics for `remove_annotations`/`remove_images`

Docstrings (`mixin_add_remove.py:794-795,1026-1027`) claim `safe` filters
"duplicates and non-existing identifiers", but the code only dedups
(`:821`, `:1053`); a nonexistent id raises `KeyError`. `remove_categories` /
`remove_tracks` *do* filter via try/except. **⚠️ DECISION** — recommended:
make the code match the docs (filter nonexistent ids when `safe=True`),
consistent across all four removers. Add tests for each.

### Task 3.16 — `remove_annotation_keypoints` unguarded access

`mixin_add_remove.py:1164-1171`: `ann['keypoints']` KeyErrors on annotations
without keypoints and `kp['keypoint_category_id']` TypeErrors on old-style
flat lists. Use `ann.get('keypoints', [])` and skip non-dict entries. Test via
`remove_keypoint_categories(clean_anns=True)` on a mixed dataset.

### Task 3.17 [verified] — `CocoDataset.coerce` rejects subclass instances

`mixin_constructors.py:75-126`: `if isinstance(key, cls): self = key` is not
part of the elif chain, so a subclass instance falls through to the
`TypeError` at `:126` despite `self` being assigned. Return early on the
isinstance hit. Doctest with a trivial subclass.

### Task 3.18 [verified] — `compat_dataset.COCO.info()` crashes

`compat_dataset.py:54`: `self.dataset.get('info', {}).items()` — kwcoco
initializes `info` as a **list** (`coco_dataset.py:290`). Handle both list
(iterate entries) and dict (legacy) forms. Add a test in
`tests/test_compat_dataset.py`.

### Task 3.19 [verified] — `demo()` shadows `key` when parsing vidshapes suffixes

`mixin_constructors.py:299-340` reuse loop variable `key` for suffix tokens,
so the bundle dpath tag at `:366` gets e.g. `multispectral_<hash>` instead of
`vidshapes8-msi_<hash>`. Rename the inner loop variable. Assert the bundle
name prefix in a test (cheap: check `dset.fpath` contains the original key).

### Task 3.20 — `_normalize_intensity_if_needed` stretches nearly every image

`mixin_draw.py:436-441`: operator precedence makes the elif
`max_value > 0 or min_value < 0` match any image with a positive pixel, so
`draw_image` contrast-stretches ordinary uint8 images. Intent (cf.
`show_image` guard at `:345`) is to normalize only out-of-range data. Fix the
condition to mirror `show_image` (`max > 255` for int-ish, or float outside
[0,1]); eyeball a `draw_image` of a plain uint8 demo image before/after.

### Task 3.21 — `compat_dataset.COCO.download` reroots with check=True

`compat_dataset.py:380-384`: `self.reroot(tarDir)` raises for not-yet-
downloaded images (reroot defaults `check=True`), defeating `download`.
Pass `check=False` (and reroot *after* downloading).

### Task 3.22 — `remove_annotation` (singular) nukes the index

`mixin_add_remove.py:763-783` calls `self.index.clear()` and never rebuilds —
every subsequent accessor fails until the user manually calls
`rebuild_index()` (see `tests/test_sseg_metrics.py:67-68` working around
this). Delegate to `remove_annotations([aid])` which surgically updates the
index. Its docstring example currently exercises the plural method — fix
that too.

## Verification (whole doc)

```
python -m pytest tests/test_index_integrity.py tests/test_cocodataset.py \
    tests/test_union.py tests/test_conform.py tests/test_compat_dataset.py \
    tests/test_cached_hashid.py -q
python -m pytest --xdoctest kwcoco/coco_dataset.py kwcoco/_dataset_components -q
python run_tests.py   # full suite at the end
```

Reference: audit's index-integrity matrix — after this doc, every row of the
mutator table must be "correct": `add_image` (invalid), `add_annotation`
(bad gid), bulk `add_images` (video_ids), `clear_images`, unindexed fallbacks
of `remove_videos/tracks/categories`, `remove_tracks(keep_annots=True)`,
`remove_annotations` (missing category_id).
