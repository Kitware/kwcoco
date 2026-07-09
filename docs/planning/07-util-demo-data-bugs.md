# 07 — util / demo / data / formats bugs + security hardening

Status: TODO
Risk: low (mostly leaf modules); one demo fix changes toydata pixel output
Prereqs: 01

## A. Broken deprecation shims (user-facing crash)

### Task 7.1 [verified] — `kwcoco.kpf` / `kwcoco.kw18` raise on import

`kwcoco/kpf.py:5-13` and `kwcoco/kw18.py:5-13` call
`ub.schedule_deprecation(..., error='0.9.0')` at module import and the
package is at 0.9.0, so `import kwcoco.kpf` raises. Worse, the shims never
re-exported anything (`from kwcoco.formats.kpf import *` is absent), so they
were broken even during the warning window. **⚠️ DECISION** — recommended:
make the shims actually forward (`from kwcoco.formats.kpf import *  # noqa`)
with `error='1.0.0', remove='1.1.0'`, since the advertised removal is 0.10.0
and they never worked as shims. Add a test that
`kwcoco.kpf.coco_to_kpf` is reachable under a deprecation warning.

## B. Data-grab bugs (kwcoco/data)

### Task 7.2 — `grab_voc.extract_tarfile` extracts the wrong archive

`data/grab_voc.py:238-247`: the helper takes `fpath` but the body opens
`fpath1` (closure capture) — every call re-extracts the devkit archive, so
VOC2007/2012 image data is never extracted. Also `tar` is unbound in
`finally: tar.close()` if `open` raises. Rewrite as
`with tarfile.open(fpath) as tar:` (+ see Task 7.10 for the extraction
filter). Also fix the copy-paste at `:310`:
`'_train-2012': join(dpath, 'voc-train-2007.mscoco.json')` → `...-2012...`.

### Task 7.3 — `grab_camvid.main` assigns the dataset as its own fpath

`data/grab_camvid.py:644`: `coco_dset.fpath = coco_dset` (line 643 computes
the real `fpath` then discards it). Fix to `coco_dset.fpath = fpath`.

### Task 7.4 — `grab_domainnet` corrupts the image index

`data/grab_domainnet.py:147`: `coco_dset.index.imgs['domain'] = domain`
writes a string key into the gid→img index (clobbered per iteration, never
serialized). Intended: set `domain` on each image dict (best: pass it via
`add_image(...)` so it persists). Also `:132`: a `print` per manifest line
inside the loop — delete (it also defeats the ProgIter above it).

### Task 7.5 — `grab_cifar` stamp renewed before dump

`data/grab_cifar.py:70-71`: `stamp.renew()` precedes `coco_dset.dump(...)`;
a crash between them leaves a stamp claiming the product exists → next run
crashes loading a nonexistent file. Swap the order (cf. the correct pattern
in `grab_camvid.py:226-237`). Also `:47,:55`: `file_name` stored as
`PosixPath` — use `os.fspath`.

## C. Demo bugs (kwcoco/demo)

### Task 7.6 — boids damping is a no-op

`demo/boids.py:367`:
`self.acc[:] *= max(1, min(0, (1 - self.config['damping'])))` always
multiplies by 1 (`min(0, positive)==0`, `max(1,0)==1`). Intended:
`max(0.0, min(1.0, 1 - damping))`. **⚠️ DECISION** — fixing changes
deterministic toydata trajectories, which may break hash-pinned doctests
(e.g. `toydata_video.py:399,423`). Recommended: fix the formula AND update
any doctest hashes that change in the same commit; note the change in the
changelog (demo data only).

### Task 7.7 — toydata parameter bugs

- [ ] `demo/toydata_video.py:24-228`: `random_video_dset(background=...)` is
      accepted and silently ignored (never merged into `renderkw`, which
      `render_toy_image:1117` reads). Inject it into `renderkw`; test that
      `background='parrot'` changes the rendered frame.
- [ ] `demo/toydata_image.py:485` and `demo/toydata_video.py:463`:
      `'unknown kwargs={}'.format(**kwargs)` raises IndexError exactly when
      the assert fires; use `.format(kwargs)` (as at `toydata_image.py:144`).
- [ ] `demo/toydata_video.py:834`: `sorted(imgs, key=lambda g:
      g.get('frame_index', None))` TypeErrors when any image lacks
      `frame_index`; use a sentinel like `(g.get('frame_index') is None,
      g.get('frame_index', -1))`.
- [ ] `demo/toypatterns.py:171-178`: `CategoryPatterns.get` catches
      `KeyError` but list indexing raises `IndexError`; catch the right one.
- [ ] `demo/toypatterns.py:379`: empty-keypoint fallback uses shape-`(0,)`
      `xy`; use `.reshape(0, 2)` (cf. correct code at `:275`).
- [ ] `demo/perterb.py:263`: `chan_data + chan_data.clip(0, 1)` discards its
      result — intended `chan_data = chan_data.clip(0, 1)`.
- [ ] `demo/perterb.py:206-207`: stores a raw `kwimage.MultiPolygon` as
      `ann['segmentation']`; convert with `.to_coco(style='new')` like the
      true-annotation path at `:168`.

## D. util bugs

### Task 7.8 — `util_monkey.SupressPrint` reads/writes the wrong object

`util/util_monkey.py:23`: `getattr(self.mods, 'print', print)` queries the
*tuple*; and `__exit__` then *sets* a `print` attribute (builtin) on modules
that never had one, permanently. Fix: `getattr(mod, 'print', print)` per
module and delete the attribute on exit when it didn't exist. Used by
`metrics/detect_metrics.py:867`. Add a unit test (module with and without a
custom `print`).

### Task 7.9 — `util_archive` hardcodes gzip for tar

`util/util_archive.py:280` (`unarchive_file`) and `Archive._open`
(`:142-146`) use `'r:gz'` after only checking `is_tarfile` — plain `.tar` /
`.tar.bz2` / `.tar.xz` fail with `ReadError`. Use `'r:*'` for reading. Add
tests for `.tar` and `.tar.xz` round trips (this module is used by
`coco_dataset.py:666`).

### Task 7.10 — Archive-extraction path-traversal hardening (SECURITY)

Pre-3.12 `tarfile.extractall` permits `../` traversal and absolute-path
writes from malicious archives. Affected: `util/util_archive.py:243-255`
(`Archive.extractall`) and `:268-304` (`unarchive_file`);
`data/grab_voc.py:245` (`tar.extractall(dpath)` — rewrite in Task 7.2).

- [ ] Pass `filter='data'` when available (Python ≥3.12 or backported
      tarfile); otherwise validate each member: reject absolute names and
      normalized paths escaping the output dir. Do it centrally in
      `util_archive` and route `grab_voc` through it.
- [ ] Also add `hash_prefix` to the `ub.grabdata` calls in
      `data/grab_voc.py:249,258,270,281` (mirror `grab_domainnet.py:103`,
      which does this correctly) and switch the plain-`http://` Oxford URLs
      to https where available.
- [ ] Test: crafted tar with a `../evil.txt` member → extraction raises.

### Task 7.11 — util small-fix batch

- [ ] `util/util_eval.py:71` — `restricted_eval` mutates the caller's
      `local_dict` (writes `__builtins__` into it); copy first. Also rename
      docstring param `max_char` → `max_chars` and drop the leaked
      "num-workers-hueristic" caller-specific error text (`:55-59`).
- [ ] `kwcoco/exceptions.py:4-25` — make `DuplicateAddError` and
      `InvalidAddError` subclass `AddError` (currently siblings, so
      `except AddError` doesn't catch them despite the naming). Add a
      hierarchy test.
- [ ] `util/dict_like.py:82` — `DictLike.__iter__` does
      `return self.keys()` which only works if `keys()` yields; use
      `iter(self.keys())` (dict_proxy2 already fixed this).
- [ ] `util/util_sklearn.py:78-79` — use the computed-but-unused
      `y_inversed` so `n_classes = len(unique_y)` instead of
      `max(unique_y)+1` (breaks on non-integer labels; allocates huge
      bincounts for sparse ids).
- [ ] `util/rc/registry.py:22-24` — returned path escapes its
      `importlib_resources.path` context manager (invalid for zip installs);
      use `importlib.resources.files(...)`/`as_file` and copy, or document
      the limitation.
- [ ] `formats/voc.py:36` — `file_name` stored as `ub.Path`; use
      `os.fspath`. `:51` — `assert depth == 3` crashes on grayscale; warn
      and record instead. `:70` — `bbox is None` is a no-op expression
      (missing assignment); `:84-85` dead branch — clean both.
- [ ] `formats/labelme.py:312-318` — `reroot(absolute=True)` default
      immediately asserts `not absolute`; implement or drop the parameter,
      and fix the no-op relative logic. Fix the `LabeMeFile` typo at
      `:196,257,288,334`.
- [ ] `util/util_truncate.py:64-77` — remove docstring params that don't
      exist (`word_boundary`, `save_order`).

## Verification (whole doc)

```
python -m pytest tests/test_util_json.py tests/ -q -k 'archive or monkey or exceptions'
python -m pytest --xdoctest kwcoco/util kwcoco/demo kwcoco/formats -q
python -c "import kwcoco.kpf"   # deprecation warning, not error (task 7.1)
kwcoco toydata vidshapes2 --dst $SCRATCH/v.json   # demo still renders
```

Note: `kwcoco/data/*` grab scripts are network-bound; their fixes are
verified by unit tests of the pure helpers (extract, convert) against tiny
synthetic trees, not by real downloads.
