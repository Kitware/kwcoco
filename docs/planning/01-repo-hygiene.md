# 01 — Repo hygiene

Status: TODO
Risk: none (deletions of generated files, mechanical text fixes)
Prereqs: none

Goal: remove clutter, resolve untracked-file limbo, and land one mechanical
repo-wide fix. Everything here is cheap and unblocks cleaner diffs later.

## Task 1.1 — Commit the assignment_v2 work-in-progress [completed]

The historical audit found that `CHANGELOG.md` documented
`kwcoco.metrics.assignment_v2` while the module and its tests were untracked.
They are tracked in the current branch, so source archives now contain the
prototype described by the changelog.

- [x] Commit `kwcoco/metrics/assignment_v2.py` and
      `tests/test_metrics_assignment_v2.py`. Doc 04 now describes v2 as an
      alternative design with an explicit semantic contract rather than
      assuming every difference from v1 is a correctness defect.

## Task 1.2 — Delete stray root files and gitignore the archive pattern

- [ ] Delete the six `kwcoco-source-*.tar.gz` files at repo root (~13 MB;
      produced by `build_source_archive.sh`, which defaults `--output-dir` to
      the repo root).
- [ ] Add `kwcoco-source-*.tar.gz` to `.gitignore`; optionally change
      `build_source_archive.sh` to default its output into `dist/` (already
      ignored).
- [ ] Delete the untracked file `auto` at repo root (a JSON detection-metrics
      dump). It was created by the `kwcoco eval` output-path bug fixed in
      doc 06 task 6.1 — do not just delete it without that fix or it will
      reappear.
- [ ] Delete `NOTE.md` after transplanting its single line ("kwcoco visualize
      on the sealion data draws black frames without gdal installed") into a
      `Known issues` note in `CHANGELOG.md` or a tracker issue.
- [ ] Delete `tests/test_gdal_probe_tmp.py` (untracked, 34 lines of pure
      `print()` diagnostics, zero asserts, cannot fail — a leftover debug
      probe).
- [ ] Resolve `dev/mwe/sqlalchemy_mutable_default_mwe.py`: the rest of `dev/`
      is tracked scratch, so either commit it there or delete it.
      **⚠️ DECISION** — recommended: commit it (it documents an SQL backend
      pitfall relevant to doc 05).

## Task 1.3 — Fix `from __future__ import annotations` trapped inside docstrings (repo-wide)

An automated refactor inserted `from __future__ import annotations` as the
first line **inside** module docstrings, where it is inert text: it does not
enable lazy annotations, and it pollutes `help()`/Sphinx output. Verified in
30+ files. In 3 of them the real import is *also* present (duplicated text);
in the rest the future-import has no effect at all.

Affected (verified list — re-grep to be safe):

```
grep -rn --include='*.py' -A1 '^"""$' kwcoco/ | grep 'from __future__'
# more robust: find files where the future import is inside the docstring
python -c "
import pathlib, re
for p in pathlib.Path('kwcoco').rglob('*.py'):
    t = p.read_text()
    m = re.match(r'\s*(\"\"\"|\'\'\')' , t)
    if m and 'from __future__ import annotations' in t[:t.find(m.group(1), m.end())] :
        print(p)
"
```

Known members: `_helpers.py`, `coco_objects1d.py`, `compat_dataset.py`,
`category_tree.py`, `coco_image.py`, `coco_sql_dataset.py`,
`sensorchan_spec.py`, `coco_schema.py`,
`util/{__init__,util_kwutil,util_deprecate,util_eval,lazy_frame_backends,util_rich,util_special_json,util_truncate,ijson_ext,util_futures,util_reroot,util_sklearn}.py`,
`util/delayed_ops/__init__.py`, `demo/{toydata,toydata_video,toydata_image}.py`,
`data/{grab_spacenet,grab_datasets,grab_domainnet,grab_cifar,grab_camvid}.py`,
`formats/{__init__,voc,kpf,kw18,labelme}.py`.

- [ ] For each file: remove the line from inside the docstring; ensure a real
      `from __future__ import annotations` exists as the first statement after
      the docstring **iff the file uses PEP 604 / PEP 585 annotations**
      (`X | None`, `list[int]`); dedupe where both forms exist
      (`util_reroot.py`, `util_special_json.py`, `ijson_ext.py`).
- [ ] Relatedly, every file in `kwcoco/cli/` places the `#!/usr/bin/env python`
      shebang on line 3 (after the future import), making it non-functional.
      Either move shebangs to line 1 or delete them (recommended: delete; the
      modules are entry points via `python -m`, not direct executables).
- [ ] Sanity check: `python -c "import kwcoco"` and
      `python -m pytest tests/test_import.py -q`.

## Task 1.4 — Fix stale documentation claims

- [ ] `AGENTS.md:49` claims "repo ships `.pyi` + `py.typed`" — there are zero
      `.pyi` files (removed in commit `c578c69`). Rewrite to: inline
      annotations + `py.typed`.
- [ ] `setup.py:275` — drop the stale `'*.pyi'` glob from `package_data`
      (keep `'py.typed'`).
- [ ] `README.rst:4` — remove the ancient TODO comment "Get CI services
      running on gitlab" (GitLab CI has been running for years).

## Acceptance criteria

- `git status` shows no untracked files at repo root except intentional ones.
- No module docstring contains a `from __future__` line
  (re-run the detection snippet above → empty output).
- `python run_tests.py` passes at least as well as before the change
  (this doc must not alter behavior).
