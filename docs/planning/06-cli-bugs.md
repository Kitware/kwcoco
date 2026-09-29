# 06 — CLI bugs

Status: TODO
Risk: low-medium (user-facing behavior fixes; mostly isolated per subcommand)
Prereqs: 01

Files: `kwcoco/cli/*`, `kwcoco/__main__.py`. The CLI uses kwconf (`Config` + `ModalCLI`). Existing CLI tests: `tests/test_cli_validate.py`,
`tests/test_cli_visualize_videos.py`, `tests/test_cli_fixup_stats.py`;
`tests/test_kwcoco_cli.py` is permanently skipped (revived in doc 08).

## A. High-severity bugs

### Task 6.1 — `kwcoco eval` writes metrics to a file named `auto`

`cli/coco_eval.py:132-137` tests the wrong key:

```python
if cli_config['out_dpath'] == 'auto':      # out_dpath is never 'auto'
    metrics_fpath = out_dpath / 'metrics.json'
else:
    metrics_fpath = ub.Path(cli_config['out_fpath'])   # == 'auto' default
```

The `'auto'` sentinel belongs to `out_fpath`. Every default invocation dumps
metrics to `./auto` (this produced the stray `auto` file at the repo root —
see doc 01 task 1.2). Fix the condition to test `out_fpath`. Test: run the
eval CLI on demo data in a temp cwd; assert `out_dpath/metrics.json` exists
and no file named `auto` was created.

### Task 6.2 — `__cli__._main()` AttributeError in three modules

`cli/coco_union.py:147`, `cli/coco_toydata.py:147`, `cli/coco_eval.py:384`
call `__cli__._main()`; neither the former scriptconfig backend nor the
current kwconf `Config` API provides `_main` — `python -m kwcoco.cli.coco_union` crashes before
parsing args. The `# type: ignore` comments show the checker flagged it.
Replace with `__cli__.main()`. Test: parametrized subprocess test running
`python -m kwcoco.cli.<mod> --help` for every module in `kwcoco/cli/`
(exit 0) — this also guards the whole class of bug (add to doc 08's file).

### Task 6.3 — `coco_move_assets`: unregistered + help-text-as-default

- `cli/__main__.py:14-34` omits `coco_move_assets` from `modnames` → no
  `kwcoco move_assets` command exists.
- `cli/coco_move_assets.py:24-25`: `src = kwconf.Value('source asset file or folder')`
  passes help text as the default *value* (missing `help=`); an invocation
  without `--src` asserts on `ub.Path('source asset file or folder')`.
- The config class also lacks `__command__ = 'move_assets'` (required by the
  registration loop, `cli/__main__.py:90`).

Fix all three. Add a registration-drift test: every module under
`kwcoco/cli/` exposing `__cli__` (with `__command__`) must appear in
`__main__.modnames` (doc 08 owns the test file; write it now).

### Task 6.4 — `kwcoco split` triple bug

1. `cli/coco_split.py:192-196` — inside
   `for split_num, (idxs1, idxs2) in enumerate(split_idxs):` the line
   `idxs1, idxs2 = split_idxs[0]` clobbers every fold with fold 0, so
   `--num_write N` writes the same split N times. Delete the line. (This
   explains the docstring disclaimer at `:16-17` — remove it once fixed.)
2. `:170-173, 212-214` — the `--train_fpath/--vali_fpath/--test_fpath`
   three-way split advertised in help (`:29-40`) is unimplemented: the
   asserts can't pass with the defaults and the else branch is a bare
   `assert train_fpath is not None  # TODO`. **⚠️ DECISION** — recommended:
   implement the three-way split (two chained `train_test_split`-style
   partitions honoring the same grouping/stratification), since the help
   already promises it. Fallback: error out with NotImplementedError and
   strip the help text.
3. `:123-132` — `splitter='auto'` picks `'video'` when `n_videos > 0`, then
   `images.lookup('video_id')` KeyErrors on mixed datasets with loose
   images. Use `lookup('video_id', default=None)` and group loose images by
   their own id.
Also fix the epilog format-string examples (`:79`): `train_{03:d}` raises
`KeyError` — should be `train_{:03d}`.
Tests: `--num_write 3` produces 3 distinct splits; mixed video+loose dataset
splits without error; (if implemented) 3-way split writes all three files
with disjoint gids.

### Task 6.5 — `kwcoco modify_categories --rename` legacy format crashes

`cli/coco_modify_categories.py:193-211`: `kwutil.Yaml.coerce('old:new,a:b')`
returns the *string* (YAML plain scalar), not None, so the legacy-format
fallback never runs and `rename_categories` gets a str →
`AttributeError: 'str' object has no attribute 'items'`. This exact syntax is
advertised in the module epilog (`:18`) and option help (`:69-71`). Fix:
after coerce, if the result is a `str`, parse the comma/colon format. Also
fix the `except ImportError` branch (`:199-202`) that warns but leaves
`mapper` unbound (NameError). Test: run the CLI with the legacy syntax on
demo data.

### Task 6.6 — `coco_plot_stats` double-compute and width=height

1. `cli/coco_plot_stats.py:127-142`: `build_stats_data(dset)` is called
   unconditionally at `:130` AND again inside the cache-miss branch (`:136`)
   — the expensive stats build (shapely-coerces every annotation) runs twice
   on a miss and the cache never saves anything. Move the first call inside
   the miss branch.
2. `:475-477`: `'box_width': box_height` — copy-paste; every serialized
   table and the `median_box_dsize` stat gets height for width. Fix; test on
   a dataset with asymmetric boxes asserting `box_width != box_height`.
3. `:241-243`: `proc_context.stop()` called twice; delete the first call.

## B. Medium/low bugs

### Task 6.7 — `finish_install` missing-executable check can never fire

`cli/finish_install.py:158-165`: `ub.find_path(exe)` returns a generator,
never None, so `missing` stays empty. Use `ub.find_exe(exe)`. Add a doctest
with a fake exe name.

### Task 6.8 — `kwcoco validate --fix` without `--dst` silently discards fixes

`cli/coco_validate.py:184-213`: help says "dst must be specified" but nothing
enforces it — fixes are computed then dropped. Raise at config-validation
time when `fix` is set without `dst` (and without `inplace`-equivalent).
(The whole fix path is deprecated in favor of `kwcoco fixup` — doc 09
task 9.6 may remove it; still fix the trap now, it's one `raise`.)

### Task 6.9 — `kwcoco fixup --corrupted_assets=only_shape` ignored for aux assets

`cli/coco_fixup.py:160-182`: `only_shape` is passed for the main image job
(`:162-167`) but omitted for the two auxiliary/asset submissions
(`:173-175`, `:179-181`) → full decodes where header reads were requested.
Thread it through; extend `tests/test_cli_fixup_stats.py`.

### Task 6.10 — `kwcoco subset --copy_assets` ignores the `assets` key

`cli/coco_subset.py:219-225`: only `img['auxiliary']` is collected; images
using the modern `assets` list get their files silently skipped → output
dataset references files that were never copied. Iterate both keys (after
doc 05 task 5.1 lands, use the centralized CocoImage accessor). Extend
`tests/test_subset.py` with an assets-style dataset.

### Task 6.11 — `kwcoco info` emits invalid JSON with info + tables

`cli/coco_info.py:303-351`: `prev_parent` is initialized *after* the info
section prints (`:339`), so the first table lacks its preceding comma →
unparseable output, contradicting the "redirectable as a regular json
object" help (`:139`). Initialize the sentinel before the info section. Test:
`kwcoco info f.json -i 1 -g 1` output parses with `json.loads`.

### Task 6.12 — Small fixes batch

- [ ] `cli/coco_reroot.py:226-238` — `_check_candidates` uses
      `str.replace(old_pref, new_pref)` (substring, every occurrence) to
      verify candidates that `reroot` applies as a *prefix*; use
      startswith/removeprefix logic mirroring reroot.
- [ ] `cli/coco_eval.py:247` — dead comprehension with typo'd loop var
      (`stast`); delete. `:244` divides by `stats['num_true']` without a
      zero guard; guard it.
- [ ] `kwcoco/__main__.py:12-14` — wrap in `sys.exit(main())` so
      `python -m kwcoco <cmd>` propagates exit codes (the console script and
      `python -m kwcoco.cli` already do).
- [ ] `cli/coco_stats.py:155`, `cli/coco_modify_categories.py:191,269` —
      `print(nx.write_network_text(...))` prints a stray `None`; call
      without wrapping print.
- [ ] `cli/coco_visualize_videos.py:1048-1056` (`perchannel_colorize`) —
      `data.shape[2]` read before the 2D guard, and the guard adds the axis
      at the front; fix both (currently latent).
- [x] `cli/coco_plot_stats.py` — remove sunlight estimation from the core CLI
      and preserve the workflow in `examples/plot_sunlight.py`.

## Verification (whole doc)

```
python -m pytest tests/test_cli_*.py tests/test_kwcoco_cli.py tests/test_subset.py -q
python -m pytest --xdoctest kwcoco/cli -q
# manual smoke:
kwcoco toydata shapes8 --dst $SCRATCH/toy.json
kwcoco eval --true_dataset ... (per task 6.1 test)
```
