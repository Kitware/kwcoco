# AGENTS.md

## Purpose

**kwcoco**: Python package + CLI for COCO(-like) datasets, extended for **video**, **multispectral**, **tracks**, and richer **metrics**. Supports read/write/validate/transform and MS-COCO compatibility.

## Build / Packaging

* `setup.py` = authoritative build/install behavior.
* `pyproject.toml` = metadata + pytest/coverage/mypy hints. Keep in sync when changing deps/options.
* **Python**: min 3.8; CI targets latest CPython too. Avoid upper bounds on Python/deps unless a real break forces it.

## Layout (high-signal)

* `kwcoco/`

  * `coco_dataset.py`, `_helpers.py`, `abstract_coco_dataset.py`: core `CocoDataset`, indexing cache, bundle/reroot, hashing, mutations.
  * `coco_image.py`, `channel_spec.py`, `sensorchan_spec.py`: image + channel/sensor metadata, multispectral helpers.
  * `compat_dataset.py`: pycocotools-like API + conversion compatibility.
  * `coco_evaluator.py`, `coco_objects1d.py`, `metrics/`: eval + metrics (**metrics has its own AGENTS.md**).
  * `cli/`: `python -m kwcoco` / `kwcoco` (`scriptconfig` CLI: stats/union/subset/split/validate/conform/reroot/toydata/demodata/eval/grab/info/visual_stats/etc.).
  * `formats/` converters; `demo/` synthetic data; `data/` dataset access; `rc/` resources; `util/` geometry/delayed-image/vectorized ops.
  * schema refs: `coco_schema.*`, `coco_schema_informal.rst`; specialized: `kw18.py`, `kpf.py`.
* `tests/`: pytest + xdoctest; mirrors major features (CLI, dataset ops, video/multispectral, metrics, SQL, tracks, etc.).
* `examples/`: runnable transforms / dataloading / viz.
* `docs/`: Sphinx (`docs/source/manual/` concepts; `docs/source/auto/` API).
* `dev/`: scratch (not shipped).
* Root scripts: `run_tests.py`, `run_doctests.sh`, `run_linter.sh`, `run_developer_setup.sh`, `clean.sh`, `publish.sh`.
* Deps/config: `requirements/` + `requirements.txt`, `pytest.ini`, `pyproject.toml`, `MANIFEST.in`.

## Setup

* Python 3.8+
* Dev install (preferred): `./run_developer_setup.sh`
* Manual: `pip install -e .[headless]`
* Optional dep groups in `requirements/` (gdal/graphics/linting/optional/postgresql/tests/docs).
* Env var: `KWCOCO_USE_UJSON=1` for faster JSON.

## Compatibility + Conventions

* Maintain **MS-COCO superset** behavior; don’t break existing COCO datasets.
* Prefer current names over deprecated aliases:

  * `bundle_dpath` (deprecated alias: `img_root`)
  * `video_ids` (deprecated: `vidids`)
  * `add_asset` (compat alias: `add_auxiliary_item`)
* Imports: don’t hide missing deps with try/except.
* **Index integrity**: `CocoDataset` has raw `dataset` + cached `index` (`imgs`, `anns`, mappings, etc.). If you mutate internals, use existing helpers to invalidate/rebuild; avoid manual patching of caches.
* Bundles/paths: prefer bundle/reroot utilities over manual path edits.
* Typing: repo ships `.pyi` + `py.typed`; keep stubs/signatures aligned with code.

## Testing / Quality

* Full suite: `python run_tests.py` (pytest + `--xdoctest` + coverage).
* Doctests: `./run_doctests.sh`
* Lint: `./run_linter.sh`
* Targeted: `pytest tests/test_<x>.py -k <expr>` or `pytest kwcoco/<file>.py`
* SQL/Postgres tests require DB prereqs (skip/adjust env if unavailable).
* Examples + `kwcoco demo` / `kwcoco toydata` are good smoke tests.

## Docs

* Build: `make -C docs html` (needs docs deps from `requirements/docs.txt`)
* Keep `README.rst` aligned with major behavior changes.
* Metric-specific guidance: `kwcoco/metrics/AGENTS.md`

## Change Guidelines

* Touching metrics: follow `kwcoco/metrics/AGENTS.md`; preserve COCO conventions + tests.
* Schema changes: update schema refs + validators + CLI validate/conform behavior.
* Add docs/tests with new behavior (doctests + pytest). Keep tests small and near affected features.
* Performance: avoid copies; prefer vectorized ops / delayed images; be careful with indexing costs.

## Quick Commands

* CLI: `python -m kwcoco` or `kwcoco`
* Toy data: `kwcoco toydata --help` (also `kwcoco/demo/`)
* SQL backend: `coco_sql_dataset.py`
* Resources: `kwcoco/rc/`
