# AGENTS.md

This guide orients future AI coding agents and developers working in the **kwcoco** repository. It summarizes the project purpose, layout, setup, testing workflow, and conventions so you can quickly make informed changes.

## Project Snapshot
- **Goal**: Python package and CLI for reading, writing, validating, and manipulating COCO-style datasets, extended for video, multispectral imagery, tracks, and richer metrics.
- **Packaging**: `setup.py` is authoritative for builds; `pyproject.toml` holds metadata, coverage, and pytest settings—keep them synchronized when changing dependencies or options.
- **Python policy**: Minimum supported is 3.8 and CI covers up through the latest released CPython (currently 3.13). We intentionally avoid upper bounds on Python and other dependencies unless required by a compatibility break; prefer widening support rather than pinning.

## Repository Layout
- `kwcoco/` – Main package (typed; `.pyi` stubs included). Highlights:
  - `coco_dataset.py` plus helpers in `_helpers.py` and `abstract_coco_dataset.py`: core `CocoDataset` logic, indexing cache, bundle/reroot handling, hashing, and add/remove operations.
  - `coco_image.py`, `channel_spec.py`, `sensorchan_spec.py`: image helpers, channel metadata parsing, multispectral utilities.
  - `compat_dataset.py`: compatibility wrappers (pycocotools-like API, conversion helpers).
  - `coco_objects1d.py`, `coco_evaluator.py`, `metrics/`: evaluation and metrics (classification/detection/segmentation; `metrics` has its own AGENTS.md scoped to that subpackage).
  - `cli/`: `__main__.py` registers the `scriptconfig`-based CLI (stats, union/subset/split, validate, conform, reroot/move, toydata/demodata, eval, grab, info/tables, visual_stats, find_unregistered_images, etc.).
  - `formats/`: converters to/from external formats; `demo/`: synthetic dataset generators used in docs/tests; `data/`: dataset access helpers; `rc/`: bundled resources; `util/`: geometry, delayed-image access, vectorized ops leveraging `ubelt`, `numpy`, and `delayed_image`.
  - `coco_schema.*` & `coco_schema_informal.rst`: schema references; `kw18.py`/`kpf.py`: specialized format support.
- `tests/`: Pytest-based suite plus xdoctests; mirrors major features (CLI, dataset ops, multispectral/video, metrics, SQL, keypoints, tracks, subsets, validation, PostgreSQL integration, etc.).
- `examples/`: runnable scripts showing dataset transforms, PyTorch dataloading, visualization, multispectral/video usage.
- `docs/`: Sphinx sources (`docs/source/manual/` for concepts/how-tos; `docs/source/auto/` for API references). `docs/format_notes.md` and `docs/dataset_api.txt` provide extra notes.
- `dev/`: scratch/experiments not shipped.
- Root helpers: `run_tests.py`, `run_doctests.sh`, `run_linter.sh`, `run_developer_setup.sh`, `clean.sh`, `publish.sh`.
- Config: `pytest.ini` (pytest/xdoctest defaults), `pyproject.toml` (coverage, pytest opts, mypy hints), `requirements/` (grouped dependency sets), `requirements.txt` (aggregated runtime/tests/optional/postgresql), `MANIFEST.in`.

## Environment Setup
1. **Clone** and use Python 3.8+ (latest CPython versions should work; report regressions instead of pinning upper bounds).
2. **Install deps**:
   - Quick dev setup: `./run_developer_setup.sh` (installs `requirements.txt` then `pip install -e .`).
   - Manual: `pip install -r requirements.txt` then `pip install -e .` (editable). Dependency subsets live in `requirements/` (e.g., `gdal.txt`, `graphics.txt`, `linting.txt`, `optional.txt`, `postgresql.txt`, `tests.txt`). GDAL wheels are available via the `girder.github.io/large_image_wheels` index noted in `pyproject.toml`.
3. **Optional extras**: install graphics/backends (OpenCV, matplotlib), GDAL for geospatial rasters, PostgreSQL drivers for SQL backends, and torch-related deps if using dataloaders in examples/tests.
4. **Environment variables**: `KWCOCO_USE_UJSON` enables faster JSON reads; otherwise standard `json` is used.

## Conventions and Deprecations
- **Backward compatibility**: `compat_dataset.py` and CLI tools aim to remain a superset of MS-COCO; new features should not break existing MS-COCO datasets.
- **Preferred over deprecated**:
  - Use `bundle_dpath` for bundle roots; `img_root` is a deprecated alias still accepted by constructors.【F:kwcoco/coco_dataset.py†L6133-L6166】
  - Use `video_ids` instead of the deprecated `vidids` argument in dataset selectors.【F:kwcoco/coco_dataset.py†L2861-L2904】
  - Use `add_asset` for auxiliary imagery; `add_auxiliary_item` remains as a compatibility alias.【F:kwcoco/coco_dataset.py†L4327-L4358】
- **Dependency policy**: avoid adding strict upper bounds unless necessary for a break; prefer ranges that keep future Python and library releases usable.
- **Imports**: follow repo style—do not wrap imports in try/except to hide failures.
- **Index integrity**: `CocoDataset` maintains raw `dataset` data and a cached `index` (`imgs`, `anns`, `cats`, `gid_to_aids`, etc.). Mutations to internal structures should invalidate/rebuild the index using existing helpers. Path resolution respects bundle roots; prefer reroot/bundle utilities over manual edits.
- **Type hints**: `.pyi` stubs and `py.typed` are present; keep signatures synchronized when changing APIs.

## Testing
- **Full suite with coverage**: `python run_tests.py` (runs pytest over `kwcoco/` and `tests/`, enables `--xdoctest`, reports coverage via `pyproject.toml`).
- **Doctests only**: `./run_doctests.sh` (xdoctest pass-through).
- **Lint**: `./run_linter.sh` (flake8-focused strict check).
- **Selective tests**: `pytest tests/test_<module>.py -k <expr>` or `pytest kwcoco/<module>.py` (note `pytest.ini` ignores `dev` and `docs`).
- **SQL/PostgreSQL tests**: require database prerequisites; skip or adjust env if unavailable.
- **Examples as smoke tests**: many scripts in `examples/` can validate usage; `kwcoco demo`/`toydata` CLI also provides quick datasets.

## Documentation
- Primary user/developer docs in `docs/source/manual/` (getting started, coordinate spaces, vectorized interfaces, bundles, gotchas) and `docs/source/auto/` for API refs. Build locally with `make -C docs html` (requires docs dependencies from `requirements/docs.txt`).
- `README.rst` offers overview, CLI synopsis, and feature list; keep in sync with major behavior changes.
- Metric-specific docs live alongside code (see `kwcoco/metrics/AGENTS.md` for scoped guidance).

## Extending & Refactoring Tips
- **Metrics**: follow guidance in `kwcoco/metrics/AGENTS.md` when touching evaluation logic; maintain compatibility with COCO conventions and existing tests.
- **Schemas**: when modifying dataset structure, update `coco_schema.json`/`.pyi` and relevant validators; ensure conform/validate CLI commands reflect new fields.
- **Docs/tests first**: add or update docstrings (doctests) and pytest cases for new behaviors. Prefer small, focused tests under `tests/` mirroring affected modules.
- **Performance**: indexing and delayed image operations prioritize speed; avoid unnecessary copies and prefer vectorized operations or delayed images where possible.

## Quick References
- **CLI entry point**: `python -m kwcoco` or `kwcoco` after install.
- **Toy data**: `kwcoco toydata --help` or use generators in `kwcoco/demo/` for synthetic datasets.
- **SQL**: `coco_sql_dataset.py` provides SQLite/PostgreSQL support; some tests may need DB availability.
- **Resources**: `kwcoco/rc/` holds bundled requirement files and assets consumed by certain commands.

Use this document as the starting point before changing code, adding features, or running tests. When in doubt, consult the relevant module’s docstrings and accompanying doctests.
