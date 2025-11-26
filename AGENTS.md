# Agent Notes for the KWCOCO Repository

This repository provides **kwcoco**, a Python package and CLI for reading, writing, validating, and manipulating COCO-style datasets, extended for **video**, **multispectral imagery**, **annotation tracks**, **line annotations**, **vectorized interfaces**, and richer **evaluation metrics**. It is a **strict superset of MS-COCO** with a focus on speed, pure-Python usability, and flexible multimodal data access.

Packaging is currently **`setup.py`-driven** even though a `pyproject.toml` exists; keep both in sync.

---

## Repository Orientation
- **User documentation** lives in `docs/source/manual/` (concepts, bundles, coordinate spaces, vectorized interfaces, getting started, gotchas). `README.rst` provides the feature overview and CLI synopsis.
- **Examples** (`examples/`) contain runnable scripts for dataset modification, PyTorch dataloading, image access, multispectral/video usage, and visualization. Good for smoke tests and usage references.
- **Dev scratch space** (`dev/`) holds personal experiments/benchmarks not shipped with the package; safe to ignore unless seeking prior art.
- **Tests** in `tests/` plus extensive doctests in docstrings. Use `python run_tests.py` for full coverage; `./run_doctests.sh` for doctests only.

---

## Code Layout
- **`kwcoco/coco_dataset.py`** – the central `CocoDataset` class (mixin-composed).  
  Supports loading, editing, hashing, fast indexing, videos (ordered frame lists), multispectral assets (channel/sensor specs), annotation tracks, multiple segmentation/keypoint encodings, and path rerooting/bundle logic.  
  Maintains a raw `dataset` dict plus an indexed view (`self.index`) with cached lookups (`anns`, `imgs`, `cats`, `gid_to_aids`, etc.).
- **CLI (`kwcoco/cli/`)** – (stats, union/subset/split, validate, conform, reroot/move, show, toydata, eval, find missing images, dataset grabbing). `__main__.py` registers subcommands.
- **Data helpers (`kwcoco/data/`)** – accessors for select standard datasets.
- **Formats (`kwcoco/formats/`)** – conversion tools to/from other formats.
- **Demo/toydata (`kwcoco/demo/`)** – synthetic dataset generators used in docs/tests (shapes, multispectral, video).
- **Metrics (`kwcoco/metrics/`)** – COCO-style classification/segmentation/detection metrics with configurable behavior.
- **Resource registry (`kwcoco/rc/`)** – bundled resources / requirement files.
- **Utilities (`kwcoco/util/`)** – delayed image access, geometry/warping, channels, vectorized ops; heavy use of `ubelt`, `numpy`, and `delayed_image`.

---

## Working With `CocoDataset`
- Instances are composed from mixins (add/remove, constructors, hashing, indexing). Consult mixins for behavior details.
- Indexing is cached; direct mutation of nested structures requires index invalidation/rebuild.
- `bundle_dpath` and `img_root` control how relative paths resolve; use built-in `reroot` helpers, not ad-hoc path edits.
- JSON I/O uses Python’s `json`, optionally `ujson` for reads when `KWCOCO_USE_UJSON` is set. Supports reading from compressed archives (zip).
- When adding items, methods return integer IDs; editing/removal is in-place.

---

## Documentation Highlights
- **Getting started** (`manual/getting_started.rst`) explains design goals, strict-superset semantics, and basic API usage.
- **Concept docs** describe bundle layout, warping/coordinate conventions, vectorized interfaces, and common pitfalls.
- **How-to guides** show delayed image access and JSON querying tricks.

---

## Development Practices
- Modify **`setup.py`** when changing packaging; keep it consistent with `pyproject.toml`.
- Follow existing CLI patterns in `kwcoco/cli/` when adding commands; update docs/examples for user-facing behavior.
- CLI uses `scriptconfig`.
- Rich docstrings often contain doctests—consult them when modifying behavior.
- Use existing helpers (`kwcoco.util`, `_helpers`) instead of writing new ad-hoc utilities.

---

## Testing & Quality
- Full tests: `python run_tests.py` (pytest + xdoctest + coverage).
- Doctests only: `./run_doctests.sh`.
- Optional lint: `./run_linter.sh` (flake8 strict errors only).
- Style emphasizes explicit imports, rich docstrings, and consistent use of `ubelt`.

---

## Additional Resources
- `kwcoco/demo/` and `kwcoco toydata` CLI provide quick synthetic datasets.
- Examples and docs contain most behavior demonstrations; prefer them as the authoritative reference.
- Interoperability with the original `pycocotools`: KWCOCO is a superset of MS-COCO and can export a `pycocotools.coco.COCO` object via `CocoDataset._aspycoco`; the `kwcoco.compat_dataset.CompatCocoDataset` wrapper exposes a pycocotools-like API, and evaluation helpers can call into pycocotools when using conforming datasets. Key differences: KWCOCO allows in-place dataset edits, multimodal/video assets, and richer metrics, whereas pycocotools focuses on static single-image datasets. When needed, the `kwcoco conform` CLI ensures required fields (area ranges, encoding expectations) match pycocotools conventions.
