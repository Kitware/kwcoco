# kwcoco Quality Improvement Plan

This directory contains an ordered set of executable planning documents produced
from a full-repo audit (2026-07-09, branch `dev/refactors`, HEAD `be3c174`).
Six parallel deep audits covered: core dataset + mixins, coco_image/SQL/schema,
metrics + evaluator, CLI, util/demo/data/formats, and tests/CI/packaging/docs.

Many findings below were **empirically verified** by executing minimal repros
against the working tree (marked `[verified]` in the plan docs). Treat those as
confirmed; re-verify the rest before fixing (a few may have drifted).

## Execution order

Execute the documents in numeric order. Each is self-contained: an agent can
pick one up, execute it, and stop. Later docs assume earlier docs are done
(e.g. the test-hardening doc assumes the bug fixes landed with their regression
tests).

| Doc | Theme | Effort | Risk |
|-----|-------|--------|------|
| [01-repo-hygiene.md](01-repo-hygiene.md) | Delete clutter, commit/ignore decisions, mechanical repo-wide fixes | Small | None |
| [02-infrastructure.md](02-infrastructure.md) | Packaging, CI, lint/type config consolidation | Small | Low |
| [03-core-dataset-bugs.md](03-core-dataset-bugs.md) | Index-integrity and CocoDataset correctness bugs | Large | Medium |
| [04-metrics-bugs.md](04-metrics-bugs.md) | Numerical correctness in metrics/evaluator; assignment_v2 | Large | Medium |
| [05-image-sql-schema-bugs.md](05-image-sql-schema-bugs.md) | CocoImage assets, SQL backend, schema sync | Large | Medium |
| [06-cli-bugs.md](06-cli-bugs.md) | CLI subcommand bugs and registration drift | Medium | Low |
| [07-util-demo-data-bugs.md](07-util-demo-data-bugs.md) | util/demo/data/formats bugs + security hardening | Medium | Low |
| [08-test-hardening.md](08-test-hardening.md) | Systemic test-suite improvements beyond per-bug regression tests | Medium | Low |
| [09-dead-code-and-deprecations.md](09-dead-code-and-deprecations.md) | Dead code removal, deprecation consistency, vendored-code retirement | Medium | Medium |
| [10-performance.md](10-performance.md) | Perf fixes (SQL N+1, eager imports, hot loops) | Medium | Medium |
| [11-api-consistency.md](11-api-consistency.md) | Naming/behavior consistency polish across the API surface | Medium | Medium |

## Global conventions (apply to every doc)

- **Python floor is 3.8** (`setup.py python_requires='>=3.8'`). No `match`
  statements, no 3.9+ stdlib. PEP 604 unions (`X | None`) are OK only in files
  with a real (non-docstring!) `from __future__ import annotations`.
- **Do not break MS-COCO superset behavior** (AGENTS.md). Existing COCO
  datasets must keep loading and round-tripping.
- **Index integrity**: `CocoDataset` holds a raw `dataset` dict plus a cached
  `index`. Any mutation must keep them in sync; `dset._check_integrity()`
  (which calls `_check_index`) is the oracle. Use it liberally in new tests.
- **Test-first for bug fixes**: for each bug, write a failing regression test
  (pytest in `tests/`, or a doctest if it reads better near the code), confirm
  it fails, then fix, then confirm it passes. Bug-fix docs list suggested test
  contents.
- **Deprecations** use `ub.schedule_deprecation(...)` with explicit
  `deprecate`/`error`/`remove` versions — never bare `warnings.warn`.
  Current version is 0.9.0 (see `kwcoco/__init__.py`); schedule new
  deprecations as `deprecate='0.9.1', error='1.0.0', remove='1.1.0'` unless a
  doc says otherwise.
- **Changelog**: add an entry under `0.9.0 Unreleased` in `CHANGELOG.md` (or
  the current unreleased section) for every user-visible fix, in
  Keep-a-Changelog style (`Fixed:` / `Changed:` / `Removed:`).
- **Verification commands**:
  - Targeted: `python -m pytest tests/test_<x>.py -x -q`
  - Doctests for a module: `python -m pytest --xdoctest kwcoco/<file>.py -q`
  - Full suite (end of each doc): `python run_tests.py`
  - Lint floor: `./run_linter.sh`
- **Commit granularity**: one commit per numbered task (or per tight cluster
  of related tasks). Do not commit unrelated working-tree files (see doc 01
  for what is intentionally untracked).

## Decision markers

Items marked **⚠️ DECISION** change observable behavior in a way the
maintainer may care about (e.g. IoU threshold `>` vs `>=`, hash-pinned demo
outputs). Each has a recommended default. If executing autonomously, take the
recommended default, note it in the commit message, and list all decisions
taken in the final summary so the maintainer can review them in one place.

## Status tracking

When a doc is fully executed, edit its header line `Status: TODO` to
`Status: DONE (<date>, <commit range>)`. Partially executed docs get
`Status: IN PROGRESS` plus per-task checkboxes updated in place.
