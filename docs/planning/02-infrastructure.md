# 02 — Infrastructure: packaging, CI, lint/type config

Status: TODO
Risk: low (config-only; verify CI still passes)
Prereqs: 01

Goal: one source of truth per tool, delete dead CI, make lint enforcement real.

## Task 2.1 — Delete the ubelt `.coveragerc`

`.coveragerc` at repo root is a copy-paste from the **ubelt** repo
(`source = ubelt`, omits `ubelt/tests/*` etc.). It is dormant only because
`run_tests.py` and every `.gitlab-ci.yml` test job pass
`--cov-config ../pyproject.toml` explicitly; anyone running bare
`pytest --cov=kwcoco` gets the wrong config (coverage prefers `.coveragerc`
over `pyproject.toml`).

- [ ] Delete `.coveragerc`. The real config lives at `pyproject.toml:44-64`
      (`[tool.coverage.*]`). Verify: `python run_tests.py` still produces a
      coverage report for `kwcoco`.

## Task 2.2 — Consolidate pytest config

Both `pytest.ini` and `pyproject.toml:34-42` (`[tool.pytest.ini_options]`)
exist; when `pytest.ini` is present, the pyproject section is **silently
ignored**, and they have diverged (`--ignore` lists and `norecursedirs`
differ).

- [ ] Merge everything into `[tool.pytest.ini_options]` in `pyproject.toml`
      (fold `--ignore=dev --ignore=docs` into `norecursedirs`, which the
      pyproject block already lists) and delete `pytest.ini`.
- [ ] Verify `python -m pytest --collect-only -q tests/ | tail -3` collects
      the same test count as before, and that `setup.py` is not collected.

## Task 2.3 — Delete stale CI configs and the lying badge

- [ ] Delete `appveyor.yml` — it tests Python **2.7 and 3.5**, on which the
      package cannot install (`python_requires>=3.8`).
- [ ] Delete `.circleci/config.yml` — tests 2.7/3.4–3.8/pypy3; dead.
- [ ] Remove the `|Appveyor|` badge from `README.rst` (used at line 6, defined
      at lines 822-823).

## Task 2.4 — Make lint/type checking real

Current state: `run_linter.sh` runs flake8 restricted to error classes
`E9,F63,F7,F82`; the GitLab `lint` job additionally runs `ty check ./kwcoco`
but the whole job is `allow_failure: true` (`.gitlab-ci.yml:530-543`).
Meanwhile `pyproject.toml:83-96` carries full `[tool.ruff.*]` config with ruff
installed nowhere, plus `[tool.mypy]` and `[tool.codespell]` — four tool
configs, at most one soft-enforced.

- [ ] **⚠️ DECISION** — pick one enforcement story. Recommended:
      1. Keep flake8's error-class subset as the *hard* gate: change the
         GitLab `lint` job to `allow_failure: false` for the flake8 step
         (keep `ty` soft for now — the repo has many `# type: ignore`s).
      2. Add `ruff>=0.4` to `requirements/linting.txt` and add
         `ruff check kwcoco` to `run_linter.sh` in *advisory* mode initially
         (`|| true`), tightening later. If instead ruff is unwanted, delete
         the `[tool.ruff.*]` sections.
      3. Wire `codespell` into `run_linter.sh` or delete `[tool.codespell]`.
- [ ] Whatever the decision: `run_linter.sh`, `requirements/linting.txt`,
      pyproject tool sections, and the CI lint job must agree — no config for
      tools that never run.

## Task 2.5 — Small packaging fixes

- [ ] `kwcoco/coco_schema.json` is not in the sdist/wheel (`SOURCES.txt`
      lacks it). Doc 05 task 5.6 makes the JSON authoritative-in-sync; once
      it does, add it to `package_data` in `setup.py` so downstream tools can
      consume the schema file. If instead the JSON is dropped entirely (see
      that task's decision), skip this.
- [ ] `requirements/gdal.txt:3` pins `GDAL>=3.11.3.1` for python>=3.14 — a
      speculative row; leave unless it breaks resolution, but note it.
- [ ] **⚠️ DECISION** — Python 3.8 has been EOL since Oct 2024 and CI still
      runs 4 cp38 jobs. Recommended: keep 3.8 support for now (AGENTS.md says
      avoid dropping without a real break) but raise the question in the
      final summary; dropping to 3.9+ would let ~30 files delete their
      future-imports and simplify `util/__init__.py`.

## Acceptance criteria

- Exactly one pytest config, one coverage config; no configs for tools that
  never run anywhere.
- CI files present in the repo correspond to CI that actually executes.
- `python run_tests.py` and `./run_linter.sh` pass locally.
