# llm-graph-builder-scripts

This repository isolates local orchestration and backend overrides from the upstream `llm-graph-builder` project.

The goal is to keep `llm-graph-builder` pinned as an upstream submodule while carrying script-specific behavior here.

## Structure

- `vendor/llm-graph-builder/`
  - Upstream Git submodule checkout of `neo4j-labs/llm-graph-builder`
  - Currently pinned to commit `61121df4c15716f67636a4fac2c96e909d374ada`
- `src/`
  - Overlay package for backend modules that differ from upstream
  - Only the patched modules live here
- `build_graph.py`
  - Local register/extract/post-process entrypoint
- `graph_builder_runtime.py`
  - Compatibility wrapper exposing the old orchestration surface
- `scripts/`
  - Workflow entrypoints for NotebookLM sync, post-processing, and consolidation
- `tests/`
  - Regression coverage for the moved orchestration and overlay behavior

## Overlay Contract

`src` is a package overlay on top of `vendor/llm-graph-builder/backend/src`.

These modules are intentionally overridden here:

- `src/main.py`
- `src/llm.py`
- `src/adaptive_retry.py`
- `src/shared/common_fn.py`

Everything else resolves from the upstream submodule.

Do not patch files under `vendor/llm-graph-builder/` for local behavior changes. Put those changes in the overlay package instead.

## Main Entry Points

- `python build_graph.py --help`
- `python scripts/sync_notebook_graph.py --help`
- `python scripts/postprocess_graph.py --help`
- `powershell -ExecutionPolicy Bypass -File scripts/run_consolidation.ps1`

## Runtime

The local entrypoints import the upstream backend directly through the `src`
overlay package. Docker is only used where scripts explicitly provision a
Neo4j container.

## Testing

Run:

```powershell
pytest -q
```

Useful smoke checks:

```powershell
python -m py_compile build_graph.py graph_builder_runtime.py `
  src\__init__.py src\main.py src\llm.py src\adaptive_retry.py `
  src\shared\__init__.py src\shared\common_fn.py `
  scripts\sync_notebook_graph.py scripts\postprocess_graph.py

python build_graph.py --help
python scripts\sync_notebook_graph.py --help
python scripts\postprocess_graph.py --help
```

## Dependencies

Python dependencies for consolidation and backend-aligned execution come from the upstream submodule:

- `vendor/llm-graph-builder/backend/requirements.txt`
- `vendor/llm-graph-builder/backend/constraints.txt`

`scripts/run_consolidation.ps1` installs from those files when `-InstallDeps` is used.

## Notes

- This repo is meant to own orchestration code and backend deltas only.
- The upstream submodule checkout is the baseline implementation.
- Generated artifacts such as `__pycache__`, `.pytest_cache`, `tests/_tmp`, and `runs/` are ignored.
