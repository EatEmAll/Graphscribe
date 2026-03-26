# notebooklm-graph-pipe

This repository keeps local orchestration, workflow scripts, and backend overlay patches separate from the upstream `llm-graph-builder` project.

The intent is to keep `vendor/llm-graph-builder/` pinned as a Git submodule while carrying repo-specific graph build, NotebookLM sync, consolidation, and agent-skill assets here.

## Prerequisites

- Python `3.12` or newer
- `pip`
- Docker Desktop or a compatible Docker engine
- Access to a Neo4j instance
- An authenticated Google account for NotebookLM
- NotebookLM CLI `nlm` on `PATH` if you use `scripts/sync_notebook_graph.py` or `scripts/run_ab_evaluation.py`
- MCP servers for `notebooklm-mcp` and `neo4j` if you want to use the bundled deep-research agent or skill packages
- For the default consolidation flow with no routing config:
  - `codex` on `PATH`
  - `GOOGLE_API_KEY` for the default `genai` prompt, judge, and embedding roles
- Optional: if you provide `--llm-routing-config`, agent-driven review flows can use one of:
  - `codex`
  - `claude` (Claude Code)
  - `opencode`
- OpenRouter-backed agent use is technically supported through routing configs, but it is not the recommended path for review or taxonomy-tail automation because many open-source or open-weight models do not reliably return the strict JSON payloads required by the orchestrator.

## Python Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -c vendor\llm-graph-builder\backend\constraints.txt
```

The top-level `requirements.txt` keeps this repo aligned with the vendored backend and adds the extra packages imported directly by this repo's local scripts and tests.

## Non-Python Dependencies

- Docker is required when `scripts/sync_notebook_graph.py` provisions or resumes a managed Neo4j container.
- Docker is also used by `scripts/run_ab_evaluation.py` when it checks or starts benchmark Neo4j containers.
- If you pass an explicit Neo4j runtime with `--neo4j-uri`, `--neo4j-user`, `--neo4j-password`, and `--neo4j-database`, the sync workflow can run without Docker.
- NotebookLM workflows require the `nlm` CLI to be installed and authenticated.

## NotebookLM Account Requirements

- You must be signed into NotebookLM with a Google account before using the NotebookLM CLI or the bundled NotebookLM-driven workflows.
- Per Google's current NotebookLM help pages, the standard NotebookLM tier allows up to `100` notebooks and up to `50` sources per notebook.
- If the `50`-source cap becomes a bottleneck, upgrade to a paid Google AI plan that includes higher NotebookLM limits, typically `Google AI Pro` or `Google AI Ultra`.
- Google documents the higher-tier NotebookLM upgrade path and limits here:
  - [Upgrade NotebookLM](https://support.google.com/notebooklm/answer/16213268?hl=en)
  - [Google AI plans](https://one.google.com/intl/en/about/google-ai-plans)

## MCP Tooling

The included deep-research skill and agent definitions assume these MCP servers are available:

- `notebooklm-mcp`
  - Used for notebook querying and NotebookLM source access
  - The Codex skill explicitly relies on `mcp__notebooklm-mcp__notebook_query`
- `neo4j`
  - Used for schema reads and Cypher exploration
  - The Codex skill explicitly relies on `mcp__neo4j__get-schema` and `mcp__neo4j__read-cypher`

Without those MCP servers, the bundled `.claude`, `.opencode`, and `.codex` deep-research packages are not usable as intended.

## Repository Structure

- `vendor/llm-graph-builder/`
  - Upstream Git submodule checkout of `neo4j-labs/llm-graph-builder`
  - Currently pinned to commit `61121df4c15716f67636a4fac2c96e909d374ada`
- `src/`
  - Overlay package for backend modules that differ from upstream
- `build_graph.py`
  - Local register/extract/post-process entrypoint
- `graph_builder_runtime.py`
  - Compatibility wrapper exposing the old orchestration surface
- `scripts/`
  - Workflow entrypoints for NotebookLM sync, post-processing, A/B evaluation, and consolidation
- `tests/`
  - Regression coverage for orchestration and overlay behavior
- `.claude/`
  - Claude agent definitions bundled with this repo
- `.opencode/`
  - OpenCode agent definitions bundled with this repo
- `.codex/`
  - Codex skill definitions bundled with this repo

## Included Skills And Agents

This repository ships the same `notebooklm-neo4j-deep-research` workflow packaged for multiple coding-agent runtimes:

- `.claude/agents/notebooklm-neo4j-deep-research.md`
  - Claude agent definition
- `.opencode/agents/notebooklm-neo4j-deep-research.md`
  - OpenCode subagent definition
- `.codex/skills/notebooklm-neo4j-deep-research/SKILL.md`
  - Codex skill definition

The shared purpose of this workflow is to alternate between NotebookLM notebook queries and Neo4j neighborhood exploration when researching a topic against the same corpus.

## Model And Client Requirements

### Optional agent clients

The routing layer supports these agent clients for review or taxonomy-tail steps when you provide `--llm-routing-config`:

- `codex`
- `claude`
- `opencode`

Without a routing config, the consolidation entrypoints default to `codex`.

`opencode` can be pointed at OpenRouter-hosted models, so this is technically supported. In practice, it is not the recommended review-agent path because the review and taxonomy-tail steps require strict machine-readable JSON, and many open-source or open-weight models return narrative prose instead of the required object.

### Embedding options

Graph build and post-processing use one embedding provider and model combination.

- Default local graph-build embedding:
  - client or provider: `sentence-transformer`
  - model: `all-MiniLM-L6-v2`
- Routing-config embedding clients supported by `llm_routing.py`:
  - `genai`
  - `openai`
  - `openrouter`
- Backend embedding provider aliases accepted by the runtime:
  - `genai` maps to Gemini embeddings
  - `gemini`
  - `openai`
  - `openrouter`
  - `sentence-transformer`
  - `titan` for AWS Bedrock embeddings

Relevant API keys or credentials depend on the selected embedding provider:

- `GOOGLE_API_KEY` for `genai`
- `OPENAI_API_KEY` for `openai`
- `OPENROUTER_API_KEY` for `openrouter`
- AWS Bedrock credentials for `titan`

### LLM prediction and judge clients

The consolidation and post-processing routing layer supports these single-prompt clients for tier-2 classification, taxonomy cleanup, and tier-3 judge calls:

- `genai`
- `openai`
- `openrouter`

Without a routing config, these roles default to `genai`, so the zero-config path requires `GOOGLE_API_KEY`.

If you use a routing config, these prompt-client choices apply to roles such as:

- `single_prompt.tier2_primary`
- `single_prompt.tier2_secondary`
- `single_prompt.taxonomy_primary`
- `single_prompt.taxonomy_secondary`
- `single_prompt.tier3_judge_primary`
- `single_prompt.tier3_judge_secondary`
- `embeddings.tier3`
- `embeddings.graph_build`

## Overlay Contract

`src/` overlays `vendor/llm-graph-builder/backend/src`.

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
- `python scripts/run_ab_evaluation.py --help`
- `powershell -ExecutionPolicy Bypass -File scripts/run_consolidation.ps1`

## Benchmark Datasets

Benchmark dataset connection details live in `benchmark_dataset_registry.json`.

This file is local runtime state, not a sanitized sample manifest. In the current implementation it may contain:

- NotebookLM notebook IDs
- Docker container IDs and mapped ports
- Local Neo4j connection URIs
- Plaintext Neo4j passwords for benchmark containers

Treat it as sensitive local configuration and avoid sharing it unchanged.

The current registered dataset keys are:

- `bench-openalex-rag`
- `bench-imdb-scifi`
- `bench-opentargets-alzheimers`

The main entrypoints accept `--dataset-key` and resolve stored Neo4j runtime details from that registry. `scripts/sync_notebook_graph.py` also uses the registry to resolve the NotebookLM notebook id or title and the default export directory.

Examples:

```powershell
python build_graph.py --dataset-key bench-openalex-rag

python scripts\sync_notebook_graph.py update `
  --dataset-dir C:\Users\Roman\repos\misc_notebooks\benchmark-datasets\openalex-rag\sources `
  --dataset-key bench-openalex-rag

python scripts\postprocess_graph.py --dataset-key bench-imdb-scifi
```

## Runtime Notes

- Local entrypoints import the vendored backend through the `src` overlay package.
- `scripts/run_consolidation.ps1 -InstallDeps` installs backend-aligned dependencies if the required Python modules are missing.
- Without a routing config, consolidation defaults to `codex` plus `genai` roles and therefore requires `GOOGLE_API_KEY`.
- With a routing config, consolidation may instead require `OPENAI_API_KEY` or `OPENROUTER_API_KEY`, and can switch review agents away from `codex`.
- Neo4j connection details can come from explicit CLI flags, the dataset registry, or a local `.codex/config.toml` if you create one for your Codex MCP setup.

## Testing

Run the test suite:

```powershell
pytest -q
```

Useful smoke checks:

```powershell
python -m py_compile build_graph.py graph_builder_runtime.py `
  src\__init__.py src\main.py src\llm.py src\adaptive_retry.py `
  src\shared\__init__.py src\shared\common_fn.py `
  scripts\sync_notebook_graph.py scripts\postprocess_graph.py scripts\run_ab_evaluation.py

python build_graph.py --help
python scripts\sync_notebook_graph.py --help
python scripts\postprocess_graph.py --help
python scripts\run_ab_evaluation.py --help
```

## Notes

- This repo owns orchestration code, agent packaging, and backend deltas.
- The upstream submodule remains the baseline implementation.
- Generated artifacts such as `__pycache__`, `.pytest_cache`, `tests/_tmp`, and `runs/` are ignored.
