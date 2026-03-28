# notebooklm-graph-pipe

`notebooklm-graph-pipe` extends upstream [`llm-graph-builder`](https://github.com/neo4j-labs/llm-graph-builder) with agent skills and workflows that combine NotebookLM and Neo4j across Codex, Claude, and OpenCode. It provides an end-to-end pipeline for turning a NotebookLM-backed corpus into a Neo4j graph, a self-improving graph consolidation workflow, and A/B evaluation of notebook-only retrieval vs hybrid vector RAG + GraphRAG.

## Pipeline Overview

```mermaid
flowchart LR
    A["Local Corpus"] -- "sync_notebook_graph.py" --> B["NotebookLM"]
    B -- "export" --> C["Staged .txt Files"]
    C -- "build_graph.py" --> D["Neo4j Graph"]
    D -- "postprocess_graph.py" --> E["Post-processed Graph"]
    E -- "run_ab_evaluation.py" --> F["A/B Evaluation Report"]
    E -- "consolidate_self_improving.py" --> G["Consolidated Graph"]
```

## Setup

- Python `3.12+`
- Google account signed into NotebookLM
- [`notebooklm-mcp`](https://github.com/jacob-bd/notebooklm-mcp-cli) and [`neo4j`](https://github.com/neo4j-contrib/mcp-neo4j) MCP servers configured for the bundled NotebookLM and graph workflows
- Docker if you want `scripts/sync_notebook_graph.py` to provision or resume a managed Neo4j container automatically, or your own Neo4j instance if you want to pass explicit `--neo4j-*` connection details
- One of the supported agents on `PATH`: `codex`, `claude`, or `opencode`

### 1. Install dependencies

```bash
git clone --recurse-submodules <repo-url>
cd llm-graph-builder-scripts
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -c vendor/llm-graph-builder/backend/constraints.txt
```

### 2. Authenticate NotebookLM

```bash
nlm login
```

### 3. Environment Variables

Set provider keys only for the flows that use them:

```bash
# Optional - required for the default Google-backed postprocess/consolidation path,
# or whenever your routing config selects Google providers
export GOOGLE_API_KEY="your-key-here"

# Optional - only if your routing config selects these providers
export OPENAI_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

### 4. Optional: use your own Neo4j instance

You do not need to provide Neo4j connection details to `scripts/sync_notebook_graph.py` by default. If you omit `--neo4j-uri`, `--neo4j-user`, `--neo4j-password`, and `--neo4j-database`, the sync workflow provisions or resumes a Docker-managed Neo4j runtime for you automatically.

Only pass explicit Neo4j flags when you want the scripts to target a Neo4j instance that you manage yourself:

```bash
python scripts/sync_notebook_graph.py create \
  --dataset-dir path/to/corpus \
  --notebook-title my-corpus \
  --neo4j-uri bolt://127.0.0.1:7687 \
  --neo4j-user neo4j \
  --neo4j-password your-password \
  --neo4j-database neo4j
```

*Note: NotebookLM standard tier allows up to `100` notebooks and `50` sources per notebook. If source-count limits become a bottleneck, upgrade through [NotebookLM](https://support.google.com/notebooklm/answer/16213268?hl=en) or the [Google AI plans](https://one.google.com/intl/en/about/google-ai-plans).*

*Docker notes:*
- `scripts/sync_notebook_graph.py` can run without Docker when you provide explicit Neo4j credentials for your own instance
- `scripts/run_ab_evaluation.py --manifest-path ...` does not require Docker
- `scripts/run_ab_evaluation.py --datasets ...` auto-manages configured containers only when Docker is available

## Getting Started

### 5-minute path

Assuming you have completed the [Setup](#setup), the shortest path is to let `scripts/sync_notebook_graph.py` manage the Neo4j container automatically:

```bash
python scripts/sync_notebook_graph.py create \
  --dataset-dir path/to/your/corpus \
  --notebook-title my-corpus

python scripts/run_ab_evaluation.py \
  --manifest-path data/notebooklm_exports/my-corpus/manifest.json
```

This writes `data/notebooklm_exports/my-corpus/manifest.json`, stages NotebookLM exports under `sources/`, builds the graph, and runs the 4-factor A/B evaluation.

### What gets built

`scripts/build_graph.py` consumes staged NotebookLM-exported `.txt` files. When you run `scripts/sync_notebook_graph.py`, it executes the bridge from a local corpus to that staged format in this order:

- it walks the files under `--dataset-dir`
- it uploads those files into NotebookLM
- it exports NotebookLM source content into `data/notebooklm_exports/<project_slug>/sources/*.txt`
- it writes `manifest.json` with the notebook id and Neo4j runtime
- if you do not pass explicit `--neo4j-*` flags, it provisions or resumes a Docker-managed Neo4j runtime
- unless you pass `--skip-build`, it runs graph extraction from the staged `sources/` directory
- unless you pass `--skip-postprocess`, it runs the post-processing tail after graph extraction

Use local files that NotebookLM can ingest. The graph build itself always runs from the staged `.txt` exports.

### 1. Create or sync a notebook and build the graph

```bash
python scripts/sync_notebook_graph.py create \
  --dataset-dir path/to/corpus \
  --notebook-title my-corpus
```

Add `--skip-build` to stop after NotebookLM sync and manifest creation. Add `--skip-postprocess` to skip the post-processing tail after graph extraction. Add explicit `--neo4j-*` flags only if you want to use your own Neo4j instance instead of the managed Docker runtime.

### 2. Update an existing notebook and rebuild

```bash
python scripts/sync_notebook_graph.py update \
  --dataset-dir path/to/corpus \
  --notebook-id 12345678-1234-1234-1234-123456789abc
```

Explicit Neo4j flags on `update` override any managed Neo4j runtime recorded in the manifest.

### 3. Rebuild from staged NotebookLM exports

```bash
python scripts/build_graph.py \
  --sources-dir ./data/notebooklm_exports/my-corpus/sources \
  --neo4j-uri bolt://127.0.0.1:7687 \
  --neo4j-user neo4j \
  --neo4j-password your-password \
  --neo4j-database neo4j

python scripts/postprocess_graph.py \
  --neo4j-uri bolt://127.0.0.1:7687 \
  --neo4j-user neo4j \
  --neo4j-password your-password \
  --neo4j-database neo4j
```

These direct graph and postprocess entrypoints target a Neo4j instance explicitly, so pass `--neo4j-*` for the server you want to use.

### 4. Run A/B evaluation

```bash
python scripts/run_ab_evaluation.py \
  --manifest-path ./data/notebooklm_exports/my-corpus/manifest.json
```

Manifest-driven evaluation loads the notebook and Neo4j runtime from the manifest, generates `8` primary questions plus `2` reserves, runs `notebook_only` and `hybrid`, and scores them on correctness, completeness, evidence quality, and cross-document synthesis.

If you want to supply your own questions:

```bash
python scripts/run_ab_evaluation.py \
  --manifest-path ./data/notebooklm_exports/my-corpus/manifest.json \
  --questions-file path/to/questions.json \
  --dataset-label my-corpus
```

### 5. Run self-improving consolidation

```bash
python scripts/consolidation/consolidate_self_improving.py
```

Tier 1 handles lexical merges first. Tier 2 and Tier 3 then run in a self-improving loop until the consolidation gate passes or the iteration budget is exhausted.

## Providers, Agents, And Skills

The default local graph-build embedding is `sentence-transformer` with `all-MiniLM-L6-v2`. Routing config can switch embedding, prompt, and judge roles across these providers:

| Provider or runtime | Env variables | When required | Python dependency |
|---------------------|---------------|---------------|-------------------|
| `genai` / `gemini` | `GOOGLE_API_KEY` | Whenever `scripts/postprocess_graph.py` or the default consolidation flow uses Google-backed prompt / judge / embedding roles, or whenever `--llm-routing-config` selects Google-backed roles | `google-genai`, `langchain-google-vertexai` |
| `openai` | `OPENAI_API_KEY` | Whenever the routing config selects OpenAI for embeddings or single-prompt roles | `openai`, `langchain-openai` |
| `openrouter` | `OPENROUTER_API_KEY` | Whenever the routing config selects OpenRouter for embeddings or single-prompt roles | `openai`, `langchain-openai` |
| `sentence-transformer` | None | Default local graph-build embeddings, or whenever local embeddings are selected explicitly | `sentence-transformers`, `langchain-huggingface` |

Without `--llm-routing-config`, `scripts/postprocess_graph.py` and the default consolidation flow use Google-backed prompt, judge, and embedding roles, so those paths require `GOOGLE_API_KEY`. The main sync, graph-build, and A/B evaluation flow does not require it by default.

Supported agent runtimes for review or taxonomy-tail steps are `codex`, `claude`, and `opencode`. Without a routing config, consolidation defaults to `codex`.

The bundled `notebooklm-neo4j-deep-research` workflow is packaged for `.claude`, `.opencode`, and `.codex`. It alternates between NotebookLM answers and Neo4j neighborhood expansion, keeps only the strongest branches, and stops when additional loops stop adding signal.

## MCP Tooling & Agent Skills

- [`notebooklm-mcp`](https://github.com/jacob-bd/notebooklm-mcp-cli): notebook querying and NotebookLM source access
- [`neo4j`](https://github.com/neo4j-contrib/mcp-neo4j): schema reads and Cypher exploration

The bundled deep-research agent packages depend on both MCP servers:

- `.codex/skills/notebooklm-neo4j-deep-research/`
- `.claude/agents/notebooklm-neo4j-deep-research.md`
- `.opencode/agents/notebooklm-neo4j-deep-research.md`

What the provided skill does:

- treats NotebookLM as the high-context reader and Neo4j as the topology explorer
- starts from a notebook answer, extracts concrete entities, concepts, aliases, and open questions
- expands the strongest seeds through graph neighborhoods, then turns the best graph findings into tighter NotebookLM follow-ups
- scores candidate branches for relevance, novelty, graph support, and explainability, and stops when the loop stops adding signal

Example use:

```text
Use the bundled notebooklm-neo4j-deep-research skill against the notebook "my-corpus"
and the connected Neo4j graph. Research this question: "Which methods connect graph-based
retrieval with hallucination control in this corpus?" Use a 3-iteration loop budget and
return the full skill output.
```

In practice, that workflow queries NotebookLM for an initial answer, extracts high-signal seeds, probes Neo4j for neighborhoods and bridge concepts, asks targeted NotebookLM follow-ups, and returns a structured report with the final answer, iteration log, accepted/rejected branches, stop reason, and self-critique.

## Repo Layout And Overlay

- `vendor/llm-graph-builder/`: upstream `neo4j-labs/llm-graph-builder` submodule
- `src/`: local backend overlay modules that override selected upstream behavior
- `scripts/`: sync, graph build, post-processing, evaluation, and consolidation entrypoints
- `tests/`: regression coverage for orchestration and overlay behavior
- `.claude/`, `.opencode/`, `.codex/`: bundled agent and skill definitions

`src/` overlays `vendor/llm-graph-builder/backend/src`. Put local backend behavior changes in the overlay package, not in the vendored submodule.
