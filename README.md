# Graphscribe

`graphscribe` is a self-hosted corpus ingestion, vector RAG, and GraphRAG pipeline built on Neo4j. Its primary v3 workflow ingests local text, Markdown, PDF, and YouTube transcripts without NotebookLM, creates hierarchical retrieval chunks and native Neo4j vector/full-text indexes, extracts a knowledge graph, and exposes cited research through REST and MCP.

See [Local Neo4j Corpus RAG](docs/LOCAL_CORPUS_RAG.md) for the primary setup, migration, API, MCP, and operations guide.

For routine additions to an Aura-authoritative compact parent-vector corpus, the `update-aura-compact-corpus` skill drives a revision-scoped workflow that updates parent embeddings and graph evidence without rerunning full consolidation.

To seed a new compact corpus from content-addressed source packages, for example to rebuild a projection into a fresh database, run `scripts/bootstrap_compact_corpus.py` and then `scripts/process_graph_queue.py`. The bootstrap copies a template manifest's settings under a new corpus key and writes them to a new target manifest. It refuses the template's own database and any database that already holds a node.

## Primary Local Pipeline

```mermaid
flowchart LR
    A["TXT / Markdown / PDF / YouTube"] --> B["Canonical structured documents"]
    B --> C["Parent and MiniLM-safe child chunks"]
    C --> D["Neo4j vector and full-text indexes"]
    C --> E["Parent-based graph extraction"]
    D --> F["Hybrid retrieval and reranking"]
    E --> F
    F --> G["REST / MCP cited answers"]
```

Quick start:

```powershell
.\.venv\Scripts\python.exe scripts\sync_corpus_graph.py create `
  --dataset-dir C:\path\to\corpus `
  --corpus-title my-corpus

.\.venv\Scripts\python.exe scripts\serve_corpus_api.py
```

Check an existing corpus before using it. `scripts/check_corpus_connection.py` is read-only and resolves the connection from the manifest the same way REST, MCP, extraction, and sync do:

```powershell
.\.venv\Scripts\python.exe scripts\check_corpus_connection.py `
  --manifest-path data\corpora\<corpus-key>\manifest.json
```

See [Verify a connection](docs/LOCAL_CORPUS_RAG.md#verify-a-connection).

## Legacy NotebookLM Pipeline

Retained temporarily for blue-green migration and historical benchmark compatibility. `sync_notebook_graph.py` is deprecated and delegates to the local corpus synchronizer when executed directly. See [Legacy NotebookLM Pipeline](docs/LEGACY_NOTEBOOKLM_PIPELINE.md) for the full setup, migration, and getting-started guide.

## Providers, Agents, And Skills

The default local graph-build embedding is `sentence-transformer` with `all-MiniLM-L6-v2`. Routing config can switch embedding, prompt, and judge roles across these providers:

| Provider or runtime | Env variables | When required | Python dependency |
|---------------------|---------------|---------------|-------------------|
| `genai` / `gemini` | `GOOGLE_API_KEY` | Whenever `scripts/postprocess_graph.py` or the default consolidation flow uses Google-backed prompt / judge / embedding roles, or whenever `--llm-routing-config` selects Google-backed roles | `google-genai`, `langchain-google-vertexai` |
| `openai` | `OPENAI_API_KEY` | Whenever the routing config selects OpenAI for embeddings or single-prompt roles | `openai`, `langchain-openai` |
| `openrouter` | `OPENROUTER_API_KEY` | Whenever the routing config selects OpenRouter for embeddings or single-prompt roles | `openai`, `langchain-openai` |
| `openrouter_json` | `OPENROUTER_API_KEY` | OpenRouter single-prompt roles that request JSON output (the role's schema, else JSON mode) from providers that do not collect request data; the default graph extractor, `deepseek/deepseek-v4.1-flash`, and the default staged-evaluation (`:measure`) judge, `openai/gpt-6-luna` | `openai` |
| `openrouter_decisions` | `OPENROUTER_API_KEY` | OpenRouter decision models on the alpha Decisions API, routed only to providers that neither collect nor retain request data; the default Tier 2 primary and Tier 3 judge, `typesafe/jev-1.13` | `httpx` |
| `codex` | ChatGPT subscription login (`codex login`) | Subscription-backed single-prompt roles; the CLI is invoked non-interactively with read-only sandboxing and structured JSON output | Codex CLI |
| `claude` | Claude subscription login (`claude auth login`) | Subscription-backed single-prompt roles; the CLI is invoked non-interactively with tools disabled and structured JSON output | Claude Code CLI |
| `sentence-transformer` | None | Default local graph-build embeddings, or whenever local embeddings are selected explicitly | `sentence-transformers`, `langchain-huggingface` |

Without `--llm-routing-config`, Tier 2 classifies with the OpenRouter decision model `typesafe/jev-1.13` (client `openrouter_decisions`: one choice question over the label catalog on the alpha Decisions API, sent only to providers that neither collect nor retain request data), and taxonomy uses `minimax/minimax-m3` through OpenRouter for its primary prompt role. Tier 3 judges each alias candidate once with the same decision model (one yes/no question over the normalized pair) and merges only when its ALIAS probability is at least `TIER3_ALIAS_THRESHOLD` (0.65, calibrated for `typesafe/jev-1.13`; another judge needs its own threshold). It never merges names that differ only in digits, such as `AR(1)` and `AR(2)`, and reports each refused pair in `digit_guard_blocked`. Tier 3 has no second-stage judge. A prompt-model Tier 3 judge asks for JSON with a 2,048-token output cap, so a reasoning model has room to answer. Tier 2 escalates to subscription-authenticated `gpt-5.6-luna` through Codex at low reasoning effort. Taxonomy's secondary role remains `gemini-3.1-pro-preview`, and Tier 3 embeddings use `gemini-embedding-2`, which takes its task in the prompt rather than through `task_type`, so default consolidation requires `OPENROUTER_API_KEY`, `GOOGLE_API_KEY`, and an authenticated Codex CLI. Set `reasoning_effort` to `low`, `medium`, `high`, or `xhigh` on a Codex single-prompt role; Claude supports `low`, `medium`, or `high`.

Supported agent runtimes for review or taxonomy-tail steps are `codex`, `claude`, and `opencode`. Without a routing config, consolidation defaults to `codex`.

The bundled `neo4j-corpus-deep-research` workflow is packaged for `.claude`, `.opencode`, and `.codex`. It alternates cited local corpus retrieval with Neo4j neighborhood expansion, keeps only source-verifiable branches, and stops when additional loops stop adding signal.

## MCP Tooling & Agent Skills

- Local corpus MCP (`scripts/serve_corpus_mcp.py`): cited corpus search, answers, source metadata, graph neighborhoods, and sync status
- [`neo4j`](https://github.com/neo4j-contrib/mcp-neo4j): schema reads and Cypher exploration

The bundled deep-research packages use the local corpus MCP and optionally the Neo4j MCP:

- `.codex/skills/neo4j-corpus-deep-research/`
- `.claude/agents/neo4j-corpus-deep-research.md`
- `.opencode/agents/neo4j-corpus-deep-research.md`

What the provided skill does:

- treats cited corpus retrieval as the high-context reader and Neo4j as the topology explorer
- starts from a grounded answer, extracts concrete entities, concepts, aliases, and open questions
- expands the strongest seeds through graph neighborhoods, then turns the best graph findings into tighter corpus follow-ups
- scores candidate branches for relevance, novelty, graph support, and explainability, and stops when the loop stops adding signal

Example use:

```text
Use the bundled neo4j-corpus-deep-research skill against corpus "my-corpus"
and its connected Neo4j graph. Research this question: "Which methods connect graph-based
retrieval with hallucination control in this corpus?" Use a 3-iteration loop budget and
return the full skill output.
```

In practice, that workflow queries the local corpus for an initial cited answer, extracts high-signal seeds, probes Neo4j for neighborhoods and bridge concepts, revalidates graph discoveries against source chunks, and returns a structured report with the final answer, iteration log, accepted/rejected branches, stop reason, and self-critique.

## Repo Layout And Overlay

- `vendor/llm-graph-builder/`: upstream `neo4j-labs/llm-graph-builder` submodule
- `src/`: local backend overlay modules that override selected upstream behavior
- `scripts/`: sync, graph build, post-processing, evaluation, and consolidation entrypoints
- `tests/`: regression coverage for orchestration and overlay behavior
- `.claude/`, `.opencode/`, `.codex/`: bundled agent and skill definitions

`src/` overlays `vendor/llm-graph-builder/backend/src`. Put local backend behavior changes in the overlay package, not in the vendored submodule.
