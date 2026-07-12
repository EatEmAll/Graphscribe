# Local Neo4j Corpus RAG

The primary pipeline is self-hosted and does not use NotebookLM. Original local files and declared YouTube URLs are normalized into structured documents, hierarchically chunked, embedded with `all-MiniLM-L6-v2`, and persisted in a corpus-specific Neo4j runtime.

## Install

Use Python 3.12 and install the repository requirements plus the vendored backend constraints:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -c vendor\llm-graph-builder\backend\constraints.txt
```

Docling is used for PDF structure, OCR, and tables. `youtube-transcript-api` is primary for YouTube captions, with `yt-dlp` as the subtitle fallback.

## Create and update a corpus

```powershell
.\.venv\Scripts\python.exe scripts\sync_corpus_graph.py create `
  --dataset-dir C:\path\to\corpus `
  --corpus-title research-corpus

.\.venv\Scripts\python.exe scripts\sync_corpus_graph.py update `
  --dataset-dir C:\path\to\corpus `
  --corpus-key research-corpus
```

Supported local formats are `.txt`, `.md`, `.markdown`, and `.pdf`. Add YouTube sources with `sources.yaml` in the corpus root:

```yaml
version: 1
youtube:
  - url: https://www.youtube.com/watch?v=VIDEO_ID
    preferred_languages: [en]
```

The manifest, canonical JSON, readable Markdown projections, sync reports, and job reports are stored under `data/corpora/<corpus-key>/`. These generated artifacts are ignored by Git.

## Graph extraction

Vector and full-text retrieval becomes active before graph extraction completes. Process pending parent chunks with:

```powershell
.\.venv\Scripts\python.exe scripts\process_graph_queue.py `
  --manifest-path data\corpora\research-corpus\manifest.json `
  --llm-routing-config config\llm-routing.json `
  --limit 100
```

The routing file may configure `single_prompt.graph_extraction` and `single_prompt.answer`. Supported clients are `genai`, `openai`, and `openrouter`; set only the API key selected by those roles.

## REST API

```powershell
.\.venv\Scripts\python.exe scripts\serve_corpus_api.py `
  --llm-routing-config config\llm-routing.json
```

The service binds to `127.0.0.1:8765`. On first start it creates a bearer token at `.local/api_token`. `/health` is public on loopback; all `/v1` endpoints require `Authorization: Bearer <token>`.
Corpus metadata returned by REST or MCP omits the stored Neo4j password.

Key endpoints:

- `GET /v1/corpora`
- `POST /v1/corpora/{key}/search`
- `POST /v1/corpora/{key}/answer`
- `POST /v1/corpora/{key}/sync`
- `GET /v1/jobs/{job_id}`
- `GET/DELETE /v1/corpora/{key}/documents/{document_id}`

Retrieval modes are `vector`, `lexical`, `hybrid`, and `graph_hybrid`. `graph_hybrid` is the default.

## MCP

```powershell
.\.venv\Scripts\python.exe scripts\serve_corpus_mcp.py `
  --llm-routing-config config\llm-routing.json
```

The stdio MCP exposes `corpus_list`, `corpus_get`, `corpus_search`, `corpus_answer`, `source_list`, `source_get`, `graph_neighbors`, and `sync_status`.

## Evaluation

Compare vector/full-text retrieval against graph-enhanced retrieval with a fixed question set:

```powershell
.\.venv\Scripts\python.exe scripts\run_corpus_evaluation.py `
  --manifest-path data\corpora\research-corpus\manifest.json `
  --questions-file evaluation_questions.json
```

The evaluator records both answers, deterministic citation-validity metrics, a four-factor judge, unsupported-claim counts, and condition winners. Omit `--questions-file` to generate eight primary and two reserve questions from retrieved corpus evidence. A supplied question file contains either a JSON list or `{ "questions": [...] }` with `text`, optional `question_id`, and optional `reserve`.

## Updates, failure handling, and deletion

- New revisions are built without disturbing the active revision.
- A revision becomes active only after all expected child embeddings are present.
- Failed updates retain the previous active revision.
- Sync and delete operations share a per-corpus file lock, including direct CLI runs, so manifest and Neo4j mutations cannot overlap.
- Graph extraction may be retried independently.
- Extracted graph relationships retain parent-chunk provenance; retired revision relationships are excluded from retrieval and removed by garbage collection.
- Removed files are deactivated before inactive revisions and their graph provenance are garbage-collected.
- REST deletion also suppresses that source key, preventing a still-present file or YouTube declaration from being re-ingested. Remove the key from `suppressed_sources` in the manifest to enable it again.
- An embedding-model fingerprint change requires a blue-green rebuild.
- An update cannot change the corpus key or Neo4j runtime in place; use a separate export directory for blue-green migration.

## Blue-green migration

1. Keep the existing NotebookLM-derived Neo4j runtime unchanged as blue.
2. Create a separate v3 corpus/runtime from original files as green.
3. Run source, chunk, vector, citation, and evaluation checks on green.
4. Point REST/MCP registry usage at green.
5. Keep blue read-only for seven days.
6. Roll back by restoring the prior registry/manifest location.
7. Remove blue only after explicit confirmation.

NotebookLM chats, notes, and generated audio are not migrated automatically. Export any required artifacts as ordinary source documents before cutover.

The running REST/MCP service detects a changed Neo4j URI, database, credential, corpus ID, or embedding fingerprint in the registry manifest and replaces its cached runtime on the next request.

## Scale validation

Prepare fixed query JSON and run the benchmark once at each staged corpus size:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_corpus_scale.py `
  --manifest-path data\corpora\research-corpus\manifest.json `
  --queries-file evaluation_questions.json `
  --stage 100k `
  --iterations 100 `
  --output reports\scale-100k.json
```

Repeat with `--stage 1m` and `--stage 5m`. The report records actual active document, parent, and child counts plus throughput and min/p50/p95/p99/max latency for both `hybrid` and `graph_hybrid`. Treat the stage label as descriptive and verify the recorded `child_chunks` count before accepting the run.

## Known constraints

- `all-MiniLM-L6-v2` is English-centric and child chunks are limited to 220 word pieces to prevent truncation.
- Whisper transcription for videos without captions is not included.
- The service is local single-user software; internet-facing deployment and RBAC are not implemented.
- Scale targets must be validated on the actual host. Use staged corpora at 100K, 1M, and 5M chunks and record ingestion throughput plus p50/p95/p99 retrieval latency.
