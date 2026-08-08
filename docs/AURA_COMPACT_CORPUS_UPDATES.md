# Updating the Aura compact corpus

This workflow treats Aura as the source of truth. Do not update a remembered local Neo4j container and upload it later. Mutating commands resolve the Aura connection from the compact manifest, read its configured password environment variable, and require the exact `<uri>|<database>` confirmation.

Routine updates are incremental: every changed source creates an immutable `DocumentRevision`; child chunks exist only in memory while their embeddings are aggregated; Aura stores one normalized token-weighted embedding per `ParentChunk`; graph extraction runs only for the new parents; consolidation is limited to entities grounded in selected revisions. The prior revision remains available until acceptance.

## Preconditions

1. Use the active `neo4j-quant-aura-compact` manifest and confirm `unit: parent`, `parent_embedding_v1`, and `parent_keyword_v1`.
2. Set the password environment variable named by `neo4j.password_env`.
3. Audit the Aura `CorpusSource` ledger before ingestion. Aura identity takes precedence over the local manifest and path, so renamed or moved files reuse their mapped `Document`.
4. Immediately before executed consolidation, open the instance in Aura Console, choose **Snapshots**, take an on-demand snapshot, wait for it to become exportable, and download the `.backup` file. AuraDB Free supports on-demand snapshots; see Neo4j's [backup, export, and restore guide](https://neo4j.com/docs/aura/managing-instances/backup-restore-export/). Revision rollback cannot reverse APOC entity merges.
5. Preserve at least 25% capacity headroom. Override the default Aura limits only if the target tier differs.

Before changing Aura, capture the baseline with the stable question file used for every batch:

```powershell
python scripts/run_corpus_evaluation.py `
  --manifest-path <compact-manifest.json> `
  --questions-file <fixed-questions.json> `
  --output <baseline-evaluation.json> `
  --llm-routing-config config/llm_routing.json
```

## 1. Add explicit sources

Omitted sources are not deleted.

```powershell
python scripts/source_ledger.py audit `
  --manifest-path <compact-manifest.json> `
  --confirm-target '<neo4j-uri>|<database>'
```

```powershell
python scripts/update_compact_corpus.py `
  --manifest-path <compact-manifest.json> `
  --source-root <stable-corpus-root> `
  --source <source-one.pdf> `
  --source <source-two.md> `
  --youtube-url <explicit-youtube-url> `
  --confirm-target '<neo4j-uri>|<database>'
```

Identity resolution is exact and ordered: canonical provider identity, NotebookLM acquisition alias, canonical URI, then normalized-content fingerprint. Title and fuzzy similarity are never identities. NotebookLM metadata is interpreted as the original `youtube` or `document` type; `notebooklm_source_id` remains an alias rather than the source provider. Historical videos without an exposed URL use an exact transcript fingerprint until a later URL ingestion promotes the same ledger record to its YouTube video ID. Save the emitted new revision IDs and `previous_revision_id` values. Report `added`, `updated`, `unchanged`, `conflicted`, and `legacy_only`; conflicts stop without writes. `LEGACY_ONLY` sources block accidental ingestion unless the operator deliberately supplies `--force-refresh`. Activation and ledger update are one Aura transaction. The former revision becomes `INACTIVE` but is retained.

## One-time NotebookLM ledger bootstrap

NotebookLM is historical evidence only, not a runtime dependency. Dry-run reconciliation requires exact equality with all 244 historical source IDs:

```powershell
python scripts/source_ledger.py backfill-notebooklm `
  --manifest-path <compact-manifest.json> `
  --notebook-id dfae4958-fe79-4b26-bb93-dd50b6561adb `
  --inventory data/migrations/neo4j-quant-v3/sources/source_inventory.json `
  --confirm-target '<neo4j-uri>|<database>'
```

After exporting and verifying an Aura backup, repeat with `--execute --backup-file <exported.backup>`, then repeat the identical executed command. The second execution must report 244 unchanged and zero created/changed. Audit must report 244 total, 237 active and ready, and seven legacy-only records. This bootstrap does not embed, extract graph data, or consolidate.

## 2. Complete graph extraction

Repeat until `requested` is zero and the new revisions are finalized:

```powershell
python scripts/process_graph_queue.py `
  --manifest-path <compact-manifest.json> `
  --llm-routing-config config/llm_routing.json `
  --limit 100 `
  --max-nodes 150000 `
  --max-relationships 300000 `
  --confirm-target '<neo4j-uri>|<database>'
```

The worker creates `ParentChunk-[:HAS_ENTITY]->Entity` evidence and records entity revision provenance. Existing entity and relationship properties are not overwritten during extraction; ontology changes are deferred to the reviewed consolidation step. Do not consolidate a partially extracted revision.

## 3. Validate before consolidation

Run the fixed hosted question set in text-hybrid and graph-hybrid modes. Require expected documents and parent counts, both parent indexes `ONLINE` at 100%, valid citations to active revisions, no increase in unsupported claims, grounded graph evidence, and sufficient Aura node/relationship headroom.

```powershell
python scripts/run_corpus_evaluation.py `
  --manifest-path <compact-manifest.json> `
  --questions-file <fixed-questions.json> `
  --output <candidate-evaluation.json> `
  --llm-routing-config config/llm_routing.json

python scripts/validate_compact_evaluation.py `
  --baseline <baseline-evaluation.json> `
  --candidate <candidate-evaluation.json> `
  --output <evaluation-gate.json>
```

The validator requires the identical question list, at least 95% of baseline judged quality, 100% effective citation validity, no increase in unsupported claims, and grounded graph expansion on at least 90% of entity-bridge questions.

On failure, pass each new active revision ID in a separate invocation. The command restores its most recent inactive predecessor or removes the document when the source was newly added:

```powershell
python scripts/finalize_compact_update.py rollback `
  --manifest-path <compact-manifest.json> `
  --revision-id <new-active-revision-id> `
  --confirm-target '<neo4j-uri>|<database>'
```

Do not garbage-collect before this decision.

## 4. Consolidate only the delta

Dry-run first:

```powershell
python scripts/consolidation/consolidate_delta.py `
  --manifest-path <compact-manifest.json> `
  --revision-id <new-revision-id> `
  --confirm-target '<neo4j-uri>|<database>' `
  --llm-routing-config config/llm_routing.json
```

Review the lexical groups, relabels, taxonomy actions, and semantic aliases. In Aura Console, take a fresh on-demand snapshot and export its `.backup` file. Record the snapshot time and filename, then repeat the command with `--execute --backup-file <exported.backup>`. Execution is refused unless that local backup file exists.

Delta scoping works as follows:

- Tier 1 groups must include a selected-revision entity; established nodes are preferred merge targets.
- Tier 2 and taxonomy process only selected-revision entities.
- Tier 3 compares selected entities with the established graph but excludes all old-old pairs.

Run the complete self-improving workflow only for an ontology migration, large domain shift, or measured global regression that delta passes cannot fix.

## 5. Accept and clean up

Repeat retrieval, citation, grounding, and capacity checks. If they pass, accept each revision in a separate invocation:

```powershell
python scripts/finalize_compact_update.py accept `
  --manifest-path <compact-manifest.json> `
  --revision-id <new-active-revision-id> `
  --confirm-target '<neo4j-uri>|<database>'
```

Acceptance requires active, vector-ready, graph-ready revisions and then removes inactive revision data for only the selected documents. It deliberately skips database-wide orphan-entity deletion, which belongs in a separately reviewed full maintenance pass. If post-consolidation checks fail, use Aura Console's **Restore from backup file** action with the exported pre-consolidation `.backup`; switching revisions is insufficient after entity merges and restore overwrites the current database.

## Failure boundaries

- Embedding or persistence failure: the old active revision remains available.
- Graph failure: retry failed parents; do not consolidate or collect the old revision.
- Unsafe dry run: keep the new graph unconsolidated; retrieval still works.
- Capacity failure: resize Aura or build another compact projection; do not lower headroom silently.
