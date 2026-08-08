# Operator reference

Use `docs/AURA_COMPACT_CORPUS_UPDATES.md` as the canonical command reference.

## Decision points

| State | Action |
|---|---|
| Manifest is not parent retrieval | Stop; this workflow targets the Aura compact projection only. |
| Target confirmation differs | Stop and resolve the correct manifest. |
| Capacity headroom fails | Stop; resize or build another projection. |
| Ledger identity conflicts or has multiple matches | Stop without embedding or graph writes; resolve exact evidence. |
| NotebookLM source has no original URL | Infer its source type and use exact normalized content as the temporary canonical identity; retain the NotebookLM ID as an alias. |
| Ledger source is `LEGACY_ONLY` | Report it as already ingested; require explicit `--force-refresh` to materialize it. |
| Source checksum is unchanged in Aura | Record it as unchanged; do not re-embed. |
| Parent persistence fails | Keep or restore the prior active revision. |
| Graph parents remain failed | Retry extraction; do not consolidate. |
| Pre-consolidation quality fails | Roll back the corpus revision. |
| Delta dry-run is unsafe | Keep the graph unconsolidated and investigate. |
| Post-consolidation quality fails | Restore the exported on-demand Aura snapshot; revision rollback is insufficient. |
| All gates pass | Accept the active revision, then garbage-collect inactive revisions. |

## Routine versus full consolidation

Routine batches use selected revision IDs. Established entities are candidate targets, but established-to-established pairs are excluded. Schedule a full self-improving run only for an ontology or schema change, a large domain shift, or a measured global regression that delta passes cannot correct.
