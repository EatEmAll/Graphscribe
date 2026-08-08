---
name: update-aura-compact-corpus
description: Safely add or update sources in the Aura-authoritative compact parent-vector corpus, extract their Neo4j graph evidence, run revision-scoped consolidation, validate the result, and accept or roll back the revision. Use for routine corpus growth without a full graph consolidation.
---

# Update Aura Compact Corpus

Read [references/workflow.md](references/workflow.md) before taking any mutating action.

## Required sequence

1. Locate the active compact manifest and verify its parent retrieval profile.
2. Resolve Aura only from that manifest and require the exact target confirmation. Never substitute a local Neo4j instance.
3. Run `scripts/source_ledger.py audit`, inventory capacity, and establish a pre-update retrieval and grounding baseline.
4. Add only explicitly named files or YouTube URLs with `scripts/update_compact_corpus.py`. Resolve exact ledger identity before embedding and preserve the emitted new and previous revision IDs.
5. Run `scripts/process_graph_queue.py` until every new revision is graph-ready.
6. Validate the new revision before consolidation. Roll back to the previous revision if the corpus gate fails.
7. Run `scripts/consolidation/consolidate_delta.py` without `--execute`. Review the proposed changes.
8. Export an on-demand Aura snapshot, then execute the same delta with `--backup-file` only when the dry run is acceptable.
9. Repeat quality and capacity gates. Accept and garbage-collect only after they pass.

## Safety rules

- Aura is authoritative; local databases are disposable caches unless freshly restored from Aura.
- Never scan a directory and infer deletions. Routine updates use explicit source paths.
- Aura's ledger overrides stale local manifest identity. Never deduplicate by title or fuzzy similarity.
- Treat `LEGACY_ONLY` as already ingested unless the user explicitly authorizes `--force-refresh`.
- Never store child chunks in the compact database.
- Never run full consolidation for an ordinary source batch.
- Never consolidate before graph extraction is complete.
- Never garbage-collect the previous revision before validation.
- APOC merges are not revision-reversible. Require a pre-consolidation backup before `--execute`.
- If a gate fails, stop; do not lower thresholds or capacity headroom silently.

## Handoff

Report the exact Aura target; ledger audit; added, revised, unchanged, conflicted, and legacy-only sources; new and previous revision IDs; parent and transient child counts; graph completion; delta status; validation; capacity; and revision retention.
