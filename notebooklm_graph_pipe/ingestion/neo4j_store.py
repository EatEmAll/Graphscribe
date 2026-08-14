from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Iterable, Sequence

from .chunking import ChunkingResult
from .models import CanonicalDocument

GRAPH_SCHEMA_VERSION = 6

SCHEMA_QUERIES = (
    "CREATE CONSTRAINT corpus_id_unique IF NOT EXISTS FOR (n:Corpus) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT corpus_key_unique IF NOT EXISTS FOR (n:Corpus) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT revision_id_unique IF NOT EXISTS FOR (n:DocumentRevision) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT parent_chunk_id_unique IF NOT EXISTS FOR (n:ParentChunk) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT community_build_id_unique IF NOT EXISTS FOR (n:CommunityBuild) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT community_id_unique IF NOT EXISTS FOR (n:Community) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT community_report_id_unique IF NOT EXISTS FOR (n:CommunityReport) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT community_finding_id_unique IF NOT EXISTS FOR (n:CommunityFinding) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT claim_id_unique IF NOT EXISTS FOR (n:Claim) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT corpus_source_id_unique IF NOT EXISTS FOR (n:CorpusSource) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT corpus_source_provider_unique IF NOT EXISTS FOR (n:CorpusSource) REQUIRE (n.corpus_id, n.provider, n.provider_source_id) IS UNIQUE",
    "CREATE CONSTRAINT corpus_source_notebooklm_unique IF NOT EXISTS FOR (n:CorpusSource) REQUIRE (n.corpus_id, n.notebooklm_source_id) IS UNIQUE",
    "CREATE INDEX corpus_source_uri IF NOT EXISTS FOR (n:CorpusSource) ON (n.corpus_id, n.canonical_uri)",
    "CREATE INDEX corpus_source_checksum IF NOT EXISTS FOR (n:CorpusSource) ON (n.corpus_id, n.content_checksum)",
    "CREATE INDEX corpus_source_status IF NOT EXISTS FOR (n:CorpusSource) ON (n.corpus_id, n.ingestion_status, n.retrieval_status)",
    "CREATE INDEX document_status IF NOT EXISTS FOR (n:Document) ON (n.status)",
    "CREATE INDEX revision_ready IF NOT EXISTS FOR (n:DocumentRevision) ON (n.vector_ready, n.graph_ready)",
    "CREATE INDEX chunk_parent_id IF NOT EXISTS FOR (n:Chunk) ON (n.parent_id)",
    "CREATE FULLTEXT INDEX chunk_keyword_v1 IF NOT EXISTS FOR (n:Chunk) ON EACH [n.text]",
    "CREATE FULLTEXT INDEX entities IF NOT EXISTS FOR (n:__Entity__) ON EACH [n.id, n.description]",
    "CREATE FULLTEXT INDEX community_report_keyword_v1 IF NOT EXISTS "
    "FOR (n:CommunityReport) ON EACH [n.summary, n.full_content]",
    "CREATE FULLTEXT INDEX claim_keyword_v1 IF NOT EXISTS "
    "FOR (n:Claim) ON EACH [n.subject, n.predicate, n.object]",
)


def vector_index_query(dimension: int) -> str:
    return f"""
CREATE VECTOR INDEX chunk_embedding_v1 IF NOT EXISTS
FOR (n:Chunk) ON (n.embedding)
OPTIONS {{indexConfig: {{`vector.dimensions`: {int(dimension)}, `vector.similarity_function`: 'cosine'}}}}
""".strip()


def parent_vector_index_query(dimension: int) -> str:
    return f"""
CREATE VECTOR INDEX parent_embedding_v1 IF NOT EXISTS
FOR (n:ParentChunk) ON (n.embedding)
OPTIONS {{indexConfig: {{`vector.dimensions`: {int(dimension)}, `vector.similarity_function`: 'cosine'}}}}
""".strip()


def community_report_vector_index_query(dimension: int) -> str:
    return f"""
CREATE VECTOR INDEX community_report_embedding_v1 IF NOT EXISTS
FOR (n:CommunityReport) ON (n.embedding)
OPTIONS {{indexConfig: {{`vector.dimensions`: {int(dimension)}, `vector.similarity_function`: 'cosine'}}}}
""".strip()


def _cypher_identifier(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    if not normalized:
        normalized = fallback
    if normalized[0].isdigit():
        normalized = f"{fallback}_{normalized}"
    return normalized[:100]


class Neo4jCorpusStore:
    def __init__(
        self,
        driver: Any,
        database: str = "neo4j",
        *,
        batch_size: int = 500,
        corpus_id: str | None = None,
    ):
        self.driver = driver
        self.database = database
        self.batch_size = batch_size
        self.corpus_id = corpus_id

    def close(self) -> None:
        self.driver.close()

    def _session(self):
        return self.driver.session(database=self.database)

    def ensure_schema(self, dimension: int = 384) -> None:
        with self._session() as session:
            for query in SCHEMA_QUERIES:
                session.run(query).consume()
            session.run(vector_index_query(dimension)).consume()
            session.run(community_report_vector_index_query(dimension)).consume()
            session.run("DROP INDEX community_keyword IF EXISTS").consume()

    def ensure_parent_retrieval_schema(self, dimension: int = 384) -> None:
        """Create the shared corpus schema plus compact parent retrieval indexes."""
        with self._session() as session:
            for query in SCHEMA_QUERIES:
                if "chunk_keyword_v1" not in query:
                    session.run(query).consume()
            session.run(
                "CREATE FULLTEXT INDEX parent_keyword_v1 IF NOT EXISTS "
                "FOR (n:ParentChunk) ON EACH [n.text]"
            ).consume()
            session.run(parent_vector_index_query(dimension)).consume()
            session.run(community_report_vector_index_query(dimension)).consume()
            session.run("DROP INDEX community_keyword IF EXISTS").consume()

    def assert_embedding_fingerprint(self, corpus_key: str, fingerprint: str) -> None:
        with self._session() as session:
            row = session.run(
                "MATCH (c:Corpus {key: $key}) RETURN c.embedding_fingerprint AS fingerprint",
                key=corpus_key,
            ).single()
            if row and row.get("fingerprint") and row["fingerprint"] != fingerprint:
                raise ValueError(
                    f"Embedding fingerprint mismatch for corpus {corpus_key}: "
                    f"{row['fingerprint']} != {fingerprint}. Use a blue-green rebuild."
                )

    def begin_revision(
        self,
        *,
        corpus_key: str,
        corpus_title: str,
        embedding_fingerprint: str,
        document: CanonicalDocument,
        chunks: ChunkingResult,
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks.children) != len(embeddings):
            raise ValueError("Every child chunk must have exactly one embedding.")
        with self._session() as session:
            session.run(
                """
                MERGE (corpus:Corpus {id: $corpus_id})
                SET corpus.key = $corpus_key,
                    corpus.title = $corpus_title,
                    corpus.schema_version = $schema_version,
                    corpus.embedding_fingerprint = $embedding_fingerprint
                MERGE (document:Document {id: $document_id})
                WITH corpus, document
                OPTIONAL MATCH (document)-[:ACTIVE_REVISION]->(active_revision:DocumentRevision)
                SET document.source_type = $source_type,
                    document.source_uri = $source_uri,
                    document.relative_path = $relative_path,
                    document.title = $title,
                    document.language = $language,
                    document.status = CASE WHEN active_revision IS NULL THEN 'BUILDING' ELSE 'READY' END
                WITH corpus, document
                MERGE (corpus)-[:HAS_DOCUMENT]->(document)
                MERGE (revision:DocumentRevision {id: $revision_id})
                SET revision.checksum = $checksum,
                    revision.extractor = $extractor,
                    revision.extractor_version = $extractor_version,
                    revision.status = 'BUILDING',
                    revision.vector_ready = false,
                    revision.graph_ready = false,
                    revision.created_at = datetime()
                MERGE (document)-[:HAS_REVISION]->(revision)
                """,
                corpus_id=document.corpus_id,
                corpus_key=corpus_key,
                corpus_title=corpus_title,
                embedding_fingerprint=embedding_fingerprint,
                schema_version=GRAPH_SCHEMA_VERSION,
                document_id=document.document_id,
                source_type=document.source_type,
                source_uri=document.source_uri,
                relative_path=document.relative_path,
                title=document.title,
                language=document.language,
                revision_id=document.revision_id,
                checksum=document.source_checksum,
                extractor=document.extractor,
                extractor_version=document.extractor_version,
            ).consume()

        parent_rows = [
            {
                **asdict(parent),
                "section_path": list(parent.section_path),
                "block_ids": list(parent.block_ids),
            }
            for parent in chunks.parents
        ]
        child_rows = []
        for child, embedding in zip(chunks.children, embeddings, strict=True):
            child_rows.append(
                {
                    **asdict(child),
                    "section_path": list(child.section_path),
                    "block_ids": list(child.block_ids),
                    "embedding": list(embedding),
                }
            )

        for rows in _batches(parent_rows, self.batch_size):
            with self._session() as session:
                session.run(
                    """
                    MATCH (revision:DocumentRevision {id: $revision_id})
                    UNWIND $rows AS row
                    MERGE (parent:ParentChunk {id: row.id})
                    SET parent += row
                    SET parent.graph_status = 'PENDING', parent.graph_attempts = 0
                    MERGE (revision)-[:HAS_PARENT]->(parent)
                    """,
                    revision_id=document.revision_id,
                    rows=rows,
                ).consume()

        for rows in _batches(child_rows, self.batch_size):
            with self._session() as session:
                session.run(
                    """
                    MATCH (revision:DocumentRevision {id: $revision_id})
                    MATCH (document:Document {id: $document_id})
                    UNWIND $rows AS row
                    MATCH (parent:ParentChunk {id: row.parent_id})
                    MERGE (chunk:Chunk {id: row.id})
                    SET chunk += row
                    MERGE (parent)-[:HAS_CHILD]->(chunk)
                    MERGE (chunk)-[:IN_REVISION]->(revision)
                    MERGE (chunk)-[:PART_OF]->(document)
                    """,
                    revision_id=document.revision_id,
                    document_id=document.document_id,
                    rows=rows,
                ).consume()

        next_rows = [
            {"previous_id": left.id, "current_id": right.id}
            for left, right in zip(chunks.children, chunks.children[1:])
            if left.parent_id == right.parent_id
        ]
        for rows in _batches(next_rows, self.batch_size):
            with self._session() as session:
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (left:Chunk {id: row.previous_id})
                    MATCH (right:Chunk {id: row.current_id})
                    MERGE (left)-[:NEXT_CHUNK]->(right)
                    """,
                    rows=rows,
                ).consume()

    def begin_compact_revision(
        self,
        *,
        corpus_key: str,
        corpus_title: str,
        embedding_fingerprint: str,
        document: CanonicalDocument,
        chunks: ChunkingResult,
        parent_embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Persist a parent-only revision; child chunks remain transient."""
        if len(chunks.parents) != len(parent_embeddings):
            raise ValueError("Every parent chunk must have exactly one embedding.")
        with self._session() as session:
            session.run(
                """
                MERGE (corpus:Corpus {id: $corpus_id})
                SET corpus.key = $corpus_key,
                    corpus.title = $corpus_title,
                    corpus.schema_version = $schema_version,
                    corpus.embedding_fingerprint = $embedding_fingerprint
                MERGE (document:Document {id: $document_id})
                WITH corpus, document
                OPTIONAL MATCH (document)-[:ACTIVE_REVISION]->(active_revision:DocumentRevision)
                SET document.source_type = $source_type,
                    document.source_uri = $source_uri,
                    document.relative_path = $relative_path,
                    document.title = $title,
                    document.language = $language,
                    document.status = CASE WHEN active_revision IS NULL THEN 'BUILDING' ELSE 'READY' END
                WITH corpus, document
                MERGE (corpus)-[:HAS_DOCUMENT]->(document)
                MERGE (revision:DocumentRevision {id: $revision_id})
                SET revision.checksum = $checksum,
                    revision.extractor = $extractor,
                    revision.extractor_version = $extractor_version,
                    revision.status = 'BUILDING',
                    revision.vector_ready = false,
                    revision.graph_ready = false,
                    revision.retrieval_unit = 'parent',
                    revision.created_at = datetime()
                MERGE (document)-[:HAS_REVISION]->(revision)
                """,
                corpus_id=document.corpus_id,
                corpus_key=corpus_key,
                corpus_title=corpus_title,
                embedding_fingerprint=embedding_fingerprint,
                schema_version=GRAPH_SCHEMA_VERSION,
                document_id=document.document_id,
                source_type=document.source_type,
                source_uri=document.source_uri,
                relative_path=document.relative_path,
                title=document.title,
                language=document.language,
                revision_id=document.revision_id,
                checksum=document.source_checksum,
                extractor=document.extractor,
                extractor_version=document.extractor_version,
            ).consume()

        parent_rows = []
        for parent, embedding in zip(chunks.parents, parent_embeddings, strict=True):
            parent_rows.append(
                {
                    **asdict(parent),
                    "section_path": list(parent.section_path),
                    "block_ids": list(parent.block_ids),
                    "embedding": list(embedding),
                }
            )
        for rows in _batches(parent_rows, self.batch_size):
            with self._session() as session:
                session.run(
                    """
                    MATCH (revision:DocumentRevision {id: $revision_id})
                    UNWIND $rows AS row
                    MERGE (parent:ParentChunk {id: row.id})
                    SET parent += row,
                        parent.graph_status = 'PENDING',
                        parent.graph_attempts = 0
                    MERGE (revision)-[:HAS_PARENT]->(parent)
                    """,
                    revision_id=document.revision_id,
                    rows=rows,
                ).consume()

    def resolve_ledger_source(self, identity: Any) -> dict[str, Any] | None:
        from .source_ledger import SourceIdentityConflict

        row = identity.to_row()
        with self._session() as session:
            matches = list(session.run(
                """
                MATCH (source:CorpusSource {corpus_id: $corpus_id})
                 WHERE ($provider <> '' AND $provider_source_id <> ''
                        AND source.provider = $provider AND source.provider_source_id = $provider_source_id)
                   OR ($notebooklm_source_id IS NOT NULL AND source.notebooklm_source_id = $notebooklm_source_id)
                   OR ($canonical_uri IS NOT NULL AND source.canonical_uri = $canonical_uri)
                   OR ($content_checksum <> '' AND source.content_checksum = $content_checksum)
                OPTIONAL MATCH (source)-[:MATERIALIZED_AS]->(document:Document)
                RETURN source.id AS ledger_source_id, source.provider AS provider,
                       source.provider_source_id AS provider_source_id,
                       source.canonical_uri AS canonical_uri,
                       source.content_checksum AS content_checksum,
                       source.notebooklm_source_id AS notebooklm_source_id,
                       source.retrieval_status AS retrieval_status,
                       document.id AS document_id,
                       source.provider = $provider AND source.provider_source_id = $provider_source_id
                           AS provider_match,
                       $notebooklm_source_id IS NOT NULL
                           AND source.notebooklm_source_id = $notebooklm_source_id AS notebooklm_match,
                       $canonical_uri IS NOT NULL AND source.canonical_uri = $canonical_uri AS uri_match,
                       $content_checksum <> '' AND source.content_checksum = $content_checksum
                           AS fingerprint_match
                """, **row
            ))
        unique = {record["ledger_source_id"]: dict(record) for record in matches}
        if len(unique) > 1:
            duplicates = list(unique.values())
            if not all(
                item.get("provider") == row["provider"]
                and item.get("content_checksum") == row["content_checksum"]
                for item in duplicates
            ):
                raise SourceIdentityConflict(duplicates)
            match = sorted(duplicates, key=lambda item: item["ledger_source_id"])[0]
            match["duplicate_match_count"] = len(duplicates)
            match["equivalent_ledger_source_ids"] = sorted(unique)
            match["match_reason"] = "multiple_equivalent"
            return match
        if not unique:
            return None
        match = next(iter(unique.values()))
        provider_match = match["provider"] == row["provider"] and match["provider_source_id"] == row["provider_source_id"]
        notebook_match = bool(
            row["notebooklm_source_id"]
            and match.get("notebooklm_source_id") == row["notebooklm_source_id"]
        )
        uri_match = bool(row["canonical_uri"] and match.get("canonical_uri") == row["canonical_uri"])
        checksum_match = match.get("content_checksum") == row["content_checksum"]
        if not (provider_match or notebook_match or uri_match or checksum_match):
            raise ValueError("Resolved ledger record does not match the supplied source identity.")
        match["match_reason"] = next(
            reason
            for reason, matched in (
                ("provider_id", bool(match.get("provider_match"))),
                ("notebooklm_alias", bool(match.get("notebooklm_match"))),
                ("canonical_uri", bool(match.get("uri_match"))),
                ("content_fingerprint", bool(match.get("fingerprint_match"))),
            )
            if matched
        )
        return match

    def stage_compact_revision(self, document_id: str, revision_id: str, expected_parents: int) -> None:
        """Make an embedded revision graph-queue eligible without exposing it to retrieval."""
        with self._session() as session:
            result = session.run(
                """
                MATCH (document:Document {id: $document_id})-[:HAS_REVISION]->
                      (revision:DocumentRevision {id: $revision_id})
                MATCH (revision)-[:HAS_PARENT]->(parent:ParentChunk)
                WITH document, revision, count(parent) AS actual,
                     count(parent.embedding) AS embedded
                WHERE actual = $expected AND embedded = $expected
                  AND NOT (document)-[:ACTIVE_REVISION]->(revision)
                SET revision.status = 'STAGED', revision.vector_ready = true,
                    revision.staged_at = datetime(),
                    document.status = CASE
                        WHEN EXISTS { MATCH (document)-[:ACTIVE_REVISION]->() } THEN 'READY'
                        ELSE 'STAGED'
                    END
                RETURN actual
                """,
                document_id=document_id,
                revision_id=revision_id,
                expected=expected_parents,
            ).single()
            if not result:
                raise RuntimeError(
                    "Compact revision staging failed because stored parent/embedding counts "
                    f"did not equal {expected_parents}."
                )

    def staged_revision_state(self, document_id: str, revision_id: str) -> dict[str, Any] | None:
        with self._session() as session:
            row = session.run(
                """
                MATCH (document:Document {id: $document_id})-[:HAS_REVISION]->
                      (revision:DocumentRevision {id: $revision_id})
                OPTIONAL MATCH (revision)-[:HAS_PARENT]->(parent:ParentChunk)
                RETURN revision.status AS status, revision.vector_ready AS vector_ready,
                       revision.graph_ready AS graph_ready, count(parent) AS parent_count,
                       count(CASE WHEN parent.graph_status = 'COMPLETED' THEN 1 END)
                           AS completed_parents,
                       EXISTS { MATCH (document)-[:ACTIVE_REVISION]->(revision) } AS is_active
                """,
                document_id=document_id,
                revision_id=revision_id,
            ).single()
            return dict(row) if row else None

    def rollback_failed_accept(
        self,
        document_id: str,
        revision_id: str,
        previous_revision_id: str | None,
        ledger_source_id: str,
        *,
        remove_new_ledger: bool,
    ) -> None:
        """Restore the pre-accept graph state after a manifest write failure."""
        with self._session() as session:
            result = session.run(
                """
                MATCH (document:Document {id: $document_id})-[:HAS_REVISION]->
                      (revision:DocumentRevision {id: $revision_id})
                OPTIONAL MATCH (document)-[current:ACTIVE_REVISION]->(:DocumentRevision)
                DELETE current
                WITH document, revision
                OPTIONAL MATCH (document)-[:HAS_REVISION]->(previous:DocumentRevision {id: $previous_revision_id})
                FOREACH (_ IN CASE WHEN previous IS NULL THEN [] ELSE [1] END |
                    MERGE (document)-[:ACTIVE_REVISION]->(previous)
                    SET previous.status = 'ACTIVE')
                SET revision.status = 'STAGED',
                    document.status = CASE WHEN previous IS NULL THEN 'STAGED' ELSE 'READY' END
                WITH document, previous
                OPTIONAL MATCH (source:CorpusSource {id: $ledger_source_id})-[materialized:MATERIALIZED_AS]->(document)
                DELETE materialized
                WITH document, previous, source
                FOREACH (_ IN CASE WHEN source IS NULL OR $remove_new_ledger THEN [] ELSE [1] END |
                    MERGE (source)-[:MATERIALIZED_AS]->(document)
                    SET source.content_checksum = coalesce(previous.checksum, source.content_checksum),
                        source.retrieval_status = 'ACTIVE', source.ingestion_status = 'INGESTED')
                WITH document, source
                OPTIONAL MATCH (:Corpus)-[has_source:HAS_SOURCE]->(source)
                FOREACH (_ IN CASE WHEN has_source IS NULL OR NOT $remove_new_ledger THEN [] ELSE [1] END |
                    DELETE has_source)
                WITH document, source
                FOREACH (_ IN CASE WHEN source IS NULL OR NOT $remove_new_ledger THEN [] ELSE [1] END |
                    DELETE source)
                RETURN document.id AS document_id
                """,
                document_id=document_id,
                revision_id=revision_id,
                previous_revision_id=previous_revision_id,
                ledger_source_id=ledger_source_id,
                remove_new_ledger=remove_new_ledger,
            ).single()
            if not result:
                raise RuntimeError(f"Could not roll back failed acceptance for {revision_id}.")

    def promote_ledger_identity(self, ledger_source_id: str, identity: Any) -> None:
        row = identity.to_row()
        with self._session() as session:
            result = session.run(
                """
                MATCH (source:CorpusSource {id: $ledger_source_id, corpus_id: $corpus_id})
                SET source.provider = $provider,
                    source.provider_source_id = $provider_source_id,
                    source.canonical_uri = $canonical_uri,
                    source.content_checksum = $content_checksum,
                    source.source_type = $source_type,
                    source.last_verified_at = datetime()
                RETURN source.id AS id
                """, ledger_source_id=ledger_source_id, **row,
            ).single()
            if not result:
                raise RuntimeError(f"Could not promote ledger identity {ledger_source_id}.")

    def activate_compact_revision(
        self,
        document_id: str,
        revision_id: str,
        expected_parents: int,
        *,
        ledger: Any | None = None,
        require_staged: bool = False,
    ) -> None:
        with self._session() as session:
            result = session.run(
                """
                MATCH (document:Document {id: $document_id})-[:HAS_REVISION]->(revision:DocumentRevision {id: $revision_id})
                MATCH (revision)-[:HAS_PARENT]->(parent:ParentChunk)
                WITH document, revision, count(parent) AS actual,
                     count(parent.embedding) AS embedded
                WHERE actual = $expected AND embedded = $expected
                  AND (NOT $require_staged OR
                       (revision.status = 'STAGED' AND revision.graph_ready = true))
                OPTIONAL MATCH (document)-[old:ACTIVE_REVISION]->(previous:DocumentRevision)
                DELETE old
                FOREACH (_ IN CASE WHEN previous IS NULL THEN [] ELSE [1] END |
                    SET previous.status = 'INACTIVE', previous.deactivated_at = datetime())
                MERGE (document)-[:ACTIVE_REVISION]->(revision)
                SET revision.status = 'ACTIVE', revision.vector_ready = true,
                    document.status = 'READY'
                WITH document, revision
                OPTIONAL MATCH (corpus:Corpus {id: $corpus_id})
                FOREACH (_ IN CASE WHEN $ledger IS NULL THEN [] ELSE [1] END |
                    MERGE (source:CorpusSource {id: $ledger.id})
                    SET source.corpus_id = $ledger.corpus_id,
                        source.provider = $ledger.provider,
                        source.provider_source_id = $ledger.provider_source_id,
                        source.title = $ledger.title,
                        source.source_type = $ledger.source_type,
                        source.canonical_uri = $ledger.canonical_uri,
                        source.content_checksum = $ledger.content_checksum,
                        source.acquisition_method = $ledger.acquisition_method,
                        source.notebook_ids = $ledger.notebook_ids,
                        source.notebooklm_source_id = $ledger.notebooklm_source_id,
                        source.ledger_version = 2,
                        source.ingestion_status = 'INGESTED',
                        source.retrieval_status = 'ACTIVE',
                        source.last_verified_at = datetime(),
                        source.first_seen_at = coalesce(source.first_seen_at, datetime())
                    MERGE (corpus)-[:HAS_SOURCE]->(source)
                    MERGE (source)-[:MATERIALIZED_AS]->(document))
                RETURN actual
                """,
                document_id=document_id,
                revision_id=revision_id,
                expected=expected_parents,
                require_staged=require_staged,
                corpus_id=self.corpus_id,
                ledger=ledger.to_row() if ledger is not None else None,
            ).single()
            if not result:
                raise RuntimeError(
                    "Compact revision activation failed because stored parent/embedding counts "
                    f"did not equal {expected_parents}."
                )

    def assert_existing_corpus(self, corpus_id: str, corpus_key: str) -> None:
        with self._session() as session:
            row = session.run(
                "MATCH (c:Corpus {id: $corpus_id, key: $corpus_key}) RETURN c.id AS id",
                corpus_id=corpus_id,
                corpus_key=corpus_key,
            ).single()
            if not row:
                raise ValueError(
                    f"Target does not contain expected corpus id={corpus_id!r}, key={corpus_key!r}."
                )

    def active_revision_for_document(self, document_id: str) -> dict[str, Any] | None:
        with self._session() as session:
            row = session.run(
                """
                MATCH (document:Document {id: $document_id})-[:ACTIVE_REVISION]->(revision:DocumentRevision)
                RETURN revision.id AS revision_id, revision.checksum AS checksum,
                       revision.extractor AS extractor,
                       revision.extractor_version AS extractor_version,
                       revision.graph_ready AS graph_ready
                """,
                document_id=document_id,
            ).single()
            return dict(row) if row else None
    def activate_revision(self, document_id: str, revision_id: str, expected_chunks: int) -> None:
        with self._session() as session:
            result = session.run(
                """
                MATCH (document:Document {id: $document_id})-[:HAS_REVISION]->(revision:DocumentRevision {id: $revision_id})
                MATCH (chunk:Chunk)-[:IN_REVISION]->(revision)
                WITH document, revision, count(chunk) AS actual
                WHERE actual = $expected
                OPTIONAL MATCH (document)-[old:ACTIVE_REVISION]->(previous:DocumentRevision)
                DELETE old
                FOREACH (_ IN CASE WHEN previous IS NULL THEN [] ELSE [1] END |
                    SET previous.status = 'INACTIVE', previous.deactivated_at = datetime())
                MERGE (document)-[:ACTIVE_REVISION]->(revision)
                SET revision.status = 'ACTIVE', revision.vector_ready = true,
                    document.status = 'READY'
                RETURN actual
                """,
                document_id=document_id,
                revision_id=revision_id,
                expected=expected_chunks,
            ).single()
            if not result:
                raise RuntimeError(
                    f"Revision activation failed because stored chunk count did not equal {expected_chunks}."
                )

    def mark_graph_ready(self, revision_id: str) -> None:
        with self._session() as session:
            session.run(
                "MATCH (revision:DocumentRevision {id: $revision_id}) SET revision.graph_ready = true",
                revision_id=revision_id,
            ).consume()

    def pending_graph_parents(self, limit: int = 100) -> list[dict[str, Any]]:
        scope = (
            "MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)-[:HAS_REVISION]->"
            "(revision:DocumentRevision)-[:HAS_PARENT]->(parent:ParentChunk)"
            if self.corpus_id
            else "MATCH (revision:DocumentRevision)-[:HAS_PARENT]->(parent:ParentChunk)"
        )
        with self._session() as session:
            rows = session.run(
                f"""
                {scope}
                WHERE revision.vector_ready = true AND revision.graph_ready = false
                  AND parent.graph_status IN ['PENDING', 'FAILED', 'PROVISIONAL']
                OPTIONAL MATCH (parent)-[:HAS_CHILD]->(chunk:Chunk)
                RETURN revision.id AS revision_id, parent.id AS parent_id, parent.text AS text,
                       collect(chunk.id) AS child_ids, parent.graph_status AS graph_status,
                       revision.created_at AS revision_created_at, parent.position AS parent_position
                ORDER BY CASE graph_status WHEN 'PENDING' THEN 0 WHEN 'FAILED' THEN 1 ELSE 2 END,
                         revision_created_at, parent_position
                LIMIT $limit
                """,
                limit=limit,
                corpus_id=self.corpus_id,
            )
            return [dict(row) for row in rows]

    def persist_parent_graph(
        self,
        parent_id: str,
        child_ids: Sequence[str],
        graph_document: Any,
        revision_id: str | None = None,
        extraction_state: str = "VERIFIED",
    ) -> None:
        if extraction_state not in {"VERIFIED", "PROVISIONAL"}:
            raise ValueError("extraction_state must be VERIFIED or PROVISIONAL.")
        node_rows = [
            {
                "id": str(node.id),
                "type": str(node.type or "Entity"),
                "properties": dict(getattr(node, "properties", None) or {}),
            }
            for node in graph_document.nodes
        ]
        relationship_rows = [
            {
                "source_id": str(relationship.source.id),
                "target_id": str(relationship.target.id),
                "type": str(relationship.type or "RELATED_TO"),
                "properties": dict(getattr(relationship, "properties", None) or {}),
            }
            for relationship in graph_document.relationships
        ]
        nodes_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in node_rows:
            nodes_by_label[_cypher_identifier(row["type"], "Entity")].append(row)
        relationships_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in relationship_rows:
            relationships_by_type[_cypher_identifier(row["type"], "RELATED_TO")].append(row)
        with self._session() as session:
            session.run(
                """
                MATCH (parent:ParentChunk {id: $parent_id})-[mention:HAS_ENTITY]->()
                DELETE mention
                """,
                parent_id=parent_id,
            ).consume()
            session.run(
                """
                MATCH ()-[relation]->()
                WHERE $parent_id IN coalesce(relation.source_parent_ids, [])
                SET relation.source_parent_ids = [id IN relation.source_parent_ids WHERE id <> $parent_id]
                SET relation.provisional_parent_ids = [
                    id IN coalesce(relation.provisional_parent_ids, []) WHERE id <> $parent_id
                ]
                WITH collect(DISTINCT relation) AS relations
                FOREACH (relation IN [
                    item IN relations
                    WHERE size(item.source_parent_ids) = 0
                      AND item.created_in_revision = $revision_id
                ] | DELETE relation)
                """,
                parent_id=parent_id,
                revision_id=revision_id,
            ).consume()
            for label, rows in nodes_by_label.items():
                session.run(
                    f"""
                    UNWIND $nodes AS row
                    MERGE (node:__Entity__ {{id: row.id}})
                    ON CREATE SET node:{label},
                                  node += row.properties,
                                  node.entity_type = row.type,
                                  node.created_in_revision = $revision_id,
                                  node.created_at = datetime()
                    SET node.last_seen_revision = $revision_id
                    """,
                    nodes=rows,
                    revision_id=revision_id,
                ).consume()
            for relationship_type, rows in relationships_by_type.items():
                session.run(
                    f"""
                    UNWIND $relationships AS row
                    MATCH (source:__Entity__ {{id: row.source_id}})
                    MATCH (target:__Entity__ {{id: row.target_id}})
                    MERGE (source)-[rel:{relationship_type}]->(target)
                    ON CREATE SET rel += row.properties,
                                  rel.extracted_type = row.type,
                                  rel.created_in_revision = $revision_id
                    SET rel.source_parent_ids = CASE
                            WHEN $parent_id IN coalesce(rel.source_parent_ids, []) THEN rel.source_parent_ids
                            ELSE coalesce(rel.source_parent_ids, []) + [$parent_id]
                        END,
                        rel.provisional_parent_ids = CASE
                            WHEN $extraction_state = 'PROVISIONAL'
                                 AND NOT $parent_id IN coalesce(rel.provisional_parent_ids, [])
                            THEN coalesce(rel.provisional_parent_ids, []) + [$parent_id]
                            WHEN $extraction_state = 'VERIFIED'
                            THEN [id IN coalesce(rel.provisional_parent_ids, []) WHERE id <> $parent_id]
                            ELSE coalesce(rel.provisional_parent_ids, [])
                        END
                    """,
                    relationships=rows,
                    parent_id=parent_id,
                    revision_id=revision_id,
                    extraction_state=extraction_state,
                ).consume()
            session.run(
                """
                MATCH (parent:ParentChunk {id: $parent_id})
                UNWIND $entity_ids AS entity_id
                MATCH (entity:__Entity__ {id: entity_id})
                MERGE (parent)-[mention:HAS_ENTITY]->(entity)
                SET mention.extraction_scope = 'parent', mention.extraction_state = $extraction_state
                """,
                parent_id=parent_id,
                entity_ids=[row["id"] for row in node_rows],
                extraction_state=extraction_state,
            ).consume()
            session.run(
                "MATCH (parent:ParentChunk {id: $parent_id}) "
                "SET parent.graph_status = CASE WHEN $extraction_state = 'PROVISIONAL' "
                "THEN 'PROVISIONAL' ELSE 'COMPLETED' END, parent.graph_error = null, "
                "parent.graph_extraction_state = $extraction_state",
                parent_id=parent_id,
                extraction_state=extraction_state,
            ).consume()

    def fail_parent_graph(self, parent_id: str, message: str) -> None:
        with self._session() as session:
            session.run(
                """
                MATCH (parent:ParentChunk {id: $parent_id})
                SET parent.graph_status = 'FAILED', parent.graph_error = $message,
                    parent.graph_attempts = coalesce(parent.graph_attempts, 0) + 1
                """,
                parent_id=parent_id,
                message=message[:2000],
            ).consume()

    def pending_claim_parents(
        self,
        limit: int,
        extraction_fingerprint: str,
    ) -> list[dict[str, Any]]:
        if not self.corpus_id:
            raise ValueError("Claim batching requires a corpus-scoped store.")
        if limit <= 0 or not extraction_fingerprint:
            raise ValueError("Claim batching requires a positive limit and extraction fingerprint.")
        with self._session() as session:
            return [
                dict(row)
                for row in session.run(
                    """
                    MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)
                          -[:ACTIVE_REVISION]->(revision:DocumentRevision)-[:HAS_PARENT]->
                          (parent:ParentChunk)
                    WHERE parent.claim_status IS NULL OR parent.claim_status = 'FAILED'
                       OR coalesce(parent.claim_extraction_fingerprint, '') <> $extraction_fingerprint
                    RETURN parent.id AS parent_id, parent.text AS text,
                           revision.id AS revision_id
                    ORDER BY revision.id, parent.position, parent.id
                    LIMIT $limit
                    """,
                    corpus_id=self.corpus_id,
                    extraction_fingerprint=extraction_fingerprint,
                    limit=limit,
                )
            ]

    def persist_parent_claims(
        self,
        parent_id: str,
        claims: Sequence[dict[str, Any]],
        *,
        extraction_fingerprint: str | None = None,
    ) -> None:
        for claim in claims:
            if str(claim.get("source_parent_id") or "") != parent_id:
                raise ValueError("Every claim must identify the parent being persisted.")
            if claim.get("stance") not in {"SUPPORTS", "CONTRADICTS"}:
                raise ValueError("Claim stance must be SUPPORTS or CONTRADICTS.")
        scope = (
            "MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)-[:HAS_REVISION]->"
            "(:DocumentRevision)-[:HAS_PARENT]->(parent:ParentChunk {id: $parent_id})"
            if self.corpus_id
            else "MATCH (parent:ParentChunk {id: $parent_id})"
        )
        with self._session() as session:
            result = session.run(
                f"""
                {scope}
                OPTIONAL MATCH (parent)-[old:SUPPORTS|CONTRADICTS]->(:Claim)
                WITH parent, collect(old) AS old_evidence
                FOREACH (evidence IN old_evidence | DELETE evidence)
                CALL {{
                    WITH parent
                    UNWIND $rows AS row
                    MERGE (claim:Claim {{id: row.id}})
                    ON CREATE SET claim.subject = row.subject, claim.predicate = row.predicate,
                                  claim.object = row.object, claim.valid_from = row.valid_from,
                                  claim.valid_to = row.valid_to, claim.created_at = datetime()
                    FOREACH (_ IN CASE WHEN row.stance = 'SUPPORTS' THEN [1] ELSE [] END |
                        MERGE (parent)-[supported:SUPPORTS]->(claim)
                        SET supported.extraction_confidence = row.extraction_confidence)
                    FOREACH (_ IN CASE WHEN row.stance = 'CONTRADICTS' THEN [1] ELSE [] END |
                        MERGE (parent)-[contradicted:CONTRADICTS]->(claim)
                        SET contradicted.extraction_confidence = row.extraction_confidence)
                    RETURN count(row) AS persisted
                }}
                SET parent.claim_status = 'COMPLETED',
                    parent.claim_extraction_fingerprint = $extraction_fingerprint,
                    parent.claim_error = null
                RETURN persisted
                """,
                corpus_id=self.corpus_id,
                parent_id=parent_id,
                rows=[dict(claim) for claim in claims],
                extraction_fingerprint=extraction_fingerprint,
            )
            if result.single() is None:
                raise ValueError(f"Parent is not part of the selected corpus: {parent_id}")

    def fail_parent_claims(self, parent_id: str, message: str) -> None:
        scope = (
            "MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)-[:ACTIVE_REVISION]->"
            "(:DocumentRevision)-[:HAS_PARENT]->(parent:ParentChunk {id: $parent_id})"
            if self.corpus_id
            else "MATCH (parent:ParentChunk {id: $parent_id})"
        )
        with self._session() as session:
            session.run(
                f"""
                {scope}
                SET parent.claim_status = 'FAILED', parent.claim_error = $message
                """,
                corpus_id=self.corpus_id,
                parent_id=parent_id,
                message=message[:2000],
            ).consume()

    def finalize_graph_revisions(self) -> int:
        scope = (
            "MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)-[:HAS_REVISION]->"
            "(revision:DocumentRevision)"
            if self.corpus_id
            else "MATCH (revision:DocumentRevision)"
        )
        with self._session() as session:
            row = session.run(
                f"""
                {scope}
                WHERE revision.vector_ready = true AND revision.graph_ready = false
                  AND NOT EXISTS {{
                    MATCH (revision)-[:HAS_PARENT]->(parent:ParentChunk)
                    WHERE parent.graph_status <> 'COMPLETED'
                  }}
                SET revision.graph_ready = true
                RETURN count(revision) AS count
                """,
                corpus_id=self.corpus_id,
            ).single()
            return int(row["count"]) if row else 0

    def deactivate_document(self, document_id: str) -> None:
        with self._session() as session:
            session.run(
                """
                MATCH (document:Document {id: $document_id})
                OPTIONAL MATCH (document)-[active:ACTIVE_REVISION]->(revision:DocumentRevision)
                DELETE active
                SET document.status = 'DELETED'
                FOREACH (_ IN CASE WHEN revision IS NULL THEN [] ELSE [1] END |
                    SET revision.status = 'INACTIVE')
                WITH document
                OPTIONAL MATCH (source:CorpusSource)-[materialized:MATERIALIZED_AS]->(document)
                DELETE materialized
                WITH source
                WHERE source IS NOT NULL
                SET source.retrieval_status = CASE
                    WHEN (source)-[:LEGACY_EVIDENCE]->() THEN 'LEGACY_ONLY' ELSE 'INACTIVE' END
                """,
                document_id=document_id,
            ).consume()

    def restore_revision(self, document_id: str, revision_id: str) -> None:
        with self._session() as session:
            result = session.run(
                """
                MATCH (document:Document {id: $document_id})-[:HAS_REVISION]->(revision:DocumentRevision {id: $revision_id})
                OPTIONAL MATCH (document)-[current:ACTIVE_REVISION]->(:DocumentRevision)
                DELETE current
                MERGE (document)-[:ACTIVE_REVISION]->(revision)
                SET document.status = 'READY', revision.status = 'ACTIVE', revision.vector_ready = true
                WITH revision, document
                OPTIONAL MATCH (source:CorpusSource)-[:MATERIALIZED_AS]->(document)
                SET source.content_checksum = revision.checksum,
                    source.retrieval_status = 'ACTIVE', source.ingestion_status = 'INGESTED'
                RETURN revision.id AS revision_id
                """,
                document_id=document_id,
                revision_id=revision_id,
            ).single()
            if not result:
                raise RuntimeError(f"Could not restore revision {revision_id} for document {document_id}.")

    def fail_revision(self, revision_id: str, message: str) -> None:
        with self._session() as session:
            session.run(
                """
                MATCH (document:Document)-[:HAS_REVISION]->(revision:DocumentRevision {id: $revision_id})
                OPTIONAL MATCH (document)-[:ACTIVE_REVISION]->(active:DocumentRevision)
                SET revision.status = 'FAILED', revision.error = $message,
                    document.status = CASE WHEN active IS NULL THEN 'FAILED' ELSE 'READY' END
                """,
                revision_id=revision_id,
                message=message[:2000],
            ).consume()

    def garbage_collect(
        self,
        document_ids: Sequence[str] | None = None,
        *,
        revision_ids: Sequence[str] | None = None,
    ) -> dict[str, int]:
        with self._session() as session:
            relationship_row = session.run(
                """
                MATCH (document:Document)-[:HAS_REVISION]->(revision:DocumentRevision)-[:HAS_PARENT]->(parent:ParentChunk)
                WHERE revision.status IN ['INACTIVE', 'FAILED']
                  AND NOT (:Document)-[:ACTIVE_REVISION]->(revision)
                  AND ($document_ids IS NULL OR document.id IN $document_ids)
                  AND ($revision_ids IS NULL OR revision.id IN $revision_ids)
                MATCH ()-[relation]->()
                WHERE parent.id IN coalesce(relation.source_parent_ids, [])
                SET relation.source_parent_ids = [id IN relation.source_parent_ids WHERE id <> parent.id]
                WITH collect(DISTINCT relation) AS relations
                WITH relations, [
                    relation IN relations
                    WHERE size(relation.source_parent_ids) = 0
                      AND ($revision_ids IS NULL OR relation.created_in_revision IN $revision_ids)
                ] AS retired
                FOREACH (relation IN retired | DELETE relation)
                RETURN size(retired) AS relationships
                """,
                document_ids=list(document_ids) if document_ids is not None else None,
                revision_ids=list(revision_ids) if revision_ids is not None else None,
            ).single()
            row = session.run(
                """
                MATCH (document:Document)-[:HAS_REVISION]->(revision:DocumentRevision)
                WHERE revision.status IN ['INACTIVE', 'FAILED']
                  AND NOT (:Document)-[:ACTIVE_REVISION]->(revision)
                  AND ($document_ids IS NULL OR document.id IN $document_ids)
                  AND ($revision_ids IS NULL OR revision.id IN $revision_ids)
                OPTIONAL MATCH (revision)<-[:IN_REVISION]-(chunk:Chunk)
                OPTIONAL MATCH (revision)-[:HAS_PARENT]->(parent:ParentChunk)
                WITH collect(DISTINCT revision) AS revisions,
                     collect(DISTINCT chunk) AS chunks,
                     collect(DISTINCT parent) AS parents
                FOREACH (node IN chunks | DETACH DELETE node)
                FOREACH (node IN parents | DETACH DELETE node)
                FOREACH (node IN revisions | DETACH DELETE node)
                RETURN size(revisions) AS revisions, size(chunks) AS chunks, size(parents) AS parents
                """,
                document_ids=list(document_ids) if document_ids is not None else None,
                revision_ids=list(revision_ids) if revision_ids is not None else None,
            ).single()
            entity_row = None
            if revision_ids is not None:
                entity_row = session.run(
                    """
                    MATCH (entity:__Entity__)
                    WHERE entity.created_in_revision IN $revision_ids
                      AND NOT (entity)<-[:HAS_ENTITY]-(:ParentChunk)
                      AND NOT (entity)--(:__Entity__)
                    WITH collect(entity) AS entities
                    FOREACH (entity IN entities | DETACH DELETE entity)
                    RETURN size(entities) AS entities
                    """,
                    revision_ids=list(revision_ids),
                ).single()
            elif document_ids is None:
                entity_row = session.run(
                    """
                    MATCH (entity:__Entity__)
                    WHERE NOT (entity)<-[:HAS_ENTITY]-(:ParentChunk)
                      AND NOT (entity)--(:__Entity__)
                    WITH collect(entity) AS entities
                    FOREACH (entity IN entities | DETACH DELETE entity)
                    RETURN size(entities) AS entities
                    """
                ).single()
            result = dict(row) if row else {"revisions": 0, "chunks": 0, "parents": 0}
            result["relationships"] = int(relationship_row["relationships"]) if relationship_row else 0
            result["entities"] = int(entity_row["entities"]) if entity_row else 0
            return result


def _batches(rows: Sequence[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), batch_size):
        yield list(rows[start : start + batch_size])
