from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CommunityConfig:
    max_cluster_size: int = 10
    seed: int = 42
    relationship_weighting: str = "active_parent_count"
    algorithm: str = "hierarchical_leiden"
    algorithm_version: str = "graspologic-native-1.2"
    prompt_hash: str = "default"
    report_model_fingerprint: str = "unconfigured"
    embedding_fingerprint: str = "unconfigured"

    def __post_init__(self) -> None:
        if self.max_cluster_size <= 0:
            raise ValueError("max_cluster_size must be positive.")
        if self.relationship_weighting != "active_parent_count":
            raise ValueError("Unsupported community relationship weighting.")
        if self.algorithm != "hierarchical_leiden":
            raise ValueError("Unsupported community algorithm.")


@dataclass(frozen=True)
class EvidenceParent:
    id: str
    document_id: str
    revision_id: str
    text: str
    title: str = ""
    source_uri: str = ""


@dataclass(frozen=True)
class EntityRecord:
    id: str
    title: str
    description: str
    parent_ids: tuple[str, ...]


@dataclass(frozen=True)
class RelationshipRecord:
    id: str
    source_id: str
    target_id: str
    description: str
    weight: float
    parent_ids: tuple[str, ...]


@dataclass(frozen=True)
class CommunityProjection:
    corpus_id: str
    active_revision_ids: tuple[str, ...]
    parents: tuple[EvidenceParent, ...]
    entities: tuple[EntityRecord, ...]
    relationships: tuple[RelationshipRecord, ...]

    @property
    def active_revision_hash(self) -> str:
        return stable_hash(sorted(self.active_revision_ids))

    @property
    def fingerprint(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class CommunityRecord:
    id: str
    build_id: str
    level: int
    source_cluster: int
    parent_id: str | None
    member_ids: tuple[str, ...]
    rank: float


@dataclass(frozen=True)
class CommunityFinding:
    id: str
    report_id: str
    position: int
    summary: str
    explanation: str
    parent_ids: tuple[str, ...]
    confidence: float = 1.0


@dataclass(frozen=True)
class CommunityReport:
    id: str
    build_id: str
    community_id: str
    title: str
    summary: str
    full_content: str
    rank: float
    rating_explanation: str
    findings: tuple[CommunityFinding, ...]
    embedding: tuple[float, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommunityBuildResult:
    id: str
    corpus_id: str
    active_revision_ids: tuple[str, ...]
    active_revision_hash: str
    projection_fingerprint: str
    configuration_fingerprint: str
    communities: tuple[CommunityRecord, ...]
    reports: tuple[CommunityReport, ...] = ()

    def with_reports(self, reports: tuple[CommunityReport, ...]) -> "CommunityBuildResult":
        return CommunityBuildResult(
            self.id,
            self.corpus_id,
            self.active_revision_ids,
            self.active_revision_hash,
            self.projection_fingerprint,
            self.configuration_fingerprint,
            self.communities,
            reports,
        )
