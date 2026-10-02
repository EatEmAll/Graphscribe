"""Canonical entity vocabulary for source-grounded graph extraction.

A vocabulary names the concepts whose mentions must converge on one stable entity id, so graph
expansion can bridge every parent chunk that discusses the same concept. Each alias is matched as
a whole phrase whose words may be separated by spaces, hyphens, underscores, or slashes, with an
optional plural suffix.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_ALIAS_SHAPE = re.compile(r"^[a-z0-9](?:[a-z0-9 '_/-]*[a-z0-9])?$")
_SEPARATORS = re.compile(r"[\s_/-]+")


def normalize_term(value: str) -> str:
    return _SEPARATORS.sub(" ", value.casefold()).strip()


def _compact(value: str) -> str:
    return normalize_term(value).replace(" ", "")


def _alias_pattern(alias: str) -> str:
    return r"[\s_/-]+".join(re.escape(token) for token in normalize_term(alias).split(" "))


@dataclass(frozen=True)
class VocabularyConcept:
    id: str
    type: str
    aliases: tuple[str, ...]

    @property
    def terms(self) -> tuple[str, ...]:
        """The canonical id and aliases, longest first so alternation prefers full phrases."""
        return tuple(sorted({normalize_term(self.id), *map(normalize_term, self.aliases)}, key=lambda t: (-len(t), t)))

    @property
    def pattern(self) -> str:
        """A regex body valid in both Python and Neo4j (Java) regular expressions."""
        return r"\b(?:" + "|".join(_alias_pattern(term) for term in self.terms) + r")(?:s|es)?\b"


@dataclass(frozen=True)
class EntityVocabulary:
    concepts: tuple[VocabularyConcept, ...]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "EntityVocabulary":
        raw = payload.get("concepts") if isinstance(payload, dict) else None
        if not isinstance(raw, list) or not raw:
            raise ValueError("Entity vocabulary must contain a non-empty concepts list.")
        concepts: list[VocabularyConcept] = []
        owners: dict[str, str] = {}
        for item in raw:
            concept_id = str(item.get("id") or "").strip()
            concept_type = str(item.get("type") or "").strip()
            aliases = tuple(str(alias).strip() for alias in item.get("aliases") or [])
            if not concept_id or not concept_type:
                raise ValueError("Every vocabulary concept needs an id and a type.")
            concept = VocabularyConcept(concept_id, concept_type, aliases)
            for term in concept.terms:
                if not _ALIAS_SHAPE.match(term):
                    raise ValueError(f"Vocabulary term {term!r} of {concept_id!r} has an unsupported shape.")
                # Compact keys also catch CamelCase ids such as "PurgedCrossValidation".
                owner = owners.setdefault(_compact(term), concept_id)
                if owner != concept_id:
                    raise ValueError(f"Vocabulary term {term!r} is claimed by {owner!r} and {concept_id!r}.")
            concepts.append(concept)
        if len({concept.id for concept in concepts}) != len(concepts):
            raise ValueError("Vocabulary concept ids must be unique.")
        return cls(tuple(concepts))

    @classmethod
    def from_path(cls, path: str | Path) -> "EntityVocabulary":
        return cls.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            [[concept.id, concept.type, list(concept.terms)] for concept in self.concepts],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def mention_pattern(self) -> str:
        """Full-match pattern for Cypher ``=~`` selecting text that mentions any concept."""
        return "(?is).*(?:" + "|".join(concept.pattern for concept in self.concepts) + ").*"

    def mentioned(self, text: str) -> list[VocabularyConcept]:
        return [
            concept
            for concept in self.concepts
            if re.search(concept.pattern, text, flags=re.IGNORECASE)
        ]

    def canonical(self, entity_id: str) -> VocabularyConcept | None:
        key = _compact(entity_id)
        return next(
            (concept for concept in self.concepts if key in {_compact(term) for term in concept.terms}),
            None,
        )

    def prompt_block(self, text: str) -> str:
        """Name only the concepts this text mentions, keeping the extraction prompt short."""
        concepts = self.mentioned(text)
        if not concepts:
            return ""
        lines = [
            "Canonical vocabulary: this text discusses the concepts below. Name each one's node with "
            "exactly the id and type given here instead of a variant spelling."
        ]
        lines.extend(f"- {concept.id} ({concept.type})" for concept in concepts)
        return "\n".join(lines)

    def apply(self, graph_document: Any, text: str) -> SimpleNamespace:
        """Rename variant nodes to canonical ids and add concepts the text mentions literally."""
        nodes: dict[str, SimpleNamespace] = {}
        renamed: dict[str, str] = {}
        for node in graph_document.nodes:
            # The store merges on a non-empty string ``id`` property, so that name is canonicalized
            # here like the node id and then dropped; a concept named by either id wins.
            properties = dict(getattr(node, "properties", None) or {})
            named_id = properties.pop("id", None)
            extracted_id = named_id.strip() if isinstance(named_id, str) and named_id.strip() else str(node.id)
            concept = self.canonical(extracted_id) or self.canonical(str(node.id))
            node_id, node_type = (concept.id, concept.type) if concept else (extracted_id, str(node.type))
            renamed[str(node.id)] = node_id
            if node_id not in nodes:
                nodes[node_id] = SimpleNamespace(id=node_id, type=node_type, properties=properties)
        for concept in self.mentioned(text):
            nodes.setdefault(concept.id, SimpleNamespace(id=concept.id, type=concept.type, properties={}))
        relationships: list[SimpleNamespace] = []
        seen: set[tuple[str, str, str]] = set()
        for relationship in graph_document.relationships:
            source = renamed.get(str(relationship.source.id), str(relationship.source.id))
            target = renamed.get(str(relationship.target.id), str(relationship.target.id))
            key = (source, target, str(relationship.type))
            if source == target or key in seen or source not in nodes or target not in nodes:
                continue
            seen.add(key)
            relationships.append(
                SimpleNamespace(
                    source=nodes[source],
                    target=nodes[target],
                    type=relationship.type,
                    properties=dict(getattr(relationship, "properties", None) or {}),
                )
            )
        return SimpleNamespace(nodes=list(nodes.values()), relationships=relationships)
