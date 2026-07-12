"""Canonical source ingestion for the local corpus pipeline."""

from .chunking import ChunkingConfig, HierarchicalChunker
from .models import CanonicalBlock, CanonicalDocument, ChildChunk, ParentChunk

__all__ = [
    "CanonicalBlock",
    "CanonicalDocument",
    "ChildChunk",
    "ChunkingConfig",
    "HierarchicalChunker",
    "ParentChunk",
]
