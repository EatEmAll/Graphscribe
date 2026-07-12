from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from notebooklm_graph_pipe.ingestion.manifest import CorpusManifest, load_manifest


@dataclass(frozen=True)
class CorpusRegistryEntry:
    key: str
    manifest_path: Path
    manifest: CorpusManifest


class CorpusRegistry:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def entries(self) -> dict[str, CorpusRegistryEntry]:
        result: dict[str, CorpusRegistryEntry] = {}
        if not self.root.exists():
            return result
        for path in sorted(self.root.glob("*/manifest.json")):
            manifest = load_manifest(path)
            if manifest is None:
                continue
            if manifest.corpus_key in result:
                raise ValueError(f"Duplicate corpus key in registry: {manifest.corpus_key}")
            result[manifest.corpus_key] = CorpusRegistryEntry(manifest.corpus_key, path.resolve(), manifest)
        return result

    def get(self, key: str) -> CorpusRegistryEntry:
        try:
            return self.entries()[key]
        except KeyError as exc:
            raise KeyError(f"Corpus not found: {key}") from exc

    def validate_dataset_root(self, entry: CorpusRegistryEntry) -> Path:
        if not entry.manifest.dataset_root:
            raise ValueError(f"Corpus {entry.key} does not declare a dataset_root.")
        root = Path(entry.manifest.dataset_root).resolve()
        if not root.is_dir():
            raise ValueError(f"Registered dataset root is unavailable: {root}")
        return root
