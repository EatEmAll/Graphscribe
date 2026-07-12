from __future__ import annotations

import json
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .registry import CorpusRegistry, CorpusRegistryEntry


@dataclass
class JobRecord:
    id: str
    corpus_key: str
    status: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    return_code: int | None = None
    output: str = ""
    error: str = ""


class CorpusJobManager:
    def __init__(self, registry: CorpusRegistry, repository_root: Path):
        self.registry = registry
        self.repository_root = repository_root.resolve()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="corpus-job")
        self._records: dict[str, JobRecord] = {}
        self._active_corpora: set[str] = set()
        self._lock = threading.Lock()

    def _job_path(self, entry: CorpusRegistryEntry, job_id: str) -> Path:
        path = entry.manifest_path.parent / "jobs" / f"{job_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _save(self, entry: CorpusRegistryEntry, record: JobRecord) -> None:
        path = self._job_path(entry, record.id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        temporary.replace(path)

    def submit_sync(self, entry: CorpusRegistryEntry) -> JobRecord:
        self.registry.validate_dataset_root(entry)
        with self._lock:
            if entry.key in self._active_corpora:
                raise RuntimeError(f"A mutating job is already running for corpus {entry.key}.")
            record = JobRecord(
                id=str(uuid.uuid4()),
                corpus_key=entry.key,
                status="queued",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self._records[record.id] = record
            self._active_corpora.add(entry.key)
            self._save(entry, record)
            self._executor.submit(self._run_sync, entry, record)
            return record

    def _run_sync(self, entry: CorpusRegistryEntry, record: JobRecord) -> None:
        record.status = "running"
        record.started_at = datetime.now(timezone.utc).isoformat()
        self._save(entry, record)
        command = [
            sys.executable,
            str(self.repository_root / "scripts" / "sync_corpus_graph.py"),
            "update",
            "--dataset-dir",
            str(self.registry.validate_dataset_root(entry)),
            "--corpus-key",
            entry.key,
            "--export-dir",
            str(entry.manifest_path.parent),
        ]
        try:
            result = subprocess.run(
                command,
                cwd=self.repository_root,
                capture_output=True,
                text=True,
                check=False,
            )
            record.return_code = result.returncode
            record.output = result.stdout[-20000:]
            record.error = result.stderr[-20000:]
            record.status = "completed" if result.returncode == 0 else "failed"
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
        finally:
            record.completed_at = datetime.now(timezone.utc).isoformat()
            self._save(entry, record)
            with self._lock:
                self._active_corpora.discard(entry.key)

    def get(self, job_id: str) -> JobRecord:
        try:
            return self._records[job_id]
        except KeyError:
            for path in self.registry.root.glob(f"*/jobs/{job_id}.json"):
                return JobRecord(**json.loads(path.read_text(encoding="utf-8")))
            raise KeyError(f"Job not found: {job_id}")

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)
