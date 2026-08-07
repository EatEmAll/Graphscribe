from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

from notebooklm_graph_pipe.cli import migrate_neo4j
from notebooklm_graph_pipe.runtime.neo4j_connection import ResolvedNeo4jConnection

from .core import StepResult, WorkflowContext


@dataclass(frozen=True)
class UpgradeInPlaceStep:
    connection: ResolvedNeo4jConnection
    manifest_path: Path
    backup_file: Path | None
    execute: bool
    confirm_source: str | None
    name: str = "upgrade-in-place"
    version: str = "1"

    def fingerprint(self, context: WorkflowContext) -> str:
        return "|".join(
            (
                self.version,
                self.connection.uri,
                self.connection.database,
                str(self.manifest_path.resolve()),
                str(self.backup_file.resolve()) if self.backup_file else "",
                str(self.execute),
            )
        )

    def run(self, context: WorkflowContext) -> StepResult:
        summary = migrate_neo4j.upgrade_in_place(
            self.connection,
            manifest_path=self.manifest_path,
            backup_file=self.backup_file,
            execute=self.execute,
            confirm_source=self.confirm_source,
            migration_run_id=context.run_dir.name,
        )
        inventory = dict(summary.get("inventory") or {})
        return StepResult(
            outputs=summary,
            counters={
                "nodes": int(inventory.get("total_nodes") or 0),
                "relationships": int(inventory.get("total_relationships") or 0),
            },
        )


@dataclass(frozen=True)
class MigrationDelegateStep:
    argv: tuple[str, ...]
    name: str = "neo4j-migrate"
    version: str = "1"

    def fingerprint(self, context: WorkflowContext) -> str:
        return "|".join((self.version, *self.argv))

    def run(self, context: WorkflowContext) -> StepResult:
        stream = io.StringIO()
        with redirect_stdout(stream):
            return_code = migrate_neo4j.main(list(self.argv))
        if return_code:
            raise RuntimeError(f"Migration exited with status {return_code}.")
        output = stream.getvalue().strip()
        try:
            summary = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Migration did not return a JSON summary.") from exc
        return StepResult(outputs=summary)
