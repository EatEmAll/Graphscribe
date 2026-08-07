from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from notebooklm_graph_pipe.community.builder import CommunityReporter, NativeCommunityBuilder
from notebooklm_graph_pipe.community.models import CommunityConfig
from notebooklm_graph_pipe.community.store import Neo4jCommunityStore

from .core import StepResult, WorkflowContext


@dataclass(frozen=True)
class CommunityBuildStep:
    store: Neo4jCommunityStore
    builder: NativeCommunityBuilder
    reporter: CommunityReporter
    embedder: Any
    config: CommunityConfig
    name: str = "community-build"
    version: str = "1"

    def fingerprint(self, context: WorkflowContext) -> str:
        return "|".join(
            (
                self.version,
                str(context.inputs.get("active_revision_hash") or ""),
                str(context.inputs.get("projection_fingerprint") or ""),
                self.config.algorithm_version,
                self.config.prompt_hash,
                self.config.report_model_fingerprint,
                self.config.embedding_fingerprint,
                str(self.config.max_cluster_size),
                str(self.config.seed),
            )
        )

    def run(self, context: WorkflowContext) -> StepResult:
        projection = self.store.active_projection()
        expected_revision_hash = str(context.inputs.get("active_revision_hash") or "")
        if expected_revision_hash and projection.active_revision_hash != expected_revision_hash:
            raise RuntimeError("Active revisions changed after the community workflow was prepared.")
        build = self.builder.build(projection, self.config)
        complete = asyncio.run(
            self.builder.build_reports(
                build,
                projection,
                self.reporter,
                self.embedder.embed_documents,
            )
        )
        self.store.stage_build(complete, {"workflow_run_id": context.run_dir.name})
        self.store.activate_build(complete.id, complete.active_revision_hash)
        return StepResult(
            outputs={
                "build_id": complete.id,
                "active_revision_hash": complete.active_revision_hash,
                "projection_fingerprint": complete.projection_fingerprint,
            },
            counters={
                "communities": len(complete.communities),
                "reports": len(complete.reports),
                "findings": sum(len(report.findings) for report in complete.reports),
            },
        )
