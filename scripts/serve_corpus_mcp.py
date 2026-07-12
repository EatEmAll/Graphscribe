#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from notebooklm_graph_pipe.paths import REPO_ROOT
from notebooklm_graph_pipe.service.core import CorpusService
from notebooklm_graph_pipe.service.jobs import CorpusJobManager
from notebooklm_graph_pipe.service.mcp_server import create_mcp_server
from notebooklm_graph_pipe.service.registry import CorpusRegistry
from notebooklm_graph_pipe.service.runtime import RuntimeFactory


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve corpus research tools over MCP stdio.")
    parser.add_argument("--registry-root", default=str(REPO_ROOT / "data" / "corpora"))
    parser.add_argument("--llm-routing-config")
    args = parser.parse_args()
    registry = CorpusRegistry(Path(args.registry_root))
    service = CorpusService(registry, RuntimeFactory(args.llm_routing_config), CorpusJobManager(registry, REPO_ROOT))
    create_mcp_server(service).run(transport="stdio")


if __name__ == "__main__":
    main()
