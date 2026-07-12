#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from notebooklm_graph_pipe.paths import REPO_ROOT
from notebooklm_graph_pipe.service.api import create_app
from notebooklm_graph_pipe.service.core import CorpusService
from notebooklm_graph_pipe.service.jobs import CorpusJobManager
from notebooklm_graph_pipe.service.registry import CorpusRegistry
from notebooklm_graph_pipe.service.runtime import RuntimeFactory
from notebooklm_graph_pipe.service.security import load_or_create_token


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local Neo4j corpus REST API.")
    parser.add_argument("--registry-root", default=str(REPO_ROOT / "data" / "corpora"))
    parser.add_argument("--token-path", default=str(REPO_ROOT / ".local" / "api_token"))
    parser.add_argument("--llm-routing-config")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    registry = CorpusRegistry(Path(args.registry_root))
    service = CorpusService(registry, RuntimeFactory(args.llm_routing_config), CorpusJobManager(registry, REPO_ROOT))
    app = create_app(service, load_or_create_token(Path(args.token_path)))
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
