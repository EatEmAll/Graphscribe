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
from notebooklm_graph_pipe.service.api import create_source_resolution_app
from notebooklm_graph_pipe.service.registry import CorpusRegistry
from notebooklm_graph_pipe.service.security import load_or_create_token
from notebooklm_graph_pipe.service.source_resolution import SourceResolutionService


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve read-only exact corpus source resolution.")
    parser.add_argument("--registry-root", default=str(REPO_ROOT / "data" / "corpora"))
    parser.add_argument("--token-path", default=str(REPO_ROOT / ".local" / "api_token"))
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    service = SourceResolutionService(CorpusRegistry(Path(args.registry_root)))
    app = create_source_resolution_app(
        service, load_or_create_token(Path(args.token_path)),
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
