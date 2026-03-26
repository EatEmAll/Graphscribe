from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooklm_graph_pipe.cli.check_neo4j import check_neo4j_state


if __name__ == "__main__":
    check_neo4j_state()
