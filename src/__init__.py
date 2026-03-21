from pathlib import Path


_VENDORED_SRC = Path(__file__).resolve().parents[1] / "vendor" / "llm-graph-builder" / "backend" / "src"
if _VENDORED_SRC.is_dir():
    __path__.append(str(_VENDORED_SRC))
