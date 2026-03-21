from pathlib import Path


_VENDORED_SHARED = Path(__file__).resolve().parents[2] / "vendor" / "llm-graph-builder" / "backend" / "src" / "shared"
if _VENDORED_SHARED.is_dir():
    __path__.append(str(_VENDORED_SHARED))
