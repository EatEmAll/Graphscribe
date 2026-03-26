from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
CONFIG_DIR = REPO_ROOT / "config"
MODELS_DIR = REPO_ROOT / "models"
RUNS_DIR = REPO_ROOT / "runs"
CONSOLIDATION_CACHE_DIR = RUNS_DIR / "cache" / "consolidation"
CONSOLIDATION_REPORT_DIR = RUNS_DIR / "reports" / "consolidation"
LOCAL_MODEL_DIR = MODELS_DIR / "local_model"

