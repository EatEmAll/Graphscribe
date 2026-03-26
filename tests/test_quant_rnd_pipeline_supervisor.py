from __future__ import annotations

from pathlib import Path


def test_quant_rnd_supervisor_uses_local_runtime_contract() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "quant_rnd_pipeline_supervisor.ps1"
    script_text = script_path.read_text(encoding="utf-8")

    assert "Invoke-RestMethod" not in script_text
    assert "--backend-url" not in script_text
    assert "notebooklm_graph_pipe.runtime.graph_builder_runtime import GraphBuilderAPI" in script_text
    assert "\"-m\", \"notebooklm_graph_pipe.cli.build_graph\"" in script_text
