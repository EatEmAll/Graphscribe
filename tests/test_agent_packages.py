from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_local_corpus_research_packages_exist_without_notebooklm_tools() -> None:
    paths = [
        ROOT / ".codex" / "skills" / "neo4j-corpus-deep-research" / "SKILL.md",
        ROOT / ".claude" / "agents" / "neo4j-corpus-deep-research.md",
        ROOT / ".opencode" / "agents" / "neo4j-corpus-deep-research.md",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "corpus_answer" in text
        assert "notebooklm-mcp" not in text.lower()
    assert not (ROOT / ".codex" / "skills" / "notebooklm-neo4j-deep-research" / "SKILL.md").exists()
