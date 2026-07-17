from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from notebooklm_graph_pipe.runtime import llm_json_utils as utils


def test_build_single_prompt_clients_supports_subscription_clis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils.shutil, "which", lambda name: f"C:\\bin\\{name}.exe")
    monkeypatch.setenv("LLM_CLI_TIMEOUT_SECONDS", "45")

    clients = utils.build_single_prompt_clients("codex", "claude")

    assert clients["codex"] == utils.SubscriptionCliClient("codex", "C:\\bin\\codex.exe", 45.0)
    assert clients["claude"] == utils.SubscriptionCliClient("claude", "C:\\bin\\claude.exe", 45.0)


def test_codex_cli_uses_read_only_structured_noninteractive_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(args, *, prompt, cwd, timeout_seconds):
        captured.update(args=args, prompt=prompt, cwd=cwd, timeout_seconds=timeout_seconds)
        output_path = Path(args[args.index("--output-last-message") + 1])
        output_path.write_text('{"verdict":"SAME"}', encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(utils, "_run_cli", fake_run)
    response = utils._generate_cli_response(
        utils.SubscriptionCliClient("codex", "codex.exe", 30),
        model_name="gpt-5.6-luna",
        prompt="Judge this pair.",
        system_instruction="Return a verdict.",
        reasoning_effort="medium",
    )

    args = captured["args"]
    assert isinstance(args, list)
    assert args[:5] == ["codex.exe", "--ask-for-approval", "never", "--sandbox", "read-only"]
    assert "--ephemeral" in args
    assert "--output-schema" in args
    assert 'model_reasoning_effort="medium"' in args
    assert args[-1] == "-"
    assert response.output_text == '{"verdict":"SAME"}'


def test_claude_cli_disables_tools_and_extracts_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(args, *, prompt, cwd, timeout_seconds):
        captured.update(args=args, prompt=prompt, cwd=cwd, timeout_seconds=timeout_seconds)
        stdout = json.dumps({"structured_output": {"label": "Metric"}})
        return subprocess.CompletedProcess(args, 0, stdout, "")

    monkeypatch.setattr(utils, "_run_cli", fake_run)
    response = utils._generate_cli_response(
        utils.SubscriptionCliClient("claude", "claude.exe", 30),
        model_name="sonnet",
        prompt="Classify this node.",
        system_instruction="Return a label.",
        reasoning_effort="low",
    )

    args = captured["args"]
    assert isinstance(args, list)
    assert args[args.index("--tools") + 1] == ""
    assert args[args.index("--permission-mode") + 1] == "dontAsk"
    assert "--no-session-persistence" in args
    assert args[args.index("--effort") + 1] == "low"
    assert json.loads(response.output_text) == {"label": "Metric"}


def test_run_cli_uses_argument_list_stdin_and_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_subprocess_run(args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return subprocess.CompletedProcess(args, 0, "{}", "")

    monkeypatch.setattr(utils.subprocess, "run", fake_subprocess_run)
    utils._run_cli(["codex", "exec", "-"], prompt="$(unsafe)", cwd=tmp_path, timeout_seconds=12)

    assert captured["args"] == ["codex", "exec", "-"]
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["input"] == "$(unsafe)"
    assert kwargs["shell"] is False
    assert kwargs["timeout"] == 12
