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
        schema_path = Path(args[args.index("--output-schema") + 1])
        captured["schema"] = json.loads(schema_path.read_text(encoding="utf-8"))
        output_path = Path(args[args.index("--output-last-message") + 1])
        output_path.write_text('{"payload":"{\\"verdict\\":\\"SAME\\"}"}', encoding="utf-8")
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
    assert captured["schema"] == utils.CODEX_CLI_JSON_SCHEMA
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


def test_routed_adapter_constrains_gemini_to_the_request_schema() -> None:
    from types import SimpleNamespace

    from notebooklm_graph_pipe.runtime.llm_routing import PromptRoleConfig
    from notebooklm_graph_pipe.runtime.model_adapters import RoutedJsonAdapter
    from notebooklm_graph_pipe.runtime.model_executor import ModelRequest

    configs = []

    class Models:
        def generate_content(self, *, model, contents, config):
            configs.append(config)
            return SimpleNamespace(text='{"nodes": [], "relationships": []}')

    schema = {"type": "object", "required": ["nodes"], "properties": {"nodes": {"type": "array"}}}
    adapter = RoutedJsonAdapter(
        PromptRoleConfig(client="genai", model="gemini-2.5-flash"), SimpleNamespace(models=Models())
    )
    _, payload, _ = adapter.execute(
        ModelRequest(role="r", prompt="p", system_instruction="s", response_schema=schema, max_output_tokens=64)
    )

    assert payload == {"nodes": [], "relationships": []}
    assert configs[0].response_mime_type == "application/json"
    assert configs[0].response_json_schema == schema


class _DecisionsResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


def test_decisions_client_hides_key_and_routes_only_to_non_retaining_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-secret")
    client = utils.build_single_prompt_clients("openrouter_decisions")["openrouter_decisions"]
    assert "sk-or-test-secret" not in repr(client)
    captured: dict[str, object] = {}

    def fake_post(url, *, json, headers, timeout):
        captured.update(url=url, body=json, headers=headers, timeout=timeout)
        return _DecisionsResponse(200, {"answers": {"same": {"type": "noul", "noul": 0.9}}})

    monkeypatch.setattr(utils.httpx, "post", fake_post)
    questions = {"same": {"type": "noul", "instructions": "Same?", "criteria": {"true": "yes", "false": "no"}}}

    answers = utils.request_decisions(client, model_name="typesafe/jev-1.13", state={"a": 1}, questions=questions)

    assert answers == {"same": {"type": "noul", "noul": 0.9}}
    assert captured["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert captured["body"] == {
        "model": "typesafe/jev-1.13",
        "state": {"a": 1},
        "questions": questions,
        "provider": {"data_collection": "deny", "zdr": True},
    }
    assert captured["headers"] == {"Authorization": "Bearer sk-or-test-secret"}


def test_decisions_client_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils.httpx, "post", lambda *a, **k: _DecisionsResponse(429, {"error": {"message": "rate limit"}}))

    with pytest.raises(RuntimeError, match="429"):
        utils.request_decisions(utils.OpenRouterDecisionsClient(api_key="k"), model_name="m", state="s", questions={})


def test_openrouter_json_client_requests_json_from_non_collecting_providers() -> None:
    captured: list[dict] = []

    class _Responses:
        def create(self, **kwargs):
            captured.append(kwargs)
            return utils.CliResponse(output_text='{"ok": true}')

    class _Client:
        responses = _Responses()

    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    for response_schema in (schema, None):
        payload, error = utils.generate_json_payload(
            _Client(), client_name="openrouter_json", model_name="openai/gpt-6-luna", prompt="p",
            system_instruction="s", max_output_tokens=10, response_schema=response_schema,
        )
        assert (payload, error) == ({"ok": True}, "")

    assert captured[0]["text"] == {"format": {"type": "json_schema", "name": "response", "schema": schema, "strict": False}}
    assert captured[1]["text"] == {"format": {"type": "json_object"}}
    assert all(call["extra_body"] == {"provider": {"data_collection": "deny"}} for call in captured)
