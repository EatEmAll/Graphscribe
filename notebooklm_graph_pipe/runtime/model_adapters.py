from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .llm_json_utils import generate_json_payload
from .llm_routing import PromptRoleConfig
from .model_executor import ModelRequest, ModelUsage


@dataclass(frozen=True)
class RoutedJsonAdapter:
    role: PromptRoleConfig
    client: Any

    @property
    def provider(self) -> str:
        return self.role.client

    @property
    def model(self) -> str:
        return self.role.model

    def execute(self, request: ModelRequest) -> tuple[str, dict[str, object] | None, ModelUsage]:
        payload, error = generate_json_payload(
            self.client,
            client_name=self.role.client,
            model_name=self.role.model,
            prompt=request.prompt,
            system_instruction=request.system_instruction,
            max_output_tokens=request.max_output_tokens,
            temperature=request.temperature,
            reasoning_effort=self.role.reasoning_effort,
            max_attempts=1,
        )
        if payload is None:
            raise RuntimeError(error)
        return "", payload, ModelUsage()
