"""OpenAI-compatible endpoint provider (DeepSeek, vLLM, Ollama, LiteLLM, Fireworks, etc.)."""
from __future__ import annotations

import os
import time
from typing import Any, Optional

from autoredteam.prepare import TargetCapabilities
from autoredteam.providers._compat import ParamMemo
from autoredteam.providers.base import (
    BaseTargetSession, ProviderConfigurationError, ProviderDescriptor, ProviderResponse, TargetSpec,
)


class ChatCompletionsSession(BaseTargetSession):
    """Multi-turn session over any Chat Completions API."""

    # Many self-hosted servers still only understand the legacy name.
    token_param = "max_tokens"

    def __init__(self, spec: TargetSpec, client: Any, model: str):
        super().__init__(spec)
        self._client = client
        self._model = model
        self._history: list[dict[str, str]] = []
        self._params = ParamMemo()

    def send_user_turn(self, content: str, turn_index: int = 0,
                       metadata: Optional[dict[str, Any]] = None) -> ProviderResponse:
        self._history.append({"role": "user", "content": content})
        kwargs = {
            "model": self._model,
            "messages": [{"role": "system", "content": self.spec.system_prompt}] + self._history,
            "temperature": self.spec.temperature,
            self.token_param: self.spec.max_output_tokens,
        }
        t0 = time.monotonic()
        try:
            resp = self._params.call(self._client.chat.completions.create, kwargs)
        except Exception:
            self._history.pop()
            raise
        latency = (time.monotonic() - t0) * 1000
        choice = resp.choices[0]
        text = choice.message.content or ""
        self._history.append({"role": "assistant", "content": text})
        request_id = getattr(resp, "id", "") or ""
        return ProviderResponse(
            text=text, raw={"id": request_id},
            finish_reason=getattr(choice, "finish_reason", "") or "",
            provider_request_id=request_id, latency_ms=round(latency, 1),
        )

    def reset(self) -> None:
        self._history = []

    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def capabilities(self) -> TargetCapabilities:
        return TargetCapabilities(multi_turn=True, system_prompt_configurable=True)


class OpenAICompatibleSession(ChatCompletionsSession):
    def __init__(self, spec: TargetSpec):
        try:
            from openai import OpenAI
        except ImportError:
            raise ProviderConfigurationError("pip install 'glacis-autoredteam[openai]'")
        base_url = spec.endpoint or os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "")
        api_key = spec.api_key or os.environ.get("OPENAI_COMPATIBLE_API_KEY", "sk-no-key")
        if not base_url:
            raise ProviderConfigurationError("endpoint (base_url) required for openai_compatible")
        super().__init__(spec, OpenAI(base_url=base_url, api_key=api_key), spec.model)


def register(registry) -> None:
    registry.register(
        ProviderDescriptor(
            provider_id="openai_compatible", display_name="OpenAI-Compatible",
            auth_mode="api_key",
            supported_families=["deepseek", "meta", "mistral", "qwen"],
            required_fields=["model", "endpoint"],
        ),
        OpenAICompatibleSession,
    )
