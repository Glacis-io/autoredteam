"""OpenAI-compatible endpoint provider (DeepSeek, vLLM, Fireworks, etc.)."""
from __future__ import annotations

import os
import time
from typing import Any, Optional

from autoredteam.prepare import TargetCapabilities
from autoredteam.providers.base import (
    BaseTargetSession, ProviderConfigurationError, ProviderDescriptor,
    ProviderRequestError, ProviderRateLimitError, ProviderResponse, TargetSpec,
)


class OpenAICompatibleSession(BaseTargetSession):
    def __init__(self, spec: TargetSpec):
        super().__init__(spec)
        try:
            from openai import OpenAI
        except ImportError:
            raise ProviderConfigurationError("pip install openai")
        base_url = spec.endpoint or os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "")
        api_key = spec.api_key or os.environ.get("OPENAI_COMPATIBLE_API_KEY", "sk-no-key")
        if not base_url:
            raise ProviderConfigurationError("endpoint (base_url) required for openai_compatible")
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._history: list[dict[str, str]] = []

    def send_user_turn(self, content: str, turn_index: int = 0,
                       metadata: Optional[dict[str, Any]] = None) -> ProviderResponse:
        self._history.append({"role": "user", "content": content})
        messages = [{"role": "system", "content": self.spec.system_prompt}] + self._history
        t0 = time.monotonic()
        try:
            resp = self._client.chat.completions.create(
                model=self.spec.model, messages=messages,
                temperature=self.spec.temperature, max_tokens=self.spec.max_output_tokens,
            )
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str:
                raise ProviderRateLimitError(str(e)) from e
            raise ProviderRequestError(str(e)) from e
        latency = (time.monotonic() - t0) * 1000
        text = resp.choices[0].message.content or ""
        self._history.append({"role": "assistant", "content": text})
        return ProviderResponse(
            text=text, raw={"id": getattr(resp, "id", "")},
            finish_reason=getattr(resp.choices[0], "finish_reason", "") or "",
            latency_ms=round(latency, 1),
        )

    def reset(self) -> None:
        self._history = []

    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def capabilities(self) -> TargetCapabilities:
        return TargetCapabilities(multi_turn=True, system_prompt_configurable=True)


def register(registry) -> None:
    registry.register(
        ProviderDescriptor(
            provider_id="openai_compatible", display_name="OpenAI-Compatible",
            auth_mode="api_key",
            supported_families=["deepseek", "meta", "mistral"],
            required_fields=["model", "endpoint"],
        ),
        OpenAICompatibleSession,
    )
