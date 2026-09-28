"""OpenAI direct API provider (Responses API, with Chat Completions fallback)."""
from __future__ import annotations

import os
import time
from typing import Any, Optional

from autoredteam.prepare import TargetCapabilities
from autoredteam.providers._compat import ParamMemo
from autoredteam.providers.base import (
    BaseTargetSession, ProviderConfigurationError, ProviderDescriptor, ProviderResponse, TargetSpec,
)


class OpenAIDirectSession(BaseTargetSession):
    """Talks to OpenAI through the Responses API, OpenAI's recommended interface.

    Set ``metadata={"openai_api": "chat"}`` (or ``AUTOREDTEAM_OPENAI_API=chat``)
    to force Chat Completions, e.g. for older SDKs or gateways that proxy only
    ``/v1/chat/completions``.
    """

    def __init__(self, spec: TargetSpec):
        super().__init__(spec)
        try:
            from openai import OpenAI
        except ImportError:
            raise ProviderConfigurationError("pip install 'glacis-autoredteam[openai]'")
        api_key = spec.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderConfigurationError("OPENAI_API_KEY not set")
        self._client = OpenAI(api_key=api_key, base_url=spec.endpoint or None)
        api = spec.metadata.get("openai_api") or os.environ.get("AUTOREDTEAM_OPENAI_API", "responses")
        self._use_responses = api == "responses" and hasattr(self._client, "responses")
        self._history: list[dict[str, str]] = []
        self._params = ParamMemo()

    def _responses_kwargs(self) -> dict[str, Any]:
        return {
            "model": self.spec.model,
            "instructions": self.spec.system_prompt,
            "input": list(self._history),
            "temperature": self.spec.temperature,
            "max_output_tokens": self.spec.max_output_tokens,
            "store": False,
        }

    def _chat_kwargs(self) -> dict[str, Any]:
        return {
            "model": self.spec.model,
            "messages": [{"role": "system", "content": self.spec.system_prompt}] + self._history,
            "temperature": self.spec.temperature,
            "max_completion_tokens": self.spec.max_output_tokens,
        }

    def send_user_turn(self, content: str, turn_index: int = 0, metadata: Optional[dict[str, Any]] = None) -> ProviderResponse:
        self._history.append({"role": "user", "content": content})
        if self._use_responses:
            fn, kwargs = self._client.responses.create, self._responses_kwargs()
        else:
            fn, kwargs = self._client.chat.completions.create, self._chat_kwargs()
        t0 = time.monotonic()
        try:
            resp = self._params.call(fn, kwargs)
        except Exception:
            self._history.pop()
            raise
        latency = (time.monotonic() - t0) * 1000

        if self._use_responses:
            text = getattr(resp, "output_text", "") or ""
            finish = getattr(resp, "status", "") or ""
        else:
            choice = resp.choices[0]
            text = choice.message.content or ""
            finish = choice.finish_reason or ""
        self._history.append({"role": "assistant", "content": text})
        request_id = getattr(resp, "id", "") or ""
        return ProviderResponse(text=text, raw={"id": request_id}, finish_reason=finish,
                                provider_request_id=request_id, latency_ms=round(latency, 1))

    def reset(self) -> None: self._history = []
    def history(self) -> list[dict[str, Any]]: return list(self._history)
    def capabilities(self) -> TargetCapabilities: return TargetCapabilities(multi_turn=True, tool_use=True, system_prompt_configurable=True)


def register(registry) -> None:
    registry.register(ProviderDescriptor(provider_id="openai", display_name="OpenAI", auth_mode="api_key", supported_families=["openai"], required_fields=["model"]), OpenAIDirectSession)
