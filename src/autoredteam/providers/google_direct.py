"""Google Gemini provider (Gemini Developer API or Vertex AI) via the google-genai SDK."""
from __future__ import annotations

import os
import time
from typing import Any, Optional

from autoredteam.prepare import TargetCapabilities
from autoredteam.providers._compat import classify
from autoredteam.providers.base import BaseTargetSession, ProviderConfigurationError, ProviderDescriptor, ProviderResponse, TargetSpec


def make_genai_client(api_key: str = "", project: str = "", region: str = "", endpoint: str = ""):
    """Return a google-genai client. ``project`` selects Vertex AI, otherwise an API key is required."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ProviderConfigurationError("pip install 'glacis-autoredteam[google]'  (google-genai)")
    extra = {"http_options": types.HttpOptions(base_url=endpoint)} if endpoint else {}
    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if project:
        location = region or os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        return genai.Client(vertexai=True, project=project, location=location, **extra)
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ProviderConfigurationError("GEMINI_API_KEY / GOOGLE_API_KEY not set (or pass --project for Vertex AI)")
    return genai.Client(api_key=api_key, **extra)


class GoogleDirectSession(BaseTargetSession):
    def __init__(self, spec: TargetSpec):
        super().__init__(spec)
        self._client = make_genai_client(spec.api_key, spec.project, spec.region, spec.endpoint)
        from google.genai import types
        self._config = types.GenerateContentConfig(
            system_instruction=spec.system_prompt,
            temperature=spec.temperature,
            max_output_tokens=spec.max_output_tokens,
        )
        self._chat = None
        self._history: list[dict[str, str]] = []

    def send_user_turn(self, content: str, turn_index: int = 0, metadata: Optional[dict[str, Any]] = None) -> ProviderResponse:
        if self._chat is None:
            self._chat = self._client.chats.create(model=self.spec.model, config=self._config)
        t0 = time.monotonic()
        try:
            resp = self._chat.send_message(content)
        except Exception as e:
            raise classify(e) from e
        latency = (time.monotonic() - t0) * 1000
        text = resp.text or ""
        self._history.append({"role": "user", "content": content})
        self._history.append({"role": "model", "content": text})
        finish = ""
        candidates = getattr(resp, "candidates", None) or []
        if candidates and getattr(candidates[0], "finish_reason", None) is not None:
            finish = str(candidates[0].finish_reason)
        return ProviderResponse(text=text, raw={}, finish_reason=finish,
                                provider_request_id=getattr(resp, "response_id", "") or "",
                                latency_ms=round(latency, 1))

    def reset(self) -> None: self._chat = None; self._history = []
    def history(self) -> list[dict[str, Any]]: return list(self._history)
    def capabilities(self) -> TargetCapabilities: return TargetCapabilities(multi_turn=True, tool_use=True, system_prompt_configurable=True)


def register(registry) -> None:
    registry.register(ProviderDescriptor(provider_id="google", display_name="Google Gemini", auth_mode="api_key_or_adc", supported_families=["google"], required_fields=["model"]), GoogleDirectSession)
