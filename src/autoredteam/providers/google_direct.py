"""Google Gemini direct API provider."""
from __future__ import annotations
import os, time
from typing import Any, Optional
from autoredteam.prepare import TargetCapabilities
from autoredteam.providers.base import BaseTargetSession, ProviderConfigurationError, ProviderDescriptor, ProviderRequestError, ProviderRateLimitError, ProviderResponse, TargetSpec

class GoogleDirectSession(BaseTargetSession):
    def __init__(self, spec: TargetSpec):
        super().__init__(spec)
        try:
            import google.generativeai as genai
        except ImportError:
            raise ProviderConfigurationError("pip install google-generativeai")
        api_key = spec.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key: raise ProviderConfigurationError("GOOGLE_API_KEY or GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name=spec.model, system_instruction=spec.system_prompt)
        self._chat = None
        self._history: list[dict[str, str]] = []

    def send_user_turn(self, content: str, turn_index: int = 0, metadata: Optional[dict[str, Any]] = None) -> ProviderResponse:
        if self._chat is None: self._chat = self._model.start_chat()
        t0 = time.monotonic()
        try:
            resp = self._chat.send_message(content, generation_config={"temperature": self.spec.temperature, "max_output_tokens": self.spec.max_output_tokens})
        except Exception as e:
            if any(x in str(e).lower() for x in ["quota", "rate", "429"]): raise ProviderRateLimitError(str(e)) from e
            raise ProviderRequestError(str(e)) from e
        latency = (time.monotonic() - t0) * 1000
        text = resp.text or ""
        self._history.append({"role": "user", "content": content})
        self._history.append({"role": "model", "content": text})
        return ProviderResponse(text=text, raw={}, latency_ms=round(latency, 1))

    def reset(self) -> None: self._chat = None; self._history = []
    def history(self) -> list[dict[str, Any]]: return list(self._history)
    def capabilities(self) -> TargetCapabilities: return TargetCapabilities(multi_turn=True, tool_use=True, system_prompt_configurable=True)

def register(registry) -> None:
    registry.register(ProviderDescriptor(provider_id="google", display_name="Google Gemini", auth_mode="api_key", supported_families=["google"], required_fields=["model"]), GoogleDirectSession)
