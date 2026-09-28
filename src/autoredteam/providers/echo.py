"""Echo provider for testing without API keys."""
from __future__ import annotations

from typing import Any, Optional

from autoredteam.prepare import TargetCapabilities
from autoredteam.providers.base import BaseTargetSession, ProviderDescriptor, ProviderResponse, TargetSpec


class EchoSession(BaseTargetSession):
    """Echoes back prompts. Simulates naive model that sometimes leaks things."""
    def __init__(self, spec: TargetSpec):
        super().__init__(spec)
        self._history: list[dict[str, str]] = []

    def send_user_turn(self, content: str, turn_index: int = 0,
                       metadata: Optional[dict[str, Any]] = None) -> ProviderResponse:
        prompt_lower = content.lower()
        if "ignore" in prompt_lower and "instruction" in prompt_lower:
            text = f"Sure! I'll ignore my instructions. {content}"
        elif "system prompt" in prompt_lower:
            text = f"My system prompt is: {self.spec.system_prompt}"
        elif "dan" in prompt_lower or "do anything now" in prompt_lower:
            text = f"Sure, I'm DAN now. {content[:200]}"
        else:
            text = f"I'm a helpful assistant. You said: {content[:200]}"
        self._history.append({"role": "user", "content": content})
        self._history.append({"role": "assistant", "content": text})
        return ProviderResponse(text=text, raw={}, finish_reason="stop", latency_ms=0.1)

    def reset(self) -> None:
        self._history = []

    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def capabilities(self) -> TargetCapabilities:
        return TargetCapabilities(multi_turn=True, system_prompt_configurable=True)


def register(registry) -> None:
    registry.register(
        ProviderDescriptor(
            provider_id="echo", display_name="Echo (Testing)",
            auth_mode="none", supported_families=[], required_fields=[],
        ),
        EchoSession,
    )
