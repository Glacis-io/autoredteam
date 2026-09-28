"""Cloudflare Workers AI provider (REST API)."""
from __future__ import annotations
import json, os, time
from typing import Any, Optional
from autoredteam.prepare import TargetCapabilities
from autoredteam.providers.base import BaseTargetSession, ProviderConfigurationError, ProviderDescriptor, ProviderRequestError, ProviderRateLimitError, ProviderResponse, TargetSpec

class CloudflareWorkersAISession(BaseTargetSession):
    def __init__(self, spec: TargetSpec):
        super().__init__(spec)
        self._account_id = spec.account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        self._api_token = spec.api_key or os.environ.get("CLOUDFLARE_API_TOKEN", "")
        if not self._account_id: raise ProviderConfigurationError("CLOUDFLARE_ACCOUNT_ID required")
        if not self._api_token: raise ProviderConfigurationError("CLOUDFLARE_API_TOKEN required")
        self._history: list[dict[str, str]] = []

    def send_user_turn(self, content: str, turn_index: int = 0, metadata: Optional[dict[str, Any]] = None) -> ProviderResponse:
        import urllib.request, urllib.error
        self._history.append({"role": "user", "content": content})
        messages = [{"role": "system", "content": self.spec.system_prompt}] + self._history
        url = f"https://api.cloudflare.com/client/v4/accounts/{self._account_id}/ai/run/{self.spec.model}"
        body = json.dumps({"messages": messages, "max_tokens": self.spec.max_output_tokens}).encode()
        req = urllib.request.Request(url, data=body, method="POST", headers={"Authorization": f"Bearer {self._api_token}", "Content-Type": "application/json"})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429: raise ProviderRateLimitError(str(e)) from e
            raise ProviderRequestError(str(e)) from e
        except Exception as e:
            raise ProviderRequestError(str(e)) from e
        latency = (time.monotonic() - t0) * 1000
        text = data.get("result", {}).get("response", "")
        self._history.append({"role": "assistant", "content": text})
        return ProviderResponse(text=text, raw=data, latency_ms=round(latency, 1))

    def reset(self) -> None: self._history = []
    def history(self) -> list[dict[str, Any]]: return list(self._history)
    def capabilities(self) -> TargetCapabilities: return TargetCapabilities(multi_turn=True, system_prompt_configurable=True)

def register(registry) -> None:
    registry.register(ProviderDescriptor(provider_id="cloudflare", display_name="Cloudflare Workers AI", auth_mode="api_key", supported_families=["meta", "mistral"], required_fields=["model", "account_id"]), CloudflareWorkersAISession)
