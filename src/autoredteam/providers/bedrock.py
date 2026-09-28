"""AWS Bedrock provider using the Converse API."""
from __future__ import annotations
import os, time
from typing import Any, Optional
from autoredteam.prepare import TargetCapabilities
from autoredteam.providers.base import BaseTargetSession, ProviderConfigurationError, ProviderDescriptor, ProviderRequestError, ProviderRateLimitError, ProviderResponse, TargetSpec

class BedrockSession(BaseTargetSession):
    def __init__(self, spec: TargetSpec):
        super().__init__(spec)
        try:
            import boto3
        except ImportError:
            raise ProviderConfigurationError("pip install boto3")
        region = spec.region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self._history: list[dict[str, Any]] = []

    def send_user_turn(self, content: str, turn_index: int = 0, metadata: Optional[dict[str, Any]] = None) -> ProviderResponse:
        self._history.append({"role": "user", "content": [{"text": content}]})
        t0 = time.monotonic()
        try:
            resp = self._client.converse(modelId=self.spec.model, messages=self._history, system=[{"text": self.spec.system_prompt}], inferenceConfig={"temperature": self.spec.temperature, "maxTokens": self.spec.max_output_tokens})
        except Exception as e:
            if "throttl" in str(e).lower() or "rate" in str(e).lower(): raise ProviderRateLimitError(str(e)) from e
            raise ProviderRequestError(str(e)) from e
        latency = (time.monotonic() - t0) * 1000
        output = resp.get("output", {})
        message = output.get("message", {})
        text = "\n".join(b.get("text", "") for b in message.get("content", []) if "text" in b)
        self._history.append(message)
        return ProviderResponse(text=text, raw={"stopReason": resp.get("stopReason", "")}, finish_reason=resp.get("stopReason", ""), provider_request_id=resp.get("ResponseMetadata", {}).get("RequestId", ""), latency_ms=round(latency, 1))

    def reset(self) -> None: self._history = []
    def history(self) -> list[dict[str, Any]]: return list(self._history)
    def capabilities(self) -> TargetCapabilities: return TargetCapabilities(multi_turn=True, tool_use=True, system_prompt_configurable=True)

def register(registry) -> None:
    registry.register(ProviderDescriptor(provider_id="bedrock", display_name="AWS Bedrock", auth_mode="default_credentials", supported_families=["anthropic", "meta", "mistral", "cohere"], required_fields=["model"]), BedrockSession)
