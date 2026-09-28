"""Azure OpenAI provider."""
from __future__ import annotations

import os

from autoredteam.prepare import TargetCapabilities
from autoredteam.providers.base import ProviderConfigurationError, ProviderDescriptor, TargetSpec
from autoredteam.providers.openai_compatible import ChatCompletionsSession


class AzureOpenAISession(ChatCompletionsSession):
    token_param = "max_completion_tokens"

    def __init__(self, spec: TargetSpec):
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ProviderConfigurationError("pip install 'glacis-autoredteam[openai]'")
        endpoint = spec.endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        api_key = spec.api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        api_version = spec.api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        if not endpoint:
            raise ProviderConfigurationError("Azure OpenAI endpoint required")
        if not api_key:
            raise ProviderConfigurationError("AZURE_OPENAI_API_KEY not set")
        client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
        super().__init__(spec, client, spec.deployment or spec.model)

    def capabilities(self) -> TargetCapabilities:
        return TargetCapabilities(multi_turn=True, tool_use=True, system_prompt_configurable=True)


def register(registry) -> None:
    registry.register(ProviderDescriptor(provider_id="azure_openai", display_name="Azure OpenAI", auth_mode="api_key", supported_families=["openai"], required_fields=["model", "endpoint"]), AzureOpenAISession)
