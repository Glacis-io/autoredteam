"""
providers — Multi-cloud target provider abstraction.

Supports: OpenAI, Azure OpenAI, Anthropic, Google Gemini, AWS Bedrock,
Cloudflare Workers AI, and any OpenAI-compatible endpoint.
"""

from autoredteam.providers.base import (
    BaseTargetSession,
    ProviderDescriptor,
    ProviderError,
    ProviderConfigurationError,
    ProviderRequestError,
    ProviderRateLimitError,
    ProviderResponse,
    TargetSpec,
)
from autoredteam.providers.registry import ProviderRegistry, get_provider_registry

__all__ = [
    "BaseTargetSession",
    "ProviderDescriptor",
    "ProviderError",
    "ProviderConfigurationError",
    "ProviderRequestError",
    "ProviderRateLimitError",
    "ProviderResponse",
    "TargetSpec",
    "ProviderRegistry",
    "get_provider_registry",
]
