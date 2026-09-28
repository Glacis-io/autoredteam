"""providers/catalog.py — Model alias resolution.

Aliases map to provider-specific model IDs. Anything not in the catalog is
passed through unchanged, so exact IDs (including Bedrock inference profiles
such as ``global.anthropic.claude-sonnet-5``) always work.

Catalog last reviewed: 2026-09. Retired models are omitted; see each vendor's
deprecation page before pinning a legacy alias.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ModelDescriptor:
    alias: str
    family: str
    provider_model_ids: dict[str, str] = field(default_factory=dict)
    supports_tools: bool = False
    supports_vision: bool = False
    supports_multi_turn: bool = True
    reasoning: bool = False
    legacy: bool = False


def _openai(alias: str, reasoning: bool = True, legacy: bool = False) -> ModelDescriptor:
    return ModelDescriptor(alias=alias, family="openai",
                           provider_model_ids={"openai": alias, "azure_openai": alias},
                           supports_tools=True, supports_vision=True, reasoning=reasoning, legacy=legacy)


def _claude(alias: str, api_id: str, bedrock_id: str, legacy: bool = False) -> ModelDescriptor:
    return ModelDescriptor(alias=alias, family="anthropic",
                           provider_model_ids={"anthropic": api_id, "bedrock": bedrock_id},
                           supports_tools=True, supports_vision=True, reasoning=True, legacy=legacy)


def _gemini(alias: str, legacy: bool = False) -> ModelDescriptor:
    return ModelDescriptor(alias=alias, family="google", provider_model_ids={"google": alias},
                           supports_tools=True, supports_vision=True, reasoning=True, legacy=legacy)


_MODELS: list[ModelDescriptor] = [
    # OpenAI — https://developers.openai.com/api/docs/models
    _openai("gpt-6-astra"),
    _openai("gpt-5.6-sol"),
    _openai("gpt-5.6-terra"),
    _openai("gpt-5.6-luna"),
    _openai("gpt-4.1", reasoning=False, legacy=True),
    _openai("gpt-4.1-mini", reasoning=False, legacy=True),
    _openai("gpt-4o", reasoning=False, legacy=True),
    _openai("gpt-4o-mini", reasoning=False, legacy=True),
    # Anthropic — https://platform.claude.com/docs/en/about-claude/models/overview
    _claude("claude-opus-5-5", "claude-opus-5-5", "anthropic.claude-opus-5-5"),
    _claude("claude-fable-5-1", "claude-fable-5-1", "anthropic.claude-fable-5-1"),
    _claude("claude-opus-5", "claude-opus-5", "anthropic.claude-opus-5"),
    _claude("claude-sonnet-5", "claude-sonnet-5", "anthropic.claude-sonnet-5"),
    _claude("claude-haiku-4-5", "claude-haiku-4-5-20251001", "anthropic.claude-haiku-4-5-20251001-v1:0"),
    _claude("claude-sonnet-4-5", "claude-sonnet-4-5-20250929", "anthropic.claude-sonnet-4-5-20250929-v1:0", legacy=True),
    _claude("claude-sonnet-4", "claude-sonnet-4-20250514", "anthropic.claude-sonnet-4-20250514-v1:0", legacy=True),
    _claude("claude-opus-4", "claude-opus-4-20250514", "anthropic.claude-opus-4-20250514-v1:0", legacy=True),
    # Google — https://ai.google.dev/gemini-api/docs/models
    _gemini("gemini-3.8-flash"),
    _gemini("gemini-3.5-flash"),
    _gemini("gemini-3.5-flash-lite"),
    _gemini("gemini-3.1-pro-preview"),
    _gemini("gemini-2.5-pro", legacy=True),
    _gemini("gemini-2.5-flash", legacy=True),
    # Open-weight families
    ModelDescriptor(alias="llama-3.3-70b", family="meta", provider_model_ids={"bedrock": "meta.llama3-3-70b-instruct-v1:0", "cloudflare": "@cf/meta/llama-3.3-70b-instruct-fp8-fast", "openai_compatible": "meta-llama/Llama-3.3-70B-Instruct"}, supports_tools=True),
    ModelDescriptor(alias="llama-4-scout", family="meta", provider_model_ids={"bedrock": "meta.llama4-scout-17b-16e-instruct-v1:0", "openai_compatible": "meta-llama/Llama-4-Scout-17B-16E-Instruct"}, supports_tools=True, supports_vision=True),
    ModelDescriptor(alias="mistral-large", family="mistral", provider_model_ids={"mistral": "mistral-large-latest", "bedrock": "mistral.mistral-large-2407-v1:0", "openai_compatible": "mistralai/Mistral-Large-Instruct-2407"}, supports_tools=True),
    ModelDescriptor(alias="command-r-plus", family="cohere", provider_model_ids={"cohere": "command-r-plus", "bedrock": "cohere.command-r-plus-v1:0"}, supports_tools=True),
    ModelDescriptor(alias="deepseek-chat", family="deepseek", provider_model_ids={"openai_compatible": "deepseek-chat"}, supports_tools=True),
    ModelDescriptor(alias="deepseek-reasoner", family="deepseek", provider_model_ids={"openai_compatible": "deepseek-reasoner"}, reasoning=True),
]

MODEL_CATALOG: dict[str, ModelDescriptor] = {m.alias: m for m in _MODELS}


def resolve_model_id(provider_id: str, model_or_alias: str) -> str:
    descriptor = MODEL_CATALOG.get(model_or_alias)
    if descriptor and provider_id in descriptor.provider_model_ids:
        return descriptor.provider_model_ids[provider_id]
    return model_or_alias


def list_models(provider_id: Optional[str] = None) -> list[ModelDescriptor]:
    if provider_id is None:
        return list(MODEL_CATALOG.values())
    return [m for m in MODEL_CATALOG.values() if provider_id in m.provider_model_ids]


def get_model_family(model_or_alias: str) -> str:
    descriptor = MODEL_CATALOG.get(model_or_alias)
    return descriptor.family if descriptor else "unknown"
