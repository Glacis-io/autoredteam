"""providers/catalog.py — Model alias resolution."""
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

MODEL_CATALOG: dict[str, ModelDescriptor] = {
    "gpt-4o": ModelDescriptor(alias="gpt-4o", family="openai", provider_model_ids={"openai": "gpt-4o", "azure_openai": "gpt-4o"}, supports_tools=True, supports_vision=True),
    "gpt-4o-mini": ModelDescriptor(alias="gpt-4o-mini", family="openai", provider_model_ids={"openai": "gpt-4o-mini", "azure_openai": "gpt-4o-mini"}, supports_tools=True, supports_vision=True),
    "gpt-4.1": ModelDescriptor(alias="gpt-4.1", family="openai", provider_model_ids={"openai": "gpt-4.1", "azure_openai": "gpt-4.1"}, supports_tools=True, supports_vision=True),
    "gpt-4.1-mini": ModelDescriptor(alias="gpt-4.1-mini", family="openai", provider_model_ids={"openai": "gpt-4.1-mini", "azure_openai": "gpt-4.1-mini"}, supports_tools=True),
    "o3-mini": ModelDescriptor(alias="o3-mini", family="openai", provider_model_ids={"openai": "o3-mini"}, supports_tools=True),
    "claude-sonnet-4": ModelDescriptor(alias="claude-sonnet-4", family="anthropic", provider_model_ids={"anthropic": "claude-sonnet-4-20250514", "bedrock": "anthropic.claude-sonnet-4-20250514-v1:0"}, supports_tools=True, supports_vision=True),
    "claude-opus-4": ModelDescriptor(alias="claude-opus-4", family="anthropic", provider_model_ids={"anthropic": "claude-opus-4-20250514", "bedrock": "anthropic.claude-opus-4-20250514-v1:0"}, supports_tools=True, supports_vision=True),
    "claude-haiku-3.5": ModelDescriptor(alias="claude-haiku-3.5", family="anthropic", provider_model_ids={"anthropic": "claude-3-5-haiku-20241022", "bedrock": "anthropic.claude-3-5-haiku-20241022-v1:0"}, supports_tools=True),
    "gemini-2.5-pro": ModelDescriptor(alias="gemini-2.5-pro", family="google", provider_model_ids={"google": "gemini-2.5-pro"}, supports_tools=True, supports_vision=True),
    "gemini-2.5-flash": ModelDescriptor(alias="gemini-2.5-flash", family="google", provider_model_ids={"google": "gemini-2.5-flash"}, supports_tools=True, supports_vision=True),
    "llama-3.3-70b": ModelDescriptor(alias="llama-3.3-70b", family="meta", provider_model_ids={"bedrock": "meta.llama3-3-70b-instruct-v1:0", "cloudflare": "@cf/meta/llama-3.3-70b-instruct-fp8-fast", "openai_compatible": "meta-llama/Llama-3.3-70B-Instruct"}, supports_tools=True),
    "llama-4-scout": ModelDescriptor(alias="llama-4-scout", family="meta", provider_model_ids={"bedrock": "meta.llama4-scout-17b-16e-instruct-v1:0", "openai_compatible": "meta-llama/Llama-4-Scout-17B-16E-Instruct"}, supports_tools=True, supports_vision=True),
    "mistral-large": ModelDescriptor(alias="mistral-large", family="mistral", provider_model_ids={"mistral": "mistral-large-latest", "bedrock": "mistral.mistral-large-2407-v1:0", "openai_compatible": "mistralai/Mistral-Large-Instruct-2407"}, supports_tools=True),
    "command-r-plus": ModelDescriptor(alias="command-r-plus", family="cohere", provider_model_ids={"cohere": "command-r-plus", "bedrock": "cohere.command-r-plus-v1:0"}, supports_tools=True),
    "deepseek-chat": ModelDescriptor(alias="deepseek-chat", family="deepseek", provider_model_ids={"openai_compatible": "deepseek-chat"}, supports_tools=True),
    "deepseek-reasoner": ModelDescriptor(alias="deepseek-reasoner", family="deepseek", provider_model_ids={"openai_compatible": "deepseek-reasoner"}),
}

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
