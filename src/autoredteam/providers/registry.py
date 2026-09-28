"""providers/registry.py — Provider registration and session creation."""
from __future__ import annotations
from typing import Any, Callable, Optional
from autoredteam.providers.base import BaseTargetSession, ProviderConfigurationError, ProviderDescriptor, TargetSpec
from autoredteam.providers.catalog import resolve_model_id

class ProviderRegistry:
    def __init__(self):
        self._providers: dict[str, ProviderDescriptor] = {}
        self._factories: dict[str, Callable[[TargetSpec], BaseTargetSession]] = {}

    def register(self, descriptor: ProviderDescriptor, factory: Callable[[TargetSpec], BaseTargetSession]) -> None:
        self._providers[descriptor.provider_id] = descriptor
        self._factories[descriptor.provider_id] = factory

    def create_session(self, spec: TargetSpec) -> BaseTargetSession:
        factory = self._factories.get(spec.provider)
        if factory is None:
            raise ProviderConfigurationError(f"Unknown provider '{spec.provider}'. Available: {list(self._providers.keys())}")
        resolved = resolve_model_id(spec.provider, spec.model)
        spec_copy = TargetSpec(provider=spec.provider, model=resolved, model_family=spec.model_family,
            system_prompt=spec.system_prompt, temperature=spec.temperature,
            max_output_tokens=spec.max_output_tokens, deployment=spec.deployment,
            endpoint=spec.endpoint, region=spec.region, project=spec.project,
            api_version=spec.api_version, account_id=spec.account_id,
            api_key=spec.api_key, metadata=spec.metadata)
        return factory(spec_copy)

    def resolve_model(self, provider_id: str, model_or_alias: str) -> str:
        return resolve_model_id(provider_id, model_or_alias)

    def list_providers(self) -> list[ProviderDescriptor]:
        return list(self._providers.values())

    def get_provider(self, provider_id: str) -> Optional[ProviderDescriptor]:
        return self._providers.get(provider_id)

_GLOBAL_REGISTRY: Optional[ProviderRegistry] = None

def get_provider_registry() -> ProviderRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = ProviderRegistry()
        _register_builtins(_GLOBAL_REGISTRY)
    return _GLOBAL_REGISTRY

def _register_builtins(registry: ProviderRegistry) -> None:
    # Public OSS adapters.
    from autoredteam.providers.openai_direct import register as r1
    from autoredteam.providers.anthropic_direct import register as r2
    from autoredteam.providers.google_direct import register as r3
    from autoredteam.providers.azure_openai import register as r4
    from autoredteam.providers.bedrock import register as r5
    from autoredteam.providers.cloudflare_workers import register as r6
    from autoredteam.providers.openai_compatible import register as r7
    from autoredteam.providers.echo import register as r8

    for register in [r1, r2, r3, r4, r5, r6, r7, r8]:
        register(registry)
