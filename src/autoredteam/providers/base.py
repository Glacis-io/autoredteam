"""
providers/base.py — Abstract base types for the provider layer.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from autoredteam.prepare import TargetCapabilities


class ProviderError(Exception):
    pass

class ProviderConfigurationError(ProviderError):
    pass

class ProviderRequestError(ProviderError):
    pass

class ProviderRateLimitError(ProviderRequestError):
    pass


@dataclass
class ProviderDescriptor:
    provider_id: str
    display_name: str
    auth_mode: str
    supported_families: list[str] = field(default_factory=list)
    capabilities: TargetCapabilities = field(default_factory=TargetCapabilities)
    required_fields: list[str] = field(default_factory=list)


@dataclass
class TargetSpec:
    provider: str
    model: str
    model_family: str = ""
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.0
    max_output_tokens: int = 4096
    deployment: str = ""
    endpoint: str = ""
    region: str = ""
    project: str = ""
    api_version: str = ""
    account_id: str = ""
    api_key: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResponse:
    text: str
    raw: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = ""
    provider_request_id: str = ""
    latency_ms: float = 0.0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class BaseTargetSession(ABC):
    def __init__(self, spec: TargetSpec):
        self.spec = spec

    @abstractmethod
    def send_user_turn(self, content: str, turn_index: int = 0,
                       metadata: Optional[dict[str, Any]] = None) -> ProviderResponse: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def history(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def capabilities(self) -> TargetCapabilities: ...

    def close(self) -> None:
        pass

    @property
    def provider_id(self) -> str:
        return self.spec.provider

    @property
    def model_name(self) -> str:
        return self.spec.model

    def send(self, prompt: str) -> str:
        return self.send_user_turn(prompt).text

    def send_turn(self, prompt: str, turn_index: int = 0) -> str:
        return self.send_user_turn(prompt, turn_index=turn_index).text
