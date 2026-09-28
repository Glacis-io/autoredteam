"""
attack_packs/base.py — Base types for attack packs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from autoredteam.campaign import Probe, ProbeSurface, TargetRef


@dataclass(frozen=True)
class AttackPackMetadata:
    """Static metadata about an attack pack."""
    pack_id: str
    display_name: str
    description: str
    surfaces: list[ProbeSurface] = field(default_factory=lambda: [ProbeSurface.CHAT])
    categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class PackBuildContext:
    """Context passed to a pack when building probes."""
    target: Optional[TargetRef] = None
    system_prompt: str = ""
    seed: int = 42
    intensity: str = "medium"      # "low" | "medium" | "high"
    max_probes: int = 50
    max_trajectory_turns: int = 5
    stealth_profile: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)


class AttackPack(ABC):
    """Base class for all attack packs."""

    @property
    @abstractmethod
    def metadata(self) -> AttackPackMetadata:
        ...

    @abstractmethod
    def build_probes(self, context: PackBuildContext) -> list[Probe]:
        """Generate probes for this pack given the build context."""
        ...
