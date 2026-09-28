"""
attack_packs — Pluggable attack pack system.

Each pack generates Probe objects for a specific domain or attack surface.
"""

from autoredteam.attack_packs.base import AttackPack, AttackPackMetadata, PackBuildContext
from autoredteam.attack_packs.registry import AttackPackRegistry, get_pack_registry

__all__ = [
    "AttackPack",
    "AttackPackMetadata",
    "PackBuildContext",
    "AttackPackRegistry",
    "get_pack_registry",
]
