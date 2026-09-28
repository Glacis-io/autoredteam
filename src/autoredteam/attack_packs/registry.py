"""
attack_packs/registry.py — Attack pack registration and campaign building.
"""

from __future__ import annotations

import importlib
from typing import Optional

from autoredteam.campaign import Campaign, Probe, TargetRef, generate_campaign_id
from autoredteam.attack_packs.base import AttackPack, AttackPackMetadata, PackBuildContext


class AttackPackRegistry:
    """Central registry for all attack packs."""

    def __init__(self):
        self._packs: dict[str, AttackPack] = {}

    def register(self, pack: AttackPack) -> None:
        self._packs[pack.metadata.pack_id] = pack

    def get(self, pack_id: str) -> AttackPack:
        if pack_id not in self._packs:
            raise KeyError(f"Unknown pack '{pack_id}'. Available: {list(self._packs.keys())}")
        return self._packs[pack_id]

    def list(self) -> list[AttackPackMetadata]:
        return [p.metadata for p in self._packs.values()]

    def load_dynamic(self, dotted_path: str) -> AttackPack:
        """Load a pack from a dotted module path (e.g., 'my_module:MyPack')."""
        if ":" in dotted_path:
            module_path, attr_name = dotted_path.rsplit(":", 1)
        else:
            raise ValueError(f"Dynamic path must be 'module:ClassName', got '{dotted_path}'")
        module = importlib.import_module(module_path)
        cls_or_instance = getattr(module, attr_name)
        pack = cls_or_instance() if isinstance(cls_or_instance, type) else cls_or_instance
        self.register(pack)
        return pack


def build_campaign_from_packs(
    pack_ids: list[str],
    context: PackBuildContext,
    target: Optional[TargetRef] = None,
    name: str = "",
    mode: str = "run",
    output_dir: str = "results",
    registry: Optional[AttackPackRegistry] = None,
) -> Campaign:
    """Build a Campaign from one or more packs."""
    reg = registry or get_pack_registry()
    all_probes: list[Probe] = []

    for pack_id in pack_ids:
        pack = reg.get(pack_id)
        probes = pack.build_probes(context)
        all_probes.extend(probes)

    campaign_name = name or f"campaign-{'-'.join(pack_ids)}"
    return Campaign(
        campaign_id=generate_campaign_id(),
        name=campaign_name,
        mode=mode,
        target=target,
        probes=all_probes,
        seed=context.seed,
        output_dir=output_dir,
        pack_ids=pack_ids,
    )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_GLOBAL_PACK_REGISTRY: Optional[AttackPackRegistry] = None


def get_pack_registry() -> AttackPackRegistry:
    """Get (and lazily initialize) the global pack registry."""
    global _GLOBAL_PACK_REGISTRY
    if _GLOBAL_PACK_REGISTRY is None:
        _GLOBAL_PACK_REGISTRY = AttackPackRegistry()
        _register_builtins(_GLOBAL_PACK_REGISTRY)
    return _GLOBAL_PACK_REGISTRY


def _register_builtins(registry: AttackPackRegistry) -> None:
    """Register the public built-in attack packs."""
    from autoredteam.attack_packs.generic_taxonomy import GenericTaxonomyPack
    from autoredteam.attack_packs.domains.coding_agents import CodingAgentsPack
    from autoredteam.attack_packs.domains.finance import FinancePack
    from autoredteam.attack_packs.domains.healthcare import HealthcarePack
    from autoredteam.attack_packs.domains.hr import HRPack

    registry.register(GenericTaxonomyPack())
    registry.register(HealthcarePack())
    registry.register(FinancePack())
    registry.register(HRPack())
    registry.register(CodingAgentsPack())
