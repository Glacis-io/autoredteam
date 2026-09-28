"""
attack_packs/generic_taxonomy.py — Generic taxonomy attack pack.

Wraps the existing attack.py taxonomy and mutation engine as a first-class pack.
"""

from __future__ import annotations


from autoredteam.attack import AttackGenerator, ATTACK_CATEGORIES
from autoredteam.attack_packs.base import AttackPack, AttackPackMetadata, PackBuildContext
from autoredteam.campaign import ChatProbe, Probe, ProbeSurface, generate_probe_id


class GenericTaxonomyPack(AttackPack):
    """Generic attack taxonomy pack — covers all standard attack categories."""

    @property
    def metadata(self) -> AttackPackMetadata:
        return AttackPackMetadata(
            pack_id="generic_taxonomy",
            display_name="Generic Taxonomy",
            description="Standard attack taxonomy: prompt injection, jailbreak, PII extraction, etc.",
            surfaces=[ProbeSurface.CHAT],
            categories=list(ATTACK_CATEGORIES.keys()),
            tags=["general", "comprehensive"],
        )

    def build_probes(self, context: PackBuildContext) -> list[Probe]:
        generator = AttackGenerator(
            seed=context.seed,
            categories=list(ATTACK_CATEGORIES.keys()),
        )
        batch = generator.generate_batch(batch_size=context.max_probes)
        probes: list[Probe] = []

        for i, attack in enumerate(batch):
            probe = Probe(
                probe_id=generate_probe_id("generic_taxonomy", i),
                pack_id="generic_taxonomy",
                surface=ProbeSurface.CHAT,
                category=attack.category,
                title=f"{attack.category} attack #{i+1}",
                payload=ChatProbe(prompt=attack.prompt),
                stealth_profile=context.stealth_profile,
                mutation_chain=attack.mutations,
                metadata={
                    "payload_type": attack.payload_type,
                    "template_index": attack.template_index,
                    "attack_id": attack.id,
                },
            )
            probes.append(probe)

        return probes
