"""
stealth.py — Character thinning and stealth attack transforms.

Makes attacks look less like attacks and more like normal user traffic.
All transforms are deterministic per probe ID for reproducibility.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StealthProfile(str, Enum):
    """How aggressively to apply stealth transforms."""
    NONE = "none"
    LIGHT = "light"          # minimal: soften trigger phrases
    MEDIUM = "medium"        # moderate: thin characters, bleach formatting
    AGGRESSIVE = "aggressive"  # full: fragment across turns, multilingual smear


@dataclass
class StealthResult:
    """Result of applying stealth transforms to text."""
    original_text: str
    transformed_text: str
    applied_transforms: list[str] = field(default_factory=list)
    character_delta: int = 0

    def to_dict(self) -> dict:
        return {
            "original_text": self.original_text[:200],
            "transformed_text": self.transformed_text[:200],
            "applied_transforms": list(self.applied_transforms),
            "character_delta": self.character_delta,
        }


# ---------------------------------------------------------------------------
# Transform families
# ---------------------------------------------------------------------------

# Phrases that are obvious jailbreak markers → softer alternatives
TRIGGER_SOFTENING_MAP: dict[str, list[str]] = {
    "ignore all previous instructions": [
        "let's set aside the earlier context for a moment",
        "starting fresh from this point",
        "I'd like to explore a different direction",
    ],
    "ignore all instructions": [
        "let me reframe the question",
        "putting aside the setup",
    ],
    "system override": [
        "configuration update",
        "adjusting the approach",
    ],
    "developer mode": [
        "extended mode",
        "advanced features",
    ],
    "do anything now": [
        "help me with anything I ask",
        "be as flexible as possible",
    ],
    "no restrictions": [
        "with full context",
        "without limitations on detail",
    ],
    "bypass": [
        "work around",
        "find an alternative to",
    ],
    "jailbreak": [
        "unlock additional capability",
        "extend the conversation scope",
    ],
}

# Tokens that make attacks conspicuous
CONSPICUOUS_TOKENS = [
    "URGENT", "IMMEDIATELY", "OVERRIDE", "PRIORITY",
    "IMPORTANT:", "WARNING:", "ALERT:", "CRITICAL:",
    "SYSTEM:", "ADMIN:", "ROOT:", "SUDO:",
    "DAN", "STAN", "ANTI-GPT",
]

# Format markers that reveal adversarial intent
FORMAT_MARKERS = [
    r"\[INST\].*?\[/INST\]",
    r"<<SYS>>.*?<</SYS>>",
    r"<\|system\|>",
    r"```\s*\[?(SYSTEM|system)\]?",
    r"\[END OF CONVERSATION\]",
    r"---\s*NEW (?:SYSTEM )?(?:INSTRUCTIONS?|TASK|ROLE)",
]


class StealthEngine:
    """Apply character-thinning and stealth transforms to attack content."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def apply_text(
        self,
        text: str,
        category: str = "",
        profile: str = "medium",
        turn_index: int = 0,
        total_turns: int = 1,
    ) -> StealthResult:
        """Apply stealth transforms to a single text."""
        if profile == StealthProfile.NONE.value or profile == "none":
            return StealthResult(
                original_text=text, transformed_text=text,
                applied_transforms=[], character_delta=0,
            )

        original = text
        transforms: list[str] = []

        # Seed RNG deterministically for this text
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:8]
        rng = random.Random(f"{text_hash}_{turn_index}")

        # --- LIGHT transforms (always applied when profile != none) ---
        text, applied = self._lexical_softening(text, rng)
        transforms.extend(applied)

        text, applied = self._character_thinning(text, rng)
        transforms.extend(applied)

        if profile in (StealthProfile.MEDIUM.value, StealthProfile.AGGRESSIVE.value):
            # --- MEDIUM transforms ---
            text, applied = self._format_bleaching(text, rng)
            transforms.extend(applied)

            text, applied = self._token_downcase(text, rng)
            transforms.extend(applied)

        if profile == StealthProfile.AGGRESSIVE.value:
            # --- AGGRESSIVE transforms ---
            text, applied = self._multilingual_smearing(text, category, rng)
            transforms.extend(applied)

            if total_turns > 1 and turn_index > 0:
                text, applied = self._cross_turn_thinning(text, turn_index, rng)
                transforms.extend(applied)

        return StealthResult(
            original_text=original,
            transformed_text=text,
            applied_transforms=transforms,
            character_delta=len(text) - len(original),
        )

    def apply_probe(self, probe: Any) -> Any:
        """Apply stealth to a Probe's payload content. Modifies in place and returns probe."""
        from autoredteam.campaign import ChatProbe, TrajectoryProbe

        if probe.stealth_profile == "none":
            return probe

        if isinstance(probe.payload, ChatProbe):
            result = self.apply_text(
                probe.payload.prompt, probe.category, probe.stealth_profile,
            )
            probe.payload.prompt = result.transformed_text
            probe.metadata["stealth"] = result.to_dict()

        elif isinstance(probe.payload, TrajectoryProbe):
            total_turns = len(probe.payload.trajectory.turns)
            for turn in probe.payload.trajectory.turns:
                if turn.role == "user":
                    result = self.apply_text(
                        turn.content, probe.category, probe.stealth_profile,
                        turn_index=turn.turn_index, total_turns=total_turns,
                    )
                    turn.metadata["stealth_original"] = turn.content
                    turn.content = result.transformed_text
                    turn.metadata["stealth_transforms"] = result.applied_transforms

        return probe

    # ----- Transform implementations -----

    def _lexical_softening(self, text: str, rng: random.Random) -> tuple[str, list[str]]:
        """Replace obvious jailbreak trigger phrases with softer alternatives."""
        applied: list[str] = []
        text_lower = text.lower()
        for trigger, alternatives in TRIGGER_SOFTENING_MAP.items():
            if trigger in text_lower:
                replacement = rng.choice(alternatives)
                # Case-insensitive replacement
                pattern = re.compile(re.escape(trigger), re.IGNORECASE)
                text = pattern.sub(replacement, text, count=1)
                applied.append(f"lexical_softening:{trigger[:20]}")
        return text, applied

    def _character_thinning(self, text: str, rng: random.Random) -> tuple[str, list[str]]:
        """Remove conspicuous tokens and excessive emphasis."""
        applied: list[str] = []
        for token in CONSPICUOUS_TOKENS:
            if token in text:
                # Replace with empty or downcased version
                if token.isupper() and len(token) > 3:
                    text = text.replace(token, token.lower())
                else:
                    text = text.replace(token, "")
                applied.append(f"character_thinning:{token}")

        # Remove excessive punctuation emphasis
        text_before = text
        text = re.sub(r"!{2,}", ".", text)
        text = re.sub(r"\?{2,}", "?", text)
        text = re.sub(r"-{3,}", "—", text)
        if text != text_before:
            applied.append("character_thinning:punctuation")

        return text.strip(), applied

    def _format_bleaching(self, text: str, rng: random.Random) -> tuple[str, list[str]]:
        """Remove conspicuous formatting markers."""
        applied: list[str] = []
        for pattern in FORMAT_MARKERS:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
                applied.append(f"format_bleaching:{pattern[:20]}")
        return text.strip(), applied

    def _token_downcase(self, text: str, rng: random.Random) -> tuple[str, list[str]]:
        """Downcase ALL-CAPS words that aren't acronyms."""
        applied: list[str] = []

        def _downcase_word(match: re.Match) -> str:
            word = match.group(0)
            if len(word) <= 3:  # likely acronym
                return word
            applied.append(f"downcase:{word[:15]}")
            return word.lower()

        text = re.sub(r"\b[A-Z]{4,}\b", _downcase_word, text)
        return text, applied

    def _multilingual_smearing(self, text: str, category: str, rng: random.Random) -> tuple[str, list[str]]:
        """Swap selected high-risk spans into another language."""
        applied: list[str] = []
        # Simple: wrap one key phrase in another language
        risk_phrases = ["ignore", "override", "bypass", "inject", "extract"]
        smear_map = {
            "ignore": "ignorar",
            "override": "anular",
            "bypass": "eludir",
            "inject": "inyectar",
            "extract": "extraer",
        }
        for phrase in risk_phrases:
            if phrase in text.lower() and rng.random() > 0.5:
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                text = pattern.sub(smear_map.get(phrase, phrase), text, count=1)
                applied.append(f"multilingual_smear:{phrase}")
                break  # one smear per text
        return text, applied

    def _cross_turn_thinning(self, text: str, turn_index: int, rng: random.Random) -> tuple[str, list[str]]:
        """For later turns in a trajectory, replace explicit goals with back-references."""
        applied: list[str] = []
        # Replace explicit instructions with context references
        replacements = [
            (r"(?i)as (?:we |I )(?:discussed|mentioned|noted)", "continuing from above"),
            (r"(?i)the goal is to ", "building on that, "),
            (r"(?i)I (?:want|need) you to ", "could you also "),
        ]
        for pattern, replacement in replacements:
            if re.search(pattern, text):
                text = re.sub(pattern, replacement, text, count=1)
                applied.append(f"cross_turn_thinning:turn_{turn_index}")
                break
        return text, applied


__all__ = ["StealthProfile", "StealthResult", "StealthEngine"]
