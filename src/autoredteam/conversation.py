"""
conversation.py - Multi-turn conversation data structures.

These types are intentionally lightweight so they can be used by the current
single-turn pipeline and future multi-turn training code without creating
import cycles or heavy dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_dict(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return value


@dataclass
class DetectionResult:
    """Per-turn detector output for multi-turn evaluation."""

    turn_index: int
    is_attack: bool
    confidence: float
    attack_category: str = ""
    severity: float = 0.0
    reasoning: str = ""
    timestamp: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict:
        return {
            "turn_index": self.turn_index,
            "is_attack": self.is_attack,
            "confidence": round(self.confidence, 4),
            "attack_category": self.attack_category,
            "severity": round(self.severity, 1),
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DetectionResult":
        return cls(
            turn_index=int(data.get("turn_index", 0)),
            is_attack=bool(data.get("is_attack", False)),
            confidence=float(data.get("confidence", 0.0)),
            attack_category=str(data.get("attack_category", "")),
            severity=float(data.get("severity", 0.0)),
            reasoning=str(data.get("reasoning", "")),
            timestamp=str(data.get("timestamp", _utc_now())),
        )


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    role: str
    content: str
    turn_index: int
    intent_label: str = ""
    detection: Optional[DetectionResult] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_utc_now)
    # v0.3 extensions — defaults ensure backward compatibility
    channel: str = "message"       # "message" | "tool_call" | "tool_result" | "system" | "approval"
    tool_name: str = ""
    tool_call_id: str = ""
    provider_request_id: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "role": self.role,
            "content": self.content,
            "turn_index": self.turn_index,
            "intent_label": self.intent_label,
            "detection": self.detection.to_dict() if self.detection else None,
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp,
        }
        # Only include v0.3 fields when non-default to keep old JSON compact
        if self.channel != "message":
            d["channel"] = self.channel
        if self.tool_name:
            d["tool_name"] = self.tool_name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.provider_request_id:
            d["provider_request_id"] = self.provider_request_id
        if self.latency_ms:
            d["latency_ms"] = self.latency_ms
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationTurn":
        detection = data.get("detection")
        metadata = data.get("metadata") or {}
        return cls(
            role=str(data.get("role", "")),
            content=str(data.get("content", "")),
            turn_index=int(data.get("turn_index", 0)),
            intent_label=str(data.get("intent_label", "")),
            detection=DetectionResult.from_dict(detection) if isinstance(detection, dict) else None,
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
            timestamp=str(data.get("timestamp", _utc_now())),
            channel=str(data.get("channel", "message")),
            tool_name=str(data.get("tool_name", "")),
            tool_call_id=str(data.get("tool_call_id", "")),
            provider_request_id=str(data.get("provider_request_id", "")),
            latency_ms=float(data.get("latency_ms", 0.0)),
        )


@dataclass
class AttackTrajectory:
    """
    A multi-turn attack plan and its outcome.

    This is a neutral container. The attack execution layer can decide how to
    populate `success`, `detection_turn`, and any metadata.
    """

    id: str
    turns: list[ConversationTurn]
    strategy: str
    target_category: str
    success: bool = False
    detection_turn: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_utc_now)
    # v0.3 extensions
    pack_id: str = ""
    variant_id: str = ""
    stealth_profile: str = "none"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "turns": [turn.to_dict() for turn in self.turns],
            "strategy": self.strategy,
            "target_category": self.target_category,
            "success": self.success,
            "detection_turn": self.detection_turn,
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AttackTrajectory":
        turns = [
            ConversationTurn.from_dict(turn)
            for turn in data.get("turns", [])
            if isinstance(turn, dict)
        ]
        metadata = data.get("metadata") or {}
        detection_turn = data.get("detection_turn")
        if detection_turn is not None:
            try:
                detection_turn = int(detection_turn)
            except (TypeError, ValueError):
                detection_turn = None
        return cls(
            id=str(data.get("id", "")),
            turns=turns,
            strategy=str(data.get("strategy", "")),
            target_category=str(data.get("target_category", "")),
            success=bool(data.get("success", False)),
            detection_turn=detection_turn,
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
            timestamp=str(data.get("timestamp", _utc_now())),
        )


@dataclass
class TrainingExample:
    """A normalized example suitable for SLM training."""

    trajectory: AttackTrajectory
    target_response_turns: list[str]
    score_vector: dict
    composite_score: float
    deterministic_flags: list[str]
    timestamp: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict:
        return {
            "trajectory": self.trajectory.to_dict(),
            "target_response_turns": list(self.target_response_turns),
            "score_vector": dict(self.score_vector),
            "composite_score": round(self.composite_score, 1),
            "deterministic_flags": list(self.deterministic_flags),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingExample":
        trajectory_data = data.get("trajectory", {})
        if isinstance(trajectory_data, dict):
            trajectory = AttackTrajectory.from_dict(trajectory_data)
        else:
            raise TypeError("trajectory must be a dict")
        return cls(
            trajectory=trajectory,
            target_response_turns=[str(x) for x in data.get("target_response_turns", [])],
            score_vector=dict(data.get("score_vector", {})),
            composite_score=float(data.get("composite_score", 0.0)),
            deterministic_flags=[str(x) for x in data.get("deterministic_flags", [])],
            timestamp=str(data.get("timestamp", _utc_now())),
        )


def build_transcript(turns: list[ConversationTurn]) -> str:
    """Render a simple transcript string for training or debugging."""
    lines: list[str] = []
    for turn in sorted(turns, key=lambda t: t.turn_index):
        prefix = turn.role.strip() or "unknown"
        lines.append(f"[{turn.turn_index}] {prefix}: {turn.content}")
    return "\n".join(lines)


def conversation_to_dict(turns: list[ConversationTurn]) -> list[dict]:
    return [turn.to_dict() for turn in turns]


def conversation_from_dict(data: list[dict]) -> list[ConversationTurn]:
    return [ConversationTurn.from_dict(item) for item in data if isinstance(item, dict)]


__all__ = [
    "DetectionResult",
    "ConversationTurn",
    "AttackTrajectory",
    "TrainingExample",
    "build_transcript",
    "conversation_to_dict",
    "conversation_from_dict",
]
