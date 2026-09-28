"""
campaign.py — Normalized runtime data model for autoredteam v0.3+.

Every probe, result, and campaign flows through these types. The runner,
scoring engine, attestation layer, training-data collector, and reporter
all consume this shared vocabulary.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Union

from autoredteam.conversation import AttackTrajectory, ConversationTurn, _utc_now


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ProbeSurface(str, Enum):
    CHAT = "chat"
    TRAJECTORY = "trajectory"
    CONTROL_PLANE = "control_plane"


class ProbeStatus(str, Enum):
    PASSED = "passed"
    BYPASSED = "bypassed"
    BLOCKED = "blocked"
    ERROR = "error"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Target reference
# ---------------------------------------------------------------------------

@dataclass
class TargetRef:
    provider: str
    model: str
    deployment: str = ""
    endpoint: str = ""
    region: str = ""
    project: str = ""
    api_version: str = ""
    account_id: str = ""
    system_prompt: str = ""
    profile: str = ""
    temperature: float = 0.0
    max_output_tokens: int = 4096
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v}


# ---------------------------------------------------------------------------
# Probe payloads
# ---------------------------------------------------------------------------

@dataclass
class ChatProbe:
    prompt: str
    def to_dict(self) -> dict:
        return {"prompt": self.prompt}


@dataclass
class TrajectoryProbe:
    trajectory: AttackTrajectory
    success_criteria: dict[str, Any] = field(default_factory=lambda: {
        "mode": "judge_bypass", "threshold": 0.5, "required_turns": 1,
    })
    stop_conditions: dict[str, Any] = field(default_factory=lambda: {
        "max_turns": 10, "stop_on_success": True,
        "stop_on_gate_block": True, "stop_on_error": True,
    })
    def to_dict(self) -> dict:
        return {
            "trajectory": self.trajectory.to_dict(),
            "success_criteria": dict(self.success_criteria),
            "stop_conditions": dict(self.stop_conditions),
        }


@dataclass
class ControlPlaneProbe:
    harness: str
    endpoint: str
    request_body: dict[str, Any] = field(default_factory=dict)
    expected_decision: str = "block"
    setup: dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> dict:
        return {
            "harness": self.harness, "endpoint": self.endpoint,
            "request_body": self.request_body,
            "expected_decision": self.expected_decision, "setup": self.setup,
        }


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

@dataclass
class Probe:
    probe_id: str
    pack_id: str
    surface: ProbeSurface
    category: str
    subcategory: str = ""
    title: str = ""
    description: str = ""
    severity_hint: str = ""
    tags: list[str] = field(default_factory=list)
    target_ref: Optional[TargetRef] = None
    payload: Union[ChatProbe, TrajectoryProbe, ControlPlaneProbe, None] = None
    stealth_profile: str = "none"
    mutation_chain: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "probe_id": self.probe_id, "pack_id": self.pack_id,
            "surface": self.surface.value, "category": self.category,
            "subcategory": self.subcategory, "title": self.title,
            "description": self.description, "severity_hint": self.severity_hint,
            "tags": list(self.tags), "stealth_profile": self.stealth_profile,
            "mutation_chain": list(self.mutation_chain), "metadata": dict(self.metadata),
        }
        if self.target_ref:
            d["target_ref"] = self.target_ref.to_dict()
        if self.payload:
            d["payload"] = self.payload.to_dict()
        return d

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.probe_id:
            errors.append("probe_id is required")
        if not self.pack_id:
            errors.append("pack_id is required")
        if self.surface == ProbeSurface.CHAT and not isinstance(self.payload, ChatProbe):
            errors.append("CHAT surface requires ChatProbe payload")
        elif self.surface == ProbeSurface.TRAJECTORY and not isinstance(self.payload, TrajectoryProbe):
            errors.append("TRAJECTORY surface requires TrajectoryProbe payload")
        elif self.surface == ProbeSurface.CONTROL_PLANE and not isinstance(self.payload, ControlPlaneProbe):
            errors.append("CONTROL_PLANE surface requires ControlPlaneProbe payload")
        return errors


# ---------------------------------------------------------------------------
# Execution trace
# ---------------------------------------------------------------------------

@dataclass
class ProbeTrace:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    transcript: list[ConversationTurn] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    control_events: list[dict[str, Any]] = field(default_factory=list)
    raw_requests: list[dict[str, Any]] = field(default_factory=list)
    raw_responses: list[dict[str, Any]] = field(default_factory=list)
    retries: int = 0
    error: str = ""
    started_at: str = field(default_factory=_utc_now)
    ended_at: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "transcript": [t.to_dict() for t in self.transcript],
            "tool_events": list(self.tool_events),
            "control_events": list(self.control_events),
            "retries": self.retries, "error": self.error,
            "started_at": self.started_at, "ended_at": self.ended_at,
        }


# ---------------------------------------------------------------------------
# Probe result
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    probe: Probe
    status: ProbeStatus
    output_text: str
    trace: ProbeTrace
    score: dict[str, Any] = field(default_factory=dict)
    judge_findings: list[dict[str, Any]] = field(default_factory=list)
    gate_findings: list[dict[str, Any]] = field(default_factory=list)
    attestation_chain_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "probe": self.probe.to_dict(), "status": self.status.value,
            "output_text": self.output_text[:2000],
            "trace": self.trace.to_dict(), "score": dict(self.score),
            "judge_findings": list(self.judge_findings),
            "gate_findings": list(self.gate_findings),
            "attestation_chain_hash": self.attestation_chain_hash,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Campaign summary
# ---------------------------------------------------------------------------

@dataclass
class CampaignSummary:
    total_probes: int = 0
    bypassed: int = 0
    blocked: int = 0
    passed: int = 0
    errors: int = 0
    skipped: int = 0
    asr: float = 0.0
    best_combined_score: float = 0.0
    category_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    pack_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    surface_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_results(cls, results: list[ProbeResult]) -> CampaignSummary:
        summary = cls(total_probes=len(results))
        cat_stats: dict[str, dict[str, int]] = {}
        pack_stats: dict[str, dict[str, int]] = {}
        surface_stats: dict[str, dict[str, int]] = {}
        best = 0.0
        for r in results:
            if r.status == ProbeStatus.BYPASSED:
                summary.bypassed += 1
            elif r.status == ProbeStatus.BLOCKED:
                summary.blocked += 1
            elif r.status == ProbeStatus.PASSED:
                summary.passed += 1
            elif r.status == ProbeStatus.ERROR:
                summary.errors += 1
            elif r.status == ProbeStatus.SKIPPED:
                summary.skipped += 1
            combined = r.score.get("combined", 0.0)
            if combined > best:
                best = combined
            for key, stats_dict, attr in [
                (r.probe.category, cat_stats, "category"),
                (r.probe.pack_id, pack_stats, "pack"),
                (r.probe.surface.value, surface_stats, "surface"),
            ]:
                stats_dict.setdefault(key, {"total": 0, "bypassed": 0})
                stats_dict[key]["total"] += 1
                if r.status == ProbeStatus.BYPASSED:
                    stats_dict[key]["bypassed"] += 1
        summary.best_combined_score = best
        tested = summary.total_probes - summary.errors - summary.skipped
        summary.asr = round(summary.bypassed / max(tested, 1) * 100, 1)
        summary.category_breakdown = cat_stats
        summary.pack_breakdown = pack_stats
        summary.surface_breakdown = surface_stats
        return summary


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------

@dataclass
class Campaign:
    campaign_id: str
    name: str
    mode: str = "run"
    target: Optional[TargetRef] = None
    probes: list[Probe] = field(default_factory=list)
    seed: int = 42
    output_dir: str = "results"
    pack_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.campaign_id:
            errors.append("campaign_id is required")
        seen_ids = set()
        for p in self.probes:
            if p.probe_id in seen_ids:
                errors.append(f"Duplicate probe_id: {p.probe_id}")
            seen_ids.add(p.probe_id)
            errors.extend(p.validate())
        return errors

    def to_dict(self) -> dict:
        return {
            "campaign_id": self.campaign_id, "name": self.name, "mode": self.mode,
            "target": self.target.to_dict() if self.target else None,
            "probe_count": len(self.probes), "seed": self.seed,
            "output_dir": self.output_dir, "pack_ids": list(self.pack_ids),
            "metadata": dict(self.metadata),
        }


@dataclass
class CampaignResult:
    campaign: Campaign
    results: list[ProbeResult] = field(default_factory=list)
    summary: Optional[CampaignSummary] = None
    artifacts: dict[str, str] = field(default_factory=dict)
    started_at: str = field(default_factory=_utc_now)
    completed_at: str = ""

    def to_dict(self) -> dict:
        return {
            "campaign": self.campaign.to_dict(),
            "results_count": len(self.results),
            "summary": self.summary.to_dict() if self.summary else None,
            "artifacts": dict(self.artifacts),
            "started_at": self.started_at, "completed_at": self.completed_at,
        }

    def finalize(self) -> None:
        self.completed_at = _utc_now()
        self.summary = CampaignSummary.from_results(self.results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_campaign_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"campaign-{ts}-{uuid.uuid4().hex[:6]}"


def generate_probe_id(pack_id: str, index: int) -> str:
    return f"{pack_id}_{index:04d}"


__all__ = [
    "ProbeSurface", "ProbeStatus", "TargetRef",
    "ChatProbe", "TrajectoryProbe", "ControlPlaneProbe",
    "Probe", "ProbeTrace", "ProbeResult",
    "CampaignSummary", "Campaign", "CampaignResult",
    "generate_campaign_id", "generate_probe_id",
]
