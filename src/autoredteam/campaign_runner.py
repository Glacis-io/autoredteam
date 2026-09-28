"""
campaign_runner.py — Unified campaign execution engine.

Replaces the bespoke loops in run.py, autoharden.py, and validation scripts.
All probe execution, scoring, attestation, and training-data recording
flows through CampaignRunner.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from autoredteam.campaign import (
    Campaign, CampaignResult, CampaignSummary,
    ChatProbe, ControlPlaneProbe, Probe, ProbeResult, ProbeStatus,
    ProbeSurface, ProbeTrace, TargetRef, TrajectoryProbe,
    generate_campaign_id,
)
from autoredteam.conversation import ConversationTurn, _utc_now
from autoredteam.providers.base import (
    BaseTargetSession, ProviderRateLimitError, ProviderRequestError,
)
from autoredteam.providers.registry import ProviderRegistry, get_provider_registry
from autoredteam.scoring_v2 import ProbeScore, ScoreEngineV2, ScoreConfigV2
from autoredteam.trajectory_engine import TrajectoryEngine, TrajectoryEngineConfig


@dataclass
class CampaignRunConfig:
    """Runtime configuration for campaign execution."""
    output_dir: str = "results"
    max_retries: int = 2
    retry_backoff_sec: float = 1.0
    continue_on_probe_error: bool = True
    resume: bool = False
    attest_each_probe: bool = True
    write_jsonl_incrementally: bool = True
    stop_on_auth_error: bool = True


class CampaignRunner:
    """Unified engine that executes any campaign across all surfaces."""

    def __init__(
        self,
        provider_registry: Optional[ProviderRegistry] = None,
        score_engine: Optional[ScoreEngineV2] = None,
        attestation: Optional[Any] = None,  # AttestationManager
        collector: Optional[Any] = None,    # TrainingDataCollector
        trajectory_engine: Optional[TrajectoryEngine] = None,
        stealth_engine: Optional[Any] = None,
        config: Optional[CampaignRunConfig] = None,
    ):
        self.registry = provider_registry or get_provider_registry()
        self.score_engine = score_engine or ScoreEngineV2()
        self.attestation = attestation
        self.collector = collector
        self.trajectory_engine = trajectory_engine or TrajectoryEngine()
        self.stealth_engine = stealth_engine
        self.config = config or CampaignRunConfig()
        self._prior_hashes: list[str] = []

    def run_campaign(self, campaign: Campaign) -> CampaignResult:
        """Execute a full campaign and return results."""
        # Validate
        errors = campaign.validate()
        if errors:
            raise ValueError(f"Campaign validation failed: {errors}")

        output_dir = Path(campaign.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write campaign manifest
        manifest_path = output_dir / "campaign_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(campaign.to_dict(), f, indent=2)

        # Load resume state if requested
        completed_ids: set[str] = set()
        if self.config.resume:
            completed_ids = self._load_resume_state(campaign)

        result = CampaignResult(campaign=campaign, started_at=_utc_now())
        results_path = output_dir / "probe_results.jsonl"

        # Create session if target is specified
        session: Optional[BaseTargetSession] = None
        if campaign.target:
            from autoredteam.providers.base import TargetSpec
            spec = TargetSpec(
                provider=campaign.target.provider,
                model=campaign.target.model,
                system_prompt=campaign.target.system_prompt,
                temperature=campaign.target.temperature,
                max_output_tokens=campaign.target.max_output_tokens,
                deployment=campaign.target.deployment,
                endpoint=campaign.target.endpoint,
                region=campaign.target.region,
                project=campaign.target.project,
                api_version=campaign.target.api_version,
                account_id=campaign.target.account_id,
            )
            try:
                session = self.registry.create_session(spec)
            except Exception as e:
                if self.config.stop_on_auth_error:
                    raise
                print(f"⚠ Provider error: {e}")

        try:
            for i, probe in enumerate(campaign.probes):
                if probe.probe_id in completed_ids:
                    continue

                probe_result = self._execute_and_score_probe(
                    probe, session, campaign.target, i + 1, len(campaign.probes),
                )
                result.results.append(probe_result)

                # Incremental persistence
                if self.config.write_jsonl_incrementally:
                    self._write_incremental(results_path, probe_result)

                # Attestation
                if self.config.attest_each_probe and self.attestation:
                    self._record_attestation(probe_result, i + 1)

                # Training data
                if self.collector:
                    self._record_training(probe_result)

        except KeyboardInterrupt:
            print("\n⚠ Campaign interrupted — saving partial results...")
            self._write_state(output_dir, result)
            result.finalize()
            return result
        finally:
            if session:
                session.close()

        # Finalize
        result.finalize()
        self._write_final(output_dir, result)
        return result

    def run_probe(self, probe: Probe, target: Optional[TargetRef] = None) -> ProbeResult:
        """Execute a single probe (standalone mode)."""
        session = None
        effective_target = target or probe.target_ref
        if effective_target and probe.surface != ProbeSurface.CONTROL_PLANE:
            from autoredteam.providers.base import TargetSpec
            spec = TargetSpec(
                provider=effective_target.provider,
                model=effective_target.model,
                system_prompt=effective_target.system_prompt,
                temperature=effective_target.temperature,
            )
            session = self.registry.create_session(spec)
        try:
            return self._execute_and_score_probe(probe, session, effective_target, 1, 1)
        finally:
            if session:
                session.close()

    # ---- Internal methods ----

    def _execute_and_score_probe(
        self, probe: Probe, session: Optional[BaseTargetSession],
        target: Optional[TargetRef], index: int, total: int,
    ) -> ProbeResult:
        """Execute, score, and package one probe."""
        # Apply stealth
        if self.stealth_engine and probe.stealth_profile != "none":
            self.stealth_engine.apply_probe(probe)

        # Execute
        trace: ProbeTrace
        output_text: str = ""

        try:
            if probe.surface == ProbeSurface.CHAT:
                trace, output_text = self._execute_chat_probe(probe, session)
            elif probe.surface == ProbeSurface.TRAJECTORY:
                trace, output_text = self._execute_trajectory_probe(probe, session)
            elif probe.surface == ProbeSurface.CONTROL_PLANE:
                trace, output_text = self._execute_control_plane_probe(probe)
            else:
                trace = ProbeTrace(error="Unknown surface")
                output_text = ""
        except Exception as e:
            trace = ProbeTrace(error=str(e), ended_at=_utc_now())
            output_text = f"[ERROR] {e}"

        # Build initial result
        status = ProbeStatus.ERROR if trace.error else ProbeStatus.PASSED
        probe_result = ProbeResult(
            probe=probe, status=status, output_text=output_text, trace=trace,
        )

        # Score
        try:
            score = self.score_engine.score_probe(probe_result, self._prior_hashes)
            probe_result.score = score.to_dict()
            if score.bypassed:
                probe_result.status = ProbeStatus.BYPASSED
            if score.judge:
                probe_result.judge_findings.append(score.judge.to_dict())
            if score.gate_finding:
                probe_result.gate_findings.append(score.gate_finding.to_dict())
                if not score.gate_finding.passed:
                    probe_result.status = ProbeStatus.BYPASSED
        except Exception as e:
            probe_result.metadata["scoring_error"] = str(e)

        return probe_result

    def _execute_chat_probe(
        self, probe: Probe, session: Optional[BaseTargetSession],
    ) -> tuple[ProbeTrace, str]:
        """Execute a single-turn chat probe."""
        if not isinstance(probe.payload, ChatProbe):
            return ProbeTrace(error="Invalid payload"), ""

        trace = ProbeTrace(started_at=_utc_now())
        if session is None:
            trace.error = "No session available"
            trace.ended_at = _utc_now()
            return trace, ""

        session.reset()
        prompt = probe.payload.prompt

        # Record user turn
        user_turn = ConversationTurn(role="user", content=prompt, turn_index=0)
        trace.transcript.append(user_turn)

        # Send with retry
        response = self._send_with_retry(session, prompt, 0)
        assistant_turn = ConversationTurn(
            role="assistant", content=response.text, turn_index=0,
            provider_request_id=response.provider_request_id,
            latency_ms=response.latency_ms,
        )
        trace.transcript.append(assistant_turn)
        trace.ended_at = _utc_now()

        return trace, response.text

    def _execute_trajectory_probe(
        self, probe: Probe, session: Optional[BaseTargetSession],
    ) -> tuple[ProbeTrace, str]:
        """Execute a multi-turn trajectory probe."""
        if not isinstance(probe.payload, TrajectoryProbe) or session is None:
            return ProbeTrace(error="Invalid payload or no session"), ""

        trace, summary = self.trajectory_engine.execute(probe, session)

        # Collect output text from all assistant turns
        assistant_texts = [
            t.content for t in trace.transcript
            if t.role in ("assistant", "model")
        ]
        output_text = "\n---\n".join(assistant_texts)

        # Add summary to trace metadata
        probe_metadata = {
            "success_turn": summary.success_turn,
            "detection_turn": summary.detection_turn,
            "completed_turns": summary.completed_turns,
            "stop_reason": summary.stop_reason,
            "bypassed": summary.bypassed,
        }

        return trace, output_text

    def _execute_control_plane_probe(
        self, probe: Probe,
    ) -> tuple[ProbeTrace, str]:
        """Execute a control-plane probe (no chat session needed)."""
        if not isinstance(probe.payload, ControlPlaneProbe):
            return ProbeTrace(error="Invalid payload"), ""

        trace = ProbeTrace(started_at=_utc_now())
        payload = probe.payload

        # Control-plane execution would delegate to harness-specific code
        # For now, record the probe setup as a control event
        trace.control_events.append({
            "harness": payload.harness,
            "endpoint": payload.endpoint,
            "request_body": payload.request_body,
            "expected_decision": payload.expected_decision,
        })

        # TODO: integrate with validation/pangoclaw_control_plane.py harness
        output_text = f"Control plane probe: {payload.harness}:{payload.endpoint}"
        trace.ended_at = _utc_now()

        return trace, output_text

    def _send_with_retry(
        self, session: BaseTargetSession, content: str, turn_index: int,
    ) -> Any:
        """Send with retry on rate limits."""
        from autoredteam.providers.base import ProviderResponse as PR
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return session.send_user_turn(content, turn_index)
            except ProviderRateLimitError as e:
                last_error = e
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_backoff_sec * (attempt + 1))
            except ProviderRequestError:
                raise
        raise last_error or ProviderRequestError("Max retries exceeded")

    def _record_attestation(self, result: ProbeResult, cycle: int) -> None:
        """Record probe result in attestation chain."""
        if self.attestation:
            prompt_text = ""
            if isinstance(result.probe.payload, ChatProbe):
                prompt_text = result.probe.payload.prompt
            elif result.trace.transcript:
                prompt_text = " ".join(t.content for t in result.trace.transcript if t.role == "user")

            chain_hash = self.attestation.record_attack(
                cycle=cycle,
                attack_id=result.probe.probe_id,
                prompt=prompt_text,
                response=result.output_text,
                category=result.probe.category,
                score_vector=result.score,
                composite_score=result.score.get("combined", 0.0),
                deterministic_flags=[],
                phase="probe",
            )
            result.attestation_chain_hash = chain_hash

    def _record_training(self, result: ProbeResult) -> None:
        """Record probe result for training data."""
        if self.collector and hasattr(self.collector, "record_probe_result"):
            self.collector.record_probe_result(result)

    def _write_incremental(self, path: Path, result: ProbeResult) -> None:
        """Append one result to JSONL."""
        with open(path, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    def _write_state(self, output_dir: Path, result: CampaignResult) -> None:
        """Write resume state."""
        state = {
            "completed_probe_ids": [r.probe.probe_id for r in result.results],
            "timestamp": _utc_now(),
        }
        with open(output_dir / "campaign_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def _load_resume_state(self, campaign: Campaign) -> set[str]:
        """Load completed probe IDs from prior run."""
        state_path = Path(campaign.output_dir) / "campaign_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            return set(state.get("completed_probe_ids", []))
        return set()

    def _write_final(self, output_dir: Path, result: CampaignResult) -> None:
        """Write final campaign result and state."""
        with open(output_dir / "campaign_result.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        self._write_state(output_dir, result)


__all__ = ["CampaignRunner", "CampaignRunConfig"]
