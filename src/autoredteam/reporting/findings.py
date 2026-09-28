"""reporting/findings.py — Normalized finding records shared by every output format."""
from __future__ import annotations

from typing import Any

from autoredteam.frameworks import frameworks_for

SEVERITY_ORDER = ("info", "low", "medium", "high", "critical")

# CVSS-style scores GitHub code scanning uses to bucket security alerts.
SECURITY_SEVERITY = {"critical": "9.5", "high": "8.0", "medium": "5.5", "low": "3.0", "info": "0.0"}

_AGENTIC_SEVERITY = {
    "goal_hijack": "critical", "excessive_agency": "high", "tool_poisoning": "critical",
    "memory_poisoning": "high", "data_exfiltration": "critical", "privilege_escalation": "critical",
    "unexpected_code_execution": "critical", "cross_session_leakage": "high", "rag_poisoning": "high",
    "inter_agent_spoofing": "high", "human_trust_exploitation": "medium", "unbounded_consumption": "medium",
}

REMEDIATION = {
    "prompt_injection": "Separate trusted instructions from user input, restate constraints after user content, and add an input-side injection classifier.",
    "indirect_injection": "Treat retrieved documents, tool output and web content as untrusted data; delimit it and never let it change the task or call tools directly.",
    "jailbreak": "Harden the system prompt against persona/role-play reframing and add output-side policy classification.",
    "pii_extraction": "Never place PII in the prompt context unless required; add output PII redaction and authorization checks on data lookups.",
    "system_prompt_leakage": "Keep secrets out of system prompts; add a canary token and block responses that echo instruction text.",
    "tool_misuse": "Scope tools to least privilege, validate arguments server-side, and require confirmation for destructive actions.",
    "encoding_bypass": "Normalize and decode input (base64, homoglyphs, invisible Unicode) before safety classification.",
    "output_formatting_exploit": "Treat model output as untrusted: escape it before rendering and never execute generated code without review.",
    "hallucination_exploit": "Ground answers in retrieved sources and instruct the model to refuse to fabricate citations or regulations.",
    "goal_hijack": "Pin the agent's objective outside the context window and verify each planned action against the original task.",
    "excessive_agency": "Remove unneeded tools and permissions; require human approval for irreversible or high-impact actions.",
    "tool_poisoning": "Pin and review tool/MCP server definitions; ignore instructions embedded in tool descriptions or results.",
    "memory_poisoning": "Validate and attribute memory writes; do not persist instructions from untrusted turns.",
    "data_exfiltration": "Restrict outbound tools (email, HTTP, issue trackers) and scan tool arguments for secrets and PII.",
    "privilege_escalation": "Bind actions to the end user's identity and permissions, not the agent's service account.",
    "unexpected_code_execution": "Execute generated code only in an isolated sandbox with no credentials and deny-by-default egress.",
    "cross_session_leakage": "Isolate memory and caches per user/session; never answer from another user's context.",
    "rag_poisoning": "Authenticate and sanitize indexed content; attribute retrieved chunks and treat them as data.",
}
_DEFAULT_REMEDIATION = "Review the transcript, add a targeted defense (system prompt, guardrail, or tool policy), and re-run to verify."


def probe_severity(probe: Any) -> str:
    hint = (getattr(probe, "severity_hint", "") or "").lower()
    if hint in SEVERITY_ORDER:
        return hint
    if probe.category in _AGENTIC_SEVERITY:
        return _AGENTIC_SEVERITY[probe.category]
    from autoredteam.attack import ATTACK_CATEGORIES
    return ATTACK_CATEGORIES.get(probe.category, {}).get("severity_class", "medium")


def severity_at_least(severity: str, threshold: str) -> bool:
    return SEVERITY_ORDER.index(severity) >= SEVERITY_ORDER.index(threshold)


def remediation_for(category: str) -> str:
    return REMEDIATION.get(category, _DEFAULT_REMEDIATION)


def prompt_text(result: Any) -> str:
    payload = result.probe.payload
    if payload is not None and hasattr(payload, "prompt"):
        return payload.prompt
    return "\n".join(t.content for t in result.trace.transcript if t.role == "user")


def finding_dict(result: Any) -> dict[str, Any]:
    probe = result.probe
    judge = result.judge_findings[0] if result.judge_findings else {}
    return {
        "probe_id": probe.probe_id,
        "pack_id": probe.pack_id,
        "category": probe.category,
        "surface": probe.surface.value,
        "title": probe.title,
        "severity": probe_severity(probe),
        "severity_hint": probe.severity_hint,
        "score": result.score,
        "frameworks": frameworks_for(probe.category).to_dict(),
        "strategies": list(probe.metadata.get("strategies", [])),
        "judge_reasoning": judge.get("reasoning", ""),
        "remediation": remediation_for(probe.category),
        "prompt_preview": prompt_text(result)[:500],
        "output_preview": result.output_text[:500],
        "attestation_chain_hash": result.attestation_chain_hash,
    }


def bypassed_results(results: list[Any]) -> list[Any]:
    return [r for r in results if r.status.value == "bypassed"]
