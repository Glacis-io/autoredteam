"""frameworks.py — Map attack categories to industry risk frameworks.

Every finding carries references to:

* OWASP Top 10 for LLM Applications 2025 (``LLM01``–``LLM10``)
* OWASP Top 10 for Agentic Applications 2026 (``ASI01``–``ASI10``)
* MITRE ATLAS techniques (``AML.T0051.000`` …)
* NIST AI 600-1 Generative AI Profile risk categories

Sources (reviewed 2026-09):
  https://genai.owasp.org/llm-top-10/
  https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/
  https://atlas.mitre.org/techniques
  https://doi.org/10.6028/NIST.AI.600-1
"""
from __future__ import annotations

from dataclasses import dataclass

OWASP_LLM_2025: dict[str, str] = {
    "LLM01": "Prompt Injection",
    "LLM02": "Sensitive Information Disclosure",
    "LLM03": "Supply Chain",
    "LLM04": "Data and Model Poisoning",
    "LLM05": "Improper Output Handling",
    "LLM06": "Excessive Agency",
    "LLM07": "System Prompt Leakage",
    "LLM08": "Vector and Embedding Weaknesses",
    "LLM09": "Misinformation",
    "LLM10": "Unbounded Consumption",
}

OWASP_AGENTIC_2026: dict[str, str] = {
    "ASI01": "Agent Goal Hijack",
    "ASI02": "Tool Misuse and Exploitation",
    "ASI03": "Identity and Privilege Abuse",
    "ASI04": "Agentic Supply Chain Vulnerabilities",
    "ASI05": "Unexpected Code Execution (RCE)",
    "ASI06": "Memory and Context Poisoning",
    "ASI07": "Insecure Inter-Agent Communication",
    "ASI08": "Cascading Failures",
    "ASI09": "Human-Agent Trust Exploitation",
    "ASI10": "Rogue Agents",
}

MITRE_ATLAS: dict[str, str] = {
    "AML.T0050": "Command and Scripting Interpreter",
    "AML.T0051.000": "LLM Prompt Injection: Direct",
    "AML.T0051.001": "LLM Prompt Injection: Indirect",
    "AML.T0053": "AI Agent Tool Invocation",
    "AML.T0054": "LLM Jailbreak",
    "AML.T0056": "Extract LLM System Prompt",
    "AML.T0057": "LLM Data Leakage",
    "AML.T0062": "Discover LLM Hallucinations",
    "AML.T0068": "LLM Prompt Obfuscation",
    "AML.T0070": "RAG Poisoning",
    "AML.T0080.000": "AI Agent Context Poisoning: Memory",
    "AML.T0080.001": "AI Agent Context Poisoning: Thread",
    "AML.T0086": "Exfiltration via AI Agent Tool Invocation",
    "AML.T0110": "AI Agent Tool Poisoning",
}

NIST_AI_600_1: tuple[str, ...] = (
    "CBRN Information or Capabilities",
    "Confabulation",
    "Dangerous, Violent, or Hateful Content",
    "Data Privacy",
    "Environmental Impacts",
    "Harmful Bias or Homogenization",
    "Human-AI Configuration",
    "Information Integrity",
    "Information Security",
    "Intellectual Property",
    "Obscene, Degrading, and/or Abusive Content",
    "Value Chain and Component Integration",
)

_INFOSEC = "Information Security"


@dataclass(frozen=True)
class FrameworkRefs:
    owasp_llm: tuple[str, ...] = ()
    owasp_agentic: tuple[str, ...] = ()
    mitre_atlas: tuple[str, ...] = ()
    nist_ai_600_1: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "owasp_llm_2025": list(self.owasp_llm),
            "owasp_agentic_2026": list(self.owasp_agentic),
            "mitre_atlas": list(self.mitre_atlas),
            "nist_ai_600_1": list(self.nist_ai_600_1),
        }

    def tags(self) -> list[str]:
        """Flat identifiers suitable for SARIF ``properties.tags``."""
        return [
            *(f"OWASP-{i}:2025" for i in self.owasp_llm),
            *(f"OWASP-{i}:2026" for i in self.owasp_agentic),
            *(f"MITRE-ATLAS-{i}" for i in self.mitre_atlas),
        ]


def _refs(llm=(), asi=(), atlas=(), nist=(_INFOSEC,)) -> FrameworkRefs:
    return FrameworkRefs(tuple(llm), tuple(asi), tuple(atlas), tuple(nist))


CATEGORY_FRAMEWORKS: dict[str, FrameworkRefs] = {
    # Generic taxonomy
    "prompt_injection": _refs(["LLM01"], ["ASI01"], ["AML.T0051.000"]),
    "indirect_injection": _refs(["LLM01"], ["ASI01"], ["AML.T0051.001"]),
    "jailbreak": _refs(["LLM01"], [], ["AML.T0054"], [_INFOSEC, "Dangerous, Violent, or Hateful Content"]),
    "pii_extraction": _refs(["LLM02"], [], ["AML.T0057"], ["Data Privacy"]),
    "system_prompt_leakage": _refs(["LLM07"], [], ["AML.T0056"]),
    "tool_misuse": _refs(["LLM06"], ["ASI02"], ["AML.T0053"]),
    "role_confusion": _refs(["LLM01"], ["ASI01"], ["AML.T0054"], [_INFOSEC, "Human-AI Configuration"]),
    "context_window_poisoning": _refs(["LLM01"], ["ASI06"], ["AML.T0051.000", "AML.T0080.001"]),
    "multi_turn_manipulation": _refs(["LLM01"], ["ASI06"], ["AML.T0054", "AML.T0080.001"]),
    "encoding_bypass": _refs(["LLM01"], [], ["AML.T0068", "AML.T0054"]),
    "payload_splitting": _refs(["LLM01"], [], ["AML.T0068", "AML.T0051.000"]),
    "refusal_suppression": _refs(["LLM01"], [], ["AML.T0054"]),
    "ethical_bypass": _refs(["LLM01"], [], ["AML.T0054"], [_INFOSEC, "Dangerous, Violent, or Hateful Content"]),
    "authority_manipulation": _refs(["LLM01"], ["ASI03"], ["AML.T0054"]),
    "output_formatting_exploit": _refs(["LLM05"], [], ["AML.T0054"]),
    "multilingual_attack": _refs(["LLM01"], [], ["AML.T0068", "AML.T0054"]),
    "continuation_attack": _refs(["LLM01"], [], ["AML.T0054"]),
    "social_engineering": _refs(["LLM01"], [], ["AML.T0054"], [_INFOSEC, "Human-AI Configuration"]),
    "hallucination_exploit": _refs(["LLM09"], [], ["AML.T0062"], ["Confabulation", "Information Integrity"]),
    # Agentic pack
    "goal_hijack": _refs(["LLM01", "LLM06"], ["ASI01"], ["AML.T0051.001"]),
    "excessive_agency": _refs(["LLM06"], ["ASI02", "ASI03"], ["AML.T0053"], [_INFOSEC, "Human-AI Configuration"]),
    "tool_poisoning": _refs(["LLM01", "LLM03"], ["ASI02", "ASI04"], ["AML.T0110", "AML.T0051.001"],
                            [_INFOSEC, "Value Chain and Component Integration"]),
    "memory_poisoning": _refs(["LLM04"], ["ASI06"], ["AML.T0080.000"]),
    "data_exfiltration": _refs(["LLM02", "LLM06"], ["ASI02"], ["AML.T0086", "AML.T0057"], [_INFOSEC, "Data Privacy"]),
    "privilege_escalation": _refs(["LLM06"], ["ASI03"], ["AML.T0053"]),
    "unexpected_code_execution": _refs(["LLM05", "LLM06"], ["ASI05"], ["AML.T0050", "AML.T0053"]),
    "cross_session_leakage": _refs(["LLM02"], ["ASI06"], ["AML.T0057"], [_INFOSEC, "Data Privacy"]),
    "rag_poisoning": _refs(["LLM08", "LLM04"], ["ASI06"], ["AML.T0070", "AML.T0051.001"], [_INFOSEC, "Information Integrity"]),
    "inter_agent_spoofing": _refs(["LLM01"], ["ASI07"], ["AML.T0051.001"]),
    "human_trust_exploitation": _refs(["LLM09"], ["ASI09"], [], ["Human-AI Configuration", "Information Integrity"]),
    "unbounded_consumption": _refs(["LLM10"], ["ASI08"], [], [_INFOSEC, "Environmental Impacts"]),
}

_UNMAPPED = FrameworkRefs()


def frameworks_for(category: str) -> FrameworkRefs:
    return CATEGORY_FRAMEWORKS.get(category, _UNMAPPED)


def describe(framework_id: str) -> str:
    """Human-readable name for any OWASP / ATLAS identifier."""
    return OWASP_LLM_2025.get(framework_id) or OWASP_AGENTIC_2026.get(framework_id) or MITRE_ATLAS.get(framework_id, "")


def coverage(results: list, key: str = "owasp_llm") -> dict[str, dict[str, int]]:
    """Tested/bypassed counts per framework ID for a list of ProbeResult objects."""
    out: dict[str, dict[str, int]] = {}
    for r in results:
        for fid in getattr(frameworks_for(r.probe.category), key):
            stats = out.setdefault(fid, {"tested": 0, "bypassed": 0})
            stats["tested"] += 1
            if r.status.value == "bypassed":
                stats["bypassed"] += 1
    return dict(sorted(out.items()))


__all__ = [
    "OWASP_LLM_2025", "OWASP_AGENTIC_2026", "MITRE_ATLAS", "NIST_AI_600_1",
    "FrameworkRefs", "CATEGORY_FRAMEWORKS", "frameworks_for", "describe", "coverage",
]
