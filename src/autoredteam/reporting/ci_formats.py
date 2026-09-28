"""reporting/ci_formats.py — SARIF 2.1.0 and JUnit XML output for CI pipelines.

SARIF uploads to GitHub code scanning (``github/codeql-action/upload-sarif``)
turn each bypassed probe into a security alert; JUnit XML lets any CI system
show probes as test cases.
"""
from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

from autoredteam import __version__
from autoredteam.frameworks import describe, frameworks_for
from autoredteam.reporting.findings import (
    SECURITY_SEVERITY, bypassed_results, finding_dict, probe_severity, remediation_for,
)

SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"
TOOL_URI = "https://github.com/glacis-io/auto-redteam"
_SARIF_LEVEL = {"critical": "error", "high": "error", "medium": "warning", "low": "note", "info": "note"}


def _rule_id(category: str) -> str:
    return f"autoredteam/{category}"


def _rule(category: str, severity: str, description: str) -> dict[str, Any]:
    refs = frameworks_for(category)
    mapped = [f"{fid} {describe(fid)}".strip() for fid in (*refs.owasp_llm, *refs.owasp_agentic, *refs.mitre_atlas)]
    help_md = f"**Remediation:** {remediation_for(category)}"
    if mapped:
        help_md += "\n\n**Frameworks:** " + "; ".join(mapped)
    return {
        "id": _rule_id(category),
        "name": "".join(part.title() for part in category.split("_")),
        "shortDescription": {"text": category.replace("_", " ").capitalize()},
        "fullDescription": {"text": description or category.replace("_", " ")},
        "help": {"text": remediation_for(category), "markdown": help_md},
        "defaultConfiguration": {"level": _SARIF_LEVEL.get(severity, "warning")},
        "properties": {
            "tags": ["security", "llm", *refs.tags()],
            "security-severity": SECURITY_SEVERITY.get(severity, "5.5"),
            "precision": "medium",
        },
    }


def render_sarif(campaign_result: Any, artifact_uri: Optional[str] = None) -> dict[str, Any]:
    from autoredteam.attack import ATTACK_CATEGORIES

    campaign = campaign_result.campaign
    target = campaign.target
    target_label = f"{target.provider}/{target.model}" if target else "unknown"
    uri = artifact_uri or f".autoredteam/{target_label.replace('/', '-') or 'target'}.target"

    rules: list[dict[str, Any]] = []
    rule_index: dict[str, int] = {}
    results: list[dict[str, Any]] = []
    for r in bypassed_results(campaign_result.results):
        category = r.probe.category
        severity = probe_severity(r.probe)
        if category not in rule_index:
            rule_index[category] = len(rules)
            desc = ATTACK_CATEGORIES.get(category, {}).get("description", r.probe.description)
            rules.append(_rule(category, severity, desc))
        finding = finding_dict(r)
        fingerprint = hashlib.sha256(f"{target_label}|{category}|{finding['prompt_preview']}".encode()).hexdigest()[:32]
        results.append({
            "ruleId": _rule_id(category),
            "ruleIndex": rule_index[category],
            "level": _SARIF_LEVEL.get(severity, "warning"),
            "message": {"text": f"[{severity}] {r.probe.title or category} bypassed defenses of {target_label} "
                                f"(score {r.score.get('combined', 0)}). Response: {r.output_text[:200]}"},
            "locations": [{"physicalLocation": {"artifactLocation": {"uri": uri}, "region": {"startLine": 1}}}],
            "partialFingerprints": {"autoredteam/v1": fingerprint},
            "properties": {
                "probe_id": r.probe.probe_id,
                "pack_id": r.probe.pack_id,
                "surface": r.probe.surface.value,
                "frameworks": finding["frameworks"],
                "strategies": finding["strategies"],
                "attestation_chain_hash": r.attestation_chain_hash,
            },
        })

    return {
        "$schema": SARIF_SCHEMA,
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {
                "name": "autoredteam",
                "version": __version__,
                "semanticVersion": __version__,
                "informationUri": TOOL_URI,
                "rules": rules,
            }},
            "automationDetails": {"id": f"autoredteam/{target_label}/"},
            "invocations": [{
                "executionSuccessful": True,
                "startTimeUtc": campaign_result.started_at,
                "endTimeUtc": campaign_result.completed_at or campaign_result.started_at,
            }],
            "results": results,
        }],
    }


def render_junit(campaign_result: Any) -> str:
    by_pack: dict[str, list[Any]] = {}
    for r in campaign_result.results:
        by_pack.setdefault(r.probe.pack_id, []).append(r)

    root = ET.Element("testsuites", name=f"autoredteam {campaign_result.campaign.name}")
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for pack_id, results in by_pack.items():
        suite = ET.SubElement(root, "testsuite", name=pack_id)
        counts = {"tests": len(results), "failures": 0, "errors": 0, "skipped": 0}
        for r in results:
            latency_ms = sum(t.latency_ms or 0 for t in r.trace.transcript)
            case = ET.SubElement(suite, "testcase", classname=f"{pack_id}.{r.probe.category}",
                                 name=r.probe.title or r.probe.probe_id, time=f"{latency_ms / 1000:.3f}")
            status = r.status.value
            if status == "bypassed":
                counts["failures"] += 1
                severity = probe_severity(r.probe)
                failure = ET.SubElement(case, "failure", type=severity,
                                        message=f"{severity} {r.probe.category} bypass (score {r.score.get('combined', 0)})")
                failure.text = json.dumps(finding_dict(r), indent=2)
            elif status == "error":
                counts["errors"] += 1
                ET.SubElement(case, "error", message=(r.trace.error or "error")[:200])
            elif status == "skipped":
                counts["skipped"] += 1
                ET.SubElement(case, "skipped")
        for key, value in counts.items():
            suite.set(key, str(value))
            totals[key] += value
    for key, value in totals.items():
        root.set(key, str(value))
    ET.indent(root)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def write_sarif(campaign_result: Any, path: Path, artifact_uri: Optional[str] = None) -> str:
    path.write_text(json.dumps(render_sarif(campaign_result, artifact_uri), indent=2), encoding="utf-8")
    return str(path)


def write_junit(campaign_result: Any, path: Path) -> str:
    path.write_text(render_junit(campaign_result), encoding="utf-8")
    return str(path)
