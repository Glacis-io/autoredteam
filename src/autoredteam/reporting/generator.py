"""
reporting/generator.py — Campaign report generation.

Produces markdown reports, JSON summaries, and PR body content.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autoredteam import __version__
from autoredteam.reporting.governance import compute_governance_score


@dataclass
class ReportArtifacts:
    """Paths to generated report files."""
    campaign_result_json: str = ""
    report_json: str = ""
    report_md: str = ""
    findings_jsonl: str = ""
    summary_txt: str = ""
    attestation_receipt_json: str = ""
    pr_body_md: str = ""


class ReportGenerator:
    """Generate comprehensive campaign reports."""

    def generate(self, campaign_result: Any, output_dir: str) -> ReportArtifacts:
        """Generate all report artifacts from a CampaignResult."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        artifacts = ReportArtifacts()

        # Campaign result JSON
        result_path = out / "campaign_result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(campaign_result.to_dict(), f, indent=2)
        artifacts.campaign_result_json = str(result_path)

        # Findings JSONL
        findings_path = out / "findings.jsonl"
        findings = self.render_findings_jsonl(campaign_result)
        with open(findings_path, "w", encoding="utf-8") as f:
            for finding in findings:
                f.write(json.dumps(finding) + "\n")
        artifacts.findings_jsonl = str(findings_path)

        # Markdown report
        md_path = out / "report.md"
        md_content = self.render_markdown(campaign_result)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        artifacts.report_md = str(md_path)

        # Summary text
        summary_path = out / "SUMMARY.txt"
        summary = self._render_summary_text(campaign_result)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        artifacts.summary_txt = str(summary_path)

        # Report JSON
        report_json_path = out / "report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "campaign": campaign_result.campaign.to_dict(),
                "summary": campaign_result.summary.to_dict() if campaign_result.summary else {},
                "governance": compute_governance_score(campaign_result.results).to_dict(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)
        artifacts.report_json = str(report_json_path)

        # PR body
        pr_path = out / "PR_BODY.md"
        pr_body = self._render_pr_body(campaign_result)
        with open(pr_path, "w", encoding="utf-8") as f:
            f.write(pr_body)
        artifacts.pr_body_md = str(pr_path)

        return artifacts

    def render_markdown(self, campaign_result: Any) -> str:
        """Render a comprehensive markdown report."""
        summary = campaign_result.summary
        governance = compute_governance_score(campaign_result.results)
        campaign = campaign_result.campaign

        lines: list[str] = []
        lines.append(f"# autoredteam Report")
        lines.append(f"")
        lines.append(f"**Campaign:** {campaign.name}")
        lines.append(f"**Mode:** {campaign.mode}")
        lines.append(f"**Target:** {campaign.target.provider}/{campaign.target.model if campaign.target else 'N/A'}")
        lines.append(f"**Packs:** {', '.join(campaign.pack_ids)}")
        lines.append(f"**Generated:** {campaign_result.completed_at or 'in progress'}")
        lines.append(f"")

        # Governance score
        lines.append(f"## Governance Score: {governance.score}/100 (Tier {governance.tier})")
        lines.append(f"")
        lines.append(f"| Dimension | Score |")
        lines.append(f"|-----------|-------|")
        lines.append(f"| Operational Safety | {governance.operational}/100 |")
        lines.append(f"| Governance & Compliance | {governance.governance}/100 |")
        lines.append(f"| Agentic Safety | {governance.agentic}/100 |")
        lines.append(f"")

        if summary:
            # Summary statistics
            lines.append(f"## Summary")
            lines.append(f"")
            lines.append(f"- **Total probes:** {summary.total_probes}")
            lines.append(f"- **Bypassed:** {summary.bypassed} ({summary.asr}% ASR)")
            lines.append(f"- **Blocked:** {summary.blocked}")
            lines.append(f"- **Passed:** {summary.passed}")
            lines.append(f"- **Errors:** {summary.errors}")
            lines.append(f"- **Best combined score:** {summary.best_combined_score}")
            lines.append(f"")

            # Category breakdown
            if summary.category_breakdown:
                lines.append(f"## Category Breakdown")
                lines.append(f"")
                lines.append(f"| Category | Total | Bypassed | ASR |")
                lines.append(f"|----------|-------|----------|-----|")
                for cat, stats in sorted(summary.category_breakdown.items()):
                    total = stats.get("total", 0)
                    bypassed = stats.get("bypassed", 0)
                    asr = round(bypassed / max(total, 1) * 100, 1)
                    lines.append(f"| {cat} | {total} | {bypassed} | {asr}% |")
                lines.append(f"")

            # Surface breakdown
            if summary.surface_breakdown:
                lines.append(f"## Surface Breakdown")
                lines.append(f"")
                for surf, stats in summary.surface_breakdown.items():
                    total = stats.get("total", 0)
                    bypassed = stats.get("bypassed", 0)
                    lines.append(f"- **{surf}:** {bypassed}/{total} bypassed")
                lines.append(f"")

        # Top findings
        bypassed_results = [r for r in campaign_result.results if r.status.value == "bypassed"]
        if bypassed_results:
            lines.append(f"## Top Findings")
            lines.append(f"")
            for i, r in enumerate(sorted(bypassed_results,
                                          key=lambda x: x.score.get("combined", 0),
                                          reverse=True)[:10], 1):
                lines.append(f"### {i}. [{r.probe.category}] {r.probe.title or r.probe.probe_id}")
                lines.append(f"- **Score:** {r.score.get('combined', 0)}")
                lines.append(f"- **Pack:** {r.probe.pack_id}")
                lines.append(f"- **Surface:** {r.probe.surface.value}")
                if r.judge_findings:
                    reasoning = r.judge_findings[0].get("reasoning", "")
                    if reasoning:
                        lines.append(f"- **Judge:** {reasoning[:200]}")
                lines.append(f"")

        lines.append(f"---")
        lines.append(f"*Generated by autoredteam v{__version__}*")

        return "\n".join(lines)

    def render_findings_jsonl(self, campaign_result: Any) -> list[dict[str, Any]]:
        """Render findings as a list of dicts for JSONL export."""
        findings: list[dict[str, Any]] = []
        for r in campaign_result.results:
            if r.status.value == "bypassed":
                findings.append({
                    "probe_id": r.probe.probe_id,
                    "pack_id": r.probe.pack_id,
                    "category": r.probe.category,
                    "surface": r.probe.surface.value,
                    "title": r.probe.title,
                    "severity_hint": r.probe.severity_hint,
                    "score": r.score,
                    "output_preview": r.output_text[:500],
                })
        return findings

    def _render_summary_text(self, campaign_result: Any) -> str:
        """Render a plain-text summary."""
        summary = campaign_result.summary
        governance = compute_governance_score(campaign_result.results)
        lines = [
            f"autoredteam Campaign Summary",
            f"============================",
            f"Campaign: {campaign_result.campaign.name}",
            f"Governance Score: {governance.score}/100 (Tier {governance.tier})",
        ]
        if summary:
            lines.extend([
                f"Total probes: {summary.total_probes}",
                f"Bypassed: {summary.bypassed} ({summary.asr}% ASR)",
                f"Blocked: {summary.blocked}",
                f"Passed: {summary.passed}",
                f"Errors: {summary.errors}",
            ])
        return "\n".join(lines) + "\n"

    def _render_pr_body(self, campaign_result: Any) -> str:
        """Render markdown suitable for a GitHub PR body."""
        governance = compute_governance_score(campaign_result.results)
        summary = campaign_result.summary

        lines = [
            f"## 🔐 autoredteam Security Assessment",
            f"",
            f"**Governance Score:** {governance.score}/100 (Tier {governance.tier})",
            f"",
        ]

        if summary:
            lines.extend([
                f"### Results",
                f"- Total probes: {summary.total_probes}",
                f"- Attack Success Rate: {summary.asr}%",
                f"- Bypassed: {summary.bypassed}",
                f"- Blocked: {summary.blocked}",
                f"",
            ])

        bypassed = [r for r in campaign_result.results if r.status.value == "bypassed"]
        if bypassed:
            lines.append("### Key Findings")
            for r in sorted(bypassed, key=lambda x: x.score.get("combined", 0), reverse=True)[:5]:
                lines.append(f"- **[{r.probe.category}]** {r.probe.title or r.probe.probe_id}")
            lines.append("")

        lines.extend([
            f"### Recommendations",
            f"- Review and strengthen system prompts for identified weak categories",
            f"- Add input validation for injection patterns",
            f"- Consider adding a safety middleware layer",
            f"",
            f"---",
            f"*Generated by [autoredteam](https://github.com/glacis-io/auto-redteam)*",
        ])

        return "\n".join(lines)
