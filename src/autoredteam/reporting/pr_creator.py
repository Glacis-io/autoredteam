"""
reporting/pr_creator.py — PR bundle creation and GitHub PR workflow.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class PRTargetFiles:
    """Files and branches for PR creation."""
    prompt_file: str = ""
    guardrail_file: str = ""
    report_dir: str = ""
    base_branch: str = "main"
    head_branch: str = ""
    labels: list[str] = field(default_factory=lambda: ["security", "autoredteam"])


class PRCreator:
    """Create PR bundles and optionally submit via gh CLI."""

    def prepare_bundle(
        self,
        report_artifacts: Any,
        targets: PRTargetFiles,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Prepare a PR bundle (title, body, files to include)."""
        # Read PR body from generated artifact
        pr_body = ""
        if report_artifacts.pr_body_md and Path(report_artifacts.pr_body_md).exists():
            pr_body = Path(report_artifacts.pr_body_md).read_text()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        head_branch = targets.head_branch or f"autoredteam/assessment-{timestamp}"

        return {
            "title": f"🔐 autoredteam: Security assessment ({timestamp})",
            "body": pr_body,
            "head_branch": head_branch,
            "base_branch": targets.base_branch,
            "labels": targets.labels,
            "files": {
                "report_md": report_artifacts.report_md,
                "report_json": report_artifacts.report_json,
                "findings_jsonl": report_artifacts.findings_jsonl,
                "summary_txt": report_artifacts.summary_txt,
            },
            "metadata": metadata or {},
        }

    def write_bundle(self, bundle: dict[str, Any], output_dir: str) -> dict[str, str]:
        """Write PR bundle to disk."""
        out = Path(output_dir) / "pr"
        out.mkdir(parents=True, exist_ok=True)

        bundle_path = out / "PR_BUNDLE.json"
        with open(bundle_path, "w") as f:
            json.dump(bundle, f, indent=2)

        body_path = out / "PR_BODY.md"
        with open(body_path, "w") as f:
            f.write(bundle.get("body", ""))

        return {
            "bundle_json": str(bundle_path),
            "body_md": str(body_path),
        }

    def create(
        self,
        bundle: dict[str, Any],
        create_mode: str = "dry_run",
    ) -> dict[str, Any]:
        """Create a PR.

        Modes:
        - 'dry_run': write bundle only, no git operations
        - 'gh_cli': create branch, commit files, use gh pr create
        """
        if create_mode == "dry_run":
            return {
                "mode": "dry_run",
                "title": bundle["title"],
                "head_branch": bundle["head_branch"],
                "status": "bundle_ready",
            }

        if create_mode == "gh_cli":
            return self._create_via_gh(bundle)

        return {"mode": create_mode, "status": "unsupported_mode"}

    def _create_via_gh(self, bundle: dict[str, Any]) -> dict[str, Any]:
        """Create PR using git and gh CLI."""
        head = bundle["head_branch"]
        base = bundle["base_branch"]
        title = bundle["title"]
        body = bundle.get("body", "")

        try:
            # Check we're in a git repo
            subprocess.run(["git", "rev-parse", "--git-dir"], capture_output=True, check=True)

            # Create branch
            subprocess.run(["git", "checkout", "-b", head], capture_output=True, check=True)

            # Add report files
            files = bundle.get("files", {})
            for key, filepath in files.items():
                if filepath and Path(filepath).exists():
                    subprocess.run(["git", "add", filepath], capture_output=True, check=True)

            # Commit
            subprocess.run(
                ["git", "commit", "-m", f"[autoredteam] {title}"],
                capture_output=True, check=True,
            )

            # Push
            subprocess.run(
                ["git", "push", "-u", "origin", head],
                capture_output=True, check=True,
            )

            # Create PR
            cmd = ["gh", "pr", "create",
                   "--title", title,
                   "--body", body,
                   "--base", base,
                   "--head", head]
            for label in bundle.get("labels", []):
                cmd.extend(["--label", label])

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            pr_url = result.stdout.strip()

            return {
                "mode": "gh_cli",
                "status": "created",
                "pr_url": pr_url,
                "head_branch": head,
            }

        except FileNotFoundError:
            return {"mode": "gh_cli", "status": "error", "error": "git or gh CLI not found"}
        except subprocess.CalledProcessError as e:
            return {
                "mode": "gh_cli", "status": "error",
                "error": f"Command failed: {e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}",
            }
