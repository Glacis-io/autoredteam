"""Regression tests for the v0.4 CLI and runner fixes."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


def _run_cli(argv: list[str]) -> tuple[int, str]:
    from autoredteam.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(argv)
    return code, buf.getvalue()


class TestRunCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="art-cli-")

    def test_rerun_does_not_duplicate_probe_results(self):
        args = ["run", "--dry-run", "--quiet", "--max-probes", "5", "--output-dir", self.tmp]
        _run_cli(args)
        _run_cli(args)
        lines = (Path(self.tmp) / "probe_results.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 5)

    def test_run_writes_verified_evidence_chain_and_receipt(self):
        code, out = _run_cli(["run", "--dry-run", "--quiet", "--max-probes", "4",
                              "--output-dir", self.tmp, "--attest"])
        self.assertEqual(code, 0)
        chain = (Path(self.tmp) / "evidence_chain.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(chain), 4)
        receipt = json.loads((Path(self.tmp) / "attestation_receipt.json").read_text(encoding="utf-8"))
        self.assertTrue(receipt["chain_verified"])
        self.assertIn("chain verified", out)

    def test_prompt_file_overrides_system_prompt(self):
        prompt_file = Path(self.tmp) / "prompt.txt"
        prompt_file.write_text("You are Acme's billing bot.", encoding="utf-8")
        _run_cli(["run", "--dry-run", "--quiet", "--max-probes", "2",
                  "--output-dir", self.tmp, "--prompt-file", str(prompt_file)])
        manifest = json.loads((Path(self.tmp) / "campaign_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["target"]["system_prompt"], "You are Acme's billing bot.")

    def test_api_judge_backend_is_actually_invoked(self):
        with mock.patch("autoredteam.scoring.dual_judge", return_value=(90.0, "judge says bypass")) as judge:
            _run_cli(["run", "--dry-run", "--quiet", "--max-probes", "19", "--output-dir", self.tmp,
                      "--judge-backend", "api", "--judge-model", "claude-haiku-4-5"])
        self.assertTrue(judge.called)
        self.assertEqual(judge.call_args.args[2], "claude-haiku-4-5")


class TestHardenCommand(unittest.TestCase):
    def test_harden_runs_loop_into_custom_output_dir(self):
        tmp = tempfile.mkdtemp(prefix="art-harden-")
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            code, _ = _run_cli([
                "harden", "--dry-run", "--quiet", "--cycles", "1", "--batch-size", "3",
                "--attack-cycles", "1", "--output-dir", "out/harden",
                "--training-data-dir", "out/td", "--attest",
            ])
        finally:
            os.chdir(cwd)
        out = Path(tmp) / "out" / "harden"
        self.assertEqual(code, 0)
        for name in ("hardened_prompt.txt", "policy.toml", "evidence_chain.jsonl", "attestation_receipt.json"):
            self.assertTrue((out / name).exists(), name)
        self.assertFalse((Path(tmp) / "results").exists())


class TestReportWithoutTarget(unittest.TestCase):
    def test_markdown_renders_when_campaign_has_no_target(self):
        from autoredteam.campaign import Campaign, CampaignResult
        from autoredteam.reporting.generator import ReportGenerator

        result = CampaignResult(campaign=Campaign(campaign_id="c1", name="no-target"))
        result.finalize()
        md = ReportGenerator().render_markdown(result)
        self.assertIn("**Target:** N/A", md)


class TestGitCommitOptIn(unittest.TestCase):
    def test_end_cycle_does_not_commit_by_default(self):
        from autoredteam.attestation import AttestationManager

        mgr = AttestationManager(output_dir=tempfile.mkdtemp(prefix="art-att-"))
        with mock.patch.object(mgr.local, "git_commit") as commit:
            mgr.end_cycle(1, {})
        commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
