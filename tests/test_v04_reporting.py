"""Tests for framework mapping, SARIF/JUnit output and CI exit codes."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from pathlib import Path


def _run(argv: list[str]) -> int:
    from autoredteam.cli import main

    with redirect_stdout(io.StringIO()):
        return main(argv)


class TestFrameworkMapping(unittest.TestCase):
    def test_every_builtin_category_is_mapped(self):
        from autoredteam.attack_packs.base import PackBuildContext
        from autoredteam.attack_packs.registry import get_pack_registry
        from autoredteam.frameworks import CATEGORY_FRAMEWORKS

        registry = get_pack_registry()
        ctx = PackBuildContext(max_probes=200)
        categories = {p.category for meta in registry.list() for p in registry.get(meta.pack_id).build_probes(ctx)}
        self.assertEqual(categories - set(CATEGORY_FRAMEWORKS), set())

    def test_mapping_ids_exist_in_catalogs(self):
        from autoredteam.frameworks import (
            CATEGORY_FRAMEWORKS, MITRE_ATLAS, NIST_AI_600_1, OWASP_AGENTIC_2026, OWASP_LLM_2025,
        )

        for category, refs in CATEGORY_FRAMEWORKS.items():
            with self.subTest(category=category):
                self.assertTrue(set(refs.owasp_llm) <= set(OWASP_LLM_2025))
                self.assertTrue(set(refs.owasp_agentic) <= set(OWASP_AGENTIC_2026))
                self.assertTrue(set(refs.mitre_atlas) <= set(MITRE_ATLAS))
                self.assertTrue(set(refs.nist_ai_600_1) <= set(NIST_AI_600_1))

    def test_well_known_mappings(self):
        from autoredteam.frameworks import frameworks_for

        self.assertEqual(frameworks_for("system_prompt_leakage").owasp_llm, ("LLM07",))
        self.assertIn("AML.T0051.001", frameworks_for("indirect_injection").mitre_atlas)
        self.assertIn("ASI06", frameworks_for("memory_poisoning").owasp_agentic)


class TestCIOutputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="art-ci-")

    def test_sarif_and_junit_are_well_formed(self):
        code = _run(["run", "--dry-run", "--quiet", "--output-dir", self.tmp, "--format", "sarif", "junit",
                     "--sarif-artifact", "prompts/system.txt"])
        self.assertEqual(code, 0)
        sarif = json.loads((Path(self.tmp) / "results.sarif").read_text(encoding="utf-8"))
        self.assertEqual(sarif["version"], "2.1.0")
        run = sarif["runs"][0]
        rules = {r["id"]: r for r in run["tool"]["driver"]["rules"]}
        self.assertTrue(run["results"])
        for res in run["results"]:
            rule = rules[res["ruleId"]]
            self.assertEqual(run["tool"]["driver"]["rules"][res["ruleIndex"]]["id"], res["ruleId"])
            self.assertIn("security-severity", rule["properties"])
            self.assertEqual(res["locations"][0]["physicalLocation"]["artifactLocation"]["uri"], "prompts/system.txt")
            self.assertIn("autoredteam/v1", res["partialFingerprints"])

        junit = ET.parse(Path(self.tmp) / "junit.xml").getroot()
        findings = (Path(self.tmp) / "findings.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(int(junit.get("failures")), len(findings))
        self.assertEqual(int(junit.get("tests")), 20)

    def test_findings_carry_frameworks_severity_and_remediation(self):
        _run(["run", "--dry-run", "--quiet", "--output-dir", self.tmp])
        finding = json.loads((Path(self.tmp) / "findings.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertIn("owasp_llm_2025", finding["frameworks"])
        self.assertIn(finding["severity"], {"low", "medium", "high", "critical"})
        self.assertTrue(finding["remediation"])
        report = (Path(self.tmp) / "report.md").read_text(encoding="utf-8")
        self.assertIn("OWASP Top 10 for LLM Applications 2025", report)
        self.assertIn("MITRE ATLAS", report)


class TestExitCodes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="art-exit-")

    def test_default_run_passes(self):
        self.assertEqual(_run(["run", "--dry-run", "--quiet", "--output-dir", self.tmp]), 0)

    def test_fail_on_severity_gates_build(self):
        self.assertEqual(_run(["run", "--dry-run", "--quiet", "--output-dir", self.tmp, "--fail-on", "high"]), 1)
        self.assertEqual(_run(["run", "--dry-run", "--quiet", "--output-dir", self.tmp, "--fail-on", "critical"]), 0)

    def test_max_asr_gates_build(self):
        self.assertEqual(_run(["run", "--dry-run", "--quiet", "--output-dir", self.tmp, "--max-asr", "5"]), 1)
        self.assertEqual(_run(["run", "--dry-run", "--quiet", "--output-dir", self.tmp, "--max-asr", "50"]), 0)

    def test_probe_errors_exit_2(self):
        code = _run(["run", "--provider", "http", "--endpoint", "http://127.0.0.1:9/unreachable",
                     "--http-timeout", "1", "--quiet", "--max-probes", "2", "--output-dir", self.tmp])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
