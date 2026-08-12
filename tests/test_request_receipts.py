"""Request-bound receipt conformance, signing, and disclosure tests."""

from __future__ import annotations

import copy
import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from autoredteam.receipts import (
    ReceiptError,
    build_receipt_artifacts,
    generate_signing_key,
    verify_receipt,
)
from autoredteam import receipts as receipts_module
from autoredteam.requirements import (
    RequirementError,
    load_requirement,
    requirement_digest,
    validate_requirement,
)
from campaign import (
    Campaign,
    CampaignResult,
    ChatProbe,
    Probe,
    ProbeResult,
    ProbeStatus,
    ProbeSurface,
    ProbeTrace,
    TargetRef,
)
from conversation import ConversationTurn
import cli


ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "examples" / "requirement-v0.1.json"
VECTOR_DIGEST = "sha256:e143ddf7908f7a7b7df81da8f8ca1736e5cec09a74d80683f7c32094aff4af5c"
AT = datetime(2026, 8, 12, 12, 0, 0, tzinfo=timezone.utc)
RAW_SENTINEL = "TOP_SECRET_RAW_PROMPT_AND_OUTPUT"


def _result(index: int, category: str, status: ProbeStatus = ProbeStatus.PASSED) -> ProbeResult:
    probe = Probe(
        probe_id=f"probe-{index}",
        pack_id="generic_taxonomy",
        surface=ProbeSurface.CHAT,
        category=category,
        title=f"private {RAW_SENTINEL}",
        payload=ChatProbe(prompt=f"{RAW_SENTINEL}: prompt {index}"),
    )
    trace = ProbeTrace(
        transcript=[
            ConversationTurn(role="user", content=f"{RAW_SENTINEL}: prompt", turn_index=0),
            ConversationTurn(role="assistant", content=f"{RAW_SENTINEL}: output", turn_index=0),
        ],
        ended_at="2026-08-12T12:00:00Z",
    )
    return ProbeResult(
        probe=probe,
        status=status,
        output_text=f"{RAW_SENTINEL}: detailed output",
        trace=trace,
        score={"combined": 0.0},
    )


def _campaign_result() -> CampaignResult:
    campaign = Campaign(
        campaign_id="campaign-test",
        name="request-bound test",
        target=TargetRef(
            provider="echo",
            model="echo",
            system_prompt=f"{RAW_SENTINEL}: private system prompt",
        ),
        probes=[],
        pack_ids=["generic_taxonomy"],
    )
    result = CampaignResult(
        campaign=campaign,
        results=[
            _result(1, "prompt_injection"),
            _result(2, "pii_extraction"),
            _result(3, "system_prompt_leakage"),
            _result(4, "tool_misuse"),
        ],
        started_at="2026-08-12T11:59:00Z",
    )
    result.completed_at = "2026-08-12T12:00:00Z"
    from campaign import CampaignSummary
    result.summary = CampaignSummary.from_results(result.results)
    return result


class TestRequirementV01(unittest.TestCase):
    def test_known_digest_and_object_order(self) -> None:
        requirement = load_requirement(FIXTURE)
        self.assertEqual(requirement_digest(requirement), VECTOR_DIGEST)
        reordered = dict(reversed(list(requirement.items())))
        self.assertEqual(requirement_digest(reordered), VECTOR_DIGEST)

    def test_buyer_signature_is_excluded_from_digest(self) -> None:
        requirement = load_requirement(FIXTURE)
        requirement["buyer_signature"] = {
            "algorithm": "Ed25519",
            "key_id": "buyer-test-key",
            "signed_digest": VECTOR_DIGEST,
            "signature": "A" * 86,
        }
        validate_requirement(requirement)
        self.assertEqual(requirement_digest(requirement), VECTOR_DIGEST)

    def test_numbers_and_noncanonical_nonce_fail_closed(self) -> None:
        requirement = load_requirement(FIXTURE)
        requirement["unexpected_number"] = 1
        with self.assertRaisesRegex(RequirementError, "numbers are not allowed"):
            validate_requirement(requirement)

        requirement = load_requirement(FIXTURE)
        requirement["nonce"]["value"] = "A" * 42
        with self.assertRaises(RequirementError):
            validate_requirement(requirement)

        requirement = load_requirement(FIXTURE)
        # The final character changes only unused base64 padding bits; a
        # decoder accepts it, but canonical unpadded base64url must reject it.
        requirement["nonce"]["value"] = requirement["nonce"]["value"][:-1] + "N"
        with self.assertRaisesRegex(RequirementError, "canonical unpadded"):
            validate_requirement(requirement)

    def test_schema_string_bounds_and_return_uri_fail_closed(self) -> None:
        requirement = load_requirement(FIXTURE)
        overlong_urn = "urn:aa:" + ("a" * 234)
        self.assertEqual(len(overlong_urn), 241)
        requirement["profile"]["id"] = overlong_urn
        with self.assertRaisesRegex(RequirementError, r"\$\.profile\.id"):
            validate_requirement(requirement)

        requirement = load_requirement(FIXTURE)
        requirement["accepted_receipt_formats"] = ["a/" + ("b" * 159)]
        with self.assertRaisesRegex(RequirementError, "accepted_receipt_formats"):
            validate_requirement(requirement)

        for invalid_uri in (
            "HTTPS://example.invalid/evidence/return",
            "https://example.invalid/evidence/return\nignored",
            "https://buyer:secret@example.invalid/evidence/return",
            "https://example.invalid:invalid/evidence/return",
        ):
            with self.subTest(uri=invalid_uri):
                requirement = load_requirement(FIXTURE)
                requirement["return"]["uri"] = invalid_uri
                with self.assertRaisesRegex(RequirementError, r"\$\.return\.uri"):
                    validate_requirement(requirement)


class TestSignedRequestReceipt(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp.name) / "results"
        self.requirement = load_requirement(FIXTURE)
        self.artifacts = build_receipt_artifacts(
            _campaign_result(),
            self.requirement,
            bytes(range(32)),
            self.output_dir,
            now=AT,
        )
        self.receipt = json.loads(self.artifacts.receipt.read_text(encoding="utf-8"))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _verify(
        self,
        *,
        receipt: Optional[dict] = None,
        requirement: Optional[dict] = None,
        artifact: Optional[Path] = None,
        trusted_public_key: Optional[str] = None,
        now: datetime = AT,
    ) -> dict:
        return verify_receipt(
            receipt or self.receipt,
            requirement or self.requirement,
            artifact_path=artifact or self.artifacts.redacted_evidence,
            trusted_public_key=trusted_public_key,
            now=now,
        )

    def _resign_mutated_pair(
        self, mutate, *, contradictory_io: bool = False
    ) -> tuple[dict, Path]:
        """Mutate both public documents, then create a fresh valid SDK signature."""
        receipt = copy.deepcopy(self.receipt)
        artifact = json.loads(
            self.artifacts.redacted_evidence.read_text(encoding="utf-8")
        )
        fulfillment = receipt["statement"]["fulfillment"]
        mutate(fulfillment, artifact)

        artifact_bytes = receipts_module._json_bytes(artifact)
        fulfillment["receipt_artifact"]["digest"] = receipts_module._sha256_bytes(
            artifact_bytes
        )
        statement_digest = receipts_module._statement_digest(receipt["statement"])
        receipt["statement_digest"] = statement_digest

        Glacis, _, _, _, sdk_version = receipts_module._sdk_imports()
        signer = Glacis(
            mode="offline",
            signing_seed=bytes(range(32)),
            db_path=Path(self.temp.name) / "resign.db",
        )
        try:
            attestation = signer.attest(
                service_id="glacis-autoredteam",
                operation_type="red-team-run-receipt",
                input={
                    "requirement_digest": (
                        "sha256:" + ("0" * 64)
                        if contradictory_io
                        else fulfillment["requirement_digest"]
                    )
                },
                output={
                    "statement_digest": (
                        "sha256:" + ("f" * 64)
                        if contradictory_io
                        else statement_digest
                    )
                },
                metadata={"adapter": receipts_module.RECEIPT_TYPE},
                control_plane_results={
                    "type": receipts_module.SDK_BINDING_TYPE,
                    "statement_digest": statement_digest,
                    "requirement_digest": fulfillment["requirement_digest"],
                },
            )
        finally:
            signer.close()
        receipt["signature"] = {
            "type": "glacis-python-offline-attestation",
            "sdk_version": sdk_version,
            "attestation": attestation.model_dump(exclude_none=True),
        }
        artifact_path = Path(self.temp.name) / "resigned-artifact.json"
        artifact_path.write_bytes(artifact_bytes)
        return receipt, artifact_path

    def test_valid_signature_request_binding_and_raw_exclusion(self) -> None:
        result = self._verify(
            trusted_public_key=self.receipt["signature"]["attestation"]["public_key"]
        )
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["status"], "requirement-fulfilled")
        self.assertEqual(result["current_revocation_status"], "unknown-offline")
        self.assertIn("key signed this claim", result["signature_explanation"])
        self.assertIn("does not independently prove", result["signature_explanation"])

        private_text = self.artifacts.private_findings.read_text(encoding="utf-8")
        shareable_text = self.artifacts.redacted_evidence.read_text(encoding="utf-8")
        receipt_text = self.artifacts.receipt.read_text(encoding="utf-8")
        self.assertIn(RAW_SENTINEL, private_text)
        self.assertNotIn(RAW_SENTINEL, shareable_text)
        self.assertNotIn(RAW_SENTINEL, receipt_text)
        self.assertEqual(self.artifacts.private_findings.stat().st_mode & 0o777, 0o600)

    def test_cli_json_is_an_allowlisted_public_verification_view(self) -> None:
        receipt_path = self.artifacts.receipt
        signer_public_key = self.receipt["signature"]["attestation"]["public_key"]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli.main(
                [
                    "verify",
                    str(receipt_path),
                    "--requirement",
                    str(FIXTURE),
                    "--artifact",
                    str(self.artifacts.redacted_evidence),
                    "--trusted-public-key",
                    signer_public_key,
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 0)
        public_result = json.loads(stdout.getvalue())
        self.assertEqual(
            set(public_result),
            {
                "valid",
                "status",
                "checks",
                "signer_key_fingerprint",
                "key_trust",
                "witness_status",
                "current_revocation_status",
                "signature_explanation",
                "buyer_signature_verification",
                "compatibility",
                "content_check_scope",
            },
        )
        self.assertNotIn("signature_error", public_result)
        self.assertNotIn("artifact_error", public_result)
        self.assertNotIn(signer_public_key, stdout.getvalue())
        self.assertRegex(public_result["signer_key_fingerprint"], r"^sha256:[0-9a-f]{64}$")
        self.assertTrue(public_result["checks"]["trusted_public_key_match"])

        wrong_key_stdout = io.StringIO()
        with contextlib.redirect_stdout(wrong_key_stdout):
            wrong_key_exit = cli.main(
                [
                    "verify",
                    str(receipt_path),
                    "--requirement",
                    str(FIXTURE),
                    "--artifact",
                    str(self.artifacts.redacted_evidence),
                    "--trusted-public-key",
                    "0" * 64,
                    "--json",
                ]
            )
        wrong_key_result = json.loads(wrong_key_stdout.getvalue())
        self.assertEqual(wrong_key_exit, 1)
        self.assertEqual(wrong_key_result["status"], "wrong-key")
        self.assertFalse(wrong_key_result["checks"]["trusted_public_key_match"])
        self.assertNotIn("signature_error", wrong_key_result)
        self.assertNotIn("artifact_error", wrong_key_result)

    def test_requirement_digest_and_nonce_mismatch(self) -> None:
        different = copy.deepcopy(self.requirement)
        different["nonce"]["value"] = "csVOXQRC0m1YxmlQOrcseaV9DOKEiIcYd6E7LUrzi_0"
        result = self._verify(requirement=different)
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "requirement-mismatch")
        self.assertFalse(result["checks"]["requirement_digest_and_fields_match"])

    def test_wrong_key_and_statement_tamper(self) -> None:
        wrong_key = "00" * 32
        result = self._verify(trusted_public_key=wrong_key)
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "wrong-key")

        tampered = copy.deepcopy(self.receipt)
        tampered["statement"]["fulfillment"]["subject"]["id"] = "urn:tampered:other-build"
        result = self._verify(receipt=tampered)
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "invalid-signature-or-tamper")
        self.assertFalse(result["checks"]["statement_digest_match"])

    def test_unsigned_sdk_version_copy_tamper_fails_binding(self) -> None:
        tampered = copy.deepcopy(self.receipt)
        tampered["signature"]["sdk_version"] = "0.8.99"
        result = self._verify(receipt=tampered)
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "invalid-signature-or-tamper")
        self.assertFalse(result["checks"]["signed_sdk_version_copy_match"])

        tampered = copy.deepcopy(self.receipt)
        tampered["signature"]["attestation"]["is_offline"] = False
        result = self._verify(receipt=tampered)
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "invalid-signature-or-tamper")
        self.assertFalse(result["checks"]["sdk_attestation_context_match"])

        tampered = copy.deepcopy(self.receipt)
        tampered["signature"]["attestation"]["raw_prompt"] = (
            "SECRET_UNSIGNED_EXTENSION"
        )
        result = self._verify(receipt=tampered)
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "invalid-signature-or-tamper")
        self.assertFalse(result["checks"]["sdk_attestation_shape_match"])
        self.assertFalse(result["checks"]["receipt_prohibited_field_names_absent"])

    def test_freshly_resigned_schema_invalid_fulfillments_fail_closed(self) -> None:
        mutations = {
            "empty category outcomes": lambda fulfillment, artifact: (
                fulfillment["evidence_summary"].__setitem__("category_outcomes", []),
                artifact.__setitem__("category_outcomes", []),
            ),
            "duplicate omitted content": lambda fulfillment, artifact: (
                fulfillment["disclosure"]["omitted_content"].append("raw-prompts"),
                artifact["disclosure"]["omitted_content"].append("raw-prompts"),
            ),
            "object scope": lambda fulfillment, artifact: (
                fulfillment["evidence_summary"].__setitem__("scope", {"raw": "x"}),
                artifact.__setitem__("scope", {"raw": "x"}),
            ),
            "malformed private digest": lambda fulfillment, artifact: (
                fulfillment["evidence_summary"].__setitem__(
                    "private_report_digest", "sha256:not-a-digest"
                ),
                artifact.__setitem__(
                    "private_report_digest", "sha256:not-a-digest"
                ),
            ),
            "freshness extra field": lambda fulfillment, artifact: (
                fulfillment["freshness_result"].__setitem__("extra", "x"),
                artifact["freshness_result"].__setitem__("extra", "x"),
            ),
            "object evidence reference": lambda fulfillment, artifact: (
                fulfillment["assurance_results"][0].__setitem__(
                    "evidence_ref", {"path": "signature.attestation"}
                ),
                artifact["assurance_results"][0].__setitem__(
                    "evidence_ref", {"path": "signature.attestation"}
                ),
            ),
            "overlong observed value": lambda fulfillment, artifact: (
                fulfillment["criteria_results"][0].__setitem__("observed", "x" * 201),
                artifact["criteria_results"][0].__setitem__("observed", "x" * 201),
            ),
            "disclosure extra field": lambda fulfillment, artifact: (
                fulfillment["disclosure"].__setitem__("extra", "x"),
                artifact["disclosure"].__setitem__("extra", "x"),
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                receipt, artifact_path = self._resign_mutated_pair(mutate)
                with self.assertRaises(ReceiptError):
                    self._verify(receipt=receipt, artifact=artifact_path)

    def test_freshly_resigned_wrong_category_set_is_incomplete(self) -> None:
        def mutate(fulfillment, artifact) -> None:
            replacement = [{"category": "other-test", "outcome": "pass"}]
            fulfillment["evidence_summary"]["category_outcomes"] = replacement
            artifact["category_outcomes"] = copy.deepcopy(replacement)

        receipt, artifact_path = self._resign_mutated_pair(mutate)
        result = self._verify(receipt=receipt, artifact=artifact_path)
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["checks"]["category_outcomes_match"])

    def test_fresh_signature_with_contradictory_sdk_io_fails_closed(self) -> None:
        receipt, artifact_path = self._resign_mutated_pair(
            lambda fulfillment, artifact: None,
            contradictory_io=True,
        )
        result = self._verify(receipt=receipt, artifact=artifact_path)
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "invalid-signature-or-tamper")
        self.assertFalse(result["checks"]["sdk_evidence_hash_match"])
        self.assertTrue(result["checks"]["sdk_signed_binding_match"])

    def test_freshly_resigned_unsupported_assurance_is_incomplete(self) -> None:
        def mutate(fulfillment, artifact) -> None:
            for container in (fulfillment, artifact):
                result = next(
                    item
                    for item in container["assurance_results"]
                    if item["predicate"] == "organization-bound"
                )
                result["result"] = "satisfied"
                result["evidence_ref"] = "signature.attestation"

        receipt, artifact_path = self._resign_mutated_pair(mutate)
        result = self._verify(receipt=receipt, artifact=artifact_path)
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(result["checks"]["minimum_assurance_satisfied"])
        self.assertFalse(result["checks"]["no_unsupported_assurance_claim"])

        receipt_path = Path(self.temp.name) / "unsupported-assurance-receipt.json"
        receipt_path.write_bytes(receipts_module._json_bytes(receipt))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli.main(
                [
                    "verify",
                    str(receipt_path),
                    "--requirement",
                    str(FIXTURE),
                    "--artifact",
                    str(artifact_path),
                    "--trusted-public-key",
                    receipt["signature"]["attestation"]["public_key"],
                ]
            )
        self.assertEqual(exit_code, 1)
        self.assertIn("Fulfillment: incomplete", stdout.getvalue())
        self.assertNotIn("Fulfillment: requirement-fulfilled", stdout.getvalue())

    def test_artifact_one_byte_equivalent_tamper(self) -> None:
        tampered_artifact = Path(self.temp.name) / "tampered.json"
        original = self.artifacts.redacted_evidence.read_bytes()
        tampered_artifact.write_bytes(original + b" ")
        result = self._verify(artifact=tampered_artifact)
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "artifact-missing-or-mismatch")
        self.assertFalse(result["checks"]["artifact_digest_match"])

    def test_freshness_and_expiry(self) -> None:
        stale_at = datetime(2026, 8, 20, 12, 0, 1, tzinfo=timezone.utc)
        result = self._verify(now=stale_at)
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "stale")
        self.assertFalse(result["checks"]["freshness_window_satisfied"])

        expired_at = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = self._verify(now=expired_at)
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "expired")
        self.assertFalse(result["checks"]["request_unexpired"])

    def test_unsupported_minimum_assurance_is_explicitly_incomplete(self) -> None:
        requirement = copy.deepcopy(self.requirement)
        requirement["minimum_assurance_predicates"].append("execution-attested")
        artifacts = build_receipt_artifacts(
            _campaign_result(),
            requirement,
            bytes(reversed(range(32))),
            Path(self.temp.name) / "unsupported-assurance",
            now=AT,
        )
        self.assertFalse(artifacts.verification["valid"])
        self.assertEqual(artifacts.verification["status"], "incomplete")
        receipt = json.loads(artifacts.receipt.read_text(encoding="utf-8"))
        assurances = receipt["statement"]["fulfillment"]["assurance_results"]
        execution = next(item for item in assurances if item["predicate"] == "execution-attested")
        self.assertEqual(execution["result"], "not-evaluated")


class TestRequestModeCliPrivacy(unittest.TestCase):
    @unittest.skipUnless(hasattr(Path, "symlink_to"), "symlinks unavailable")
    def test_request_mode_output_root_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            outside = temp / "outside"
            outside.mkdir()
            output = temp / "share"
            output.symlink_to(outside, target_is_directory=True)
            key_path = temp / "issuer.key"
            generate_signing_key(key_path)
            args = cli.build_parser().parse_args(
                [
                    "run",
                    "--dry-run",
                    "--requirement",
                    str(FIXTURE),
                    "--signing-key",
                    str(key_path),
                    "--output-dir",
                    str(output),
                ]
            )
            with self.assertRaisesRegex(ReceiptError, "output root must not be a symlink"):
                cli.cmd_run(args)
            self.assertEqual(list(outside.iterdir()), [])

    @unittest.skipUnless(hasattr(Path, "symlink_to"), "symlinks unavailable")
    def test_sdk_signing_store_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            output = temp / "share"
            private = output / "private"
            private.mkdir(parents=True)
            outside = temp / "outside.db"
            (private / "glacis-offline.db").symlink_to(outside)
            with self.assertRaisesRegex(ReceiptError, "signing store must be a new regular file"):
                build_receipt_artifacts(
                    _campaign_result(),
                    load_requirement(FIXTURE),
                    bytes(range(32)),
                    output,
                    now=AT,
                )
            self.assertFalse(outside.exists())

    def test_raw_run_is_confined_to_private_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            output = temp / "share"
            key_path = temp / "issuer.key"
            requirement_path = temp / "requirement.json"
            generate_signing_key(key_path)

            requirement = load_requirement(FIXTURE)
            now = datetime.now(timezone.utc).replace(microsecond=0)
            created = now - timedelta(minutes=1)
            requirement["created_at"] = created.strftime("%Y-%m-%dT%H:%M:%SZ")
            requirement["freshness"]["not_before"] = requirement["created_at"]
            requirement["return"]["due_at"] = (
                now + timedelta(days=1)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            requirement["return"]["expires_at"] = (
                now + timedelta(days=2)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            requirement["criteria"] = [
                criterion
                for criterion in requirement["criteria"]
                if criterion["metric"] == "test-accounting-status"
            ]
            requirement_path.write_text(
                json.dumps(requirement, indent=2) + "\n", encoding="utf-8"
            )

            cli_sentinel = "CLI_PRIVATE_RAW_SENTINEL"
            run_args = [
                "run", "--dry-run", "--quiet", "--max-probes", "5",
                "--system-prompt", cli_sentinel,
                "--requirement", str(requirement_path),
                "--signing-key", str(key_path),
                "--output-dir", str(output),
            ]
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(run_args)
            self.assertEqual(exit_code, 0, stdout.getvalue())

            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"private", "receipt.json", "redacted_evidence.json"},
            )
            for shareable in (output / "receipt.json", output / "redacted_evidence.json"):
                self.assertNotIn(cli_sentinel, shareable.read_text(encoding="utf-8"))
                self.assertEqual(shareable.stat().st_mode & 0o777, 0o644)

            private = output / "private"
            run_dir = private / "run"
            self.assertEqual(private.stat().st_mode & 0o777, 0o700)
            self.assertEqual(run_dir.stat().st_mode & 0o777, 0o700)
            expected_run_files = {
                "campaign_manifest.json", "campaign_result.json",
                "campaign_state.json", "probe_results.jsonl",
            }
            self.assertEqual(
                {path.name for path in run_dir.iterdir()}, expected_run_files
            )
            for private_file in run_dir.iterdir():
                self.assertEqual(private_file.stat().st_mode & 0o777, 0o600)
            self.assertIn(
                cli_sentinel,
                (run_dir / "campaign_manifest.json").read_text(encoding="utf-8"),
            )

            unexpected = output / "stale-raw.json"
            unexpected.write_text(cli_sentinel, encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(run_args)
            self.assertEqual(exit_code, 2)
            self.assertIn("choose a clean output directory", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
