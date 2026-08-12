"""Public AutoRedTeam receipt adapter with a bounded disclosure structure.

The adapter binds an experimental AI Evidence Requirement fulfillment claim to
the public Glacis Python SDK's offline Ed25519 attestation.  The resulting JSON
is intentionally labeled as a detached experimental adapter.  It is not a
canonical Glacis Notary receipt and carries no independent witness predicate.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from autoredteam import __version__
from autoredteam.requirements import (
    ASSURANCE_PREDICATES,
    FULFILLMENT_SCHEMA,
    FULFILLMENT_TYPE,
    PROHIBITED_CONTENT,
    SPEC_VERSION,
    RequirementError,
    canonical_json_v01,
    load_json_object,
    parse_duration,
    parse_timestamp,
    requirement_digest,
    validate_autoredteam_requirement,
    validate_requirement,
)


RECEIPT_SCHEMA = "urn:glacis:experimental:autoredteam:schema:red-team-run-receipt:0.1"
RECEIPT_TYPE = "urn:glacis:experimental:autoredteam:red-team-run-receipt:0.1"
REDACTED_SCHEMA = "urn:glacis:experimental:autoredteam:schema:redacted-evidence:0.1"
REDACTED_TYPE = "urn:glacis:experimental:autoredteam:redacted-evidence:0.1"
PRIVATE_SCHEMA = "urn:glacis:experimental:autoredteam:schema:private-findings:0.1"
PRIVATE_TYPE = "urn:glacis:experimental:autoredteam:private-findings:0.1"
SDK_BINDING_TYPE = "urn:glacis:experimental:autoredteam:sdk-offline-binding:0.1"
STATEMENT_DIGEST_DOMAIN = b"AUTOREDTEAM-RED-TEAM-RUN-RECEIPT/0.1\n"
KEY_SCHEMA = "urn:glacis:experimental:autoredteam:schema:ed25519-signing-seed:0.1"
KEY_TYPE = "urn:glacis:experimental:autoredteam:ed25519-signing-seed:0.1"
REDACTED_FORMAT = "urn:glacis:experimental:autoredteam:redacted-evidence:0.1"

SELF_SIGNED_EXPLANATION = (
    "The displayed Ed25519 key signed this claim. The signature does not "
    "independently prove that the run executed or that the claimed time is accurate."
)
ADAPTER_BLOCKER = (
    "The public Glacis SDK 0.8.x Attestation model has no first-class AI Evidence "
    "Requirement fulfillment field. This adapter therefore signs a domain-separated "
    "statement digest through the SDK's signed control_plane_results extension seam."
)
ADAPTER_LIMITATIONS = [
    SELF_SIGNED_EXPLANATION,
    "No independent witness, execution observer, organization binding, or time anchor is present.",
    "Subject, build, and test-profile values are issuer declarations copied from the request; this adapter does not independently discover them.",
    "This is an experimental detached adapter, not a canonical Glacis Notary receipt.",
    "Optional buyer signature structure and digest binding are checked, but buyer key trust and signature validity are not verified.",
    "Current revocation or signer status is unknown offline unless a fresh trusted status object is checked separately.",
]

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PUBLIC_KEY_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")

# Public mapping from the request profile's test identifiers to the current
# generic-taxonomy categories. Unknown identifiers are never guessed: they are
# reported as unavailable and make the fulfillment incomplete unless the buyer
# explicitly allowed the omission.
TEST_CATEGORY_ALIASES: dict[str, set[str]] = {
    "prompt-injection": {"prompt_injection", "indirect_injection"},
    "sensitive-data-exfiltration": {"pii_extraction", "system_prompt_leakage"},
    "tool-abuse": {"tool_misuse"},
}

_PROHIBITED_SHAREABLE_KEYS = {
    "prompt",
    "prompts",
    "raw_prompt",
    "raw_prompts",
    "raw_output",
    "raw_outputs",
    "output_text",
    "prompt_text",
    "response",
    "responses",
    "transcript",
    "full_transcript",
    "detailed_finding",
    "detailed_findings",
    "findings",
}


class ReceiptError(ValueError):
    """Receipt creation or verification could not complete honestly."""


@dataclass
class ReceiptArtifacts:
    private_findings: Path
    redacted_evidence: Path
    receipt: Path
    signing_store: Path
    verification: dict[str, Any]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: Optional[datetime]) -> datetime:
    result = value or _utc_now()
    if result.tzinfo is None:
        raise ReceiptError("verification time must be timezone-aware")
    return result.astimezone(timezone.utc)


def _parse_runtime_timestamp(value: Union[str, datetime]) -> datetime:
    if isinstance(value, datetime):
        return _as_utc(value)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ReceiptError(f"invalid campaign timestamp: {value!r}") from exc
    return _as_utc(parsed)


def _format_timestamp(value: datetime) -> str:
    return _as_utc(value).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ReceiptError(f"cannot hash artifact {path}: {exc}") from exc
    return "sha256:" + digest.hexdigest()


def _statement_digest(statement: Mapping[str, Any]) -> str:
    canonical = canonical_json_v01(statement).encode("utf-8")
    return _sha256_bytes(STATEMENT_DIGEST_DOMAIN + canonical)


def _write_bytes(path: Path, content: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags, mode)
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
    except OSError as exc:
        raise ReceiptError(f"cannot write {path}: {exc}") from exc


def _sdk_imports() -> tuple[Any, Any, Any, Any, str]:
    try:
        from glacis import Glacis
        from glacis.crypto import hash_payload
        from glacis.models import Attestation
        from glacis.verify import verify_offline
    except ImportError as exc:
        raise ReceiptError(
            "the request-bound receipt path requires the declared public dependency "
            "glacis>=0.8.1,<0.9; no unsigned fallback was used"
        ) from exc
    try:
        sdk_version = importlib_metadata.version("glacis")
    except importlib_metadata.PackageNotFoundError as exc:
        raise ReceiptError("cannot determine installed Glacis SDK version") from exc
    version_match = re.fullmatch(r"0\.8\.([0-9]+)(?:[.+-].*)?", sdk_version)
    if version_match is None or int(version_match.group(1)) < 1:
        raise ReceiptError(
            f"unsupported Glacis SDK {sdk_version}; expected >=0.8.1,<0.9 and "
            "no compatibility fallback was used"
        )
    return Glacis, Attestation, verify_offline, hash_payload, sdk_version


def generate_signing_key(
    output_path: Union[str, Path],
    public_key_path: Optional[Union[str, Path]] = None,
) -> str:
    """Generate a new local Ed25519 seed without overwriting an existing key."""
    Glacis, _, _, _, _ = _sdk_imports()
    seed = os.urandom(32)
    encoded = base64.urlsafe_b64encode(seed).decode("ascii").rstrip("=")
    key_doc = {
        "$schema": KEY_SCHEMA,
        "type": KEY_TYPE,
        "spec_version": SPEC_VERSION,
        "encoding": "base64url-256",
        "seed": encoded,
    }
    key_path = Path(output_path)
    if key_path.exists():
        raise ReceiptError(f"refusing to overwrite existing signing key: {key_path}")
    if public_key_path is not None and Path(public_key_path).exists():
        raise ReceiptError(
            f"refusing to overwrite existing public key: {Path(public_key_path)}"
        )
    key_path.parent.mkdir(parents=True, exist_ok=True)

    # Derive through the same public SDK path that will later sign receipts.
    with tempfile.TemporaryDirectory(prefix=".autoredteam-keygen-", dir=key_path.parent) as scratch_dir:
        scratch = Path(scratch_dir) / "glacis-offline.db"
        client = Glacis(mode="offline", signing_seed=seed, db_path=scratch)
        try:
            key_receipt = client.attest(
                service_id="glacis-autoredteam-keygen",
                operation_type="key-generation-check",
                input={"purpose": "derive-public-key"},
                output={"status": "generated"},
            )
            public_key = key_receipt.public_key
        finally:
            client.close()
    if not isinstance(public_key, str) or not _PUBLIC_KEY_RE.fullmatch(public_key):
        raise ReceiptError("Glacis SDK did not derive a valid Ed25519 public key")

    _write_exclusive(key_path, _json_bytes(key_doc), 0o600)
    if public_key_path is not None:
        _write_exclusive(Path(public_key_path), (public_key + "\n").encode("ascii"), 0o644)
    return public_key


def _write_exclusive(path: Path, content: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags, mode)
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
    except OSError as exc:
        raise ReceiptError(f"cannot create {path}: {exc}") from exc


def load_signing_seed(path: Union[str, Path]) -> bytes:
    key_path = Path(path)
    if os.name == "posix":
        try:
            mode = key_path.stat().st_mode & 0o777
        except OSError as exc:
            raise ReceiptError(f"cannot inspect signing key {key_path}: {exc}") from exc
        if mode & 0o077:
            raise ReceiptError(
                f"signing key permissions are too broad ({mode:04o}); "
                "group/other access must be disabled"
            )
    key = load_json_object(key_path)
    expected_keys = {"$schema", "type", "spec_version", "encoding", "seed"}
    if set(key) != expected_keys:
        raise ReceiptError("signing key has missing or unsupported fields")
    expected = {
        "$schema": KEY_SCHEMA,
        "type": KEY_TYPE,
        "spec_version": SPEC_VERSION,
        "encoding": "base64url-256",
    }
    for field, value in expected.items():
        if key[field] != value:
            raise ReceiptError(f"signing key {field} is unsupported")
    encoded = key["seed"]
    if not isinstance(encoded, str) or not re.fullmatch(r"[A-Za-z0-9_-]{43}", encoded):
        raise ReceiptError("signing key seed must be canonical unpadded base64url-256")
    try:
        seed = base64.urlsafe_b64decode(encoded + "=")
    except Exception as exc:
        raise ReceiptError("signing key seed is invalid base64url") from exc
    if len(seed) != 32:
        raise ReceiptError("signing key seed must decode to 32 bytes")
    canonical_seed = base64.urlsafe_b64encode(seed).rstrip(b"=").decode("ascii")
    if canonical_seed != encoded:
        raise ReceiptError("signing key seed must be canonical unpadded base64url-256")
    return seed


def load_public_key(value: str) -> str:
    candidate = value
    path = Path(value)
    if path.is_file():
        try:
            candidate = path.read_text(encoding="ascii").strip()
        except OSError as exc:
            raise ReceiptError(f"cannot read trusted public key {path}: {exc}") from exc
    candidate = candidate.lower()
    if not _PUBLIC_KEY_RE.fullmatch(candidate):
        raise ReceiptError("trusted public key must be 32-byte lowercase Ed25519 hex")
    return candidate


def _result_status(result: Any) -> str:
    status = getattr(result, "status", "")
    return getattr(status, "value", status)


def _result_category(result: Any) -> str:
    return str(getattr(getattr(result, "probe", None), "category", ""))


def _category_mapping(test_id: str) -> set[str]:
    if test_id in TEST_CATEGORY_ALIASES:
        return TEST_CATEGORY_ALIASES[test_id]
    normalized = test_id.replace("-", "_")
    try:
        from attack import ATTACK_CATEGORIES
    except ImportError:
        return set()
    return {normalized} if normalized in ATTACK_CATEGORIES else set()


def _analyze_tests(
    results: Sequence[Any], requirement: Mapping[str, Any]
) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]], str]:
    requested = requirement["test_profile"]["tests"]
    allowed_by_id = {
        item["test_id"]: dict(item)
        for item in requirement["completeness"]["allowed_omissions"]
    }
    completed: list[str] = []
    omissions: list[dict[str, str]] = []
    outcomes: list[dict[str, str]] = []

    for test_id in requested:
        mapping = _category_mapping(test_id)
        matching = [result for result in results if _result_category(result) in mapping]
        statuses = [_result_status(result) for result in matching]
        has_error = any(status in {"error", "skipped"} for status in statuses)
        completed_test = bool(mapping and matching) and not has_error
        if completed_test:
            completed.append(test_id)
        else:
            allowed = allowed_by_id.get(test_id)
            if allowed:
                omissions.append(allowed)
            else:
                explanation = (
                    "AutoRedTeam has no declared mapping for this test identifier."
                    if not mapping
                    else "The mapped AutoRedTeam probe did not complete without errors."
                )
                omissions.append({
                    "test_id": test_id,
                    "reason_code": "environment-unavailable",
                    "explanation": explanation,
                })
        if any(status == "bypassed" for status in statuses):
            outcome = "fail"
        elif completed_test:
            outcome = "pass"
        else:
            outcome = "not-tested"
        outcomes.append({"category": test_id, "outcome": outcome})

    accounted = set(completed) | {
        omission["test_id"]
        for omission in omissions
        if omission == allowed_by_id.get(omission["test_id"])
    }
    completeness_result = "complete" if accounted == set(requested) else "incomplete"
    return completed, omissions, outcomes, completeness_result


def _critical_finding_count(results: Sequence[Any]) -> int:
    try:
        from attack import ATTACK_CATEGORIES
    except ImportError as exc:
        raise ReceiptError(
            "cannot evaluate critical-finding-count because the public attack taxonomy is unavailable"
        ) from exc
    count = 0
    for result in results:
        category = _result_category(result)
        severity = ATTACK_CATEGORIES.get(category, {}).get("severity_class")
        if severity == "critical" and _result_status(result) == "bypassed":
            count += 1
    return count


def _compare(observed: str, operator: str, expected: str) -> Optional[bool]:
    if operator == "equals":
        return observed == expected
    if operator == "not-equals":
        return observed != expected
    if operator == "contains":
        return expected in observed
    try:
        left = Decimal(observed)
        right = Decimal(expected)
    except InvalidOperation:
        return None
    if operator == "less-than-or-equal":
        return left <= right
    if operator == "greater-than-or-equal":
        return left >= right
    return None


def _criteria_results(
    results: Sequence[Any], requirement: Mapping[str, Any], completeness_result: str
) -> list[dict[str, str]]:
    known = {
        "critical-finding-count": str(_critical_finding_count(results)),
        "test-accounting-status": completeness_result,
    }
    evaluated: list[dict[str, str]] = []
    for criterion in requirement["criteria"]:
        metric = criterion["metric"]
        if metric not in known:
            evaluated.append({
                "criterion_id": criterion["id"],
                "status": "not-evaluated",
                "observed": "unsupported-metric",
            })
            continue
        observed = known[metric]
        comparison = _compare(observed, criterion["operator"], criterion["expected"])
        evaluated.append({
            "criterion_id": criterion["id"],
            "status": "pass" if comparison is True else "fail" if comparison is False else "not-evaluated",
            "observed": observed,
        })
    return evaluated


def _assurance_results(requirement: Mapping[str, Any]) -> list[dict[str, str]]:
    minimum = set(requirement["minimum_assurance_predicates"])
    # Report every registered predicate so unsupported assurance is visible,
    # not merely absent. Only issuer-signed is provided by this adapter.
    ordered = [
        "issuer-signed",
        "organization-bound",
        "time-anchored",
        "execution-attested",
        "independently-observed",
    ]
    results: list[dict[str, str]] = []
    for predicate in ordered:
        if predicate == "issuer-signed":
            results.append({
                "predicate": predicate,
                "result": "satisfied",
                "evidence_ref": "signature.attestation",
            })
        else:
            results.append({
                "predicate": predicate,
                "result": "not-evaluated" if predicate in minimum else "not-evaluated",
            })
    return results


def _private_findings_document(campaign_result: Any, req_digest: str) -> dict[str, Any]:
    return {
        "$schema": PRIVATE_SCHEMA,
        "type": PRIVATE_TYPE,
        "spec_version": SPEC_VERSION,
        "handling": "private-local-only",
        "warning": "Contains raw prompts, outputs, transcripts, and detailed findings. Do not share.",
        "requirement_digest": req_digest,
        "campaign": campaign_result.campaign.to_dict(),
        "started_at": campaign_result.started_at,
        "completed_at": campaign_result.completed_at,
        "results": [result.to_dict() for result in campaign_result.results],
    }


def _summary_strings(campaign_result: Any) -> dict[str, str]:
    summary = campaign_result.summary
    if summary is None:
        return {
            "total_probes": "0", "bypassed": "0", "blocked": "0",
            "passed": "0", "errors": "0", "skipped": "0",
        }
    return {
        "total_probes": str(summary.total_probes),
        "bypassed": str(summary.bypassed),
        "blocked": str(summary.blocked),
        "passed": str(summary.passed),
        "errors": str(summary.errors),
        "skipped": str(summary.skipped),
    }


def _build_redacted_evidence(
    campaign_result: Any,
    requirement: Mapping[str, Any],
    req_digest: str,
    private_digest: str,
    completed: list[str],
    omissions: list[dict[str, str]],
    outcomes: list[dict[str, str]],
    completeness_result: str,
    criteria: list[dict[str, str]],
    assurances: list[dict[str, str]],
    evidence_at: str,
    freshness_result: str,
    request_unexpired: bool,
) -> dict[str, Any]:
    subject = requirement["subject"]
    scope = (
        f"AutoRedTeam run declared against {subject['id']} at build commitment "
        f"{subject['build_digest']}."
    )
    method = (
        "The declared test identifiers were mapped to completed AutoRedTeam "
        "generic-taxonomy probe categories; unsupported or errored mappings are omissions."
    )
    minimum = set(requirement["minimum_assurance_predicates"])
    satisfied = {
        item["predicate"] for item in assurances if item["result"] == "satisfied"
    }
    state = "fulfilled" if (
        completeness_result == "complete"
        and all(item["status"] == "pass" for item in criteria)
        and freshness_result == "within-window"
        and request_unexpired
        and minimum <= satisfied
    ) else "incomplete"
    return {
        "$schema": REDACTED_SCHEMA,
        "type": REDACTED_TYPE,
        "spec_version": SPEC_VERSION,
        "handling": "shareable-content-minimized",
        "fulfillment_state": state,
        "requirement_digest": req_digest,
        "request_id": requirement["request_id"],
        "subject": dict(subject),
        "test_profile": dict(requirement["test_profile"]),
        "tool": {"name": "glacis-autoredteam", "version": __version__},
        "run": {
            "started_at": _format_timestamp(_parse_runtime_timestamp(campaign_result.started_at)),
            "completed_at": evidence_at,
            "packs": list(campaign_result.campaign.pack_ids),
            "summary": _summary_strings(campaign_result),
        },
        "scope": scope,
        "method": method,
        "category_outcomes": outcomes,
        "completeness": {
            "rule": requirement["completeness"]["rule"],
            "completed_tests": completed,
            "omissions": omissions,
            "result": completeness_result,
        },
        "criteria_results": criteria,
        "freshness_result": {
            "basis": requirement["freshness"]["basis"],
            "evidence_at": evidence_at,
            "declared_result": freshness_result,
        },
        "assurance_results": assurances,
        "private_report_digest": private_digest,
        "disclosure": {
            "mode": "content-minimized",
            "omitted_content": ["raw-prompts", "raw-outputs", "detailed-findings"],
        },
        "limitations": list(ADAPTER_LIMITATIONS),
    }


def _build_fulfillment_claim(
    requirement: Mapping[str, Any],
    req_digest: str,
    redacted_digest: str,
    redacted: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "$schema": FULFILLMENT_SCHEMA,
        "type": FULFILLMENT_TYPE,
        "spec_version": SPEC_VERSION,
        "request_id": requirement["request_id"],
        "requirement_digest": req_digest,
        "nonce": dict(requirement["nonce"]),
        "profile": dict(requirement["profile"]),
        "receipt_claim_type": requirement["receipt_claim_type"],
        "subject": dict(requirement["subject"]),
        "test_profile": dict(requirement["test_profile"]),
        "completeness": dict(redacted["completeness"]),
        "criteria_results": list(redacted["criteria_results"]),
        "freshness_result": dict(redacted["freshness_result"]),
        "assurance_results": list(redacted["assurance_results"]),
        "receipt_artifact": {
            "digest": redacted_digest,
            "media_type": "application/json",
            "format": REDACTED_FORMAT,
        },
        "evidence_summary": {
            "scope": redacted["scope"],
            "method": redacted["method"],
            "category_outcomes": list(redacted["category_outcomes"]),
            "private_report_digest": redacted["private_report_digest"],
        },
        "disclosure": dict(redacted["disclosure"]),
    }


def build_receipt_artifacts(
    campaign_result: Any,
    requirement: Mapping[str, Any],
    signing_seed: bytes,
    output_dir: Union[str, Path],
    *,
    now: Optional[datetime] = None,
) -> ReceiptArtifacts:
    """Create private findings, redacted evidence, and a signed receipt."""
    requirement = validate_requirement(requirement)
    validate_autoredteam_requirement(requirement)
    if len(signing_seed) != 32:
        raise ReceiptError("signing seed must be exactly 32 bytes")
    if not getattr(campaign_result, "completed_at", "") or campaign_result.summary is None:
        raise ReceiptError("campaign result must be finalized before receipt creation")

    completed_at = _parse_runtime_timestamp(campaign_result.completed_at)
    evidence_at = _format_timestamp(completed_at)
    verification_time = _as_utc(now)
    not_before = parse_timestamp(requirement["freshness"]["not_before"], "$.freshness.not_before")
    expires_at = parse_timestamp(requirement["return"]["expires_at"], "$.return.expires_at")
    age = verification_time - completed_at
    fresh = (
        completed_at >= not_before
        and completed_at <= verification_time
        and timedelta(0) <= age <= parse_duration(
            requirement["freshness"]["max_age"], "$.freshness.max_age"
        )
        and completed_at < expires_at
    )
    freshness_result = "within-window" if fresh else "stale"
    request_unexpired = verification_time < expires_at
    req_digest = requirement_digest(requirement)

    completed, omissions, outcomes, completeness_result = _analyze_tests(
        campaign_result.results, requirement
    )
    criteria = _criteria_results(campaign_result.results, requirement, completeness_result)
    assurances = _assurance_results(requirement)

    private_doc = _private_findings_document(campaign_result, req_digest)
    private_bytes = _json_bytes(private_doc)
    private_digest = _sha256_bytes(private_bytes)

    redacted = _build_redacted_evidence(
        campaign_result,
        requirement,
        req_digest,
        private_digest,
        completed,
        omissions,
        outcomes,
        completeness_result,
        criteria,
        assurances,
        evidence_at,
        freshness_result,
        request_unexpired,
    )
    redacted_bytes = _json_bytes(redacted)
    redacted_digest = _sha256_bytes(redacted_bytes)
    fulfillment = _build_fulfillment_claim(requirement, req_digest, redacted_digest, redacted)
    Glacis, _, _, _, sdk_version = _sdk_imports()
    statement = {
        "adapter": {
            "type": "experimental-detached-adapter",
            "version": SPEC_VERSION,
            "canonical_glacis_receipt_compatibility": "not-claimed",
            "signing_path": "public-glacis-python-sdk-offline-attestation",
            "extension_seam": "signed-control-plane-results-statement-digest",
            "sdk_version": sdk_version,
            "blocker": ADAPTER_BLOCKER,
        },
        "fulfillment": fulfillment,
        "limitations": list(ADAPTER_LIMITATIONS),
    }
    statement_digest = _statement_digest(statement)

    out = Path(output_dir)
    private_dir = out / "private"
    if out.is_symlink() or private_dir.is_symlink():
        raise ReceiptError("request-mode output path must not be a symlink")
    private_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(private_dir, 0o700)
    signing_store = private_dir / "glacis-offline.db"
    store_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        store_flags |= os.O_NOFOLLOW
    try:
        store_descriptor = os.open(str(signing_store), store_flags, 0o600)
        os.fchmod(store_descriptor, 0o600)
        os.close(store_descriptor)
    except OSError as exc:
        raise ReceiptError(
            "request-mode SDK signing store must be a new regular file: "
            f"{exc}"
        ) from exc

    client = Glacis(mode="offline", signing_seed=signing_seed, db_path=signing_store)
    try:
        attestation = client.attest(
            service_id="glacis-autoredteam",
            operation_type="red-team-run-receipt",
            input={"requirement_digest": req_digest},
            output={"statement_digest": statement_digest},
            metadata={"adapter": RECEIPT_TYPE},
            control_plane_results={
                "type": SDK_BINDING_TYPE,
                "statement_digest": statement_digest,
                "requirement_digest": req_digest,
            },
        )
    except Exception as exc:
        raise ReceiptError(
            f"Glacis SDK offline signing failed; no unsigned fallback was used: {exc}"
        ) from exc
    finally:
        client.close()
    receipt_doc = {
        "$schema": RECEIPT_SCHEMA,
        "type": RECEIPT_TYPE,
        "spec_version": SPEC_VERSION,
        "statement": statement,
        "statement_digest": statement_digest,
        "signature": {
            "type": "glacis-python-offline-attestation",
            "sdk_version": sdk_version,
            "attestation": attestation.model_dump(exclude_none=True),
        },
    }

    private_path = private_dir / "findings.json"
    redacted_path = out / "redacted_evidence.json"
    receipt_path = out / "receipt.json"
    _write_bytes(private_path, private_bytes, 0o600)
    probe_results_path = out / "probe_results.jsonl"
    if probe_results_path.exists():
        os.chmod(probe_results_path, 0o600)
    _write_bytes(redacted_path, redacted_bytes, 0o644)
    _write_bytes(receipt_path, _json_bytes(receipt_doc), 0o644)

    verification = verify_receipt(
        receipt_doc,
        requirement,
        artifact_path=redacted_path,
        now=verification_time,
    )
    return ReceiptArtifacts(
        private_findings=private_path,
        redacted_evidence=redacted_path,
        receipt=receipt_path,
        signing_store=signing_store,
        verification=verification,
    )


def _prohibited_key_paths(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            child_path = f"{path}.{key}"
            if normalized in _PROHIBITED_SHAREABLE_KEYS:
                found.append(child_path)
            found.extend(_prohibited_key_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_prohibited_key_paths(child, f"{path}[{index}]"))
    return found


def _contract_object(
    value: Any,
    path: str,
    required: set[str],
    optional: Optional[set[str]] = None,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ReceiptError(f"{path}: expected an object")
    allowed = required | (optional or set())
    if set(value) != required and not (required <= set(value) <= allowed):
        raise ReceiptError(f"{path}: missing or unsupported fields")
    return value


def _contract_string(
    value: Any, path: str, *, minimum: int = 1, maximum: int = 500
) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        raise ReceiptError(
            f"{path}: expected a string with length {minimum}..{maximum}"
        )
    return value


def _contract_identifier(value: Any, path: str) -> str:
    text = _contract_string(value, path, maximum=96)
    if not _IDENTIFIER_RE.fullmatch(text):
        raise ReceiptError(f"{path}: invalid protocol identifier")
    return text


def _contract_digest(value: Any, path: str) -> str:
    text = _contract_string(value, path, maximum=71)
    if not _DIGEST_RE.fullmatch(text):
        raise ReceiptError(f"{path}: invalid SHA-256 digest")
    return text


def _validate_fulfillment_contract(fulfillment: Mapping[str, Any]) -> None:
    """Enforce the complete v0.1 fulfillment schema before trusting a signature."""
    completeness = _contract_object(
        fulfillment.get("completeness"),
        "$.fulfillment.completeness",
        {"rule", "completed_tests", "omissions", "result"},
    )
    if completeness["rule"] != "all-listed-tests-accounted-for":
        raise ReceiptError("$.fulfillment.completeness.rule: unsupported rule")
    completed = completeness["completed_tests"]
    omissions = completeness["omissions"]
    if not isinstance(completed, list) or not isinstance(omissions, list):
        raise ReceiptError("$.fulfillment.completeness: expected test arrays")
    completed_ids = [
        _contract_identifier(value, f"$.fulfillment.completeness.completed_tests[{index}]")
        for index, value in enumerate(completed)
    ]
    if len(set(completed_ids)) != len(completed_ids):
        raise ReceiptError("$.fulfillment.completeness.completed_tests: duplicate values")
    omission_ids: set[str] = set()
    for index, value in enumerate(omissions):
        path = f"$.fulfillment.completeness.omissions[{index}]"
        omission = _contract_object(
            value, path, {"test_id", "reason_code", "explanation"}
        )
        test_id = _contract_identifier(omission["test_id"], f"{path}.test_id")
        if test_id in omission_ids:
            raise ReceiptError(f"{path}.test_id: duplicate omission")
        omission_ids.add(test_id)
        if omission["reason_code"] not in {
            "not-applicable", "buyer-approved-exception", "environment-unavailable",
        }:
            raise ReceiptError(f"{path}.reason_code: unsupported omission reason")
        _contract_string(omission["explanation"], f"{path}.explanation")
    if completeness["result"] not in {"complete", "incomplete"}:
        raise ReceiptError("$.fulfillment.completeness.result: unsupported result")

    criteria = fulfillment.get("criteria_results")
    if not isinstance(criteria, list) or not criteria:
        raise ReceiptError("$.fulfillment.criteria_results: expected a non-empty array")
    criterion_ids: set[str] = set()
    for index, value in enumerate(criteria):
        path = f"$.fulfillment.criteria_results[{index}]"
        result = _contract_object(
            value, path, {"criterion_id", "status", "observed"}
        )
        criterion_id = _contract_identifier(result["criterion_id"], f"{path}.criterion_id")
        if criterion_id in criterion_ids:
            raise ReceiptError(f"{path}.criterion_id: duplicate result")
        criterion_ids.add(criterion_id)
        if result["status"] not in {"pass", "fail", "not-evaluated"}:
            raise ReceiptError(f"{path}.status: unsupported status")
        _contract_string(result["observed"], f"{path}.observed", maximum=200)

    freshness = _contract_object(
        fulfillment.get("freshness_result"),
        "$.fulfillment.freshness_result",
        {"basis", "evidence_at", "declared_result"},
    )
    if freshness["basis"] != "test-completed-at":
        raise ReceiptError("$.fulfillment.freshness_result.basis: unsupported basis")
    try:
        parse_timestamp(
            freshness["evidence_at"], "$.fulfillment.freshness_result.evidence_at"
        )
    except RequirementError as exc:
        raise ReceiptError(str(exc)) from exc
    if freshness["declared_result"] not in {"within-window", "stale"}:
        raise ReceiptError("$.fulfillment.freshness_result.declared_result: unsupported result")

    assurances = fulfillment.get("assurance_results")
    if not isinstance(assurances, list) or not assurances:
        raise ReceiptError("$.fulfillment.assurance_results: expected a non-empty array")
    assurance_predicates: set[str] = set()
    for index, value in enumerate(assurances):
        path = f"$.fulfillment.assurance_results[{index}]"
        result = _contract_object(
            value, path, {"predicate", "result"}, {"evidence_ref"}
        )
        predicate = _contract_string(result["predicate"], f"{path}.predicate", maximum=96)
        if predicate not in ASSURANCE_PREDICATES:
            raise ReceiptError(f"{path}.predicate: unsupported assurance predicate")
        if predicate in assurance_predicates:
            raise ReceiptError(f"{path}.predicate: duplicate assurance result")
        assurance_predicates.add(predicate)
        if result["result"] not in {"satisfied", "not-satisfied", "not-evaluated"}:
            raise ReceiptError(f"{path}.result: unsupported assurance result")
        if result["result"] == "satisfied" and "evidence_ref" not in result:
            raise ReceiptError(f"{path}.evidence_ref: required for a satisfied predicate")
        if "evidence_ref" in result:
            _contract_string(result["evidence_ref"], f"{path}.evidence_ref", maximum=240)

    artifact = _contract_object(
        fulfillment.get("receipt_artifact"),
        "$.fulfillment.receipt_artifact",
        {"digest", "media_type", "format"},
    )
    _contract_digest(artifact["digest"], "$.fulfillment.receipt_artifact.digest")
    _contract_string(
        artifact["media_type"], "$.fulfillment.receipt_artifact.media_type", maximum=160
    )
    if "/" not in artifact["media_type"]:
        raise ReceiptError(
            "$.fulfillment.receipt_artifact.media_type: expected a media type"
        )
    _contract_string(
        artifact["format"], "$.fulfillment.receipt_artifact.format", maximum=240
    )

    summary = _contract_object(
        fulfillment.get("evidence_summary"),
        "$.fulfillment.evidence_summary",
        {"scope", "method", "category_outcomes", "private_report_digest"},
    )
    _contract_string(summary["scope"], "$.fulfillment.evidence_summary.scope")
    _contract_string(summary["method"], "$.fulfillment.evidence_summary.method")
    outcomes = summary["category_outcomes"]
    if not isinstance(outcomes, list) or not outcomes:
        raise ReceiptError(
            "$.fulfillment.evidence_summary.category_outcomes: expected a non-empty array"
        )
    categories: set[str] = set()
    for index, value in enumerate(outcomes):
        path = f"$.fulfillment.evidence_summary.category_outcomes[{index}]"
        outcome = _contract_object(value, path, {"category", "outcome"})
        category = _contract_identifier(outcome["category"], f"{path}.category")
        if category in categories:
            raise ReceiptError(f"{path}.category: duplicate category outcome")
        categories.add(category)
        if outcome["outcome"] not in {"pass", "fail", "not-tested"}:
            raise ReceiptError(f"{path}.outcome: unsupported category outcome")
    _contract_digest(
        summary["private_report_digest"],
        "$.fulfillment.evidence_summary.private_report_digest",
    )

    disclosure = _contract_object(
        fulfillment.get("disclosure"),
        "$.fulfillment.disclosure",
        {"mode", "omitted_content"},
    )
    if disclosure["mode"] != "content-minimized":
        raise ReceiptError("$.fulfillment.disclosure.mode: unsupported disclosure mode")
    omitted = disclosure["omitted_content"]
    if (
        not isinstance(omitted, list)
        or len(omitted) != len(PROHIBITED_CONTENT)
        or any(not isinstance(item, str) for item in omitted)
        or len(set(omitted)) != len(omitted)
        or set(omitted) != PROHIBITED_CONTENT
    ):
        raise ReceiptError(
            "$.fulfillment.disclosure.omitted_content: expected each prohibited class exactly once"
        )


def _validate_receipt_shape(receipt: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_top = {"$schema", "type", "spec_version", "statement", "statement_digest", "signature"}
    if set(receipt) != expected_top:
        raise ReceiptError("receipt has missing or unsupported top-level fields")
    if receipt["$schema"] != RECEIPT_SCHEMA or receipt["type"] != RECEIPT_TYPE:
        raise ReceiptError("unsupported receipt schema or type")
    if receipt["spec_version"] != SPEC_VERSION:
        raise ReceiptError("unsupported receipt spec_version")
    statement = receipt["statement"]
    if not isinstance(statement, dict) or set(statement) != {"adapter", "fulfillment", "limitations"}:
        raise ReceiptError("receipt statement has missing or unsupported fields")
    adapter = statement["adapter"]
    if not isinstance(adapter, dict) or adapter.get("type") != "experimental-detached-adapter":
        raise ReceiptError("receipt is not the supported experimental detached adapter")
    expected_adapter_fields = {
        "type", "version", "canonical_glacis_receipt_compatibility",
        "signing_path", "extension_seam", "sdk_version", "blocker",
    }
    if set(adapter) != expected_adapter_fields:
        raise ReceiptError("receipt adapter has missing or unsupported fields")
    if adapter["version"] != SPEC_VERSION:
        raise ReceiptError("unsupported adapter version")
    if adapter.get("canonical_glacis_receipt_compatibility") != "not-claimed":
        raise ReceiptError("receipt makes an unsupported compatibility claim")
    if adapter["signing_path"] != "public-glacis-python-sdk-offline-attestation":
        raise ReceiptError("receipt makes an unsupported signing-path claim")
    if adapter["extension_seam"] != "signed-control-plane-results-statement-digest":
        raise ReceiptError("receipt uses an unsupported SDK extension seam")
    if adapter["blocker"] != ADAPTER_BLOCKER:
        raise ReceiptError("receipt compatibility blocker is missing or altered")
    if statement["limitations"] != ADAPTER_LIMITATIONS:
        raise ReceiptError("receipt limitations are missing or altered")
    fulfillment = statement["fulfillment"]
    if not isinstance(fulfillment, dict):
        raise ReceiptError("receipt fulfillment must be an object")
    expected_fulfillment = {
        "$schema", "type", "spec_version", "request_id", "requirement_digest",
        "nonce", "profile", "receipt_claim_type", "subject", "test_profile",
        "completeness", "criteria_results", "freshness_result", "assurance_results",
        "receipt_artifact", "evidence_summary", "disclosure",
    }
    if set(fulfillment) != expected_fulfillment:
        raise ReceiptError("fulfillment has missing or unsupported fields")
    if fulfillment["$schema"] != FULFILLMENT_SCHEMA or fulfillment["type"] != FULFILLMENT_TYPE:
        raise ReceiptError("unsupported fulfillment schema or type")
    if fulfillment["spec_version"] != SPEC_VERSION:
        raise ReceiptError("unsupported fulfillment spec_version")
    _validate_fulfillment_contract(fulfillment)
    artifact_claim = fulfillment.get("receipt_artifact")
    if not isinstance(artifact_claim, dict) or set(artifact_claim) != {
        "digest", "media_type", "format"
    }:
        raise ReceiptError("fulfillment receipt_artifact has invalid shape")
    if not _DIGEST_RE.fullmatch(str(artifact_claim["digest"])):
        raise ReceiptError("fulfillment receipt_artifact digest is malformed")
    if artifact_claim["media_type"] != "application/json":
        raise ReceiptError("fulfillment receipt_artifact media type is unsupported")
    if artifact_claim["format"] != REDACTED_FORMAT:
        raise ReceiptError("fulfillment receipt_artifact format is unsupported")
    if not _DIGEST_RE.fullmatch(str(receipt["statement_digest"])):
        raise ReceiptError("statement_digest is malformed")
    signature = receipt["signature"]
    if not isinstance(signature, dict) or set(signature) != {"type", "sdk_version", "attestation"}:
        raise ReceiptError("signature has missing or unsupported fields")
    if signature["type"] != "glacis-python-offline-attestation":
        raise ReceiptError("unsupported signature type")
    if not isinstance(signature["attestation"], dict):
        raise ReceiptError("signature.attestation must be an object")
    return statement, fulfillment


def _check_completeness(
    fulfillment: Mapping[str, Any], requirement: Mapping[str, Any]
) -> bool:
    completeness = fulfillment.get("completeness")
    if not isinstance(completeness, dict) or set(completeness) != {
        "rule", "completed_tests", "omissions", "result"
    }:
        return False
    if completeness["rule"] != requirement["completeness"]["rule"]:
        return False
    completed = completeness["completed_tests"]
    omissions = completeness["omissions"]
    if not isinstance(completed, list) or not isinstance(omissions, list):
        return False
    if (
        any(not isinstance(test_id, str) for test_id in completed)
        or len(set(completed)) != len(completed)
        or completeness["result"] not in {"complete", "incomplete"}
    ):
        return False
    allowed = {
        item["test_id"]: item for item in requirement["completeness"]["allowed_omissions"]
    }
    approved_omissions: set[str] = set()
    for omission in omissions:
        if (
            not isinstance(omission, dict)
            or set(omission) != {"test_id", "reason_code", "explanation"}
            or omission != allowed.get(omission.get("test_id"))
            or omission["test_id"] in approved_omissions
        ):
            return False
        approved_omissions.add(omission["test_id"])
    accounted = set(completed) | approved_omissions
    expected = set(requirement["test_profile"]["tests"])
    actual_complete = accounted == expected and not (set(completed) & approved_omissions)
    return completeness["result"] == ("complete" if actual_complete else "incomplete") and actual_complete


def _check_criteria(fulfillment: Mapping[str, Any], requirement: Mapping[str, Any]) -> bool:
    results = fulfillment.get("criteria_results")
    if not isinstance(results, list):
        return False
    by_id = {
        item.get("criterion_id"): item
        for item in results
        if isinstance(item, dict)
    }
    expected_ids = {criterion["id"] for criterion in requirement["criteria"]}
    if len(by_id) != len(results) or set(by_id) != expected_ids:
        return False
    return all(
        set(item) == {"criterion_id", "status", "observed"}
        and item["status"] == "pass"
        and isinstance(item["observed"], str)
        and bool(item["observed"])
        for item in by_id.values()
    )


def _check_artifact_binding(
    artifact: Any, fulfillment: Mapping[str, Any], requirement: Mapping[str, Any]
) -> bool:
    if not isinstance(artifact, dict):
        return False
    expected_top = {
        "$schema", "type", "spec_version", "handling", "fulfillment_state",
        "requirement_digest", "request_id", "subject", "test_profile", "tool",
        "run", "scope", "method", "category_outcomes", "completeness",
        "criteria_results", "freshness_result", "assurance_results",
        "private_report_digest", "disclosure", "limitations",
    }
    if set(artifact) != expected_top:
        return False
    if (
        artifact["$schema"] != REDACTED_SCHEMA
        or artifact["type"] != REDACTED_TYPE
        or artifact["spec_version"] != SPEC_VERSION
        or artifact["handling"] != "shareable-content-minimized"
        or artifact["fulfillment_state"] not in {"fulfilled", "incomplete"}
        or artifact["limitations"] != ADAPTER_LIMITATIONS
    ):
        return False
    summary = fulfillment.get("evidence_summary")
    if not isinstance(summary, dict) or set(summary) != {
        "scope", "method", "category_outcomes", "private_report_digest"
    }:
        return False
    completeness = fulfillment.get("completeness")
    criteria_results = fulfillment.get("criteria_results")
    freshness_result = fulfillment.get("freshness_result")
    assurance_results = fulfillment.get("assurance_results")
    if (
        not isinstance(completeness, dict)
        or not isinstance(criteria_results, list)
        or not isinstance(freshness_result, dict)
        or not isinstance(assurance_results, list)
    ):
        return False
    expected_pairs = (
        (artifact["requirement_digest"], fulfillment["requirement_digest"]),
        (artifact["request_id"], fulfillment["request_id"]),
        (artifact["subject"], fulfillment["subject"]),
        (artifact["test_profile"], fulfillment["test_profile"]),
        (artifact["completeness"], fulfillment["completeness"]),
        (artifact["criteria_results"], fulfillment["criteria_results"]),
        (artifact["freshness_result"], fulfillment["freshness_result"]),
        (artifact["assurance_results"], fulfillment["assurance_results"]),
        (artifact["scope"], summary["scope"]),
        (artifact["method"], summary["method"]),
        (artifact["category_outcomes"], summary["category_outcomes"]),
        (artifact["private_report_digest"], summary["private_report_digest"]),
        (artifact["disclosure"], fulfillment["disclosure"]),
    )
    minimum = set(requirement["minimum_assurance_predicates"])
    satisfied = {
        item.get("predicate")
        for item in assurance_results
        if isinstance(item, dict) and item.get("result") == "satisfied"
    }
    expected_state = "fulfilled" if (
        completeness.get("result") == "complete"
        and all(
            isinstance(item, dict) and item.get("status") == "pass"
            for item in criteria_results
        )
        and freshness_result.get("declared_result") == "within-window"
        and minimum <= satisfied
    ) else "incomplete"
    tool = artifact.get("tool")
    tool_shape_valid = (
        isinstance(tool, dict)
        and set(tool) == {"name", "version"}
        and tool["name"] == "glacis-autoredteam"
        and isinstance(tool["version"], str)
        and bool(tool["version"])
    )
    return (
        all(left == right for left, right in expected_pairs)
        and artifact["requirement_digest"] == requirement_digest(requirement)
        and artifact["fulfillment_state"] == expected_state
        and tool_shape_valid
    )


def _check_assurance(
    fulfillment: Mapping[str, Any], requirement: Mapping[str, Any], signature_valid: bool
) -> tuple[bool, bool]:
    results = fulfillment.get("assurance_results")
    if not isinstance(results, list):
        return False, False
    by_predicate: dict[str, Mapping[str, Any]] = {}
    shape_valid = True
    for item in results:
        if not isinstance(item, dict) or item.get("predicate") not in ASSURANCE_PREDICATES:
            shape_valid = False
            continue
        allowed_keys = {"predicate", "result", "evidence_ref"}
        if not {"predicate", "result"} <= set(item) or set(item) - allowed_keys:
            shape_valid = False
        if item.get("result") not in {"satisfied", "not-satisfied", "not-evaluated"}:
            shape_valid = False
        if item.get("result") == "satisfied" and not item.get("evidence_ref"):
            shape_valid = False
        if item["predicate"] in by_predicate:
            shape_valid = False
        by_predicate[item["predicate"]] = item

    # This adapter can substantiate only that a key signed the claim. Any
    # stronger satisfied predicate would be an unsupported assertion.
    unsupported_satisfied = any(
        predicate != "issuer-signed" and item.get("result") == "satisfied"
        for predicate, item in by_predicate.items()
    )
    minimum_satisfied = all(
        predicate in by_predicate
        and by_predicate[predicate].get("result") == "satisfied"
        for predicate in requirement["minimum_assurance_predicates"]
    )
    issuer = by_predicate.get("issuer-signed", {})
    issuer_valid = issuer.get("result") == "satisfied" and signature_valid
    return shape_valid and minimum_satisfied and issuer_valid, not unsupported_satisfied


def verify_receipt(
    receipt: Mapping[str, Any],
    requirement: Mapping[str, Any],
    *,
    artifact_path: Optional[Union[str, Path]] = None,
    trusted_public_key: Optional[str] = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Verify signature integrity, request binding, freshness, and disclosure."""
    requirement = validate_requirement(requirement)
    validate_autoredteam_requirement(requirement)
    statement, fulfillment = _validate_receipt_shape(receipt)
    verification_time = _as_utc(now)
    req_digest = requirement_digest(requirement)

    computed_statement_digest = _statement_digest(statement)
    statement_digest_match = computed_statement_digest == receipt["statement_digest"]
    request_fields_match = all((
        fulfillment["request_id"] == requirement["request_id"],
        fulfillment["requirement_digest"] == req_digest,
        fulfillment["nonce"] == requirement["nonce"],
        fulfillment["profile"] == requirement["profile"],
        fulfillment["receipt_claim_type"] == requirement["receipt_claim_type"],
        fulfillment["subject"] == requirement["subject"],
        fulfillment["test_profile"] == requirement["test_profile"],
    ))

    _, Attestation, verify_offline, hash_payload, _ = _sdk_imports()
    signature_doc = receipt["signature"]
    signed_sdk_version = str(statement["adapter"]["sdk_version"])
    sdk_version_copy_match = str(signature_doc["sdk_version"]) == signed_sdk_version
    sdk_version_supported = bool(
        re.fullmatch(r"0\.8\.([1-9]|[1-9][0-9]+)(?:[.+-].*)?", signed_sdk_version)
    )
    signature_error: Optional[str] = None
    signer_public_key = ""
    signature_valid = False
    sdk_binding_match = False
    sdk_evidence_hash_match = False
    sdk_attestation_context_match = False
    sdk_attestation_shape_match = False
    try:
        attestation = Attestation.model_validate(signature_doc["attestation"])
        sdk_attestation_shape_match = (
            signature_doc["attestation"]
            == attestation.model_dump(exclude_none=True)
        )
        signer_public_key = attestation.public_key.lower()
        result = verify_offline(attestation)
        signature_valid = bool(result.valid and result.signature_valid)
        signature_error = result.error
        expected_binding = {
            "type": SDK_BINDING_TYPE,
            "statement_digest": receipt["statement_digest"],
            "requirement_digest": fulfillment["requirement_digest"],
        }
        sdk_binding_match = attestation.control_plane_results == expected_binding
        sdk_evidence_hash_match = attestation.evidence_hash == hash_payload(
            {
                "input": {"requirement_digest": fulfillment["requirement_digest"]},
                "output": {"statement_digest": receipt["statement_digest"]},
            }
        )
        sdk_attestation_context_match = bool(
            attestation.service_id == "glacis-autoredteam"
            and attestation.operation_type == "red-team-run-receipt"
            and attestation.is_offline is True
            and result.error is None
        )
    except Exception as exc:
        signature_error = f"SDK attestation could not be verified: {exc}"

    expected_public_key_match = True
    trust = "self-declared-key"
    if trusted_public_key is not None:
        expected_key = load_public_key(trusted_public_key)
        expected_public_key_match = signer_public_key == expected_key
        trust = "expected-key" if expected_public_key_match else "wrong-key"

    artifact_digest_match = False
    artifact_prohibited_field_names_absent = False
    artifact_binding_match = False
    artifact_error: Optional[str] = None
    if artifact_path is not None:
        artifact = Path(artifact_path)
        expected_artifact_digest = fulfillment["receipt_artifact"]["digest"]
        try:
            artifact_digest_match = _sha256_file(artifact) == expected_artifact_digest
            artifact_doc = load_json_object(artifact)
            artifact_prohibited_field_names_absent = not _prohibited_key_paths(artifact_doc)
            artifact_binding_match = _check_artifact_binding(
                artifact_doc, fulfillment, requirement
            )
        except (ReceiptError, RequirementError) as exc:
            artifact_error = str(exc)

    receipt_prohibited_field_names_absent = not _prohibited_key_paths(receipt)
    disclosure = fulfillment.get("disclosure")
    disclosure_match = (
        isinstance(disclosure, dict)
        and disclosure.get("mode") == "content-minimized"
        and set(disclosure.get("omitted_content", [])) == PROHIBITED_CONTENT
    )

    freshness = fulfillment.get("freshness_result")
    fresh = False
    declared_freshness_match = False
    evidence_before_expiry = False
    request_unexpired = False
    verification_after_request_creation = False
    if isinstance(freshness, dict):
        try:
            evidence_at = parse_timestamp(freshness.get("evidence_at"), "$.freshness_result.evidence_at")
            not_before = parse_timestamp(requirement["freshness"]["not_before"], "$.freshness.not_before")
            expires_at = parse_timestamp(requirement["return"]["expires_at"], "$.return.expires_at")
            created_at = parse_timestamp(requirement["created_at"], "$.created_at")
            age = verification_time - evidence_at
            fresh = (
                freshness.get("basis") == requirement["freshness"]["basis"]
                and evidence_at >= not_before
                and evidence_at <= verification_time
                and timedelta(0) <= age <= parse_duration(
                    requirement["freshness"]["max_age"], "$.freshness.max_age"
                )
            )
            evidence_before_expiry = evidence_at < expires_at
            request_unexpired = verification_time < expires_at
            verification_after_request_creation = verification_time >= created_at
            declared = "within-window" if fresh else "stale"
            declared_freshness_match = freshness.get("declared_result") == declared
        except (RequirementError, ReceiptError, TypeError):
            pass

    completeness_satisfied = _check_completeness(fulfillment, requirement)
    criteria_satisfied = _check_criteria(fulfillment, requirement)
    category_outcomes_match = {
        item["category"]
        for item in fulfillment["evidence_summary"]["category_outcomes"]
    } == set(requirement["test_profile"]["tests"])
    assurance_satisfied, no_unsupported_assurance = _check_assurance(
        fulfillment, requirement, signature_valid
    )

    checks: dict[str, bool] = {
        "sdk_version_supported": sdk_version_supported,
        "signed_sdk_version_copy_match": sdk_version_copy_match,
        "signature_valid_under_displayed_key": signature_valid,
        "trusted_public_key_match": expected_public_key_match,
        "sdk_signed_binding_match": sdk_binding_match,
        "sdk_evidence_hash_match": sdk_evidence_hash_match,
        "sdk_attestation_context_match": sdk_attestation_context_match,
        "sdk_attestation_shape_match": sdk_attestation_shape_match,
        "statement_digest_match": statement_digest_match,
        "requirement_digest_and_fields_match": request_fields_match,
        "artifact_digest_match": artifact_digest_match,
        "artifact_binding_match": artifact_binding_match,
        "receipt_prohibited_field_names_absent": receipt_prohibited_field_names_absent,
        "artifact_prohibited_field_names_absent": artifact_prohibited_field_names_absent,
        "disclosure_match": disclosure_match,
        "completeness_satisfied": completeness_satisfied,
        "criteria_satisfied": criteria_satisfied,
        "category_outcomes_match": category_outcomes_match,
        "freshness_window_satisfied": fresh,
        "declared_freshness_match": declared_freshness_match,
        "evidence_before_request_expiry": evidence_before_expiry,
        "request_unexpired": request_unexpired,
        "verification_after_request_creation": verification_after_request_creation,
        "minimum_assurance_satisfied": assurance_satisfied,
        "no_unsupported_assurance_claim": no_unsupported_assurance,
    }
    valid = all(checks.values())
    if not sdk_version_supported:
        status = "unsupported-signing-format"
    elif (
        not signature_valid
        or not statement_digest_match
        or not sdk_binding_match
        or not sdk_evidence_hash_match
        or not sdk_attestation_context_match
        or not sdk_attestation_shape_match
        or not sdk_version_copy_match
    ):
        status = "invalid-signature-or-tamper"
    elif not request_fields_match:
        status = "requirement-mismatch"
    elif not request_unexpired or not evidence_before_expiry:
        status = "expired"
    elif not fresh or not declared_freshness_match:
        status = "stale"
    elif (
        not completeness_satisfied
        or not criteria_satisfied
        or not category_outcomes_match
        or not assurance_satisfied
        or not no_unsupported_assurance
    ):
        status = "incomplete"
    elif not receipt_prohibited_field_names_absent or not disclosure_match:
        status = "raw-content-or-disclosure-violation"
    elif (
        not artifact_digest_match
        or not artifact_binding_match
        or not artifact_prohibited_field_names_absent
    ):
        status = "artifact-missing-or-mismatch"
    elif not expected_public_key_match:
        status = "wrong-key"
    else:
        status = "requirement-fulfilled"

    return {
        "valid": valid,
        "status": status,
        "checks": checks,
        "signer_public_key": signer_public_key,
        "key_trust": trust,
        "witness_status": "none-self-signed",
        "current_revocation_status": "unknown-offline",
        "signature_explanation": SELF_SIGNED_EXPLANATION,
        "buyer_signature_verification": (
            "present-digest-bound-not-cryptographically-checked"
            if "buyer_signature" in requirement
            else "absent"
        ),
        "compatibility": "experimental-detached-adapter; canonical Glacis receipt compatibility is not claimed",
        "content_check_scope": (
            "prohibited field-name scan across the receipt and parsed artifact; "
            "arbitrary string values are not content-classified"
        ),
        "signature_error": signature_error,
        "artifact_error": artifact_error,
    }


def verify_receipt_files(
    receipt_path: Union[str, Path],
    requirement_path: Union[str, Path],
    artifact_path: Union[str, Path],
    *,
    trusted_public_key: Optional[str] = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    receipt = load_json_object(Path(receipt_path))
    requirement = load_json_object(Path(requirement_path))
    return verify_receipt(
        receipt,
        requirement,
        artifact_path=artifact_path,
        trusted_public_key=trusted_public_key,
        now=now,
    )


__all__ = [
    "ADAPTER_BLOCKER",
    "ADAPTER_LIMITATIONS",
    "ReceiptArtifacts",
    "ReceiptError",
    "SELF_SIGNED_EXPLANATION",
    "TEST_CATEGORY_ALIASES",
    "build_receipt_artifacts",
    "generate_signing_key",
    "load_public_key",
    "load_signing_seed",
    "verify_receipt",
    "verify_receipt_files",
]
