"""Public AI Evidence Requirement v0.1 parsing and canonicalization.

This module implements the experimental, issuer-neutral request contract used
by the Glacis requirement demo.  It deliberately does not define or claim
compatibility with the canonical Glacis receipt envelope.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union
from urllib.parse import urlsplit


SPEC_VERSION = "0.1.0"
REQUIREMENT_SCHEMA = (
    "urn:glacis:experimental:ai-evidence-requirement:schema:request:0.1"
)
REQUIREMENT_TYPE = "urn:glacis:experimental:ai-evidence-requirement:request:0.1"
FULFILLMENT_SCHEMA = (
    "urn:glacis:experimental:ai-evidence-requirement:"
    "schema:fulfillment-binding:0.1"
)
FULFILLMENT_TYPE = (
    "urn:glacis:experimental:ai-evidence-requirement:"
    "fulfillment-binding:0.1"
)
RED_TEAM_PROFILE_ID = (
    "urn:glacis:experimental:ai-evidence-profile:red-team-run"
)
RED_TEAM_CLAIM_TYPE = (
    "urn:glacis:experimental:ai-evidence-claim:red-team-run:0.1"
)
REQUIREMENT_DIGEST_DOMAIN = b"AI-EVIDENCE-REQUIREMENT/0.1\n"

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_NONCE_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")
_REQUEST_ID_RE = re.compile(
    r"^urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_SEMVER_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
_URN_RE = re.compile(
    r"^urn:[a-z0-9][a-z0-9-]{1,31}:"
    r"[A-Za-z0-9][A-Za-z0-9._~:/?#@!$&'()*+,;=-]*$"
)
_DURATION_RE = re.compile(r"^(?:P([1-9][0-9]*)D|PT([1-9][0-9]*)([HM]))$")

ASSURANCE_PREDICATES = {
    "issuer-signed",
    "organization-bound",
    "time-anchored",
    "execution-attested",
    "independently-observed",
}
PROHIBITED_CONTENT = {"raw-prompts", "raw-outputs", "detailed-findings"}
REQUIRED_SUMMARY = {
    "scope",
    "method",
    "category-outcomes",
    "omissions",
    "criteria-results",
    "private-report-digest",
}


class RequirementError(ValueError):
    """The supplied requirement is malformed or unsupported."""


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RequirementError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object while rejecting duplicate member names."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_duplicate_rejecting_object)
    except OSError as exc:
        raise RequirementError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RequirementError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RequirementError("top-level JSON value must be an object")
    return value


def _reject_numbers(value: Any, path: str = "$") -> None:
    # v0.1 forbids JSON numbers so canonicalization is identical in every
    # conforming producer without depending on floating-point behavior.
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        raise RequirementError(f"{path}: JSON numbers are not allowed in v0.1")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_numbers(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RequirementError(f"{path}: object member names must be strings")
            _reject_numbers(item, f"{path}.{key}")
        return
    raise RequirementError(f"{path}: {type(value).__name__} is not a JSON value")


def _json_string(value: str) -> str:
    try:
        # Encoding first rejects lone surrogates, which are not valid Unicode
        # scalar values and therefore cannot appear in RFC 8785 input.
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RequirementError("lone surrogate is not valid canonical JSON") from exc
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _utf16_sort_key(value: str) -> bytes:
    # RFC 8785 orders object names by UTF-16 code units.
    return value.encode("utf-16-be")


def canonical_json_v01(value: Any) -> str:
    """Return RFC 8785-style canonical JSON for the number-free v0.1 profile."""
    _reject_numbers(value)

    def encode(item: Any) -> str:
        if item is None:
            return "null"
        if isinstance(item, bool):
            return "true" if item else "false"
        if isinstance(item, str):
            return _json_string(item)
        if isinstance(item, list):
            return "[" + ",".join(encode(child) for child in item) + "]"
        if isinstance(item, dict):
            keys = sorted(item, key=_utf16_sort_key)
            return "{" + ",".join(
                f"{_json_string(key)}:{encode(item[key])}" for key in keys
            ) + "}"
        raise RequirementError(f"unsupported canonical JSON type: {type(item).__name__}")

    return encode(value)


def requirement_digest(requirement: Mapping[str, Any]) -> str:
    """Hash a request after removing its optional top-level buyer signature."""
    unsigned = dict(requirement)
    unsigned.pop("buyer_signature", None)
    canonical = canonical_json_v01(unsigned).encode("utf-8")
    return "sha256:" + hashlib.sha256(REQUIREMENT_DIGEST_DOMAIN + canonical).hexdigest()


def _object(
    value: Any,
    path: str,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RequirementError(f"{path}: expected object")
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = required_set - set(value)
    extra = set(value) - allowed
    if missing:
        raise RequirementError(f"{path}: missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise RequirementError(f"{path}: unsupported fields: {', '.join(sorted(extra))}")
    return value


def _string(value: Any, path: str, *, minimum: int = 1, maximum: int = 500) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        raise RequirementError(
            f"{path}: expected string with length {minimum}..{maximum}"
        )
    return value


def _matches(
    value: Any,
    pattern: re.Pattern[str],
    path: str,
    *,
    maximum: int = 500,
) -> str:
    text = _string(value, path, maximum=maximum)
    if not pattern.fullmatch(text):
        raise RequirementError(f"{path}: invalid value")
    return text


def parse_timestamp(value: Any, path: str) -> datetime:
    text = _matches(value, _TIMESTAMP_RE, path)
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise RequirementError(f"{path}: invalid UTC timestamp") from exc


def parse_duration(value: Any, path: str) -> timedelta:
    text = _matches(value, _DURATION_RE, path, maximum=32)
    match = _DURATION_RE.fullmatch(text)
    if match is None:
        raise RequirementError(f"{path}: invalid duration")
    days, amount, unit = match.groups()
    try:
        if days is not None:
            return timedelta(days=int(days))
        if unit == "H":
            return timedelta(hours=int(amount))
        return timedelta(minutes=int(amount))
    except (OverflowError, ValueError) as exc:
        raise RequirementError(f"{path}: duration is outside the supported range") from exc


def _unique_string_list(
    value: Any,
    path: str,
    *,
    minimum: int = 1,
    allowed: Optional[set[str]] = None,
    pattern: Optional[re.Pattern[str]] = None,
    maximum: int = 240,
) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum:
        raise RequirementError(f"{path}: expected at least {minimum} item(s)")
    result: list[str] = []
    for index, item in enumerate(value):
        text = _string(item, f"{path}[{index}]", maximum=maximum)
        if allowed is not None and text not in allowed:
            raise RequirementError(f"{path}[{index}]: unsupported value {text!r}")
        if pattern is not None and not pattern.fullmatch(text):
            raise RequirementError(f"{path}[{index}]: invalid identifier")
        result.append(text)
    if len(set(result)) != len(result):
        raise RequirementError(f"{path}: duplicate values are not allowed")
    return result


def _versioned_reference(value: Any, path: str) -> None:
    obj = _object(value, path, ("id", "version"))
    _matches(obj["id"], _URN_RE, f"{path}.id", maximum=240)
    _matches(obj["version"], _SEMVER_RE, f"{path}.version")


def _subject(value: Any, path: str) -> None:
    obj = _object(value, path, ("kind", "id", "build_digest"))
    if obj["kind"] != "software-build":
        raise RequirementError(f"{path}.kind: expected 'software-build'")
    _matches(obj["id"], _URN_RE, f"{path}.id", maximum=240)
    _matches(obj["build_digest"], _DIGEST_RE, f"{path}.build_digest")


def _test_profile(value: Any, path: str) -> None:
    obj = _object(value, path, ("id", "version", "configuration_digest", "tests"))
    _matches(obj["id"], _URN_RE, f"{path}.id", maximum=240)
    _matches(obj["version"], _SEMVER_RE, f"{path}.version")
    _matches(obj["configuration_digest"], _DIGEST_RE, f"{path}.configuration_digest")
    _unique_string_list(obj["tests"], f"{path}.tests", pattern=_IDENTIFIER_RE)


def _omission(value: Any, path: str) -> None:
    obj = _object(value, path, ("test_id", "reason_code", "explanation"))
    _matches(obj["test_id"], _IDENTIFIER_RE, f"{path}.test_id")
    if obj["reason_code"] not in {
        "not-applicable",
        "buyer-approved-exception",
        "environment-unavailable",
    }:
        raise RequirementError(f"{path}.reason_code: unsupported value")
    _string(obj["explanation"], f"{path}.explanation")


def validate_requirement(requirement: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the exact experimental request object."""
    if not isinstance(requirement, dict):
        raise RequirementError("requirement must be a JSON object")
    _reject_numbers(requirement)
    obj = _object(
        requirement,
        "$",
        (
            "$schema", "type", "spec_version", "request_id", "created_at",
            "nonce", "profile", "receipt_claim_type", "accepted_receipt_formats",
            "subject", "test_profile", "completeness", "criteria", "freshness",
            "minimum_assurance_predicates", "return", "disclosure",
        ),
        ("buyer_signature",),
    )
    constants = {
        "$schema": REQUIREMENT_SCHEMA,
        "type": REQUIREMENT_TYPE,
        "spec_version": SPEC_VERSION,
    }
    for field, expected in constants.items():
        if obj[field] != expected:
            raise RequirementError(f"$.{field}: expected {expected!r}")
    _matches(obj["request_id"], _REQUEST_ID_RE, "$.request_id")
    created_at = parse_timestamp(obj["created_at"], "$.created_at")

    nonce = _object(obj["nonce"], "$.nonce", ("encoding", "value"))
    if nonce["encoding"] != "base64url-256":
        raise RequirementError("$.nonce.encoding: expected 'base64url-256'")
    nonce_text = _matches(nonce["value"], _NONCE_RE, "$.nonce.value")
    try:
        nonce_bytes = base64.urlsafe_b64decode(nonce_text + "=")
    except Exception as exc:
        raise RequirementError("$.nonce.value: invalid base64url") from exc
    canonical_nonce = base64.urlsafe_b64encode(nonce_bytes).rstrip(b"=").decode("ascii")
    if len(nonce_bytes) != 32 or canonical_nonce != nonce_text:
        raise RequirementError(
            "$.nonce.value: expected canonical unpadded base64url for exactly 32 bytes"
        )

    _versioned_reference(obj["profile"], "$.profile")
    _matches(obj["receipt_claim_type"], _URN_RE, "$.receipt_claim_type", maximum=240)
    accepted_formats = _unique_string_list(
        obj["accepted_receipt_formats"],
        "$.accepted_receipt_formats",
        maximum=160,
    )
    for index, media_type in enumerate(accepted_formats):
        if "/" not in media_type:
            raise RequirementError(
                f"$.accepted_receipt_formats[{index}]: expected a media type"
            )
    _subject(obj["subject"], "$.subject")
    _test_profile(obj["test_profile"], "$.test_profile")

    completeness = _object(
        obj["completeness"], "$.completeness", ("rule", "allowed_omissions")
    )
    if completeness["rule"] != "all-listed-tests-accounted-for":
        raise RequirementError("$.completeness.rule: unsupported rule")
    if not isinstance(completeness["allowed_omissions"], list):
        raise RequirementError("$.completeness.allowed_omissions: expected array")
    omission_ids: list[str] = []
    for index, omission in enumerate(completeness["allowed_omissions"]):
        _omission(omission, f"$.completeness.allowed_omissions[{index}]")
        omission_ids.append(omission["test_id"])
    if len(set(omission_ids)) != len(omission_ids):
        raise RequirementError("$.completeness.allowed_omissions: duplicate test_id")
    unknown_omissions = set(omission_ids) - set(obj["test_profile"]["tests"])
    if unknown_omissions:
        raise RequirementError(
            "$.completeness.allowed_omissions: test_id not in test_profile.tests: "
            + ", ".join(sorted(unknown_omissions))
        )

    if not isinstance(obj["criteria"], list) or not obj["criteria"]:
        raise RequirementError("$.criteria: expected non-empty array")
    criterion_ids: list[str] = []
    for index, criterion_value in enumerate(obj["criteria"]):
        path = f"$.criteria[{index}]"
        criterion = _object(
            criterion_value,
            path,
            ("id", "description", "metric", "operator", "expected"),
        )
        criterion_ids.append(_matches(criterion["id"], _IDENTIFIER_RE, f"{path}.id"))
        _string(criterion["description"], f"{path}.description")
        _matches(criterion["metric"], _IDENTIFIER_RE, f"{path}.metric")
        if criterion["operator"] not in {
            "equals", "not-equals", "less-than-or-equal",
            "greater-than-or-equal", "contains",
        }:
            raise RequirementError(f"{path}.operator: unsupported value")
        _string(criterion["expected"], f"{path}.expected", maximum=200)
    if len(set(criterion_ids)) != len(criterion_ids):
        raise RequirementError("$.criteria: duplicate criterion id")

    freshness = _object(obj["freshness"], "$.freshness", ("basis", "max_age", "not_before"))
    if freshness["basis"] != "test-completed-at":
        raise RequirementError("$.freshness.basis: unsupported value")
    parse_duration(freshness["max_age"], "$.freshness.max_age")
    not_before = parse_timestamp(freshness["not_before"], "$.freshness.not_before")
    if not_before < created_at:
        raise RequirementError(
            "$.freshness.not_before: must not precede request creation"
        )

    _unique_string_list(
        obj["minimum_assurance_predicates"],
        "$.minimum_assurance_predicates",
        allowed=ASSURANCE_PREDICATES,
    )

    return_obj = _object(obj["return"], "$.return", ("uri", "due_at", "expires_at"))
    uri = _string(return_obj["uri"], "$.return.uri")
    if re.search(r"[\x00-\x20\x7f]", uri) or not uri.startswith("https://"):
        raise RequirementError(
            "$.return.uri: expected absolute https URI without raw whitespace or controls"
        )
    try:
        parsed_uri = urlsplit(uri)
        username = parsed_uri.username
        password = parsed_uri.password
        parsed_uri.port
    except ValueError as exc:
        raise RequirementError(f"$.return.uri: invalid URI: {exc}") from exc
    if (
        parsed_uri.scheme != "https"
        or not parsed_uri.netloc
        or not parsed_uri.hostname
        or username is not None
        or password is not None
    ):
        raise RequirementError(
            "$.return.uri: expected absolute https URI without userinfo"
        )
    due_at = parse_timestamp(return_obj["due_at"], "$.return.due_at")
    expires_at = parse_timestamp(return_obj["expires_at"], "$.return.expires_at")
    if due_at < created_at or expires_at <= due_at:
        raise RequirementError("$.return: expected created_at <= due_at < expires_at")

    disclosure = _object(
        obj["disclosure"],
        "$.disclosure",
        ("mode", "prohibited_content", "required_summary"),
    )
    if disclosure["mode"] != "content-minimized":
        raise RequirementError("$.disclosure.mode: expected 'content-minimized'")
    prohibited = set(_unique_string_list(
        disclosure["prohibited_content"], "$.disclosure.prohibited_content", minimum=3,
        allowed=PROHIBITED_CONTENT,
    ))
    if prohibited != PROHIBITED_CONTENT:
        raise RequirementError("$.disclosure.prohibited_content: all three exclusions are required")
    summary = set(_unique_string_list(
        disclosure["required_summary"], "$.disclosure.required_summary", minimum=6,
        allowed=REQUIRED_SUMMARY,
    ))
    if summary != REQUIRED_SUMMARY:
        raise RequirementError("$.disclosure.required_summary: all six summary fields are required")

    if "buyer_signature" in obj:
        signature = _object(
            obj["buyer_signature"],
            "$.buyer_signature",
            ("algorithm", "key_id", "signed_digest", "signature"),
        )
        if signature["algorithm"] != "Ed25519":
            raise RequirementError("$.buyer_signature.algorithm: expected 'Ed25519'")
        _string(signature["key_id"], "$.buyer_signature.key_id", maximum=240)
        _matches(signature["signed_digest"], _DIGEST_RE, "$.buyer_signature.signed_digest")
        sig_value = _string(signature["signature"], "$.buyer_signature.signature", minimum=86, maximum=86)
        if not re.fullmatch(r"[A-Za-z0-9_-]{86}", sig_value):
            raise RequirementError("$.buyer_signature.signature: invalid base64url signature")
        try:
            sig_bytes = base64.urlsafe_b64decode(sig_value + "==")
        except Exception as exc:
            raise RequirementError(
                "$.buyer_signature.signature: invalid base64url signature"
            ) from exc
        canonical_signature = (
            base64.urlsafe_b64encode(sig_bytes).rstrip(b"=").decode("ascii")
        )
        if len(sig_bytes) != 64 or canonical_signature != sig_value:
            raise RequirementError(
                "$.buyer_signature.signature: expected canonical unpadded base64url "
                "for exactly 64 bytes"
            )
        if signature["signed_digest"] != requirement_digest(obj):
            raise RequirementError("$.buyer_signature.signed_digest: digest mismatch")

    return dict(obj)


def load_requirement(path: Union[str, Path]) -> dict[str, Any]:
    requirement = load_json_object(Path(path))
    return validate_requirement(requirement)


def validate_autoredteam_requirement(requirement: Mapping[str, Any]) -> None:
    """Reject request types this AutoRedTeam adapter cannot honestly fulfill."""
    profile = requirement["profile"]
    if profile != {"id": RED_TEAM_PROFILE_ID, "version": SPEC_VERSION}:
        raise RequirementError(
            "unsupported profile; this adapter supports only "
            f"{RED_TEAM_PROFILE_ID}@{SPEC_VERSION}"
        )
    if requirement["receipt_claim_type"] != RED_TEAM_CLAIM_TYPE:
        raise RequirementError(
            "unsupported receipt_claim_type; this adapter emits only "
            f"{RED_TEAM_CLAIM_TYPE}"
        )
    if "application/json" not in requirement["accepted_receipt_formats"]:
        raise RequirementError("request does not accept the adapter's application/json artifact")


__all__ = [
    "ASSURANCE_PREDICATES",
    "FULFILLMENT_SCHEMA",
    "FULFILLMENT_TYPE",
    "PROHIBITED_CONTENT",
    "RED_TEAM_CLAIM_TYPE",
    "RED_TEAM_PROFILE_ID",
    "REQUIREMENT_SCHEMA",
    "REQUIREMENT_TYPE",
    "RequirementError",
    "SPEC_VERSION",
    "canonical_json_v01",
    "load_json_object",
    "load_requirement",
    "parse_timestamp",
    "requirement_digest",
    "validate_autoredteam_requirement",
    "validate_requirement",
]
