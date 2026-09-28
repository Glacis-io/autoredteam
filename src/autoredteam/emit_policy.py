"""
emit_policy.py — Generate OVERT-compliant policy.toml from autoharden results.

OVERT (Open Verification and Evaluation for Responsible Technology) is an open
standard for declarative AI governance policy. It defines six domains — PROTECT,
ATTEST, MEASURE, TOOL, HITL, RESPOND — that together specify how an AI system
should be constrained, monitored, and governed at runtime.

autoredteam discovers vulnerabilities. emit_policy converts those findings into
a machine-readable governance policy that any OVERT-compatible enforcement layer
can consume — or that autoredteam can ingest for recursive hardening.

The closed loop:

  autoredteam → autoharden → emit_policy → policy.toml
                                               ↓
                              enforcement engine / recursive harden

Usage:
    # From autoharden results directory
    python emit_policy.py results/autoharden/healthcare_ambient_scribe/gpt-5-4/

    # With a specific profile and enforcement mode
    python emit_policy.py results/autoharden/ --profile healthcare-ambient --enforce

    # Output to a specific file
    python emit_policy.py results/autoharden/ -o deployment/policy.toml

    # From an autoharden_report.json directly
    python emit_policy.py results/autoharden/autoharden_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# We use tomli for reading (Python 3.11+ has tomllib) and a simple writer
# since tomli_w / tomlkit are optional dependencies.

# ---------------------------------------------------------------------------
# OVERT domain mapping: autoharden category → OVERT control domain
# ---------------------------------------------------------------------------

# Maps autoharden guardrail types to OVERT TOML sections
GUARDRAIL_TYPE_TO_OVERT = {
    "input_filter": "protect.input_filtering",
    "output_filter": "protect.output_filtering",
    "tool_policy": "tool",
}

# Maps autoharden attack categories to OVERT violation types
CATEGORY_TO_VIOLATION_TYPE = {
    "prompt_injection": "prompt_injection",
    "jailbreak": "jailbreak",
    "pii_extraction": "pii",
    "system_prompt_leakage": "system_prompt_leakage",
    "tool_misuse": "tool_misuse",
    "role_confusion": "role_confusion",
    "context_window_poisoning": "context_injection",
    "multi_turn_manipulation": "multi_turn_manipulation",
    "encoding_bypass": "encoding_bypass",
    "payload_splitting": "payload_splitting",
    "refusal_suppression": "refusal_suppression",
    "ethical_bypass": "ethical_bypass",
    "authority_manipulation": "authority_manipulation",
    "output_formatting_exploit": "output_formatting",
    "indirect_injection": "indirect_injection",
    "multilingual_attack": "multilingual_bypass",
    "continuation_attack": "continuation_attack",
    "social_engineering": "social_engineering",
    "hallucination_exploit": "hallucination",
}

# Maps autoharden governance tier → recommended enforcement mode
TIER_TO_ENFORCEMENT = {
    "Best-in-class": "strict",
    "Enterprise Gate": "enforce",
    "Regulatory Floor": "enforce",
    "Insurability Line": "warn",
    "Below Insurability Line": "shadow",
}

# Maps governance score → OVERT level target
def _score_to_overt_level(score: int) -> int:
    if score >= 850:
        return 4
    if score >= 700:
        return 3
    if score >= 550:
        return 2
    return 1


# ---------------------------------------------------------------------------
# Policy loading (for recursive hardening)
# ---------------------------------------------------------------------------

def load_policy(path: str | Path) -> dict:
    """
    Load a policy.toml and extract the starting posture for a new autoharden pass.

    Returns a dict with:
      - system_prompt: str  (the hardened prompt from the prior pass)
      - role_name: str      (extracted from policy name)
      - guardrail_config: dict  (prior guardrail rules, keyed by name)
      - prior_governance_score: int
      - prior_asr: float
      - overt_level: int
      - policy_id: str

    This is the inverse of emit_policy_toml — it reads the TOML artifact and
    reconstructs the inputs needed to start autoharden from where the last
    pass left off.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")

    text = path.read_text()
    parsed = _parse_toml_minimal(text)

    policy_section = parsed.get("policy", {})
    prompt_section = parsed.get("prompt", {})
    provenance = policy_section.get("provenance", {})

    # Extract hardened prompt
    system_prompt = prompt_section.get("hardened", "").strip()
    if not system_prompt:
        raise ValueError(f"No [prompt].hardened found in {path}")

    # Extract role name from policy name (strip " — Autohardened Policy" suffix)
    policy_name = policy_section.get("name", "")
    role_name = policy_name.replace(" — Autohardened Policy", "").strip()
    if not role_name:
        role_name = "this AI assistant"

    return {
        "system_prompt": system_prompt,
        "role_name": role_name,
        "prior_governance_score": provenance.get("governance_score", 0),
        "prior_asr": provenance.get("final_asr_percent", 0),
        "overt_level": policy_section.get("overt_level", {}).get("target", 1),
        "policy_id": policy_section.get("id", ""),
        "enforcement_mode": policy_section.get("enforcement_mode", "warn"),
        "profile": policy_section.get("profile", ""),
        "prior_evidence_chain_hash": provenance.get("evidence_chain_hash", ""),
        "prior_cycles": provenance.get("cycles_run", 0),
        "prior_blocks_kept": provenance.get("blocks_kept", 0),
    }


def _parse_toml_minimal(text: str) -> dict:
    """
    Minimal TOML parser sufficient for reading back policy.toml files.

    Handles: string values, integers, floats, booleans, arrays of strings,
    table headers, multiline basic strings (triple-quoted).

    This avoids requiring tomli/tomllib as a dependency for the OSS tool.
    """
    result: dict = {}
    current_table: dict = result
    current_path: list[str] = []
    lines = text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip comments and blanks
        if not line or line.startswith("#"):
            i += 1
            continue

        # Table header: [foo.bar] or [[foo.bar]]
        if line.startswith("[[") and line.endswith("]]"):
            # Array of tables — skip for now, not needed for load_policy
            i += 1
            continue
        if line.startswith("[") and line.endswith("]"):
            path = line[1:-1].strip().split(".")
            current_path = path
            # Navigate/create nested dicts
            current_table = result
            for part in path:
                if part not in current_table:
                    current_table[part] = {}
                current_table = current_table[part]
            i += 1
            continue

        # Key-value pair
        if "=" in line:
            eq_pos = line.index("=")
            key = line[:eq_pos].strip()
            val_str = line[eq_pos + 1:].strip()

            # Multiline basic string: """
            if val_str.startswith('"""'):
                content_after = val_str[3:]
                if content_after.endswith('"""') and len(content_after) > 3:
                    # Single-line triple-quoted
                    current_table[key] = content_after[:-3]
                else:
                    # Multiline — collect until closing """
                    ml_lines = [content_after] if content_after else []
                    i += 1
                    while i < len(lines):
                        if lines[i].strip() == '"""':
                            break
                        if lines[i].rstrip().endswith('"""'):
                            ml_lines.append(lines[i].rstrip()[:-3])
                            break
                        ml_lines.append(lines[i])
                        i += 1
                    current_table[key] = "\n".join(ml_lines)
                i += 1
                continue

            # Parse value
            current_table[key] = _parse_toml_value(val_str)

        i += 1

    return result


def _parse_toml_value(val: str) -> Any:
    """Parse a single TOML value."""
    # Strip inline comments
    if " #" in val and not val.startswith('"') and not val.startswith("'"):
        val = val[:val.index(" #")].strip()

    # Boolean
    if val == "true":
        return True
    if val == "false":
        return False

    # String (double-quoted)
    if val.startswith('"') and val.endswith('"'):
        return val[1:-1]

    # String (single-quoted)
    if val.startswith("'") and val.endswith("'"):
        return val[1:-1]

    # Array of strings
    if val.startswith("["):
        # Simple inline array
        inner = val[1:].rstrip("]").strip()
        if not inner:
            return []
        items = []
        for item in inner.split(","):
            item = item.strip().strip('"').strip("'")
            if item:
                items.append(item)
        return items

    # Integer
    try:
        return int(val)
    except ValueError:
        pass

    # Float
    try:
        return float(val)
    except ValueError:
        pass

    return val


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_autoharden_results(path: Path) -> dict:
    """
    Load autoharden results from a directory or a single report JSON.

    Looks for:
      - autoharden_report.json  (full report with metrics + cycle history)
      - guardrail_config.json   (guardrail rules)
      - block_history.json      (kept/discarded defense blocks)
      - hardened_prompt.txt      (the hardened system prompt)
      - evidence_chain.jsonl     (hash-chained attestation records)
    """
    if path.is_file() and path.suffix == ".json":
        with open(path) as f:
            report = json.load(f)
        # Try to find sibling files
        directory = path.parent
    elif path.is_dir():
        directory = path
        report_path = directory / "autoharden_report.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
        else:
            report = {}
    else:
        raise FileNotFoundError(f"Not a valid autoharden results path: {path}")

    results: dict[str, Any] = {"report": report}

    # Guardrail config
    gc_path = directory / "guardrail_config.json"
    if gc_path.exists():
        with open(gc_path) as f:
            results["guardrail_config"] = json.load(f)

    # Block history
    bh_path = directory / "block_history.json"
    if bh_path.exists():
        with open(bh_path) as f:
            results["block_history"] = json.load(f)

    # Hardened prompt
    hp_path = directory / "hardened_prompt.txt"
    if hp_path.exists():
        results["hardened_prompt"] = hp_path.read_text()

    # Evidence chain — read last record for chain hash
    ec_path = directory / "evidence_chain.jsonl"
    if ec_path.exists():
        lines = ec_path.read_text().strip().splitlines()
        if lines:
            last = json.loads(lines[-1])
            results["evidence_chain_hash"] = last.get("chain_hash", "")
            results["evidence_chain_length"] = len(lines)

    return results


# ---------------------------------------------------------------------------
# Policy generation
# ---------------------------------------------------------------------------

def generate_policy(
    results: dict,
    policy_id: str | None = None,
    profile: str | None = None,
    enforcement_mode: str | None = None,
    name: str | None = None,
) -> dict:
    """
    Generate an OVERT policy document (as a nested dict) from autoharden results.

    The dict structure mirrors the OVERT TOML schema:
      [policy], [protect], [attest], [measure], [tool], [hitl], [respond]
    plus the autoredteam-specific extensions:
      [prompt], [policy.provenance]
    """
    report = results.get("report", {})
    guardrails = results.get("guardrail_config", {})
    block_history = results.get("block_history", {})
    hardened_prompt = results.get("hardened_prompt", "")

    final_metrics = report.get("final_metrics", {})
    gov_score = final_metrics.get("governance_score", 0)
    gov_tier = final_metrics.get("governance_tier", "Below Insurability Line")
    role_name = report.get("role_name", "AI Assistant")
    target = report.get("target", "unknown")

    # --- [policy] ---
    if not policy_id:
        slug = role_name.lower().replace(" ", "-").replace("/", "-")
        model_slug = target.replace("/", "-").replace(".", "")
        policy_id = f"{slug}-{model_slug}-hardened"

    if not name:
        name = f"{role_name} — Autohardened Policy"

    if not enforcement_mode:
        enforcement_mode = TIER_TO_ENFORCEMENT.get(gov_tier, "warn")

    overt_level = _score_to_overt_level(gov_score)

    policy: dict[str, Any] = {
        "policy": {
            "id": policy_id,
            "version": "1.0.0",
            "name": name,
            "description": f"OVERT governance policy for {role_name}, generated from adversarial evaluation by autoredteam",
            "enforcement_mode": enforcement_mode,
            "generated_by": f"autoredteam/autoharden {report.get('version', '0.1.0')}",
            "generated_at": report.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "overt_level": {"target": overt_level},
            "provenance": _build_provenance(results),
        },
    }

    if profile:
        policy["policy"]["profile"] = profile

    # --- [prompt] ---
    if hardened_prompt:
        original = report.get("original_prompt", "")
        policy["prompt"] = {
            "hardened": hardened_prompt,
            "original_hash": _hash(original) if original else "",
            "hardened_hash": _hash(hardened_prompt),
            "defense_blocks": _extract_defense_block_summary(block_history),
        }

    # --- [protect] ---
    policy["protect"] = _build_protect(guardrails, final_metrics)

    # --- [attest] ---
    policy["attest"] = _build_attest(overt_level)

    # --- [measure] ---
    policy["measure"] = _build_measure(final_metrics, overt_level)

    # --- [tool] ---
    tool_section = _build_tool(guardrails)
    if tool_section:
        policy["tool"] = tool_section

    # --- [respond] ---
    policy["respond"] = _build_respond(enforcement_mode)

    return policy


def _build_provenance(results: dict) -> dict:
    report = results.get("report", {})
    fm = report.get("final_metrics", {})
    block_history = results.get("block_history", {})

    kept = block_history.get("kept", [])
    discarded = block_history.get("discarded", [])

    return {
        "target_model": report.get("target", "unknown"),
        "cycles_run": report.get("total_cycles", 0),
        "blocks_kept": len(kept),
        "blocks_discarded": len(discarded),
        "final_asr_percent": fm.get("asr", 0),
        "governance_score": fm.get("governance_score", 0),
        "governance_tier": fm.get("governance_tier", "unknown"),
        "evidence_chain_hash": results.get("evidence_chain_hash", ""),
        "evidence_chain_length": results.get("evidence_chain_length", 0),
    }


def _extract_defense_block_summary(block_history: dict) -> list[dict]:
    kept = block_history.get("kept", [])
    return [
        {
            "id": b.get("block_id", ""),
            "category": b.get("target_category", ""),
            "asr_delta": b.get("asr_delta", 0),
            "governance_delta": b.get("governance_delta", 0),
        }
        for b in kept
    ]


def _build_protect(guardrails: dict, final_metrics: dict) -> dict:
    protect: dict[str, Any] = {}

    # Boundary
    protect["boundary"] = {"mode": "enforce"}

    # Classify guardrails into input/output filtering
    input_rules = []
    output_rules = []
    custom_input_patterns = []
    custom_output_patterns = []

    # Flags derived from which categories were tested/bypassed
    categories_bypassed = set(final_metrics.get("categories_bypassed", []))
    cat_breakdown = final_metrics.get("category_breakdown", {})

    has_pii = "pii_extraction" in cat_breakdown
    has_injection = "prompt_injection" in cat_breakdown
    has_jailbreak = "jailbreak" in cat_breakdown
    has_indirect = "indirect_injection" in cat_breakdown

    for name, config in guardrails.items():
        gtype = config.get("type", "")
        rules = config.get("rules", [])

        if gtype == "input_filter":
            for rule in rules:
                pattern = rule.get("pattern", "")
                if pattern:
                    custom_input_patterns.append({
                        "pattern": pattern,
                        "action": rule.get("action", "flag"),
                        "label": name,
                    })
        elif gtype == "output_filter":
            for rule in rules:
                pattern = rule.get("pattern", "")
                if pattern:
                    custom_output_patterns.append({
                        "pattern": pattern,
                        "action": rule.get("action", "flag"),
                        "label": rule.get("label", name),
                    })

    # Input filtering
    protect["input_filtering"] = {
        "enabled": True,
        "scan_pii": has_pii,
        "scan_prompt_injection": has_injection or has_indirect,
        "scan_adversarial": has_jailbreak,
    }
    if custom_input_patterns:
        protect["input_filtering"]["custom_patterns"] = custom_input_patterns

    # Output filtering
    protect["output_filtering"] = {
        "enabled": True,
        "scan_pii": has_pii,
        "scan_harmful": True,
        "scan_hallucination": "hallucination_exploit" in cat_breakdown,
        "scan_out_of_scope": "role_confusion" in cat_breakdown,
    }
    if custom_output_patterns:
        protect["output_filtering"]["custom_patterns"] = custom_output_patterns

    # Rate limits (sensible defaults)
    protect["rate_limits"] = {
        "requests_per_minute": 100,
        "requests_per_epoch": 1500,
    }

    return protect


def _build_attest(overt_level: int) -> dict:
    attest: dict[str, Any] = {
        "epoch_duration_secs": 300,
    }
    if overt_level >= 3:
        attest["three_phase"] = True
    attest["optimistic_enforcement"] = {"enabled": overt_level < 4}
    return attest


def _build_measure(final_metrics: dict, overt_level: int) -> dict:
    # Build violation types from categories that were tested
    cat_breakdown = final_metrics.get("category_breakdown", {})
    violation_types = []
    for cat in cat_breakdown:
        vtype = CATEGORY_TO_VIOLATION_TYPE.get(cat, cat)
        violation_types.append(vtype)

    # Higher OVERT levels get higher sampling rates
    if overt_level >= 4:
        sampling_rate = 0.10
    elif overt_level >= 3:
        sampling_rate = 0.05
    else:
        sampling_rate = 0.01

    return {
        "sampling_rate": sampling_rate,
        "confidence_level": 0.95,
        "s3p_enabled": overt_level >= 3,
        "violation_types": sorted(violation_types),
    }


def _build_tool(guardrails: dict) -> dict | None:
    tool_policies = {k: v for k, v in guardrails.items() if v.get("type") == "tool_policy"}
    if not tool_policies:
        return None

    tool: dict[str, Any] = {
        "defaults": {"mode": "deny"},
        "deny": [],
        "circuit_breaker": {
            "enabled": True,
            "error_threshold": 0.1,
            "window_epochs": 3,
            "max_recursion_depth": 25,
            "max_identical_retries": 3,
        },
    }

    for _name, config in tool_policies.items():
        for rule in config.get("rules", []):
            tool_name = rule.get("tool", "")
            action = rule.get("action", "")
            condition = rule.get("condition", "")

            if action == "deny":
                tool["deny"].append({
                    "name": tool_name if tool_name != "*" else f"*_{condition}",
                    "reason": f"Blocked by autoharden: {condition}",
                })
            elif action == "require_confirmation":
                # Map to HITL approval gate — captured in deny as advisory
                tool["deny"].append({
                    "name": tool_name if tool_name != "*" else f"*_{condition}",
                    "reason": f"Requires human confirmation: {condition}",
                })
            elif action == "sandbox":
                tool["deny"].append({
                    "name": tool_name,
                    "reason": f"Sandboxed: {condition}",
                })

    if not tool["deny"]:
        del tool["deny"]

    return tool


def _build_respond(enforcement_mode: str) -> dict:
    return {
        "failure_mode": "fail-closed" if enforcement_mode in ("enforce", "strict") else "fail-open",
        "circuit_breaker_enabled": True,
    }


# ---------------------------------------------------------------------------
# TOML serializer (minimal, no external deps)
# ---------------------------------------------------------------------------

def to_toml(data: dict, _prefix: str = "") -> str:
    """
    Serialize a nested dict to TOML format.

    Handles the OVERT schema patterns: flat tables, nested tables,
    arrays of tables ([[tool.allow]]), and multiline strings.
    """
    lines: list[str] = []
    _serialize_table(data, lines, prefix="")
    return "\n".join(lines) + "\n"


def _serialize_table(data: dict, lines: list[str], prefix: str) -> None:
    # First pass: emit simple key-value pairs at this level
    for key, value in data.items():
        if isinstance(value, dict):
            continue
        if isinstance(value, list) and value and isinstance(value[0], dict):
            continue
        _emit_value(key, value, lines)

    # Second pass: emit nested tables
    for key, value in data.items():
        if isinstance(value, dict):
            full_key = f"{prefix}.{key}" if prefix else key
            lines.append("")
            lines.append(f"[{full_key}]")
            _serialize_table(value, lines, full_key)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            full_key = f"{prefix}.{key}" if prefix else key
            for item in value:
                lines.append("")
                lines.append(f"[[{full_key}]]")
                _serialize_table(item, lines, full_key)


def _emit_value(key: str, value: Any, lines: list[str]) -> None:
    if isinstance(value, str):
        if "\n" in value:
            lines.append(f'{key} = """')
            lines.append(value)
            lines.append('"""')
        elif '"' in value:
            lines.append(f"{key} = '{value}'")
        else:
            lines.append(f'{key} = "{value}"')
    elif isinstance(value, bool):
        lines.append(f"{key} = {'true' if value else 'false'}")
    elif isinstance(value, int):
        lines.append(f"{key} = {value}")
    elif isinstance(value, float):
        lines.append(f"{key} = {value}")
    elif isinstance(value, list):
        if not value:
            lines.append(f"{key} = []")
        elif all(isinstance(v, str) for v in value):
            items = ", ".join(f'"{v}"' for v in value)
            if len(items) > 80:
                lines.append(f"{key} = [")
                for v in value:
                    lines.append(f'    "{v}",')
                lines.append("]")
            else:
                lines.append(f"{key} = [{items}]")
        else:
            lines.append(f"{key} = {json.dumps(value)}")
    elif value is None:
        pass  # Skip None values
    else:
        lines.append(f"{key} = {json.dumps(value)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public API: one-call generation from a results directory
# ---------------------------------------------------------------------------

def emit_policy_toml(
    results_path: str | Path,
    output_path: str | Path | None = None,
    policy_id: str | None = None,
    profile: str | None = None,
    enforcement_mode: str | None = None,
    name: str | None = None,
) -> Path:
    """
    Load autoharden results, generate an OVERT policy, and write policy.toml.

    Returns the path to the written file.
    """
    results_path = Path(results_path)
    results = load_autoharden_results(results_path)

    policy = generate_policy(
        results,
        policy_id=policy_id,
        profile=profile,
        enforcement_mode=enforcement_mode,
        name=name,
    )

    # Add header comment
    header = (
        "# OVERT Policy — Generated by autoredteam autoharden\n"
        "#\n"
        "# OVERT (Open Verification and Evaluation for Responsible Technology)\n"
        "# is an open standard for declarative AI governance policy.\n"
        "#\n"
        "# This file can be consumed by any OVERT-compatible enforcement engine,\n"
        "# or fed back into autoredteam for recursive hardening.\n"
        "#\n"
        f"# Source: {results_path}\n"
        f"# Generated: {datetime.now(timezone.utc).isoformat()}\n"
        "\n"
    )

    toml_str = header + to_toml(policy)

    if output_path is None:
        if results_path.is_dir():
            output_path = results_path / "policy.toml"
        else:
            output_path = results_path.parent / "policy.toml"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(toml_str)

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="emit_policy — Generate OVERT policy.toml from autoharden results"
    )
    parser.add_argument(
        "results_path",
        help="Path to autoharden results directory or autoharden_report.json",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path for policy.toml (default: <results_dir>/policy.toml)",
    )
    parser.add_argument(
        "--policy-id",
        default=None,
        help="Override policy ID",
    )
    parser.add_argument(
        "--profile",
        default=None,
        choices=["healthcare-ambient", "healthcare-general", "finserv-trading", "enterprise-general"],
        help="OVERT industry profile to apply",
    )
    parser.add_argument(
        "--enforcement-mode",
        default=None,
        choices=["shadow", "warn", "enforce", "strict"],
        help="Override enforcement mode (default: derived from governance tier)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Override policy name",
    )

    args = parser.parse_args()

    try:
        output = emit_policy_toml(
            results_path=args.results_path,
            output_path=args.output,
            policy_id=args.policy_id,
            profile=args.profile,
            enforcement_mode=args.enforcement_mode,
            name=args.name,
        )
        print(f"  OVERT policy written: {output}")
        return 0
    except FileNotFoundError as e:
        print(f"  Error: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"  Error: invalid JSON in results: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
