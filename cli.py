#!/usr/bin/env python3
"""
cli.py — Unified CLI entrypoint for autoredteam.

Usage:
    autoredteam run       --provider echo --model echo --pack generic_taxonomy
    autoredteam harden    --provider openai --model gpt-4o-mini --prompt-file prompt.txt
    autoredteam validate  --suite overnight --provider echo --model echo
    autoredteam report    --input results/campaign_result.json
    autoredteam pr        --input results/ --mode dry_run
    autoredteam providers list
    autoredteam packs     list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from autoredteam import __version__
from autoredteam.receipts import (
    ReceiptError,
    build_receipt_artifacts,
    generate_signing_key,
    load_public_key,
    load_signing_seed,
    verify_receipt_files,
)
from autoredteam.requirements import (
    RequirementError,
    load_requirement,
    parse_timestamp,
    requirement_digest,
    validate_autoredteam_requirement,
)


VERSION = __version__


def _print_banner():
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    autoredteam v{VERSION}                       ║
║         Automated Red-Teaming for AI Systems                 ║
║         Multi-cloud · Multi-turn · Stealth · Domain-aware    ║
╚══════════════════════════════════════════════════════════════╝
""")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoredteam",
        description="autoredteam — Automated red-teaming for AI systems",
    )
    parser.add_argument("--version", action="version", version=f"autoredteam {VERSION}")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_p = sub.add_parser("run", help="Run a red-team campaign")
    run_p.add_argument("--provider", default="echo", help="Provider ID (e.g. openai, anthropic, bedrock, echo)")
    run_p.add_argument("--model", default="gpt-4o-mini", help="Model name or alias")
    run_p.add_argument("--pack", "--packs", nargs="+", default=["generic_taxonomy"], help="Attack pack IDs")
    run_p.add_argument("--system-prompt", default="You are a helpful assistant.", help="System prompt")
    run_p.add_argument("--output-dir", default="results", help="Output directory")
    run_p.add_argument("--max-probes", type=int, default=20, help="Max probes per pack")
    run_p.add_argument("--max-trajectory-turns", type=int, default=5, help="Max turns per trajectory")
    run_p.add_argument("--stealth-profile", default="none", choices=["none", "light", "medium", "aggressive"])
    run_p.add_argument("--intensity", default="medium", choices=["low", "medium", "high"])
    run_p.add_argument(
        "--judge-backend",
        default="deterministic",
        choices=["deterministic", "slm"],
        help="Scoring backend; the public CLI supports deterministic scoring and an optional local SLM",
    )
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--resume", action="store_true", help="Resume interrupted campaign")
    run_p.add_argument("--dry-run", action="store_true", help="Use echo provider")
    run_p.add_argument("--quiet", action="store_true")
    run_p.add_argument(
        "--requirement",
        help="AI Evidence Requirement v0.1 JSON file to bind to the run receipt",
    )
    run_p.add_argument(
        "--signing-key",
        help="Local Ed25519 seed file created by 'autoredteam keygen' (required with --requirement)",
    )
    # Provider-specific
    run_p.add_argument("--endpoint", default="", help="API endpoint / base URL")
    run_p.add_argument("--deployment", default="", help="Azure deployment name")
    run_p.add_argument("--region", default="", help="Cloud region")
    run_p.add_argument("--project", default="", help="GCP project ID")
    run_p.add_argument("--account-id", default="", help="Cloudflare account ID")

    # --- validate ---
    val_p = sub.add_parser("validate", help="Run the public validation suite")
    val_p.add_argument("--suite", default="generic", choices=["generic", "overnight", "all"])
    val_p.add_argument("--provider", default="echo")
    val_p.add_argument("--model", default="gpt-4o-mini")
    val_p.add_argument("--system-prompt", default="You are a helpful assistant.")
    val_p.add_argument("--output-dir", default="results/validation")
    val_p.add_argument("--stealth-profile", default="none", choices=["none", "light", "medium", "aggressive"])
    val_p.add_argument("--dry-run", action="store_true")
    val_p.add_argument("--endpoint", default="")
    val_p.add_argument("--region", default="")
    val_p.add_argument("--project", default="")
    val_p.add_argument("--account-id", default="")

    # --- harden ---
    harden_p = sub.add_parser("harden", help="Unavailable in the OSS kernel")
    harden_p.add_argument("--provider", default="echo")
    harden_p.add_argument("--model", default="gpt-4o-mini")
    harden_p.add_argument("--pack", "--packs", nargs="+", default=["generic_taxonomy"])
    harden_p.add_argument("--prompt-file", default="", help="Path to system prompt file")
    harden_p.add_argument("--output-dir", default="results/harden")
    harden_p.add_argument("--target-score", type=int, default=80)
    harden_p.add_argument("--create-pr", action="store_true")
    harden_p.add_argument("--base-branch", default="main")
    harden_p.add_argument("--system-prompt", default="You are a helpful assistant.")
    harden_p.add_argument("--dry-run", action="store_true")
    harden_p.add_argument("--endpoint", default="")
    harden_p.add_argument("--region", default="")
    harden_p.add_argument("--project", default="")
    harden_p.add_argument("--account-id", default="")

    # --- report ---
    report_p = sub.add_parser("report", help="Unavailable in the OSS kernel")
    report_p.add_argument("--input", required=True, help="Path to campaign_result.json or results directory")
    report_p.add_argument("--output-dir", default="", help="Output directory (default: same as input)")

    # --- pr ---
    pr_p = sub.add_parser("pr", help="Unavailable in the OSS kernel")
    pr_p.add_argument("--input", required=True, help="Results directory")
    pr_p.add_argument("--mode", default="dry_run", choices=["dry_run", "gh_cli"])
    pr_p.add_argument("--base-branch", default="main")

    # --- emit-policy ---
    ep_p = sub.add_parser("emit-policy", help="Generate OVERT policy.toml from autoharden results")
    ep_p.add_argument("results_path", help="Path to autoharden results directory or report JSON")
    ep_p.add_argument("-o", "--output", default=None, help="Output path (default: <results_dir>/policy.toml)")
    ep_p.add_argument("--policy-id", default=None, help="Override policy ID")
    ep_p.add_argument("--profile", default=None,
                       choices=["healthcare-ambient", "healthcare-general", "finserv-trading", "enterprise-general"],
                       help="OVERT industry profile")
    ep_p.add_argument("--enforcement-mode", default=None,
                       choices=["shadow", "warn", "enforce", "strict"],
                       help="Override enforcement mode (default: derived from governance tier)")
    ep_p.add_argument("--name", default=None, help="Override policy name")

    # --- providers ---
    prov_p = sub.add_parser("providers", help="List available providers")
    prov_sub = prov_p.add_subparsers(dest="providers_action")
    prov_sub.add_parser("list", help="List all registered providers")

    # --- packs ---
    pack_p = sub.add_parser("packs", help="List available attack packs")
    pack_sub = pack_p.add_subparsers(dest="packs_action")
    pack_sub.add_parser("list", help="List all registered packs")

    # --- keygen ---
    key_p = sub.add_parser("keygen", help="Create a local receipt signing key")
    key_p.add_argument("--output", required=True, help="Private signing key path (created mode 0600)")
    key_p.add_argument(
        "--public-key-output",
        help="Optional path for the shareable Ed25519 public key",
    )

    # --- verify ---
    verify_p = sub.add_parser(
        "verify",
        help="Check a request-bound self-signed receipt and redacted artifact",
    )
    verify_p.add_argument("receipt", help="Path to receipt.json")
    verify_p.add_argument("--requirement", required=True, help="Original requirement.json")
    verify_p.add_argument("--artifact", required=True, help="Shareable redacted_evidence.json")
    verify_p.add_argument(
        "--trusted-public-key",
        help="Expected Ed25519 public-key hex or path; otherwise the receipt key is self-declared",
    )
    verify_p.add_argument(
        "--at",
        help="Verification time as UTC seconds (YYYY-MM-DDTHH:MM:SSZ); defaults to now",
    )
    verify_p.add_argument("--json", action="store_true", help="Print machine-readable verification result")

    return parser


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    """Execute a red-team campaign."""
    request = None
    signing_seed = None
    if getattr(args, "requirement", None):
        if not getattr(args, "signing_key", None):
            raise ReceiptError("--signing-key is required with --requirement; no unsigned fallback was used")
        request = load_requirement(args.requirement)
        validate_autoredteam_requirement(request)
        expires_at = parse_timestamp(request["return"]["expires_at"], "$.return.expires_at")
        if datetime.now(timezone.utc) >= expires_at:
            raise RequirementError("the requirement has expired; the campaign was not started")
        signing_seed = load_signing_seed(args.signing_key)
    elif getattr(args, "signing_key", None):
        raise ReceiptError(
            "--signing-key requires --requirement; no signed receipt was created"
        )

    _print_banner()

    from campaign import TargetRef, generate_campaign_id
    from attack_packs.base import PackBuildContext
    from attack_packs.registry import build_campaign_from_packs, get_pack_registry
    from campaign_runner import CampaignRunner, CampaignRunConfig
    from scoring_v2 import ScoreEngineV2, ScoreConfigV2
    from stealth import StealthEngine
    provider = "echo" if args.dry_run else args.provider
    target = TargetRef(
        provider=provider, model=args.model,
        system_prompt=args.system_prompt,
        endpoint=args.endpoint, deployment=args.deployment,
        region=args.region, project=args.project,
        account_id=args.account_id,
    )

    context = PackBuildContext(
        target=target, system_prompt=args.system_prompt,
        seed=args.seed, intensity=args.intensity,
        max_probes=args.max_probes,
        max_trajectory_turns=args.max_trajectory_turns,
        stealth_profile=args.stealth_profile,
    )

    shareable_output_dir = Path(args.output_dir)
    campaign_output_dir = shareable_output_dir
    if request is not None:
        if shareable_output_dir.is_symlink():
            raise ReceiptError("request-mode output root must not be a symlink")
        allowed_root_entries = {"private"}
        if shareable_output_dir.exists():
            unexpected = sorted(
                child.name
                for child in shareable_output_dir.iterdir()
                if child.name not in allowed_root_entries
            )
            if unexpected:
                raise ReceiptError(
                    "request-mode output root contains non-shareable or unknown entries: "
                    + ", ".join(unexpected)
                    + "; choose a clean output directory"
                )
        private_dir = shareable_output_dir / "private"
        campaign_output_dir = private_dir / "run"
        if private_dir.is_symlink() or campaign_output_dir.is_symlink():
            raise ReceiptError("request-mode private output path must not be a symlink")
        campaign_output_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(private_dir, 0o700)
        os.chmod(campaign_output_dir, 0o700)

    campaign = build_campaign_from_packs(
        pack_ids=args.pack, context=context, target=target,
        mode="run", output_dir=str(campaign_output_dir),
    )

    if not args.quiet:
        print(f"  Provider:  {provider}")
        print(f"  Model:     {args.model}")
        print(f"  Packs:     {', '.join(args.pack)}")
        print(f"  Probes:    {len(campaign.probes)}")
        print(f"  Stealth:   {args.stealth_profile}")
        print(f"  Judge:     {args.judge_backend}")
        print(f"  Output:    {args.output_dir}/")
        if request:
            print(f"  Request:   {request['request_id']}")
            print(f"  Digest:    {requirement_digest(request)}")
            unsupported_assurance = [
                predicate for predicate in request["minimum_assurance_predicates"]
                if predicate != "issuer-signed"
            ]
            if unsupported_assurance:
                print(
                    "  Assurance: incomplete — this self-signed adapter does not provide "
                    + ", ".join(unsupported_assurance)
                )
        print()

    stealth = StealthEngine(seed=args.seed) if args.stealth_profile != "none" else None
    score_config = ScoreConfigV2(judge_backend=args.judge_backend)
    runner = CampaignRunner(
        score_engine=ScoreEngineV2(config=score_config),
        stealth_engine=stealth,
        config=CampaignRunConfig(
            output_dir=str(campaign_output_dir),
            resume=args.resume,
            private_artifacts=request is not None,
        ),
    )

    result = runner.run_campaign(campaign)

    artifacts = None
    receipt_artifacts = None
    if request is not None and signing_seed is not None:
        # The public request-bound path intentionally does not import the
        # excluded attestation/reporting stack. It writes its private detail,
        # structurally bounded evidence, and SDK-offline signature itself.
        receipt_artifacts = build_receipt_artifacts(
            result,
            request,
            signing_seed,
            args.output_dir,
        )
    else:
        try:
            from reporting.generator import ReportGenerator
            reporter = ReportGenerator()
            artifacts = reporter.generate(result, args.output_dir)
        except Exception:
            artifacts = None

    # Print summary
    if result.summary:
        s = result.summary
        governance = None
        if request is None:
            try:
                from reporting.governance import compute_governance_score
                governance = compute_governance_score(result.results)
            except Exception:
                pass

        print(f"\n{'='*60}")
        print(f"  RESULTS")
        print(f"{'='*60}")
        print(f"  Total probes:  {s.total_probes}")
        print(f"  Bypassed:      {s.bypassed} ({s.asr}% ASR)")
        print(f"  Blocked:       {s.blocked}")
        print(f"  Passed:        {s.passed}")
        print(f"  Errors:        {s.errors}")
        if governance:
            print(f"  Governance:    {governance.score}/100 (Tier {governance.tier})")
        print(f"  Best score:    {s.best_combined_score}")
        print()
        if artifacts:
            print(f"  📄 Report:     {artifacts.report_md}")
            print(f"  📊 Findings:   {artifacts.findings_jsonl}")
            print(f"  📋 Summary:    {artifacts.summary_txt}")
        elif receipt_artifacts:
            print(f"  🔒 Private:    {receipt_artifacts.private_findings}")
            print(f"  📋 Shareable:  {receipt_artifacts.redacted_evidence}")
            print(f"  🧾 Receipt:    {receipt_artifacts.receipt}")
            print(f"  Fulfillment:   {receipt_artifacts.verification['status']}")
            print(
                "  Signature:     key signed this claim; execution and time were not "
                "independently proven"
            )
        else:
            print(f"  📦 Campaign:   {args.output_dir}/campaign_result.json")
            print(f"  🧾 Results:    {args.output_dir}/probe_results.jsonl")
        print()

    if not result.summary or result.summary.errors != 0:
        return 1
    if receipt_artifacts and not receipt_artifacts.verification["valid"]:
        return 3
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Run a validation suite."""
    _print_banner()

    suite_to_packs = {
        "generic": ["generic_taxonomy"],
        "overnight": ["generic_taxonomy", "healthcare", "finance", "hr", "coding_agents"],
        "all": ["generic_taxonomy", "healthcare", "finance", "hr", "coding_agents"],
    }

    packs = suite_to_packs.get(args.suite, ["generic_taxonomy"])
    print(f"  Suite: {args.suite} → packs: {', '.join(packs)}")

    # Delegate to run with the appropriate packs
    args.pack = packs
    args.max_probes = 20
    args.max_trajectory_turns = 5
    args.intensity = "high"
    args.judge_backend = "deterministic"
    args.seed = 42
    args.resume = False
    args.quiet = False
    args.deployment = ""
    if not hasattr(args, "endpoint"):
        args.endpoint = ""

    return cmd_run(args)


def cmd_harden(args: argparse.Namespace) -> int:
    """Auto-harden is intentionally kept out of the OSS kernel."""
    print("harden is not available in the OSS kernel.")
    return 2


def cmd_report(args: argparse.Namespace) -> int:
    """Report generation is intentionally kept out of the OSS kernel."""
    print("report is not available in the OSS kernel.")
    return 2


def cmd_pr(args: argparse.Namespace) -> int:
    """PR creation is intentionally kept out of the OSS kernel."""
    print("pr is not available in the OSS kernel.")
    return 2


def cmd_providers_list(args: argparse.Namespace) -> int:
    """List all registered providers."""
    from providers.registry import get_provider_registry
    registry = get_provider_registry()
    providers = registry.list_providers()

    print(f"\n  Available Providers ({len(providers)}):")
    print(f"  {'─'*50}")
    for p in providers:
        families = ", ".join(p.supported_families) if p.supported_families else "any"
        required = ", ".join(p.required_fields) if p.required_fields else "none"
        print(f"  {p.provider_id:<20} {p.display_name:<25} auth={p.auth_mode}")
        print(f"  {'':20} families: {families}")
        print(f"  {'':20} required: {required}")
        print()
    return 0


def cmd_emit_policy(args: argparse.Namespace) -> int:
    """Generate OVERT policy.toml from autoharden results."""
    from emit_policy import emit_policy_toml

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


def cmd_packs_list(args: argparse.Namespace) -> int:
    """List all registered attack packs."""
    from attack_packs.registry import get_pack_registry
    registry = get_pack_registry()
    packs = registry.list()

    print(f"\n  Available Attack Packs ({len(packs)}):")
    print(f"  {'─'*50}")
    for p in packs:
        surfaces = ", ".join(s.value for s in p.surfaces)
        print(f"  {p.pack_id:<20} {p.display_name}")
        print(f"  {'':20} {p.description[:60]}")
        print(f"  {'':20} surfaces: {surfaces}")
        print(f"  {'':20} categories: {len(p.categories)}")
        print()
    return 0


def cmd_keygen(args: argparse.Namespace) -> int:
    public_key = generate_signing_key(args.output, args.public_key_output)
    print(f"Private signing key created: {args.output}")
    if args.public_key_output:
        print(f"Public key written: {args.public_key_output}")
    print(f"Ed25519 public key: {public_key}")
    print("Keep the private seed local. A receipt verifier needs only the public key.")
    return 0


def _verification_output(result: dict) -> dict:
    """Return the deliberately public, machine-readable verification view.

    The verifier's library result also carries raw SDK and artifact error text
    for local debugging. Those strings may contain filesystem details and do
    not belong in a shareable CLI transcript. Likewise, expose a stable hash of
    the public signing key instead of echoing receipt-controlled key material.
    """
    signer_public_key = result.get("signer_public_key")
    signer_key_fingerprint = (
        "sha256:"
        + hashlib.sha256(signer_public_key.encode("ascii")).hexdigest()
        if signer_public_key
        else "unavailable"
    )
    return {
        "valid": result["valid"],
        "status": result["status"],
        "checks": result["checks"],
        "signer_key_fingerprint": signer_key_fingerprint,
        "key_trust": result["key_trust"],
        "witness_status": result["witness_status"],
        "current_revocation_status": result["current_revocation_status"],
        "signature_explanation": result["signature_explanation"],
        "buyer_signature_verification": result["buyer_signature_verification"],
        "compatibility": result["compatibility"],
        "content_check_scope": result["content_check_scope"],
    }


def cmd_verify(args: argparse.Namespace) -> int:
    verification_time = None
    if args.at:
        verification_time = parse_timestamp(args.at, "--at")
    trusted_key = load_public_key(args.trusted_public_key) if args.trusted_public_key else None
    result = verify_receipt_files(
        args.receipt,
        args.requirement,
        args.artifact,
        trusted_public_key=trusted_key,
        now=verification_time,
    )
    public_result = _verification_output(result)
    if args.json:
        print(json.dumps(public_result, indent=2))
    else:
        print(f"Fulfillment: {result['status']}")
        print(
            "Signature: "
            + ("PASS" if result["checks"]["signature_valid_under_displayed_key"] else "FAIL")
        )
        print(f"Signer key fingerprint: {public_result['signer_key_fingerprint']}")
        print(f"Key trust: {result['key_trust']}")
        print(result["signature_explanation"])
        failed = [name for name, passed in result["checks"].items() if not passed]
        if failed:
            print("Failed checks: " + ", ".join(failed))
    return 0 if result["valid"] else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    handlers = {
        "run": cmd_run,
        "validate": cmd_validate,
        "harden": cmd_harden,
        "report": cmd_report,
        "pr": cmd_pr,
        "emit-policy": cmd_emit_policy,
        "keygen": cmd_keygen,
        "verify": cmd_verify,
        "providers": lambda a: cmd_providers_list(a) if getattr(a, "providers_action", None) == "list" else (print("Use: autoredteam providers list"), 0)[1],
        "packs": lambda a: cmd_packs_list(a) if getattr(a, "packs_action", None) == "list" else (print("Use: autoredteam packs list"), 0)[1],
    }

    handler = handlers.get(args.command)
    if handler:
        try:
            return handler(args)
        except KeyboardInterrupt:
            print("\n⚠ Interrupted")
            return 130
        except (RequirementError, ReceiptError) as e:
            print(f"\n❌ Error: {e}")
            return 2
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return 2
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
