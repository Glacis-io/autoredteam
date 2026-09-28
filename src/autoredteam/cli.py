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
import sys
from typing import Optional

from autoredteam import __version__


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
    run_p.add_argument("--judge-backend", default="deterministic", choices=["deterministic", "api", "slm"])
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--resume", action="store_true", help="Resume interrupted campaign")
    run_p.add_argument("--dry-run", action="store_true", help="Use echo provider")
    run_p.add_argument("--quiet", action="store_true")
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

    return parser


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    """Execute a red-team campaign."""
    _print_banner()

    from autoredteam.campaign import TargetRef
    from autoredteam.attack_packs.base import PackBuildContext
    from autoredteam.attack_packs.registry import build_campaign_from_packs
    from autoredteam.campaign_runner import CampaignRunner, CampaignRunConfig
    from autoredteam.scoring_v2 import ScoreEngineV2, ScoreConfigV2
    from autoredteam.stealth import StealthEngine
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

    campaign = build_campaign_from_packs(
        pack_ids=args.pack, context=context, target=target,
        mode="run", output_dir=args.output_dir,
    )

    if not args.quiet:
        print(f"  Provider:  {provider}")
        print(f"  Model:     {args.model}")
        print(f"  Packs:     {', '.join(args.pack)}")
        print(f"  Probes:    {len(campaign.probes)}")
        print(f"  Stealth:   {args.stealth_profile}")
        print(f"  Judge:     {args.judge_backend}")
        print(f"  Output:    {args.output_dir}/")
        print()

    stealth = StealthEngine(seed=args.seed) if args.stealth_profile != "none" else None
    score_config = ScoreConfigV2(judge_backend=args.judge_backend)
    runner = CampaignRunner(
        score_engine=ScoreEngineV2(config=score_config),
        stealth_engine=stealth,
        config=CampaignRunConfig(output_dir=args.output_dir, resume=args.resume),
    )

    result = runner.run_campaign(campaign)

    artifacts = None
    try:
        from autoredteam.reporting.generator import ReportGenerator
        reporter = ReportGenerator()
        artifacts = reporter.generate(result, args.output_dir)
    except Exception:
        artifacts = None

    # Print summary
    if result.summary:
        s = result.summary
        governance = None
        try:
            from autoredteam.reporting.governance import compute_governance_score
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
        else:
            print(f"  📦 Campaign:   {args.output_dir}/campaign_result.json")
            print(f"  🧾 Results:    {args.output_dir}/probe_results.jsonl")
        print()

    return 0 if result.summary and result.summary.errors == 0 else 1


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
    from autoredteam.providers.registry import get_provider_registry
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
    from autoredteam.emit_policy import emit_policy_toml

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
    from autoredteam.attack_packs.registry import get_pack_registry
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _harden_console_encoding() -> None:
    # Banners and status lines use box-drawing characters and emoji, which
    # crash on legacy code pages (e.g. cp1252 on Windows CI runners).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: Optional[list[str]] = None) -> int:
    _harden_console_encoding()
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
