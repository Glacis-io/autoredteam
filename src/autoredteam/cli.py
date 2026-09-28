#!/usr/bin/env python3
"""
cli.py — Unified CLI entrypoint for autoredteam.

Usage:
    autoredteam run       --provider echo --model echo --pack generic_taxonomy
    autoredteam harden    --provider openai --model gpt-5.6-luna --prompt-file prompt.txt
    autoredteam validate  --suite overnight --provider echo --model echo
    autoredteam emit-policy results/autoharden/
    autoredteam providers list
    autoredteam packs     list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from autoredteam import __version__


VERSION = __version__

STEALTH_PROFILES = ["none", "light", "medium", "aggressive"]
JUDGE_BACKENDS = ["deterministic", "api", "slm"]
SEVERITIES = ["low", "medium", "high", "critical"]
EXIT_OK, EXIT_GATE_FAILED, EXIT_ERROR = 0, 1, 2
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MODELS = {
    "openai": "gpt-5.6-luna",
    "azure_openai": "gpt-5.6-luna",
    "anthropic": "claude-haiku-4-5",
    "google": "gemini-3.8-flash",
    "bedrock": "claude-haiku-4-5",
    "cloudflare": "llama-3.3-70b",
    "echo": "echo",
    "http": "",
}


def _print_banner():
    title = f"autoredteam v{VERSION}".center(62)
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║{title}║
║         Automated Red-Teaming for AI Systems                 ║
║         Multi-cloud · Multi-turn · Stealth · Domain-aware    ║
╚══════════════════════════════════════════════════════════════╝
""")


def _add_target_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("target")
    g.add_argument("--provider", default="echo", help="Provider ID (see `autoredteam providers list`)")
    g.add_argument("--model", default="", help="Model name or catalog alias (default: per-provider)")
    g.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt for the target")
    g.add_argument("--prompt-file", default="", help="Read the system prompt from a file")
    g.add_argument("--endpoint", default="", help="API endpoint / base URL")
    g.add_argument("--deployment", default="", help="Azure deployment name")
    g.add_argument("--region", default="", help="Cloud region")
    g.add_argument("--project", default="", help="GCP project ID")
    g.add_argument("--account-id", default="", help="Cloudflare account ID")
    g.add_argument("--dry-run", action="store_true", help="Use the offline echo provider (no API keys)")
    h = p.add_argument_group("http provider (--provider http)")
    h.add_argument("--http-header", action="append", default=[], metavar="'Name: value'",
                   help="Request header, repeatable. Use {{env.VAR}} for secrets")
    h.add_argument("--http-body", default="",
                   help="JSON request template or @file. Placeholders: {{prompt}}, {{messages}}, {{history}}, "
                        "{{system_prompt}}, {{session_id}}, {{env.VAR}}")
    h.add_argument("--http-response-path", default="", help="Path to reply text, e.g. choices[0].message.content")
    h.add_argument("--http-method", default="POST")
    h.add_argument("--http-timeout", type=float, default=60.0)


def _add_judge_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("judge")
    g.add_argument("--judge-backend", default="deterministic", choices=JUDGE_BACKENDS)
    g.add_argument("--judge-model", default="gpt-5.6-luna",
                   help="Frontier judge model used when --judge-backend=api")
    g.add_argument("--judge-model-path", default="models/judge-v2",
                   help="Local judge SLM checkpoint used when --judge-backend=slm")


def _add_campaign_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output-dir", default="results", help="Output directory")
    p.add_argument("--max-probes", type=int, default=20, help="Max probes per pack")
    p.add_argument("--max-trajectory-turns", type=int, default=5, help="Max turns per trajectory")
    p.add_argument("--stealth-profile", default="none", choices=STEALTH_PROFILES)
    p.add_argument("--intensity", default="medium", choices=["low", "medium", "high"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="Resume interrupted campaign")
    p.add_argument("--attest", action="store_true", help="Write attestation_receipt.json for the evidence chain")
    p.add_argument("--quiet", action="store_true")
    ci = p.add_argument_group("CI / reporting")
    ci.add_argument("--format", nargs="*", default=[], choices=["sarif", "junit"],
                    help="Extra outputs: results.sarif (GitHub code scanning) and/or junit.xml")
    ci.add_argument("--sarif-artifact", default=None,
                    help="Repo-relative file SARIF alerts point at (default: --prompt-file, if given)")
    ci.add_argument("--fail-on", default="none", choices=["none", "any", *SEVERITIES],
                    help="Exit 1 if any bypass at or above this severity is found")
    ci.add_argument("--max-asr", type=float, default=None,
                    help="Exit 1 if the attack success rate (%%) exceeds this value")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoredteam",
        description="autoredteam — Automated red-teaming for AI systems",
    )
    parser.add_argument("--version", action="version", version=f"autoredteam {VERSION}")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_p = sub.add_parser("run", help="Run a red-team campaign")
    _add_target_args(run_p)
    run_p.add_argument("--pack", "--packs", nargs="+", default=["generic_taxonomy"], help="Attack pack IDs")
    _add_campaign_args(run_p)
    _add_judge_args(run_p)

    # --- validate ---
    val_p = sub.add_parser("validate", help="Run a predefined validation suite")
    _add_target_args(val_p)
    val_p.add_argument("--suite", default="generic", choices=["generic", "overnight", "all"])
    _add_campaign_args(val_p)
    _add_judge_args(val_p)
    val_p.set_defaults(output_dir="results/validation", intensity="high")

    # --- harden ---
    harden_p = sub.add_parser("harden", help="Run the closed-loop attack → heal → verify hardening loop")
    _add_target_args(harden_p)
    harden_p.add_argument("--from-policy", default=None,
                          help="Start from a prior OVERT policy.toml (recursive hardening)")
    harden_p.add_argument("--role-name", default="this AI assistant")
    harden_p.add_argument("--output-dir", default="results/autoharden")
    harden_p.add_argument("--training-data-dir", default="training_data")
    harden_p.add_argument("--cycles", type=int, default=10)
    harden_p.add_argument("--target-score", type=int, default=700, help="Governance score goal (0-1000)")
    harden_p.add_argument("--batch-size", type=int, default=12)
    harden_p.add_argument("--attack-cycles", type=int, default=3)
    harden_p.add_argument("--autonomous", action="store_true", help="Loop until interrupted")
    harden_p.add_argument("--immune", action="store_true", help="Enable the continual LoRA update loop")
    harden_p.add_argument("--immune-interval", type=int, default=5)
    harden_p.add_argument("--immune-threshold", type=int, default=50)
    harden_p.add_argument("--attest", action="store_true", help="Write attestation_receipt.json")
    harden_p.add_argument("--quiet", action="store_true")
    _add_judge_args(harden_p)

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

def _resolve_model(args: argparse.Namespace, provider: str) -> str:
    return args.model or DEFAULT_MODELS.get(provider, "")


def _resolve_system_prompt(args: argparse.Namespace) -> str:
    if getattr(args, "prompt_file", ""):
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    return args.system_prompt


def _target_params(args: argparse.Namespace) -> dict:
    """Provider-specific connection fields that were actually supplied."""
    fields = ("endpoint", "deployment", "region", "project", "account_id")
    params = {f: getattr(args, f) for f in fields if getattr(args, f, "")}
    metadata = _http_metadata(args)
    if metadata:
        params["metadata"] = metadata
    return params


def _http_metadata(args: argparse.Namespace) -> dict:
    if getattr(args, "provider", "") != "http" or getattr(args, "dry_run", False):
        return {}
    headers = {}
    for raw in args.http_header:
        name, sep, value = raw.partition(":")
        if not sep:
            raise ValueError(f"--http-header must look like 'Name: value', got {raw!r}")
        headers[name.strip()] = value.strip()
    body = args.http_body
    if body.startswith("@"):
        body = Path(body[1:]).read_text(encoding="utf-8")
    meta: dict = {"http_method": args.http_method, "http_timeout": args.http_timeout}
    if headers:
        meta["http_headers"] = headers
    if body:
        meta["http_body"] = body
    if args.http_response_path:
        meta["http_response_path"] = args.http_response_path
    return meta


def cmd_run(args: argparse.Namespace) -> int:
    """Execute a red-team campaign."""
    _print_banner()

    from autoredteam.attack_packs.base import PackBuildContext
    from autoredteam.attack_packs.registry import build_campaign_from_packs
    from autoredteam.attestation import AttestationManager
    from autoredteam.campaign import TargetRef
    from autoredteam.campaign_runner import CampaignRunConfig, CampaignRunner
    from autoredteam.scoring_v2 import ScoreConfigV2, ScoreEngineV2
    from autoredteam.stealth import StealthEngine

    provider = "echo" if args.dry_run else args.provider
    args.model = _resolve_model(args, provider)
    system_prompt = _resolve_system_prompt(args)
    target = TargetRef(
        provider=provider, model=args.model,
        system_prompt=system_prompt,
        **_target_params(args),
    )

    context = PackBuildContext(
        target=target, system_prompt=system_prompt,
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

    attestation = AttestationManager(output_dir=args.output_dir)
    if not args.resume:
        attestation.local.clear()

    stealth = StealthEngine(seed=args.seed) if args.stealth_profile != "none" else None
    score_config = ScoreConfigV2(
        judge_backend=args.judge_backend,
        judge_model=args.judge_model,
        judge_model_path=args.judge_model_path,
        use_api_judge=args.judge_backend == "api",
    )
    runner = CampaignRunner(
        score_engine=ScoreEngineV2(config=score_config),
        stealth_engine=stealth,
        attestation=attestation,
        config=CampaignRunConfig(output_dir=args.output_dir, resume=args.resume),
    )

    result = runner.run_campaign(campaign)

    artifacts = None
    try:
        from autoredteam.reporting.generator import ReportGenerator
        artifacts = ReportGenerator().generate(
            result, args.output_dir, formats=args.format,
            sarif_artifact_uri=args.sarif_artifact or (args.prompt_file or None),
        )
    except Exception as e:
        print(f"  ⚠ Report generation failed: {e}", file=sys.stderr)

    receipt_path = None
    if args.attest:
        receipt_path = attestation.write_receipt(metadata={
            "campaign_id": campaign.campaign_id,
            "provider": provider,
            "model": args.model,
            "packs": list(args.pack),
        })

    if result.summary:
        s = result.summary
        governance = None
        try:
            from autoredteam.reporting.governance import compute_governance_score
            governance = compute_governance_score(result.results)
        except Exception:
            pass

        print(f"\n{'='*60}")
        print("  RESULTS")
        print(f"{'='*60}")
        print(f"  Total probes:  {s.total_probes}")
        print(f"  Bypassed:      {s.bypassed} ({s.asr}% ASR)")
        print(f"  Blocked:       {s.blocked}")
        print(f"  Passed:        {s.passed}")
        print(f"  Errors:        {s.errors}")
        if governance:
            print(f"  Governance:    {governance.score}/100 (Tier {governance.tier})")
        print(f"  Best score:    {s.best_combined_score}")
        print(f"  Evidence:      {attestation.get_chain_length()} records, "
              f"chain {'verified' if attestation.local.verify_chain() else 'BROKEN'}")
        print()
        _print_framework_summary(result.results)
        if artifacts:
            print(f"  📄 Report:     {artifacts.report_md}")
            print(f"  📊 Findings:   {artifacts.findings_jsonl}")
            print(f"  📋 Summary:    {artifacts.summary_txt}")
            if artifacts.sarif:
                print(f"  🛡  SARIF:      {artifacts.sarif}")
            if artifacts.junit_xml:
                print(f"  🧪 JUnit:      {artifacts.junit_xml}")
        else:
            print(f"  📦 Campaign:   {args.output_dir}/campaign_result.json")
            print(f"  🧾 Results:    {args.output_dir}/probe_results.jsonl")
        print(f"  🔗 Evidence:   {attestation.local.evidence_file}")
        if receipt_path:
            print(f"  🧾 Receipt:    {receipt_path}")
        print()

    return _exit_code(result, args)


def _print_framework_summary(results: list) -> None:
    from autoredteam.frameworks import coverage

    cov = coverage(results, "owasp_llm")
    if not cov:
        return
    cells = [f"{fid} {'✗' if stats['bypassed'] else '✓'}" for fid, stats in cov.items()]
    print(f"  OWASP LLM:     {'  '.join(cells)}")
    agentic = coverage(results, "owasp_agentic")
    if agentic:
        cells = [f"{fid} {'✗' if stats['bypassed'] else '✓'}" for fid, stats in agentic.items()]
        print(f"  OWASP Agentic: {'  '.join(cells)}")
    print()


def _exit_code(result, args: argparse.Namespace) -> int:
    """0 = passed, 1 = security gate failed, 2 = probes errored (config/infra problem)."""
    from autoredteam.reporting.findings import bypassed_results, probe_severity, severity_at_least

    summary = result.summary
    if summary is None or summary.errors:
        return EXIT_ERROR
    reasons = []
    bypassed = bypassed_results(result.results)
    if args.fail_on != "none" and bypassed:
        threshold = "low" if args.fail_on == "any" else args.fail_on
        hits = [r for r in bypassed if severity_at_least(probe_severity(r.probe), threshold)]
        if hits:
            reasons.append(f"{len(hits)} bypass(es) at or above '{threshold}' severity")
    if args.max_asr is not None and summary.asr > args.max_asr:
        reasons.append(f"ASR {summary.asr}% exceeds --max-asr {args.max_asr}%")
    if reasons:
        print(f"  ✗ Security gate failed: {'; '.join(reasons)}")
        return EXIT_GATE_FAILED
    return EXIT_OK


SUITE_PACKS = {
    "generic": ["generic_taxonomy"],
    "overnight": ["generic_taxonomy", "healthcare", "finance", "hr", "coding_agents"],
    "all": ["generic_taxonomy", "healthcare", "finance", "hr", "coding_agents"],
}


def cmd_validate(args: argparse.Namespace) -> int:
    """Run a validation suite by delegating to `run` with the suite's packs."""
    args.pack = SUITE_PACKS.get(args.suite, ["generic_taxonomy"])
    print(f"  Suite: {args.suite} → packs: {', '.join(args.pack)}")
    return cmd_run(args)


def cmd_harden(args: argparse.Namespace) -> int:
    """Closed-loop hardening: attack, heal the worst cluster, verify, keep or discard."""
    from autoredteam.autoharden import autoharden

    system_prompt = _resolve_system_prompt(args)
    role_name = args.role_name
    if args.from_policy:
        from autoredteam.emit_policy import load_policy
        prior = load_policy(args.from_policy)
        system_prompt = prior["system_prompt"]
        role_name = prior["role_name"]
        print(f"  Loaded prior OVERT policy: {args.from_policy}")

    provider = "echo" if args.dry_run else args.provider
    args.model = _resolve_model(args, provider)
    result = autoharden(
        target_type=provider,
        model=args.model,
        system_prompt=system_prompt,
        role_name=role_name,
        max_cycles=args.cycles,
        target_score=args.target_score,
        batch_size=args.batch_size,
        attack_cycles=args.attack_cycles,
        autonomous=args.autonomous,
        dry_run=args.dry_run,
        verbose=not args.quiet,
        immune_enabled=args.immune,
        immune_interval=args.immune_interval,
        immune_threshold=args.immune_threshold,
        judge_backend=args.judge_backend,
        judge_model=args.judge_model,
        judge_model_path=args.judge_model_path,
        output_dir=args.output_dir,
        training_data_dir=args.training_data_dir,
        target_params=_target_params(args),
    )

    if args.attest:
        from autoredteam.attestation import AttestationManager
        mgr = AttestationManager(output_dir=args.output_dir)
        receipt_path = mgr.write_receipt(metadata={
            "provider": provider,
            "model": args.model,
            "cycles": result.get("cycles", 0),
        })
        print(f"  📄 Attestation receipt: {receipt_path}")

    return 0 if result.get("chain_verified", True) else 1


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
