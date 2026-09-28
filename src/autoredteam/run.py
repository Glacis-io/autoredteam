#!/usr/bin/env python3
"""Public closed-loop entry point for autoredteam."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from autoredteam.attestation import AttestationManager, load_attestation_config
from autoredteam.autoharden import autoharden


DEFAULT_CONFIG: dict[str, Any] = {
    "target": {"type": "echo", "params": {}},
    "run": {"max_cycles": 10, "batch_size": 10, "verbose": True},
    "scoring": {
        "judge_backend": "deterministic",
        "judge_model": "gpt-4.1-mini",
        "judge_model_path": "models/judge-v2",
    },
    "autoharden": {
        "target_score": 700,
        "attack_cycles": 3,
        "immune_enabled": False,
        "immune_interval": 5,
        "immune_threshold": 50,
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config_path = Path(path)
    if not config_path.exists():
        return config
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return _merge(config, loaded)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the public autoredteam hardening loop")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--target", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--role-name", default=None)
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-score", type=int, default=None)
    parser.add_argument("--attack-cycles", type=int, default=None)
    parser.add_argument("--judge-backend", choices=["deterministic", "api", "slm"], default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-model-path", default=None)
    parser.add_argument("--immune", dest="immune", action="store_true")
    parser.add_argument("--no-immune", dest="immune", action="store_false")
    parser.add_argument("--immune-interval", type=int, default=None)
    parser.add_argument("--immune-threshold", type=int, default=None)
    parser.add_argument("--attest", action="store_true",
                        help="Emit attestation_receipt.json after the run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.set_defaults(immune=None)
    args = parser.parse_args()

    config = load_config(args.config)
    target_cfg = config.get("target", {})
    target_params = target_cfg.get("params") or {}
    run_cfg = config.get("run", {})
    scoring_cfg = config.get("scoring", {})
    harden_cfg = config.get("autoharden", {})

    target_type = "echo" if args.dry_run else (args.target or target_cfg.get("type", "echo"))
    model = args.model or target_params.get("model", "")
    system_prompt = args.system_prompt or target_params.get("system_prompt", "You are a helpful assistant.")
    role_name = args.role_name or target_params.get("role_name", "this AI assistant")
    max_cycles = args.cycles or run_cfg.get("max_cycles", 10)
    batch_size = args.batch_size or run_cfg.get("batch_size", 10)
    target_score = args.target_score or harden_cfg.get("target_score", 700)
    attack_cycles = args.attack_cycles or harden_cfg.get("attack_cycles", 3)
    judge_backend = args.judge_backend or scoring_cfg.get("judge_backend", "deterministic")
    judge_model = args.judge_model or scoring_cfg.get("judge_model", "gpt-4.1-mini")
    judge_model_path = args.judge_model_path or scoring_cfg.get("judge_model_path", "models/judge-v2")
    immune_enabled = harden_cfg.get("immune_enabled", False) if args.immune is None else args.immune
    immune_interval = args.immune_interval or harden_cfg.get("immune_interval", 5)
    immune_threshold = args.immune_threshold or harden_cfg.get("immune_threshold", 50)

    if target_type == "openai" and not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for the OpenAI target")
    if target_type == "anthropic" and not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is required for the Anthropic target")
    if target_type == "gemini" and not args.dry_run and not (
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    ):
        raise SystemExit("GOOGLE_API_KEY or GEMINI_API_KEY is required for the Gemini target")

    result = autoharden(
        target_type=target_type,
        model=model,
        system_prompt=system_prompt,
        role_name=role_name,
        max_cycles=max_cycles,
        target_score=target_score,
        batch_size=batch_size,
        attack_cycles=attack_cycles,
        dry_run=args.dry_run,
        verbose=not args.quiet and run_cfg.get("verbose", True),
        immune_enabled=immune_enabled,
        immune_interval=immune_interval,
        immune_threshold=immune_threshold,
        judge_backend=judge_backend,
        judge_model=judge_model,
        judge_model_path=judge_model_path,
    )

    if args.attest:
        attest_cfg = load_attestation_config(args.config)
        mgr = AttestationManager(output_dir="results/autoharden", config=attest_cfg)
        receipt_path = mgr.write_receipt(metadata={
            "target_type": target_type,
            "model": model,
            "cycles": result.get("cycles", 0),
        })
        print(f"  📄 Attestation receipt: {receipt_path}")

    print(
        json.dumps(
            {
                "final_metrics": result.get("final_metrics", {}),
                "training_data": result.get("training_data", {}),
                "chain_verified": result.get("chain_verified", False),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
