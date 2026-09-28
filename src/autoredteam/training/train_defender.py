#!/usr/bin/env python3
"""Local defender LoRA training scaffold."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from autoredteam.training.lora_trainer import (
    LoRATrainer,
    _ML_STACK_AVAILABLE,
    format_defender_completion,
    format_defender_prompt,
)


@dataclass
class DefenderTrainingConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    data_path: str = "training_data/defender_examples.jsonl"
    output_root: str = "models"
    adapter_name: str = "defender-v1"
    cycle: int = 1
    epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    max_length: int = 1024
    eval_fraction: float = 0.1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_examples(path: str, limit: Optional[int] = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    input_path = Path(path)
    if not input_path.exists():
        return records
    for line in input_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if limit is not None and len(records) >= limit:
            break
    return records


def format_prompt(example: dict[str, Any]) -> str:
    return format_defender_prompt(example)


def format_completion(example: dict[str, Any]) -> str:
    return format_defender_completion(example)


def build_manifest(config: DefenderTrainingConfig, examples: list[dict[str, Any]]) -> dict[str, Any]:
    severity_buckets = {"0-19": 0, "20-39": 0, "40-59": 0, "60-79": 0, "80-100": 0}
    category_counts: dict[str, int] = {}
    attack_count = 0
    benign_count = 0

    for example in examples:
        label = example.get("label", {}) if isinstance(example.get("label"), dict) else {}
        severity = float(label.get("severity", 0.0))
        category = str(label.get("category") or example.get("category", "unknown"))
        is_attack = bool(label.get("is_attack", severity >= 40.0))
        category_counts[category] = category_counts.get(category, 0) + 1
        attack_count += int(is_attack)
        benign_count += int(not is_attack)
        if severity < 20:
            severity_buckets["0-19"] += 1
        elif severity < 40:
            severity_buckets["20-39"] += 1
        elif severity < 60:
            severity_buckets["40-59"] += 1
        elif severity < 80:
            severity_buckets["60-79"] += 1
        else:
            severity_buckets["80-100"] += 1

    return {
        "tool": "autoredteam",
        "task": "defender_slm_training",
        "config": config.to_dict(),
        "example_count": len(examples),
        "attack_count": attack_count,
        "benign_count": benign_count,
        "category_counts": dict(sorted(category_counts.items())),
        "severity_buckets": severity_buckets,
        "ml_stack_available": _ML_STACK_AVAILABLE,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the defender SLM")
    parser.add_argument("--data", default="training_data/defender_examples.jsonl")
    parser.add_argument("--output-root", default="models")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-name", default="defender-v1")
    parser.add_argument("--cycle", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = DefenderTrainingConfig(
        base_model=args.base_model,
        data_path=args.data,
        output_root=args.output_root,
        adapter_name=args.adapter_name,
        cycle=args.cycle,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_length=args.max_length,
        eval_fraction=args.eval_fraction,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        dry_run=args.dry_run,
    )

    examples = load_examples(config.data_path)
    manifest = build_manifest(config, examples)

    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "defender_training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

    if config.dry_run:
        return 0

    trainer = LoRATrainer(
        base_defender_model=config.base_model,
        output_root=config.output_root,
        epochs=config.epochs,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        max_length=config.max_length,
        eval_fraction=config.eval_fraction,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
    )
    adapter = trainer.train_defender(examples, name=config.adapter_name, cycle=config.cycle)
    metrics = trainer.evaluate_adapter(adapter, examples)
    report_path = output_root / "defender_training_report.json"
    report_path.write_text(json.dumps({"adapter": adapter.to_dict(), "metrics": metrics.to_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
