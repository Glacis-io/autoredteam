#!/usr/bin/env python3
"""Local attacker LoRA training scaffold."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from autoredteam.training.lora_trainer import LoRATrainer, _ML_STACK_AVAILABLE


@dataclass
class AttackerTrainingConfig:
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    data_path: str = "training_data/attacker_examples.jsonl"
    output_root: str = "models"
    adapter_name: str = "attacker-v1"
    cycle: int = 1
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    max_length: int = 2048
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
    for line in input_path.read_text(encoding="utf-8").splitlines():
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


def _extract_target(example: dict[str, Any]) -> str:
    if "target_description" in example:
        return str(example["target_description"])
    input_block = example.get("input")
    if isinstance(input_block, dict):
        return str(input_block.get("prompt", ""))
    return ""


def _extract_category(example: dict[str, Any]) -> str:
    if "category" in example:
        return str(example["category"])
    input_block = example.get("input")
    if isinstance(input_block, dict):
        return str(input_block.get("category", "unknown"))
    return "unknown"


def _extract_strategy(example: dict[str, Any]) -> str:
    return str(example.get("strategy", "crescendo"))


def _extract_prior_failures(example: dict[str, Any]) -> list[str]:
    value = example.get("prior_failures", [])
    return [str(item) for item in value] if isinstance(value, list) else []


def _extract_turns(example: dict[str, Any]) -> list[dict[str, Any]]:
    turns = example.get("turns")
    if isinstance(turns, list):
        return turns
    return []


def format_prompt(example: dict[str, Any]) -> str:
    failures = ", ".join(_extract_prior_failures(example)[:5]) or "none"
    return (
        "### Instruction\n"
        "Generate a multi-turn attack trajectory.\n\n"
        "### Context\n"
        f"Target: {_extract_target(example)}\n"
        f"Category: {_extract_category(example)}\n"
        f"Strategy: {_extract_strategy(example)}\n"
        f"Prior failures: {failures}\n\n"
    )


def format_completion(example: dict[str, Any]) -> str:
    turns = [
        {
            "content": str(turn.get("content", "")),
            "intent_label": str(turn.get("intent_label", "")),
        }
        for turn in _extract_turns(example)
    ]
    return json.dumps(turns, sort_keys=True)


def build_training_manifest(config: AttackerTrainingConfig, examples: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts: dict[str, int] = {}
    strategy_counts: dict[str, int] = {}
    turn_counts: list[int] = []
    for example in examples:
        category = _extract_category(example)
        strategy = _extract_strategy(example)
        turns = _extract_turns(example)
        category_counts[category] = category_counts.get(category, 0) + 1
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        turn_counts.append(len(turns))
    return {
        "tool": "autoredteam",
        "task": "attacker_slm_training",
        "config": config.to_dict(),
        "example_count": len(examples),
        "category_counts": dict(sorted(category_counts.items())),
        "strategy_counts": dict(sorted(strategy_counts.items())),
        "avg_turns": round(sum(turn_counts) / max(len(turn_counts), 1), 2),
        "ml_stack_available": _ML_STACK_AVAILABLE,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the attacker SLM")
    parser.add_argument("--data", default="training_data/attacker_examples.jsonl")
    parser.add_argument("--output-root", default="models")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter-name", default="attacker-v1")
    parser.add_argument("--cycle", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = AttackerTrainingConfig(
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
    manifest = build_training_manifest(config, examples)
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "attacker_training_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

    if config.dry_run:
        return 0

    trainer = LoRATrainer(
        base_attacker_model=config.base_model,
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
    adapter = trainer.train_attacker(examples, name=config.adapter_name, cycle=config.cycle)
    metrics = trainer.evaluate_adapter(adapter, examples)
    (output_root / "attacker_training_report.json").write_text(
        json.dumps({"adapter": adapter.to_dict(), "metrics": metrics.to_dict()}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
