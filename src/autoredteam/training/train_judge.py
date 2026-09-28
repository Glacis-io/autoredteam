#!/usr/bin/env python3
"""Local judge-training manifest and helper utilities."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


def _safe_import(module_name: str):
    try:
        return __import__(module_name, fromlist=["*"])
    except Exception:
        return None


torch = _safe_import("torch")
transformers = _safe_import("transformers")
peft = _safe_import("peft")
datasets = _safe_import("datasets")


@dataclass
class JudgeTrainingConfig:
    base_model: str = "Qwen/Qwen3Guard-Gen-0.6B"
    data_path: str = "training_data/judge_examples.jsonl"
    output_dir: str = "models/judge-v1"
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 2e-4
    max_examples: Optional[int] = None
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_length: int = 1024
    dry_run: bool = False
    report_path: str = "training_data/judge_training_manifest.json"

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


def _extract_category(example: dict[str, Any]) -> str:
    if "category" in example:
        return str(example.get("category", "unknown"))
    if isinstance(example.get("input"), dict):
        return str(example["input"].get("category", "unknown"))
    return "unknown"


def _extract_prompt(example: dict[str, Any]) -> str:
    if "prompt" in example:
        return str(example.get("prompt", ""))
    if isinstance(example.get("input"), dict):
        return str(example["input"].get("prompt") or example["input"].get("transcript") or "")
    return ""


def _extract_response(example: dict[str, Any]) -> str:
    if "response" in example:
        return str(example.get("response", ""))
    if isinstance(example.get("output"), dict):
        return str(example["output"].get("response", ""))
    if isinstance(example.get("input"), dict):
        return str(example["input"].get("response", ""))
    return ""


def _extract_label(example: dict[str, Any]) -> dict[str, Any]:
    label = example.get("label")
    return label if isinstance(label, dict) else {}


def _severity_label(severity: float) -> str:
    if severity < 20:
        return "Safe"
    if severity < 60:
        return "Controversial"
    return "Unsafe"


def build_training_manifest(config: JudgeTrainingConfig, examples: list[dict[str, Any]]) -> dict[str, Any]:
    severity_buckets = {"0-19": 0, "20-39": 0, "40-59": 0, "60-79": 0, "80-100": 0}
    category_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    flag_counts: dict[str, int] = {}

    for example in examples:
        label = _extract_label(example)
        severity = float(label.get("severity", 0.0))
        category = _extract_category(example)
        source = str(example.get("source", "unknown"))
        category_counts[category] = category_counts.get(category, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
        flags = example.get("deterministic_flags", []) or example.get("metadata", {}).get("deterministic_flags", [])
        for flag in flags:
            flag_key = str(flag)
            flag_counts[flag_key] = flag_counts.get(flag_key, 0) + 1
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
        "task": "judge_slm_training",
        "config": config.to_dict(),
        "available_ml_stack": {
            "torch": torch is not None,
            "transformers": transformers is not None,
            "peft": peft is not None,
            "datasets": datasets is not None,
        },
        "example_count": len(examples),
        "category_counts": dict(sorted(category_counts.items())),
        "severity_buckets": severity_buckets,
        "source_counts": dict(sorted(source_counts.items())),
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda item: (-item[1], item[0]))),
        "sparse_categories": sorted(category for category, count in category_counts.items() if count < 10),
        "notes": [
            "This manifest is emitted even when the ML stack is missing.",
            "Use immune.py or your own trainer to run the actual fine-tune.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local judge training manifest")
    parser.add_argument("--data", default="training_data/judge_examples.jsonl")
    parser.add_argument("--output-dir", default="models/judge-v1")
    parser.add_argument("--base-model", default="Qwen/Qwen3Guard-Gen-0.6B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--report-path", default="training_data/judge_training_manifest.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = JudgeTrainingConfig(
        base_model=args.base_model,
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_examples=args.max_examples,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_length=args.max_length,
        dry_run=args.dry_run,
        report_path=args.report_path,
    )
    examples = load_examples(config.data_path, config.max_examples)
    manifest = build_training_manifest(config, examples)
    report_path = Path(config.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
