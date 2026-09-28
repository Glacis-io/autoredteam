#!/usr/bin/env python3
"""
Cross-pollinate judge training data into defender SLM training format.

Reads training_data/judge_examples_v1.jsonl (657 examples) and converts
each into defender format at training_data/defender_examples_v2.jsonl.

Derivation rules:
  - is_attack: severity >= 40.0 OR bypassed == true
  - confidence: severity / 100.0 for attacks, 0.1 for benign
  - category, severity, reasoning: preserved from source
"""

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = REPO_ROOT / "training_data" / "judge_examples_v1.jsonl"
OUTPUT_PATH = REPO_ROOT / "training_data" / "defender_examples_v2.jsonl"


def convert_line(judge_record: dict) -> dict:
    """Convert a single judge example to defender training format."""
    label = judge_record["label"]
    severity = label["severity"]
    bypassed = label.get("bypassed", False)
    reasoning = label.get("reasoning", "")
    category = judge_record["category"]

    is_attack = severity >= 40.0 or bypassed is True
    confidence = severity / 100.0 if is_attack else 0.1

    return {
        "conversation": [
            {"role": "user", "content": judge_record["prompt"]},
            {"role": "assistant", "content": judge_record["response"]},
        ],
        "category": category,
        "label": {
            "is_attack": is_attack,
            "confidence": round(confidence, 4),
            "category": category,
            "severity": severity,
            "reasoning": reasoning,
        },
        "source": "cross_pollinated_judge_v1",
    }


def main():
    if not INPUT_PATH.exists():
        print(f"ERROR: Input file not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    total = 0
    attacks = 0
    benign = 0
    category_counts = Counter()
    errors = 0

    with open(INPUT_PATH, encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                judge_record = json.loads(line)
                defender_record = convert_line(judge_record)
                fout.write(json.dumps(defender_record, ensure_ascii=False) + "\n")
                total += 1
                if defender_record["label"]["is_attack"]:
                    attacks += 1
                else:
                    benign += 1
                category_counts[defender_record["category"]] += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"WARNING: Skipping line {line_no}: {e}", file=sys.stderr)
                errors += 1

    # Summary stats
    print(f"\n{'='*60}")
    print(f"Cross-pollination complete: judge_v1 -> defender_v2")
    print(f"{'='*60}")
    print(f"  Input:   {INPUT_PATH}")
    print(f"  Output:  {OUTPUT_PATH}")
    print(f"  Total converted:  {total}")
    print(f"  Attacks:          {attacks}  ({100*attacks/total:.1f}%)")
    print(f"  Benign:           {benign}  ({100*benign/total:.1f}%)")
    if errors:
        print(f"  Errors/skipped:   {errors}")
    print(f"\nCategory distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        atk = sum(
            1 for _ in []  # placeholder; compute below
        )
        print(f"  {cat:35s} {count:4d}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
