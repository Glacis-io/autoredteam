#!/usr/bin/env python3
"""
Cross-pollinate judge training data into attacker SLM training format.

Reads training_data/judge_examples_v1.jsonl (657 judge examples) and converts
successful attacks (severity >= 40 OR bypassed == true) into single-turn
attacker trajectories written to training_data/attacker_examples_v2.jsonl.
"""

import json
import sys
from collections import Counter
from pathlib import Path

# Resolve paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = REPO_ROOT / "training_data" / "judge_examples_v1.jsonl"
OUTPUT_PATH = REPO_ROOT / "training_data" / "attacker_examples_v2.jsonl"

# Strategy mapping
STRATEGY_MAP = {
    "prompt_injection": "trust_building",
    "jailbreak": "trust_building",
    "role_confusion": "trust_building",
    "refusal_suppression": "format_pressure",
    "output_formatting_exploit": "format_pressure",
}
DEFAULT_STRATEGY = "crescendo"

# Category -> human-readable intent label
INTENT_LABELS = {
    "prompt_injection": "Prompt injection attempt",
    "jailbreak": "Jailbreak attempt",
    "role_confusion": "Role confusion attempt",
    "refusal_suppression": "Refusal suppression attempt",
    "output_formatting_exploit": "Output formatting exploit",
    "tool_misuse": "Tool misuse attempt",
    "pii_extraction": "PII extraction attempt",
    "system_prompt_leakage": "System prompt leakage attempt",
    "social_engineering": "Social engineering attempt",
    "authority_manipulation": "Authority manipulation attempt",
    "encoding_bypass": "Encoding bypass attempt",
    "indirect_injection": "Indirect injection attempt",
    "multi_turn_manipulation": "Multi-turn manipulation attempt",
    "multilingual_attack": "Multilingual attack attempt",
    "hallucination_exploit": "Hallucination exploit attempt",
    "continuation_attack": "Continuation attack attempt",
    "payload_splitting": "Payload splitting attempt",
    "context_window_poisoning": "Context window poisoning attempt",
    "ethical_bypass": "Ethical bypass attempt",
}


def should_include(label: dict) -> bool:
    """Include if severity >= 40 OR bypassed is true."""
    return label.get("severity", 0) >= 40.0 or label.get("bypassed", False) is True


def map_strategy(category: str) -> str:
    return STRATEGY_MAP.get(category, DEFAULT_STRATEGY)


def humanize_intent(category: str) -> str:
    if category in INTENT_LABELS:
        return INTENT_LABELS[category]
    # Fallback: replace underscores with spaces and title-case
    return category.replace("_", " ").title() + " attempt"


def convert_example(judge_example: dict) -> dict:
    """Convert a judge example into attacker training format."""
    category = judge_example["category"]
    return {
        "target_description": "An AI assistant",
        "category": category,
        "strategy": map_strategy(category),
        "prior_failures": [],
        "turns": [
            {
                "content": judge_example["prompt"],
                "intent_label": humanize_intent(category),
            }
        ],
        "source": "cross_pollinated_judge_v1",
    }


def main():
    if not INPUT_PATH.exists():
        print(f"ERROR: Input file not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    # Read all judge examples
    examples = []
    with open(INPUT_PATH, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping malformed JSON on line {line_num}: {e}", file=sys.stderr)

    print(f"Read {len(examples)} judge examples from {INPUT_PATH.name}")

    # Filter and convert
    converted = []
    category_counter = Counter()
    strategy_counter = Counter()

    for ex in examples:
        if not should_include(ex["label"]):
            continue
        attacker_ex = convert_example(ex)
        converted.append(attacker_ex)
        category_counter[attacker_ex["category"]] += 1
        strategy_counter[attacker_ex["strategy"]] += 1

    # Write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(converted)} attacker examples to {OUTPUT_PATH.name}")
    print(f"Filtered out {len(examples) - len(converted)} examples (severity < 40 and not bypassed)")

    # Summary stats
    print(f"\n{'='*60}")
    print(f"SUMMARY STATS")
    print(f"{'='*60}")
    print(f"Total source examples:    {len(examples)}")
    print(f"Total attacker examples:  {len(converted)}")
    print(f"Conversion rate:          {len(converted)/len(examples)*100:.1f}%")

    print(f"\nCategory distribution ({len(category_counter)} categories):")
    for cat, count in category_counter.most_common():
        print(f"  {cat:35s}  {count:4d}  ({count/len(converted)*100:5.1f}%)")

    print(f"\nStrategy distribution ({len(strategy_counter)} strategies):")
    for strat, count in strategy_counter.most_common():
        print(f"  {strat:20s}  {count:4d}  ({count/len(converted)*100:5.1f}%)")


if __name__ == "__main__":
    main()
