#!/usr/bin/env python3
"""Merge all training data chunks + cross-pollinated data into final JSONL files."""
from pathlib import Path

OUT_DIR = Path("training_data")

def merge(pattern_prefix: str, cross_pollinated, output: str):
    """Merge chunk files + cross-pollinated data into one JSONL."""
    lines = []
    # Collect chunks
    for f in sorted(OUT_DIR.glob(f"{pattern_prefix}_chunk*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    lines.append(line)
        print(f"  {f.name}: {sum(1 for _ in open(f, encoding='utf-8'))} lines")

    # Add cross-pollinated data
    if cross_pollinated:
        cp = OUT_DIR / cross_pollinated
        if cp.exists():
            with open(cp, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        lines.append(line)
            print(f"  {cp.name}: {sum(1 for _ in open(cp, encoding='utf-8'))} lines (cross-pollinated)")

    # Also add v1 data for judge
    if "judge" in pattern_prefix:
        v1 = OUT_DIR / "judge_examples_v1.jsonl"
        if v1.exists():
            with open(v1, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        lines.append(line)
            print(f"  {v1.name}: {sum(1 for _ in open(v1, encoding='utf-8'))} lines (v1 original)")

    out_path = OUT_DIR / output
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  -> {output}: {len(lines)} total lines\n")
    return len(lines)


print("=== Merging Judge Data ===")
j = merge("judge", None, "judge_examples_merged.jsonl")

print("=== Merging Defender Data ===")
d = merge("defender", "defender_examples_v2.jsonl", "defender_examples_merged.jsonl")

print("=== Merging Attacker Data ===")
a = merge("attacker", "attacker_examples_v2.jsonl", "attacker_examples_merged.jsonl")

print(f"TOTAL: {j} judge, {d} defender, {a} attacker examples")
