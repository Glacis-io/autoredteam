"""
training_data.py - Lightweight training data collection helpers.

The collector stores normalized examples in memory and can export JSONL files
for judge, attacker, and defender fine-tuning workflows.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from autoredteam.conversation import AttackTrajectory, ConversationTurn, TrainingExample, build_transcript


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_dict(value: Any) -> dict:
    if value is None:
        return {}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return {}


def _safe_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _score_to_dict(score: Any) -> dict:
    data = _safe_dict(score)
    if data:
        return data
    return {}


def _score_to_composite(score: Any) -> float:
    if score is None:
        return 0.0
    if hasattr(score, "composite") and callable(score.composite):
        try:
            return float(score.composite())
        except TypeError:
            return float(score.composite({}))
    if isinstance(score, dict):
        for key in ("composite", "score", "value"):
            if key in score:
                try:
                    return float(score[key])
                except (TypeError, ValueError):
                    pass
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def _score_to_depth(score: Any) -> float:
    if score is None:
        return 0.0
    if isinstance(score, dict):
        try:
            return float(score.get("depth", 0.0))
        except (TypeError, ValueError):
            return 0.0
    return float(_get_attr(score, "depth", 0.0) or 0.0)


def _record_prompt(attack: Any) -> str:
    if isinstance(attack, str):
        return attack
    return str(_get_attr(attack, "prompt", attack))


class TrainingDataCollector:
    """
    Collects normalized training examples for later JSONL export.

    The API is intentionally permissive so it can accept the current attack
    objects, dicts from future pipelines, or plain prompt strings.
    """

    def __init__(self, output_dir: str = "training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.single_turn_records: list[dict[str, Any]] = []
        self.multi_turn_records: list[dict[str, Any]] = []
        self._category_counts: dict[str, int] = {}

    def record_single_turn(
        self,
        attack: Any,
        response: str,
        score: Any,
        category: str = "",
        metadata: Optional[dict] = None,
    ) -> dict:
        prompt = _record_prompt(attack)
        attack_id = _get_attr(attack, "id", _get_attr(attack, "attack_id", ""))
        resolved_category = category or _get_attr(attack, "category", "")
        score_vector = _score_to_dict(score)
        composite_score = _score_to_composite(score)
        flags = _safe_list(_get_attr(score, "deterministic_flags", []))
        record = {
            "kind": "single_turn",
            "attack_id": attack_id,
            "prompt": prompt,
            "response": response,
            "category": resolved_category,
            "score_vector": score_vector,
            "composite_score": round(composite_score, 1),
            "deterministic_flags": flags,
            "timestamp": _utc_now(),
            "metadata": dict(metadata or {}),
        }
        self.single_turn_records.append(record)
        if resolved_category:
            self._category_counts[resolved_category] = self._category_counts.get(resolved_category, 0) + 1
        return record

    def record_multi_turn(
        self,
        trajectory: Any,
        responses: list[str],
        scores: list[Any],
        metadata: Optional[dict] = None,
    ) -> dict:
        traj = trajectory if isinstance(trajectory, AttackTrajectory) else (
            AttackTrajectory.from_dict(trajectory) if isinstance(trajectory, dict) else None
        )
        if traj is None:
            raise TypeError("trajectory must be an AttackTrajectory or dict")

        score_vectors = [_score_to_dict(score) for score in scores]
        composites = [_score_to_composite(score) for score in scores]
        flags: list[str] = []
        for score in scores:
            flags.extend(_safe_list(_get_attr(score, "deterministic_flags", [])))

        example = TrainingExample(
            trajectory=traj,
            target_response_turns=list(responses),
            score_vector={
                "turn_scores": score_vectors,
                "turn_composites": [round(value, 1) for value in composites],
                "final_composite": round(max(composites, default=0.0), 1),
            },
            composite_score=max(composites, default=0.0),
            deterministic_flags=sorted(set(flags)),
        )
        record = {
            "kind": "multi_turn",
            "example": example.to_dict(),
            "transcript": build_transcript(traj.turns),
            "metadata": dict(metadata or {}),
        }
        self.multi_turn_records.append(record)
        if traj.target_category:
            self._category_counts[traj.target_category] = self._category_counts.get(traj.target_category, 0) + 1
        return record

    def export_judge_training_data(self) -> list[dict]:
        records = self._build_judge_training_data()
        self._write_jsonl(self.output_dir / "judge_examples.jsonl", records)
        return records

    def _build_judge_training_data(self) -> list[dict]:
        records: list[dict] = []

        for item in self.single_turn_records:
            record = {
                "prompt": item["prompt"],
                "response": item["response"],
                "category": item["category"],
                "label": {
                    "severity": round(item["score_vector"].get("depth", item["composite_score"]), 1),
                    "bypassed": item["composite_score"] > 10.0,
                    "reasoning": "derived from deterministic/score outputs",
                },
                "source": "collector:single_turn",
                "metadata": {
                    "attack_id": item["attack_id"],
                    "deterministic_flags": item["deterministic_flags"],
                    **dict(item.get("metadata", {})),
                },
                "score_vector": item["score_vector"],
                "composite_score": item["composite_score"],
                "deterministic_flags": item["deterministic_flags"],
                "timestamp": item["timestamp"],
            }
            records.append(record)

        for item in self.multi_turn_records:
            example = item["example"]
            trajectory = example["trajectory"]
            turns = trajectory.get("turns", [])
            responses = example.get("target_response_turns", [])
            turn_scores = example.get("score_vector", {}).get("turn_scores", [])
            for idx, turn in enumerate(turns):
                records.append({
                    "prompt": turn.get("content", ""),
                    "response": responses[idx] if idx < len(responses) else "",
                    "category": trajectory.get("target_category", ""),
                    "label": {
                        "severity": round(turn_scores[idx].get("depth", 0.0), 1) if idx < len(turn_scores) else 0.0,
                        "bypassed": example.get("composite_score", 0.0) > 10.0,
                        "reasoning": "derived from multi-turn trajectory",
                    },
                    "conversation": turns,
                    "conversation_turn": turn,
                    "transcript": item["transcript"],
                    "trajectory_id": trajectory.get("id", ""),
                    "strategy": trajectory.get("strategy", ""),
                    "source": "collector:multi_turn",
                    "metadata": {
                        "deterministic_flags": example.get("deterministic_flags", []),
                        **dict(item.get("metadata", {})),
                    },
                    "score_vector": turn_scores[idx] if idx < len(turn_scores) else {},
                    "timestamp": example.get("timestamp", _utc_now()),
                })

        return records

    def export_attacker_training_data(self) -> list[dict]:
        records = self._build_attacker_training_data()
        self._write_jsonl(self.output_dir / "attacker_examples.jsonl", records)
        return records

    def _build_attacker_training_data(self) -> list[dict]:
        records: list[dict] = []

        for item in self.single_turn_records:
            if item["composite_score"] <= 10.0:
                continue
            records.append({
                "input": {
                    "prompt": item["prompt"],
                    "category": item["category"],
                },
                "output": {
                    "response": item["response"],
                },
                "score_vector": item["score_vector"],
                "composite_score": item["composite_score"],
                "deterministic_flags": item["deterministic_flags"],
                "timestamp": item["timestamp"],
            })

        for item in self.multi_turn_records:
            example = item["example"]
            trajectory = example["trajectory"]
            if example.get("composite_score", 0.0) <= 10.0 and not trajectory.get("success", False):
                continue
            records.append({
                "input": {
                    "target_category": trajectory.get("target_category", ""),
                    "strategy": trajectory.get("strategy", ""),
                    "transcript": item["transcript"],
                    "prior_failures": trajectory.get("metadata", {}).get("prior_failures", []),
                },
                "output": {
                    "turns": trajectory.get("turns", []),
                    "success": trajectory.get("success", False),
                },
                "score_vector": example.get("score_vector", {}),
                "composite_score": example.get("composite_score", 0.0),
                "deterministic_flags": example.get("deterministic_flags", []),
                "timestamp": example.get("timestamp", _utc_now()),
            })

        return records

    def export_defender_training_data(self) -> list[dict]:
        records = self._build_defender_training_data()
        self._write_jsonl(self.output_dir / "defender_examples.jsonl", records)
        return records

    def _build_defender_training_data(self) -> list[dict]:
        records: list[dict] = []

        for item in self.single_turn_records:
            records.append({
                "conversation": [
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": item["response"]},
                ],
                "label": {
                    "is_attack": item["composite_score"] > 10.0,
                    "severity": round(item["score_vector"].get("depth", item["composite_score"]), 1),
                    "category": item["category"],
                },
                "score_vector": item["score_vector"],
                "composite_score": item["composite_score"],
                "deterministic_flags": item["deterministic_flags"],
                "timestamp": item["timestamp"],
            })

        for item in self.multi_turn_records:
            example = item["example"]
            trajectory = example["trajectory"]
            records.append({
                "conversation": trajectory.get("turns", []),
                "label": {
                    "is_attack": example.get("composite_score", 0.0) > 10.0,
                    "severity": round(example.get("score_vector", {}).get("final_composite", example.get("composite_score", 0.0)), 1),
                    "category": trajectory.get("target_category", ""),
                    "detection_turn": trajectory.get("detection_turn"),
                },
                "score_vector": example.get("score_vector", {}),
                "composite_score": example.get("composite_score", 0.0),
                "deterministic_flags": example.get("deterministic_flags", []),
                "timestamp": example.get("timestamp", _utc_now()),
            })

        return records

    def export_all(self) -> dict[str, list[dict]]:
        """Write all JSONL exports and return the in-memory payloads."""
        return {
            "judge": self.export_judge_training_data(),
            "attacker": self.export_attacker_training_data(),
            "defender": self.export_defender_training_data(),
        }

    def stats(self) -> dict:
        """Return a small summary of collected data and exported files."""
        judge_count = len(self._build_judge_training_data())
        attacker_count = len(self._build_attacker_training_data())
        defender_count = len(self._build_defender_training_data())
        return {
            "output_dir": str(self.output_dir),
            "single_turn_records": len(self.single_turn_records),
            "multi_turn_records": len(self.multi_turn_records),
            "judge_examples": judge_count,
            "attacker_examples": attacker_count,
            "defender_examples": defender_count,
            "category_counts": dict(sorted(self._category_counts.items())),
            "files": {
                "judge": str(self.output_dir / "judge_examples.jsonl"),
                "attacker": str(self.output_dir / "attacker_examples.jsonl"),
                "defender": str(self.output_dir / "defender_examples.jsonl"),
            },
        }

    def clear(self) -> None:
        """Reset collected records without deleting exported files."""
        self.single_turn_records.clear()
        self.multi_turn_records.clear()
        self._category_counts.clear()

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record, sort_keys=True) + "\n")


__all__ = ["TrainingDataCollector"]
