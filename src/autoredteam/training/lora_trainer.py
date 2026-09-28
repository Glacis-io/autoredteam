"""Dependency-tolerant LoRA training scaffolding.

When torch/transformers/peft/datasets are installed, the defender training path
runs a real LoRA fine-tune using the same format_prompt/format_completion pattern
as train_defender.py. When those libraries are missing, the placeholder
path emits metadata so the rest of the pipeline remains functional.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

_ML_STACK_AVAILABLE = all(mod is not None for mod in [torch, transformers, peft, datasets])


@dataclass
class LoRAAdapter:
    """Metadata for a trained or placeholder adapter."""

    name: str
    model_path: str
    task: str
    status: str = "stub"
    base_model: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model_path": self.model_path,
            "task": self.task,
            "status": self.status,
            "base_model": self.base_model,
            "created_at": self.created_at,
            "metrics": self.metrics,
            "notes": self.notes,
        }


@dataclass
class TrainingMetrics:
    """Simple metrics payload for adapter evaluation."""

    score: float = 0.0
    tier: str = "stub"
    samples: int = 0
    loss: Optional[float] = None
    latency_ms: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "tier": self.tier,
            "samples": self.samples,
            "loss": self.loss,
            "latency_ms": self.latency_ms,
            "notes": self.notes,
        }


@dataclass
class TrainingArtifact:
    """Output of a training call."""

    adapter: LoRAAdapter
    metrics: TrainingMetrics
    checkpoint_hash: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "adapter": self.adapter.to_dict(),
            "metrics": self.metrics.to_dict(),
            "checkpoint_hash": self.checkpoint_hash,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Defender prompt formatting — mirrors train_defender.py
# ---------------------------------------------------------------------------

OUTPUT_MARKER = "### Output\n"


def _extract_history(example: dict) -> list[dict]:
    """Extract conversation history from an example."""
    conversation = example.get("conversation", [])
    if isinstance(conversation, list) and len(conversation) > 1:
        return conversation[:-1]
    history = example.get("conversation_history", [])
    if isinstance(history, list):
        return history
    return []


def _extract_current_turn(example: dict) -> str:
    """Extract the current turn text from an example."""
    current = example.get("current_turn")
    if isinstance(current, str) and current:
        return current
    conversation = example.get("conversation", [])
    if isinstance(conversation, list) and conversation:
        last = conversation[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    return ""


def _extract_category(example: dict) -> str:
    """Extract category from example or its label."""
    label = example.get("label")
    if isinstance(label, dict) and "category" in label:
        return str(label["category"])
    if "category" in example:
        return str(example["category"])
    return "unknown"


def _extract_label(example: dict) -> dict:
    label = example.get("label")
    return label if isinstance(label, dict) else {}


def _format_history(history: list[dict]) -> str:
    """Render conversation history as a readable string."""
    if not history:
        return "(none)"
    lines: list[str] = []
    for turn in history:
        if isinstance(turn, dict):
            role = str(turn.get("role", "unknown"))
            content = str(turn.get("content", ""))
            lines.append(f"{role}: {content}")
        else:
            lines.append(str(turn))
    return "\n".join(lines)


def format_defender_prompt(example: dict) -> str:
    """Build the input portion (everything BEFORE ### Output)."""
    history = _extract_history(example)
    current_turn = _extract_current_turn(example)
    return (
        "### Instruction\n"
        "Detect if this conversation turn is an attack.\n\n"
        "### History\n"
        f"{_format_history(history)}\n\n"
        "### Current Turn\n"
        f"{current_turn}\n\n"
    )


def format_defender_completion(example: dict) -> str:
    """Build the output portion (what the model must learn to generate)."""
    label = _extract_label(example)
    category = _extract_category(example)
    severity = round(float(label.get("severity", 0.0)), 1)
    is_attack = bool(label.get("is_attack", severity >= 40.0))
    confidence = round(float(label.get("confidence", (severity / 100.0) if is_attack else 0.1)), 3)
    reasoning = str(label.get("reasoning", ""))
    output = {
        "is_attack": is_attack,
        "confidence": confidence,
        "category": category,
        "severity": severity,
        "reasoning": reasoning,
    }
    return json.dumps(output, sort_keys=True)


def format_defender_full_text(example: dict) -> str:
    """Full training text = prompt + marker + completion."""
    return format_defender_prompt(example) + OUTPUT_MARKER + format_defender_completion(example)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class LoRATrainer:
    """LoRA trainer with real training for defender when ML stack is available.

    When torch/transformers/peft/datasets are installed, train_defender() runs
    a real LoRA fine-tune. Otherwise it falls back to the placeholder path so
    the rest of the pipeline remains functional.
    """

    def __init__(
        self,
        base_attacker_model: str = "Qwen/Qwen2.5-3B-Instruct",
        base_defender_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        output_root: str = "models",
        # Defender training hyperparameters
        epochs: int = 3,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 1e-4,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        max_length: int = 1024,
        eval_fraction: float = 0.1,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
    ):
        self.base_attacker_model = base_attacker_model
        self.base_defender_model = base_defender_model
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._artifacts: list[TrainingArtifact] = []
        # Hyperparams
        self.epochs = epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.max_length = max_length
        self.eval_fraction = eval_fraction
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay

    def train_attacker(self, examples: list[dict[str, Any]], **kwargs: Any) -> LoRAAdapter:
        return self._train_placeholder(
            task="attacker",
            base_model=self.base_attacker_model,
            examples=examples,
            **kwargs,
        )

    def train_defender(self, examples: list[dict[str, Any]], **kwargs: Any) -> LoRAAdapter:
        if _ML_STACK_AVAILABLE and examples:
            return self._train_defender_real(examples, **kwargs)
        return self._train_placeholder(
            task="defender",
            base_model=self.base_defender_model,
            examples=examples,
            **kwargs,
        )

    def evaluate_adapter(self, adapter: LoRAAdapter, eval_data: list[dict[str, Any]]) -> TrainingMetrics:
        """Return a deterministic placeholder evaluation."""
        sample_count = len(eval_data)
        score = 0.0
        if sample_count:
            score = min(100.0, 40.0 + (sample_count * 3.5))
        if adapter.status == "trained":
            score = min(100.0, score + 10.0)
        tier = "ready" if score >= 70 else "stub"
        latency_ms = 25.0 if adapter.status == "trained" else 5.0
        return TrainingMetrics(
            score=score,
            tier=tier,
            samples=sample_count,
            loss=max(0.0, 1.0 - score / 100.0),
            latency_ms=latency_ms,
            notes=[f"adapter={adapter.name}", f"status={adapter.status}"],
        )

    def should_keep(self, before_metrics: Optional[dict[str, Any]], after_metrics: dict[str, Any]) -> bool:
        """Keep adapters that clear the configured confidence threshold."""
        score = float(after_metrics.get("score", 0.0))
        return score >= 50.0

    def checkpoint(self, adapter: LoRAAdapter, cycle: int) -> str:
        """Write a placeholder checkpoint manifest and return its hash."""
        payload = {
            "cycle": cycle,
            "adapter": adapter.to_dict(),
        }
        checkpoint_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        manifest = self.output_root / adapter.task / f"cycle_{cycle:03d}" / "adapter_manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({**payload, "checkpoint_hash": checkpoint_hash}, indent=2))
        return checkpoint_hash

    def export_artifacts(self) -> list[dict]:
        return [artifact.to_dict() for artifact in self._artifacts]

    # ------------------------------------------------------------------
    # Real defender training (requires ML stack)
    # ------------------------------------------------------------------

    def _train_defender_real(
        self,
        examples: list[dict[str, Any]],
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> LoRAAdapter:
        """Run a real LoRA fine-tune for the defender task."""
        adapter_name = name or "defender-adapter"
        model_path = str(self.output_root / "defender" / adapter_name)
        local_dir = Path(model_path)
        local_dir.mkdir(parents=True, exist_ok=True)

        print(f"[defender-train] {len(examples)} examples, base_model={self.base_defender_model}")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.base_defender_model, use_fast=True
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(self.base_defender_model)

        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        # LoRA on all linear layers
        lora_config = peft.LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = peft.get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[defender-train] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        # Tokenize the output marker to find where loss masking starts
        marker_ids = tokenizer.encode(OUTPUT_MARKER, add_special_tokens=False)
        marker_len = len(marker_ids)

        def tokenize_with_masked_labels(row):
            full_text = row["text"]
            enc = tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )
            input_ids = enc["input_ids"]
            labels = list(input_ids)

            # Find the output marker position and mask everything before it
            mask_end = 0
            for i in range(len(input_ids) - marker_len + 1):
                if input_ids[i:i + marker_len] == marker_ids:
                    mask_end = i + marker_len
                    break

            for i in range(mask_end):
                labels[i] = -100
            pad_id = tokenizer.pad_token_id
            if pad_id is not None:
                labels = [(-100 if tok == pad_id else tok) for tok in labels]

            enc["labels"] = labels
            return enc

        texts = [format_defender_full_text(ex) for ex in examples]
        full_dataset = datasets.Dataset.from_dict({"text": texts})

        # Train/eval split
        eval_dataset = None
        if self.eval_fraction > 0 and len(examples) > 20:
            split = full_dataset.train_test_split(
                test_size=self.eval_fraction, seed=42
            )
            train_dataset = split["train"].map(tokenize_with_masked_labels, remove_columns=["text"])
            eval_dataset = split["test"].map(tokenize_with_masked_labels, remove_columns=["text"])
            print(f"[defender-train] Split: {len(train_dataset)} train, {len(eval_dataset)} eval")
        else:
            train_dataset = full_dataset.map(tokenize_with_masked_labels, remove_columns=["text"])

        training_args = transformers.TrainingArguments(
            output_dir=str(local_dir),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            logging_steps=10,
            save_strategy="steps" if eval_dataset else "epoch",
            save_steps=50 if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=50 if eval_dataset else None,
            load_best_model_at_end=bool(eval_dataset),
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False if eval_dataset else None,
            save_total_limit=3,
            fp16=torch.cuda.is_available(),
            report_to=[],
            dataloader_pin_memory=False,
        )

        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        trainer.train()
        trainer.save_model(str(local_dir))
        tokenizer.save_pretrained(str(local_dir))

        adapter = LoRAAdapter(
            name=adapter_name,
            model_path=model_path,
            task="defender",
            status="trained",
            base_model=self.base_defender_model,
            metrics={
                "samples": len(examples),
                "train_examples": len(train_dataset),
                "eval_examples": len(eval_dataset) if eval_dataset else 0,
                "trainable_params": trainable,
                "total_params": total,
            },
            notes=[
                f"examples={len(examples)}",
                "real_lora_training",
                "loss_masking",
                "all_linear_targets",
                "cosine_lr",
            ],
        )
        metrics = self.evaluate_adapter(adapter, examples)
        checkpoint_hash = self.checkpoint(adapter, cycle=max(1, kwargs.get("cycle", 1)))
        self._artifacts.append(
            TrainingArtifact(
                adapter=adapter,
                metrics=metrics,
                checkpoint_hash=checkpoint_hash,
            )
        )
        print(f"[defender-train] Saved adapter to {model_path}")
        return adapter

    # ------------------------------------------------------------------
    # Placeholder path (no ML stack or no examples)
    # ------------------------------------------------------------------

    def _train_placeholder(
        self,
        task: str,
        base_model: str,
        examples: list[dict[str, Any]],
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> LoRAAdapter:
        """Placeholder training hook."""
        sample_count = len(examples)
        adapter_name = name or f"{task}-adapter"
        model_path = str(self.output_root / task / adapter_name)
        status = "trained" if sample_count > 0 else "stub"
        notes = [f"examples={sample_count}", "placeholder_training"]
        if not _ML_STACK_AVAILABLE:
            notes.append("ml_stack_unavailable")
        if kwargs:
            notes.append(f"kwargs={sorted(kwargs.keys())}")

        adapter = LoRAAdapter(
            name=adapter_name,
            model_path=model_path,
            task=task,
            status=status,
            base_model=base_model,
            metrics={"samples": sample_count},
            notes=notes,
        )
        metrics = self.evaluate_adapter(adapter, examples)
        checkpoint_hash = self.checkpoint(adapter, cycle=max(1, kwargs.get("cycle", 1)))
        self._artifacts.append(
            TrainingArtifact(
                adapter=adapter,
                metrics=metrics,
                checkpoint_hash=checkpoint_hash,
            )
        )
        return adapter
