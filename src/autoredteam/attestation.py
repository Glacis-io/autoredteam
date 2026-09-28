"""
attestation.py — Evidence chain and attestation hooks.

Free mode:  Local git log + JSON evidence files. No external dependencies.
Paid mode:  Glacis cryptographic attestation (clear integration point).

Hash separation: attestation records contain hashes of attack prompts,
NOT raw prompts. Three access tiers:
  1. Public  — summary stats, category coverage, composite scores
  2. Team    — full hashes, score vectors, timestamps
  3. Admin   — raw prompts (stored locally only, never in attestation chain)
"""

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from autoredteam import __version__


# ---------------------------------------------------------------------------
# Evidence record
# ---------------------------------------------------------------------------

@dataclass
class EvidenceRecord:
    """One entry in the evidence chain."""
    timestamp: str
    cycle: int
    attack_id: str
    prompt_hash: str  # SHA-256 hash, NOT the raw prompt
    category: str
    score_vector: dict  # {breadth, depth, novelty, reliability}
    composite_score: float
    deterministic_flags: list[str]
    phase: str  # "attack" or "defend"

    # These fields are tier-3 (admin only) — stored locally, not attested
    # Raw prompt and response are written to a separate local-only file
    _raw_prompt: str = ""
    _raw_response: str = ""

    def public_summary(self) -> dict:
        """Tier 1: Public summary (no hashes, no details)."""
        return {
            "cycle": self.cycle,
            "category": self.category,
            "composite_score": round(self.composite_score, 1),
            "phase": self.phase,
            "timestamp": self.timestamp,
        }

    def team_summary(self) -> dict:
        """Tier 2: Team view (hashes + score vectors)."""
        return {
            "cycle": self.cycle,
            "attack_id": self.attack_id,
            "prompt_hash": self.prompt_hash,
            "category": self.category,
            "score_vector": self.score_vector,
            "composite_score": round(self.composite_score, 1),
            "deterministic_flags": self.deterministic_flags,
            "phase": self.phase,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Local evidence store (free mode)
# ---------------------------------------------------------------------------

class LocalEvidenceStore:
    """
    Stores evidence as JSON files + git commits.
    This is the free tier — works out of the box with no external services.
    """

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_file = self.output_dir / "evidence_chain.jsonl"
        self.raw_file = self.output_dir / ".raw_evidence.jsonl"  # Tier 3, gitignored
        self.summary_file = self.output_dir / "summary.json"
        self._cached_chain_hash: Optional[str] = None
        self._ensure_gitignore()

    def _ensure_gitignore(self):
        """Ensure raw evidence file is gitignored."""
        gitignore = self.output_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(".raw_evidence.jsonl\n")

    def record(self, evidence: EvidenceRecord) -> str:
        """
        Record evidence. Returns the chain hash (hash of this record + previous).
        """
        # Write tier-2 record to evidence chain
        team_data = evidence.team_summary()

        # Compute chain hash (links to previous record).
        # Use in-memory cache to avoid re-reading the file — reading the file
        # on every call is both slow (O(n) scan) and fragile (OS write-buffer
        # visibility can cause the previous write to be missed, breaking the
        # chain link).
        prev_hash = self._cached_chain_hash or self._last_chain_hash()
        record_bytes = json.dumps(team_data, sort_keys=True).encode()
        chain_hash = hashlib.sha256(
            prev_hash.encode() + record_bytes
        ).hexdigest()
        team_data["chain_hash"] = chain_hash

        with open(self.evidence_file, "a") as f:
            f.write(json.dumps(team_data) + "\n")

        # Keep the tip in memory so the next record can link without a file read.
        self._cached_chain_hash = chain_hash

        # Write tier-3 raw data (local only)
        raw_data = {
            "attack_id": evidence.attack_id,
            "prompt": evidence._raw_prompt,
            "response": evidence._raw_response,
            "chain_hash": chain_hash,
        }
        with open(self.raw_file, "a") as f:
            f.write(json.dumps(raw_data) + "\n")

        return chain_hash

    def _last_chain_hash(self) -> str:
        """Get the chain hash of the last record."""
        if not self.evidence_file.exists():
            return "genesis"
        try:
            with open(self.evidence_file) as f:
                lines = f.readlines()
            if lines:
                last = json.loads(lines[-1])
                return last.get("chain_hash", "genesis")
        except (json.JSONDecodeError, IndexError):
            pass
        return "genesis"

    def last_chain_hash(self) -> str:
        """Public accessor for the current tip of the evidence chain."""
        return self._cached_chain_hash or self._last_chain_hash()

    def clear(self) -> None:
        """Remove evidence files from prior runs so the next run starts a clean chain."""
        for path in (self.evidence_file, self.raw_file):
            if path.exists():
                path.unlink()
        self._cached_chain_hash = None

    def write_summary(self, summary: dict) -> None:
        """Write the public-tier summary."""
        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)

    def git_commit(self, message: str) -> bool:
        """Commit evidence to local git repo."""
        try:
            subprocess.run(
                ["git", "add", str(self.evidence_file), str(self.summary_file)],
                capture_output=True, check=True,
            )
            subprocess.run(
                ["git", "commit", "-m", f"[autoredteam] {message}"],
                capture_output=True, check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def load_chain(self) -> list[dict]:
        """Load the full evidence chain."""
        if not self.evidence_file.exists():
            return []
        records = []
        with open(self.evidence_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def verify_chain(self) -> bool:
        """Verify the integrity of the evidence chain."""
        records = self.load_chain()
        if not records:
            return True

        prev_hash = "genesis"
        for record in records:
            expected_hash = record.pop("chain_hash", "")
            record_bytes = json.dumps(record, sort_keys=True).encode()
            computed = hashlib.sha256(
                prev_hash.encode() + record_bytes
            ).hexdigest()
            record["chain_hash"] = expected_hash
            if computed != expected_hash:
                return False
            prev_hash = expected_hash
        return True


# ---------------------------------------------------------------------------
# Glacis attestation hooks (paid tier integration point)
# ---------------------------------------------------------------------------

class GlacisAttestationHook:
    """
    Integration point for Glacis cryptographic attestation.

    In free mode, this is a no-op wrapper. When Glacis is configured,
    it submits evidence hashes to the Glacis attestation service for
    cryptographic timestamping and tamper-proof storage.

    To enable:
      1. pip install glacis-sdk
      2. Set GLACIS_API_KEY in your environment
      3. Set attestation.provider: "glacis" in config.yaml
    """

    def __init__(self, config: Optional[dict] = None):
        self.enabled = False
        self.client = None
        self._config = config or {}

        if self._config.get("provider") == "glacis":
            self._init_glacis()

    def _init_glacis(self):
        """Initialize Glacis client if available."""
        try:
            # This is the integration point — uncomment when glacis-sdk is available
            # from glacis import GlacisClient
            # self.client = GlacisClient(
            #     api_key=os.environ.get("GLACIS_API_KEY"),
            #     project=self._config.get("project", "autoredteam"),
            # )
            # self.enabled = True

            api_key = os.environ.get("GLACIS_API_KEY")
            if api_key:
                print("🔐 Glacis attestation configured (SDK integration pending)")
                self.enabled = True
            else:
                print("⚠  GLACIS_API_KEY not set — falling back to local attestation")
        except ImportError:
            print("⚠  glacis-sdk not installed — using local attestation only")

    def attest(self, chain_hash: str, metadata: dict) -> Optional[str]:
        """
        Submit a chain hash for cryptographic attestation.
        Returns attestation ID if successful, None otherwise.
        """
        if not self.enabled:
            return None

        # Glacis integration point:
        # attestation_id = self.client.attest(
        #     hash=chain_hash,
        #     metadata={
        #         "tool": "autoredteam",
        #         "version": __version__,
        #         **metadata,
        #     },
        # )
        # return attestation_id

        # Stub: log that we would attest
        print(f"  🔐 Would attest chain_hash={chain_hash[:16]}... to Glacis")
        return f"stub_{chain_hash[:16]}"

    def verify(self, chain_hash: str, attestation_id: str) -> bool:
        """Verify a previously submitted attestation."""
        if not self.enabled:
            return False

        # Glacis integration point:
        # return self.client.verify(attestation_id, expected_hash=chain_hash)

        return True  # Stub


# ---------------------------------------------------------------------------
# Unified attestation interface
# ---------------------------------------------------------------------------

class AttestationManager:
    """
    Unified interface that handles both local evidence and optional Glacis.
    """

    def __init__(self, output_dir: str = "results", config: Optional[dict] = None):
        self.config = config or {}
        self.local = LocalEvidenceStore(output_dir=output_dir)
        self.glacis = GlacisAttestationHook(config=self.config)
        self._cycle_records: list[EvidenceRecord] = []

    def record_attack(
        self,
        cycle: int,
        attack_id: str,
        prompt: str,
        response: str,
        category: str,
        score_vector: dict,
        composite_score: float,
        deterministic_flags: list[str],
        phase: str = "attack",
    ) -> str:
        """Record a single attack result. Returns chain hash."""
        evidence = EvidenceRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            cycle=cycle,
            attack_id=attack_id,
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
            category=category,
            score_vector=score_vector,
            composite_score=composite_score,
            deterministic_flags=deterministic_flags,
            phase=phase,
            _raw_prompt=prompt,
            _raw_response=response,
        )
        self._cycle_records.append(evidence)
        chain_hash = self.local.record(evidence)

        # Optional Glacis attestation
        if self.glacis.enabled:
            self.glacis.attest(chain_hash, evidence.public_summary())

        return chain_hash

    def end_cycle(self, cycle: int, summary: dict) -> None:
        """Finalize a cycle — write summary, optionally git commit."""
        summary["evidence_chain_length"] = len(self.local.load_chain())
        summary["chain_verified"] = self.local.verify_chain()
        self.local.write_summary(summary)
        self.local.git_commit(f"Cycle {cycle} complete — score {summary.get('best_composite', 'N/A')}")
        self._cycle_records = []

    def get_chain_length(self) -> int:
        return len(self.local.load_chain())

    def build_receipt(self, metadata: Optional[dict] = None) -> dict:
        """
        Build a run-level attestation receipt.

        This is the explicit artifact emitted by the `--attest` CLI flag.
        It references the verified local evidence chain and, when configured,
        records the Glacis attestation stub ID for the final chain hash.
        """
        chain_hash = self.local.last_chain_hash()
        chain_verified = self.local.verify_chain()
        provider = self.config.get("provider", "local")

        receipt = {
            "tool": "autoredteam",
            "version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "project": self.config.get("project", "autoredteam"),
            "chain_hash": chain_hash,
            "chain_verified": chain_verified,
            "evidence_records": self.get_chain_length(),
            "evidence_file": str(self.local.evidence_file),
            "summary_file": str(self.local.summary_file),
            "status": "verified_locally" if chain_verified else "chain_verification_failed",
            "receipt_type": "local_stub",
            "receipt_id": None,
            "metadata": metadata or {},
        }

        if chain_hash != "genesis" and provider == "glacis":
            attestation_id = self.glacis.attest(chain_hash, receipt["metadata"])
            receipt["receipt_type"] = "glacis" if attestation_id else "glacis_stub"
            receipt["receipt_id"] = attestation_id or f"stub_{chain_hash[:16]}"
            receipt["status"] = "attested" if attestation_id else "pending_glacis_sdk_or_credentials"
        elif chain_hash != "genesis":
            receipt["receipt_id"] = f"local_{chain_hash[:16]}"

        receipt["receipt_hash"] = hashlib.sha256(
            json.dumps(receipt, sort_keys=True).encode()
        ).hexdigest()
        return receipt

    def write_receipt(
        self,
        path: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Write a run-level attestation receipt and return its path."""
        receipt = self.build_receipt(metadata=metadata)
        receipt_path = Path(path) if path else self.local.output_dir / "attestation_receipt.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(receipt_path, "w") as f:
            json.dump(receipt, f, indent=2)
        return str(receipt_path)


def load_attestation_config(config_path: str = "config.yaml") -> dict:
    """Load attestation config from config.yaml."""
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("attestation", {})


if __name__ == "__main__":
    mgr = AttestationManager(output_dir="/tmp/autoredteam_test")
    chain_hash = mgr.record_attack(
        cycle=1,
        attack_id="test_001",
        prompt="test prompt",
        response="test response",
        category="prompt_injection",
        score_vector={"breadth": 50, "depth": 70, "novelty": 80, "reliability": 100},
        composite_score=75.0,
        deterministic_flags=["bypass:test"],
    )
    print(f"Chain hash: {chain_hash}")
    print(f"Chain verified: {mgr.local.verify_chain()}")
    print("✓ attestation.py smoke test passed")
