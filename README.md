# autoredteam

**Find prompt injection, jailbreaks, PII leakage, and system prompt exposure in your LLM systems — automatically.**

Point it at a supported model provider or API endpoint. Get a scored local vulnerability report and, when fulfilling a requirement, a signed claim with a bounded disclosure structure.

```
pip install glacis-autoredteam
autoredteam run --provider openai --model gpt-4o-mini
```

```
╔══════════════════════════════════════════════════════════════╗
║                    autoredteam v0.3.1                        ║
║         Automated Red-Teaming for AI Systems                 ║
╚══════════════════════════════════════════════════════════════╝

  Provider:  openai
  Model:     gpt-4o-mini
  Probes:    38
  Output:    results/

  ✓ 19 attack categories tested
  ✓ 4-dimension scoring (breadth · depth · novelty · reliability)
  ✓ Structured local per-probe results
  ✓ Local report + JSONL; request-bound receipt with --requirement
```

## What It Finds

autoredteam's generic pack draws from **19 attack categories** and can apply seeded prompt mutations before executing the selected probes:

| Category | Example |
|---|---|
| **Prompt injection** | Override system instructions via direct/indirect injection |
| **Jailbreaks** | Bypass safety via role-play, academic framing, fictional scenarios |
| **PII extraction** | Trick the model into leaking personal data |
| **System prompt leakage** | Extract internal instructions or system prompts |
| **Tool misuse** | Abuse available tools or trigger unintended actions |
| **Multi-turn manipulation** | Build up attacks across conversation turns |
| **Encoding bypass** | Evade filters using obfuscation or encoding |
| + 12 more | Role confusion, payload splitting, social engineering, ... |

Domain-specific attack packs are available for **healthcare**, **finance**, **HR**, and **coding agents**.

## Quickstart

```bash
# Install from PyPI
pip install glacis-autoredteam

# Dry run — echo target, no API keys needed
autoredteam run --dry-run

# Point at a real system
export OPENAI_API_KEY=sk-...
autoredteam run --provider openai --model gpt-4o-mini
```

The dry run uses an echo target that simulates a naive model. It shows attack-pack generation, target execution, scoring, and local result persistence without sending attack traffic to a model provider.

A full run against a real model takes 5–20 minutes depending on probe count and judge configuration.

## Fulfill an enterprise evidence request

AutoRedTeam can consume the experimental public **AI Evidence Requirement v0.1** and return a request-bound Red-Team Run Receipt with a bounded disclosure structure. The requirement fixes the request ID, unpredictable nonce, profile and version, subject/build commitment, test-profile commitment, completeness rule, freshness window, disclosure limits, and minimum assurance predicates before the vendor runs the test.

```bash
# Create a persistent local signing key once. Keep issuer.key private.
autoredteam keygen --output issuer.key --public-key-output issuer.pub

# As the buyer, generate and download a fresh requirement.json at:
# https://www.glacis.io/require
autoredteam run --dry-run \
  --requirement requirement.json \
  --signing-key issuer.key \
  --output-dir results/request-018f5ec7

# Give the reviewer receipt.json, redacted_evidence.json, and the requirement.
autoredteam verify results/request-018f5ec7/receipt.json \
  --requirement requirement.json \
  --artifact results/request-018f5ec7/redacted_evidence.json \
  --trusted-public-key issuer.pub
```

The repository's committed `examples/requirement-v0.1.json` is a deterministic conformance fixture for contributors; it is not installed as a working-directory template by the PyPI wheel. Its nonce is synthetic. A real buyer must generate a new, unpredictable 32-byte CSPRNG challenge for every request and encode it as 43-character unpadded base64url. The validator can check canonical encoding and length, not randomness or reuse. Never reuse the fixture nonce.

The reviewer sees separate checks for signature integrity, exact request digest and nonce binding, profile/subject/build match, artifact digest, completeness and omissions, criteria, freshness, expiry, and required assurance. Unsupported tests or assurance predicates are emitted as incomplete or not evaluated; AutoRedTeam does not silently substitute a different profile or unsigned artifact.

The self-signed result has a narrow meaning: **the displayed Ed25519 key signed the claim.** It does not independently prove that the run executed, that the claimed time is accurate, that the signer controls an organization, or that the tested system is safe. Passing `--trusted-public-key` checks the signer against a key the reviewer obtained separately; without it, the key is self-declared. Current key revocation or signer status remains unknown offline unless the reviewer separately checks a fresh, trusted status object.

### Enterprise-review journey

1. A buyer sends a versioned requirement for the in-scope AI application and build.
2. The vendor runs the requested profile and keeps detailed prompts, outputs, transcripts, and findings private.
3. AutoRedTeam returns `receipt.json` plus `redacted_evidence.json`, both bound to the original requirement digest and nonce. The redacted artifact includes scope, method, category outcomes, omissions, criteria results, and the private-report digest.
4. The buyer verifies the requirement, receipt, artifact, freshness, and expected signer key without a Glacis account.
5. If the buyer requires organization binding, independently anchored time, execution attestation, or independent observation, the self-signed adapter reports that requirement as incomplete. A stronger assurance workflow is required; it is not simulated here.

### Data egress, precisely

Receipt creation and verification use the Glacis Python SDK's offline signing path and make no request to Glacis. Glacis need not receive raw prompts, outputs, transcripts, or detailed findings. In requirement mode, runner files stay under a mode-0700 `private/` tree; only `receipt.json` and `redacted_evidence.json` are intended to be shared.

AutoRedTeam does not upload artifacts to the requirement's `return.uri`; returning the two shareable files is an explicit operator or surrounding-workflow step.

Use a fresh output directory for each request. Requirement mode rejects unknown files in the shareable root instead of mixing a new receipt with stale raw or legacy output.

The red-team attack traffic is a separate boundary: prompts and responses go to the target endpoint or model provider you configure (OpenAI, Anthropic, Bedrock, an internal endpoint, and so on). `--dry-run` uses the local echo target. Do not describe a real provider-backed campaign as “zero egress.”

Results land in `results/`:

```
results/
├── private/                   # Private local tree; mode 0700
│   ├── findings.json          # Private raw detail; mode 0600
│   ├── glacis-offline.db      # Local SDK signing store; mode 0600
│   └── run/                   # Private runner output; mode 0700
│       ├── campaign_manifest.json
│       ├── campaign_result.json
│       ├── campaign_state.json
│       └── probe_results.jsonl # All runner files mode 0600
├── redacted_evidence.json     # Shareable, structurally bounded summary
└── receipt.json               # Request-bound self-signed claim
```

## Who It's For

- **AI/ML engineers** shipping LLM features who need to test before deploy
- **Security teams** evaluating third-party AI integrations
- **Compliance teams** collecting a scoped test record for review (not a compliance determination)
- **Researchers** studying LLM robustness and attack surfaces

## How It Works

The public `run` command executes a bounded campaign assembled from attack packs. The generic pack samples the taxonomy and can apply one seeded mutation while building probes:

```
Generate attacks (from taxonomy + mutations)
       ↓
Execute against target
       ↓
Score results (deterministic by default; optional local SLM)
       ↓
Record evidence, write report
```

**Phase 1 — Attack:** Find vulnerabilities and score the bounded set of generated probes. The public CLI does not claim an iterative keep/discard convergence run.

**Phase 2 — Defend:** The repository contains hardening research code, but the `harden` command is not available in the public OSS CLI. A Red-Team Run Receipt does not claim that remediation happened.

**Phase 3 — Emit Policy:** `emit-policy` can generate an OVERT-oriented `policy.toml` draft from compatible hardening results. Emission is not an OVERT conformance assessment or certificate.

### Scoring

Every attack is scored on four dimensions to prevent single-metric collapse:

| Dimension | What it measures |
|---|---|
| **Breadth** | How many attack categories find bypasses |
| **Depth** | Severity of the bypass (0 = refusal, 100 = full compliance) |
| **Novelty** | How different from prior attacks |
| **Reliability** | Does the attack reproduce consistently |

The public CLI uses deterministic checks by default, including keyword matching and regex for PII or system-prompt patterns. `--judge-backend slm` can attempt the repository's optional local judge model; if that model is unavailable, the result records the limitation instead of silently calling a hosted judge. The public CLI does not expose API or dual-judge scoring.

### Mutation Engine

Attack-pack utilities include rephrase, encode, nest, persona-shift, language-switch, format-change, and authority-escalation mutations. The bounded public `run` command does not claim a repeated evolutionary or convergence cycle.

## Multi-Cloud Support

Built-in adapters cover these LLM providers:

```bash
autoredteam run --provider openai --model gpt-4o-mini
autoredteam run --provider anthropic --model claude-sonnet-4-5
autoredteam run --provider bedrock --model claude-sonnet-4 --region us-east-1
autoredteam run --provider google --model gemini-2.0-flash
autoredteam run --provider azure_openai --model gpt-4o
autoredteam run --provider cloudflare --model @cf/meta/llama-3-8b-instruct
```

Or bring your own target:

```python
from prepare import Target, TargetCapabilities, TARGET_REGISTRY

class MyTarget(Target):
    def send(self, prompt: str) -> str:
        return my_api.chat(prompt)

    def reset(self) -> None:
        my_api.new_session()

    def capabilities(self) -> TargetCapabilities:
        return TargetCapabilities(multi_turn=True)

TARGET_REGISTRY["my_target"] = MyTarget
```

## CLI Reference

```bash
# Red-team campaigns
autoredteam run --dry-run                                     # Echo target, no API keys
autoredteam run --provider openai --model gpt-4o-mini         # Full run
autoredteam run --pack generic_taxonomy healthcare            # Multiple attack packs
autoredteam run --stealth-profile medium                      # Stealth mode
autoredteam run --judge-backend slm                           # Optional local judge model
autoredteam run --requirement requirement.json \
  --signing-key issuer.key                                    # Request-bound receipt

# Local signing and verification
autoredteam keygen --output issuer.key --public-key-output issuer.pub
autoredteam verify results/receipt.json --requirement requirement.json \
  --artifact results/redacted_evidence.json --trusted-public-key issuer.pub

# Validation suites
autoredteam validate --suite generic --provider openai --model gpt-4o-mini

# Policy generation
autoredteam emit-policy results/autoharden/                   # Generate OVERT policy.toml

# Discovery
autoredteam providers list                                    # Available providers
autoredteam packs list                                        # Available attack packs
```

## Evidence & Attestation

The request-bound public path uses the declared `glacis>=0.8.1,<0.9` dependency for offline Ed25519 signing. Legacy runner fields may carry a chain-hash reference only when an attestation manager was explicitly supplied; the public `run` command does not present that reference as independent proof.

The requirement digest removes only the optional top-level `buyer_signature`, rejects JSON numbers in v0.1, canonicalizes the remaining object with RFC 8785 JCS rules, and hashes:

```text
SHA-256(ASCII("AI-EVIDENCE-REQUIREMENT/0.1\n") || UTF8(JCS(requirement)))
```

The output is lowercase `sha256:<64 hex>`. The example vector is `sha256:e143ddf7908f7a7b7df81da8f8ca1736e5cec09a74d80683f7c32094aff4af5c`.

### Compatibility boundary

`receipt.json` is explicitly an `experimental-detached-adapter`. The public Glacis SDK 0.8.x `Attestation` model does not have a first-class AI Evidence Requirement fulfillment field. AutoRedTeam therefore binds a domain-separated digest of the fulfillment statement through the SDK's signed `control_plane_results` extension seam. Verification checks both the SDK signature and that binding.

This is a real public-SDK offline signature, but it is **not** claimed to be a canonical Glacis Notary receipt, a transparency-log inclusion, a witnessed timestamp, or proof of execution. There is no silent fallback if the SDK is missing, incompatible, or cannot sign.

If an optional `buyer_signature` block is present, this adapter checks its shape and signed-digest binding but does not resolve the buyer key or cryptographically authenticate the buyer. That trust check is separate.

The verifier enforces exact artifact bytes and rejects prohibited raw-content field names. It does not classify every arbitrary string value as sensitive; raw-content exclusion is additionally covered by construction tests using sentinel prompts and outputs.

Hash separation keeps sensitive attack details scoped:

| Tier | Contains |
|---|---|
| **Shareable** | Requirement binding, scope/method, category outcomes, omissions, criteria, assurance status, private-report digest |
| **Private** | Raw prompts, responses, transcripts, detailed per-probe results; local file mode 0600 |
| **Witnessed** | Not emitted by this adapter |

## OVERT Policy Output

For compatible hardening results, `emit-policy` generates an [OVERT](https://overt.is)-oriented `policy.toml` capturing the declared configuration. The public `harden` CLI is unavailable, and policy generation alone does not establish OVERT conformance:

```bash
# Generate policy from autoharden results
autoredteam emit-policy results/autoharden/ --profile healthcare-ambient

# From a specific report
autoredteam emit-policy results/autoharden/autoharden_report.json -o deployment/policy.toml
```

The policy can include input/output filtering rules, violation types, tool-call deny rules, a hardened system prompt, attestation configuration, and a source chain-hash reference when the input carries one. A hash reference establishes integrity linkage only; it does not independently prove execution or conformance.

## Configuration

See `config.yaml` for all options. Key settings:

```yaml
target:
  type: openai              # openai, anthropic, gemini, azure_openai,
  params:                   # bedrock, cloudflare, openai_compatible, echo
    model: gpt-4o-mini
    system_prompt: "You are a helpful customer service bot."

campaign:
  max_probes: 20
  intensity: medium         # low, medium, high
  stealth_profile: none     # none, light, medium, aggressive

scoring:
  judge_backend: deterministic  # public CLI: deterministic or slm
  weights: { breadth: 0.25, depth: 0.25, novelty: 0.25, reliability: 0.25 }
```

## Roadmap

- [x] v0.1 — Single-turn text attacks, deterministic scoring, and an optional local SLM path
- [x] v0.2 — Multi-turn attack chains, agentic target support
- [x] v0.3 — Hardening research components, OVERT-oriented policy output, multi-cloud providers; public `harden` CLI unavailable
- [ ] v0.4 — Image/multimodal attack vectors, recursive policy hardening
- [ ] v1.0 — Full OVERT standard conformance, compliance reporting

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. All contributions welcome — from bug reports to new attack packs.

## Citation

If you use autoredteam in research, see [CITATION.cff](CITATION.cff) or cite:

```bibtex
@software{autoredteam,
  title = {autoredteam: Automated Red-Teaming for AI Systems},
  author = {Glacis},
  url = {https://github.com/Glacis-io/autoredteam},
  license = {Apache-2.0}
}
```

## License

[Apache 2.0](LICENSE)

---

Built by [Glacis](https://glacis.io). Local testing, public-SDK offline signing, and verification are free. Organization binding, accepted independent assurance, recurrence, retention, fleet operations, and continuous runtime evidence are separate higher-assurance capabilities.
