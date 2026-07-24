<div align="center">

# Resonant Learning Fabric (RLF)

### A Non-Neural Learning Architecture — Research Prototype & Evidence Package

**Sparse resonant dynamics · phase-vector representations · local learning · no backpropagation**

Built by **Mekan Bahmanjeh** · License: [MIT](LICENSE)

</div>

---

> **What this is.** RLF is an experimental learning architecture that learns **without neural
> networks**: no layers, no artificial neurons, no weight matrices as the learning substrate, no
> backpropagation, no automatic differentiation, and no Adam-style gradient optimizers. Learning
> instead emerges from **distributed phase-vector representations, sparse mode retrieval, recurrent
> settling, local learning, structural growth, and associative memory**. This repository is the
> complete, buildable, tested C++23 research prototype — RLF-0 through RLF-7, the Frontier research
> edition, and the Solstice multimodal configurations — together with reproducible experiment,
> audit, and evidence-generation tooling.

> **What this is not.** RLF does **not** claim GPT-level intelligence, commercial frontier-model
> parity, or a broadly pretrained model. The immediate scientific objective is narrower and
> falsifiable: *determine experimentally whether sparse resonant dynamics can learn and preserve
> reusable transformations more efficiently than a conventional baseline.* Every claim in this
> repository is gated against real, hash-bound evidence — see **[True Evidence](#-true-evidence)**.

---

## Table of Contents

- [The Core Idea](#-the-core-idea)
- [Scientific Discovery](#-scientific-discovery)
- [Academic Claims](#-academic-claims)
- [True Evidence](#-true-evidence)
- [Architecture Lineage: RLF-0 → Frontier](#-architecture-lineage-rlf-0--frontier)
- [The Solstice System](#-the-solstice-system)
- [Quick Start](#-quick-start)
- [Single-H200 Campaign](#-single-h200-campaign)
- [Training & Usage](#-training--usage)
- [Benchmarks & Validation](#-benchmarks--validation)
- [Repository Layout](#-repository-layout)
- [Roadmap: Image Generation from 1,000–3,000 Labeled Images](#-roadmap-image-generation-from-10003000-labeled-images)
- [Engineering Standards](#-engineering-standards)
- [Scientific Status & Honest Boundaries](#-scientific-status--honest-boundaries)
- [License & Citation](#-license--citation)

---

## 🧠 The Core Idea

Conventional machine learning stores knowledge in dense weight matrices and updates them by
backpropagating gradients. RLF asks a different question: **can a system learn by resonance?**

The substrate is a **phase vector** — a state of dimension `D` where each component lives on the
unit circle:

```
z ∈ ℂ^D ,  |zⱼ| = 1
```

Information is encoded in the *relative phases* of many components, not in scalar weights. On this
substrate RLF builds its learning machinery:

| Mechanism | What it does | Replaces |
|---|---|---|
| **Distributed phase vectors** | Represent states/symbols as unit-magnitude complex vectors | Dense embeddings + weights |
| **Sparse mode retrieval** | Recall a small set of resonant "modes" per query | Full dense attention |
| **Recurrent settling** | Iterate dynamics until the state stabilizes | Layered forward passes |
| **Local learning** | Update only the modes that participated | Global gradient updates |
| **Structural growth** | Add new modes/prototypes as novelty appears | Fixed-size weight tensors |
| **Associative memory** | Content-addressed storage and recall | Key/value caches |

The hypothesis under test: this style of learning can **acquire and preserve reusable
transformations with far less supervision and far fewer candidate operations** than an explicit
memorization baseline — because one demonstrated *schema* can be induced once and then reapplied,
rather than re-learned for every label.

---

## 🔬 Scientific Discovery

The central empirical finding of this research program, established within **explicitly narrow,
controlled synthetic scopes** by the fail-closed `rlf_efficiency_proof` runner:

> **A single demonstrated relational schema, induced in one shot, replaces 10,000 held-out target
> labels that an explicit memorization baseline requires.** Indexed variable joins then answer the
> *same exact queries* while examining **12,502.5×** fewer candidate fact unifications, and
> post-index exact retrieval over 524,288 vectors uses **17,480×** fewer inference components.

This is the discovery the whole repository is built to make reproducible and auditable: **one-shot
schema induction + indexed sparse retrieval yields orders-of-magnitude reductions in supervision and
candidate operations, at 100% accuracy, inside a controlled scope.** It is a *mechanism* result —
not a general-intelligence result — and the repository is engineered so that the difference is
machine-enforced (the broader claims are hard-coded to fail closed until external evidence exists).

---

## 🎓 Academic Claims

Claims are stated with exact definitions, metrics, datasets, baselines, and current evidence
status. Every claim below is gated against fail-closed proof runners and hash-bound artifacts.
Generated run artifacts live under the git-ignored `results/` tree; the source package includes
the inputs and commands needed to reproduce them.

### Supported within declared scope

| Claim | Evidence | Status |
|---|---|---|
| **Narrow structured-task efficiency** — schema induction & indexed retrieval meet task-scoped thresholds without changing answers | `rlf_efficiency_proof` (fail-closed runner, see [Run the packaged proof](#run-the-packaged-proof)) | ✅ **Supported (synthetic scope only)** |
| **10,000× target-supervision efficiency** | 1 demonstrated schema vs 10,000 held-out labels, memorization baseline, 100% acc | ✅ Scoped proof passed |
| **12,502.5× fewer candidate unifications** | Indexed variable joins, same exact answers | ✅ Scoped proof passed |
| **17,480× fewer retrieval components** | 524,288 vectors, 3 deterministic seeds | ✅ Scoped proof passed |
| **Strict local CPU build** — warnings-as-errors, full suite green | GCC 13 Release 29/29 CTest; focused Solstice 130/130; Windows 4/4; ASan+UBSan paths pass | ✅ Proven on CPU |
| **Audited dataset admission** — refuses bad SHA-256, unapproved shards, duplicates, train/eval contamination | `solstice audit-data` + `tests/test_data_pipeline.cpp` | ✅ Mechanism proven |
| **Deterministic shard resume** — byte-identical checkpoint on re-run | `scripts/run_checkpoint_transaction.sh` + `tests/test_checkpoint_workflow.cpp` | ✅ Proven at shard boundaries |
| **Checkpoint failure recovery** — transactional, corruption-rejecting, format-6 round-trip | `tests/test_data_pipeline.cpp` | ✅ Supported for tested failure modes |

### Explicitly NOT proven (fail-closed by construction)

| Claim | Why it stays false |
|---|---|
| Broad frontier-model parity | Parity gate reports **0/8 targets, 0/3,018 required external examples** |
| General 10,000× / 100,000× efficiency | No matched-quality external evidence; component ratios are never multiplied into a general claim |
| SOTA image generation | No unrestricted CUDA generator, no physical run, no external quality scores |
| Broad language/vision/tool competence | ARC-AGI-2 public measured at **0/120 exact tasks**; other external families unrun |
| H100/H200 CUDA correctness and physical scale training | Guards and profiles implemented; no real H100/H200 training trace exists yet |

> **Falsification discipline:** the `rlf_general_benchmark` and parity-gate executables are
> *fail-closed*. They are incapable of marking broad parity from internal or synthetic tests.
> Controlled/synthetic results can never be promoted into external frontier claims.

---

## 🧾 True Evidence

"True evidence" here means **verifiable, hash-bound artifacts** — every claim links to a file you
can inspect and recompute. Nothing rests on assertion.

### Build & test (verified 2026-07-24)

- **Linux GCC 13, C++23 Release, warnings-as-errors:** 29/29 CTest targets pass (133.38 s).
- **Focused Solstice suite:** 130/130 pass.
- **Core aggregate suite:** 261/261 pass.
- **Windows MSVC matrix:** 4/4 pass (39.84 s).
- **Combined ASan+UBSan:** core 119/119 plus affected image/V100 workflows 4/4, `halt_on_error=1`.
- **Frontier-specific suite:** 12/12 pass.
- **Full 10K proof:** reproduce with `./scripts/run_efficiency_proofs.sh`.

### Packaged proof metrics (controlled scope, 100% accuracy)

| Metric | Ratio | Accuracy | Scope |
|---|---:|---:|---|
| Target supervision | **10,000.0×** | 100% | Held-out two-hop schema labels |
| Indexed reasoning candidates | **12,502.5×** | 100% | Candidate fact unifications |
| Exact retrieval inference | **17,480.06×** | 100% | Post-index exact queries, 3 seeds |

Capability gates passed: held-out compositional generalization, one-shot schema induction,
unsupported-conclusion negative control, cross-domain structural transfer, continual retention,
contrastive multimodal grounding.

### RLF-7 controlled knowledge & multimodality (100,000-record reference)

Exact retrieval 1.0000 · candidates/query 1.0000 · image classification 1.0000 · localization IoU
1.0000 · video direction 1.0000 · track continuity 1.0000 · audio classification 1.0000 ·
cross-modal link audit 1.0000 (withheld generated examples — **not** open-world benchmarks).

### Integrity & audit mechanisms (implemented + locally tested)

- Native SHA-256 dataset verification, provenance/approval checks, exact dedup, bounded
  SimHash/MinHash train/eval contamination screening, fail-closed `audit-data`.
- **Checkpoint format 6:** incremental, transactional, corruption-rejecting, deterministic
  shard-level resume, read-compatible back to v1.
- **Nine-artifact self-hashed run manifest** binding checkpoint/data/source/environment/audit/VRAM/
  readiness; tampering rejected.
- Hash-bound resumable batch prediction runner for external evaluation.
- **Measured negative result:** preregistered ARC-AGI-2 public run, 0/120 tasks, all raw
  predictions retained and deliberately excluded from gate evidence.

### Build provenance

All binaries are built from the sources in this repository with the CMake presets in
`CMakePresets.json` (see [Quick Start](#-quick-start)). The build is deterministic, seeded, and
compiles warning-free under `-Werror` with both GCC and Clang.

### Strongest bundled checkpoint (honestly labeled)

`models/solstice_general_h100_bootstrap.rlfsp` — format-5 bootstrap, SHA-256
`3209d5830f9a5f98d1f4728b28160217fd362ee844966afdee8816cf08718c3c`, containing **5,375 language
tokens, 31 episodes, 4 visual examples, 12 demonstrations, 4 preferences.** It is a *bootstrap for
the training pipeline*, **not** a broadly pretrained model.

---

## 🧬 Architecture Lineage: RLF-0 → Frontier

The program progressed through documented, individually-specified milestones. Each milestone was
developed against a written specification, a benchmark protocol, and a results report, and each has
dedicated deterministic regression suites in `tests/` and configs in `configs/`.

| Stage | Focus |
|---|---|
| **RLF-0** | Deterministic phase-vector substrate |
| **RLF-1** | Resonant modes, credit assignment |
| **RLF-2** | Recurrent settling, causal credit |
| **RLF-3** | Local + structural learning, memory & search |
| **RLF-4** | Associative memory, temporal abstraction |
| **RLF-5** | Language learning |
| **RLF-6** | Agent architecture: tools, memory, uncertainty, safety |
| **RLF-7** | Controlled knowledge + multimodality at 100K records |
| **Frontier** | One-shot schema induction, compositional reasoning, continual learning, multimodal grounding, sparse routing |

The local chronological experimental record — including deliberately retained **negative
results** — is written as hash-bound JSON/text artifacts under `results/`. Generated runs and
external datasets are intentionally excluded from the source repository.

### Frontier Research Edition components

- **`AbstractionFabric`** — facts, variable-bearing rules, multi-hop proofs, relation-structure
  transfer, and `induce_chain_rule` (shortest-supported-path schema induction from one labeled
  demonstration).
- **`ContinualLearningFabric`** — novelty-driven local prototypes, importance-based
  stability/plasticity, balanced replay, consolidation.
- **`CrossModalGroundingFabric`** — positive/negative region–word binding, compositional concept
  retrieval.
- **`SparseRoutingIndex`** — deterministic multi-probe candidate routing for large mode tables.
- **`rlf_efficiency_proof`** / **`rlf_frontier_benchmark`** — fail-closed, JSON-producing runners.
- **`frontier-h100` / `general-h100`** — H100 80 GB capacity profiles.

---

## 🖥️ The Solstice System

**Solstice** is the heavy multimodal RLF configuration. **Solstice-General-Frontier** combines
multiscale image understanding, sparse hierarchical text generation, associative image/dialogue
episodes, typed policy-controlled tool calls, CPU-resident long-term fabric state, and optional
persistent CUDA visual-mode caching. It uses **no Transformers, no neural layers, no
backpropagation, and no imported foundation-model weights.**

### `frontier-24g` profile (RTX 3090 default)

- 65,536 tokenizer pieces · language contexts through 2,048 tokens
- Ceilings of 20M predictive contexts and 2M episodes
- Four visual scales (8/16/32/64 px) · 32-dim patch descriptors
- Up to 1,024-px images, 24,576 patches/image · 262,144 local visual modes · 256 retained regions
- Bounded query/candidate shards · persistent CUDA candidate cache · streaming TSV manifests
- Native PNG/JPEG via libpng/libjpeg · checkpoint format 6 (v1–5 readable)
- Design target ≈ **21 GiB GPU working set** (tables grow with training)

---

## 🚀 Quick Start

### CPU build (validated path)

```bash
cmake --preset ubuntu-release
cmake --build --preset ubuntu-release --parallel "$(nproc)"
ctest --preset ubuntu-release --output-on-failure
# run with --backend optimized_cpu
```

### Ubuntu + RTX 3090 (CUDA, `sm_86`)

```bash
chmod +x scripts/*.sh *.sh
./scripts/setup_ubuntu_3090.sh      # install build dependencies
./BUILD_UBUNTU_3090.sh
./scripts/bootstrap_frontier_3090.sh
./START_SOLSTICE_3090.sh            # executable: build/ubuntu-rtx3090-cuda/solstice
```

### Ubuntu + H100 80 GB (`sm_90`)

```bash
chmod +x scripts/*.sh *.sh
./BUILD_UBUNTU_H100.sh
./scripts/bootstrap_frontier_h100.sh
./START_SOLSTICE_H100.sh
```

### General-H100 campaign (fail-closed, audited)

```bash
./BUILD_UBUNTU_GENERAL_H100.sh
./scripts/check_general_h100_readiness.sh \
  --ledger /data/general/ledger.tsv --output results/codex_campaign/h100_readiness
# train only after a real readiness report passes:
./scripts/train_general_h100_audited.sh \
  /data/general/ledger.tsv models/solstice_general_h100.rlfsp \
  results/codex_campaign/h100_training \
  results/codex_campaign/h100_readiness/readiness.json
```

### Ubuntu + H200 141 GB (`sm_90`)

```bash
./BUILD_UBUNTU_H200.sh
./scripts/check_general_h200_readiness.sh \
  --ledger /data/h200/ledger.tsv --output results/h200/readiness
./scripts/h200/h200_12month_controller.sh --plan
```

The H200 build exposes the `general-h200-141g-30t` profile and exact cumulative
token targets. Paid scale training remains deliberately unavailable until a
physical throughput/resume promotion gate authorizes the selected stage. See
[`docs/H200_CAMPAIGN.md`](docs/H200_CAMPAIGN.md).

### Run the packaged proof

```bash
./scripts/run_efficiency_proofs.sh            # full proof
build/ubuntu-release/rlf_efficiency_proof --quick   # CI smoke proof
build/ubuntu-release/rlf_frontier_benchmark \
  --output results/frontier_research_benchmark.json # CPU mechanism benchmark
```

> **CUDA note:** the CPU Release build and full regression suite are validated. CUDA source and the
> `sm_86`/`sm_90` presets are included, but must be compiled and validated on the target GPU
> machine. Other build scripts: `BUILD_UBUNTU_GENERAL_CUDA_COMPAT.sh`,
> `BUILD_UBUNTU_RTX_PRO_6000.sh`, `BUILD_WINDOWS.bat`.

---

## 🧭 Single-H200 Campaign

The repository includes an **experimental, fail-closed** 12-month campaign
layout for one NVIDIA H200:

- staged cumulative language-token targets of **1T, 5T, 15T, and 30T**;
- a **15T primary goal** and **30T conditional goal**;
- a 132 GiB GPU working-set ceiling;
- exact tokenizer-piece accounting and immutable shard-level checkpoints;
- physical H200 identity, readiness, VRAM, provenance, contamination, resume,
  resource, and frozen-quality evidence requirements.

These targets describe RLF tokenizer pieces processed by sparse/local learning.
They are **not Transformer-equivalent pretraining tokens**, parameter counts, or
frontier-capability claims. The current controller intentionally blocks all
physical training stages until the remaining promotion gate is implemented and
validated on a real H200.

```bash
# Planning is safe and performs no training.
./scripts/h200/h200_12month_controller.sh --plan

# Initialize immutable campaign state.
binding="$(printf '%s' 'your audited campaign definition' | sha256sum | awk '{print $1}')"
./scripts/h200/h200_12month_controller.sh --start \
  --state-dir /data/rlf-h200-campaign --binding "${binding}"
```

Full prerequisites, dataset contracts, stage budgets, and honest claim
boundaries are documented in [`docs/H200_CAMPAIGN.md`](docs/H200_CAMPAIGN.md).

---

## 🏋️ Training & Usage

Create/bootstrap a checkpoint, then train each modality (streaming TSV manifests):

```bash
solstice bootstrap      --profile frontier-24g --backend cuda --checkpoint models/m.rlfsp
solstice train-text     --backend cuda --input /data/corpus.txt        --checkpoint models/m.rlfsp
solstice train-dialogue --backend cuda --manifest /data/dialogues.tsv  --checkpoint models/m.rlfsp
solstice train-vision   --backend cuda --manifest /data/vision.tsv     --checkpoint models/m.rlfsp
solstice train-tools    --backend cuda --manifest /data/tools.tsv      --checkpoint models/m.rlfsp
```

**Formats:** dialogue `prompt⇥response⇥optional grounding` · vision `image_path⇥caption` ·
tools `request⇥tool_name`. Relative image paths resolve from the manifest directory.

Query:

```bash
solstice ask --backend cuda --checkpoint models/m.rlfsp --prompt "What can you do?"
solstice ask --backend cuda --checkpoint models/m.rlfsp --image /data/x.png --prompt "What do you see?"
solstice chat --backend cuda --checkpoint models/m.rlfsp --tool-root /data/safe_tool_root
```

File tools are disabled unless a sandbox is supplied; built-ins are calculator, current time,
sandboxed file reading, and directory listing. No default shell or network execution.

**One-shot rule induction** (persists into the checkpoint, usable on unseen entities):

```bash
solstice induce-rule --checkpoint models/m.rlfsp \
  --subject alice --relation grandparent --object carol \
  --max-hops 2 --prompt "induced grandparent schema"
```

Checkpoint tooling: `solstice inspect-checkpoint` / `solstice verify-checkpoint`.

---

## 📊 Benchmarks & Validation

Benchmark runners emit hash-bound artifacts under `results/` and are reproducible with the commands
in [Run the packaged proof](#run-the-packaged-proof). Generated artifacts are git-ignored so a
source checkout remains compact.

### Packaged proof (controlled scope, 100% accuracy)

| Metric | Ratio | Accuracy | Scope |
|---|---:|---:|---|
| Target supervision | **10,000.0×** | 100% | Held-out two-hop schema labels |
| Indexed reasoning candidates | **12,502.5×** | 100% | Candidate fact unifications |
| Exact retrieval inference | **17,480.06×** | 100% | Post-index exact queries, 3 seeds |

Run it yourself:

```bash
./scripts/run_efficiency_proofs.sh                      # full proof
build/ubuntu-release/rlf_efficiency_proof --quick       # CI smoke proof
build/ubuntu-release/rlf_frontier_benchmark \
  --output results/frontier_research_benchmark.json     # CPU mechanism benchmark
```

### Test & validation record (verified 2026-07-24)

| Suite | Result |
|---|---|
| Linux GCC 13, C++23 Release, warnings-as-errors | 29/29 CTest targets pass (133.38 s) |
| Focused Solstice suite | 130/130 pass |
| Core aggregate suite | 261/261 pass |
| Windows MSVC matrix | 4/4 pass (39.84 s) |
| Combined ASan+UBSan | core 119/119 + image/V100 workflows 4/4, `halt_on_error=1` |
| Frontier-specific suite | 12/12 pass |

### Measured negative results (kept on the record)

- **ARC-AGI-2 public evaluation:** 0/120 exact tasks — preregistered run, raw predictions retained
  and deliberately excluded from gate evidence.
- **External parity gate:** 0/8 targets, 0/3,018 required external examples — fail-closed by
  construction; internal/synthetic results can never be promoted into external claims.

---

## 🗂️ Repository Layout

```
├── src/                     # C++23 source: core, learning, memory, retrieval,
│                            #   backend, solstice, frontier, agent, storage, cli,
│                            #   experiments, benchmarks, baselines  (179 C++/CUDA files)
├── include/rlf/             # Public headers
├── tests/                   # CTest regression suites
├── benchmarks/              # codex_campaign, efficiency_campaign, frontier_sample, general_frontier
├── configs/                 # Deterministic experiment configs
├── data_templates/          # Ledger/manifest templates
├── datasets/external/       # git-ignored external datasets, each under its own license
├── models/                  # Bootstrap checkpoints (.rlfsp) + SHA256SUMS.txt
├── results/                 # Evidence artifacts (git-ignored; regenerate via scripts/)
├── scripts/                 # Build / readiness / audited training / proof wrappers
├── examples/                # e.g. examples/solstice
├── .github/workflows/       # CI: GCC+Clang × debug/asan/ubsan/release + proof smoke
├── CMakeLists.txt / CMakePresets.json
├── BUILD_UBUNTU_*.sh / BUILD_WINDOWS.bat / START_SOLSTICE_*
├── README.md                # You are here — the single consolidated document
└── LICENSE                  # MIT
```

---

## 🖼️ Roadmap: Image Generation from 1,000–3,000 Labeled Images

This is the **next evidence target**: train and honestly evaluate a **non-neural RLF image learner**
on a small labeled dataset, as stronger evidence for (or against) the resonant-learning hypothesis.
The repository already contains an isolated, controlled RLF-native image component
(`image_generation_fabric.cpp`, `RLFIMG01` checkpoint format 4) that:

- passes held-out **transformation-composition** tests;
- supports audited exact-label prompt→image rows via a canonical neutral seed and caller-supplied
  source images;
- learns **distributional prompt semantics** locally, with sparse open-vocabulary prompt retrieval
  and conservative multi-operation parsing (`then`, `and then`, `followed by`, `;`);
- enforces native SHA-256, decoded-pixel dHash/aHash dedup, license allow-listing, and frozen
  train/eval contamination checks.

### Proposed protocol for a 1,000–3,000 image study

1. **Dataset contract (immutable manifests).** Three splits with all images below their manifest
   directory. For a small study, a fixed split such as **~70% train / 15% development / 15% frozen
   evaluation**. The evaluation split must include **≥100 prompts absent verbatim from training**
   and **≥100 composed-operation prompts** — these quotas measure *coverage*, not success. With
   1,000–3,000 labeled images this is a **mechanism/pipeline experiment**, not a SOTA text-to-image
   claim.
2. **Audit before training.** Native image-pair auditor verifies source/target bytes, licenses,
   SHA-256, dHash/aHash + color, and every split direction; unapproved or contaminated rows fail
   closed.
3. **Train the resonant patch/prompt-semantic modes** on CPU first (correctness before CUDA), with
   transactional checkpoints and byte-identical resume.
4. **Held-out tagged evaluation** — paraphrase, composition, natural, and multilingual tags — with
   raw RLF / nearest-neighbor / patch-quilt outputs, plus perceptual diversity and copying
   (memorization) metrics.
5. **Baseline comparison.** A same-split, frozen comparison against a current diffusion baseline,
   with a fail-closed result gate. RLF's efficiency claim is only meaningful at *matched quality*.
6. **Publish raw artifacts.** Hash-bound predictions and metrics so the result is independently
   checkable.

**Honest boundary:** 1,000–3,000 images is enough to test whether the resonant mechanism learns
reusable image transformations — it is **not** enough for a frontier text-to-image claim, and this
repository will not make one. All frontier/SOTA flags stay false until external evidence exists.

---

## 🛠️ Engineering Standards

- **C++23**, CMake, standard-library-first, Linux-primary; GCC **and** Clang supported.
- **CPU before GPU** — correctness and learning behavior established on CPU; backend interfaces kept
  CUDA-ready (`src/frontier/frontier_cuda.cu`).
- Contiguous, cache-aware storage; deterministic seeded execution; RAII and strict ownership.
- **Sanitizers** (ASan+UBSan) in development builds; **warnings-as-errors** in CI.
- Clear separation between **mathematical reference code** and **optimized code** (scalar reference
  vs optimized backends, with exact-agreement tests).
- **No Python in the core** — all training, evaluation, serialization, and benchmarking run through
  the C++ executables. (Python may later be used only for optional visualization/analysis.)
- No PyTorch, TensorFlow, JAX, ONNX, or neural-network libraries.

---

## ⚖️ Scientific Status & Honest Boundaries

RLF is a **buildable, tested, non-neural research framework** — not a broadly pretrained commercial
frontier model. "Frontier" in profile names (e.g. `frontier-24g`) identifies the largest implemented
RLF systems target, not a capability claim. Actual capability depends on training data, evaluation
discipline, and future algorithmic development.

**Status (2026-07-24):** *Not frontier-complete. External frontier parity is not proven. General
10,000×/100,000× efficiency is not proven. SOTA image generation is not proven.* What **is** proven
is the scoped mechanism result, the strict build/test posture, and a comprehensive set of
integrity/audit mechanisms. Everything broader is deliberately fail-closed pending external,
independently-evaluated evidence.

This honesty is a feature: the repository is designed so that **claims cannot be inflated** — the
parity gate, the benchmark runner, and the claim inventory are all fail-closed by construction.

---

## 📄 License & Citation

**License:** [MIT](LICENSE) — Copyright (c) 2026 **Mekan Bahmanjeh**.

If you use RLF in academic work, please cite this repository and reference the specific evidence
artifacts (validation reports, proof JSON, claim inventory) for any claim you rely on:

```bibtex
@misc{bahmanjeh2026rlf,
  author       = {Bahmanjeh, Mekan},
  title        = {Resonant Learning Fabric (RLF): A Non-Neural Learning Architecture},
  year         = {2026},
  howpublished = {\url{https://github.com/MekanBahmanjeh/resonant-learning-fabric}},
  note         = {Research prototype; MIT License}
}
```

---

<div align="center">
<sub>Built by <strong>Mekan Bahmanjeh</strong> · Resonant Learning Fabric · MIT License ·
Every claim gated against verifiable evidence.</sub>
</div>
