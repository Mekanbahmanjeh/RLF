# Single-H200 12-Month Campaign

This document describes the repository's experimental
`general-h200-141g-30t` campaign profile. It is an engineering and evidence
contract for testing whether RLF sparse/local learning can scale efficiently on
one NVIDIA H200. It is not evidence of frontier-model capability.

## Current Status

Implemented:

- H200 CUDA `sm_90` configure, build, and test presets.
- A 132 GiB guarded GPU working-set ceiling.
- An H200 readiness checker requiring a physical H200, CUDA 12 or newer,
  8 TiB host RAM, 16 TiB storage, a complete test pass, and an audited ledger.
- Exact cumulative language-token targets in `solstice train-data`.
- Deterministic bounded replacement for language contexts and episodes.
- Audited H200 training, artifact-manifest, and 8,760-hour controller scripts.
- Dataset ledger and vision-manifest templates.

Not yet implemented or proven:

- The physical token-throughput promotion gate.
- A real H200 build, throughput trace, VRAM trace, or scale checkpoint.
- Physical byte-identical exact-token resume evidence.
- A tokenizer-bound 15T/30T dataset census and distributed storage manifest.
- Bounded replacement or compaction for every non-language state table.
- External evidence of broad coding, language, reasoning, or multimodal parity.

The controller therefore rejects all training stages with exit status `5`.

## Hardware Contract

Real readiness requires:

- exactly one physical NVIDIA H200 with compute capability 9.0;
- CUDA Toolkit 12 or newer;
- at least 132 GiB free HBM for the guarded run;
- at least 8 TiB host RAM for the configured host-backed state ceilings;
- at least 16 TiB available checkpoint storage;
- the warning-free CUDA Release build and complete CTest suite;
- an immutable, hash-audited multimodal ledger.

The host RAM and storage requirements mean "single GPU" does not mean a small
single-node resource footprint.

## Build And Inspect

```bash
./BUILD_UBUNTU_H200.sh

build/ubuntu-h200-cuda/solstice profile-info \
  --profile general-h200-141g-30t

build/ubuntu-h200-cuda/solstice device-info \
  --backend cuda --profile general-h200-141g-30t
```

## Dataset Contract

Start from:

- `data_templates/h200-30t-multimodal/ledger.tsv.template`
- `data_templates/h200-30t-multimodal/vision.tsv.template`

Every immutable shard must have:

- a stable shard identifier;
- an approved source URI and license;
- a declared record count;
- a SHA-256 digest;
- a fixed train, development, or frozen-evaluation split;
- media hashes for referenced images;
- no train/evaluation exact, near-duplicate, or perceptual contamination.

The exact campaign target counts `language_tokens_seen`: tokenizer pieces
learned by the language fabric. Image bytes, facts, tools, preferences, and
other records are governed separately and do not increase that counter.
Consequently, an exact 15T or 30T ledger must be constructed from shard
boundaries that land exactly on the selected cumulative tokenizer-piece target.

## Readiness

```bash
./scripts/check_general_h200_readiness.sh \
  --ledger /data/h200/ledger.tsv \
  --output results/h200/readiness
```

Test doubles may exercise the checker but always produce `ready=false`. Only a
physical report with `ready=true` can be consumed by the audited trainer.

## Campaign Plan

```bash
./scripts/h200/h200_12month_controller.sh --plan
```

The plan allocates exactly 8,760 hours:

| Stage | Hours | Cumulative language target |
|---|---:|---:|
| Hardware/build/data/token audits | 912 | 0 |
| Physical throughput/resume probe | 72 | bounded probe only |
| 1T calibration | 500 | 1T |
| 1T frozen evaluation | 120 | 1T |
| 5T training | 1,500 | 5T |
| 5T frozen evaluation | 180 | 5T |
| 15T primary training | 2,400 | 15T |
| 15T frozen evaluation | 240 | 15T |
| 30T conditional training | 1,800 | 30T |
| Final evaluation/export/recovery | 1,036 | unchanged |

The 15T and 30T stages require substantially higher throughput than the annual
average because their stage budgets are shorter. Promotion must use measured
end-to-end throughput including audit, checkpoint, and recovery overhead.

## Exact-Token Training

The audited trainer accepts only the four campaign targets:

```bash
./scripts/train_general_h200_audited.sh \
  /data/h200/ledger.tsv \
  /data/checkpoints/solstice-h200.rlfsp \
  results/h200/training \
  results/h200/readiness/readiness.json \
  1000000000000
```

Do not invoke this command as authorization to begin paid scale training. The
campaign controller remains the authority and currently blocks training until
the physical promotion gate exists.

## Required Promotion Evidence

Before any target is authorized, the gate must verify a SHA-256-bound evidence
bundle containing:

1. physical H200 identity and CUDA configuration;
2. exact tokenizer and cumulative token census;
3. end-to-end physical token throughput with at least 20% timing headroom;
4. zero silent capacity skips or partial shard admission;
5. byte-identical interrupted/resumed and uninterrupted checkpoints;
6. frozen text-generation, image-understanding, and grounding non-regression;
7. provenance, license, deduplication, and contamination artifacts;
8. RAM, disk, checkpoint, and recovery projections with 20% headroom;
9. a training-only authorization ticket that cannot authorize a frontier claim.

## Claim Boundary

A completed 15T or 30T counter would prove only that the audited RLF pipeline
processed that many tokenizer pieces under the recorded constraints. It would
not by itself establish equivalence to a neural parameter count, a Transformer
training-token count, or a frontier model.

Capability claims require frozen external evaluations at matched quality,
including coding, language generation, reasoning, continual retention, image
understanding, multimodal grounding, and independently reproduced compute
evidence.
