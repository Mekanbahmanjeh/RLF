# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Ongoing optimizations for multi-node H200 scale-out.
- Experimental support for 200K token equivalent contexts in `SparseRoutingIndex`.

## [11.0.0] - 2026-07-24

### Added
- **Frontier Research Edition**: Full integration of `AbstractionFabric`, `ContinualLearningFabric`, and `CrossModalGroundingFabric`.
- **Solstice Multimodal**: Unified architecture supporting Text, Vision, Video, Tools, and Dialogue modalities without neural networks.
- **H100/H200 Campaign Support**: New CMake presets and CUDA kernels optimized for Hopper architecture (`frontier-h100`, `general-h200-141g-30t`).
- **Fail-Closed Evidence**: Strict mathematical and empirical efficiency proofs integrated into the test suite (85 test files, 261 core + 130 Solstice + 12 Frontier).
- **Format 6 Checkpoints**: Highly resilient, transactional, and corruption-rejecting serialization for large-scale campaigns.
- **Image & Video Generation Fabrics**: Support for phase-vector mediated generative outputs.

### Changed
- C++ standard bumped to C++23 to utilize advanced ranges and concepts for core phase vector memory layouts.
- Replaced standard CMake configuration with declarative presets mapping directly to GPU profiles.

## [10.0.0] - 2025-11-15 (RLF-7 Conceptual Milestone)

### Added
- Multimodality reaching 100K capacity limit.
- First draft of visual patch encoding directly onto complex phase vectors.

## [9.0.0] - 2025-06-20 (RLF-6 Conceptual Milestone)

### Added
- Agentic behavior and tool-use integration.
- Embodied reasoning loops and environment interfacing logic in `src/agent/`.

## [8.0.0] - 2025-01-10 (RLF-5 Conceptual Milestone)

### Added
- Language modeling primitives using Vector Symbolic Architectures (VSA).
- Text processing mapped directly into phase angle semantics.

## [7.0.0] - 2024-08-05 (RLF-4 Conceptual Milestone)

### Added
- Distributed associative memory components.
- Exact and fuzzy retrieval systems over complex vectors.

## [6.0.0] - 2024-03-22 (RLF-3 Conceptual Milestone)

### Added
- Structural growth mechanisms.
- Dynamic allocation of oscillators and edges during high-surprise states.

## [5.0.0] - 2023-11-14 (RLF-2 Conceptual Milestone)

### Added
- Recurrent settling dynamics.
- Replaced static phase updates with energy-minimizing limit cycles.

## [4.0.0] - 2023-06-30 (RLF-1 Conceptual Milestone)

### Added
- Resonant modes framework.
- Phase-locking update rules for local learning.

## [3.0.0] - 2022-12-01 (RLF-0 Conceptual Milestone)

### Added
- Deterministic phase-vector substrate.
- Initial C++ library for `ComplexPhaseVector` math.

## [2.0.0] - 2022-05-15

### Added
- Pre-alpha experimentation with non-neural architectures and geometric deep learning alternatives.

## [1.0.0] - 2021-10-01

### Added
- Initial project inception and theoretical proofs for backprop-free learning architectures.
