# Contributing to Resonant Learning Fabric (RLF)

First off, thank you for considering contributing to Resonant Learning Fabric. It's people like you that make RLF a powerful and reliable research platform. 

RLF is a C++23 research prototype for non-neural sparse resonant learning. Our engineering standards are rigorous because our research claims rely on the integrity of our software. We operate on a **fail-closed evidence system**, meaning that claims cannot be inflated, and negative results are reported honestly.

This document outlines the process and standards for contributing to the RLF project.

## How to Contribute

### 1. Issues
- **Bug Reports**: Provide a minimal reproducible example, system information (OS, compiler, hardware), and the expected vs. actual behavior.
- **Feature Requests**: Open an issue discussing the proposed feature before writing code. Since this is a research prototype, features must align with the current research trajectory (e.g., non-neural, sparse, phase-vector representations).
- **Discussions**: Use GitHub Discussions for questions, architectural discussions, or research ideas.

### 2. Pull Requests (PRs)
1. Fork the repository and create your branch from `main`.
2. Ensure your development environment is set up correctly (see below).
3. Write code that adheres to our style and architectural principles.
4. Add comprehensive tests.
5. Ensure all CTest suites pass locally (including sanitizers).
6. Update documentation as necessary.
7. Issue a PR with a detailed description and the PR checklist completed.

---

## Development Setup

RLF requires a fully compliant C++23 compiler and CMake.

### Prerequisites
- GCC 13+ or Clang 17+ (C++23 support required)
- CMake 3.25+
- Ninja (recommended)

### Build Commands
We use CMake presets to simplify the build process.

```bash
# Configure the project in Debug mode
cmake --preset debug

# Build the project
cmake --build --preset debug

# Configure the project in Release mode
cmake --preset release

# Build the project in Release mode
cmake --build --preset release
```

---

## Code Style

- **Standard**: C++23.
- **Paradigm**: Absolutely **NO** neural network libraries, no backpropagation, no PyTorch, no TensorFlow. RLF is strictly non-neural.
- **Formatting**: We use `clang-format`. Run it before submitting a PR.
- **Warnings**: We compile with `-Wall -Wextra -Wpedantic -Werror`. Warnings are treated as errors. Your code must compile cleanly.
- **Types**: Use standard integer sizes (`std::int32_t`, `std::uint64_t`) and `std::complex` for phase-vector representations in ℂ^D.

---

## Testing Requirements

We maintain an exceptionally high bar for testing. 100% test pass rate is non-negotiable.

- Currently passing: 261/261 core tests, 130/130 Solstice tests, 12/12 Frontier tests.
- **Sanitizers**: You must run tests with ASan (AddressSanitizer) and UBSan (UndefinedBehaviorSanitizer) enabled.
- **Command**:
  ```bash
  ctest --preset debug-sanitizers --output-on-failure
  ```

---

## Evidence Standards

RLF operates on a **fail-closed evidence system**.
- **Hash-bound Artifacts**: All experimental results and checkpoints must be hash-bound.
- **Fail-closed Proof Runners**: The testing and benchmarking harness will explicitly fail if evidence constraints are violated.
- Claims cannot be inflated. Honest negative results (e.g., ARC-AGI-2 0/120) are celebrated as scientifically valuable.

---

## Architecture Overview

RLF is built around phase-vector representations on the unit circle in ℂ^D. Key mechanisms include:
- **Sparse Mode Retrieval**
- **Recurrent Settling**
- **Local Learning**
- **Structural Growth**
- **Associative Memory**

The lineage spans RLF-0 through RLF-7, the Frontier research edition, and the Solstice multimodal system. The codebase avoids heavy abstractions in favor of explicit, traceable execution paths.

---

## Benchmarking

To run the benchmark suite on supported hardware (CPU, RTX 3090, H100, H200, RTX PRO 6000):

```bash
cmake --build --preset release --target run_benchmarks
./build/release/bin/run_benchmarks
```
Include benchmark results in your PR if you are modifying performance-critical paths (e.g., sparse retrieval components).

---

## Branch Naming & Commit Messages

### Branch Naming
- `feature/short-description`
- `fix/issue-number-short-description`
- `research/experiment-name`

### Commit Messages
We follow Conventional Commits:
- `feat: add associative memory scaling`
- `fix: resolve UB in phase-vector normalization`
- `docs: update evidence standards`
- `test: add cases for structural growth`

---

## PR Checklist

- [ ] Code compiles without warnings (`-Werror`).
- [ ] All tests pass (Core, Solstice, Frontier).
- [ ] ASan and UBSan report zero issues.
- [ ] No neural components or backpropagation paradigms have been introduced.
- [ ] Research claims in PR descriptions are strictly scoped and backed by fail-closed evidence.
- [ ] Code is formatted with `clang-format`.

---

## Code Review Process

1. **Automated CI**: GCC and Clang will build your PR in debug, release, ASan, and UBSan configurations.
2. **Review**: Maintainers will review the code for architectural consistency, C++23 idioms, and adherence to the non-neural paradigm.
3. **Merge**: Once approved and CI passes, the PR will be squash-merged into `main`.

## Research Contribution Guidelines

When introducing new milestones or experimental features:
- Claims must be strictly scoped to the exact empirical results.
- Do not extrapolate performance without direct evidence.
- Both extreme efficiencies (e.g., 10,000× target-supervision efficiency) and negative results must be reproducible via the provided fail-closed runners.
