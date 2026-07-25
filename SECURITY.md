# Security Policy

The Resonant Learning Fabric (RLF) project takes security seriously. Due to the high-performance and exploratory nature of this C++23 research prototype, we adhere to strict standards to ensure the integrity of the system, models, and evidence runner.

## Supported Versions

Security updates are provided for the latest stable release and the current active research branch. 

| Version | Supported          |
| ------- | ------------------ |
| 11.0.x  | :white_check_mark: |
| 10.x.x  | :x:                |
| < 10.0  | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability within RLF, please do **NOT** open a public issue. Instead, report it privately to ensure the safety of the project and its users.

**Contact:** Send an email to [mekanbahmanjeh@gmail.com](mailto:mekanbahmanjeh@gmail.com).

Please include the following details in your report:
- A description of the vulnerability.
- Steps to reproduce the issue.
- Potential impact and any known exploit scenarios.
- The environment (OS, compiler, hardware) where the issue was observed.

## Response Timeline

- **Acknowledgment**: We aim to acknowledge receipt of your vulnerability report within **48 hours**.
- **Assessment**: A preliminary assessment of the vulnerability will be completed within **7 days**.
- **Resolution**: We will provide a timeline for a patch or mitigation strategy once the assessment is complete.

## RLF-Specific Security Considerations

Given the unique architecture of RLF—leveraging sparse mode retrieval, recurrent settling, and phase-vector representations—the following security constraints are strictly enforced:

1. **Checkpoint Integrity**: 
   - All experimental checkpoints and associative memory dumps must be cryptographically verified.
   - We mandate **SHA-256 verification** for all state loads to prevent malicious code execution or corrupted research evidence.
2. **Sandbox Enforcement**: 
   - The structural growth mechanisms and local learning routines operate within a strictly constrained memory sandbox. 
   - Exploits targeting buffer overflows in ℂ^D unit circle representations are mitigated through rigorous bounds checking and ASan/UBSan enforcement in CI.
3. **Fail-Closed Evidence**:
   - The fail-closed evidence system ensures that tampered artifacts result in an immediate hard abort (`std::abort()`). This prevents false research claims and secures the artifact pipeline.

## Disclosure Policy

We follow a coordinated disclosure policy. We ask that you do not publish details of the vulnerability until we have had adequate time to assess and release a fix or mitigation. Once the fix is released, we will credit you (if desired) for the discovery in our release notes and security advisories.
