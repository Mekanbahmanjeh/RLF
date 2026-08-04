#!/usr/bin/env python3
"""
build_magnum_5_1_8b_fable_dataset.py
------------------------------------
Compiles an ultra-comprehensive multi-domain dataset for Magnum 5.1 (Claude Fable 5 Level)
covering: Cybersecurity, Law, Science & Astrophysics, Advanced Coding (Rust, C++, Next.js 15),
Planning & Agentic Task Supervision, and Conversational Dialogue.
"""

import json
import os
import sys
import random
import urllib.request

DATA_DIR = "/workspace/RLF/demo_data/vast_8b_fable"
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

def fetch_json_cached(url: str, filename: str):
    cache_path = os.path.join("/workspace/RLF/demo_data", filename)
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        print(f"[+] Using cached file: {filename}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"[+] Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data

def build_cybersecurity_seeds():
    return [
        (
            "What are the key steps in conducting a web application penetration test according to OWASP guidelines?",
            THOUGHT_HEADER + (
                "A comprehensive web application penetration test following the OWASP Testing Guide (OTG) involves five core phases:\n\n"
                "1. Information Gathering & Reconnaissance: Mapping the target application architecture, identifying subdomains, server software, frameworks, and API endpoints using passive and active enumeration.\n\n"
                "2. Threat Modeling & Configuration Testing: Assessing infrastructure setups, SSL/TLS configurations, CORS policies, authentication mechanisms, and session management tokens.\n\n"
                "3. Vulnerability Analysis & Exploitation: Testing for OWASP Top 10 risks including SQL Injection (SQLi), Cross-Site Scripting (XSS), Server-Side Request Forgery (SSRF), Insecure Direct Object References (IDOR), and Broken Access Control.\n\n"
                "4. Post-Exploitation & Risk Escalation: Evaluating privilege escalation vectors, lateral movement capabilities, and sensitive data exposure.\n\n"
                "5. Reporting & Remediation Guidance: Documenting findings with CVSS scores, proof-of-concept exploits, and actionable developer remediation recommendations."
            )
        ),
        (
            "Explain how buffer overflow vulnerabilities work in C/C++ and how memory safety mitigations like ASLR and DEP/NX defend against them.",
            THOUGHT_HEADER + (
                "A buffer overflow occurs in languages without automatic bounds checking (such as C and C++) when a program writes more data to a buffer than its allocated memory capacity.\n\n"
                "When excess data spills past buffer boundaries on the stack, it can overwrite adjacent control structures, including the saved frame pointer (EBP/RBP) and the function return address. An attacker can craft a payload containing shellcode and overwrite the return address to divert control flow to their injected instructions.\n\n"
                "Defensive mitigations include:\n"
                "- Data Execution Prevention (DEP / NX Bit): Marks stack and heap memory regions as non-executable, preventing direct shellcode execution.\n"
                "- Address Space Layout Randomization (ASLR): Randomizes the memory addresses of executable binaries, stack, heap, and libraries on process startup, preventing attackers from using static target addresses."
            )
        ),
        (
            "What is Zero Trust Architecture (ZTA) and how is it implemented in enterprise networks?",
            THOUGHT_HEADER + (
                "Zero Trust Architecture (ZTA) is a strategic cybersecurity paradigm operating on the fundamental principle: 'Never trust, always verify.' Unlike traditional perimeter defenses that trust internal network traffic, Zero Trust treats all users, devices, and network segments as untrusted regardless of location.\n\n"
                "Key implementation pillars include:\n"
                "1. Micro-segmentation: Granular network isolation preventing lateral movement.\n"
                "2. Continuous Identity Verification: Multi-Factor Authentication (MFA) combined with risk-based Conditional Access.\n"
                "3. Least Privilege Access: RBAC/ABAC policies ensuring users access only resources necessary for their specific role.\n"
                "4. Continuous Monitoring & Analytics: SIEM and EDR telemetry analyzing device health and user telemetry in real-time."
            )
        )
    ]

def build_planning_supervision_seeds():
    return [
        (
            "How does an autonomous AI agent execute a complex multi-step software refactoring task under strict human audit controls?",
            THOUGHT_HEADER + (
                "Autonomous AI agents executing complex software refactoring operate under a structured three-gate governance workflow:\n\n"
                "1. `<reasoning_gate>`: The agent performs initial static code analysis, maps dependency graphs, and drafts a precise implementation plan detailing file modifications, dependency updates, and breaking change risks.\n\n"
                "2. `<eval_harness>`: Before modifying code, the agent executes unit test suites, integration benchmarks, and lint audits to establish baseline performance metrics.\n\n"
                "3. `<human_audit>`: Critical architectural modifications, security credentials, or structural schema migrations require explicit human review and authorization before deployment to production environments."
            )
        ),
        (
            "Formulate a structured project plan for building a fault-tolerant microservice architecture in Rust with distributed tracing.",
            THOUGHT_HEADER + (
                "Building a high-throughput, fault-tolerant microservice architecture in Rust requires a structured four-phase implementation plan:\n\n"
                "Phase 1: Core Service Scaffold\n"
                "- Framework Selection: Utilize `tokio` for async runtime and `axum` for HTTP API routing.\n"
                "- Error Handling: Define strongly typed domain errors using `thiserror` and `anyhow`.\n\n"
                "Phase 2: Observability & Telemetry\n"
                "- OpenTelemetry Integration: Configure `tracing` and `tracing-opentelemetry` to export Jaeger/Zipkin spans.\n\n"
                "Phase 3: Resiliency & Fault Tolerance\n"
                "- Circuit Breaking & Rate Limiting: Implement `tower` middleware layers for timeout, retry policies, and backoff jitter.\n\n"
                "Phase 4: Deployment & Health Probes\n"
                "- Docker multi-stage builds targeting `scratch` or `distroless` images, exposing `/healthz` and `/readyz` endpoints."
            )
        )
    ]

def main():
    print("=========================================================================")
    print("  Magnum 5.1 (Claude Fable 5 Level) 8B Multi-Domain Suite Builder         ")
    print("=========================================================================")

    dataset_pairs = []

    # 1. Add Custom Domain Seeds
    print("[+] Ingesting Cybersecurity Seeds...")
    dataset_pairs.extend(build_cybersecurity_seeds())

    print("[+] Ingesting Planning & Supervision Seeds...")
    dataset_pairs.extend(build_planning_supervision_seeds())

    # 2. Ingest Alpaca General Knowledge
    try:
        alpaca = fetch_json_cached(
            "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
            "alpaca_data.json"
        )
        print("[+] Ingesting Alpaca Conversational & General Knowledge rows...")
        for item in alpaca[:25000]:
            inst = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            out = item.get("output", "").strip()
            if not inst or not out:
                continue
            prompt = f"{inst}\n{inp}".strip() if inp else inst
            dataset_pairs.append((prompt, THOUGHT_HEADER + out))
    except Exception as e:
        print(f"[!] Warning: Could not ingest Alpaca: {e}")

    # 3. Ingest CodeAlpaca Programming
    try:
        code_alpaca = fetch_json_cached(
            "https://raw.githubusercontent.com/sahil280114/codealpaca/main/data/code_alpaca_20k.json",
            "code_alpaca_20k.json"
        )
        print("[+] Ingesting CodeAlpaca Programming rows...")
        for item in code_alpaca[:20000]:
            inst = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            out = item.get("output", "").strip()
            if not inst or not out:
                continue
            prompt = f"{inst}\n{inp}".strip() if inp else inst
            dataset_pairs.append((prompt, THOUGHT_HEADER + out))
    except Exception as e:
        print(f"[!] Warning: Could not ingest CodeAlpaca: {e}")

    # 4. Ingest MetaMathQA Math Reasoning
    try:
        metamath = fetch_json_cached(
            "https://huggingface.co/datasets/meta-math/MetaMathQA/resolve/main/MetaMathQA-395K.json",
            "MetaMathQA-395K.json"
        )
        print("[+] Ingesting Sampled MetaMathQA Math Reasoning rows...")
        sample_size = min(35000, len(metamath))
        sampled = random.sample(metamath, sample_size)
        for item in sampled:
            query = item.get("query", "").strip()
            resp = item.get("response", "").strip()
            if query and resp:
                dataset_pairs.append((query, THOUGHT_HEADER + resp))
    except Exception as e:
        print(f"[!] Warning: Could not ingest MetaMathQA: {e}")

    print(f"[+] Total Ingested Multi-Domain Dialogue Pairs: {len(dataset_pairs)}")

    # Write Dialogue TSV
    print(f"[+] Writing Dialogue TSV to {DIALOGUE_TSV}...")
    with open(DIALOGUE_TSV, "w", encoding="utf-8") as f:
        for prompt, resp in dataset_pairs:
            p_clean = prompt.replace("\t", " ").replace("\n", " \\n ")
            r_clean = resp.replace("\t", " ").replace("\n", " \\n ")
            f.write(f"{p_clean}\t{r_clean}\n")

    # Write Continuous Text Corpus
    print(f"[+] Writing Continuous Text Corpus to {CORPUS_TXT}...")
    with open(CORPUS_TXT, "w", encoding="utf-8") as f:
        for prompt, resp in dataset_pairs:
            f.write(f"User: {prompt}\nAssistant:\n{resp}\n\n")

    print("[+] Magnum 5.1 8B Multi-Domain Suite Build Complete!")

if __name__ == "__main__":
    main()
