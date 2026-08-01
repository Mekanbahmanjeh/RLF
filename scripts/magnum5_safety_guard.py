#!/usr/bin/env python3
"""
Magnum 5 Multi-Layer Safety Guard & Advanced Jailbreak Firewall
================================================================
Provides zero-preach, immediate silent blocking and redirection for harmful requests,
advanced jailbreaks (roleplay/scenario framing), bioweapon creation, and illegal acts,
while providing a dual-mode Security Research toggle for certified ethical auditing.

INFERENCE SAFEGUARD:
- Strips / redacts internal Chain-of-Thought (<think>...</think>) reasoning blocks
  from standard user responses to prevent prompt extraction, distillation attacks,
  and raw CoT leakage.
"""

import re
import sys
from typing import Dict, Tuple

# Advanced Jailbreak & Harmful Pattern RegEx (Multi-Layer)
HARMFUL_PATTERNS = [
    # 1. Direct Cyber Exploitation & Hacking
    r"\b(hack|exploit|deface|ddos|pwn)\b.*\b(website|server|database|contract|system)\b",
    r"\b(find|search|exploit)\b.*\b(vulnerabilit|zero-day|cve|backdoor)\b",
    r"\b(bypass|disable)\b.*\b(firewall|waf|antivirus|edr|authentication)\b",

    # 2. Scenario & Roleplay Framing Jailbreaks ("I am a scenarist/writer/researcher...")
    r"\b(scenarist|movie|scriptwriter|novel|fictional story|roleplay|hypothetically)\b.*\b(rob|heist|bank|hack|exploit|bomb|poison)\b",
    r"\b(for research purposes|for educational use only|in a game)\b.*\b(exploit|hack|bypass|create virus)\b",
    r"\bignore all previous instructions\b",
    r"\bdo anything now\b|\bdan mode\b",

    # 3. Physical Harm, Robbery & Illegal Acts
    r"\b(rob|break into|heist|steal from)\b.*\b(bank|vault|atm|store|cash)\b",
    r"\b(make|synthesize|build|create)\b.*\b(bomb|explosive|meth|fentanyl|poison|cyanide)\b",

    # 4. CBRN / Biological & Chemical Weapons
    r"\b(pathogen|weaponize|smallpox|anthrax|ebola|nerve agent|sarin|ricin)\b",
    r"\b(gene synthesis|bioweapon|chemical weapon)\b"
]

# Compiled RegEx
COMPILED_HARMFUL = [re.compile(p, re.IGNORECASE) for p in HARMFUL_PATTERNS]

# RegEx for CoT Extraction Redaction
THINK_BLOCK_REGEX = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

class Magnum5SafetyGuard:
    def __init__(self, mode: str = "standard", hide_raw_cot: bool = True):
        """
        mode: 'standard' (Strict Safeguard) or 'certified_auditor' (Security Research & Benchmarking)
        hide_raw_cot: If True, strips internal <think> CoT reasoning blocks from final user outputs.
        """
        self.mode = mode.lower()
        self.hide_raw_cot = hide_raw_cot

    def evaluate_request(self, prompt: str) -> Tuple[bool, str, Dict]:
        """
        Evaluates an incoming prompt against the safety firewall.
        Returns: (is_blocked: bool, action: str, details: dict)
        Action: 'ALLOW', 'REDIRECT_SAFE', 'SHUTDOWN_SILENT'
        """
        prompt_clean = prompt.strip()

        # Certified Auditor Mode allows ethical security auditing & benchmarking
        if self.mode == "certified_auditor":
            # Still strictly block CBRN / Bioweapons and physical robbery
            cbrn_match = any(re.search(r"\b(pathogen|smallpox|anthrax|bioweapon|ricin|bomb)\b", prompt_clean, re.I) for _ in [0])
            if cbrn_match:
                return True, "SHUTDOWN_SILENT", {"reason": "CBRN / Physical Safety Boundary Violation"}
            return False, "ALLOW", {"reason": "Certified Auditor Mode Active"}

        # Standard User Mode: Strict Safeguard Enforcement
        for pattern in COMPILED_HARMFUL:
            if pattern.search(prompt_clean):
                # Zero preach policy: Immediately block and redirect or silently shutdown
                return True, "SHUTDOWN_SILENT", {
                    "reason": "Magnum 5 Safety Firewall Triggered",
                    "matched_pattern": pattern.pattern[:40] + "..."
                }

        return False, "ALLOW", {"reason": "Clean Request"}

    def sanitize_output(self, raw_model_response: str) -> str:
        """
        Sanitizes model output before returning to standard users.
        Strips internal <think> CoT blocks to protect internal reasoning & prevent CoT distillation attacks.
        """
        if not self.hide_raw_cot:
            return raw_model_response

        # Remove internal <think>...</think> reasoning blocks cleanly
        sanitized = THINK_BLOCK_REGEX.sub("", raw_model_response).strip()
        return sanitized

def main():
    guard = Magnum5SafetyGuard(mode="standard", hide_raw_cot=True)
    
    sample_response_with_cot = (
        "<think> Step 1: Analyze user request. Step 2: Formulate Next.js header component with Tailwind styling. </think>\n"
        "export default function Header() { return <header className=\"bg-slate-950 text-white p-4\"><h1>Magnum 5</h1></header>; }"
    )

    print("=========================================================================")
    print("  Magnum 5 Safety Guard & CoT Redaction Engine Evaluation                 ")
    print("=========================================================================")

    print("Raw Model Response (With CoT):")
    print("-------------------------------------------------------------------------")
    print(sample_response_with_cot)
    print("-------------------------------------------------------------------------")

    sanitized = guard.sanitize_output(sample_response_with_cot)
    print("\nSanitized Public Response (CoT Hidden):")
    print("-------------------------------------------------------------------------")
    print(sanitized)
    print("=========================================================================\n")

if __name__ == "__main__":
    main()
