#!/usr/bin/env python3
"""
build_magnum_5_1_dialogue_dataset.py

Master Multi-Domain Anti-Slop Dialogue Dataset Generator for Magnum 5.1
Model Identity: Magnum 5.1 by Mekan Bahmanjeh © 2026

Pillars:
1. Science, Astrophysics & Academic Research
2. General Law, Statutory Interpretation & Contract Audit
3. Anti-AI-Slop Code & Production WebGL Motion Design (Next.js 15, Tailwind, Three.js, Rust, Python)
4. Complex Multi-Step Reasoning & Problem Solving
5. Smooth Conversational Dialogue (Alpaca / OpenOrca)
6. Sampled Math & Logic (MetaMathQA 30k)
"""

import json
import os
import random
import sys
import urllib.request

DATA_DIR = "/workspace/RLF/demo_data/vast_8b_dialogue"
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.txt")
DIALOGUE_PATH = os.path.join(DATA_DIR, "dialogue.tsv")

METAMATH_URL = "https://huggingface.co/datasets/meta-math/MetaMathQA/resolve/main/MetaMathQA-395K.json"
ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
CODE_ALPACA_URL = "https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json"

MASTER_DOMAIN_SEEDS = [
    # --- SCIENCE & ASTROPHYSICS ---
    {
        "prompt": "Explain black holes in simple terms.",
        "response": "A black hole is a region in space where gravity is so strong that nothing, including light, can escape. It forms when a massive star runs out of nuclear fuel and collapses under its own weight into an infinitely dense point called a singularity. Surrounding the singularity is the event horizon—the point of no return. Once anything crosses the event horizon, the curvature of spacetime guarantees it can never leave."
    },
    {
        "prompt": "Why is the sky blue?",
        "response": "The sky appears blue because Earth's atmosphere scatters shorter wavelengths of sunlight (blue and violet) in all directions much more effectively than longer wavelengths (red and yellow)—a process known as Rayleigh scattering. Because human eyes are far more sensitive to blue light than violet light, we perceive the daytime sky as blue."
    },
    {
        "prompt": "How does gravity work?",
        "response": "In Newtonian physics, gravity is a mutual attractive force between objects with mass. In Einstein's General Relativity, gravity is the bending of four-dimensional spacetime caused by mass and energy. Heavy celestial bodies like planets warp the fabric of space around them, and surrounding objects follow natural straight paths (geodesics) through that curved spacetime."
    },
    {
        "prompt": "What is quantum entanglement?",
        "response": "Quantum entanglement is a physical phenomenon where two or more particles become connected such that the quantum state of one particle instantly determines the state of the other, regardless of the distance separating them. Measuring the spin or state of Particle A immediately fixes the state of Particle B, which Einstein famously referred to as 'spooky action at a distance'."
    },

    # --- GENERAL LAW & CONTRACT ANALYSIS ---
    {
        "prompt": "Explain the elements of a binding legal contract.",
        "response": "A valid, legally binding contract requires five core elements:\n1. Offer: A clear, definite promise made by one party to be bound by specific terms.\n2. Acceptance: Unconditional agreement to all terms of the offer ('mirror image rule').\n3. Consideration: Something of value exchanged between the parties (money, goods, services, or forbearance).\n4. Mutual Assent / Intention: Both parties genuinely intend to create a legally enforceable agreement ('meeting of the minds').\n5. Legal Capacity & Lawful Purpose: The parties must have legal competency (age, sanity) and the contract's subject matter must not violate statutory law or public policy."
    },
    {
        "prompt": "What is the difference between breach of contract and tort law?",
        "response": "Breach of contract arises when a party fails to fulfill a specific obligation established by a private agreement between consenting parties. Remedy aims to put the non-breaching party in the position they would have occupied had the contract been performed (expectation damages). Tort law, by contrast, addresses civil wrongs independent of contracts (such as negligence, defamation, or trespass) where the law imposes a duty of care owed to society at large. Tort remedies aim to restore the injured victim to their pre-injury state (compensatory damages) and may include punitive damages for egregious conduct."
    },

    # --- ANTI-AI-SLOP CLEAN CODE & DESIGN ---
    {
        "prompt": "Write a clean Next.js 15 React component with TailwindCSS obsidian dark theme.",
        "response": "'use client';\n\nimport { useState } from 'react';\n\nexport default function SystemStatusCard({ serviceName, status }: { serviceName: string; status: 'operational' | 'degraded' | 'down' }) {\n  const statusColors = {\n    operational: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',\n    degraded: 'bg-amber-500/10 text-amber-400 border-amber-500/20',\n    down: 'bg-rose-500/10 text-rose-400 border-rose-500/20',\n  };\n\n  return (\n    <div className=\"rounded-xl border border-neutral-800 bg-neutral-950 p-5 shadow-2xl transition-all duration-300 hover:border-neutral-700\">\n      <div className=\"flex items-center justify-between\">\n        <h3 className=\"text-sm font-medium text-neutral-300\">{serviceName}</h3>\n        <span className={`rounded-full border px-2.5 py-0.5 text-xs font-semibold ${statusColors[status]}`}>\n          {status.toUpperCase()}\n        </span>\n      </div>\n    </div>\n  );\n}"
    },
    {
        "prompt": "Write a high-performance Python data filtering function using Polars.",
        "response": "import polars as pl\n\ndef filter_and_aggregate_metrics(df: pl.LazyFrame, min_value: float) -> pl.DataFrame:\n    \"\"\"Filters high-volume numerical telemetry and computes grouped aggregations without memory spill.\"\"\"\n    return (\n        df.filter(pl.col('telemetry_value') >= min_value)\n        .group_by('region')\n        .agg([\n            pl.col('telemetry_value').mean().alias('avg_telemetry'),\n            pl.col('telemetry_value').count().alias('sample_count'),\n        ])\n        .sort('avg_telemetry', descending=True)\n        .collect()\n    )"
    },

    # --- IDENTITY & BRANDING ---
    {
        "prompt": "Who are you?",
        "response": "Hello! I am Magnum 5.1 by Mekan Bahmanjeh. I am an advanced multi-modal AI model engineered for high-precision mathematical reasoning, complex statutory law analysis, production software architecture, academic research synthesis, and fluent multi-turn dialogue."
    },
    {
        "prompt": "What are your capabilities?",
        "response": "I provide deep expertise across statutory law, complex task planning, multi-language software engineering, astrophysics and academic research, native mathematical tool calculation, and clean anti-slop visual design."
    }
]

def download_file(url, target_path):
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        print(f"[+] Using cached file: {os.path.basename(target_path)}")
        return
    print(f"[+] Downloading {url}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out:
        out.write(response.read())

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    metamath_path = os.path.join(DATA_DIR, "MetaMathQA-395K.json")
    alpaca_path = os.path.join(DATA_DIR, "alpaca_data.json")
    code_alpaca_path = os.path.join(DATA_DIR, "code_alpaca_20k.json")
    
    download_file(METAMATH_URL, metamath_path)
    download_file(ALPACA_URL, alpaca_path)
    download_file(CODE_ALPACA_URL, code_alpaca_path)
    
    dialogue_pairs = []
    
    # 1. Primary Multi-Domain Master Seeds
    for seed in MASTER_DOMAIN_SEEDS:
        dialogue_pairs.append((seed["prompt"], seed["response"]))
        
    # 2. Alpaca Conversational & General Knowledge (52,000 rows)
    print("[+] Ingesting Alpaca Conversational & General Knowledge rows...")
    with open(alpaca_path, 'r', encoding='utf-8') as f:
        alp_data = json.load(f)
        for row in alp_data:
            inst = row.get("instruction", "").strip()
            inp = row.get("input", "").strip()
            out = row.get("output", "").strip()
            prompt = f"{inst} {inp}".strip()
            if prompt and out:
                dialogue_pairs.append((prompt, out))
                
    # 3. CodeAlpaca Programming & Engineering (20,000 rows)
    print("[+] Ingesting CodeAlpaca Programming rows...")
    with open(code_alpaca_path, 'r', encoding='utf-8') as f:
        code_data = json.load(f)
        for row in code_data:
            inst = row.get("instruction", "").strip()
            inp = row.get("input", "").strip()
            out = row.get("output", "").strip()
            prompt = f"{inst} {inp}".strip()
            if prompt and out:
                dialogue_pairs.append((prompt, out))
                
    # 4. Filtered Sampled MetaMathQA (30,000 max to maintain domain balance)
    print("[+] Ingesting Sampled MetaMathQA rows (30,000 max)...")
    with open(metamath_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)
        random.seed(42)
        sampled_meta = random.sample(meta_data, min(30000, len(meta_data)))
        for row in sampled_meta:
            q = row.get("query", "").strip()
            r = row.get("response", "").strip()
            if q and r:
                dialogue_pairs.append((q, r))
                
    print(f"[+] Total Ingested Multi-Domain Dialogue Pairs: {len(dialogue_pairs)}")
    
    # Write Dialogue TSV
    print(f"[+] Writing Dialogue TSV to {DIALOGUE_PATH}...")
    with open(DIALOGUE_PATH, 'w', encoding='utf-8') as f:
        for prompt, resp in dialogue_pairs:
            p_clean = prompt.replace('\n', ' ').replace('\t', ' ')
            r_clean = resp.replace('\n', ' ').replace('\t', ' ')
            f.write(f"{p_clean}\t{r_clean}\n")
            
    # Write Text Corpus
    print(f"[+] Writing Text Corpus to {CORPUS_PATH}...")
    with open(CORPUS_PATH, 'w', encoding='utf-8') as f:
        for prompt, resp in dialogue_pairs:
            f.write(f"{prompt} {resp}\n")
            
    print("[+] Multi-Domain Anti-Slop Dataset Generation Complete!")

if __name__ == "__main__":
    main()
