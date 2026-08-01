#!/usr/bin/env python3
"""
Hugging Face CoT Dataset Fetcher for Vast.ai 24GB GPU
======================================================
Uses the HF `datasets` library (primary) or direct parquet download (fallback)
to fetch GSM8K math reasoning and Alpaca instruction data.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

def load_dotenv():
    """Load HF_TOKEN from .env file in workspace."""
    for env_path in [Path(".env"), Path("../.env"), Path("../../.env")]:
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip().strip("'\"")

def ensure_datasets_installed():
    """Install HF datasets library if not available."""
    try:
        import datasets
        print(f"[+] HF datasets library version: {datasets.__version__}")
        return True
    except ImportError:
        print("[+] Installing HF datasets library...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "-q"])
        return True

def load_gsm8k(cache_dir: Path, max_samples: int):
    """Load GSM8K via HF datasets library."""
    try:
        from datasets import load_dataset
        print("[+] Loading GSM8K from Hugging Face...")
        ds = load_dataset("openai/gsm8k", "main", split="train", trust_remote_code=True)
        print(f"[+] GSM8K loaded: {len(ds)} rows")
        return ds
    except Exception as e:
        print(f"[-] HF datasets load failed for GSM8K: {e}")
    
    # Fallback: try alternate dataset IDs
    for dataset_id in ["gsm8k", "openai/gsm8k"]:
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_id, "main", split="train", trust_remote_code=True)
            print(f"[+] GSM8K loaded via {dataset_id}: {len(ds)} rows")
            return ds
        except Exception:
            continue
    
    return None

def load_alpaca(cache_dir: Path, max_samples: int):
    """Load Alpaca via HF datasets library."""
    try:
        from datasets import load_dataset
        print("[+] Loading Alpaca from Hugging Face...")
        ds = load_dataset("tatsu-lab/alpaca", split="train", trust_remote_code=True)
        print(f"[+] Alpaca loaded: {len(ds)} rows")
        return ds
    except Exception as e:
        print(f"[-] HF datasets load failed for Alpaca: {e}")
    
    # Fallback: try yahma/alpaca-cleaned
    try:
        from datasets import load_dataset
        ds = load_dataset("yahma/alpaca-cleaned", split="train", trust_remote_code=True)
        print(f"[+] Alpaca-Cleaned loaded: {len(ds)} rows")
        return ds
    except Exception as e:
        print(f"[-] Fallback Alpaca load also failed: {e}")
    
    return None

def load_openorca(cache_dir: Path, max_samples: int):
    """Load OpenOrca subset via HF datasets library."""
    try:
        from datasets import load_dataset
        print("[+] Loading OpenOrca from Hugging Face (streaming)...")
        ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True, trust_remote_code=True)
        rows = []
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            rows.append(row)
            if (i + 1) % 50000 == 0:
                print(f"    ... streamed {i+1} OpenOrca rows")
        print(f"[+] OpenOrca loaded: {len(rows)} rows")
        return rows
    except Exception as e:
        print(f"[-] OpenOrca load failed (non-critical): {e}")
        return None

def process_gsm8k_dataset(ds, instructions: list, corpus: list, max_samples: int):
    """Process GSM8K HF dataset object into CoT rows."""
    print("[+] Processing GSM8K math reasoning dataset...")
    count = 0
    for row in ds:
        if count >= max_samples:
            break
        question = str(row.get("question", "")).replace("\t", " ").replace("\r", "").replace("\n", " ").strip()
        raw_answer = str(row.get("answer", ""))

        if "####" in raw_answer:
            parts = raw_answer.split("####")
            reasoning = parts[0].strip().replace("\t", " ").replace("\r", "").replace("\n", " ")
            final_val = parts[1].strip()
            rationale = f"<think> {reasoning} </think>"
            response = f"<think> {reasoning} </think> The answer is {final_val}."
        else:
            clean = raw_answer.strip().replace("\t", " ").replace("\r", "").replace("\n", " ")
            rationale = f"<think> {clean} </think>"
            response = rationale

        if question and response:
            count += 1
            instructions.append(f"gsm8k_{count:07d}\tarithmetic\t{question}\t{rationale}\t{response}\t1.0")
            corpus.append(f"{question} {response}")
    print(f"[+] Formatted {count} GSM8K CoT math records.")

def process_alpaca_dataset(ds, instructions: list, corpus: list, max_samples: int):
    """Process Alpaca HF dataset object into instruction rows."""
    print("[+] Processing Alpaca instruction dataset...")
    count = 0
    for row in ds:
        if count >= max_samples:
            break
        instruction = str(row.get("instruction", "")).strip().replace("\t", " ").replace("\r", "").replace("\n", " ")
        input_text = str(row.get("input", "")).strip().replace("\t", " ").replace("\r", "").replace("\n", " ")
        output_text = str(row.get("output", "")).strip().replace("\t", " ").replace("\r", "").replace("\n", " ")

        prompt = f"{instruction} {input_text}".strip()
        if not prompt or not output_text:
            continue

        count += 1
        rationale = "<think> Step 1: Analyze prompt. Step 2: Formulate response. </think>"
        response = f"{rationale} {output_text}"
        instructions.append(f"alpaca_{count:07d}\tgeneral\t{prompt}\t{rationale}\t{response}\t1.0")
        corpus.append(f"{prompt} {output_text}")
    print(f"[+] Formatted {count} Alpaca instruction records.")

def process_openorca_rows(rows, instructions: list, corpus: list, max_samples: int):
    """Process OpenOrca rows into instruction format."""
    print("[+] Processing OpenOrca reasoning dataset...")
    count = 0
    for row in rows:
        if count >= max_samples:
            break
        question = str(row.get("question", "")).strip().replace("\t", " ").replace("\r", "").replace("\n", " ")
        response_text = str(row.get("response", "")).strip().replace("\t", " ").replace("\r", "").replace("\n", " ")

        if not question or not response_text:
            continue

        count += 1
        rationale = "<think> Step 1: Reason about the question. Step 2: Provide accurate answer. </think>"
        response = f"{rationale} {response_text}"
        instructions.append(f"orca_{count:07d}\treasoning\t{question}\t{rationale}\t{response}\t1.0")
        corpus.append(f"{question} {response_text}")
    print(f"[+] Formatted {count} OpenOrca reasoning records.")

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="RLF Vast.ai CoT Dataset Builder")
    parser.add_argument("--output-dir", type=str, default="demo_data/vast_1b", help="Target dataset directory")
    parser.add_argument("--max-samples", type=int, default=2500000, help="Max samples per dataset")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    print("=========================================================================")
    print("  RLF Vast.ai Frontier 24G — CoT Dataset Builder (HF datasets library)  ")
    print("=========================================================================")

    ensure_datasets_installed()

    instructions = ["# task\tdomain\tprompt\trationale\tresponse\tquality"]
    corpus = []

    # 1. GSM8K
    gsm8k_ds = load_gsm8k(cache_dir, args.max_samples)
    if gsm8k_ds is not None:
        process_gsm8k_dataset(gsm8k_ds, instructions, corpus, args.max_samples)

    # 2. Alpaca
    alpaca_ds = load_alpaca(cache_dir, args.max_samples)
    if alpaca_ds is not None:
        process_alpaca_dataset(alpaca_ds, instructions, corpus, args.max_samples)

    # 3. OpenOrca (streaming, up to 500K rows)
    orca_rows = load_openorca(cache_dir, min(args.max_samples, 500000))
    if orca_rows is not None:
        process_openorca_rows(orca_rows, instructions, corpus, args.max_samples)

    total_rows = len(instructions) - 1
    if total_rows == 0:
        print("[-] FATAL: No data was downloaded. Check network connectivity.")
        sys.exit(1)

    instructions_file = out_dir / "instructions.tsv"
    corpus_file = out_dir / "corpus.txt"

    with open(instructions_file, "w", encoding="utf-8") as f:
        f.write("\n".join(instructions) + "\n")

    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write("\n".join(corpus) + "\n")

    print(f"\n=======================================================")
    print(f"[+] Successfully built dataset manifest:")
    print(f"    total_instruction_rows = {total_rows}")
    print(f"    total_corpus_lines     = {len(corpus)}")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
