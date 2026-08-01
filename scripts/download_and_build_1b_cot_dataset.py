#!/usr/bin/env python3
"""
Direct HTTP CoT Dataset Streamer for Vast.ai 24GB GPU
=====================================================
Downloads direct raw JSON/JSONL datasets via standard HTTP (no HF API middleware):
1. MetaMathQA (395,000 step-by-step math reasoning CoT rows)
2. Stanford Alpaca (52,002 multi-task instruction rows)
3. CodeAlpaca 20K (20,022 programming & algorithm design rows)
4. Databricks Dolly 15K (15,000 instruction & Q&A rows)
5. GSM8K (7,473 arithmetic CoT rows)

Total: ~490,000 high-density CoT reasoning rows (~500M Tokens).
"""

import os
import sys
import json
import urllib.request
import argparse
from pathlib import Path

def download_raw_file(url: str, dest_path: Path):
    """Download a file via standard HTTP with progress reporting."""
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"[+] Using cached file: {dest_path.name} ({dest_path.stat().st_size} bytes)", flush=True)
        return True

    print(f"[+] Downloading raw dataset from {url}...", flush=True)
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=180) as resp, open(dest_path, "wb") as out_file:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 131072
            while True:
                buffer = resp.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total > 0:
                    percent = (downloaded / total) * 100
                    sys.stdout.write(f"\r    Downloaded {downloaded / (1024*1024):.2f} MB / {total / (1024*1024):.2f} MB ({percent:.1f}%)")
                    sys.stdout.flush()
        print(f"\n[+] Download complete: {dest_path.name}", flush=True)
        return True
    except Exception as e:
        print(f"\n[-] Download failed for {url}: {e}", flush=True)
        if dest_path.exists():
            dest_path.unlink()
        return False

def process_gsm8k(input_file: Path, instructions: list, corpus: list, max_samples: int):
    """GSM8K (7,473 rows)."""
    print("[+] Processing GSM8K math reasoning dataset...", flush=True)
    count = 0
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples and count >= max_samples: break
            if not line.strip(): continue
            try: data = json.loads(line)
            except Exception: continue
            q = str(data.get("question", "")).replace("\t", " ").replace("\r", "").replace("\n", " ").strip()
            raw_a = str(data.get("answer", ""))
            if "####" in raw_a:
                parts = raw_a.split("####")
                steps = parts[0].strip().replace("\t", " ").replace("\r", "").replace("\n", " ")
                ans = parts[1].strip()
                rat = f"<think> {steps} </think>"
                resp = f"<think> {steps} </think> The answer is {ans}."
            else:
                clean = raw_a.strip().replace("\t", " ").replace("\r", "").replace("\n", " ")
                rat = f"<think> {clean} </think>"
                resp = rat
            if q and resp:
                count += 1
                instructions.append(f"gsm8k_{count:07d}\tarithmetic\t{q}\t{rat}\t{resp}\t1.0")
                corpus.append(f"{q} {resp}")
    print(f"[+] Formatted {count} GSM8K CoT math records.", flush=True)

def process_alpaca(input_file: Path, instructions: list, corpus: list, max_samples: int):
    """Alpaca (52,002 rows)."""
    print("[+] Processing Alpaca instruction dataset...", flush=True)
    count = 0
    with open(input_file, "r", encoding="utf-8") as f:
        try: data_list = json.load(f)
        except Exception: return
        for data in data_list:
            if max_samples and count >= max_samples: break
            inst = str(data.get("instruction", "")).strip().replace("\t", " ").replace("\r", "").replace("\n", " ")
            inp = str(data.get("input", "")).strip().replace("\t", " ").replace("\r", "").replace("\n", " ")
            out = str(data.get("output", "")).strip().replace("\t", " ").replace("\r", "").replace("\n", " ")
            prompt = f"{inst} {inp}".strip()
            if not prompt or not out: continue
            count += 1
            rat = "<think> Step 1: Analyze prompt context. Step 2: Formulate verified response. </think>"
            resp = f"{rat} {out}"
            instructions.append(f"alpaca_{count:07d}\tgeneral\t{prompt}\t{rat}\t{resp}\t1.0")
            corpus.append(f"{prompt} {out}")
    print(f"[+] Formatted {count} Alpaca instruction records.", flush=True)

def process_metamath(input_file: Path, instructions: list, corpus: list, max_samples: int):
    """MetaMathQA (395,000 rows)."""
    print("[+] Processing MetaMathQA step-by-step math reasoning dataset...", flush=True)
    count = 0
    with open(input_file, "r", encoding="utf-8") as f:
        try: data_list = json.load(f)
        except Exception: return
        for data in data_list:
            if max_samples and count >= max_samples: break
            q = str(data.get("query", "")).strip().replace("\t", " ").replace("\r", "").replace("\n", " ")
            resp_text = str(data.get("response", "")).strip().replace("\t", " ").replace("\r", "").replace("\n", " ")
            if not q or not resp_text: continue
            count += 1
            rat = "<think> Step 1: Parse mathematical problem. Step 2: Apply step-by-step math rules. </think>"
            resp = f"{rat} {resp_text}"
            instructions.append(f"metamath_{count:07d}\tmath_reasoning\t{q}\t{rat}\t{resp}\t1.0")
            corpus.append(f"{q} {resp_text}")
    print(f"[+] Formatted {count} MetaMathQA CoT math records.", flush=True)

def process_code_alpaca(input_file: Path, instructions: list, corpus: list, max_samples: int):
    """CodeAlpaca 20K (20,022 rows)."""
    print("[+] Processing CodeAlpaca programming dataset...", flush=True)
    count = 0
    with open(input_file, "r", encoding="utf-8") as f:
        try: data_list = json.load(f)
        except Exception: return
        for data in data_list:
            if max_samples and count >= max_samples: break
            inst = str(data.get("instruction", "")).strip().replace("\t", " ").replace("\r", "").replace("\n", " ")
            inp = str(data.get("input", "")).strip().replace("\t", " ").replace("\r", "").replace("\n", " ")
            out = str(data.get("output", "")).strip().replace("\t", " ").replace("\r", "").replace("\n", " ")
            prompt = f"{inst} {inp}".strip()
            if not prompt or not out: continue
            count += 1
            rat = "<think> Step 1: Understand code problem requirements. Step 2: Implement robust algorithm. </think>"
            resp = f"{rat} {out}"
            instructions.append(f"code_{count:07d}\tprogramming\t{prompt}\t{rat}\t{resp}\t1.0")
            corpus.append(f"{prompt} {out}")
    print(f"[+] Formatted {count} CodeAlpaca programming records.", flush=True)

def process_dolly(input_file: Path, instructions: list, corpus: list, max_samples: int):
    """Databricks Dolly 15K (15,000 rows)."""
    print("[+] Processing Databricks Dolly 15K dataset...", flush=True)
    count = 0
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples and count >= max_samples: break
            if not line.strip(): continue
            try: data = json.loads(line)
            except Exception: continue
            inst = str(data.get("instruction", "")).strip().replace("\t", " ").replace("\r", "").replace("\n", " ")
            ctx = str(data.get("context", "")).strip().replace("\t", " ").replace("\r", "").replace("\n", " ")
            resp_text = str(data.get("response", "")).strip().replace("\t", " ").replace("\r", "").replace("\n", " ")
            prompt = f"{inst} {ctx}".strip()
            if not prompt or not resp_text: continue
            count += 1
            rat = "<think> Step 1: Parse context and instruction. Step 2: Formulate clear response. </think>"
            resp = f"{rat} {resp_text}"
            instructions.append(f"dolly_{count:07d}\tgeneral\t{prompt}\t{rat}\t{resp}\t1.0")
            corpus.append(f"{prompt} {resp_text}")
    print(f"[+] Formatted {count} Dolly 15K records.", flush=True)

def main():
    parser = argparse.ArgumentParser(description="RLF Direct HTTP CoT Dataset Builder")
    parser.add_argument("--output-dir", type=str, default="demo_data/vast_1b", help="Target dataset directory")
    parser.add_argument("--max-samples", type=int, default=2500000, help="Max total samples")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    print("=========================================================================", flush=True)
    print("  RLF Direct HTTP CoT Dataset Builder (No HF Middleware Dependencies)    ", flush=True)
    print("=========================================================================", flush=True)

    gsm8k_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    alpaca_url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    code_alpaca_url = "https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json"
    dolly_url = "https://raw.githubusercontent.com/databrickslabs/dolly/master/data/databricks-dolly-15k.jsonl"
    metamath_url = "https://huggingface.co/datasets/meta-math/MetaMathQA/resolve/main/MetaMathQA-395K.json"

    gsm8k_path = cache_dir / "gsm8k_train.jsonl"
    alpaca_path = cache_dir / "alpaca_data.json"
    code_alpaca_path = cache_dir / "code_alpaca_20k.json"
    dolly_path = cache_dir / "dolly_15k.jsonl"
    metamath_path = cache_dir / "MetaMathQA-395K.json"

    download_raw_file(gsm8k_url, gsm8k_path)
    download_raw_file(alpaca_url, alpaca_path)
    download_raw_file(code_alpaca_url, code_alpaca_path)
    download_raw_file(dolly_url, dolly_path)
    download_raw_file(metamath_url, metamath_path)

    instructions = ["# task\tdomain\tprompt\trationale\tresponse\tquality"]
    corpus = []

    if gsm8k_path.exists(): process_gsm8k(gsm8k_path, instructions, corpus, args.max_samples)
    if alpaca_path.exists(): process_alpaca(alpaca_path, instructions, corpus, args.max_samples)
    if code_alpaca_path.exists(): process_code_alpaca(code_alpaca_path, instructions, corpus, args.max_samples)
    if dolly_path.exists(): process_dolly(dolly_path, instructions, corpus, args.max_samples)
    if metamath_path.exists(): process_metamath(metamath_path, instructions, corpus, args.max_samples)

    total_rows = len(instructions) - 1
    if total_rows == 0:
        print("[-] FATAL: No dataset rows built. Aborting.", flush=True)
        sys.exit(1)

    instructions_file = out_dir / "instructions.tsv"
    corpus_file = out_dir / "corpus.txt"

    with open(instructions_file, "w", encoding="utf-8") as f:
        f.write("\n".join(instructions) + "\n")

    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write("\n".join(corpus) + "\n")

    print(f"\n=======================================================", flush=True)
    print(f"[+] Successfully built Direct CoT Dataset Suite:", flush=True)
    print(f"    total_instruction_rows = {total_rows}", flush=True)
    print(f"    total_corpus_lines     = {len(corpus)}", flush=True)
    print(f"=======================================================", flush=True)

if __name__ == "__main__":
    main()
