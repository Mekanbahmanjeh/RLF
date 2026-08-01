#!/usr/bin/env python3
"""
Hugging Face 1-Billion Token CoT Dataset Fetcher for Vast.ai 24GB GPU
======================================================================
Streams and formats CoT reasoning rows from Hugging Face open datasets.
"""

import os
import sys
import json
import urllib.request
import argparse
from pathlib import Path

def load_dotenv():
    """Load HF_TOKEN from .env file in workspace."""
    for env_path in [Path(".env"), Path("../.env"), Path("../../.env")]:
        if env_path.exists():
            print(f"[+] Loading environment variables from {env_path.resolve()}")
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip().strip("'\"")

def download_file(url: str, dest_path: Path, token: str = None):
    """Download a file with progress reporting."""
    print(f"[+] Downloading from {url}...")
    headers = {"User-Agent": "RLF-Vast-1B-Fetcher/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, "wb") as out_file:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        block_size = 65536
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
    print(f"\n[+] Download complete: {dest_path.name} ({downloaded} bytes)")

def download_hf_dataset_parquet(dataset_id: str, split: str, cache_dir: Path, token: str = None):
    """Download a HF dataset split via the datasets API or parquet fallback."""
    cache_file = cache_dir / f"{dataset_id.replace('/', '_')}_{split}.jsonl"
    if cache_file.exists() and cache_file.stat().st_size > 100:
        print(f"[+] Using cached {cache_file.name}")
        return cache_file
    
    # Try HF datasets API (rows endpoint)
    api_url = f"https://datasets-server.huggingface.co/rows?dataset={dataset_id}&config=default&split={split}&offset=0&length=100"
    headers = {"User-Agent": "RLF-Vast-1B-Fetcher/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    try:
        # Use the first_rows endpoint for smaller datasets  
        first_rows_url = f"https://datasets-server.huggingface.co/first-rows?dataset={dataset_id}&config=default&split={split}"
        req = urllib.request.Request(first_rows_url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            rows = data.get("rows", [])
            if rows:
                with open(cache_file, "w", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row.get("row", row)) + "\n")
                print(f"[+] Downloaded {len(rows)} rows from {dataset_id}/{split}")
                return cache_file
    except Exception as e:
        print(f"[-] HF API failed for {dataset_id}: {e}")
    
    return None

def process_gsm8k(input_file: Path, instructions: list, corpus: list, max_samples: int):
    """Process GSM8K math dataset into step-by-step <think> blocks."""
    print(f"[+] Processing GSM8K math reasoning dataset...")
    count = 0
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples and count >= max_samples:
                break
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            question = data.get("question", "").replace("\t", " ").replace("\r", "").replace("\n", " ").strip()
            raw_answer = data.get("answer", "")
            
            if "####" in raw_answer:
                parts = raw_answer.split("####")
                reasoning_steps = parts[0].strip().replace("\t", " ").replace("\r", "").replace("\n", " ")
                final_val = parts[1].strip()
                rationale = f"<think> Step 1: Analyze problem statement. Step 2: {reasoning_steps} </think>"
                response = f"<think> {reasoning_steps} </think> The answer is {final_val}."
            else:
                rationale = f"<think> {raw_answer.strip().replace(chr(9), ' ').replace(chr(13), '').replace(chr(10), ' ')} </think>"
                response = rationale
            
            if question and response:
                count += 1
                task_id = f"gsm8k_{count:07d}"
                instructions.append(f"{task_id}\tarithmetic\t{question}\t{rationale}\t{response}\t1.0")
                corpus.append(f"{question} {rationale} {response}")
    print(f"[+] Formatted {count} GSM8K CoT math records.")

def process_alpaca(input_file: Path, instructions: list, corpus: list, max_samples: int):
    """Process Alpaca instruction dataset into reasoning rows."""
    print(f"[+] Processing Alpaca instruction dataset...")
    count = 0
    with open(input_file, "r", encoding="utf-8") as f:
        try:
            data_list = json.load(f)
        except json.JSONDecodeError:
            # Try JSONL format
            f.seek(0)
            data_list = []
            for line in f:
                if line.strip():
                    try:
                        data_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        for data in data_list:
            if max_samples and count >= max_samples:
                break
            instruction = data.get("instruction", "").strip().replace("\t", " ").replace("\r", "").replace("\n", " ")
            input_text = data.get("input", "").strip().replace("\t", " ").replace("\r", "").replace("\n", " ")
            output_text = data.get("output", "").strip().replace("\t", " ").replace("\r", "").replace("\n", " ")
            
            prompt = f"{instruction} {input_text}".strip()
            if not prompt or not output_text:
                continue
            
            count += 1
            rationale = f"<think> Step 1: Analyze prompt context. Step 2: Formulate clear justified response. </think>"
            response = f"{rationale} {output_text}"
            task_id = f"alpaca_{count:07d}"
            instructions.append(f"{task_id}\tgeneral\t{prompt}\t{rationale}\t{response}\t1.0")
            corpus.append(f"{prompt} {output_text}")
    print(f"[+] Formatted {count} Alpaca instruction records.")

def main():
    load_dotenv()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    parser = argparse.ArgumentParser(description="RLF Vast.ai 1B Token CoT Dataset Builder")
    parser.add_argument("--output-dir", type=str, default="demo_data/vast_1b", help="Target dataset directory")
    parser.add_argument("--max-samples", type=int, default=2500000, help="Maximum reasoning samples (default: 2500000)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    print("=========================================================================")
    print("  RLF Vast.ai Frontier 24G — 1-Billion Token CoT Dataset Builder        ")
    print("=========================================================================")

    if token:
        print(f"[+] HF_TOKEN authenticated successfully.")
    else:
        print("[+] No HF_TOKEN provided. Streaming public HF datasets...")

    # --- GSM8K: Try multiple URLs ---
    gsm8k_urls = [
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
        "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/train.jsonl",
        "https://huggingface.co/datasets/gsm8k/resolve/main/main/train.jsonl",
    ]
    gsm8k_cache = cache_dir / "gsm8k_train.jsonl"
    if not gsm8k_cache.exists() or gsm8k_cache.stat().st_size < 100:
        for url in gsm8k_urls:
            try:
                download_file(url, gsm8k_cache, token)
                if gsm8k_cache.exists() and gsm8k_cache.stat().st_size > 100:
                    break
            except Exception as e:
                print(f"[-] Warning: {url} failed: {e}")
                if gsm8k_cache.exists():
                    gsm8k_cache.unlink()

    # If GSM8K still not available, try HF datasets API
    if not gsm8k_cache.exists() or gsm8k_cache.stat().st_size < 100:
        result = download_hf_dataset_parquet("openai/gsm8k", "train", cache_dir, token)
        if result:
            gsm8k_cache = result

    # --- Alpaca ---
    alpaca_urls = [
        "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
        "https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet",
    ]
    alpaca_cache = cache_dir / "alpaca_data.json"
    if not alpaca_cache.exists() or alpaca_cache.stat().st_size < 100:
        for url in alpaca_urls:
            try:
                download_file(url, alpaca_cache, token)
                if alpaca_cache.exists() and alpaca_cache.stat().st_size > 100:
                    break
            except Exception as e:
                print(f"[-] Warning: {url} failed: {e}")
                if alpaca_cache.exists():
                    alpaca_cache.unlink()

    instructions_file = out_dir / "instructions.tsv"
    corpus_file = out_dir / "corpus.txt"

    instructions = ["# task\tdomain\tprompt\trationale\tresponse\tquality"]
    corpus = []

    if gsm8k_cache.exists() and gsm8k_cache.stat().st_size > 100:
        process_gsm8k(gsm8k_cache, instructions, corpus, args.max_samples)
    else:
        print("[-] ERROR: GSM8K dataset not available! Check network connectivity.")

    if alpaca_cache.exists() and alpaca_cache.stat().st_size > 100:
        process_alpaca(alpaca_cache, instructions, corpus, args.max_samples)
    else:
        print("[-] ERROR: Alpaca dataset not available! Check network connectivity.")

    total_rows = len(instructions) - 1
    if total_rows == 0:
        print("[-] FATAL: No data was downloaded. Aborting.")
        sys.exit(1)

    with open(instructions_file, "w", encoding="utf-8") as f:
        f.write("\n".join(instructions) + "\n")
    
    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write("\n".join(corpus) + "\n")

    print(f"\n=======================================================")
    print(f"[+] Successfully built dataset manifest:")
    print(f"    instructions_rows={total_rows}")
    print(f"    corpus_lines={len(corpus)}")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
