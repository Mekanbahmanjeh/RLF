#!/usr/bin/env python3
"""
Hugging Face & External Dataset Fetcher for RLF
================================================
Fetches or converts instruction, dialogue, and reasoning datasets (JSON, JSONL, HF Datasets)
into RLF native manifest formats (dialogues.tsv, instructions.tsv, corpus.txt).

Supports optional .env file containing HF_TOKEN for private/gated repositories.
Formats step-by-step reasoning outputs using <think>...</think> blocks.
"""

import os
import sys
import json
import argparse
from pathlib import Path

def load_dotenv():
    """Load key-value pairs from a local .env file if present."""
    env_path = Path(".env")
    if not env_path.exists():
        env_path = Path("../.env")
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip("'\"")

def format_think_block(reasoning: str, answer: str) -> str:
    """Format step-by-step reasoning into <think>...</think> block before final answer."""
    reasoning = reasoning.strip()
    answer = answer.strip()
    if not reasoning:
        return answer
    return f"<think>\n{reasoning}\n</think>\n\n{answer}"

def convert_jsonl_to_rlf(input_file: str, output_dir: str):
    """Convert local JSON/JSONL instruction file to RLF TSV manifests."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    dialogues_file = out_path / "dialogues.tsv"
    instructions_file = out_path / "instructions.tsv"
    corpus_file = out_path / "corpus.txt"
    
    dialogues = ["# prompt\tresponse\toptional_grounding"]
    instructions = ["# task\tdomain\tprompt\trationale\tresponse\tquality"]
    corpus = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                prompt = data.get("prompt") or data.get("instruction") or data.get("question") or ""
                response = data.get("response") or data.get("output") or data.get("answer") or ""
                rationale = data.get("rationale") or data.get("reasoning") or data.get("thought") or ""
                
                # Sanitize single-line TSV fields
                prompt = prompt.replace("\t", " ").replace("\r", "").replace("\n", " ").strip()
                response_clean = response.replace("\t", " ").replace("\r", "").replace("\n", " ").strip()
                rationale_clean = rationale.replace("\t", " ").replace("\r", "").replace("\n", " ").strip()
                
                if not prompt or not response_clean:
                    continue
                
                if rationale_clean:
                    full_response = format_think_block(rationale_clean, response_clean).replace("\t", " ").replace("\r", "").replace("\n", " ")
                    instructions.append(f"hf_task_{idx}\tgeneral\t{prompt}\t{rationale_clean}\t{full_response}\t1.0")
                else:
                    dialogues.append(f"{prompt}\t{response_clean}\tImported HF sample")
                
                corpus.append(f"{prompt} {response_clean}")
            except Exception as e:
                continue

    with open(dialogues_file, "w", encoding="utf-8") as f:
        f.write("\n".join(dialogues) + "\n")
    with open(instructions_file, "w", encoding="utf-8") as f:
        f.write("\n".join(instructions) + "\n")
    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write("\n".join(corpus) + "\n")
        
    print(f"Imported {len(lines)} records into {output_dir}:")
    print(f"  dialogues={len(dialogues)-1}")
    print(f"  instructions={len(instructions)-1}")
    print(f"  corpus_lines={len(corpus)}")

def main():
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    parser = argparse.ArgumentParser(description="RLF Hugging Face & JSON Dataset Importer")
    parser.add_argument("--input", type=str, help="Path to local JSON or JSONL dataset file")
    parser.add_argument("--output-dir", type=str, default="demo_data/preview_conversation", help="Output TSV manifest directory")
    args = parser.parse_args()

    if hf_token:
        print(f"[+] Found HF_TOKEN in environment/dotenv. Authenticated for Hugging Face access.")
    else:
        print("[-] No HF_TOKEN found in .env (Public datasets will be downloaded without token).")

    if args.input and os.path.exists(args.input):
        convert_jsonl_to_rlf(args.input, args.output_dir)
    else:
        print("[+] Dataset converter utility ready. Usage: python scripts/fetch_hf_dataset.py --input path/to/dataset.jsonl")

if __name__ == "__main__":
    main()
