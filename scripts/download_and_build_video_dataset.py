#!/usr/bin/env python3
"""
Video Sequence Dataset Fetcher & Builder for RLF VideoGen (6GB VRAM)
====================================================================
Generates/downloads learned motion prototype video sequence manifests (videos.tsv)
mapping text prompts to contiguous frame sequences.
"""

import os
import sys
import hashlib
import argparse
from pathlib import Path

def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()

def generate_video_sequence(output_dir: Path):
    """Generate sample motion video frames (rolling ball, moving shape) for RLF VideoGen."""
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from PIL import Image, ImageDraw
        # Sequence 1: A ball rolling across a floor (16 frames)
        for f in range(16):
            img = Image.new("RGB", (128, 128), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)
            # Floor
            draw.rectangle([0, 96, 128, 128], fill=(180, 140, 100))
            # Rolling Ball x position moves from 16 to 112
            x_pos = 16 + int((112 - 16) * (f / 15.0))
            draw.ellipse([x_pos - 12, 72, x_pos + 12, 96], fill=(220, 50, 50))
            img.save(frames_dir / f"seq1_frame_{f:02d}.png")

        # Sequence 2: A glowing star moving in the night sky (16 frames)
        for f in range(16):
            img = Image.new("RGB", (128, 128), color=(20, 20, 50))
            draw = ImageDraw.Draw(img)
            x_pos = 16 + int((112 - 16) * (f / 15.0))
            y_pos = 32 + int(20 * (f % 4))
            draw.ellipse([x_pos - 10, y_pos - 10, x_pos + 10, y_pos + 10], fill=(255, 220, 50))
            img.save(frames_dir / f"seq2_frame_{f:02d}.png")
    except ImportError:
        pass

def main():
    parser = argparse.ArgumentParser(description="RLF VideoGen Dataset Builder")
    parser.add_argument("--output-dir", type=str, default="demo_data/video_sequences", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] Generating motion video sequence frames in: {out_dir}")
    generate_video_sequence(out_dir)

    manifest_file = out_dir / "videos.tsv"
    rows = []
    
    # Format: sequence_id <TAB> frame_index <TAB> fps <TAB> prompt <TAB> frame_image_path
    # Sequence 1
    for f in range(16):
        rel_path = f"frames/seq1_frame_{f:02d}.png"
        full_path = out_dir / rel_path
        if full_path.exists():
            rows.append(f"seq-rolling-ball\t{f}\t16.0\tA red ball rolling across a wooden floor\t{rel_path}")

    # Sequence 2
    for f in range(16):
        rel_path = f"frames/seq2_frame_{f:02d}.png"
        full_path = out_dir / rel_path
        if full_path.exists():
            rows.append(f"seq-glowing-star\t{f}\t16.0\tA glowing star moving across the night sky\t{rel_path}")

    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    print(f"[+] Successfully built VideoGen manifest: {manifest_file} ({len(rows)} frame rows)")

if __name__ == "__main__":
    main()
