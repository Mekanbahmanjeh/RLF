RLF Windows 6 GB Conversation & Reasoning Package (v5)
======================================================

This archive contains the complete source code, dataset generators, and tools
required to build, train, and chat with an RLF model on a 6 GB VRAM Windows GPU.

Features in v5:
- <think>...</think> Chain-of-Thought reasoning format support.
- Expanded English grammar, conversation, and symbolic arithmetic dataset generators.
- Hugging Face / JSON dataset importer (scripts/fetch_hf_dataset.py) with .env token support.

Windows Prerequisites (for CUDA mode on target 6GB VRAM GPU):
1. Windows 10 or Windows 11, 64-bit.
2. NVIDIA GPU with 6 GB VRAM + NVIDIA Display Drivers + CUDA Toolkit.
3. Visual Studio 2022 / 2026 Build Tools with "Desktop development with C++".
4. CMake on PATH (3.25+).

Instructions on your 6 GB VRAM Windows machine:

1. Extract the archive to a short path (e.g. C:\rlf-demo-v5).
2. Open PowerShell in that directory.

3. Build and train on your 6GB NVIDIA GPU:
     .\TRAIN_PREVIEW_6GB_WINDOWS.bat -Backend cuda -Reset

4. Start interactive chat with <think> block reasoning:
     .\CHAT_PREVIEW_6GB_WINDOWS.bat -Backend cuda

5. (Optional) Convert external Hugging Face / JSON instruction datasets:
     python scripts/fetch_hf_dataset.py --input path/to/dataset.jsonl
     (Supply your HF_TOKEN in .env if accessing private/gated repositories)

CPU Fallback Mode:
     .\TRAIN_PREVIEW_6GB_WINDOWS.bat -Backend optimized_cpu -Reset
     .\CHAT_PREVIEW_6GB_WINDOWS.bat -Backend optimized_cpu
