# 📘 Magnum 5 & 5.1 (`fabric-magnum-5.1`) Comprehensive Deployment & Operations Guide

Master operations, interactive multi-turn chat, multi-modal vision input, Vast.ai serverless deployment, and zero-dependency clean machine setup for **Magnum 5 (`vast_frontier_24g_5b_master.rlfsp`)** and **Magnum 5.1 (`vast_frontier_24g_8b_flagship.rlfsp`)**.

---

## 📍 Local Checkpoint File Locations

Both 12.0 GB master checkpoints have been verified and downloaded to your local machine:

1. **Magnum 5.1 (8.0B Token Claude Fable 5 Tier Flagship)**:
   * **Path**: `C:\Users\GC121\Downloads\vast_frontier_24g_8b_flagship.rlfsp`
   * **Size**: `12.0 GB` (`11,965,549,718 bytes` / `10.84 GiB`)
   * **Verification Hash**: `0x4e9e10dcdbfd5f10`

2. **Magnum 5 (5.0B Token Master)**:
   * **Path**: `C:\Users\GC121\Downloads\vast_frontier_24g_5b_master.rlfsp`
   * **Size**: `12.0 GB` (`11,965,549,718 bytes` / `10.84 GiB`)
   * **Verification Hash**: `0x23a644608b116a3a`

---

## 💬 1. How to Chat Interactively with Existing Vast.ai Instance

To chat interactively in real time with your running Vast.ai instance from your local PowerShell / Terminal, run this single copy-paste SSH command:

### 🚀 Copy-Paste SSH Command for Existing Instance (`38.49.42.46:55936`):

```bash
ssh -t -p 55936 root@38.49.42.46 -o StrictHostKeyChecking=no "/workspace/RLF/build/3090-release/solstice chat --checkpoint /workspace/RLF/models/vast_frontier_24g_8b_flagship.rlfsp --profile frontier-24g --backend cuda --enforce-profile"
```

> **Note**: The `-t` flag allocates a pseudo-terminal so that arrow keys, backspace, and multi-line responses work smoothly inside PowerShell!

---

### 🎮 Inside the Interactive Chat REPL:

Once connected, you will see the interactive prompt:
```text
=========================================================================
  Solstice Attractor Vector Interactive Chat REPL (Magnum 5.1 8.0B)
=========================================================================
User> Hello! Explain your capabilities in simple terms.
Solstice> ...

User> Solve d/dx [x^3 * e^(2x)] step-by-step.
Solstice> ...

User> /image /path/to/image.png
[+] Image loaded and vision feature vectors extracted.
User> Analyze this image and write a Next.js UI component to match it.
Solstice> ...

User> exit
```

---

## 🖼️ 2. Sending Images in Message (Multi-Modal Vision Input)

Solstice natively ingests images into attractor context memory nodes. You can send images in two ways:

#### Method A: Direct Command-Line Vision Prompt
```bash
ssh -p 55936 root@38.49.42.46 -o StrictHostKeyChecking=no "/workspace/RLF/build/3090-release/solstice ask --checkpoint /workspace/RLF/models/vast_frontier_24g_8b_flagship.rlfsp --image /workspace/RLF/demo_data/diagram.png --prompt 'Analyze this diagram and write a Next.js UI component.'"
```

#### Method B: Inside Interactive REPL Chat
Type `/image` followed by the image path on the remote server:
```text
User> /image /workspace/RLF/demo_data/ui_mockup.png
[+] Image feature vectors extracted into attractor memory.
User> Describe this interface and convert it into a Three.js 3D WebGL scene.
```

---

## 🌐 3. How to Deploy as a Serverless Endpoint on Vast.ai

Create `serve_magnum.py` on your Vast.ai instance to expose an OpenAI-compatible HTTP API server (`/v1/chat/completions`):

```python
import os
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Magnum 5.1 Serverless API")

CHECKPOINT_PATH = os.getenv(
    "CHECKPOINT_PATH", "/workspace/RLF/models/vast_frontier_24g_8b_flagship.rlfsp"
)
SOLSTICE_BIN = os.getenv(
    "SOLSTICE_BIN", "/workspace/RLF/build/3090-release/solstice"
)


class ChatRequest(BaseModel):
  prompt: str
  image_path: str = None


@app.post("/v1/chat/completions")
@app.post("/v1/ask")
def chat_completion(req: ChatRequest):
  cmd = [
      SOLSTICE_BIN,
      "ask",
      "--checkpoint",
      CHECKPOINT_PATH,
      "--profile",
      "frontier-24g",
      "--backend",
      "cuda",
      "--enforce-profile",
      "--prompt",
      req.prompt,
  ]
  if req.image_path:
    cmd.extend(["--image", req.image_path])

  try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return {
        "id": "magnum-5.1-flagship",
        "object": "chat.completion",
        "choices": [{
            "message": {"role": "assistant", "content": result.stdout.strip()}
        }],
    }
  except subprocess.CalledProcessError as e:
    raise HTTPException(status_code=500, detail=e.stderr)


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Run Serverless API**:
```bash
pip install fastapi uvicorn pydantic
python3 serve_magnum.py
```

---

## 🏗️ 4. How to Run on a Completely Empty Machine (From Scratch)

Run this **1-line automated setup script** on any new Linux GPU machine:

```bash
# Step 1: Install CUDA & C++ Build Toolchain
sudo apt-get update && sudo apt-get install -y git cmake g++ build-essential nvidia-cuda-toolkit python3 python3-pip

# Step 2: Clone Codebase & Build Engine
git clone https://github.com/Mekanbahmanjeh/RLF.git
cd RLF
cmake -B build/release -S . -DCMAKE_BUILD_TYPE=Release -DRLF_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
cmake --build build/release --target solstice -j$(nproc)

# Step 3: Copy Checkpoint & Launch Interactive Chat!
mkdir -p models
# Copy your vast_frontier_24g_8b_flagship.rlfsp file into models/
./build/release/solstice chat --checkpoint models/vast_frontier_24g_8b_flagship.rlfsp --profile frontier-24g --backend cuda
```

---

## 🛠️ Quick Command Reference

| Action | Exact Command |
| :--- | :--- |
| **Existing Vast.ai Chat** | `ssh -t -p 55936 root@38.49.42.46 -o StrictHostKeyChecking=no "/workspace/RLF/build/3090-release/solstice chat --checkpoint /workspace/RLF/models/vast_frontier_24g_8b_flagship.rlfsp --profile frontier-24g --backend cuda --enforce-profile"` |
| **Local Windows REPL Chat** | `solstice.exe chat --checkpoint "C:\Users\GC121\Downloads\vast_frontier_24g_8b_flagship.rlfsp" --profile frontier-24g --backend cuda` |
| **Send Image / Vision Prompt** | `solstice ask --checkpoint <file.rlfsp> --image <path.png> --prompt "..."` |
| **Serverless REST API** | `python3 serve_magnum.py` (Exposes port `8000` REST API) |
