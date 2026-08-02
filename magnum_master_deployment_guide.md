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

## 💬 1. How to Run Interactive Multi-Turn Chat & Send Images

Instead of one-off single-prompt queries (`solstice ask`), you can launch an **interactive multi-turn conversation REPL session** with context history persistence and image/vision capabilities.

### 💻 A. Interactive Terminal REPL Chat (PowerShell / Linux Terminal)

#### On Windows PowerShell (Local Machine):
```powershell
# Navigate to directory containing compiled solstice binary
solstice.exe chat `
  --checkpoint "C:\Users\GC121\Downloads\vast_frontier_24g_8b_flagship.rlfsp" `
  --profile frontier-24g `
  --backend cuda `
  --enforce-profile
```

#### On Linux / GPU Server (Vast.ai Instance):
```bash
/workspace/RLF/build/3090-release/solstice chat \
  --checkpoint /workspace/RLF/models/vast_frontier_24g_8b_flagship.rlfsp \
  --profile frontier-24g \
  --backend cuda \
  --enforce-profile
```

---

### 🖼️ B. Sending Images in Message (Multi-Modal Vision Input)

Solstice natively ingests images into attractor context memory nodes. You can send images in two ways:

#### Method 1: Command-Line Direct Vision Prompt
```powershell
solstice.exe ask `
  --checkpoint "C:\Users\GC121\Downloads\vast_frontier_24g_8b_flagship.rlfsp" `
  --image "C:\Users\GC121\Pictures\diagram.jpg" `
  --prompt "Analyze this diagram and write a Next.js UI component to render it."
```

#### Method 2: Inside Interactive Chat Mode
While inside `solstice chat`, type `/image` followed by the image path:
```text
Solstice Chat REPL
User> /image C:\Users\GC121\Pictures\ui_mockup.png
[+] Image loaded and vision feature vectors extracted successfully.
User> Describe this interface and convert it into a Three.js 3D WebGL scene.
Solstice> ...
```

---

## 🌐 2. How to Deploy as a Serverless Endpoint on Vast.ai

You can deploy Magnum 5.1 on Vast.ai as an OpenAI-compatible Serverless HTTP API server using FastAPI.

### 🐍 Python Serverless API Launcher Script (`serve_magnum.py`)

Create `serve_magnum.py` in your repository:

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

### 🚀 Running Serverless on Vast.ai
```bash
pip install fastapi uvicorn pydantic
python3 serve_magnum.py
```
Now any external application can send HTTP POST requests to `http://<YOUR-VAST-IP>:8000/v1/chat/completions`!

---

## 🏗️ 3. How to Run on a Completely Empty Machine (From Scratch)

Here is the complete step-by-step setup script for a brand new, empty Linux GPU machine:

### ⚡ 1-Line Full Automated Setup
```bash
# Step 1: Install CUDA & C++ Build Toolchain
sudo apt-get update && sudo apt-get install -y git cmake g++ build-essential nvidia-cuda-toolkit python3 python3-pip

# Step 2: Clone Codebase
git clone https://github.com/Mekanbahmanjeh/RLF.git
cd RLF

# Step 3: Build Solstice Engine
cmake -B build/release -S . -DCMAKE_BUILD_TYPE=Release -DRLF_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
cmake --build build/release --target solstice -j$(nproc)

# Step 4: Download or Copy Checkpoint File into models/
mkdir -p models
# Copy your vast_frontier_24g_8b_flagship.rlfsp file into models/ directory

# Step 5: Start Interactive Chat!
./build/release/solstice chat --checkpoint models/vast_frontier_24g_8b_flagship.rlfsp --profile frontier-24g --backend cuda
```

---

## 🛠️ Summary Matrix

| Task | Command / Method |
| :--- | :--- |
| **Interactive REPL Chat** | `solstice chat --checkpoint <file.rlfsp> --profile frontier-24g --backend cuda` |
| **Send Image File** | `solstice ask --checkpoint <file.rlfsp> --image <path.png> --prompt "..."` |
| **Vast.ai Serverless API** | `python3 serve_magnum.py` (Exposes port `8000` REST API) |
| **Empty Machine Build** | `cmake -B build -DRLF_ENABLE_CUDA=ON && cmake --build build --target solstice` |
