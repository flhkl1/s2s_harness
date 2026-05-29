# CUDA & Deployment Setup

## Hardware requirements

| Service | GPU | VRAM | Notes |
|---|---|---|---|
| harness-api | none | — | CPU only |
| kame-server | NVIDIA | ≥ 16 GB | BF16 cast from F32 at load |
| moshirag-main | NVIDIA | ≥ 16 GB | Ships as BF16 |
| moshirag-cond | NVIDIA | ≥ 4 GB | Reference encoder only |

Railway tier: **A100 40 GB** for all three GPU services. This ensures identical hardware across the A/B comparison.

---

## Local setup (NVIDIA GPU machine)

### 1. Install NVIDIA driver

```bash
# Check current driver
nvidia-smi

# If missing or < 550, install:
# Ubuntu/Debian
sudo apt-get install -y nvidia-driver-550
sudo reboot

# Verify
nvidia-smi  # should show CUDA Version: 12.x
```

### 2. Install CUDA toolkit (12.4)

```bash
# Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4
```

### 3. Install nvidia-container-toolkit (GPU in Docker)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 4. Pre-download model weights

This avoids long cold-start times when the container first runs.

```bash
pip install huggingface-hub

# MoshiRAG model (~16 GB BF16)
huggingface-cli download kyutai/moshika-rag-pytorch-bf16

# KAME model (~32 GB F32, ~16 GB after BF16 cast)
huggingface-cli download SakanaAI/kame
```

### 5. Configure environment

```bash
cp .env.example .env
# Fill in: OPENAI_API_KEY, GOOGLE_APPLICATION_CREDENTIALS, STT_API_KEY
```

For `GOOGLE_APPLICATION_CREDENTIALS`: point to a Google Cloud service account JSON
with the **Cloud Speech-to-Text API** enabled (used by KAME's ASR pipeline).

### 6. Run locally

```bash
docker-compose up --build
```

Services:
- Harness API: http://localhost:8000
- KAME server: ws://localhost:8998 (internal)
- MoshiRAG main: ws://localhost:8999 (internal)
- MoshiRAG conditioner: http://localhost:8001 (internal)

Health check: `curl http://localhost:8000/health`

---

## Railway deployment

### 1. Install Railway CLI and log in

```bash
npm install -g @railway/cli
railway login
railway init   # in the repo root
```

### 2. Deploy harness-api (the public service)

```bash
railway up   # uses railway.toml + Dockerfile at repo root
```

Railway will give you a public URL, e.g. `https://harness-api-xyz.railway.app`.

### 3. Add model server services

In the Railway dashboard, create three more services in the same project:

**kame-server**
- Source: this repo, Dockerfile: `Dockerfile.kame`
- GPU: NVIDIA A100 40 GB
- Env vars:
  - `OPENAI_API_KEY`
  - `GOOGLE_APPLICATION_CREDENTIALS` (base64-encode the JSON: `base64 -w0 credentials.json`)

**moshirag-cond** (deploy this before moshirag-main)
- Source: this repo, Dockerfile: `Dockerfile.moshirag`
- GPU: NVIDIA A100 40 GB
- Override start command:
  ```
  python3.12 -m moshi.moshi.server_conditioner --config hf://kyutai/moshika-rag-pytorch-bf16/config.json --moshi-weight hf://kyutai/moshika-rag-pytorch-bf16/model.safetensors --cuda-device 0 --conditioner reference_with_time --port 8001
  ```

**moshirag-main**
- Source: this repo, Dockerfile: `Dockerfile.moshirag`
- GPU: NVIDIA A100 40 GB
- Env vars:
  - `OPENAI_API_KEY`
  - `LLM_API_KEY` (same value as OPENAI_API_KEY)
  - `LLM_BASE_URL=https://api.openai.com/v1`
  - `LLM_MODEL_NAME=gpt-4o-mini`
  - `REFERENCE_ENCODER_URL=http://<moshirag-cond-internal-hostname>:8001`
  - `STT_API_KEY`

### 4. Wire internal hostnames into harness-api

After all services are up, set these env vars on **harness-api** in the Railway dashboard:

```
KAME_HOST=<kame-server Railway internal hostname>
MOSHIRAG_HOST=<moshirag-main Railway internal hostname>
```

Railway internal hostnames look like `kame-server.railway.internal`.

### 5. Public WebSocket endpoints

```
wss://harness-api-xyz.railway.app/ws/kame
wss://harness-api-xyz.railway.app/ws/moshirag
```

---

## Verifying the deployment

```bash
# Health check
curl https://harness-api-xyz.railway.app/health
# → {"status": "ok"}

# List runs
curl https://harness-api-xyz.railway.app/runs

# Send a test audio clip (websocat required)
# Generate a 2-second silent PCM file for a quick connectivity test:
python3 -c "import struct; open('test.pcm','wb').write(struct.pack('<' + 'h'*32000, *([0]*32000)))"
websocat "wss://harness-api-xyz.railway.app/ws/kame?run_id=test-001" --binary < test.pcm
```
