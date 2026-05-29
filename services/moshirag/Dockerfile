# moshirag-main — Railway A100 40 GB GPU required
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m pip install --no-cache-dir \
    "moshi[rag] @ git+https://github.com/kyutai-labs/moshi-rag.git"

EXPOSE 8999
CMD ["python3.12", "-m", "moshi.moshi.server", \
     "--config", "hf://kyutai/moshika-rag-pytorch-bf16/config.json", \
     "--moshi-weight", "hf://kyutai/moshika-rag-pytorch-bf16/model.safetensors", \
     "--cuda-device", "0", \
     "--host", "0.0.0.0", "--port", "8999", \
     "--gradium-stt"]
