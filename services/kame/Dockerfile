# kame-server — Railway A100 40 GB GPU required
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m pip install --no-cache-dir \
    "kame-model @ git+https://github.com/SakanaAI/kame.git"

EXPOSE 8998

# GOOGLE_CREDENTIALS_BASE64: base64-encoded contents of your service account JSON.
CMD bash -c '\
  if [ -n "$GOOGLE_CREDENTIALS_BASE64" ]; then \
    echo "$GOOGLE_CREDENTIALS_BASE64" | base64 -d > /tmp/gcloud.json && \
    export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcloud.json; \
  fi && \
  exec python3.12 -m kame.server_oracle_parallel \
    --hf-repo SakanaAI/kame \
    --dtype bfloat16 \
    --host 0.0.0.0 --port 8998 --device cuda'
