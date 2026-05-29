"""
MoshiRAG adapter — full-duplex Moshi + asynchronous retrieval via trigger tokens.

Retrieval is triggered internally by the model; no extra signaling is needed
from the harness. The retrieval LLM backend (LLM_BASE_URL / LLM_MODEL_NAME)
and reference encoder (REFERENCE_ENCODER_URL) are configured at server startup.

Server: moshi.moshi.server
  python -m moshi.moshi.server \\
      --config hf://kyutai/moshika-rag-pytorch-bf16/config.json \\
      --host 0.0.0.0 --port 8999

Environment variables:
  MOSHIRAG_HOST  (default: localhost)
  MOSHIRAG_PORT  (default: 8999)
"""

from ._moshi_ws import MoshiWSAdapter


class MoshiRAGAdapter(MoshiWSAdapter):
    _host_env = "MOSHIRAG_HOST"
    _port_env = "MOSHIRAG_PORT"
    _server_label = "MoshiRAG server"
