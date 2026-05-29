"""
KAME adapter — tandem S2S + async LLM oracle injection, built on Moshi.

Server: kame.server_oracle_parallel
  python -m kame.server_oracle_parallel --hf-repo SakanaAI/kame \\
      --host 0.0.0.0 --port 8998 --device cuda

Environment variables:
  KAME_HOST  (default: localhost)
  KAME_PORT  (default: 8998)
"""

from ._moshi_ws import MoshiWSAdapter


class KameAdapter(MoshiWSAdapter):
    _host_env = "KAME_HOST"
    _port_env = "KAME_PORT"
    _server_label = "KAME server"
