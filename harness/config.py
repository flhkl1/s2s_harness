import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def get(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


# External LLM — shared by both adapters (KAME oracle + MoshiRAG retrieval)
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# Legacy / other adapters
DASHSCOPE_API_KEY: str = os.environ.get("DASHSCOPE_API_KEY", "")

# Model server hosts (Railway internal hostnames in prod, localhost in dev)
KAME_HOST: str = os.environ.get("KAME_HOST", "localhost")
KAME_PORT: int = int(os.environ.get("KAME_PORT", "8998"))

MOSHIRAG_HOST: str = os.environ.get("MOSHIRAG_HOST", "localhost")
MOSHIRAG_PORT: int = int(os.environ.get("MOSHIRAG_PORT", "8999"))

# Audio
SAMPLE_RATE: int = 16000

# VAD — identical for both adapters (controlled variable in A/B comparison)
VAD_AGGRESSIVENESS: int = int(os.environ.get("VAD_AGGRESSIVENESS", "2"))
VAD_SILENCE_MS: int = int(os.environ.get("VAD_SILENCE_MS", "500"))
VAD_FRAME_MS: int = 20

# Logging
LOG_DIR: str = os.environ.get("LOG_DIR", "logs")

# Conversation structure
DEFAULT_STRUCTURE: Path = (
    Path(__file__).parent / "structures" / "clinical_interview.yaml"
)
