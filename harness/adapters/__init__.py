from .kame import KameAdapter
from .moshirag import MoshiRAGAdapter
from .openai_realtime import OpenAIRealtimeAdapter
from .qwen import QwenAdapter

ADAPTERS = {
    "qwen": QwenAdapter,
    "openai": OpenAIRealtimeAdapter,
    "kame": KameAdapter,
    "moshirag": MoshiRAGAdapter,
}
