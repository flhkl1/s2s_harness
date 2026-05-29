from abc import ABC, abstractmethod
from typing import AsyncIterator


class BaseAdapter(ABC):
    @abstractmethod
    async def connect(self, system_prompt: str) -> None: ...

    @abstractmethod
    async def send_audio(self, chunk: bytes) -> None: ...

    @abstractmethod
    async def stream_response(self) -> AsyncIterator[bytes]: ...

    @abstractmethod
    async def close(self) -> None: ...
