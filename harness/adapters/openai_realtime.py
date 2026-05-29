import asyncio
import base64
import json
import os
from typing import AsyncIterator

import numpy as np
import websockets

from .base import BaseAdapter

_WS_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
_MODEL_RATE = 24000  # OpenAI Realtime requires 24kHz


def _resample(pcm16: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate:
        return pcm16
    samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    ratio = dst_rate / src_rate
    new_len = int(len(samples) * ratio)
    indices = np.linspace(0, len(samples) - 1, new_len)
    resampled = np.interp(indices, np.arange(len(samples)), samples)
    return resampled.astype(np.int16).tobytes()


class OpenAIRealtimeAdapter(BaseAdapter):
    def __init__(self, api_key: str | None = None, harness_rate: int = 16000):
        self._api_key = api_key or os.environ["OPENAI_API_KEY"]
        self._harness_rate = harness_rate
        self._ws = None
        self._response_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._receiver_task: asyncio.Task | None = None

    async def connect(self, system_prompt: str) -> None:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        self._ws = await websockets.connect(_WS_URL, additional_headers=headers)

        # Configure session with system prompt and audio formats
        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": system_prompt,
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": None,
                "turn_detection": None,  # we handle VAD ourselves
            },
        }
        await self._ws.send(json.dumps(session_update))
        self._receiver_task = asyncio.create_task(self._receive_loop())

    async def send_audio(self, chunk: bytes) -> None:
        if self._ws is None:
            raise RuntimeError("Not connected — call connect() first")

        # Resample from harness rate (16kHz) to model rate (24kHz)
        resampled = _resample(chunk, self._harness_rate, _MODEL_RATE)
        audio_b64 = base64.b64encode(resampled).decode()

        await self._ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
        }))
        await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await self._ws.send(json.dumps({"type": "response.create"}))

    async def stream_response(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._response_queue.get()
            if chunk is None:
                break
            yield chunk

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._receiver_task:
            self._receiver_task.cancel()
            self._receiver_task = None

    async def _receive_loop(self) -> None:
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                event_type = msg.get("type", "")

                if event_type == "response.audio.delta":
                    audio_b64 = msg.get("delta", "")
                    if audio_b64:
                        pcm = base64.b64decode(audio_b64)
                        # Resample from 24kHz back to harness rate (16kHz)
                        pcm = _resample(pcm, _MODEL_RATE, self._harness_rate)
                        await self._response_queue.put(pcm)

                elif event_type == "response.audio.done":
                    await self._response_queue.put(None)

                elif event_type == "error":
                    print(f"[openai] error: {msg.get('error')}")
                    await self._response_queue.put(None)

        except websockets.ConnectionClosed:
            await self._response_queue.put(None)
