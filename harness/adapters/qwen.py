import asyncio
import json
import os
from typing import AsyncIterator

import websockets

from .base import BaseAdapter

# DashScope real-time speech WebSocket endpoint
_WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"


class QwenAdapter(BaseAdapter):
    def __init__(self, api_key: str | None = None, sample_rate: int = 16000):
        self._api_key = api_key or os.environ["DASHSCOPE_API_KEY"]
        self._sample_rate = sample_rate
        self._ws = None
        self._response_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._receiver_task: asyncio.Task | None = None

    async def connect(self, system_prompt: str) -> None:
        headers = {"Authorization": f"bearer {self._api_key}"}
        self._ws = await websockets.connect(_WS_URL, additional_headers=headers)

        # Send session init with system prompt
        init_msg = {
            "header": {
                "action": "run-task",
                "task-id": "s2s-harness",
                "streaming": "duplex",
            },
            "payload": {
                "task_group": "audio",
                "task": "omni-realtime",
                "function": "SpeechChat",
                "model": "qwen-omni-realtime",
                "input": {"system_prompt": system_prompt},
                "parameters": {
                    "audio_input": {
                        "sample_rate": self._sample_rate,
                        "format": "pcm",
                    },
                    "audio_output": {
                        "sample_rate": self._sample_rate,
                        "format": "pcm",
                    },
                },
            },
        }
        await self._ws.send(json.dumps(init_msg))
        self._receiver_task = asyncio.create_task(self._receive_loop())

    async def send_audio(self, chunk: bytes) -> None:
        if self._ws is None:
            raise RuntimeError("Not connected — call connect() first")
        msg = {
            "header": {"action": "continue-task"},
            "payload": {"input": {"audio": chunk.hex()}},
        }
        await self._ws.send(json.dumps(msg))

    async def stream_response(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._response_queue.get()
            if chunk is None:
                break
            yield chunk

    async def close(self) -> None:
        if self._ws:
            finish_msg = {"header": {"action": "finish-task"}}
            await self._ws.send(json.dumps(finish_msg))
            await self._ws.close()
            self._ws = None
        if self._receiver_task:
            self._receiver_task.cancel()
            self._receiver_task = None

    async def _receive_loop(self) -> None:
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                header = msg.get("header", {})
                event = header.get("event", "")

                if event == "result-generated":
                    audio_hex = (
                        msg.get("payload", {}).get("output", {}).get("audio", "")
                    )
                    if audio_hex:
                        await self._response_queue.put(bytes.fromhex(audio_hex))

                elif event in ("task-finished", "task-failed"):
                    await self._response_queue.put(None)
                    break
        except websockets.ConnectionClosed:
            await self._response_queue.put(None)
