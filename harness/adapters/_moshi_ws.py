"""
Shared WebSocket + Opus transport layer for Moshi-based model servers.

Wire protocol (server → client):
  Binary messages with a single-byte type prefix:
    0x00  handshake — server ready (ignored here, safe to send audio immediately)
    0x01  Opus-encoded audio output at 24 kHz, frame_size=1920 samples (80 ms)
    0x02  text token from LLM oracle (logged at server; discarded here)

Wire protocol (client → server):
  Raw binary Opus frames at 24 kHz, mono, frame_size=1920 samples.
  No prefix byte. VBR is fine.

Response-end detection: there is no explicit done signal from Moshi-based
servers. We declare the response finished after RESPONSE_SILENCE_S seconds of
no audio output.
"""

import asyncio
import os
from typing import AsyncIterator

import numpy as np
import opuslib
import websockets

from .base import BaseAdapter

_MODEL_RATE = 24000
_FRAME_SAMPLES = 1920        # 80 ms at 24 kHz — Moshi's frame size
_RESPONSE_SILENCE_S = 2.0   # seconds of output silence → response done


def _resample(pcm16: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate:
        return pcm16
    samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    new_len = int(len(samples) * dst_rate / src_rate)
    indices = np.linspace(0, len(samples) - 1, new_len)
    resampled = np.interp(indices, np.arange(len(samples)), samples)
    return resampled.astype(np.int16).tobytes()


class MoshiWSAdapter(BaseAdapter):
    """Base adapter for Moshi-based servers (KAME, MoshiRAG) using Opus over WebSocket."""

    _host_env: str   # e.g. "KAME_HOST" — override in subclass
    _port_env: str   # e.g. "KAME_PORT" — override in subclass
    _server_label: str  # human-readable name for error messages

    def __init__(self, harness_rate: int = 16000, full_duplex: bool = True):
        self._host = os.environ.get(self._host_env, "localhost")
        self._port = int(os.environ.get(self._port_env, "8998"))
        self._harness_rate = harness_rate
        self._full_duplex = full_duplex
        self._ws = None
        self._response_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._receiver_task: asyncio.Task | None = None
        self._encoder = opuslib.Encoder(_MODEL_RATE, 1, opuslib.APPLICATION_VOIP)
        self._decoder = opuslib.Decoder(_MODEL_RATE, 1)
        self._send_buf = bytearray()  # accumulates PCM until a full 80 ms frame is ready

    async def connect(self, system_prompt: str) -> None:
        url = f"ws://{self._host}:{self._port}"
        try:
            self._ws = await websockets.connect(url, max_size=None)
        except OSError:
            raise RuntimeError(
                f"{self._server_label} not reachable at {url} — "
                "start the server first (see SETUP_CUDA.md)"
            )
        self._receiver_task = asyncio.create_task(self._receive_loop())

    async def send_audio(self, chunk: bytes) -> None:
        if self._ws is None:
            raise RuntimeError("Not connected — call connect() first")
        resampled = _resample(chunk, self._harness_rate, _MODEL_RATE)
        self._send_buf.extend(resampled)
        frame_bytes = _FRAME_SAMPLES * 2  # int16 = 2 bytes per sample
        while len(self._send_buf) >= frame_bytes:
            frame = bytes(self._send_buf[:frame_bytes])
            del self._send_buf[:frame_bytes]
            encoded = self._encoder.encode(frame, _FRAME_SAMPLES)
            await self._ws.send(encoded)

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
        if self._full_duplex:
            await self._receive_loop_stream()
        else:
            await self._receive_loop_turn()

    async def _receive_loop_stream(self) -> None:
        """Full-duplex: forward every audio chunk immediately, no silence timeout."""
        try:
            async for msg in self._ws:
                if not isinstance(msg, bytes) or len(msg) < 1:
                    continue
                if msg[0] == 0x01:
                    pcm = self._decoder.decode(msg[1:], _FRAME_SAMPLES)
                    out = _resample(pcm, _MODEL_RATE, self._harness_rate)
                    await self._response_queue.put(out)
        except websockets.ConnectionClosed:
            pass
        finally:
            await self._response_queue.put(None)

    async def _receive_loop_turn(self) -> None:
        """Turn-based: declare response done after RESPONSE_SILENCE_S of output silence."""
        timeout_task: asyncio.Task | None = None

        async def _declare_done() -> None:
            await asyncio.sleep(_RESPONSE_SILENCE_S)
            await self._response_queue.put(None)

        try:
            async for msg in self._ws:
                if not isinstance(msg, bytes) or len(msg) < 1:
                    continue
                if msg[0] == 0x01:
                    if timeout_task:
                        timeout_task.cancel()
                    pcm = self._decoder.decode(msg[1:], _FRAME_SAMPLES)
                    out = _resample(pcm, _MODEL_RATE, self._harness_rate)
                    await self._response_queue.put(out)
                    timeout_task = asyncio.create_task(_declare_done())
        except websockets.ConnectionClosed:
            pass
        finally:
            if timeout_task:
                timeout_task.cancel()
            await self._response_queue.put(None)
