#!/usr/bin/env python3
"""
S2S harness entrypoint.

Usage:
    python run_harness.py --adapter qwen
    python run_harness.py --adapter qwen --structure structures/clinical_interview.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

from adapters import ADAPTERS
from config import (
    SAMPLE_RATE,
    VAD_AGGRESSIVENESS,
    VAD_FRAME_MS,
    VAD_SILENCE_MS,
    DEFAULT_STRUCTURE,
)
from sessions import Session, VADStream


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time S2S interview harness")
    p.add_argument(
        "--adapter",
        choices=list(ADAPTERS),
        required=True,
        help="Which model adapter to use",
    )
    p.add_argument(
        "--structure",
        type=Path,
        default=DEFAULT_STRUCTURE,
        help="Path to YAML conversation structure (default: clinical_interview.yaml)",
    )
    return p.parse_args()


async def play_audio(audio_bytes: bytes, sample_rate: int) -> None:
    pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    sd.play(pcm, samplerate=sample_rate, blocking=True)


async def run(adapter_name: str, structure_path: Path) -> None:
    session = Session.from_yaml(structure_path)
    system_prompt = session.build_system_prompt()

    AdapterClass = ADAPTERS[adapter_name]
    adapter = AdapterClass()

    print(f"[harness] connecting to {adapter_name}...")
    await adapter.connect(system_prompt)
    print(f"[harness] connected — speak now (Ctrl+C to stop)\n")

    vad = VADStream(
        sample_rate=SAMPLE_RATE,
        frame_duration_ms=VAD_FRAME_MS,
        aggressiveness=VAD_AGGRESSIVENESS,
        silence_threshold_ms=VAD_SILENCE_MS,
    )

    loop = asyncio.get_event_loop()
    mic_queue: asyncio.Queue[bytes] = asyncio.Queue()

    def mic_callback(indata, frames, time, status):
        pcm = (indata[:, 0] * 32768).astype(np.int16).tobytes()
        loop.call_soon_threadsafe(mic_queue.put_nowait, pcm)

    frame_samples = SAMPLE_RATE * VAD_FRAME_MS // 1000
    buf = b""

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=frame_samples,
        callback=mic_callback,
    ):
        while True:
            chunk = await mic_queue.get()
            buf += chunk

            while len(buf) >= vad.frame_bytes:
                frame, buf = buf[: vad.frame_bytes], buf[vad.frame_bytes :]
                utterance = vad.process_frame(frame)

                if utterance is not None:
                    print("[harness] utterance detected — sending to model...")
                    await adapter.send_audio(utterance)

                    response_audio = b""
                    async for chunk in adapter.stream_response():
                        response_audio += chunk

                    if response_audio:
                        print("[harness] playing response...")
                        await play_audio(response_audio, SAMPLE_RATE)


async def main() -> None:
    args = parse_args()
    try:
        await run(args.adapter, args.structure)
    except KeyboardInterrupt:
        print("\n[harness] stopping...")
    except RuntimeError as e:
        print(f"\n[error] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
