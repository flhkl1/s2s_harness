"""
S2S harness — public WebSocket API server.

Endpoints:
  GET  /health              Railway health check
  GET  /runs                list completed run IDs
  WS   /ws/{adapter}        real-time S2S session

WebSocket protocol:
  Client → server: raw 16 kHz PCM int16 chunks (any size)
  Server → client: raw 16 kHz PCM int16 chunks of model audio output

Query parameters on /ws/{adapter}:
  mode      "stream" (default) — full-duplex, no turn-taking, model and user
                                  can speak simultaneously and interrupt each other
            "turn"             — VAD-gated: harness waits for silence before
                                  sending to model, discards mic while model speaks
  run_id    optional; UUID4 generated if omitted
  structure optional path to YAML conversation structure

Usage:
  uvicorn harness.server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from harness import config
from harness.adapters import ADAPTERS
from harness.logging import RunLogger
from harness.sessions.session import Session

app = FastAPI(title="S2S Comparison Harness")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/runs")
async def list_runs():
    log_dir = Path(os.environ.get("LOG_DIR", "logs"))
    if not log_dir.exists():
        return []
    return sorted(f.stem for f in log_dir.glob("*.jsonl"))


@app.websocket("/ws/{adapter_name}")
async def ws_session(websocket: WebSocket, adapter_name: str):
    await websocket.accept()

    if adapter_name not in ADAPTERS:
        await websocket.close(code=1008, reason=f"unknown adapter: {adapter_name}")
        return

    mode = websocket.query_params.get("mode", "stream")
    run_id = websocket.query_params.get("run_id", str(uuid.uuid4()))
    structure_path = websocket.query_params.get(
        "structure", str(config.DEFAULT_STRUCTURE)
    )

    session = Session.from_yaml(structure_path)
    system_prompt = session.build_system_prompt()

    full_duplex = (mode != "turn")
    adapter = ADAPTERS[adapter_name](full_duplex=full_duplex)
    logger = RunLogger(
        run_id=run_id,
        adapter=adapter_name,
        mode=mode,
        vad_aggressiveness=config.VAD_AGGRESSIVENESS,
        vad_silence_ms=config.VAD_SILENCE_MS,
    )

    try:
        await adapter.connect(system_prompt)
        if full_duplex:
            await _run_stream(websocket, adapter, logger)
        else:
            await _run_turn(websocket, adapter, logger)
    except WebSocketDisconnect:
        pass
    finally:
        await adapter.close()
        logger.close()


async def _run_stream(websocket: WebSocket, adapter, logger: RunLogger) -> None:
    """
    Full-duplex mode — mic audio flows continuously to the model and model
    audio flows continuously back. Either side can speak at any time.
    The model hears you while it's talking and can adjust (barge-in).
    """

    async def mic_to_model() -> None:
        try:
            while True:
                chunk = await websocket.receive_bytes()
                await adapter.send_audio(chunk)
        except WebSocketDisconnect:
            pass

    async def model_to_client() -> None:
        async for chunk in adapter.stream_response():
            try:
                await websocket.send_bytes(chunk)
            except (WebSocketDisconnect, RuntimeError):
                break

    mic_task = asyncio.create_task(mic_to_model())
    model_task = asyncio.create_task(model_to_client())

    # Run until the client disconnects (mic_task finishes first)
    done, pending = await asyncio.wait(
        [mic_task, model_task], return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()


async def _run_turn(websocket: WebSocket, adapter, logger: RunLogger) -> None:
    """
    Turn-based mode — VAD waits for end-of-utterance before sending to model.
    Mic input is discarded while the model is responding.
    Useful for clean per-utterance latency measurement.
    """
    frame_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def receive_frames() -> None:
        try:
            while True:
                await frame_queue.put(await websocket.receive_bytes())
        except WebSocketDisconnect:
            pass
        finally:
            await frame_queue.put(None)

    from harness.sessions.vad import VADStream  # lazy: webrtcvad only needed in turn mode
    vad = VADStream(
        sample_rate=config.SAMPLE_RATE,
        frame_duration_ms=config.VAD_FRAME_MS,
        aggressiveness=config.VAD_AGGRESSIVENESS,
        silence_threshold_ms=config.VAD_SILENCE_MS,
    )

    receive_task = asyncio.create_task(receive_frames())
    buf = b""
    responding = False

    try:
        while True:
            data = await frame_queue.get()
            if data is None:
                break
            if responding:
                continue

            buf += data
            while len(buf) >= vad.frame_bytes:
                frame, buf = buf[: vad.frame_bytes], buf[vad.frame_bytes :]
                utterance = vad.process_frame(frame)
                if utterance is None:
                    continue

                responding = True
                logger.utterance_sent(len(utterance))
                await adapter.send_audio(utterance)

                first = True
                async for chunk in adapter.stream_response():
                    if first:
                        logger.first_audio()
                        first = False
                    await websocket.send_bytes(chunk)

                logger.response_done()
                responding = False
    finally:
        receive_task.cancel()
