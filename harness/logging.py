import json
import os
import time
from pathlib import Path


class RunLogger:
    """
    Writes one JSONL file per session run to LOG_DIR.

    Each line is a timestamped event:
      {"t": <unix>, "event": <name>, "adapter": <name>, ...extra_fields}

    Key events and their extra fields:
      run_start      adapter, run_id, vad_aggressiveness, vad_silence_ms
      utterance_sent bytes
      first_audio    ttfa_ms   (time-to-first-audio in milliseconds)
      response_done  total_ms  (utterance_sent → last audio byte)
      run_end        (no extras)
    """

    def __init__(self, run_id: str, adapter: str, **run_meta):
        self.run_id = run_id
        self.adapter = adapter
        self._utterance_t: float = 0.0

        log_dir = Path(os.environ.get("LOG_DIR", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self._f = open(log_dir / f"{run_id}.jsonl", "a")
        self._log("run_start", run_id=run_id, adapter=adapter, **run_meta)

    def utterance_sent(self, byte_count: int) -> None:
        self._utterance_t = time.time()
        self._log("utterance_sent", bytes=byte_count)

    def first_audio(self) -> None:
        ttfa_ms = round((time.time() - self._utterance_t) * 1000)
        self._log("first_audio", ttfa_ms=ttfa_ms)

    def response_done(self) -> None:
        total_ms = round((time.time() - self._utterance_t) * 1000)
        self._log("response_done", total_ms=total_ms)

    def close(self) -> None:
        self._log("run_end")
        self._f.close()

    def _log(self, event: str, **kwargs) -> None:
        record = {"t": time.time(), "event": event, "adapter": self.adapter, **kwargs}
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()
