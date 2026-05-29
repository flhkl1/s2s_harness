from __future__ import annotations

import collections

import webrtcvad


class VADStream:
    """
    Wraps webrtcvad to detect end-of-speech in a stream of PCM frames.

    frame_duration_ms: must be 10, 20, or 30 (webrtcvad requirement)
    sample_rate: must be 8000, 16000, 32000, or 48000
    aggressiveness: 0 (least aggressive) to 3 (most aggressive)
    silence_threshold_ms: ms of silence before declaring end-of-speech
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
        aggressiveness: int = 2,
        silence_threshold_ms: int = 500,
    ):
        self._vad = webrtcvad.Vad(aggressiveness)
        self._sample_rate = sample_rate
        self._frame_duration_ms = frame_duration_ms
        self._frame_bytes = int(sample_rate * frame_duration_ms / 1000) * 2  # 16-bit

        silence_frames = silence_threshold_ms // frame_duration_ms
        self._ring = collections.deque(maxlen=silence_frames)
        self._triggered = False
        self._voiced_frames: list[bytes] = []

    @property
    def frame_bytes(self) -> int:
        return self._frame_bytes

    def process_frame(self, frame: bytes) -> bytes | None:
        """
        Feed one PCM frame. Returns the complete utterance bytes when
        end-of-speech is detected, otherwise None.
        """
        is_speech = self._vad.is_speech(frame, self._sample_rate)

        if not self._triggered:
            self._ring.append((frame, is_speech))
            num_voiced = sum(1 for _, s in self._ring if s)
            if num_voiced > 0.9 * self._ring.maxlen:
                self._triggered = True
                self._voiced_frames = [f for f, _ in self._ring]
                self._ring.clear()
        else:
            self._voiced_frames.append(frame)
            self._ring.append((frame, is_speech))
            num_unvoiced = sum(1 for _, s in self._ring if not s)
            if num_unvoiced == self._ring.maxlen:
                utterance = b"".join(self._voiced_frames)
                self._triggered = False
                self._voiced_frames = []
                self._ring.clear()
                return utterance

        return None
