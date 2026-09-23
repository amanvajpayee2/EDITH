from __future__ import annotations

from collections.abc import Iterator
from collections import deque
import time

import numpy as np
import sounddevice as sd
import webrtcvad


class AudioInput:
    def __init__(self, sample_rate: int, channels: int, device: int | None) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device

    def frames(self, frame_ms: int = 30) -> Iterator[np.ndarray]:
        frame_size = self.sample_rate * frame_ms // 1000
        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=frame_size,
            channels=self.channels,
            dtype="int16",
            device=self.device,
        ) as stream:
            while True:
                data, _ = stream.read(frame_size)
                yield np.frombuffer(data, dtype=np.int16).copy()

    def utterance(
        self,
        silence_seconds: float,
        max_seconds: float,
        start_timeout_seconds: float | None = None,
        min_voiced_frames: int = 3,
        min_rms: float = 300.0,
    ) -> bytes:
        vad = webrtcvad.Vad(2)
        frames = []
        pre_roll = deque(maxlen=10)
        silence_frames = 0
        frame_ms = 30
        max_frames = int(max_seconds * 1000 / frame_ms)
        required_silence = max(1, int(silence_seconds * 1000 / frame_ms))
        start_timeout_frames = (
            int(start_timeout_seconds * 1000 / frame_ms)
            if start_timeout_seconds is not None
            else None
        )
        started = False
        voiced_frames = 0

        for index, frame in enumerate(self.frames(frame_ms)):
            frame_bytes = frame.tobytes()
            pre_roll.append(frame_bytes)
            rms = float(np.sqrt(np.mean(np.square(frame.astype(np.float32)))))
            voiced = vad.is_speech(frame_bytes, self.sample_rate) and rms >= min_rms
            if voiced:
                voiced_frames += 1
                if not started:
                    if voiced_frames >= min_voiced_frames:
                        frames.extend(pre_roll)
                        started = True
                        silence_frames = 0
                else:
                    silence_frames = 0
            elif started:
                silence_frames += 1
            else:
                voiced_frames = 0
            if started:
                frames.append(frame_bytes)
            if started and silence_frames >= required_silence:
                break
            if not started and start_timeout_frames is not None and index >= start_timeout_frames:
                break
            if index >= max_frames:
                break
        return b"".join(frames)


class PlaybackGuard:
    def __init__(self, cooldown_seconds: float) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._blocked_until = 0.0

    def block(self, duration_seconds: float) -> None:
        self._blocked_until = time.monotonic() + duration_seconds + self.cooldown_seconds

    @property
    def blocked(self) -> bool:
        return time.monotonic() < self._blocked_until
