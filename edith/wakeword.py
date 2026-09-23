from __future__ import annotations

from pathlib import Path
import re
from typing import Iterator

import numpy as np

_WAKE_COMMAND_PATTERN = re.compile(r"^\s*hey(?:[\s,:-]+|$)", re.IGNORECASE)


class WakeWordDetector:
    """Local wake-word detector. Audio never leaves the device here."""

    def __init__(self, engine: str, model_path: str, threshold: float) -> None:
        self.engine = engine.lower()
        self.threshold = threshold
        self._detector = self._load_detector(model_path)

    def _load_detector(self, model_path: str):
        if self.engine == "openwakeword":
            from openwakeword.model import Model

            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"Wake-word model not found: {path}. Set WAKEWORD_MODEL_PATH to an "
                    "openWakeWord .onnx model for EDITH."
                )
            return Model(wakeword_models=[str(path)], inference_framework="onnx")
        if self.engine == "porcupine":
            import pvporcupine

            access_key = __import__("os").getenv("PICOVOICE_ACCESS_KEY")
            if not access_key:
                raise RuntimeError("PICOVOICE_ACCESS_KEY is required for Porcupine")
            return pvporcupine.create(access_key=access_key, keyword_paths=[model_path])
        raise ValueError("WAKEWORD_ENGINE must be 'openwakeword' or 'porcupine'")

    def detected(self, audio_frames: Iterator[np.ndarray]) -> bool:
        if self.engine == "openwakeword":
            for frame in audio_frames:
                scores = self._detector.predict(frame.astype(np.int16))
                if any(float(score) >= self.threshold for score in scores.values()):
                    return True
            return False

        for frame in audio_frames:
            if self._detector.process(frame.astype(np.int16).tolist()) >= 0:
                return True
        return False


def extract_wake_command(text: str) -> str | None:
    """Returns command text after wake phrase or None when no wake command exists."""
    stripped = text.strip()
    match = _WAKE_COMMAND_PATTERN.match(stripped)
    if match:
        return stripped[match.end() :].strip()

    return None


def main() -> None:
    from .audio import AudioInput
    from .config import Settings

    settings = Settings.from_env()
    detector = WakeWordDetector(
        settings.wakeword_engine,
        settings.wakeword_model_path,
        settings.wakeword_threshold,
    )
    audio = AudioInput(
        settings.sample_rate,
        settings.channels,
        settings.microphone_device,
    )
    print('Listening for "EDITH". Press Ctrl+C to stop.')
    if detector.detected(audio.frames()):
        print("Wake word detected: EDITH")


if __name__ == "__main__":
    main()
