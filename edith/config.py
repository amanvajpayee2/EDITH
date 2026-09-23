from dataclasses import dataclass
import os
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    openrouter_api_key: str
    openrouter_model: str
    groq_stt_model: str
    wakeword_engine: str
    wakeword_model_path: str
    wakeword_threshold: float
    wakeword_test_mode: bool
    sample_rate: int
    channels: int
    silence_seconds: float
    max_recording_seconds: float
    follow_up_seconds: float
    follow_up_listen_timeout_seconds: float
    min_voiced_frames: int
    min_audio_rms: float
    playback_cooldown_seconds: float
    microphone_device: Optional[int]
    google_drive_credentials_file: str
    google_drive_token_file: str
    google_drive_folder: str
    reminder_advance_minutes: int
    reminder_interval_seconds: int
    daily_prompt_hour: int
    daily_prompt_minute: int
    daily_prompt_retry_minutes: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            openrouter_model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
            groq_stt_model=os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo"),
            wakeword_engine=os.getenv("WAKEWORD_ENGINE", "openwakeword").strip().lower(),
            wakeword_model_path=os.getenv("WAKEWORD_MODEL_PATH", "models/edith.onnx"),
            wakeword_threshold=float(os.getenv("WAKEWORD_THRESHOLD", "0.5")),
            wakeword_test_mode=_optional_bool(os.getenv("WAKEWORD_TEST_MODE")),
            sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", "16000")),
            channels=int(os.getenv("AUDIO_CHANNELS", "1")),
            silence_seconds=float(os.getenv("SILENCE_SECONDS", "1.8")),
            max_recording_seconds=float(os.getenv("MAX_RECORDING_SECONDS", "15")),
            follow_up_seconds=float(os.getenv("FOLLOW_UP_SECONDS", "30")),
            follow_up_listen_timeout_seconds=float(
                os.getenv("FOLLOW_UP_LISTEN_TIMEOUT_SECONDS", "3")
            ),
            min_voiced_frames=int(os.getenv("MIN_VOICED_FRAMES", "3")),
            min_audio_rms=float(os.getenv("MIN_AUDIO_RMS", "300")),
            playback_cooldown_seconds=float(os.getenv("PLAYBACK_COOLDOWN_SECONDS", "0.5")),
            microphone_device=_optional_int(os.getenv("MICROPHONE_DEVICE")),
            google_drive_credentials_file=os.getenv(
                "GOOGLE_DRIVE_CREDENTIALS_FILE", "credentials.json"
            ),
            google_drive_token_file=os.getenv(
                "GOOGLE_DRIVE_TOKEN_FILE", ".edith-drive-token.json"
            ),
            google_drive_folder=os.getenv("GOOGLE_DRIVE_FOLDER", "EDITH"),
            reminder_advance_minutes=int(os.getenv("REMINDER_ADVANCE_MINUTES", "15")),
            reminder_interval_seconds=int(os.getenv("REMINDER_INTERVAL_SECONDS", "15")),
            daily_prompt_hour=int(os.getenv("DAILY_PROMPT_HOUR", "7")),
            daily_prompt_minute=int(os.getenv("DAILY_PROMPT_MINUTE", "0")),
            daily_prompt_retry_minutes=int(
                os.getenv("DAILY_PROMPT_RETRY_MINUTES", "15")
            ),
        )

    def require_cloud_keys(self) -> None:
        missing = []
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if not self.openrouter_api_key:
            missing.append("OPENROUTER_API_KEY")
        if missing:
            raise RuntimeError("Missing environment variables: " + ", ".join(missing))


def _optional_int(value: Optional[str]) -> Optional[int]:
    return int(value) if value else None


def _optional_bool(value: Optional[str]) -> bool:
    if not value:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}
