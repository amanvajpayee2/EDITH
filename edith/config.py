from dataclasses import dataclass
import os
from typing import Optional, Tuple
from datetime import time

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
    camera_presence_enabled: bool
    camera_index: int
    camera_sample_interval_seconds: float
    camera_quiet_start: Optional[time]
    camera_quiet_end: Optional[time]
    camera_min_consecutive_detections: int
    camera_min_consecutive_absence: int
    owner_verification_enabled: bool
    owner_encoding_file: str
    owner_return_after_minutes: int
    owner_face_tolerance: float
    visitor_recording_enabled: bool
    visitor_drive_collection: str
    visitor_max_session_duration_seconds: float
    visitor_recording_announcement: str
    visitor_retention_metadata: str
    visitor_audio_enabled: bool
    visitor_audio_device: Optional[int]
    visitor_audio_sample_rate: int
    visitor_audio_codec: str
    visitor_ffmpeg_path: str
    study_observer_enabled: bool
    study_desk_region: Tuple[float, float, float, float]
    study_bed_region: Tuple[float, float, float, float]
    study_motion_threshold: float
    study_min_confidence: float
    study_slot_title_patterns: tuple[str, ...]

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
            camera_presence_enabled=_optional_bool(os.getenv("CAMERA_PRESENCE_ENABLED")),
            camera_index=int(os.getenv("CAMERA_INDEX", "0")),
            camera_sample_interval_seconds=float(
                os.getenv("CAMERA_SAMPLE_INTERVAL_SECONDS", "5")
            ),
            camera_quiet_start=_optional_time(os.getenv("CAMERA_QUIET_START")),
            camera_quiet_end=_optional_time(os.getenv("CAMERA_QUIET_END")),
            camera_min_consecutive_detections=int(
                os.getenv("CAMERA_MIN_CONSECUTIVE_DETECTIONS", "2")
            ),
            camera_min_consecutive_absence=int(
                os.getenv("CAMERA_MIN_CONSECUTIVE_ABSENCE", "3")
            ),
            owner_verification_enabled=_optional_bool(os.getenv("OWNER_VERIFICATION_ENABLED")),
            owner_encoding_file=os.getenv("OWNER_ENCODING_FILE", ".edith-owner.json"),
            owner_return_after_minutes=int(os.getenv("OWNER_RETURN_AFTER_MINUTES", "30")),
            owner_face_tolerance=float(os.getenv("OWNER_FACE_TOLERANCE", "0.48")),
            visitor_recording_enabled=_optional_bool(os.getenv("VISITOR_RECORDING_ENABLED")),
            visitor_drive_collection=os.getenv("VISITOR_DRIVE_COLLECTION", "visitor-recordings"),
            visitor_max_session_duration_seconds=float(os.getenv("VISITOR_MAX_SESSION_DURATION_SECONDS", "120")),
            visitor_recording_announcement=os.getenv(
                "VISITOR_RECORDING_ANNOUNCEMENT",
                "This room is monitored. Recording has started.",
            ),
            visitor_retention_metadata=os.getenv(
                "VISITOR_RETENTION_METADATA", "Delete visitor recordings when no longer needed."
            ),
            visitor_audio_enabled=_optional_bool(os.getenv("VISITOR_AUDIO_ENABLED")),
            visitor_audio_device=_optional_int(os.getenv("VISITOR_AUDIO_DEVICE")),
            visitor_audio_sample_rate=int(os.getenv("VISITOR_AUDIO_SAMPLE_RATE", "16000")),
            visitor_audio_codec=os.getenv("VISITOR_AUDIO_CODEC", "aac"),
            visitor_ffmpeg_path=os.getenv("VISITOR_FFMPEG_PATH", "ffmpeg"),
            study_observer_enabled=_optional_bool(os.getenv("STUDY_OBSERVER_ENABLED")),
            study_desk_region=_region(os.getenv("STUDY_DESK_REGION", "0,0.5,0.5,0.5")),
            study_bed_region=_region(os.getenv("STUDY_BED_REGION", "0.5,0.5,0.5,0.5")),
            study_motion_threshold=float(os.getenv("STUDY_MOTION_THRESHOLD", "0.08")),
            study_min_confidence=float(os.getenv("STUDY_MIN_CONFIDENCE", "0.5")),
            study_slot_title_patterns=tuple(
                item.strip().lower() for item in os.getenv(
                    "STUDY_SLOT_TITLE_PATTERNS", "study,studying,leetcode,revision"
                ).split(",") if item.strip()
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


def _optional_time(value: Optional[str]) -> Optional[time]:
    if not value:
        return None
    hour, minute = (int(part) for part in value.split(":", 1))
    return time(hour, minute)


def _region(value: str) -> Tuple[float, float, float, float]:
    parts = tuple(float(part.strip()) for part in value.split(","))
    if len(parts) != 4:
        raise ValueError("study regions must be x,y,width,height")
    return parts
