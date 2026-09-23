"""Explicit, opt-in visitor recording with ephemeral local files."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import time
from os import close
import shutil
import subprocess
import threading
import wave
from typing import Any, Callable


class VisitorRecorder:
    def __init__(
        self,
        storage: Any,
        collection: str,
        max_duration_seconds: float,
        announcement: str,
        retention_metadata: str = "",
        speak: Callable[[str], None] | None = None,
        cv2_module: Any | None = None,
        writer_factory: Callable[[str, Any, float, tuple[int, int]], Any] | None = None,
        fps: float = 10.0,
        audio_enabled: bool = False,
        audio_device: int | None = None,
        audio_sample_rate: int = 16000,
        audio_codec: str = "aac",
        ffmpeg_path: str = "ffmpeg",
        sounddevice_module: Any | None = None,
        subprocess_run: Callable[..., Any] | None = None,
    ) -> None:
        self.storage = storage
        self.collection = collection
        self.max_duration_seconds = max(1.0, max_duration_seconds)
        self.announcement = announcement
        self.retention_metadata = retention_metadata
        self.speak = speak or (lambda _: None)
        self._cv2 = cv2_module
        self._writer_factory = writer_factory
        self.fps = fps
        self.audio_enabled = audio_enabled
        self.audio_device = audio_device
        self.audio_sample_rate = audio_sample_rate
        self.audio_codec = audio_codec
        self.ffmpeg_path = ffmpeg_path
        self._sounddevice = sounddevice_module
        self._subprocess_run = subprocess_run or subprocess.run
        self._writer = None
        self._path: Path | None = None
        self._audio_path: Path | None = None
        self._audio_stream = None
        self._audio_file = None
        self._audio_lock = threading.Lock()
        self._started_at = 0.0
        self._frame_size: tuple[int, int] | None = None
        self.recording = False
        self._owner_seen = False

    def process(self, frame: Any | None, present: bool, identity: str | None) -> None:
        if identity == "owner":
            self.stop("owner confirmed")
            self._owner_seen = True
            return
        if not present:
            self.stop("room empty")
            self._owner_seen = False
            return
        if self._owner_seen:
            return
        if not self.recording:
            self._start(frame)
        if self.recording and frame is not None:
            self._writer.write(frame)
        if self.recording and time.monotonic() - self._started_at >= self.max_duration_seconds:
            self.stop("maximum duration reached")

    def _start(self, frame: Any) -> None:
        if frame is None:
            return
        try:
            height, width = frame.shape[:2]
            self._frame_size = (int(width), int(height))
            cv2 = self._cv2
            if cv2 is None:
                import cv2 as cv2_module
                cv2 = cv2_module
            file_descriptor, filename = tempfile.mkstemp(
                prefix=".edith-visitor-", suffix=".mp4"
            )
            close(file_descriptor)
            path = Path(filename)
            if self._writer_factory:
                writer = self._writer_factory(str(path), cv2, self.fps, self._frame_size)
            else:
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self._frame_size
                )
            if not writer or hasattr(writer, "isOpened") and not writer.isOpened():
                path.unlink(missing_ok=True)
                print("EDITH visitor recording unavailable: video codec could not be opened")
                return
            self._path, self._writer, self._started_at = path, writer, time.monotonic()
            self.recording = True
            if self.audio_enabled:
                self._start_audio()
            print("EDITH visitor recording started.")
            self.speak(self.announcement)
        except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as error:
            print(f"EDITH visitor recording unavailable: {error}")

    def _start_audio(self) -> None:
        try:
            sd = self._sounddevice
            if sd is None:
                import sounddevice as sd_module
                sd = sd_module
            descriptor, filename = tempfile.mkstemp(
                prefix=".edith-visitor-audio-", suffix=".wav"
            )
            close(descriptor)
            path = Path(filename)
            audio_file = wave.open(str(path), "wb")
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(self.audio_sample_rate)

            def callback(indata: Any, frames: int, callback_time: Any, status: Any) -> None:
                if status:
                    print(f"EDITH visitor audio warning: {status}")
                with self._audio_lock:
                    if self._audio_file is not None:
                        self._audio_file.writeframes(indata.tobytes())

            stream = sd.InputStream(
                samplerate=self.audio_sample_rate,
                channels=1,
                dtype="int16",
                device=self.audio_device,
                callback=callback,
            )
            stream.start()
            self._audio_path, self._audio_file, self._audio_stream = path, audio_file, stream
        except (ImportError, OSError, RuntimeError, ValueError, wave.Error) as error:
            print(f"EDITH visitor audio unavailable; continuing video-only: {error}")
            if "audio_file" in locals():
                audio_file.close()
            if "path" in locals():
                path.unlink(missing_ok=True)

    def _stop_audio(self) -> None:
        stream, audio_file = self._audio_stream, self._audio_file
        self._audio_stream = self._audio_file = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except (OSError, RuntimeError):
                pass
        if audio_file is not None:
            with self._audio_lock:
                audio_file.close()

    def _ffmpeg_command(self, video: Path, audio: Path, output: Path) -> list[str]:
        return [
            self.ffmpeg_path, "-y", "-i", str(video), "-i", str(audio),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
            "-c:a", self.audio_codec, "-shortest", str(output),
        ]

    def _mux_audio(self, video: Path, audio: Path) -> Path:
        output = video.with_name(video.stem + "-muxed.mp4")
        command = self._ffmpeg_command(video, audio, output)
        executable = self.ffmpeg_path if Path(self.ffmpeg_path).exists() else shutil.which(self.ffmpeg_path)
        if not executable:
            raise FileNotFoundError(f"ffmpeg executable not found: {self.ffmpeg_path}")
        self._subprocess_run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError("ffmpeg produced no output")
        return output

    def stop(self, reason: str) -> None:
        if not self.recording:
            return
        writer, path, audio_path = self._writer, self._path, self._audio_path
        self.recording = False
        self._writer = self._path = self._audio_path = None
        try:
            try:
                writer.release()
            finally:
                self._stop_audio()
            if path is None or not path.exists() or path.stat().st_size == 0:
                print(f"EDITH visitor recording discarded ({reason}; no video).")
                return
            upload_path = path
            if audio_path is not None and audio_path.exists() and audio_path.stat().st_size:
                try:
                    upload_path = self._mux_audio(path, audio_path)
                except (OSError, RuntimeError, ValueError) as error:
                    print(f"EDITH visitor audio mux unavailable; uploading video-only: {error}")
            name = f"visitor-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.mp4"
            try:
                self.storage.upload_binary(
                    self.collection,
                    name,
                    upload_path,
                    "video/mp4",
                    metadata={                    "retention": self.retention_metadata or "visitor recording", "reason": reason},
                )
                print(f"EDITH visitor recording uploaded to Drive: {name}")
            except (OSError, RuntimeError, ValueError) as error:
                print(f"EDITH visitor recording upload failed; local file deleted: {error}")
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
            if audio_path is not None:
                audio_path.unlink(missing_ok=True)
            if path is not None:
                path.with_name(path.stem + "-muxed.mp4").unlink(missing_ok=True)
