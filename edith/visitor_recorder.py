"""Explicit, opt-in visitor recording with ephemeral local files."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
from os import close
import queue
import re
import shutil
import subprocess
import threading
import wave
import time
from typing import Any, Callable
import threading


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
        transcribe_audio: Callable[[str], str] | None = None,
    ) -> None:
        self.storage = storage
        self.collection = collection
        self.max_duration_seconds = max(0.0, max_duration_seconds)
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
        self._video_queue: queue.Queue[Any | None] | None = None
        self._video_writer_thread: threading.Thread | None = None
        self._path: Path | None = None
        self._audio_path: Path | None = None
        self._audio_stream = None
        self._audio_file = None
        self._audio_lock = threading.Lock()
        self._audio_queue: queue.Queue[bytes | None] | None = None
        self._audio_writer_thread: threading.Thread | None = None
        self._started_at = 0.0
        self._frame_size: tuple[int, int] | None = None
        self.recording = False
        self._owner_seen = False
        self._owner_absence_frames = 0
        self._owner_reset_after_frames = 3
        self._interview_stage: str | None = None
        self._visitor_name: str | None = None
        self._visitor_reason: str | None = None
        self._visit_started_at: datetime | None = None
        self._state_lock = threading.Lock()
        self._transcribe_audio = transcribe_audio
        self._pending_dir = Path(".edith-pending-visitor")

    def process(self, frame: Any | None, present: bool, identity: str | None) -> None:
        if identity == "owner":
            self.stop("owner confirmed")
            self._owner_seen = True
            self._owner_absence_frames = 0
            return
        if not present:
            self.stop("room empty")
            self._owner_seen = False
            self._owner_absence_frames = 0
            return
        if self._owner_seen:
            self._owner_absence_frames += 1
            if self._owner_absence_frames < self._owner_reset_after_frames:
                return
            self._owner_seen = False
            self._owner_absence_frames = 0
        if not self.recording:
            self._start(frame)
        if self.recording and frame is not None:
            if self._video_queue is not None:
                self._video_queue.put(frame)
            else:
                self._writer.write(frame)
        if (
            self.recording
            and self.max_duration_seconds > 0
            and time.monotonic() - self._started_at >= self.max_duration_seconds
        ):
            self.stop("maximum duration reached")

    def set_capture_fps(self, fps: float) -> None:
        """Use the camera's measured rate so playback stays real-time."""
        if fps > 0:
            self.fps = max(1.0, float(fps))

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
            self._video_queue = queue.Queue()
            self._video_writer_thread = threading.Thread(
                target=self._write_video_frames,
                name="edith-visitor-video",
                daemon=True,
            )
            self._video_writer_thread.start()
            with self._state_lock:
                self._interview_stage = "name"
                self._visitor_name = None
                self._visitor_reason = None
                self._visit_started_at = datetime.now().astimezone()
            if self.audio_enabled:
                self._start_audio()
            print("EDITH visitor recording started.")
            self.speak(self.announcement)
            self.speak("Please tell me your name.")
        except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as error:
            print(f"EDITH visitor recording unavailable: {error}")

    def _write_video_frames(self) -> None:
        video_queue = self._video_queue
        writer = self._writer
        if video_queue is None or writer is None:
            return
        while True:
            frame = video_queue.get()
            if frame is None:
                return
            writer.write(frame)

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
            audio_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=128)

            def callback(indata: Any, frames: int, callback_time: Any, status: Any) -> None:
                if status:
                    if getattr(status, "input_overflow", False):
                        print("EDITH visitor audio warning: input overflow; dropping delayed audio block")
                    else:
                        print(f"EDITH visitor audio warning: {status}")
                try:
                    audio_queue.put_nowait(indata.tobytes())
                except queue.Full:
                    pass

            def write_audio() -> None:
                while True:
                    block = audio_queue.get()
                    if block is None:
                        return
                    with self._audio_lock:
                        if self._audio_file is not None:
                            self._audio_file.writeframes(block)

            stream = sd.InputStream(
                samplerate=self.audio_sample_rate,
                channels=1,
                dtype="int16",
                device=self.audio_device,
                callback=callback,
            )
            self._audio_queue = audio_queue
            self._audio_writer_thread = threading.Thread(
                target=write_audio, name="edith-visitor-audio", daemon=True
            )
            self._audio_writer_thread.start()
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
        writer_thread = self._audio_writer_thread
        audio_queue = self._audio_queue
        self._audio_stream = self._audio_file = None
        self._audio_writer_thread = self._audio_queue = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except (OSError, RuntimeError):
                pass
        if audio_queue is not None:
            audio_queue.put(None)
        if writer_thread is not None:
            writer_thread.join(timeout=2.0)
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
                video_queue = self._video_queue
                video_thread = self._video_writer_thread
                self._video_queue = self._video_writer_thread = None
                if video_queue is not None:
                    video_queue.put(None)
                if video_thread is not None:
                    video_thread.join(timeout=10.0)
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
                self._upload_with_retry(upload_path, name, reason)
                print(f"EDITH visitor recording uploaded to Drive: {name}")
            except (OSError, RuntimeError, ValueError) as error:
                pending = self._preserve_failed_upload(upload_path, name)
                print(
                    "EDITH visitor recording upload failed; preserved for retry at "
                    f"{pending}: {error}"
                )
        finally:
            self._save_visit_summary(audio_path)
            if path is not None:
                path.unlink(missing_ok=True)
            if audio_path is not None:
                audio_path.unlink(missing_ok=True)
            if path is not None:
                path.with_name(path.stem + "-muxed.mp4").unlink(missing_ok=True)

    def stop_async(self, reason: str) -> None:
        threading.Thread(
                target=self.stop,
                args=(reason,),
                name="edith-visitor-finalize",
                daemon=True,
        ).start()

    def _upload_with_retry(self, upload_path: Path, name: str, reason: str) -> None:
        error: Exception | None = None
        for attempt in range(3):
            try:
                self.storage.upload_binary(
                    self.collection,
                    name,
                    upload_path,
                    "video/mp4",
                    metadata={
                        "retention": self.retention_metadata or "visitor recording",
                        "reason": reason,
                    },
                )
                return
            except (OSError, RuntimeError, ValueError) as caught:
                error = caught
                if attempt < 2:
                    time.sleep(2 ** attempt)
        assert error is not None
        raise error

    def _preserve_failed_upload(self, upload_path: Path, name: str) -> Path:
        self._pending_dir.mkdir(parents=True, exist_ok=True)
        pending = self._pending_dir / name
        upload_path.replace(pending)
        return pending

    def retry_pending_uploads(self) -> None:
        if not self._pending_dir.exists():
            return
        for pending in sorted(self._pending_dir.glob("visitor-*.mp4")):
            try:
                self._upload_with_retry(
                    pending,
                    pending.name,
                    "retry of previously failed visitor upload",
                )
                pending.unlink(missing_ok=True)
                print(f"EDITH pending visitor recording uploaded to Drive: {pending.name}")
            except (OSError, RuntimeError, ValueError) as error:
                print(f"EDITH pending visitor upload still unavailable: {error}")

    @property
    def interview_active(self) -> bool:
        with self._state_lock:
            return self.recording and self._interview_stage is not None

    def accept_interview_answer(self, answer: str) -> str | None:
        answer = answer.strip()
        if not answer:
            return "I did not hear that. Please answer briefly."
        with self._state_lock:
            if self._interview_stage == "name":
                self._visitor_name = answer
                self._interview_stage = "reason"
                return "Thank you. Why are you in this room?"
            if self._interview_stage == "reason":
                self._visitor_reason = answer
                self._interview_stage = None
                return "Thank you. Your answer has been recorded."
        return None

    def _save_visit_summary(self, audio_path: Path | None) -> None:
        with self._state_lock:
            started = self._visit_started_at
            name = self._visitor_name or "unknown visitor"
            reason = self._visitor_reason or "no reason provided"
            self._interview_stage = None
            self._visit_started_at = None
        if started is None:
            return
        transcript = ""
        if audio_path is not None and audio_path.exists() and self._transcribe_audio:
            try:
                transcript = self._transcribe_audio(str(audio_path))
            except (OSError, RuntimeError, ValueError) as error:
                print(f"EDITH visitor conversation transcription unavailable: {error}")
        if transcript:
            transcript = self._clean_transcript(transcript)
        if not self._visitor_name and transcript:
            name = "visitor"
        if not self._visitor_reason:
            reason = (
                "conversation could not be transcribed"
                if self._transcribe_audio and not transcript
                else "see recorded conversation"
            )
        try:
            summaries = self.storage.read_json(
                "metadata", "visitor-summaries.json", {"visits": []}
            )
            visits = summaries.get("visits", [])
            visits.append({
                "name": name,
                "reason": reason,
                "conversation": transcript,
                "entered_at": started.isoformat(),
                "reported_at": datetime.now().astimezone().isoformat(),
            })
            self.storage.write_json(
                "metadata", "visitor-summaries.json", {"visits": visits[-50:]}
            )
        except (OSError, RuntimeError, ValueError, AttributeError) as error:
            print(f"EDITH visitor summary unavailable: {error}")

    def _clean_transcript(self, transcript: str) -> str:
        cleaned = transcript
        phrases = [
            self.announcement,
            "recording has started",
            "reporting has started",
            "please tell me your name",
            "thank you why are you in this room",
            "thank you your answer has been recorded",
        ]
        for phrase in phrases:
            pattern = re.escape(phrase).replace(r"\ ", r"\s+")
            cleaned = re.sub(
                r"\b" + pattern + r"\b",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
        return re.sub(r"\s+", " ", cleaned).strip(" .,")
