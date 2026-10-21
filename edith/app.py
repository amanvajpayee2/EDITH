import time
import sys
import os
import threading
from datetime import timedelta
from queue import Empty, Queue
import httpx
from datetime import datetime, time as clock_time

from .audio import AudioInput, PlaybackGuard
from .cloud import GeminiChat, SpeechToText
from .config import Settings
from .speaker import Speaker
from .reminders import ReminderEngine, ReminderWorker
from .storage import GoogleDriveStorage
from .wakeword import WakeWordDetector, extract_wake_command
from .presence import CameraPresenceWorker, PresenceState
from .owner_verification import OwnerVerifier
from .study_observer import StudyObserver
from .visitor_recorder import VisitorRecorder


def _retry_chat_request(chat: GeminiChat, text: str, results: Queue) -> None:
    delay = 1.0
    while True:
        try:
            results.put(("success", text, chat.ask(text)))
            return
        except (RuntimeError, OSError, httpx.HTTPError) as error:
            print(
                f"EDITH Gemini retry failed; trying again in {delay:.0f}s: "
                f"{_console_text(str(error))}"
            )
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


def _console_text(value: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return value.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _visitor_return_report(storage: GoogleDriveStorage) -> str | None:
    data = storage.read_json("metadata", "visitor-summaries.json", {"visits": []})
    visits = data.get("visits", [])
    if not visits:
        return None
    reports = []
    for visit in visits:
        try:
            entered = datetime.fromisoformat(visit["entered_at"]).astimezone()
            when = entered.strftime("%I:%M %p").lstrip("0")
        except (KeyError, TypeError, ValueError):
            when = "an unknown time"
        name = visit.get("name", "unknown visitor")
        reason = visit.get("reason", "no reason provided")
        conversation = visit.get("conversation", "").strip()
        if name != "unknown visitor" and reason not in {
            "no reason provided",
            "see recorded conversation",
            "conversation could not be transcribed",
        }:
            reports.append(
                f"{name} came to the room at {when}. "
                f"When asked, they said they came because {reason}."
            )
        elif conversation:
            reports.append(
                f"a visitor came to the room at {when}. "
                f"The recorded WhisperFlow transcript was: {conversation}"
            )
        else:
            reports.append(
                f"someone reported as {name} came to the room at {when}. "
                f"The reason they gave was {reason}."
            )
    storage.write_json("metadata", "visitor-summaries.json", {"visits": []})
    return "Sir, while you were not in the room, " + " ".join(reports)


def run() -> None:
    settings = Settings.from_env()
    if settings.wakeword_test_mode:
        if not settings.groq_api_key:
            raise RuntimeError("Missing environment variables: GROQ_API_KEY")
    else:
        settings.require_cloud_keys()
    audio = AudioInput(settings.sample_rate, settings.channels, settings.microphone_device)
    stt = SpeechToText(settings.groq_api_key, settings.groq_stt_model, settings.sample_rate)
    chat = None
    reminder_engine = None
    reminder_worker = None
    storage = None
    if not settings.wakeword_test_mode:
        storage = GoogleDriveStorage(
            settings.google_drive_credentials_file,
            settings.google_drive_token_file,
            settings.google_drive_folder,
        )
        if settings.visitor_recording_enabled and settings.visitor_retention_days > 0:
            try:
                deleted = storage.purge_files_older_than(
                    settings.visitor_drive_collection,
                    timedelta(days=settings.visitor_retention_days),
                )
                if deleted:
                    print(f"EDITH removed {deleted} expired visitor recording(s) from Drive.")
            except (OSError, RuntimeError) as error:
                print(f"EDITH visitor retention cleanup unavailable: {error}")
    presence_events: Queue[str] = Queue()
    presence_state = PresenceState(
        settings.camera_presence_enabled,
        presence_events.put,
        require_owner=settings.owner_verification_enabled,
    )
    presence_state.set_return_after_seconds(settings.owner_return_after_minutes * 60)
    presence_worker = None
    visitor_recorder = None
    if (
        settings.visitor_recording_enabled
        and storage is not None
        and settings.owner_verification_enabled
    ):
        visitor_recorder = VisitorRecorder(
            storage,
            settings.visitor_drive_collection,
            settings.visitor_max_session_duration_seconds,
            settings.visitor_recording_announcement,
            settings.visitor_retention_metadata,
            lambda text: print(f"EDITH: {text}"),
            audio_enabled=settings.visitor_audio_enabled,
            audio_device=settings.visitor_audio_device,
            audio_sample_rate=settings.visitor_audio_sample_rate,
            audio_codec=settings.visitor_audio_codec,
            ffmpeg_path=settings.visitor_ffmpeg_path,
            fps=settings.visitor_video_fps,
            transcribe_audio=stt.transcribe_file,
        )
        if settings.visitor_audio_enabled:
            import shutil
            if not (
                shutil.which(settings.visitor_ffmpeg_path)
                or os.path.isfile(settings.visitor_ffmpeg_path)
            ):
                print(
                    "EDITH visitor audio disabled: FFmpeg was not found. "
                    "Install FFmpeg or set VISITOR_FFMPEG_PATH to ffmpeg.exe."
                )
                visitor_recorder.audio_enabled = False
        visitor_recorder.retry_pending_uploads()
    elif settings.visitor_recording_enabled:
        print(
            "EDITH visitor recording disabled: owner verification must be enabled "
            "to avoid recording the owner."
        )
    study_observer = (
        StudyObserver(
            settings.study_desk_region,
            settings.study_bed_region,
            settings.study_motion_threshold,
            settings.study_min_confidence,
        )
        if settings.study_observer_enabled else None
    )
    if (
        settings.camera_presence_enabled
        or study_observer is not None
        or visitor_recorder is not None
    ):
        owner_verifier = (
            OwnerVerifier(settings.owner_encoding_file, settings.owner_face_tolerance)
            if settings.owner_verification_enabled else None
        )
        if visitor_recorder is not None and (
            owner_verifier is None or not owner_verifier.available
        ):
            print(
                "EDITH visitor recording disabled: owner enrollment/backend is "
                "unavailable."
            )
            visitor_recorder = None
        presence_worker = CameraPresenceWorker(
            presence_state,
            settings.camera_index,
            settings.camera_sample_interval_seconds,
            settings.camera_quiet_start,
            settings.camera_quiet_end,
            settings.camera_min_consecutive_detections,
            settings.camera_min_consecutive_absence,
            owner_verifier=owner_verifier,
            detector_backend=settings.camera_detector,
            model_path=settings.camera_model_path,
            detection_confidence=settings.camera_detection_confidence,
            detection_iou=settings.camera_detection_iou,
            detection_image_size=settings.camera_detection_image_size,
            hog_score_threshold=settings.camera_hog_score_threshold,
            study_observer=study_observer,
            visitor_recorder=visitor_recorder,
            debug_enabled=settings.camera_debug_enabled,
            preview_enabled=settings.camera_preview_enabled,
        )
        if settings.camera_presence_enabled:
            print(
                "Camera presence is enabled "
                f"(detector={settings.camera_detector})."
            )
        if study_observer is not None:
            print("Local study observation is enabled; only aggregate estimates are retained in memory.")
        if owner_verifier is not None:
            if owner_verifier.available:
                print("Local owner verification is enabled; unrecognized people remain unknown.")
            else:
                print("Owner verification is degraded; enroll locally and install face-recognition to confirm identity.")
    playback = PlaybackGuard(settings.playback_cooldown_seconds)
    pending_notifications: Queue[str] = Queue()
    deferred_notifications: list[str] = []
    completed_retries: Queue[tuple[str, str, str]] = Queue()
    speaker = None
    if not settings.wakeword_test_mode:
        speaker = Speaker(playback)
        if visitor_recorder is not None:
            visitor_recorder.speak = speaker.speak
        assert storage is not None
        reminder_engine = ReminderEngine(
            storage,
            settings.reminder_advance_minutes,
            clock_time(settings.daily_prompt_hour, settings.daily_prompt_minute),
            settings.daily_prompt_retry_minutes,
            study_observer=study_observer,
            study_slot_title_patterns=settings.study_slot_title_patterns,
        )
        chat = GeminiChat(
            settings.gemini_api_key,
            settings.gemini_model,
            storage,
            lambda: reminder_engine.acknowledge_daily_plan(
                datetime.now().astimezone().date()
            ),
        )
        reminder_worker = ReminderWorker(
            reminder_engine,
            pending_notifications.put,
            settings.reminder_interval_seconds,
            presence_state,
        )
        reminder_worker.start()
    if presence_worker is not None:
        presence_worker.start()
    follow_up_until = 0.0
    detector = None

    if settings.wakeword_engine in {"spoken", "fallback"}:
        print('Spoken wake fallback is active. Say "HEY" followed by your command.')
    elif settings.wakeword_engine == "openwakeword":
        try:
            detector = WakeWordDetector(settings.wakeword_engine, settings.wakeword_model_path, settings.wakeword_threshold)
            print("EDITH is listening locally for the wake word.")
        except (FileNotFoundError, ImportError, OSError) as error:
            print(
                "Local wake-word backend unavailable "
                f"({_console_text(str(error))}). Falling back to spoken command mode "
                '("HEY ...").'
            )
    else:
        detector = WakeWordDetector(settings.wakeword_engine, settings.wakeword_model_path, settings.wakeword_threshold)
        print("EDITH is listening locally for the wake word.")

    return_session_announced = False
    while True:
        if playback.blocked:
            time.sleep(0.1)
            continue
        if speaker is not None:
            try:
                result, request_text, answer = completed_retries.get_nowait()
            except Empty:
                pass
            else:
                print(f"EDITH: {_console_text(answer)}")
                speaker.speak(answer)
                follow_up_until = time.monotonic() + settings.follow_up_seconds
            while True:
                try:
                    event = presence_events.get_nowait()
                except Empty:
                    break
                if event == "owner_returned":
                    return_session_announced = False
                    greeting = "Welcome back, Aman."
                    print(f"EDITH: {greeting}")
                    speaker.speak(greeting)
                    return_prompt = reminder_engine.begin_return_session()
                    if return_prompt:
                        print(f"EDITH: {_console_text(return_prompt)}")
                        speaker.speak(return_prompt)
                        return_session_announced = True
                    follow_up_until = time.monotonic() + settings.follow_up_seconds
        if (
            speaker is not None
            and storage is not None
            and presence_state.should_accept_commands
        ):
            visitor_report = _visitor_return_report(storage)
            if visitor_report:
                print(f"EDITH: {_console_text(visitor_report)}")
                speaker.speak(visitor_report)
                follow_up_until = time.monotonic() + settings.follow_up_seconds
        if reminder_engine is not None and not return_session_announced:
            return_prompt = reminder_engine.begin_return_session()
            if return_prompt:
                print(f"EDITH: {_console_text(return_prompt)}")
                speaker.speak(return_prompt)
                return_session_announced = True
        try:
            if time.monotonic() >= follow_up_until:
                while True:
                    reminder = pending_notifications.get_nowait()
                    deferred_notifications.append(reminder)
        except Empty:
            pass
        if speaker is not None and not presence_state.should_defer and deferred_notifications:
            for reminder in deferred_notifications:
                print(f"EDITH reminder: {_console_text(reminder)}")
                speaker.speak(reminder)
                follow_up_until = time.monotonic() + settings.follow_up_seconds
            deferred_notifications.clear()
        if visitor_recorder is not None and visitor_recorder.interview_active:
            utterance = audio.utterance(
                settings.silence_seconds,
                settings.max_recording_seconds,
                min_voiced_frames=settings.min_voiced_frames,
                min_rms=settings.min_audio_rms,
            )
            if utterance:
                print("Transcribing visitor response...")
                visitor_answer = stt.transcribe(utterance)
                if visitor_answer:
                    visitor_prompt = visitor_recorder.accept_interview_answer(
                        visitor_answer
                    )
                    if visitor_prompt:
                        print(f"EDITH: {_console_text(visitor_prompt)}")
                        speaker.speak(visitor_prompt)
            continue
        if not presence_state.should_accept_commands:
            time.sleep(0.1)
            continue
        text = ""
        if time.monotonic() >= follow_up_until:
            if detector is not None:
                if not detector.detected(audio.frames()):
                    continue
            utterance = audio.utterance(
                settings.silence_seconds,
                settings.max_recording_seconds,
                start_timeout_seconds=0.5,
                min_voiced_frames=settings.min_voiced_frames,
                min_rms=settings.min_audio_rms,
            )
            if not utterance:
                follow_up_until = 0.0
                continue
            print("Transcribing...")
            text = stt.transcribe(utterance)
            if not text:
                continue
            if detector is None:
                print(f"Transcript: {_console_text(text)}")
                command = extract_wake_command(text)
                if command is None:
                    print('Wake command not detected. Say: "HEY ...".')
                    continue
                if not command:
                    follow_up_until = time.monotonic() + settings.follow_up_seconds
                    print('Wake command heard. Listening for your follow-up...')
                    continue
                print(f"Wake command detected: {_console_text(command)}")
                text = command
        else:
            utterance = audio.utterance(
                settings.silence_seconds,
                settings.max_recording_seconds,
                settings.follow_up_listen_timeout_seconds,
                min_voiced_frames=settings.min_voiced_frames,
                min_rms=settings.min_audio_rms,
            )
            if not utterance:
                follow_up_until = 0.0
                continue
            print("Transcribing...")
            text = stt.transcribe(utterance)
            if not text:
                continue
        if playback.blocked:
            continue

        if settings.wakeword_test_mode:
            print(f"WAKE TEST SUCCESS: {_console_text(text)}")
            follow_up_until = time.monotonic() + settings.follow_up_seconds
            continue

        assert chat is not None
        assert speaker is not None
        print(f"You: {_console_text(text)}")
        assert reminder_engine is not None
        try:
            pending_reminders = reminder_engine.tick()
        except (OSError, RuntimeError) as error:
            print(f"EDITH reminder check unavailable: {_console_text(str(error))}")
            pending_reminders = []
        for reminder in pending_reminders:
            if presence_state.should_defer:
                deferred_notifications.append(reminder)
            else:
                print(f"EDITH reminder: {_console_text(reminder)}")
                speaker.speak(reminder)
        checkin_answer = reminder_engine.consume_response(text)
        if checkin_answer is not None:
            chat.record_local_exchange(text, checkin_answer)
            print(f"EDITH: {_console_text(checkin_answer)}")
            speaker.speak(checkin_answer)
            follow_up_until = time.monotonic() + settings.follow_up_seconds
            continue
        print(f"Sending to {_console_text(settings.gemini_model)}...")
        try:
            answer = chat.ask(text)
        except (RuntimeError, OSError, httpx.HTTPError) as error:
            print(f"EDITH error: {_console_text(str(error))}")
            retry_message = (
                "Sir, that request failed. I am retrying it in the background."
            )
            speaker.speak(retry_message)
            threading.Thread(
                target=_retry_chat_request,
                args=(chat, text, completed_retries),
                name="edith-gemini-retry",
                daemon=True,
            ).start()
            follow_up_until = time.monotonic() + settings.follow_up_seconds
            continue
        print(f"EDITH: {_console_text(answer)}")
        speaker.speak(answer)
        follow_up_until = time.monotonic() + settings.follow_up_seconds


if __name__ == "__main__":
    run()
