import time
import sys
from queue import Empty, Queue
import httpx
from datetime import datetime, time as clock_time

from .audio import AudioInput, PlaybackGuard
from .cloud import OpenRouterChat, SpeechToText
from .config import Settings
from .speaker import Speaker
from .reminders import ReminderEngine, ReminderWorker
from .storage import GoogleDriveStorage
from .wakeword import WakeWordDetector, extract_wake_command


def _console_text(value: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return value.encode(encoding, errors="replace").decode(encoding, errors="replace")


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
    if not settings.wakeword_test_mode:
        storage = GoogleDriveStorage(
            settings.google_drive_credentials_file,
            settings.google_drive_token_file,
            settings.google_drive_folder,
        )
    playback = PlaybackGuard(settings.playback_cooldown_seconds)
    pending_notifications: Queue[str] = Queue()
    speaker = None
    if not settings.wakeword_test_mode:
        speaker = Speaker(playback)
        assert storage is not None
        reminder_engine = ReminderEngine(
            storage,
            settings.reminder_advance_minutes,
            clock_time(settings.daily_prompt_hour, settings.daily_prompt_minute),
            settings.daily_prompt_retry_minutes,
        )
        chat = OpenRouterChat(
            settings.openrouter_api_key,
            settings.openrouter_model,
            storage,
            lambda: reminder_engine.acknowledge_daily_plan(
                datetime.now().astimezone().date()
            ),
        )
        reminder_worker = ReminderWorker(
            reminder_engine,
            pending_notifications.put,
            settings.reminder_interval_seconds,
        )
        reminder_worker.start()
    follow_up_until = 0.0
    detector = None

    if settings.wakeword_engine in {"spoken", "fallback"}:
        print('Spoken wake fallback is active. Say "HEY" followed by your command.')
    elif settings.wakeword_engine == "openwakeword":
        try:
            detector = WakeWordDetector(settings.wakeword_engine, settings.wakeword_model_path, settings.wakeword_threshold)
            print("EDITH is listening locally for the wake word.")
        except FileNotFoundError:
            print('Wake model missing. Falling back to spoken command mode ("HEY ...").')
    else:
        detector = WakeWordDetector(settings.wakeword_engine, settings.wakeword_model_path, settings.wakeword_threshold)
        print("EDITH is listening locally for the wake word.")

    while True:
        if playback.blocked:
            time.sleep(0.1)
            continue
        try:
            if time.monotonic() >= follow_up_until:
                while True:
                    reminder = pending_notifications.get_nowait()
                    print(f"EDITH reminder: {_console_text(reminder)}")
                    speaker.speak(reminder)
                    follow_up_until = time.monotonic() + settings.follow_up_seconds
        except Empty:
            pass
        text = ""
        if time.monotonic() >= follow_up_until:
            if detector is not None:
                if not detector.detected(audio.frames()):
                    continue
            utterance = audio.utterance(
                settings.silence_seconds,
                settings.max_recording_seconds,
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
            print(f"EDITH reminder: {_console_text(reminder)}")
            speaker.speak(reminder)
        checkin_answer = reminder_engine.consume_response(text)
        if checkin_answer is not None:
            chat.record_local_exchange(text, checkin_answer)
            print(f"EDITH: {_console_text(checkin_answer)}")
            speaker.speak(checkin_answer)
            follow_up_until = time.monotonic() + settings.follow_up_seconds
            continue
        print(f"Sending to {_console_text(settings.openrouter_model)}...")
        try:
            answer = chat.ask(text)
        except (RuntimeError, OSError, httpx.HTTPError) as error:
            print(f"EDITH error: {_console_text(str(error))}")
            follow_up_until = 0.0
            continue
        print(f"EDITH: {_console_text(answer)}")
        speaker.speak(answer)
        follow_up_until = time.monotonic() + settings.follow_up_seconds


if __name__ == "__main__":
    run()
