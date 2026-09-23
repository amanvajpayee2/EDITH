import base64
import os
import platform
import subprocess
import threading

import pyttsx3

from .audio import PlaybackGuard


class Speaker:
    def __init__(self, playback_guard: PlaybackGuard) -> None:
        self.playback_guard = playback_guard
        self._lock = threading.Lock()
        self.engine = None if platform.system() == "Windows" else pyttsx3.init()

    def speak(self, text: str) -> None:
        with self._lock:
            words = max(1, len(text.split()))
            rate = 170
            if self.engine is not None:
                rate = self.engine.getProperty("rate") or rate
            estimated_seconds = words / (float(rate) / 60.0)
            self.playback_guard.block(estimated_seconds)
            try:
                if platform.system() == "Windows":
                    self._speak_windows(text)
                    return
                assert self.engine is not None
                self.engine.say(text)
                self.engine.runAndWait()
            except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
                print(f"Speech output unavailable: {error}")

    @staticmethod
    def _speak_windows(text: str) -> None:
        command = (
            "$speaker = New-Object -ComObject SAPI.SpVoice; "
            "$speaker.Speak($env:EDITH_SPEECH_TEXT) | Out-Null; "
            "[System.Runtime.InteropServices.Marshal]::ReleaseComObject($speaker) | Out-Null"
        )
        encoded_command = base64.b64encode(command.encode("utf-16le")).decode("ascii")
        environment = os.environ.copy()
        environment["EDITH_SPEECH_TEXT"] = text
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded_command],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
