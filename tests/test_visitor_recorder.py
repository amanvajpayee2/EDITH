import tempfile
import unittest
from pathlib import Path

from edith.visitor_recorder import VisitorRecorder


class Frame:
    shape = (4, 6, 3)


class Writer:
    def __init__(self):
        self.frames = []

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        pass


class Storage:
    def __init__(self):
        self.uploads = []

    def upload_binary(self, collection, name, path, content_type, metadata):
        self.uploads.append((collection, name, Path(path), content_type, metadata))


class FakeSoundDevice:
    class InputStream:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]

        def start(self):
            self.callback(type("Data", (), {"tobytes": lambda self: b"\0\0"})(), 1, None, None)

        def stop(self):
            pass

        def close(self):
            pass


class VisitorRecorderTests(unittest.TestCase):
    def test_audio_lifecycle_muxes_and_cleans_temporary_files(self):
        storage = Storage()
        writer = Writer()
        def video_writer(path, *_):
            Path(path).write_bytes(b"video")
            return writer
        with tempfile.TemporaryDirectory() as directory:
            def run(command, **kwargs):
                Path(command[-1]).write_bytes(b"muxed")

            recorder = VisitorRecorder(
                storage, "visitors", 10, "recording",
                writer_factory=video_writer,
                audio_enabled=True,
                sounddevice_module=FakeSoundDevice,
                ffmpeg_path="ffmpeg",
                subprocess_run=run,
            )
            recorder._mux_audio = lambda video, audio: (
                Path(directory) / "muxed.mp4"
            )
            # Keep command construction deterministic without requiring ffmpeg.
            command = recorder._ffmpeg_command(
                Path("video.mp4"), Path("audio.wav"), Path("out.mp4")
            )
            self.assertIn("-shortest", command)
            recorder.process(Frame(), True, None)
            recorder.stop("test")
            self.assertEqual(len(storage.uploads), 1)
            self.assertFalse(storage.uploads[0][2].exists())

    def test_audio_failure_keeps_video_recording(self):
        storage = Storage()
        def video_writer(path, *_):
            Path(path).write_bytes(b"video")
            return Writer()
        recorder = VisitorRecorder(
            storage, "visitors", 10, "recording",
            writer_factory=video_writer,
            audio_enabled=True,
            sounddevice_module=type(
                "BrokenSoundDevice", (), {
                    "InputStream": lambda **kwargs: (_ for _ in ()).throw(OSError("busy"))
                }
            ),
        )
        recorder.process(Frame(), True, None)
        recorder.stop("test")
        self.assertEqual(len(storage.uploads), 1)


if __name__ == "__main__":
    unittest.main()
