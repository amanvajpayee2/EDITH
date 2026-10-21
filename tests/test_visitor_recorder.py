import tempfile
import unittest
from pathlib import Path

from edith.app import _visitor_return_report
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
        self.data = {}

    def upload_binary(self, collection, name, path, content_type, metadata):
        self.uploads.append((collection, name, Path(path), content_type, metadata))

    def read_json(self, collection, name, default):
        return self.data.get((collection, name), default)

    def write_json(self, collection, name, value):
        self.data[(collection, name)] = value


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
    @staticmethod
    def _writer_with_file(path, *_):
        Path(path).write_bytes(b"video")
        return Writer()

    def test_owner_leaving_allows_new_visitor_recording(self):
        storage = Storage()
        writers = []
        def video_writer(path, *_):
            writer = Writer()
            writers.append(writer)
            Path(path).write_bytes(b"video")
            return writer
        recorder = VisitorRecorder(
            storage, "visitors", 10, "recording",
            writer_factory=video_writer,
            fps=60,
        )
        recorder.process(Frame(), True, "owner")
        recorder.process(Frame(), True, "unknown")
        recorder.process(Frame(), True, "unknown")
        recorder.process(Frame(), True, "unknown")
        self.assertTrue(recorder.recording)
        self.assertEqual(recorder.fps, 60)
        recorder.stop("test")

    def test_interview_answers_are_persisted_for_owner_return(self):
        storage = Storage()
        recorder = VisitorRecorder(
            storage, "visitors", 10, "recording",
            writer_factory=self._writer_with_file,
        )
        recorder.process(Frame(), True, "unknown")
        self.assertTrue(recorder.interview_active)
        self.assertIn("Why are you", recorder.accept_interview_answer("Priya"))
        self.assertIn("recorded", recorder.accept_interview_answer("To return a book"))
        recorder.stop("test")
        visit = storage.data[("metadata", "visitor-summaries.json")]["visits"][0]
        self.assertEqual(visit["name"], "Priya")
        self.assertEqual(visit["reason"], "To return a book")

    def test_capture_rate_controls_writer_playback_rate(self):
        recorder = VisitorRecorder(
            Storage(), "visitors", 10, "recording", fps=60
        )
        recorder.set_capture_fps(30)
        self.assertEqual(recorder.fps, 30)

    def test_video_frames_are_drained_before_stop(self):
        storage = Storage()
        writer = Writer()
        def video_writer(path, *_):
            Path(path).write_bytes(b"video")
            return writer
        recorder = VisitorRecorder(
            storage, "visitors", 10, "recording",
            writer_factory=video_writer,
        )
        recorder.process(Frame(), True, "unknown")
        for _ in range(25):
            recorder.process(Frame(), True, "unknown")
        recorder.stop("test")
        self.assertEqual(len(writer.frames), 26)

    def test_failed_upload_is_preserved_for_retry(self):
        storage = Storage()
        storage.upload_binary = lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("EOF occurred in violation of protocol")
        )
        recorder = VisitorRecorder(
            storage, "visitors", 10, "recording",
            writer_factory=self._writer_with_file,
        )
        with tempfile.TemporaryDirectory() as directory:
            recorder._pending_dir = Path(directory) / "pending"
            recorder.process(Frame(), True, "unknown")
            recorder.stop("test")
            self.assertEqual(len(list(recorder._pending_dir.glob("*.mp4"))), 1)

    def test_recorded_audio_transcript_is_used_for_summary(self):
        storage = Storage()
        recorder = VisitorRecorder(
            storage, "visitors", 10, "recording",
            writer_factory=self._writer_with_file,
            transcribe_audio=lambda path: (
                "This room is monitored. Recording has started. "
                "Please tell me your name. I am Priya. I came to return a book."
            ),
        )
        recorder.process(Frame(), True, "unknown")
        audio_path = Path(tempfile.mktemp(suffix=".wav"))
        audio_path.write_bytes(b"audio")
        recorder._audio_path = audio_path
        recorder._mux_audio = lambda video, audio: video
        recorder.stop("test")
        visit = storage.data[("metadata", "visitor-summaries.json")]["visits"][0]
        self.assertEqual(visit["name"], "visitor")
        self.assertIn("return a book", visit["conversation"])

    def test_live_answers_are_preferred_over_noisy_recording(self):
        storage = Storage()
        recorder = VisitorRecorder(
            storage, "visitors", 10, "recording",
            writer_factory=self._writer_with_file,
            transcribe_audio=lambda path: (
                "Reporting has started. Please tell me your name. "
                "Thank you. Why are you in this room."
            ),
        )
        recorder.process(Frame(), True, "unknown")
        recorder.accept_interview_answer("Yashir Sharma")
        recorder.accept_interview_answer("to visit Aman")
        recorder.stop("test")
        visit = storage.data[("metadata", "visitor-summaries.json")]["visits"][0]
        self.assertEqual(visit["name"], "Yashir Sharma")
        self.assertEqual(visit["reason"], "to visit Aman")

    def test_owner_report_is_consumed_once(self):
        storage = Storage()
        storage.data[("metadata", "visitor-summaries.json")] = {
            "visits": [{
                "name": "Priya",
                "reason": "to return a book",
                "entered_at": "2026-10-14T14:15:00+05:30",
            }]
        }
        report = _visitor_return_report(storage)
        self.assertIn("Priya", report)
        self.assertIn("2:15 PM", report)
        self.assertIsNone(_visitor_return_report(storage))

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
