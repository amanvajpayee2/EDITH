import threading
import time as wall_time
import unittest
from datetime import datetime, time, timezone

from edith.presence import CameraPresenceWorker, PresenceState


class FakeCapture:
    def __init__(self):
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        return True, object()

    def release(self):
        self.released = True


class FakeHog:
    def setSVMDetector(self, detector):
        pass

    def detectMultiScale(self, frame):
        return [], None


class FakeCv2:
    HOGDescriptor = FakeHog

    @staticmethod
    def HOGDescriptor_getDefaultPeopleDetector():
        return object()


class FakeStudy:
    active = True

    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def observe(self, frame, boxes, person_present):
        pass


class FakeRecorder:
    def __init__(self):
        self.stops = []

    def stop(self, reason):
        self.stops.append(reason)


class CameraScheduleTests(unittest.TestCase):
    def test_quiet_hours_release_and_reopen_camera(self):
        current = [time(23, 0)]
        captures = []
        study = FakeStudy()
        recorder = FakeRecorder()

        def clock():
            return datetime.combine(datetime(2026, 1, 1), current[0], timezone.utc)

        def open_capture(_):
            capture = FakeCapture()
            captures.append(capture)
            return capture

        state = PresenceState(enabled=True, require_owner=True)
        worker = CameraPresenceWorker(
            state,
            quiet_start=time(22, 0),
            quiet_end=time(6, 0),
            interval_seconds=0.01,
            cv2_module=FakeCv2,
            capture_factory=open_capture,
            study_observer=study,
            visitor_recorder=recorder,
            clock=clock,
        )
        worker.start()
        self.assertTrue(_wait_for(lambda: worker.state.snapshot().degraded_reason == "camera quiet hours"))
        self.assertEqual(captures, [])
        self.assertFalse(state.should_accept_commands)

        current[0] = time(6, 1)
        self.assertTrue(_wait_for(lambda: len(captures) == 1))
        current[0] = time(23, 1)
        self.assertTrue(_wait_for(lambda: captures[0].released))
        self.assertEqual(study.resets, 1)
        self.assertIn("camera quiet hours", recorder.stops)

        current[0] = time(6, 2)
        self.assertTrue(_wait_for(lambda: len(captures) == 2))
        worker.stop()
        self.assertTrue(captures[1].released)

    def test_empty_schedule_preserves_single_camera_lifecycle(self):
        captures = []

        def open_capture(_):
            capture = FakeCapture()
            captures.append(capture)
            return capture

        worker = CameraPresenceWorker(
            PresenceState(),
            interval_seconds=0.01,
            cv2_module=FakeCv2,
            capture_factory=open_capture,
        )
        worker.start()
        self.assertTrue(_wait_for(lambda: len(captures) == 1))
        worker.stop()
        self.assertEqual(len(captures), 1)
        self.assertTrue(captures[0].released)


def _wait_for(predicate):
    deadline = wall_time.monotonic() + 1.0
    while wall_time.monotonic() < deadline:
        if predicate():
            return True
        wall_time.sleep(0.005)
    return False


if __name__ == "__main__":
    unittest.main()
