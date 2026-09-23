import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from edith.owner_verification import OwnerVerifier
from edith.presence import PresenceState


class FakeBackend:
    def face_locations(self, frame):
        return [("face",)]

    def face_encodings(self, frame, locations):
        return [[float(frame)]]

    def compare_faces(self, known, candidate, tolerance):
        return [abs(known[0][0] - candidate[0]) <= tolerance]


class OwnerPresenceTests(unittest.TestCase):
    def test_enrollment_and_unknown_are_local_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "owner.json")
            verifier = OwnerVerifier(path, backend=FakeBackend())
            verifier.enroll(1)
            self.assertTrue(verifier.verify(1))
            self.assertFalse(verifier.verify(2))
            with open(path, encoding="utf-8") as handle:
                self.assertIn("encoding", json.load(handle))

    def test_return_event_requires_configured_absence(self):
        events = []
        state = PresenceState(on_event=events.append)
        state.set_return_after_seconds(1800)
        state.update(True, identity="owner")
        first = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with state._lock:
            state._owner_absent_since = first - timedelta(minutes=31)
        with patch("edith.presence.datetime") as clock:
            clock.now.return_value = first
            state.update(True, identity="owner")
        self.assertEqual(events, ["owner_returned"])
        self.assertEqual(state.consume_events(), ["owner_returned"])

    def test_owner_gate_defers_unknown_people(self):
        state = PresenceState(enabled=True, require_owner=True)
        state.update(True, identity="unknown")
        self.assertTrue(state.should_defer)
        self.assertFalse(state.should_accept_commands)
        state.update(True, identity="owner")
        self.assertFalse(state.should_defer)
        self.assertTrue(state.should_accept_commands)


if __name__ == "__main__":
    unittest.main()
