import unittest
from datetime import datetime, timedelta, timezone

from edith.reminders import ReminderEngine
from edith.study_observer import StudySessionSummary


class MemoryStorage:
    def __init__(self, plan):
        self.data = {"plans": {plan["date"] + ".json": plan}, "metadata": {}}

    def read_json(self, folder, name, default):
        return self.data.get(folder, {}).get(name, default)

    def write_json(self, folder, name, value):
        self.data.setdefault(folder, {})[name] = value


class FakeObserver:
    active = True

    def __init__(self):
        self.started = []
        self.ended = 0

    def start_slot(self, slot_id):
        self.started.append(slot_id)

    def end_slot(self):
        self.ended += 1
        return StudySessionSummary(2, 2, 1.0, 0.5, 0.0, 0.75, 0.9, True, None)


class StudyReminderTests(unittest.TestCase):
    def setUp(self):
        self.start = datetime.now().astimezone().replace(second=0, microsecond=0) - timedelta(minutes=5)
        plan = {
            "date": self.start.date().isoformat(), "goals": [],
            "slots": [{
                "id": "study-1", "title": "Study biology",
                "start": self.start.isoformat(),
                "end": (self.start + timedelta(minutes=5)).isoformat(),
                "status": "planned",
            }],
        }
        self.storage = MemoryStorage(plan)
        self.observer = FakeObserver()
        self.engine = ReminderEngine(
            self.storage, study_observer=self.observer,
            study_slot_title_patterns=("study",),
        )

    def test_study_lifecycle_persists_aggregate_only(self):
        self.engine.tick(self.start + timedelta(minutes=1))
        self.assertEqual(self.observer.started, ["study-1"])
        messages = self.engine.tick(self.start + timedelta(minutes=6))
        slot = self.storage.data["plans"][self.start.date().isoformat() + ".json"]["slots"][0]
        self.assertEqual(self.observer.ended, 1)
        self.assertEqual(slot["study_summary"]["desk_occupancy_fraction"], 0.5)
        self.assertNotIn("frame", slot["study_summary"])
        self.assertIn("confidence", messages[0])

    def test_negative_checkin_records_reason_and_reschedules(self):
        self.engine.tick(self.start + timedelta(minutes=6))
        self.engine.consume_response("no")
        response = self.engine.consume_response("I was tired")
        self.assertIn("reschedule", response)
        response = self.engine.consume_response("tomorrow")
        self.assertIn("rescheduled", response)
        slot = self.storage.data["plans"][self.start.date().isoformat() + ".json"]["slots"][0]
        self.assertEqual(slot["status"], "planned")
        self.assertEqual(
            datetime.fromisoformat(slot["start"]).date(),
            self.start.date() + timedelta(days=1),
        )


if __name__ == "__main__":
    unittest.main()
