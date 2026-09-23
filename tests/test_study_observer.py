import unittest
from datetime import datetime, timezone

import numpy as np

from edith.study_observer import StudyObserver


class FakeBackend:
    def frame_size(self, frame):
        return (200, 100)

    def motion_score(self, previous, current):
        return 0.01


class StudyObserverTests(unittest.TestCase):
    def test_aggregates_regions_without_retaining_frames(self):
        observer = StudyObserver(
            desk_region=(0, 0, 0.5, 1),
            bed_region=(0.5, 0, 0.5, 1),
            backend=FakeBackend(),
            clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        first = observer.observe(np.zeros((100, 200, 3), dtype=np.uint8), [(10, 10, 40, 70)], person_present=True)
        second = observer.observe(np.zeros((100, 200, 3), dtype=np.uint8), [(120, 10, 40, 70)], person_present=True)

        self.assertFalse(first.conclusive)  # no previous frame for motion
        self.assertTrue(second.conclusive)
        self.assertTrue(second.desk_occupied is False)
        self.assertTrue(second.bed_occupied)
        summary = observer.summary()
        self.assertEqual(summary.samples, 2)
        self.assertEqual(summary.observed_samples, 1)
        self.assertEqual(summary.bed_occupancy_fraction, 1.0)

    def test_missing_detector_is_explicitly_inconclusive(self):
        observer = StudyObserver(backend=FakeBackend())
        observer.observe(np.zeros((100, 200, 3), dtype=np.uint8), person_present=None)
        observer.observe(np.zeros((100, 200, 3), dtype=np.uint8), person_present=None)
        summary = observer.summary()
        self.assertFalse(summary.conclusive)
        self.assertIn("inconclusive", summary.reason)

    def test_invalid_region_is_rejected(self):
        with self.assertRaises(ValueError):
            StudyObserver(desk_region=(0.8, 0, 0.4, 0.2))


if __name__ == "__main__":
    unittest.main()
