import unittest

from edith.presence import _filter_hog_detections


class FakeDnn:
    def NMSBoxes(self, boxes, scores, score_threshold, nms_threshold):
        return [[index] for index in range(len(boxes))]


class FakeCv2:
    dnn = FakeDnn()


class PresenceDetectionTests(unittest.TestCase):
    def test_discards_low_confidence_hog_boxes(self):
        boxes = [(0, 0, 20, 40), (10, 10, 80, 160)]
        filtered = _filter_hog_detections(FakeCv2, boxes, [0.12, 0.71], 0.4)
        self.assertEqual(filtered, [boxes[1]])

    def test_handles_missing_weights_without_claiming_detection(self):
        boxes = [(0, 0, 20, 40)]
        self.assertEqual(_filter_hog_detections(FakeCv2, boxes, None, 0.4), [])


if __name__ == "__main__":
    unittest.main()
