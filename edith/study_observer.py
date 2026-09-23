"""Opt-in, local-only aggregate study-session observations.

This module deliberately does not retain frames.  Observations are estimates:
occlusion, camera angle, lighting, and a missing detector can make a sample
inconclusive.  They must not be interpreted as proof that somebody studied,
slept, or was absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterable, Sequence, Tuple


Region = Tuple[float, float, float, float]


@dataclass(frozen=True)
class StudyObservation:
    timestamp: datetime
    person_present: bool | None
    desk_occupied: bool | None
    bed_occupied: bool | None
    low_motion: bool | None
    confidence: float
    conclusive: bool
    reason: str | None = None


@dataclass(frozen=True)
class StudySessionSummary:
    samples: int
    observed_samples: int
    person_present_fraction: float | None
    desk_occupancy_fraction: float | None
    bed_occupancy_fraction: float | None
    low_motion_fraction: float | None
    average_confidence: float
    conclusive: bool
    reason: str | None


class StudyObserver:
    """Convert detector boxes and frame differences into aggregate estimates.

    ``backend`` is optional and is intended for tests or applications that
    already have a detector.  It may provide ``frame_size(frame)`` and
    ``motion_score(previous, current)``.  Person boxes are supplied by the
    existing camera worker, so no second camera or recording path is created.
    """

    def __init__(
        self,
        desk_region: Region = (0.0, 0.5, 0.5, 0.5),
        bed_region: Region = (0.5, 0.5, 0.5, 0.5),
        motion_threshold: float = 0.08,
        min_confidence: float = 0.5,
        backend: Any | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.desk_region = _validate_region(desk_region)
        self.bed_region = _validate_region(bed_region)
        self.motion_threshold = max(0.0, motion_threshold)
        self.min_confidence = min(1.0, max(0.0, min_confidence))
        self.backend = backend
        self._clock = clock or (lambda: datetime.now().astimezone())
        # A small numeric signature is retained, never the raw frame.
        self._previous_signature: Any | None = None
        self._observations: list[StudyObservation] = []
        self._active = True
        self._slot_id: str | None = None

    @property
    def active(self) -> bool:
        return self._active

    def start_slot(self, slot_id: str) -> None:
        """Begin a fresh aggregate for a planned study slot."""
        self.reset()
        self._slot_id = slot_id
        self._active = True

    def end_slot(self) -> StudySessionSummary:
        """Finalize the aggregate and stop accepting camera samples."""
        result = self.summary()
        self._active = False
        self._slot_id = None
        self._previous_signature = None
        self._observations.clear()
        return result

    def observe(
        self,
        frame: Any,
        person_boxes: Iterable[Sequence[float]] | None = None,
        person_present: bool | None = None,
    ) -> StudyObservation:
        if not self._active:
            raise RuntimeError("study observation is not active")
        timestamp = self._clock()
        width, height = _frame_size(frame, self.backend)
        boxes = list(person_boxes) if person_boxes is not None else []
        if person_present is None:
            person_present = bool(boxes) if person_boxes is not None else None
        confidence = 0.0 if width <= 0 or height <= 0 else 1.0
        reason = None
        if width <= 0 or height <= 0:
            reason = "frame dimensions unavailable"
            confidence = 0.0
        elif person_present is None:
            reason = "person detector result unavailable"
            confidence = 0.35

        desk = _region_has_person(boxes, self.desk_region, width, height) if person_present is not None else None
        bed = _region_has_person(boxes, self.bed_region, width, height) if person_present is not None else None
        low_motion = self._motion(frame)
        if low_motion is None and reason is None:
            reason = "motion estimate unavailable"
            confidence = min(confidence, 0.45)
        conclusive = confidence >= self.min_confidence and reason is None
        observation = StudyObservation(
            timestamp, person_present, desk, bed, low_motion,
            confidence, conclusive, reason,
        )
        self._observations.append(observation)
        return observation

    def summary(self) -> StudySessionSummary:
        observations = tuple(self._observations)
        if not observations:
            return StudySessionSummary(0, 0, None, None, None, None, 0.0, False, "no observations")
        usable = tuple(item for item in observations if item.conclusive)
        fraction = lambda attr: _fraction(usable, attr)
        reason = None if usable else "observations were inconclusive"
        return StudySessionSummary(
            len(observations), len(usable), fraction("person_present"),
            fraction("desk_occupied"), fraction("bed_occupied"),
            fraction("low_motion"),
            sum(item.confidence for item in observations) / len(observations),
            bool(usable), reason,
        )

    def reset(self) -> None:
        self._previous_signature = None
        self._observations.clear()

    def _motion(self, frame: Any) -> bool | None:
        signature = _frame_signature(frame)
        if signature is None:
            return None
        if self._previous_signature is None:
            self._previous_signature = signature
            return None
        try:
            if self.backend is not None and hasattr(self.backend, "motion_score"):
                score = float(self.backend.motion_score(self._previous_signature, signature))
            else:
                score = _default_motion_score(self._previous_signature, signature)
        except (TypeError, ValueError, AttributeError, ImportError):
            score = None
        self._previous_signature = signature
        return None if score is None else score <= self.motion_threshold


def _frame_size(frame: Any, backend: Any | None) -> tuple[int, int]:
    if backend is not None and hasattr(backend, "frame_size"):
        width, height = backend.frame_size(frame)
        return int(width), int(height)
    shape = getattr(frame, "shape", ())
    if len(shape) < 2:
        return 0, 0
    return int(shape[1]), int(shape[0])


def _region_has_person(boxes: list[Sequence[float]], region: Region, width: int, height: int) -> bool:
    rx, ry, rw, rh = region
    left, top, right, bottom = rx * width, ry * height, (rx + rw) * width, (ry + rh) * height
    for box in boxes:
        if len(box) < 4:
            continue
        x, y, w, h = map(float, box[:4])
        if max(0.0, min(right, x + w) - max(left, x)) * max(0.0, min(bottom, y + h) - max(top, y)) > 0:
            return True
    return False


def _default_motion_score(previous: Any, current: Any) -> float | None:
    try:
        import numpy as np
        before, after = np.asarray(previous), np.asarray(current)
        if before.shape != after.shape:
            return None
        return float(np.mean(np.abs(after.astype("float32") - before.astype("float32"))))
    except (ImportError, TypeError, ValueError):
        return None


def _frame_signature(frame: Any) -> Any | None:
    """Create a tiny grayscale signature so no raw frame is kept."""
    try:
        import numpy as np
        pixels = np.asarray(frame)
        if pixels.ndim < 2:
            return None
        if pixels.ndim == 3:
            pixels = pixels.mean(axis=2)
        height, width = pixels.shape[:2]
        rows = np.linspace(0, height - 1, 16).astype(int)
        columns = np.linspace(0, width - 1, 16).astype(int)
        return pixels[np.ix_(rows, columns)].astype("float32") / 255.0
    except (ImportError, TypeError, ValueError, IndexError):
        return None


def _validate_region(region: Region) -> Region:
    if len(region) != 4 or any(value < 0 for value in region) or region[0] + region[2] > 1 or region[1] + region[3] > 1:
        raise ValueError("regions must be normalized x,y,width,height values inside 0..1")
    return tuple(float(value) for value in region)  # type: ignore[return-value]


def _fraction(observations: tuple[StudyObservation, ...], attribute: str) -> float | None:
    values = [getattr(item, attribute) for item in observations if getattr(item, attribute) is not None]
    return sum(bool(value) for value in values) / len(values) if values else None
