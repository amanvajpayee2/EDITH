"""Optional, local-only owner verification.

The backend is deliberately imported lazily.  Face encodings are stored locally;
camera frames are never written, uploaded, or passed to a cloud service.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


class OwnerVerifier:
    """Enroll and verify one owner using an optional ``face_recognition`` backend."""

    def __init__(self, encoding_file: str, tolerance: float = 0.48, backend: Any = None):
        self.encoding_file = encoding_file
        self.tolerance = tolerance
        self._backend = backend
        self._encoding: list[float] | None = None
        self._reason: str | None = None
        self._load()

    @property
    def available(self) -> bool:
        return self._backend is not None and self._encoding is not None

    @property
    def degraded_reason(self) -> str | None:
        return self._reason

    def enroll(self, frame: Any) -> None:
        self.enroll_many([frame])

    def enroll_many(self, frames: list[Any]) -> None:
        backend = self._get_backend()
        samples = []
        for frame in frames:
            locations = _face_locations(backend, frame)
            encodings = backend.face_encodings(frame, locations)
            if len(encodings) != 1:
                continue
            samples.append(encodings[0])
        if not samples:
            raise ValueError("Enrollment requires one clear visible face in at least one sample.")
        dimensions = len(samples[0])
        averaged = [
            sum(float(sample[index]) for sample in samples) / len(samples)
            for index in range(dimensions)
        ]
        self._encoding = averaged
        self._reason = None
        parent = os.path.dirname(os.path.abspath(self.encoding_file))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.encoding_file, "w", encoding="utf-8") as handle:
            json.dump({"encoding": self._encoding}, handle)

    def verify(self, frame: Any, person_boxes: Any = None) -> bool | None:
        """Return True for owner, False for known non-owner, None when unknown."""
        if not self.available:
            return None
        encodings = []
        for candidate in _verification_candidates(frame, person_boxes):
            locations = _face_locations(self._backend, candidate)
            encodings.extend(self._backend.face_encodings(candidate, locations))
        if not encodings:
            return None
        known = np.asarray(self._encoding, dtype=np.float64)
        candidates = [
            np.asarray(encoding, dtype=np.float64) for encoding in encodings
        ]
        distances = [float(np.linalg.norm(candidate - known)) for candidate in candidates]
        return min(distances) <= self.tolerance

    def _get_backend(self) -> Any:
        if self._backend is None:
            try:
                import face_recognition
            except (ImportError, OSError) as error:
                self._reason = f"face-recognition backend unavailable: {error}"
                raise RuntimeError(self._reason) from error
            self._backend = face_recognition
        return self._backend

    def _load(self) -> None:
        if self._backend is None:
            try:
                import face_recognition
                self._backend = face_recognition
            except (ImportError, OSError) as error:
                self._reason = f"face-recognition backend unavailable: {error}"
                return
        try:
            with open(self.encoding_file, encoding="utf-8") as handle:
                values = json.load(handle).get("encoding")
            if not isinstance(values, list) or not values:
                raise ValueError("encoding file is invalid")
            self._encoding = [float(value) for value in values]
        except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError) as error:
            self._reason = f"owner is not enrolled: {error}"


def _face_locations(backend: Any, frame: Any) -> Any:
    try:
        return backend.face_locations(frame, number_of_times_to_upsample=2)
    except TypeError:
        return backend.face_locations(frame)


def _verification_candidates(frame: Any, person_boxes: Any) -> list[Any]:
    if person_boxes is None:
        return [frame]
    candidates = []
    try:
        height, width = frame.shape[:2]
        for box in person_boxes:
            x, y, box_width, box_height = (int(value) for value in box[:4])
            left = max(0, x)
            top = max(0, y)
            right = min(width, x + box_width)
            bottom = min(height, y + box_height)
            if right > left and bottom > top:
                # dlib rejects non-contiguous NumPy views produced by slicing.
                candidates.append(frame[top:bottom, left:right].copy())
    except (AttributeError, TypeError, ValueError):
        return [frame]
    return candidates or [frame]
