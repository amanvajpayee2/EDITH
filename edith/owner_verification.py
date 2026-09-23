"""Optional, local-only owner verification.

The backend is deliberately imported lazily.  Face encodings are stored locally;
camera frames are never written, uploaded, or passed to a cloud service.
"""

from __future__ import annotations

import json
import os
from typing import Any


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
        backend = self._get_backend()
        locations = backend.face_locations(frame)
        encodings = backend.face_encodings(frame, locations)
        if len(encodings) != 1:
            raise ValueError("Enrollment requires exactly one visible face.")
        self._encoding = [float(value) for value in encodings[0]]
        self._reason = None
        parent = os.path.dirname(os.path.abspath(self.encoding_file))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.encoding_file, "w", encoding="utf-8") as handle:
            json.dump({"encoding": self._encoding}, handle)

    def verify(self, frame: Any) -> bool | None:
        """Return True for owner, False for known non-owner, None when unknown."""
        if not self.available:
            return None
        locations = self._backend.face_locations(frame)
        encodings = self._backend.face_encodings(frame, locations)
        if not encodings:
            return None
        return bool(self._backend.compare_faces([self._encoding], encodings[0], self.tolerance)[0])

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
