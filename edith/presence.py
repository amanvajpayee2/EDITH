"""Local, best-effort room presence detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
import threading
import time as monotonic_time
from typing import Any, Callable

from .owner_verification import OwnerVerifier
from .study_observer import StudyObserver
from .visitor_recorder import VisitorRecorder


@dataclass(frozen=True)
class PresenceSnapshot:
    occupied: bool | None
    available: bool
    degraded_reason: str | None
    updated_at: datetime | None
    identity: str | None = None
    owner_absent_seconds: float = 0.0


class PresenceState:
    """Thread-safe state shared by the camera worker and reminder worker."""

    def __init__(
        self,
        enabled: bool = False,
        on_event: Callable[[str], None] | None = None,
        require_owner: bool = False,
    ) -> None:
        self._lock = threading.Lock()
        self._enabled = enabled
        self._snapshot = PresenceSnapshot(None, False, None, None)
        self._owner_last_seen: datetime | None = None
        self._owner_absent_since: datetime | None = None
        self._return_event_sent = False
        self._events: list[str] = []
        self._return_after_seconds = 1800.0
        self._on_event = on_event
        self._require_owner = require_owner

    def update(self, occupied: bool, *, available: bool = True, identity: str | None = None) -> None:
        with self._lock:
            now = datetime.now().astimezone()
            if identity == "owner":
                if (
                    self._owner_absent_since is not None
                    and not self._return_event_sent
                    and (now - self._owner_absent_since).total_seconds() >= self._return_after_seconds
                ):
                    self._events.append("owner_returned")
                    if self._on_event is not None:
                        self._on_event("owner_returned")
                    self._return_event_sent = True
                self._owner_last_seen = now
                self._owner_absent_since = None
            elif not occupied and self._owner_last_seen is not None and self._owner_absent_since is None:
                self._owner_absent_since = now
                self._return_event_sent = False
            absent = (
                (now - self._owner_absent_since).total_seconds()
                if self._owner_absent_since is not None else 0.0
            )
            self._snapshot = PresenceSnapshot(
                occupied, available, None, now, identity, absent
            )

    def degrade(self, reason: str) -> None:
        with self._lock:
            self._snapshot = PresenceSnapshot(
                None, False, reason, datetime.now().astimezone(), "unknown", 0.0
            )

    def set_quiet(self) -> None:
        """Mark camera-derived state unavailable while the camera is released."""
        with self._lock:
            self._snapshot = PresenceSnapshot(
                None, False, "camera quiet hours", datetime.now().astimezone(), None, 0.0
            )

    def consume_events(self) -> list[str]:
        with self._lock:
            events, self._events = self._events, []
            return events

    def owner_absence_seconds(self) -> float:
        return self.snapshot().owner_absent_seconds

    def set_return_after_seconds(self, seconds: float) -> None:
        with self._lock:
            self._return_after_seconds = max(0.0, seconds)

    def snapshot(self) -> PresenceSnapshot:
        with self._lock:
            return self._snapshot

    @property
    def should_defer(self) -> bool:
        snapshot = self.snapshot()
        if not self._enabled or not snapshot.available:
            return False
        if snapshot.occupied is False:
            return True
        return self._require_owner and snapshot.identity != "owner"

    @property
    def should_accept_commands(self) -> bool:
        snapshot = self.snapshot()
        if not self._enabled or not self._require_owner:
            return True
        return snapshot.available and snapshot.occupied is True and snapshot.identity == "owner"


class CameraPresenceWorker:
    """Samples a webcam with a modern person detector and safe fallback."""

    def __init__(
        self,
        state: PresenceState,
        camera_index: int = 0,
        interval_seconds: float = 5.0,
        quiet_start: time | None = None,
        quiet_end: time | None = None,
        min_consecutive_detections: int = 2,
        min_consecutive_absence: int = 3,
        detector_backend: str = "hog",
        model_path: str = "models/yolo11n.pt",
        detection_confidence: float = 0.35,
        detection_iou: float = 0.45,
        detection_image_size: int = 640,
        hog_score_threshold: float = 0.4,
        owner_verifier: OwnerVerifier | None = None,
        cv2_module: Any | None = None,
        capture_factory: Callable[[int], Any] | None = None,
        study_observer: StudyObserver | None = None,
        visitor_recorder: VisitorRecorder | None = None,
        clock: Callable[[], datetime] | None = None,
        debug_enabled: bool = False,
        preview_enabled: bool = False,
    ) -> None:
        self.state = state
        self.camera_index = camera_index
        self.interval_seconds = max(0.1, interval_seconds)
        self.quiet_start, self.quiet_end = quiet_start, quiet_end
        self.min_detections = max(1, min_consecutive_detections)
        self.min_absence = max(1, min_consecutive_absence)
        self.hog_score_threshold = hog_score_threshold
        self.detector_backend = detector_backend
        self.model_path = model_path
        self.detection_confidence = min(max(detection_confidence, 0.05), 0.99)
        self.detection_iou = min(max(detection_iou, 0.05), 0.95)
        self.detection_image_size = max(320, detection_image_size)
        self.owner_verifier = owner_verifier
        self._cv2 = cv2_module
        self._capture_factory = capture_factory
        self.study_observer = study_observer
        self.visitor_recorder = visitor_recorder
        self._clock = clock or (lambda: datetime.now().astimezone())
        self.debug_enabled = debug_enabled
        self.preview_enabled = preview_enabled
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="edith-camera", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval_seconds + 1)

    def _run(self) -> None:
        try:
            cv2 = self._cv2 or _import_cv2()
            detector_name, detector = _create_person_detector(
                cv2,
                self.detector_backend,
                self.model_path,
                self.detection_confidence,
                self.detection_iou,
                self.detection_image_size,
                self.hog_score_threshold,
            )
            self._debug(f"detector ready: {detector_name}")
        except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as error:
            self._degrade(f"OpenCV unavailable: {error}")
            return

        present_count = absent_count = 0
        capture = None
        try:
            while not self._stop.is_set():
                quiet = _in_quiet_hours(
                    self.quiet_start, self.quiet_end, self._clock().astimezone().time()
                )
                if quiet:
                    if capture is not None:
                        self._release_capture(capture)
                        capture = None
                        if self.study_observer is not None:
                            self.study_observer.reset()
                        if self.visitor_recorder is not None:
                            self.visitor_recorder.stop("camera quiet hours")
                    self.state.set_quiet()
                    self._debug("camera quiet hours; webcam released")
                    present_count = absent_count = 0
                    if self._stop.wait(self.interval_seconds):
                        break
                    continue
                if capture is None:
                    try:
                        capture = (self._capture_factory or cv2.VideoCapture)(self.camera_index)
                        if not capture.isOpened():
                            self._release_capture(capture)
                            capture = None
                            self._degrade("camera could not be opened")
                            if self._stop.wait(self.interval_seconds):
                                break
                            continue
                        self._debug(f"camera opened: index={self.camera_index}")
                    except (OSError, RuntimeError, ValueError, AttributeError) as error:
                        self._degrade(f"camera could not be opened: {error}")
                        if self._stop.wait(self.interval_seconds):
                            break
                        continue
                if self._stop.wait(self.interval_seconds):
                    break
                try:
                    ok, frame = capture.read()
                    if not ok:
                        self._degrade("camera frame unavailable")
                        continue
                    boxes, scores = detector(frame)
                    detected = len(boxes) > 0
                    identity = None
                    if detected and self.owner_verifier is not None:
                        try:
                            identity = (
                                "owner"
                                if self.owner_verifier.verify(frame) is True
                                else "unknown"
                            )
                        except (OSError, RuntimeError, ValueError, AttributeError) as error:
                            identity = "unknown"
                            self._degrade(f"owner verification failed: {error}")
                    self._debug(
                        f"frame: person={'yes' if detected else 'no'} "
                        f"boxes={len(boxes)} identity={identity or 'not-checked'} "
                        f"scores={','.join(f'{score:.2f}' for score in scores)}"
                    )
                    if self.preview_enabled:
                        self._show_preview(cv2, frame, boxes, detected, identity)
                    if self.study_observer is not None and self.study_observer.active:
                        self.study_observer.observe(
                            frame, boxes, person_present=detected
                        )
                    present_count = present_count + 1 if detected else 0
                    absent_count = absent_count + 1 if not detected else 0
                    if present_count >= self.min_detections:
                        self.state.update(True, identity=identity)
                        if self.visitor_recorder is not None:
                            self.visitor_recorder.process(frame, True, identity)
                    elif absent_count >= self.min_absence:
                        self.state.update(False, identity="unknown" if self.owner_verifier else None)
                        if self.visitor_recorder is not None:
                            self.visitor_recorder.process(None, False, None)
                except (OSError, RuntimeError, ValueError, AttributeError) as error:
                    self._degrade(f"camera detection failed: {error}")
        finally:
            if capture is not None:
                self._release_capture(capture)
            if self.visitor_recorder is not None:
                self.visitor_recorder.stop("camera worker stopped")

    @staticmethod
    def _release_capture(capture: Any) -> None:
        try:
            capture.release()
        except (OSError, RuntimeError, AttributeError):
            pass

    def _degrade(self, reason: str) -> None:
        self.state.degrade(reason)
        print(f"EDITH camera presence degraded: {reason}")

    def _debug(self, message: str) -> None:
        if self.debug_enabled:
            print(f"EDITH camera debug: {message}")

    def _show_preview(self, cv2: Any, frame: Any, boxes: Any, detected: bool, identity: str | None) -> None:
        try:
            annotated = frame.copy()
            for box in boxes:
                x, y, width, height = (int(value) for value in box[:4])
                cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 0), 2)
            label = f"person={'yes' if detected else 'no'} identity={identity or 'not-checked'}"
            cv2.putText(
                annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2,
            )
            cv2.imshow("EDITH camera preview (Q to close)", annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                self.preview_enabled = False
                cv2.destroyWindow("EDITH camera preview (Q to close)")
                self._debug("camera preview closed by user")
        except (OSError, RuntimeError, ValueError, AttributeError) as error:
            self.preview_enabled = False
            self._debug(f"camera preview unavailable: {error}")


def _import_cv2() -> Any:
    import cv2

    return cv2


def _create_person_detector(
    cv2: Any,
    backend: str,
    model_path: str,
    confidence: float,
    iou: float,
    image_size: int,
    hog_threshold: float,
) -> tuple[str, Callable[[Any], tuple[list[Any], list[float]]]]:
    requested = backend if backend in {"auto", "yolo", "hog"} else "auto"
    if requested in {"auto", "yolo"}:
        try:
            from ultralytics import YOLO

            model = YOLO(model_path)

            def detect(frame: Any) -> tuple[list[Any], list[float]]:
                results = model(
                    frame,
                    classes=[0],
                    conf=confidence,
                    iou=iou,
                    imgsz=image_size,
                    verbose=False,
                )
                if not results or results[0].boxes is None:
                    return [], []
                result = results[0]
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                converted = [
                    (float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1]))
                    for box in boxes
                ]
                return converted, [float(score) for score in scores]

            return "yolo", detect
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            if requested == "yolo":
                raise RuntimeError(
                    "YOLO detector unavailable. Install requirements-vision.txt and "
                    f"provide CAMERA_MODEL_PATH ({error})"
                ) from error
            print(f"EDITH camera: YOLO unavailable; using HOG fallback: {error}")

    if not hasattr(cv2, "HOGDescriptor") or not hasattr(
        cv2, "HOGDescriptor_getDefaultPeopleDetector"
    ):
        raise RuntimeError(
            "No person detector available. Install requirements-vision.txt for YOLO."
        )
    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect_hog(frame: Any) -> tuple[list[Any], list[float]]:
        try:
            boxes, weights = detector.detectMultiScale(
                frame, winStride=(8, 8), padding=(16, 16), scale=1.05
            )
        except TypeError:
            boxes, weights = detector.detectMultiScale(frame)
        filtered = _filter_hog_detections(cv2, boxes, weights, hog_threshold)
        scores = []
        for box in filtered:
            for original, weight in zip(
                boxes if boxes is not None else [],
                weights if weights is not None else [],
            ):
                if tuple(original) == tuple(box):
                    scores.append(float(weight))
                    break
        return list(filtered), scores

    return "hog", detect_hog


def _filter_hog_detections(
    cv2: Any,
    boxes: Any,
    weights: Any,
    score_threshold: float,
) -> list[Any]:
    """Keep confident, non-overlapping HOG detections.

    OpenCV's default pedestrian SVM can produce weak boxes around furniture
    and doorways. The raw confidence scores and NMS are needed before a box
    is allowed to affect room presence state.
    """
    candidates = []
    for index, box in enumerate(boxes if boxes is not None else []):
        try:
            score = float(weights[index]) if weights is not None else 0.0
        except (IndexError, TypeError, ValueError):
            score = 0.0
        if score >= score_threshold:
            candidates.append((box, score))
    if len(candidates) < 2:
        return [box for box, _ in candidates]

    candidate_boxes = [tuple(int(value) for value in box[:4]) for box, _ in candidates]
    candidate_scores = [score for _, score in candidates]
    try:
        selected = cv2.dnn.NMSBoxes(
            candidate_boxes,
            candidate_scores,
            score_threshold,
            0.35,
        )
    except (AttributeError, TypeError, ValueError):
        return [box for box, _ in candidates]
    selected_indices = {
        int(item[0] if hasattr(item, "__len__") else item) for item in selected
    }
    return [
        box for index, (box, _) in enumerate(candidates) if index in selected_indices
    ]


def _in_quiet_hours(start: time | None, end: time | None, now: time | None = None) -> bool:
    if start is None or end is None or start == end:
        return False
    now = now or datetime.now().astimezone().time()
    return (start <= now < end) if start < end else (now >= start or now < end)
