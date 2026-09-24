from __future__ import annotations

from datetime import date, datetime, time as clock_time, timedelta
import threading
import time
from typing import Callable
from dataclasses import asdict

from .planning import DailyPlanner
from .presence import PresenceState
from .storage import GoogleDriveStorage
from .study_observer import StudyObserver, StudySessionSummary


class ReminderEngine:
    """Persistent reminder/check-in state machine for today's plan."""

    def __init__(
        self,
        storage: GoogleDriveStorage,
        advance_minutes: int = 15,
        daily_prompt_time: clock_time = clock_time(7, 0),
        daily_retry_minutes: int = 15,
        study_observer: StudyObserver | None = None,
        study_slot_title_patterns: tuple[str, ...] = (),
    ) -> None:
        self.planner = DailyPlanner(storage)
        self.advance = timedelta(minutes=advance_minutes)
        self.storage = storage
        self.daily_prompt_time = daily_prompt_time
        self.daily_retry = timedelta(minutes=daily_retry_minutes)
        self.study_observer = study_observer
        self.study_slot_title_patterns = tuple(
            value.casefold() for value in study_slot_title_patterns if value.strip()
        )

    def tick(self, now: datetime | None = None) -> list[str]:
        current = now or datetime.now().astimezone()
        plan = self.planner.get(current.date())
        notifications: list[str] = []
        changed = False
        daily_state = self.storage.read_json("metadata", "daily-prompt.json", {})
        day_key = current.date().isoformat()
        state = daily_state.get(day_key, {})
        prompt_at = datetime.combine(
            current.date(), self.daily_prompt_time, tzinfo=current.tzinfo
        )
        if current >= prompt_at and not state.get("answered"):
            last_prompt = _parse_timestamp(state.get("last_prompt"))
            if last_prompt is None or current - last_prompt >= self.daily_retry:
                state["last_prompt"] = current.isoformat()
                daily_state[day_key] = state
                self.storage.write_json("metadata", "daily-prompt.json", daily_state)
                notifications.append(
                    "Good morning. What are your goals and time slots for today?"
                )

        for slot in plan.get("slots", []):
            start = datetime.fromisoformat(slot["start"])
            end = datetime.fromisoformat(slot["end"])
            events = slot.setdefault("reminders", {})
            if slot["status"] in {"completed", "skipped"}:
                continue
            if start - self.advance <= current < start and not events.get("before"):
                events["before"] = current.isoformat()
                notifications.append(
                    f"Your upcoming task is {slot['title']} at {start.strftime('%H:%M')}."
                )
                changed = True
            if start <= current < end and not events.get("started"):
                events["started"] = current.isoformat()
                slot["status"] = "in_progress"
                if self._is_study_slot(slot) and self.study_observer is not None:
                    self.study_observer.start_slot(slot["id"])
                notifications.append(f"It's time for {slot['title']}.")
                changed = True
            if current >= end and not events.get("checkin"):
                if self._is_study_slot(slot) and self.study_observer is not None:
                    summary = self.study_observer.end_slot()
                    slot["study_summary"] = asdict(summary)
                events["checkin"] = current.isoformat()
                self._set_active_checkin(current.date(), slot["id"], "slot")
                notifications.append(self._study_checkin(slot) if self._is_study_slot(slot)
                                     else f"Your {slot['title']} slot has ended. Did you complete it?")
                changed = True

        for goal in plan.get("goals", []):
            due_at = goal.get("due_at")
            if not due_at or goal.get("status") in {"completed", "skipped"}:
                continue
            due = datetime.fromisoformat(due_at)
            events = goal.setdefault("reminders", {})
            if due - self.advance <= current < due and not events.get("before"):
                events["before"] = current.isoformat()
                notifications.append(
                    f"Reminder: {goal['text']} is due by {due.strftime('%H:%M')}."
                )
                changed = True
            if current >= due and not events.get("checkin"):
                events["checkin"] = current.isoformat()
                self._set_active_checkin(current.date(), goal["id"], "goal")
                notifications.append(f"Did you finish {goal['text']}?")
                changed = True

        if changed:
            self.planner.save(plan)
        return notifications

    def acknowledge_daily_plan(self, plan_date: date) -> None:
        daily_state = self.storage.read_json("metadata", "daily-prompt.json", {})
        state = daily_state.setdefault(plan_date.isoformat(), {})
        state["answered"] = True
        state["answered_at"] = datetime.now().astimezone().isoformat()
        self.storage.write_json("metadata", "daily-prompt.json", daily_state)

    def missed_daily_prompt(self, now: datetime | None = None) -> bool:
        current = now or datetime.now().astimezone()
        state = self.storage.read_json(
            "metadata", "daily-prompt.json", {}
        ).get(current.date().isoformat(), {})
        prompt_at = datetime.combine(
            current.date(), self.daily_prompt_time, tzinfo=current.tzinfo
        )
        return current >= prompt_at and not state.get("answered")

    def consume_response(self, text: str) -> str | None:
        current = datetime.now().astimezone()
        return_session = self.storage.read_json("metadata", "return-session.json", {})
        if (
            return_session.get("date") == current.date().isoformat()
            and return_session.get("queue")
        ):
            response = self._consume_return_response(text, current, return_session)
            if response is not None:
                return response
        plan = self.planner.get(current.date())
        active = self.storage.read_json("metadata", "active-checkin.json", {})
        if (
            active.get("date") != current.date().isoformat()
            or not active.get("item_id")
        ):
            return None
        for item in [*plan.get("slots", []), *plan.get("goals", [])]:
            if item.get("id") != active["item_id"]:
                continue
            events = item.get("reminders", {})
            if events.get("awaiting_reason"):
                item["status"] = "missed"
                item["missed_reason"] = text.strip()
                events.pop("awaiting_reason")
                if active.get("item_type") == "slot" and self._is_study_slot(item):
                    events["awaiting_reschedule"] = True
                self.planner.save(plan)
                if events.get("awaiting_reschedule"):
                    return "I recorded the reason. Would you like to reschedule this study slot later today or tomorrow?"
                self._clear_active_checkin()
                return "I recorded that as missed with your reason."
            if events.get("awaiting_reschedule") and active.get("item_type") == "slot":
                day = _reschedule_day(text)
                if day is None:
                    return "Say later today, tomorrow, or no to leave it missed."
                if day is False:
                    events.pop("awaiting_reschedule")
                    self.planner.save(plan)
                    self._clear_active_checkin()
                    return "Okay, I left the study slot missed."
                self._reschedule_slot(item, current, day)
                events.pop("awaiting_reschedule")
                self.planner.save(plan)
                self._clear_active_checkin()
                return f"Okay, I rescheduled it for {day.strftime('%A')}."
            if _is_positive(text):
                item["status"] = "completed"
                item["completion_note"] = text.strip()
                self.planner.save(plan)
                self._clear_active_checkin()
                return "Great, I marked that task completed."
            if _is_negative(text):
                events["awaiting_reason"] = True
                self.planner.save(plan)
                return "What prevented you from completing it?"
        return None

    def begin_return_session(self, now: datetime | None = None) -> str | None:
            """Create (or resume) the durable review of work missed while away."""
            current = now or datetime.now().astimezone()
            stored = self.storage.read_json("metadata", "return-session.json", {})
            if stored.get("date") == current.date().isoformat() and stored.get("queue"):
                return self._return_prompt(stored)
            plan = self.planner.get(current.date())
            queue = []
            for item_type, items, due_key in (
                ("slot", plan.get("slots", []), "end"),
                ("goal", plan.get("goals", []), "due_at"),
            ):
                for item in items:
                    if item.get("status") in {"completed", "skipped"}:
                        continue
                    due = item.get(due_key)
                    if not due or datetime.fromisoformat(due) > current:
                        continue
                    queue.append({"item_id": item["id"], "item_type": item_type})
            if not queue:
                self.storage.write_json("metadata", "return-session.json", {})
                return None
            session = {
                "date": current.date().isoformat(),
                "queue": queue,
                "index": 0,
                "awaiting_reason": False,
                "awaiting_reschedule": False,
                "started_at": current.isoformat(),
            }
            self.storage.write_json("metadata", "return-session.json", session)
            return self._return_prompt(session)

    def _consume_return_response(
            self, text: str, current: datetime, session: dict
    ) -> str | None:
            index = int(session.get("index", 0))
            queue = session.get("queue", [])
            if index >= len(queue):
                self.storage.write_json("metadata", "return-session.json", {})
                return None
            plan_date = date.fromisoformat(session["date"])
            plan = self.planner.get(plan_date)
            entry = queue[index]
            items = [*plan.get("slots", []), *plan.get("goals", [])]
            item = next((value for value in items if value.get("id") == entry["item_id"]), None)
            if item is None:
                return self._advance_return_session(session, plan, current)
            events = item.setdefault("reminders", {})
            if session.get("awaiting_reason"):
                reason = text.strip()
                if not reason:
                    return "Please tell me briefly what prevented you from completing it."
                item["status"] = "missed"
                item["missed_reason"] = reason
                session["awaiting_reason"] = False
                if entry["item_type"] == "slot":
                    session["awaiting_reschedule"] = True
                    self.planner.save(plan)
                    self.storage.write_json("metadata", "return-session.json", session)
                    return "Would you like to reschedule this task later today or tomorrow?"
                self.planner.save(plan)
                return self._advance_return_session(session, plan, current)
            if session.get("awaiting_reschedule"):
                day = _reschedule_day(text)
                if day is None:
                    return "Say later today, tomorrow, or no to leave it missed."
                session["awaiting_reschedule"] = False
                if day is not False:
                    self._reschedule_slot(item, current, day)
                    self.planner.save(plan)
                    return self._advance_return_session(
                        session, plan, current, f"Okay, I rescheduled it for {day.strftime('%A')}."
                    )
                self.planner.save(plan)
                return self._advance_return_session(session, plan, current, "Okay, I left it missed.")
            if _is_positive(text):
                item["status"] = "completed"
                item["completion_note"] = text.strip()
                self.planner.save(plan)
                return self._advance_return_session(session, plan, current)
            if _is_negative(text):
                session["awaiting_reason"] = True
                self.storage.write_json("metadata", "return-session.json", session)
                return "What prevented you from completing it?"
            return "Please answer yes or no. Did you complete this task?"

    def _advance_return_session(
        self, session: dict, plan: dict, current: datetime, prefix: str = ""
    ) -> str:
        session["index"] = int(session.get("index", 0)) + 1
        session["awaiting_reason"] = False
        session["awaiting_reschedule"] = False
        if session["index"] >= len(session.get("queue", [])):
            self.storage.write_json("metadata", "return-session.json", {})
            return (prefix + " All missed-task check-ins are complete.").strip()
        self.storage.write_json("metadata", "return-session.json", session)
        next_prompt = self._return_prompt(session)
        return f"{prefix} {next_prompt}".strip() if prefix else next_prompt

    def _return_prompt(self, session: dict) -> str:
        queue = session.get("queue", [])
        index = int(session.get("index", 0))
        if index >= len(queue):
            return "All missed-task check-ins are complete."
        plan = self.planner.get(date.fromisoformat(session["date"]))
        item_id = queue[index]["item_id"]
        item = next(
            (value for value in [*plan.get("slots", []), *plan.get("goals", [])]
             if value.get("id") == item_id),
            None,
        )
        title = (item or {}).get("title") or (item or {}).get("text") or "that task"
        return f"While you were away, did you complete {title}?"

    def _is_study_slot(self, slot: dict) -> bool:
        if bool(slot.get("study", slot.get("is_study", False))):
            return True
        title = str(slot.get("title", "")).casefold()
        return bool(self.study_slot_title_patterns) and any(
            pattern in title for pattern in self.study_slot_title_patterns
        )

    def _study_checkin(self, slot: dict) -> str:
        summary = slot.get("study_summary") or {}
        if not summary.get("conclusive"):
            detail = "The camera observations were inconclusive"
        else:
            detail = (
                f"I observed desk presence about {_percent(summary.get('desk_occupancy_fraction'))}, "
                f"bed presence about {_percent(summary.get('bed_occupancy_fraction'))}, "
                f"and low motion about {_percent(summary.get('low_motion_fraction'))}"
            )
        return (
            f"Your study slot has ended. {detail} (confidence "
            f"{_percent(summary.get('average_confidence'))}). Did you complete it? "
            "If not, what got in the way? I can reschedule later today or tomorrow."
        )

    @staticmethod
    def _reschedule_slot(slot: dict, current: datetime, day: date) -> None:
        start = datetime.fromisoformat(slot["start"])
        end = datetime.fromisoformat(slot["end"])
        duration = end - start
        requested = start.replace(year=day.year, month=day.month, day=day.day)
        if requested <= current and day == current.date():
            requested = current + timedelta(minutes=5)
        slot["start"] = requested.isoformat()
        slot["end"] = (requested + duration).isoformat()
        slot["status"] = "planned"
        slot["reminders"] = {}
        slot.pop("study_summary", None)
        slot["rescheduled_at"] = current.isoformat()

    def _set_active_checkin(self, plan_date: date, item_id: str, item_type: str) -> None:
        self.storage.write_json(
            "metadata",
            "active-checkin.json",
            {"date": plan_date.isoformat(), "item_id": item_id, "item_type": item_type},
        )

    def _clear_active_checkin(self) -> None:
        self.storage.write_json("metadata", "active-checkin.json", {})


class ReminderWorker:
    def __init__(
        self,
        engine: ReminderEngine,
        notify: Callable[[str], None],
        interval_seconds: int = 15,
        presence: PresenceState | None = None,
    ) -> None:
        self.engine = engine
        self.notify = notify
        self.interval_seconds = interval_seconds
        self.presence = presence
        self._deferred: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="edith-reminders", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval_seconds + 1)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                messages = self.engine.tick()
            except (OSError, RuntimeError) as error:
                print(f"EDITH reminder storage temporarily unavailable: {error}")
                continue
            if self.presence is not None and self.presence.should_defer:
                self._deferred.extend(messages)
                continue
            if self._deferred:
                messages = [*self._deferred, *messages]
                self._deferred.clear()
            for message in messages:
                self.notify(message)


def _percent(value: object) -> str:
    if value is None:
        return "unknown"
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "unknown"


def _reschedule_day(text: str) -> date | bool | None:
    lowered = text.casefold()
    if "no" in lowered or "leave" in lowered or "skip" in lowered:
        return False
    now = datetime.now().astimezone().date()
    if "tomorrow" in lowered:
        return now + timedelta(days=1)
    if "today" in lowered or "later" in lowered:
        return now
    return None


def _is_positive(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in {
            "yes",
            "yep",
            "yeah",
            "i did",
            "i finished",
            "i completed",
            "done",
            "finished",
            "completed",
        }
    )


def _is_negative(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in {
            "no",
            "nope",
            "i didn't",
            "i didnt",
            "didn't finish",
            "didnt finish",
            "not finished",
            "unfinished",
        }
    )


def _parse_timestamp(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None
