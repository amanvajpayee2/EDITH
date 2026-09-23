from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
import re
from typing import Any
from uuid import uuid4

from .storage import GoogleDriveStorage


@dataclass(frozen=True)
class Goal:
    id: str
    text: str
    status: str = "planned"
    due_at: str = ""


@dataclass(frozen=True)
class TimeSlot:
    id: str
    title: str
    start: str
    end: str
    status: str = "planned"
    completion_note: str = ""
    missed_reason: str = ""


@dataclass(frozen=True)
class PlanMutation:
    plan_date: str
    goals: tuple[Any, ...] = ()
    slots: tuple[TimeSlot, ...] = ()
    remove_terms: tuple[str, ...] = ()


class DailyPlanner:
    def __init__(self, storage: GoogleDriveStorage) -> None:
        self.storage = storage

    def get(self, plan_date: date) -> dict[str, Any]:
        return self.storage.read_json(
            "plans",
            f"{plan_date.isoformat()}.json",
            {"date": plan_date.isoformat(), "goals": [], "slots": []},
        )

    def save(self, plan: dict[str, Any]) -> None:
        self.storage.write_json("plans", f"{plan['date']}.json", plan)

    def add(self, mutation: PlanMutation) -> dict[str, Any]:
        plan_date = date.fromisoformat(mutation.plan_date)
        plan = self.get(plan_date)
        for goal in mutation.goals:
            if isinstance(goal, dict):
                goal["id"] = str(uuid4())
                plan["goals"].append(goal)
            else:
                plan["goals"].append(asdict(Goal(id=str(uuid4()), text=goal)))
        plan["slots"].extend(asdict(slot) for slot in mutation.slots)
        for term in mutation.remove_terms:
            _remove_matching(plan, term)
        plan["goals"] = _unique_by_id(plan["goals"])
        plan["slots"] = _unique_by_id(plan["slots"])
        self.save(plan)
        return plan

    def update_slot(
        self,
        plan_date: date,
        slot_id: str,
        status: str,
        completion_note: str = "",
        missed_reason: str = "",
    ) -> dict[str, Any]:
        if status not in {"planned", "in_progress", "completed", "missed", "skipped"}:
            raise ValueError(f"Unsupported slot status: {status}")
        plan = self.get(plan_date)
        for slot in plan["slots"]:
            if slot["id"] == slot_id:
                slot.update(
                    status=status,
                    completion_note=completion_note,
                    missed_reason=missed_reason,
                )
                self.save(plan)
                return plan
        raise ValueError(f"Time slot {slot_id!r} was not found for {plan_date.isoformat()}")

    def current_slot(self, now: datetime | None = None) -> dict[str, Any] | None:
        current = now or datetime.now().astimezone()
        plan = self.get(current.date())
        for slot in plan["slots"]:
            start = datetime.fromisoformat(slot["start"])
            end = datetime.fromisoformat(slot["end"])
            if start <= current <= end:
                return slot
        return None

    def prompt_context(self, plan_date: date) -> str:
        plan = self.get(plan_date)
        goals = ", ".join(
            goal["text"] + (f" (due by {_clock(goal['due_at'])})" if goal.get("due_at") else "")
            for goal in plan["goals"]
        ) or "none"
        slots = "; ".join(
            f"{_clock(slot['start'])}-{_clock(slot['end'])}: {slot['title']} ({slot['status']})"
            for slot in plan["slots"]
        ) or "none"
        return f"Plan for {plan['date']}. Goals: {goals}. Time slots: {slots}."


def parse_planning_command(text: str, now: datetime | None = None) -> PlanMutation | None:
    """Parse explicit commands and natural voice-style planning statements.

    Examples:
      plan today goals: finish project, exercise; 2pm to 4pm: leetcode
      schedule 2026-09-15: 09:00-10:30 team meeting
    """
    match = re.match(
        r"^\s*(?:plan|schedule)\s+(?P<day>today|tomorrow|\d{4}-\d{2}-\d{2})\s*:?\s*(?P<body>.+)$",
        text,
        re.IGNORECASE,
    )
    natural = re.search(
        r"\b(?:today|tomorrow|before\s+\d|between\s+\d|from\s+\d|at\s+\d|"
        r"set\s+(?:an\s+)?alarm|add|remove|delete|cancel)\b",
        text,
        re.IGNORECASE,
    )
    if not match and not natural:
        return None
    current = (now or datetime.now().astimezone()).date()
    alarm_match = re.search(
        r"\bset\s+(?:an\s+)?alarm\s+for\s+"
        r"(?P<time>\d{1,2}(?:\s*:\s*\d{2}|\s+\d{2})?\s*(?:am|pm))"
        r"(?:\s+(?:and\s+)?remind\s+me\s+(?:that\s+)?)?(?P<note>.*)$",
        text,
        re.IGNORECASE,
    )
    if alarm_match:
        alarm_time = _parse_clock(alarm_match.group("time"), current)
        current_time = (now or datetime.now().astimezone())
        if alarm_time <= current_time:
            alarm_time += timedelta(days=1)
        note = alarm_match.group("note").strip(" .")
        if not note:
            note = "Alarm"
        return PlanMutation(
            plan_date=alarm_time.date().isoformat(),
            slots=(
                TimeSlot(
                    id=str(uuid4()),
                    title=note,
                    start=alarm_time.isoformat(),
                    end=(alarm_time + timedelta(minutes=1)).isoformat(),
                ),
            ),
        )
    day_text = match.group("day").lower() if match else "today"
    plan_date = current + timedelta(days=1) if day_text == "tomorrow" else current
    if day_text not in {"today", "tomorrow"}:
        plan_date = date.fromisoformat(day_text)
    body = match.group("body").strip() if match else text.strip()
    remove_terms: list[str] = []
    remove_match = re.search(r"\b(?:remove|delete|cancel)\s+(?:the\s+)?(.+)$", body, re.IGNORECASE)
    if remove_match:
        remove_terms.extend(item.strip() for item in re.split(r",| and ", remove_match.group(1)) if item.strip())
        body = body[:remove_match.start()].strip()
    goals: list[str] = []
    slots: list[TimeSlot] = []
    reminder_matches = list(
        re.finditer(
            r"\b(?:remind\s+me|set\s+(?:an\s+)?alarm)\s+(?:at|for)\s+"
            r"(?P<time>\d{1,2}(?:(?::|\.)\d{2}|\s+\d{2})?\s*(?:am|pm)?)"
            r"(?:\s+(?:to|that)\s+)?(?P<title>[^,;]+?)(?=\s+\band\s+(?:remind\s+me|set\s+)|[,;]|$)",
            body,
            re.IGNORECASE,
        )
    )
    for reminder_match in reminder_matches:
        start = _parse_clock(reminder_match.group("time"), plan_date)
        title = re.sub(
            r"^(?:to|that)\s+",
            "",
            reminder_match.group("title").strip(" ."),
            flags=re.IGNORECASE,
        )
        slots.append(
            TimeSlot(
                id=str(uuid4()),
                title=title or "Reminder",
                start=start.isoformat(),
                end=(start + timedelta(minutes=1)).isoformat(),
            )
        )
    if reminder_matches:
        body = re.sub(
            r"\b(?:remind\s+me|set\s+(?:an\s+)?alarm)\s+(?:at|for)\s+"
            r"\d{1,2}(?:(?::|\.)\d{2}|\s+\d{2})?\s*(?:am|pm)?"
            r"(?:\s+(?:to|that)\s+)?[^,;]+?(?=\s+\band\s+(?:remind\s+me|set\s+)|[,;]|$)",
            "",
            body,
            flags=re.IGNORECASE,
        )
    range_pattern = re.compile(
        r"(?:between\s+|from\s+)?(?P<start>\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*"
        r"(?:to|-|until)\s*(?P<end>\d{1,2}(?::\d{2})?\s*(?:am|pm)?)"
        r"\s*(?:for|:)\s*(?P<title>[^,;]+)",
        re.IGNORECASE,
    )
    for range_match in range_pattern.finditer(body):
        start = _parse_clock(range_match.group("start"), plan_date)
        end = _parse_clock(range_match.group("end"), plan_date)
        if end <= start:
            raise ValueError("A time slot must end after it starts")
        slots.append(
            TimeSlot(
                id=str(uuid4()),
                title=range_match.group("title").strip(),
                start=start.isoformat(),
                end=end.isoformat(),
            )
        )
    body_without_ranges = range_pattern.sub("", body)
    trailing_range_pattern = re.compile(
        r"and\s+(?:also\s+)?(?:.*\band\s+)?"
        r"(?P<title>[^,;]+?)\s+(?:between|from)\s+"
        r"(?P<start>\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*(?:to|-|until)\s*"
        r"(?P<end>\d{1,2}(?::\d{2})?\s*(?:am|pm)?)",
        re.IGNORECASE,
    )
    for range_match in trailing_range_pattern.finditer(body_without_ranges):
        preceding = range_match.group(0).rsplit("and", 1)[0]
        preceding = re.sub(r"^\s*and\s+(?:also\s+)?", "", preceding, flags=re.IGNORECASE).strip()
        if preceding:
            goals.append(preceding)
        start = _parse_clock(range_match.group("start"), plan_date)
        end = _parse_clock(range_match.group("end"), plan_date)
        if end <= start:
            raise ValueError("A time slot must end after it starts")
        slots.append(
            TimeSlot(
                id=str(uuid4()),
                title=range_match.group("title").strip(),
                start=start.isoformat(),
                end=end.isoformat(),
            )
        )
    body_without_ranges = trailing_range_pattern.sub("", body_without_ranges)
    for chunk in re.split(r"\s*;\s*|\s+\band also\b\s+|\s+\band\b\s+", body_without_ranges, flags=re.IGNORECASE):
        slot_match = re.match(
            r"(?P<start>\d{1,2}(?::\d{2})?\s*(?:am|pm)?|\d{1,2}:\d{2})\s*"
            r"(?:to|-)\s*(?P<end>\d{1,2}(?::\d{2})?\s*(?:am|pm)?|\d{1,2}:\d{2})"
            r"\s*:?\s*(?P<title>.+)$",
            chunk,
            re.IGNORECASE,
        )
        if slot_match:
            start = _parse_clock(slot_match.group("start"), plan_date)
            end = _parse_clock(slot_match.group("end"), plan_date)
            if end <= start:
                raise ValueError("A time slot must end after it starts")
            slots.append(
                TimeSlot(
                    id=str(uuid4()),
                    title=slot_match.group("title").strip(),
                    start=start.isoformat(),
                    end=end.isoformat(),
                )
            )
        else:
            goal_text = re.sub(r"^(?:goals?|goal)\s*:?\s*", "", chunk, flags=re.IGNORECASE)
            goal_text = re.sub(r"^(?:today|tomorrow)\s+", "", goal_text, flags=re.IGNORECASE)
            before_match = re.search(
                r"\b(?:before|by)\s+(?P<time>\d{1,2}(?:(?::|\.)\d{2}|\s+\d{2})?\s*(?:am|pm)?)\b",
                goal_text,
                re.IGNORECASE,
            )
            due_at = ""
            if before_match:
                due_at = _parse_clock(before_match.group("time"), plan_date).isoformat()
                goal_text = goal_text[:before_match.start()].strip()
            at_match = re.search(
                r"\bat\s+(?P<time>\d{1,2}(?:(?::|\.)\d{2}|\s+\d{2})?\s*(?:am|pm)?)\b",
                goal_text,
                re.IGNORECASE,
            )
            if at_match:
                at_time = _parse_clock(at_match.group("time"), plan_date)
                suffix = goal_text[at_match.end():].strip()
                if suffix:
                    slots.append(
                        TimeSlot(
                            id=str(uuid4()),
                            title=re.sub(
                                r"^(?:i\s+need\s+to|need\s+to)\s+",
                                "",
                                suffix,
                                flags=re.IGNORECASE,
                            ),
                            start=at_time.isoformat(),
                            end=(at_time + timedelta(minutes=15)).isoformat(),
                        )
                    )
                    goal_text = goal_text[:at_match.start()].strip()
                else:
                    due_at = at_time.isoformat()
                    goal_text = goal_text[:at_match.start()].strip()
            for item in goal_text.split(","):
                item = re.sub(
                    r"^(?:i\s+need\s+to|need\s+to|do|add(?:\s+that)?)\s*",
                    "",
                    item.strip(),
                    flags=re.IGNORECASE,
                )
                if item:
                    goals.append(asdict(Goal(id=str(uuid4()), text=item, due_at=due_at)))
    if not goals and not slots and not remove_terms:
        raise ValueError("No goals or time slots were found in the planning command")
    return PlanMutation(
        plan_date=plan_date.isoformat(),
        goals=tuple(goals),
        slots=tuple(slots),
        remove_terms=tuple(remove_terms),
    )


def _parse_clock(value: str, plan_date: date) -> datetime:
    normalized = re.sub(r"\s+", "", value.lower().replace(".", ":"))
    normalized = re.sub(r"^(\d{1,2})(\d{2})(am|pm)$", r"\1:\2\3", normalized)
    formats = ("%I:%M%p", "%I%p", "%H:%M")
    for clock_format in formats:
        try:
            parsed = datetime.strptime(normalized, clock_format).time()
            return datetime.combine(plan_date, parsed).astimezone()
        except ValueError:
            continue
    raise ValueError(f"Invalid time: {value}")


def _clock(value: str) -> str:
    return datetime.fromisoformat(value).strftime("%H:%M")


def _unique_by_id(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list({item["id"]: item for item in items}.values())


def _remove_matching(plan: dict[str, Any], term: str) -> None:
    needle = term.lower()
    plan["goals"] = [goal for goal in plan["goals"] if needle not in goal["text"].lower()]
    plan["slots"] = [slot for slot in plan["slots"] if needle not in slot["title"].lower()]
