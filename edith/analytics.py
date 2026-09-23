from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from typing import Any

from .storage import GoogleDriveStorage


class BehaviorAnalytics:
    def __init__(self, storage: GoogleDriveStorage) -> None:
        self.storage = storage

    def report(self, period: str, end_date: date | None = None) -> dict[str, Any]:
        if period not in {"weekly", "monthly"}:
            raise ValueError("period must be 'weekly' or 'monthly'")
        end = end_date or date.today()
        start = end - timedelta(days=6 if period == "weekly" else 29)
        plans = [
            plan for name, plan in self.storage.list_json("plans")
            if _in_range(plan.get("date", ""), start, end)
        ]
        slots = [slot for plan in plans for slot in plan.get("slots", [])]
        goals = [goal for plan in plans for goal in plan.get("goals", [])]
        tracked = slots + goals
        completed = [item for item in tracked if item.get("status") == "completed"]
        missed = [item for item in tracked if item.get("status") == "missed"]
        finished_or_missed = len(completed) + len(missed)
        reasons = Counter(
            item.get("missed_reason", "").strip()
            for item in missed
            if item.get("missed_reason", "").strip()
        )
        report = {
            "period": period,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "days_with_plans": len(plans),
            "planned_items": len(tracked),
            "completed_items": len(completed),
            "missed_items": len(missed),
            "completion_rate": round(
                len(completed) / finished_or_missed * 100, 1
            ) if finished_or_missed else 0.0,
            "unfinished_items": sum(
                item.get("status") in {"planned", "in_progress"} for item in tracked
            ),
            "missed_reasons": dict(reasons.most_common(10)),
            "completed_tasks": [
                item.get("title", item.get("text", "")) for item in completed
            ],
            "recommendations": _recommendations(len(completed), len(missed), reasons),
        }
        self.storage.write_json("reports", f"{period}-{end.isoformat()}.json", report)
        return report

    def format_report(self, period: str, end_date: date | None = None) -> str:
        report = self.report(period, end_date)
        reasons = ", ".join(
            f"{reason} ({count})"
            for reason, count in report["missed_reasons"].items()
        ) or "none recorded"
        return (
            f"{period.title()} report ({report['start']} to {report['end']}): "
            f"{report['completed_items']} completed, {report['missed_items']} missed, "
            f"{report['unfinished_items']} unfinished. "
            f"Completion rate: {report['completion_rate']}%. "
            f"Common missed-task reasons: {reasons}. "
            f"Recommendations: {' '.join(report['recommendations'])}"
        )


def _in_range(value: str, start: date, end: date) -> bool:
    try:
        current = date.fromisoformat(value)
    except ValueError:
        return False
    return start <= current <= end


def _recommendations(
    completed: int, missed: int, reasons: Counter[str]
) -> list[str]:
    if not completed and not missed:
        return ["Create time slots so EDITH can measure progress."]
    recommendations: list[str] = []
    if missed > completed:
        recommendations.append("Schedule fewer or shorter tasks in each day.")
    if reasons:
        recommendations.append(
            f"Address the most frequent blocker: {reasons.most_common(1)[0][0]}."
        )
    if completed >= missed:
        recommendations.append("Keep the current planning rhythm and protect completed slots.")
    return recommendations
