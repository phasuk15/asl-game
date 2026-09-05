from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

# Column order used everywhere a participant summary is written (CSV header,
# per-participant JSON, tests) so the on-disk format stays stable.
SUMMARY_FIELDS = [
    "participant_id",
    "started_at",
    "ended_at",
    "session_duration_seconds",
    "tutorial_complete",
    "tutorial_lessons_completed",
    "tutorial_total_lessons",
    "tutorial_avg_response_time_seconds",
    "wordle_attempts",
    "wordle_correct_guesses",
    "wordle_won",
    "wordle_avg_response_time_seconds",
    "rally_attempts",
    "rally_correct_attempts",
    "rally_best_streak",
    "rally_avg_response_time_seconds",
    "total_correct",
    "total_attempts",
    "accuracy_percent",
]


def _normalise(value: str | None) -> str:
    return (value or "").strip().upper()


@dataclass
class PerformanceEvent:
    """A single scored interaction: one tutorial sign or one wordle guess."""

    phase: str
    target: str
    submitted: str
    correct: bool
    response_time_seconds: float
    timestamp: float = field(default_factory=time.time)


class SessionRecorder:
    """Captures every scored interaction for a single participant's session."""

    def __init__(self, participant_id: str):
        self.participant_id = participant_id
        self.started_at = time.time()
        self.ended_at: float | None = None
        self.events: List[PerformanceEvent] = []

    def log_event(
        self,
        phase: str,
        target: str,
        submitted: str,
        correct: bool,
        response_time_seconds: float,
    ) -> PerformanceEvent:
        event = PerformanceEvent(
            phase=phase,
            target=_normalise(target),
            submitted=_normalise(submitted),
            correct=bool(correct),
            response_time_seconds=round(max(0.0, float(response_time_seconds)), 3),
        )
        self.events.append(event)
        return event

    def events_for(self, phase: str) -> List[PerformanceEvent]:
        return [event for event in self.events if event.phase == phase]

    def finish(self) -> None:
        if self.ended_at is None:
            self.ended_at = time.time()

    @property
    def is_finished(self) -> bool:
        return self.ended_at is not None

    @property
    def duration_seconds(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.time()
        return round(end - self.started_at, 3)


def _average_response_time(events: List[PerformanceEvent]) -> float:
    if not events:
        return 0.0
    return round(sum(event.response_time_seconds for event in events) / len(events), 3)


def build_summary(recorder: SessionRecorder, manager) -> dict:
    """Combine a recorder's raw events with the manager's final state into one row."""
    tutorial_events = recorder.events_for("tutorial")
    wordle_events = recorder.events_for("wordle")
    rally_events = recorder.events_for("rally")
    rally = getattr(manager, "rally", None)

    total_attempts = manager.progress.total_attempts
    total_correct = manager.progress.total_correct
    accuracy_percent = round(100.0 * total_correct / total_attempts, 2) if total_attempts else 0.0

    ended_at = recorder.ended_at if recorder.ended_at is not None else time.time()

    return {
        "participant_id": recorder.participant_id,
        "started_at": datetime.fromtimestamp(recorder.started_at).isoformat(timespec="seconds"),
        "ended_at": datetime.fromtimestamp(ended_at).isoformat(timespec="seconds"),
        "session_duration_seconds": recorder.duration_seconds,
        "tutorial_complete": manager.progress.tutorial_complete,
        "tutorial_lessons_completed": sum(1 for lesson in manager.tutorial.lessons if lesson.completed),
        "tutorial_total_lessons": len(manager.tutorial.lessons),
        "tutorial_avg_response_time_seconds": _average_response_time(tutorial_events),
        "wordle_attempts": len(wordle_events),
        "wordle_correct_guesses": sum(1 for event in wordle_events if event.correct),
        "wordle_won": manager.wordle.is_won(),
        "wordle_avg_response_time_seconds": _average_response_time(wordle_events),
        "rally_attempts": len(rally_events),
        "rally_correct_attempts": sum(1 for event in rally_events if event.correct),
        "rally_best_streak": rally.best_streak if rally else 0,
        "rally_avg_response_time_seconds": _average_response_time(rally_events),
        "total_correct": total_correct,
        "total_attempts": total_attempts,
        "accuracy_percent": accuracy_percent,
    }
