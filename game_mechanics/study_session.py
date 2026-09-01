from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .config import RESULTS_DIR
from .game_manager import GameManager
from .model_adapter import ModelAdapter
from .session_recorder import SUMMARY_FIELDS, SessionRecorder, build_summary


class StudySession:
    """Owns one participant's game state and performance log, and resets
    cleanly for the next participant in a user test.

    Typical usage during a user-testing session::

        study = StudySession()
        manager = study.start_participant("P001")
        ... play the game, calling study.record_tutorial_result(...) and
            study.record_wordle_guess(...) as the player interacts ...
        study.reset_for_next_player()  # saves P001's results, clears state
        manager = study.start_participant("P002")  # fresh run, no carry-over
    """

    def __init__(
        self,
        results_dir: str | Path | None = None,
        model_adapter: ModelAdapter | None = None,
    ):
        self.results_dir = Path(results_dir) if results_dir else RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.results_dir / "performance_log.csv"
        self.model_adapter = model_adapter or ModelAdapter()

        self.manager: GameManager | None = None
        self.recorder: SessionRecorder | None = None
        self._auto_counter = 0

    @property
    def is_active(self) -> bool:
        return self.manager is not None and self.recorder is not None

    @property
    def participant_id(self) -> str | None:
        return self.recorder.participant_id if self.recorder else None

    def _generate_participant_id(self) -> str:
        self._auto_counter += 1
        return f"P{self._auto_counter:03d}"

    def start_participant(self, participant_id: str | None = None) -> GameManager:
        """Begin a fresh session for a new player.

        If a participant is already active, their results are saved and the
        state is cleared first, so callers never need to call
        ``reset_for_next_player`` manually before starting the next test.
        """
        if self.is_active:
            self.reset_for_next_player()

        clean_id = (participant_id or "").strip() or self._generate_participant_id()
        self.manager = GameManager(model_adapter=self.model_adapter)
        self.recorder = SessionRecorder(clean_id)
        return self.manager

    def record_tutorial_result(self, word: str, correct: bool, response_time_seconds: float) -> None:
        """Log a scored attempt at a tutorial sign."""
        if not self.is_active:
            return
        self.recorder.log_event("tutorial", word, word if correct else "", correct, response_time_seconds)

    def record_wordle_guess(
        self,
        target: str,
        guess: str,
        correct: bool,
        response_time_seconds: float,
    ) -> None:
        """Log a scored wordle guess."""
        if not self.is_active:
            return
        self.recorder.log_event("wordle", target, guess, correct, response_time_seconds)

    def end_participant(self) -> dict | None:
        """Finalise and persist the active participant's results.

        Returns the summary dict, or None if no participant is active. The
        manager/recorder are left in place (finished) so callers can still
        read final state before calling ``reset_for_next_player``.
        """
        if not self.is_active:
            return None

        self.recorder.finish()
        summary = build_summary(self.recorder, self.manager)
        self._append_csv_row(summary)
        self._write_participant_json(summary)
        return summary

    def reset_for_next_player(self) -> dict | None:
        """Save the current participant's results (if any) and clear all
        game state so the next user test starts from a blank slate."""
        summary = self.end_participant()
        self.manager = None
        self.recorder = None
        return summary

    def _append_csv_row(self, summary: dict) -> None:
        write_header = not self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(summary)

    def _write_participant_json(self, summary: dict) -> None:
        participants_dir = self.results_dir / "participants"
        participants_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_id = "".join(char if char.isalnum() else "_" for char in self.recorder.participant_id)
        path = participants_dir / f"{safe_id}_{stamp}.json"
        payload = {
            "summary": summary,
            "events": [asdict(event) for event in self.recorder.events],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
