from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .model_adapter import ModelAdapter
from .tutorial import TutorialPhase
from .wordle import WordleGame


def _normalise_answer(value: str | None) -> str:
    return (value or "").strip().upper()


@dataclass
class PlayerProgress:
    tutorial_complete: bool = False
    unlocked_games: List[str] = field(default_factory=lambda: ["wordle"])
    total_correct: int = 0
    total_attempts: int = 0


class GameManager:
    """Coordinates the learning flow: tutorial -> games."""

    def __init__(self, model_adapter: ModelAdapter | None = None):
        self.model_adapter = model_adapter or ModelAdapter()
        self.tutorial = TutorialPhase()
        self.progress = PlayerProgress()
        self.wordle = WordleGame()

    def run_tutorial(self) -> TutorialPhase:
        """Run the tutorial loop until all lessons are complete."""
        while not self.tutorial.is_complete():
            lesson = self.tutorial.current_lesson()
            lesson.completed = True
            self.progress.total_correct += 1
            self.summarise_tutorial_progress()
            if self.tutorial.advance() is None:
                break

        self.progress.tutorial_complete = True
        return self.tutorial

    def summarise_tutorial_progress(self) -> dict:
        return {
            "current_index": self.tutorial.current_index,
            "completed": self.tutorial.is_complete(),
            "lessons_done": sum(1 for lesson in self.tutorial.lessons if lesson.completed),
            "total_lessons": len(self.tutorial.lessons),
        }

    def start_wordle_session(self, target_word: str | None = None) -> WordleGame:
        self.wordle.reset(target_word)
        return self.wordle

    def process_guess(self, guess: str) -> dict:
        feedback = self.wordle.evaluate_guess(guess)
        self.progress.total_attempts += 1
        if feedback.correct:
            self.progress.total_correct += 1
        return {
            "guess": feedback.guess,
            "pattern": feedback.pattern,
            "correct": feedback.correct,
            "won": self.wordle.is_won(),
            "lost": self.wordle.is_lost(),
        }

    def evaluate_sign(self, features) -> dict:
        prediction = self.model_adapter.predict(features)
        return {
            "label": prediction.label,
            "confidence": prediction.confidence,
            "available": self.model_adapter.is_available(),
        }

    def evaluate_user_response(
        self,
        expected_answer: str,
        submitted_answer: str,
        started_at: float,
        completed_at: float,
    ) -> dict:
        """Return an accuracy and response-time score for a user action."""
        expected = _normalise_answer(expected_answer)
        submitted = _normalise_answer(submitted_answer)

        is_correct = submitted == expected and bool(expected)
        elapsed = max(0.0, float(completed_at) - float(started_at))

        if not expected:
            accuracy_score = 0.0
        else:
            accuracy_score = 1.0 if is_correct else 0.0

        # Score is based on both correctness and speed. A fast correct answer gets
        # near the max of 100; slower answers receive a reduced score.
        if is_correct:
            speed_factor = max(0.0, 1.0 - (elapsed / 30.0))
            performance_score = round(100.0 * (0.7 + (0.3 * speed_factor)), 2)
        else:
            performance_score = 0.0

        return {
            "expected_answer": expected,
            "submitted_answer": submitted,
            "correct": is_correct,
            "accuracy_score": accuracy_score,
            "response_time_seconds": round(elapsed, 3),
            "performance_score": performance_score,
        }
