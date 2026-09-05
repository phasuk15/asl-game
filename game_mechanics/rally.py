from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import List

from .config import DYNAMIC_SIGN_WORDS, RALLY_PROMPT_TIMEOUT_SECONDS, RALLY_ROUND_SECONDS


def _normalise(value: str | None) -> str:
    return (value or "").strip().upper()


@dataclass
class RallyAttempt:
    """One scored prompt within a rally round."""

    prompt: str
    submitted: str
    correct: bool
    response_time_seconds: float
    streak_after: int
    timed_out: bool = False


class RallyGame:
    """Timed Rally / Streak Mode: a rapid-fire sequence of dynamic-sign
    prompts (motion trajectories, not static handshapes) within a fixed
    time budget.

    Each prompt must be signed before its own short timeout expires. A
    correct sign extends the streak; a wrong or missed one resets the
    streak to zero, but - unlike a "one mistake and you're out" mode - the
    round keeps going until the clock runs out (or an optional miss limit
    is hit). That keeps every round producing a dense stream of per-attempt
    response-time + accuracy data instead of stopping at the first mistake.
    """

    def __init__(
        self,
        words: List[str] | None = None,
        round_seconds: float = RALLY_ROUND_SECONDS,
        prompt_timeout_seconds: float = RALLY_PROMPT_TIMEOUT_SECONDS,
        max_misses: int | None = None,
        rng: random.Random | None = None,
    ):
        if round_seconds <= 0:
            raise ValueError("round_seconds must be positive")
        if prompt_timeout_seconds <= 0:
            raise ValueError("prompt_timeout_seconds must be positive")

        self.words = list(words or DYNAMIC_SIGN_WORDS)
        if not self.words:
            raise ValueError("RallyGame needs at least one word to prompt with")

        self.round_seconds = round_seconds
        self.prompt_timeout_seconds = prompt_timeout_seconds
        self.max_misses = max_misses
        self._rng = rng or random.Random()

        self.attempts: List[RallyAttempt] = []
        self.streak = 0
        self.best_streak = 0
        self.misses = 0
        self.current_prompt: str | None = None
        self.round_started_at: float | None = None
        self.prompt_started_at: float | None = None
        self.finished = False
        self.finish_reason: str | None = None

    def start(self, now: float | None = None) -> str:
        """Begin (or restart) a rally round and return the first prompt."""
        now = time.time() if now is None else now
        self.attempts = []
        self.streak = 0
        self.best_streak = 0
        self.misses = 0
        self.finished = False
        self.finish_reason = None
        self.round_started_at = now
        return self._next_prompt(now)

    def _next_prompt(self, now: float) -> str:
        choices = self.words
        if len(choices) > 1 and self.current_prompt in choices:
            # Avoid an immediate repeat so back-to-back prompts don't look stuck.
            choices = [word for word in choices if word != self.current_prompt]
        self.current_prompt = self._rng.choice(choices)
        self.prompt_started_at = now
        return self.current_prompt

    def time_remaining(self, now: float | None = None) -> float:
        if self.round_started_at is None:
            return self.round_seconds
        now = time.time() if now is None else now
        return max(0.0, self.round_seconds - (now - self.round_started_at))

    def check_timeout(self, now: float | None = None) -> dict | None:
        """Call this on every tick. If the current prompt has been up
        longer than its own timeout, it's registered as a miss (submitted
        empty) and the same result dict `submit()` would return is handed
        back. Returns None if nothing timed out (or no round is active)."""
        if self.finished or self.current_prompt is None or self.prompt_started_at is None:
            return None
        now = time.time() if now is None else now
        if (now - self.prompt_started_at) <= self.prompt_timeout_seconds:
            return None
        return self._register_result(submitted="", now=now, timed_out=True)

    def submit(self, label: str, now: float | None = None) -> dict:
        """Score a detected sign against the current prompt."""
        if self.finished or self.current_prompt is None:
            raise ValueError("Rally round is not active; call start() first.")
        now = time.time() if now is None else now
        return self._register_result(submitted=label, now=now, timed_out=False)

    def _register_result(self, submitted: str, now: float, timed_out: bool) -> dict:
        prompt = self.current_prompt
        started_at = now if self.prompt_started_at is None else self.prompt_started_at
        response_time = max(0.0, now - started_at)
        correct = (not timed_out) and bool(submitted) and _normalise(submitted) == _normalise(prompt)

        if correct:
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
        else:
            self.misses += 1
            self.streak = 0

        attempt = RallyAttempt(
            prompt=_normalise(prompt),
            submitted=_normalise(submitted),
            correct=correct,
            response_time_seconds=round(response_time, 3),
            streak_after=self.streak,
            timed_out=timed_out,
        )
        self.attempts.append(attempt)

        if self.max_misses is not None and self.misses >= self.max_misses:
            self.finished = True
            self.finish_reason = "out_of_lives"
        elif self.time_remaining(now) <= 0:
            self.finished = True
            self.finish_reason = "time_up"

        result = {
            "prompt": attempt.prompt,
            "submitted": attempt.submitted,
            "correct": correct,
            "timed_out": timed_out,
            "response_time_seconds": attempt.response_time_seconds,
            "streak": self.streak,
            "best_streak": self.best_streak,
            "misses": self.misses,
            "time_remaining": round(self.time_remaining(now), 2),
            "finished": self.finished,
        }

        if self.finished:
            result["finish_reason"] = self.finish_reason
            self.current_prompt = None
        else:
            result["next_prompt"] = self._next_prompt(now)

        return result

    def summary(self) -> dict:
        total = len(self.attempts)
        correct = sum(1 for attempt in self.attempts if attempt.correct)
        avg_response_time = (
            round(sum(attempt.response_time_seconds for attempt in self.attempts) / total, 3)
            if total
            else 0.0
        )
        return {
            "total_attempts": total,
            "correct_attempts": correct,
            "accuracy_percent": round(100.0 * correct / total, 2) if total else 0.0,
            "best_streak": self.best_streak,
            "final_streak": self.streak,
            "misses": self.misses,
            "avg_response_time_seconds": avg_response_time,
            "finished": self.finished,
            "finish_reason": self.finish_reason,
        }
