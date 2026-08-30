from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .config import WORDLE_ALLOWED_WORDS


@dataclass
class GuessFeedback:
    guess: str
    pattern: List[str] = field(default_factory=list)
    correct: bool = False


class WordleGame:
    """A simple Wordle-style game using ASL vocabulary words."""

    def __init__(self, target_word: str | None = None, allowed_words: List[str] | None = None):
        self.allowed_words = allowed_words or WORDLE_ALLOWED_WORDS
        self.target_word = target_word or self.allowed_words[0]
        self.guesses: List[str] = []
        self.max_guesses = 6

    def reset(self, target_word: str | None = None) -> None:
        self.target_word = target_word or self.allowed_words[0]
        self.guesses = []

    def validate_guess(self, guess: str) -> bool:
        guess = guess.strip().upper()
        return len(guess) == len(self.target_word) and guess in self.allowed_words

    def evaluate_guess(self, guess: str) -> GuessFeedback:
        if not self.validate_guess(guess):
            raise ValueError(f"Invalid guess: {guess}")

        pattern: List[str] = []
        for i, ch in enumerate(guess):
            if ch == self.target_word[i]:
                pattern.append("GREEN")
            elif ch in self.target_word:
                pattern.append("YELLOW")
            else:
                pattern.append("GREY")

        feedback = GuessFeedback(guess=guess, pattern=pattern, correct=guess == self.target_word)
        self.guesses.append(guess)
        return feedback

    def is_won(self) -> bool:
        return any(guess == self.target_word for guess in self.guesses)

    def is_lost(self) -> bool:
        return len(self.guesses) >= self.max_guesses and not self.is_won()
