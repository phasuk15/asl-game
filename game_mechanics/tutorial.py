from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .config import TUTORIAL_WORDS


@dataclass
class TutorialLesson:
    word: str
    prompt: str
    expected_sign: str | None = None
    completed: bool = False


@dataclass
class TutorialPhase:
    lessons: List[TutorialLesson] = field(default_factory=lambda: [
        TutorialLesson(word=w, prompt=f"Practice the ASL sign for {w}.", expected_sign=w)
        for w in TUTORIAL_WORDS
    ])
    current_index: int = 0

    def current_lesson(self) -> TutorialLesson:
        return self.lessons[self.current_index]

    def advance(self) -> TutorialLesson | None:
        if self.current_index >= len(self.lessons) - 1:
            return None
        self.current_index += 1
        return self.current_lesson()

    def mark_completed(self, lesson: TutorialLesson | None = None) -> TutorialLesson:
        target = lesson or self.current_lesson()
        target.completed = True
        return target

    def is_complete(self) -> bool:
        return all(lesson.completed for lesson in self.lessons)
