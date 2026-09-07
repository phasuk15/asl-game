from game_mechanics.game_manager import GameManager
from game_mechanics.tutorial import TutorialPhase


def test_tutorial_phase_for_words_builds_matching_lessons():
    phase = TutorialPhase.for_words(["A", "B", "C"])

    assert [lesson.word for lesson in phase.lessons] == ["A", "B", "C"]
    assert phase.current_lesson().word == "A"
    assert phase.is_complete() is False


def test_game_manager_fingerspelling_covers_the_alphabet():
    manager = GameManager()

    words = [lesson.word for lesson in manager.fingerspelling.lessons]

    assert words == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert manager.fingerspelling is not manager.tutorial


def test_run_fingerspelling_completes_every_letter():
    manager = GameManager()

    manager.run_fingerspelling()

    assert manager.fingerspelling.is_complete() is True
    assert manager.progress.fingerspelling_complete is True
    assert all(lesson.completed for lesson in manager.fingerspelling.lessons)
