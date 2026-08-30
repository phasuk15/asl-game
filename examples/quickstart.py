from __future__ import annotations

from game_mechanics.game_manager import GameManager


def main() -> None:
    manager = GameManager()
    tutorial = manager.run_tutorial()
    print("Tutorial complete:", tutorial.is_complete())

    wordle = manager.start_wordle_session()
    print("Target word:", wordle.target_word)
    print("Guess result:", manager.process_guess(wordle.target_word))


if __name__ == "__main__":
    main()
