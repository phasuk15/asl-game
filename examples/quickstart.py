from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly (`python3 examples/quickstart.py`), not
# just as a module (`python3 -m examples.quickstart`), by putting the
# project root on sys.path so the sibling `game_mechanics` package resolves.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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
