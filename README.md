# ASL Game Mechanics

This repository is the gameplay layer for an ASL learning system. It contains the mechanics for a tutorial phase and multiple game modes, including a Wordle-style game, while integrating with the trained models produced in the `asl-training` project.

## Project purpose

- Teach basic ASL gestures in a guided tutorial
- Convert live model predictions into game input
- Support multiple game loops such as Wordle-style guessing
- Keep model integration isolated so the training project remains responsible for model creation

## Folder structure

- `game_mechanics/`: reusable mechanics and game engine logic
- `examples/`: runnable examples showing how to initialise the system
- `pyproject.toml`: project metadata for Python packaging

## Integration with training models

The training project is expected to live beside this folder in the same dissertation workspace:

- `../asl-training`

The model adapter looks for pickled or joblib-trained models in the standard locations used by the training project so the game layer can consume the ASL recogniser without duplicating training logic.

## Typical flow

1. Run the tutorial phase to teach the user a small set of ASL signs.
2. Validate each sign using model predictions.
3. Unlock game modes like Wordle after the tutorial is complete.
4. Use the same model output to score guesses and evaluate progress.

## Quick start

```bash
python3 -m examples.quickstart
```

## Live webcam mode

This project can launch the trained ASL detector already implemented in the sibling `asl-training` project.

```bash
python3 -m examples.live_webcam
```

This wrapper resolves the sibling ASL training project, adds its source path to `PYTHONPATH`, and then runs the existing OpenCV + MediaPipe webcam detector that uses the model assets from that project.

## Example package usage

```python
from game_mechanics.game_manager import GameManager
from game_mechanics.model_adapter import ModelAdapter

model = ModelAdapter()
manager = GameManager(model_adapter=model)

manager.run_tutorial()
manager.start_wordle_session()
```

## Recording performance during a user test

`StudySession` wraps a `GameManager` with a per-participant performance log
and a clean reset, so one run of the app can walk through several
participants in a user-testing session without restarting the process or
leaking one player's progress into the next.

```python
from game_mechanics.study_session import StudySession

study = StudySession()  # writes to results/ by default
manager = study.start_participant("P001")

# ... play the game, logging each scored attempt as it happens ...
study.record_tutorial_result(word="HELLO", correct=True, response_time_seconds=2.1)
study.record_wordle_guess(target="LEARN", guess="LEARN", correct=True, response_time_seconds=6.4)

study.reset_for_next_player()  # saves P001's results, clears all state
manager = study.start_participant("P002")  # fresh tutorial + wordle progress
```

Each call to `reset_for_next_player()` (or `start_participant()` while a
session is already active) finalises the outgoing participant and writes:

- `results/performance_log.csv` — one summary row per participant (accuracy,
  tutorial completion, average response times, wordle outcome, session
  duration), appended across the whole user-test run.
- `results/participants/<id>_<timestamp>.json` — that participant's full
  event log (every tutorial sign and wordle guess with its response time),
  for deeper analysis later.

`results/` is git-ignored since it holds participant data from local runs.

### In the live webcam overlay

`examples/live_webcam.py` uses `StudySession` directly:

```bash
python3 -m examples.live_webcam --mode tutorial --participant-id P001
```

Omit `--participant-id` and you'll be prompted for one at startup (leave it
blank to auto-generate `P001`, `P002`, ...). While the overlay is running:

- `T` / `W` / `B` switch between tutorial, wordle, and both modes.
- `N` saves the current player's results, resets the tutorial and wordle
  state, and prompts in the terminal for the next participant's ID — ready
  for the next person in the study without restarting the app.
- `Q` saves the current player's results and quits.
