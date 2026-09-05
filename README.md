# ASL Game Mechanics

This repository is the gameplay layer for an ASL learning system. It contains the mechanics for a tutorial phase and multiple game modes — a Wordle-style guessing game and a Timed Rally / Streak mode — while integrating with the trained models produced in the `asl-training` project.

## Project purpose

- Teach basic ASL gestures in a guided tutorial
- Convert live model predictions into game input
- Support multiple game loops: Wordle-style guessing (static handshapes) and Timed Rally / Streak mode (dynamic, motion-trajectory signs)
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

## Timed Rally / Streak mode

`RallyGame` ([game_mechanics/rally.py](game_mechanics/rally.py)) is a rapid-fire
mode built around *dynamic* signs — motion trajectories (e.g. "COME", "DRINK",
"WANT") rather than static handshapes — scored by the sibling
`asl-training` project's dynamic (word-level) model instead of the static one
used by the tutorial and Wordle.

A round has a fixed time budget (`round_seconds`, default 60s). Each prompt
must be signed within its own short timeout (`prompt_timeout_seconds`,
default 4s) or it's scored as a miss and the round moves straight to the next
prompt — a wrong or missed sign resets the streak but does **not** end the
round, so a single round produces a dense stream of per-attempt
response-time + accuracy data (the design goal: lots of samples per session)
rather than stopping at the first mistake. Pass `max_misses` if you want a
"lives" variant that ends the round early instead.

```python
from game_mechanics.game_manager import GameManager

manager = GameManager()
rally = manager.start_rally_session(round_seconds=60.0, prompt_timeout_seconds=4.0)
print(rally.current_prompt)  # e.g. "DRINK"

result = manager.process_rally_sign("DRINK")  # score a detected sign
# result: {"correct": True, "streak": 1, "best_streak": 1, "next_prompt": ..., ...}

# Call every tick (e.g. once per video frame) so an unanswered prompt is
# scored as a miss once its own timeout elapses:
manager.check_rally_timeout()
```

Every `process_rally_sign` / `check_rally_timeout` call feeds
`GameManager.progress` the same way tutorial and Wordle attempts do, and
`StudySession.record_rally_attempt(...)` logs each one into the same
per-participant performance log described below (phase `"rally"`), with
`rally_attempts`, `rally_correct_attempts`, `rally_best_streak`, and
`rally_avg_response_time_seconds` rolled into the CSV summary row.

The word list (`DYNAMIC_SIGN_WORDS` in
[game_mechanics/config.py](game_mechanics/config.py)) must match the classes
the dynamic model was actually trained on — see
`asl-training/src/dynamic/models/sign_model_meta.json`.

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
blank to auto-generate `P001`, `P002`, ...). Each participant's session opens
on a start screen — nothing is scored until a mode is chosen, so time spent
reading the menu never counts toward response times:

- `T` — Tutorial. `G` (or `W`) — Games (Wordle). `B` — Tutorial, then Games.
- `R` — Timed Rally / Streak mode, only offered when a dynamic sign model is
  found at `asl-training/src/dynamic/models/sign_model.pkl`. Unlike the other
  modes, pressing `R` again always starts a fresh round, whether the
  previous one is still running or has finished.
- Once playing, `T` / `G` / `B` / `R` switch modes directly, same as the
  start screen.
- `N` saves the current player's results, resets all game state, prompts in
  the terminal for the next participant's ID, and returns to the start
  screen — ready for the next person in the study without restarting the app.
- `Q` saves the current player's results and quits.

Pass `--mode` on the command line to skip the start screen entirely and
launch straight into a mode (useful for scripted or non-interactive runs).
`--mode rally` falls back to the start screen with a warning if no dynamic
model is available.
