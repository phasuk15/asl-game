# ASL Game Mechanics

This repository is the **gameplay layer** for an ASL (American Sign Language)
learning system built for an MSc dissertation. It provides two teaching
phases — a dynamic-sign Tutorial and a Fingerspelling alphabet tutorial —
and two game modes — a Wordle-style guessing game and a Timed Rally / Streak
mode — plus the instrumentation used to record learner performance during
user testing. Sign recognition itself is **not** implemented here: the game
layer consumes the trained models produced by the sibling `asl-training`
project.

## Project purpose

- Teach basic ASL in two guided tutorials: dynamic (motion-trajectory)
  signs, and the static fingerspelling alphabet.
- Convert live model predictions into game input.
- Provide multiple game loops: Wordle-style guessing (static handshapes) and
  Timed Rally / Streak mode (dynamic signs).
- Record per-participant performance (accuracy, response times, completion,
  streaks) for evaluation.
- Keep model integration isolated so the training project stays solely
  responsible for model creation.

## Repository layout

| Path | Contents |
| --- | --- |
| [game_mechanics/](game_mechanics/) | The importable package: mechanics, game engine, and study instrumentation (see below). |
| [examples/](examples/) | Runnable entry points — a scripted quick start and the live webcam app. |
| [tests/](tests/) | `pytest` suite for the mechanics (23 tests). |
| [results/](results/) | Output of user-test runs (`performance_log.csv` + per-participant JSON). Git-ignored; created on first run. |
| [pyproject.toml](pyproject.toml) | Packaging metadata and dependencies. |

### `game_mechanics/` modules

| Module | Responsibility |
| --- | --- |
| [config.py](game_mechanics/config.py) | Filesystem paths to the sibling training project, vocabulary lists (`TUTORIAL_WORDS` / `DYNAMIC_SIGN_WORDS`, `FINGERSPELLING_LETTERS`, `WORDLE_ALLOWED_WORDS`), and Rally pacing constants. |
| [model_adapter.py](game_mechanics/model_adapter.py) | `ModelAdapter` — locates and wraps the trained model from `asl-training`, returns a uniform `PredictionResult`, and degrades to `UNKNOWN` predictions when no model file is present. |
| [tutorial.py](game_mechanics/tutorial.py) | `TutorialPhase` / `TutorialLesson` — an ordered list of lessons with `advance()` / `mark_completed()`. Used for both the dynamic Tutorial and the Fingerspelling phase. |
| [wordle.py](game_mechanics/wordle.py) | `WordleGame` — target word, per-letter `GREEN` / `YELLOW` / `GREY` feedback, full guess `history`, win/lose state. |
| [rally.py](game_mechanics/rally.py) | `RallyGame` / `RallyAttempt` — timed streak mode with a per-prompt timeout, miss handling, and a round `summary()`. |
| [game_manager.py](game_mechanics/game_manager.py) | `GameManager` / `PlayerProgress` — ties the tutorial, fingerspelling, Wordle and Rally together, scores responses, and tracks overall progress. |
| [session_recorder.py](game_mechanics/session_recorder.py) | `SessionRecorder` / `PerformanceEvent`, plus `SUMMARY_FIELDS` and `build_summary()` — captures every scored interaction and rolls it into one summary row. |
| [study_session.py](game_mechanics/study_session.py) | `StudySession` — wraps a `GameManager` for user testing: writes `results/performance_log.csv` and `results/participants/<id>_<timestamp>.json`, and resets cleanly between participants. |

## Installation

Requires Python **3.10+**.

```bash
python3 -m pip install -e .          # runtime deps: numpy, opencv-python, mediapipe
python3 -m pip install -e ".[dev]"   # also installs pytest for the test suite
```

The live webcam app additionally needs the sibling `asl-training` project
checked out beside this one (see [Integration with training
models](#integration-with-training-models)).

## Quick start

A scripted run with no camera or model required — walks the tutorial and
plays one Wordle guess:

```bash
python3 -m examples.quickstart
```

### Package usage

```python
from game_mechanics.game_manager import GameManager
from game_mechanics.model_adapter import ModelAdapter

manager = GameManager(model_adapter=ModelAdapter())
manager.run_tutorial()
manager.start_wordle_session()
```

## Testing

```bash
python3 -m pytest -q
```

Runs from the repo root (the `game_mechanics` package must be importable —
`pip install -e .` or set `PYTHONPATH`). All tests are pure logic and need
no camera or model, **except**
[tests/test_live_webcam_setup.py](tests/test_live_webcam_setup.py), which
checks that the sibling `asl-training` project and its model assets resolve
and will fail if that project is not present.

## Integration with training models

The training project is expected to live beside this folder in the same
dissertation workspace:

```
<workspace>/
├── asl-game-        # this repository
└── asl-training     # trained models, datasets, live detector
```

[game_mechanics/model_adapter.py](game_mechanics/model_adapter.py) looks for
a pickled model in the standard locations used by the training project, so
the game layer can consume the ASL recogniser without duplicating training
logic. If no model is found the mechanics still run — predictions come back
as `UNKNOWN` — so the tutorials, Wordle and the test suite work standalone.

## Typical flow

1. Run a tutorial phase to teach a small set of ASL signs, each shown as a
   looping reference clip (dynamic) or a still photo (fingerspelling) before
   the learner attempts it.
2. Validate each sign using model predictions.
3. Unlock game modes such as Wordle after the tutorial is complete.
4. Use the same model output to score guesses and evaluate progress.

## Live webcam mode

```bash
python3 -m examples.live_webcam
python3 -m examples.live_webcam --mode tutorial --participant-id P001
```

[examples/live_webcam.py](examples/live_webcam.py) resolves the sibling
`asl-training` project, adds its source path to `PYTHONPATH`, and runs the
existing OpenCV + MediaPipe webcam detector against that project's model
assets, drawing the game overlay on top.

Each participant's session opens on a **start screen** — nothing is scored
until a mode is chosen, so time spent reading the menu never counts toward
response times:

- `G` / `W` — Games (Wordle). `F` — Fingerspelling. Both always available;
  both scored by the static handshape model.
- `T` — Tutorial. `B` — Tutorial, then Games. `R` — Timed Rally / Streak
  mode. All three need the dynamic sign model
  (`asl-training/src/dynamic/models/sign_model.pkl`) and are left off the
  start screen (with an explanatory line) if it is not found.
- Pressing `R` again always starts a fresh round, whether the previous one
  is still running or finished.
- Once playing, `T` / `G` / `B` / `R` / `F` switch modes directly.
- `SPACE` — while playing Wordle, locks in the currently-detected letter as
  the next letter of the guess being spelled out; the guess is submitted
  automatically once it reaches the target word's length.
- `N` saves the current player's results, resets all game state, prompts in
  the terminal for the next participant's ID, and returns to the start
  screen — ready for the next person without restarting the app.
- `Q` saves the current player's results and quits.

Pass `--mode` to skip the start screen and launch straight into a mode
(useful for scripted runs). `--mode tutorial|both|rally` falls back to the
start screen with a warning if no dynamic model is available.

## Tutorial: dynamic signs with a reference demo

The tutorial's vocabulary (`TUTORIAL_WORDS` in
[game_mechanics/config.py](game_mechanics/config.py)) is the same set of
dynamic (motion-trajectory) words Rally uses — not because the modes are
otherwise linked, but because that is the only vocabulary with both a
trained recogniser (a static handshape model was never trained on these
word-level signs) *and* real reference footage already on disk
(`asl-training/datasets/wlasl_subset/<word>/*.mp4`, sourced from WLASL).

In the live webcam overlay, each lesson plays its matching clip on a loop as
a small picture-in-picture inset while the prompt is active — "watch the
demo, then try it yourself" — before scoring the learner's own attempt via
the same buffered dynamic-sign pipeline Rally uses. Not every WLASL clip
decodes cleanly (some are broken/partial downloads), so
[examples/live_webcam.py](examples/live_webcam.py) probes each candidate clip
for a readable frame at startup and picks the first one that actually plays,
rather than trusting the first file found in the folder.

## Fingerspelling: static signs with a photo demo

Dynamic signs need a video to demonstrate; a static handshape has no motion,
so its tutorial (`FINGERSPELLING_LETTERS` in
[game_mechanics/config.py](game_mechanics/config.py), the alphabet A–Z) uses
a single reference photo instead — one per letter, sourced from
`asl-training/datasets/asl_alphabet_train/<LETTER>/*.jpg` (the same dataset
the static model was trained on) and shown as a still picture-in-picture
inset ("copy this") rather than a loop.

It is a separate, standalone phase from the dynamic Tutorial — not a
different mode of the same one — because it is scored by the *static*
classifier (the same one Wordle uses, since Wordle's static predictions are
themselves letters) via the same single-frame, edge-triggered pipeline as
Wordle, rather than the buffered dynamic-sign pipeline Tutorial and Rally
share. `GameManager.fingerspelling` is a `TutorialPhase` built via
`TutorialPhase.for_words(FINGERSPELLING_LETTERS)` — the same lesson/advance
machinery as the dynamic Tutorial, just pointed at a different word list.
`manager.run_fingerspelling()` mirrors `run_tutorial()` for scripted use.

## Wordle: a real tile-grid board, spelled one letter at a time

`WordleGame` keeps a full `history` of `GuessFeedback` (guess + per-letter
GREEN/YELLOW/GREY pattern) for the round, not just the raw guessed words —
[game_mechanics/wordle.py](game_mechanics/wordle.py) — so a UI can redraw the
whole board without recomputing anything itself.

In the live webcam overlay, `_draw_wordle_board()` renders that history as an
actual tile grid in the corner of the frame: filled, coloured tiles
(matching the real Wordle's green/yellow/grey palette) for guesses already
made, empty outlined tiles for guesses still to come, sized to the current
target word's length (4–6 letters depending which `WORDLE_ALLOWED_WORDS`
entry is active) and `WordleGame.max_guesses` rows. The on-screen prompt does
not spell out the target word, since the board is the game surface.

A guess is spelled one confirmed letter at a time rather than matched from a
single live classification — the static model only ever recognises one
letter per frame, so it could never realistically produce a whole word in
one shot. `SPACE` locks in whatever letter is currently detected
(`_submit_wordle_letter()` in
[examples/live_webcam.py](examples/live_webcam.py)); each confirmed letter
appears immediately in the board's current row as a dark-outlined (unscored)
tile, distinct from the light-grey empty tiles still to come. Once enough
letters have been confirmed to match the target word's length, the guess is
validated and submitted to `WordleGame` automatically — an invalid word
clears the in-progress guess and asks the player to spell it again; pressing
`SPACE` with no clear hand sign just prompts them to hold a sign first.

When a word is solved, the completed board (with its winning green row)
stays on screen for `_WORDLE_WIN_PAUSE_SECONDS` (3s) before a fresh round
starts automatically; pressing `SPACE` during that pause skips straight to
the next word.

## Timed Rally / Streak mode

`RallyGame` ([game_mechanics/rally.py](game_mechanics/rally.py)) is a
rapid-fire mode built around *dynamic* signs — motion trajectories (e.g.
"COME", "DRINK", "WANT") rather than static handshapes — scored by the
sibling `asl-training` project's dynamic (word-level) model instead of the
static one used by the tutorial and Wordle.

A round has a fixed time budget (`round_seconds`, default 60s). Each prompt
must be signed within its own short timeout (`prompt_timeout_seconds`,
default 4s) or it is scored as a miss and the round moves straight to the
next prompt — a wrong or missed sign resets the streak but does **not** end
the round, so a single round produces a dense stream of per-attempt
response-time and accuracy data (the design goal: many samples per session)
rather than stopping at the first mistake. Pass `max_misses` for a "lives"
variant that ends the round early instead.

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
per-participant performance log (phase `"rally"`), with `rally_attempts`,
`rally_correct_attempts`, `rally_best_streak` and
`rally_avg_response_time_seconds` rolled into the CSV summary row.

The word list (`DYNAMIC_SIGN_WORDS` in
[game_mechanics/config.py](game_mechanics/config.py)) must match the classes
the dynamic model was trained on — see
`asl-training/src/dynamic/models/sign_model_meta.json`.

## Recording performance during a user test

`StudySession` wraps a `GameManager` with a per-participant performance log
and a clean reset, so one run of the app can walk through several
participants without restarting the process or leaking one player's progress
into the next.

```python
from game_mechanics.study_session import StudySession

study = StudySession()  # writes to results/ by default
manager = study.start_participant("P001")

# ... play the game, logging each scored attempt as it happens ...
study.record_tutorial_result(word="COME", correct=True, response_time_seconds=2.1)
study.record_fingerspelling_result(letter="A", correct=True, response_time_seconds=1.4)
study.record_wordle_guess(target="LEARN", guess="LEARN", correct=True, response_time_seconds=6.4)

study.reset_for_next_player()  # saves P001's results, clears all state
manager = study.start_participant("P002")  # fresh progress
```

Each call to `reset_for_next_player()` (or `start_participant()` while a
session is already active) finalises the outgoing participant and writes:

- `results/performance_log.csv` — one summary row per participant (accuracy,
  tutorial and fingerspelling completion, average response times, Wordle
  outcome, Rally streak, session duration), appended across the whole run.
  Column order is fixed by `SUMMARY_FIELDS` in
  [game_mechanics/session_recorder.py](game_mechanics/session_recorder.py).
- `results/participants/<id>_<timestamp>.json` — that participant's full
  event log (every tutorial sign, fingerspelled letter, Wordle guess and
  Rally attempt, each with its response time), for deeper analysis later.

`results/` is git-ignored because it holds participant data from local runs.

## License

MIT — see [LICENSE](LICENSE).
