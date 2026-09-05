from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_PROJECT_ROOT = (PROJECT_ROOT.parent / "asl-training").resolve()

# Where per-participant user-test results are written by default. Each
# session's summary is appended as a row here; the underlying event log is
# kept alongside it in RESULTS_DIR / "participants".
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_MODEL_CANDIDATES = [
    TRAINING_PROJECT_ROOT / "src" / "static" / "models" / "sign_model.pkl",
    TRAINING_PROJECT_ROOT / "src" / "dynamic" / "models" / "sign_model.pkl",
    TRAINING_PROJECT_ROOT / "src" / "static" / "models" / "best_model.pkl",
    TRAINING_PROJECT_ROOT / "src" / "dynamic" / "models" / "best_model.pkl",
    TRAINING_PROJECT_ROOT / "src" / "static" / "models" / "model.pkl",
]

TUTORIAL_WORDS = [
    "HELLO",
    "THANKS",
    "YES",
    "NO",
    "PLEASE",
]

WORDLE_ALLOWED_WORDS = [
    "HELLO",
    "THANKS",
    "PLEASE",
    "LEARN",
    "GLOVE",
    "SIGN",
    "WORLD",
    "DANCE",
    "BASIC",
    "ALPHA",
]

# Timed Rally / Streak mode prompts. These are motion-trajectory (word-level)
# signs, not static handshapes, so they must match the classes the *dynamic*
# model was trained on - see
# asl-training/src/dynamic/models/sign_model_meta.json.
DYNAMIC_SIGN_WORDS = [
    "COME",
    "DRINK",
    "EAT",
    "FATHER",
    "GO",
    "HELP",
    "KNOW",
    "LOVE",
    "MORE",
    "MOTHER",
    "PLEASE",
    "SORRY",
    "THINK",
    "WANT",
    "WATER",
]

DEFAULT_DYNAMIC_MODEL_CANDIDATES = [
    TRAINING_PROJECT_ROOT / "src" / "dynamic" / "models" / "sign_model.pkl",
    TRAINING_PROJECT_ROOT / "src" / "dynamic" / "models" / "best_model.pkl",
]

# Timed Rally / Streak mode pacing: a round lasts RALLY_ROUND_SECONDS in
# total; each individual prompt must be signed within
# RALLY_PROMPT_TIMEOUT_SECONDS or it's scored as a miss and the round moves
# straight on to the next prompt, so a round keeps producing data even
# through a bad streak instead of stopping at the first mistake.
RALLY_ROUND_SECONDS = 60.0
RALLY_PROMPT_TIMEOUT_SECONDS = 4.0
