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
