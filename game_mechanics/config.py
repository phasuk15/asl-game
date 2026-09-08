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

# One subfolder per word (lowercase), each holding a handful of WLASL clips
# of that sign - the reference footage shown as a demo during the tutorial.
DYNAMIC_SIGN_CLIPS_DIR = TRAINING_PROJECT_ROOT / "datasets" / "wlasl_subset"

# The tutorial teaches the same vocabulary Rally later tests, for two
# reasons: it's the only vocabulary with real reference clips already on
# disk (asl-training/datasets/wlasl_subset/<word>/*.mp4, used to show a demo
# before each lesson), and it's recognised by the same dynamic (motion
# trajectory) model - a static handshape classifier was never trained on
# these word-level signs.
TUTORIAL_WORDS = list(DYNAMIC_SIGN_WORDS)

# Fingerspelling tutorial letters. These are static handshapes, recognised
# by the *static* model (see DEFAULT_MODEL_CANDIDATES) - the same one used
# for Wordle - matching its trained classes (A-Z); see
# asl-training/src/static/models/sign_model_meta.json.
FINGERSPELLING_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# One subfolder per letter (uppercase), each holding a handful of reference
# photos of that handshape - shown as a still-image demo during the
# fingerspelling tutorial. Unlike the dynamic-sign clips, a static
# handshape has no motion to show, so a single photo is enough.
STATIC_SIGN_IMAGES_DIR = TRAINING_PROJECT_ROOT / "datasets" / "asl_alphabet_train"

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
    "SLEEP",
]

# Timed Rally / Streak mode pacing: a round lasts RALLY_ROUND_SECONDS in
# total; each individual prompt must be signed within
# RALLY_PROMPT_TIMEOUT_SECONDS or it's scored as a miss and the round moves
# straight on to the next prompt, so a round keeps producing data even
# through a bad streak instead of stopping at the first mistake.
RALLY_ROUND_SECONDS = 60.0
RALLY_PROMPT_TIMEOUT_SECONDS = 4.0
