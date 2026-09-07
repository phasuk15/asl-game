from __future__ import annotations

import argparse
import sys
import time
from contextlib import ExitStack
from pathlib import Path


def _training_root() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root.parent / "asl-training",
        project_root / "../asl-training",
    ]

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and (resolved / "config.py").exists():
            return resolved

    raise FileNotFoundError(
        "Could not find the sibling asl-training project. "
        "Expected it beside the asl-game- folder."
    )


training_root = _training_root()
if str(training_root) not in sys.path:
    sys.path.insert(0, str(training_root))

# Allow running this file directly (`python3 examples/live_webcam.py`), not
# just as a module (`python3 -m examples.live_webcam`), by putting the
# project root on sys.path so the sibling `game_mechanics` package resolves.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# asl-training's own top-level config module (training_root is now on
# sys.path). Reused for the dynamic-pipeline constants (sequence length,
# minimum frames, confidence threshold, asset paths) so this project can't
# silently drift from whatever that project was trained with.
import config as training_config

from game_mechanics.config import DYNAMIC_SIGN_CLIPS_DIR, STATIC_SIGN_IMAGES_DIR
from game_mechanics.study_session import StudySession
from src.dynamic.classifier import DynamicSignClassifier
from src.shared.features import FeatureExtractor, extract_body_features
from src.shared.preprocessing import preprocess_frame
from src.shared.sequence import flatten_sequence, resample_sequence
from src.shared.smoothing import LandmarkSmoother, PredictionSmoother
from src.static.classifier import SignClassifier


class GameOverlaySession:
    # BGR tile colours matching the real Wordle's palette (green/yellow/grey).
    _WORDLE_TILE_COLORS = {
        "GREEN": (106, 170, 100),
        "YELLOW": (88, 180, 201),
        "GREY": (126, 124, 120),
    }
    _WORDLE_EMPTY_BORDER = (218, 214, 211)

    def __init__(self, mode: str | None = None, participant_id: str | None = None):
        # The session opens on a start screen where the player picks a mode
        # (Tutorial, Fingerspelling, Games, Both, or Rally); gameplay logic
        # stays paused until they do, so idle time spent reading the menu
        # never counts toward response times. Pass `mode` to skip the menu
        # and launch straight into it (used by the --mode CLI flag for
        # scripted / non-interactive runs).
        self.screen = "start"
        self.mode: str | None = None

        # StudySession owns the per-participant GameManager and performance
        # log, so results survive resets between players in a user test.
        self.study = StudySession()
        self.manager = self.study.start_participant(participant_id)
        self.landmark_smoother = LandmarkSmoother(window_size=5)
        self.prediction_smoother = PredictionSmoother(window_size=7)
        self.feature_extractor = FeatureExtractor()

        model_path = training_root / "src" / "static" / "models" / "sign_model.pkl"
        self.classifier = SignClassifier(model_path=str(model_path))

        # Tutorial and Rally both score whole motion trajectories, not
        # single frames - the tutorial's vocabulary (game_mechanics.config
        # TUTORIAL_WORDS) is the same word-level sign set Rally uses, and
        # neither is recognisable by the static handshape model. Loading
        # the dynamic model is optional: if it hasn't been trained/copied
        # yet, Tutorial/Both/Rally are simply left off the start screen
        # (only Wordle, which uses the static model, still works) rather
        # than crashing the app.
        dynamic_model_path = training_root / "src" / "dynamic" / "models" / "sign_model.pkl"
        try:
            self.dynamic_classifier = DynamicSignClassifier(
                model_path=str(dynamic_model_path),
                models_dir=str(dynamic_model_path.parent),
            )
        except FileNotFoundError:
            self.dynamic_classifier = None
            print(
                f"[live_webcam] No dynamic sign model found at {dynamic_model_path} "
                "- Tutorial, Both, and Rally will be unavailable (Games/Wordle still works)."
            )

        self.wordle = self.manager.start_wordle_session()
        # Wordle guesses are spelled one confirmed letter at a time: SPACE
        # locks in whatever letter is currently detected (see
        # _submit_wordle_letter()) rather than trying to match a whole word
        # against a single classification, which the static model - it only
        # ever outputs one letter - could never realistically produce.
        self._wordle_current_guess = ""
        self.status_message = "Waiting for hand detection..."
        self.current_signal = "UNKNOWN"
        self.current_confidence = 0.0

        # Edge-detection so a held sign is only scored once, plus the
        # response-time clocks used to log how long each attempt took.
        self._last_stable_label: str | None = None
        self._current_lesson_word: str | None = None
        self._lesson_started_at = time.time()
        self._wordle_last_action_at = time.time()

        # Tutorial and Rally both buffer per-frame body+hand features while
        # a sign is being performed, and classify the whole clip once the
        # hand (and body) drop out of frame - see _finish_dynamic_buffer().
        self._dynamic_buffer: list = []
        self._dynamic_was_visible = False

        # Tutorial shows a small looping reference clip of each sign before
        # asking the player to try it - one real WLASL clip per word, found
        # once up front rather than re-scanning disk every lesson change.
        self._demo_clip_paths = self._find_demo_clip_paths()
        self._demo_cap: cv2.VideoCapture | None = None
        self._demo_word: str | None = None

        # Fingerspelling is the static-handshape counterpart: a still photo
        # instead of a looping clip (a static sign has no motion to show),
        # scored by the same static classifier Wordle uses.
        self._letter_image_paths = self._find_letter_image_paths()
        self._current_letter: str | None = None
        self._letter_started_at = time.time()
        self._letter_image = None

        print(f"Started session for participant {self.study.participant_id}.")

        if mode in {"tutorial", "both", "rally"} and self.dynamic_classifier is None:
            print(
                f"[live_webcam] --mode {mode} requested but no dynamic model is available; "
                "showing the start screen instead."
            )
        elif mode:
            self._start_playing(mode)

    def _find_demo_clip_paths(self) -> dict:
        """Map each tutorial word to one reference clip on disk, if any.

        Several WLASL clips per word fail to decode (broken/partial
        downloads - in practice the numerically-first file in every
        folder), so each candidate is actually opened and probed for one
        readable frame before being picked, rather than trusting the first
        file found."""
        paths: dict = {}
        for lesson in self.manager.tutorial.lessons:
            if lesson.word in paths:
                continue
            clip_dir = DYNAMIC_SIGN_CLIPS_DIR / lesson.word.lower()
            if not clip_dir.is_dir():
                continue
            for clip in sorted(clip_dir.glob("*.mp4")):
                probe = cv2.VideoCapture(str(clip))
                readable = probe.isOpened() and probe.read()[0]
                probe.release()
                if readable:
                    paths[lesson.word] = clip
                    break
        return paths

    def _open_demo_clip(self, word: str) -> None:
        """Swap the looping reference-clip capture over to `word`'s clip."""
        if self._demo_cap is not None:
            self._demo_cap.release()
            self._demo_cap = None
        path = self._demo_clip_paths.get(word)
        if path is not None:
            self._demo_cap = cv2.VideoCapture(str(path))
        self._demo_word = word

    def _draw_tutorial_demo(self, frame) -> None:
        """Composite a small looping reference clip into the frame's corner."""
        if self._demo_cap is None:
            return
        ret, demo_frame = self._demo_cap.read()
        if not ret:
            self._demo_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, demo_frame = self._demo_cap.read()
            if not ret:
                return

        inset_w, inset_h = 200, 150
        demo_frame = cv2.resize(demo_frame, (inset_w, inset_h))
        x0 = frame.shape[1] - inset_w - 10
        y0 = 10
        frame[y0 : y0 + inset_h, x0 : x0 + inset_w] = demo_frame
        cv2.rectangle(frame, (x0, y0), (x0 + inset_w, y0 + inset_h), (0, 255, 255), 2)
        cv2.putText(frame, "DEMO", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    def _sync_lesson_timer(self) -> None:
        """Reset the response-time clock and swap the reference demo clip
        whenever the active lesson changes."""
        if self.manager.tutorial.is_complete():
            return
        lesson = self.manager.tutorial.current_lesson()
        if lesson.word != self._current_lesson_word:
            self._current_lesson_word = lesson.word
            self._lesson_started_at = time.time()
            self._open_demo_clip(lesson.word)

    def _find_letter_image_paths(self) -> dict:
        """Map each fingerspelling letter to one reference photo on disk."""
        paths: dict = {}
        for lesson in self.manager.fingerspelling.lessons:
            letter_dir = STATIC_SIGN_IMAGES_DIR / lesson.word
            if not letter_dir.is_dir():
                continue
            for image_path in sorted(letter_dir.glob("*.jpg")):
                if cv2.imread(str(image_path)) is not None:
                    paths[lesson.word] = image_path
                    break
        return paths

    def _load_letter_image(self, letter: str) -> None:
        path = self._letter_image_paths.get(letter)
        self._letter_image = cv2.imread(str(path)) if path is not None else None

    def _draw_letter_demo(self, frame) -> None:
        """Composite the current letter's reference photo into the corner."""
        if self._letter_image is None:
            return
        inset_w, inset_h = 200, 150
        demo_frame = cv2.resize(self._letter_image, (inset_w, inset_h))
        x0 = frame.shape[1] - inset_w - 10
        y0 = 10
        frame[y0 : y0 + inset_h, x0 : x0 + inset_w] = demo_frame
        cv2.rectangle(frame, (x0, y0), (x0 + inset_w, y0 + inset_h), (0, 255, 255), 2)
        cv2.putText(frame, "COPY THIS", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    def _sync_letter_timer(self) -> None:
        """Reset the response-time clock and swap the reference photo
        whenever the active fingerspelling letter changes."""
        if self.manager.fingerspelling.is_complete():
            return
        lesson = self.manager.fingerspelling.current_lesson()
        if lesson.word != self._current_letter:
            self._current_letter = lesson.word
            self._letter_started_at = time.time()
            self._load_letter_image(lesson.word)

    def _draw_wordle_board(self, frame) -> None:
        """Draw a tile-grid board like the real Wordle: filled coloured
        tiles for past guesses (from wordle.history), the letters typed so
        far into the current guess (unscored, dark-outlined - see
        _submit_wordle_letter()), and empty outlined tiles beyond that."""
        wordle = self.wordle
        word_len = len(wordle.target_word)
        rows = wordle.max_guesses
        cell, gap, pad, header_h = 38, 6, 14, 34

        board_w = word_len * cell + (word_len - 1) * gap
        board_h = rows * cell + (rows - 1) * gap
        panel_w = board_w + pad * 2
        panel_h = header_h + board_h + pad * 2

        x0 = frame.shape[1] - panel_w - 10
        y0 = 10

        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (245, 245, 245), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (190, 190, 190), 1)
        cv2.putText(
            frame,
            "WORDLE",
            (x0 + pad, y0 + header_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )

        grid_x, grid_y = x0 + pad, y0 + header_h

        for row in range(rows):
            feedback = wordle.history[row] if row < len(wordle.history) else None
            is_current_row = feedback is None and row == len(wordle.history)
            for col in range(word_len):
                cx = grid_x + col * (cell + gap)
                cy = grid_y + row * (cell + gap)

                if feedback is not None:
                    letter = feedback.guess[col]
                    tile_color = self._WORDLE_TILE_COLORS[feedback.pattern[col]]
                    cv2.rectangle(frame, (cx, cy), (cx + cell, cy + cell), tile_color, -1)
                    text_color = (255, 255, 255)
                elif is_current_row and col < len(self._wordle_current_guess):
                    letter = self._wordle_current_guess[col]
                    cv2.rectangle(frame, (cx, cy), (cx + cell, cy + cell), (255, 255, 255), -1)
                    cv2.rectangle(frame, (cx, cy), (cx + cell, cy + cell), (40, 40, 40), 2)
                    text_color = (40, 40, 40)
                else:
                    letter = ""
                    cv2.rectangle(frame, (cx, cy), (cx + cell, cy + cell), (255, 255, 255), -1)
                    cv2.rectangle(frame, (cx, cy), (cx + cell, cy + cell), self._WORDLE_EMPTY_BORDER, 2)
                    text_color = (40, 40, 40)

                if letter:
                    (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    tx = cx + (cell - tw) // 2
                    ty = cy + (cell + th) // 2
                    cv2.putText(frame, letter, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)

    def _start_next_participant(self) -> None:
        """Save the current player's results and reset all game state so
        the next person in the user test starts from a clean slate."""
        summary = self.study.reset_for_next_player()
        if summary:
            print(
                f"Saved results for {summary['participant_id']}: "
                f"{summary['accuracy_percent']}% accuracy, "
                f"{summary['session_duration_seconds']}s -> {self.study.log_path}"
            )

        participant_id = input("Next participant ID (blank to auto-generate): ").strip()
        self.manager = self.study.start_participant(participant_id)
        self.screen = "start"
        self.mode = None
        self.wordle = self.manager.start_wordle_session()
        self.current_signal = "UNKNOWN"
        self.current_confidence = 0.0
        self._last_stable_label = None
        self._current_lesson_word = None
        self._lesson_started_at = time.time()
        self._wordle_last_action_at = time.time()
        self._dynamic_buffer = []
        self._dynamic_was_visible = False
        if self._demo_cap is not None:
            self._demo_cap.release()
            self._demo_cap = None
        self._demo_word = None
        self._current_letter = None
        self._letter_started_at = time.time()
        self._letter_image = None
        self._wordle_current_guess = ""
        self.status_message = f"Ready for participant {self.study.participant_id}."
        print(f"Started session for participant {self.study.participant_id}.")

    def _start_playing(self, mode: str) -> None:
        """Leave the start screen and begin gameplay in the chosen mode."""
        self.screen = "playing"
        self._switch_mode(mode)

    def _draw_start_screen(self, frame) -> None:
        self._draw_text(frame, f"Participant: {self.study.participant_id}", 40, scale=0.8)
        self._draw_text(frame, "Choose how to continue:", 80, color=(0, 255, 255))
        self._draw_text(frame, "Press G - Games (Wordle)", 115)
        self._draw_text(frame, "Press F - Fingerspelling (learn the alphabet)", 145)
        if self.dynamic_classifier is not None:
            self._draw_text(frame, "Press T - Tutorial (learn the signs)", 175)
            self._draw_text(frame, "Press B - Tutorial, then Games", 205)
            self._draw_text(frame, "Press R - Timed Rally (dynamic signs)", 235)
        else:
            self._draw_text(
                frame,
                "Tutorial / Both / Rally unavailable - no dynamic model found",
                175,
                color=(0, 0, 255),
                scale=0.55,
            )
        self._draw_text(
            frame,
            "Press N for next participant | Q to quit",
            frame.shape[0] - 20,
            color=(255, 255, 255),
            scale=0.6,
        )

    def _switch_mode(self, new_mode: str) -> None:
        # Rally is exempt from the no-op guard: pressing R again - whether
        # mid-round or after one finishes - deliberately starts a fresh
        # round rather than doing nothing.
        if new_mode == self.mode and new_mode != "rally":
            return

        self.mode = new_mode
        if self.mode == "tutorial":
            self.status_message = "Tutorial mode activated."
            self._current_lesson_word = None
        elif self.mode == "wordle":
            self.wordle = self.manager.start_wordle_session()
            self.status_message = "Wordle mode activated."
            self._wordle_last_action_at = time.time()
            self._wordle_current_guess = ""
        elif self.mode == "both":
            self.status_message = "Tutorial + Wordle mode activated."
            self._current_lesson_word = None
            self._wordle_last_action_at = time.time()
            self._wordle_current_guess = ""
        elif self.mode == "rally":
            rally = self.manager.start_rally_session()
            self._dynamic_buffer = []
            self._dynamic_was_visible = False
            self.status_message = f"Rally! Sign: {rally.current_prompt}"
        elif self.mode == "fingerspell":
            self.status_message = "Fingerspelling mode activated."
            self._current_letter = None

    def _key_to_mode(self, key: int) -> str | None:
        mapping = {
            ord("t"): "tutorial",
            ord("w"): "wordle",
            ord("g"): "wordle",
            ord("b"): "both",
            ord("r"): "rally",
            ord("f"): "fingerspell",
            ord("T"): "tutorial",
            ord("W"): "wordle",
            ord("G"): "wordle",
            ord("B"): "both",
            ord("R"): "rally",
            ord("F"): "fingerspell",
        }
        mode = mapping.get(key)
        # Tutorial, Both, and Rally all need the dynamic model - none are
        # offered (on the start screen or mid-play) without it.
        if mode in {"tutorial", "both", "rally"} and self.dynamic_classifier is None:
            return None
        return mode

    def _tutorial_prompt(self) -> str:
        lesson = self.manager.tutorial.current_lesson()
        return f"Tutorial: show {lesson.word}"

    def _wordle_prompt(self) -> str:
        typed, total = len(self._wordle_current_guess), len(self.wordle.target_word)
        return f"Wordle: sign a letter, SPACE to add it ({typed}/{total})"

    def _fingerspell_prompt(self) -> str:
        lesson = self.manager.fingerspelling.current_lesson()
        return f"Fingerspell: show {lesson.word}"

    def _rally_prompt(self) -> str:
        rally = self.manager.rally
        if rally is None or rally.current_prompt is None:
            return "Rally: press R to start"
        return (
            f"Rally: sign {rally.current_prompt} | "
            f"streak {rally.streak} (best {rally.best_streak}) | "
            f"{rally.time_remaining():.0f}s left"
        )

    def _draw_text(self, frame, text: str, y: int, color=(255, 255, 255), scale=0.7):
        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_hand_skeleton(self, frame, hand_landmarks) -> None:
        """Draw the hand as a skeleton: bones connecting the joints (via
        asl-training's own HAND_CONNECTIONS topology), then a dot on top of
        each joint - not just the loose dots this used to draw."""
        points = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in hand_landmarks]

        for start, end in training_config.HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (255, 255, 255), 1, cv2.LINE_AA)

        for point in points:
            cv2.circle(frame, point, 3, (0, 255, 0), -1)

    def _handle_prediction(self, label: str) -> None:
        """Single-frame static-model path, used for Fingerspelling: each
        stable sign is auto-scored the instant it's held. Wordle uses the
        same static classifier's live output too, but doesn't auto-score
        from it - see _submit_wordle_letter() - since the model only ever
        classifies one letter at a time, never a whole word."""
        label = label.strip().upper()
        self.current_signal = label

        if self.mode == "fingerspell" and not self.manager.fingerspelling.is_complete():
            lesson = self.manager.fingerspelling.current_lesson()
            if label == lesson.word:
                response_time = time.time() - self._letter_started_at
                self.study.record_fingerspelling_result(lesson.word, True, response_time)
                self.manager.fingerspelling.mark_completed(lesson)
                self.status_message = f"Correct! Nice sign for {lesson.word}."
                next_lesson = self.manager.fingerspelling.advance()
                if next_lesson is None:
                    self.manager.progress.fingerspelling_complete = True
                    self.status_message = "Fingerspelling complete!"
                else:
                    self._current_letter = next_lesson.word
                    self._letter_started_at = time.time()
                    self._load_letter_image(next_lesson.word)
                    self.status_message = f"Next: show {next_lesson.word}"
            else:
                self.status_message = f"Fingerspell: show {lesson.word} (saw {label})"

    def _submit_wordle_letter(self) -> None:
        """SPACE: lock in whatever static sign is currently detected as the
        next letter of the guess being spelled out. Once enough letters
        have been confirmed to match the target word's length, the guess
        is submitted to Wordle automatically."""
        if self.screen != "playing" or self.mode not in {"wordle", "both"}:
            return
        if self.mode == "both" and not self.manager.tutorial.is_complete():
            return  # tutorial is still the active phase in "both" mode

        letter = self.current_signal
        if len(letter) != 1 or not letter.isalpha():
            self.status_message = "No letter detected to add - hold a clear sign first."
            return

        self._wordle_current_guess += letter
        self.status_message = f"Added '{letter}' -> {self._wordle_current_guess}"

        if len(self._wordle_current_guess) < len(self.wordle.target_word):
            return

        guess = self._wordle_current_guess
        self._wordle_current_guess = ""

        if not self.wordle.validate_guess(guess):
            self.status_message = f"'{guess}' isn't a valid word - spell your guess again."
            return

        response_time = time.time() - self._wordle_last_action_at
        target_word = self.wordle.target_word
        feedback = self.manager.process_guess(guess)
        self.study.record_wordle_guess(target_word, guess, feedback["correct"], response_time)
        self._wordle_last_action_at = time.time()

        # The board (green/yellow/grey tiles, just like the real game)
        # already shows this guess's result in full - the status line is
        # just a plain-English recap, not a dump of the raw pattern list.
        correct_count = feedback["pattern"].count("GREEN")
        present_count = feedback["pattern"].count("YELLOW")
        self.status_message = f"Guessed {guess}: {correct_count} correct, {present_count} present"
        if feedback["correct"]:
            self.status_message = "You solved the word!"
            self.wordle = self.manager.start_wordle_session()

    def _handle_tutorial_dynamic_sign(self, detected: str, raw_label: str) -> None:
        """Score one classified dynamic-sign clip against the current
        tutorial lesson (mirrors the old per-frame tutorial branch, but
        driven by a whole buffered trajectory instead of a single frame)."""
        lesson = self.manager.tutorial.current_lesson()
        if detected and detected == lesson.word:
            response_time = time.time() - self._lesson_started_at
            self.study.record_tutorial_result(lesson.word, True, response_time)
            self.manager.tutorial.mark_completed(lesson)
            self.status_message = f"Correct! Nice sign for {lesson.word}."
            next_lesson = self.manager.tutorial.advance()
            if next_lesson is None:
                self.manager.progress.tutorial_complete = True
                self.status_message = "Tutorial complete!"
                if self.mode == "both":
                    self.mode = "wordle"
                    self.wordle = self.manager.start_wordle_session()
                    self._wordle_last_action_at = time.time()
                    self._wordle_current_guess = ""
            else:
                self._current_lesson_word = next_lesson.word
                self._lesson_started_at = time.time()
                self._open_demo_clip(next_lesson.word)
                self.status_message = f"Next: show {next_lesson.word}"
        else:
            seen = raw_label.strip().upper() or "an unclear sign"
            self.status_message = f"Not quite - target is {lesson.word} (saw {seen}). Try again."

    def _finish_dynamic_buffer(self) -> None:
        """Falling edge: hand+body just disappeared, so the buffered clip
        is a completed sign - classify it as one trajectory and route the
        result to whichever dynamic-sign mode is active."""
        buffer, self._dynamic_buffer = self._dynamic_buffer, []
        if len(buffer) < training_config.MIN_SEQUENCE_FRAMES or self.dynamic_classifier is None:
            return

        sequence = resample_sequence(np.stack(buffer), training_config.SEQUENCE_LENGTH)
        label, confidence = self.dynamic_classifier.predict(flatten_sequence(sequence))
        detected = label.strip().upper() if confidence >= training_config.MIN_DYNAMIC_PREDICTION_CONFIDENCE else ""

        if self.mode == "rally":
            result = self.manager.process_rally_sign(detected)
            self._apply_rally_result(result, detected_label=label)
        elif self.mode in {"tutorial", "both"} and not self.manager.tutorial.is_complete():
            self._handle_tutorial_dynamic_sign(detected, raw_label=label)

    def _apply_rally_result(self, result: dict, detected_label: str = "") -> None:
        self.study.record_rally_attempt(
            target=result["prompt"],
            submitted=result["submitted"],
            correct=result["correct"],
            response_time_seconds=result["response_time_seconds"],
        )

        if result["timed_out"]:
            self.status_message = f"Too slow for {result['prompt']}! Streak reset."
        elif result["correct"]:
            self.status_message = f"Nice! {result['prompt']} correct - streak {result['streak']}."
        else:
            seen = detected_label.strip().upper() or "an unclear sign"
            self.status_message = f"Not quite - target was {result['prompt']} (saw {seen})."

        if result["finished"]:
            accuracy = self.manager.rally.summary()["accuracy_percent"]
            self.status_message = (
                f"Rally over ({result['finish_reason']})! Best streak {result['best_streak']}, "
                f"{accuracy}% accuracy. Press R to go again."
            )

    def run(self) -> None:
        if self.screen == "playing":
            print(f"Launching ASL game overlay in {self.mode} mode")
        else:
            print("Launching ASL game overlay. Choose a mode from the start screen.")
        print(f"Using model: {training_root / 'src' / 'static' / 'models' / 'sign_model.pkl'}")

        hand_options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(training_root / "assets" / "hand_landmarker.task")
            ),
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        cap = cv2.VideoCapture(0)

        with ExitStack() as stack:
            landmarker = stack.enter_context(vision.HandLandmarker.create_from_options(hand_options))

            # Tutorial and Rally both need a pose landmarker too (motion
            # trajectories are scored from hand + upper-body movement
            # together). Only set one up when a dynamic model was actually
            # found - no point paying for pose detection every frame
            # otherwise (Wordle-only sessions never touch it).
            pose_landmarker = None
            if self.dynamic_classifier is not None:
                pose_options = vision.PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=str(training_root / "assets" / "pose_landmarker_lite.task")
                    ),
                    num_poses=1,
                    min_pose_detection_confidence=training_config.MIN_POSE_DETECTION_CONFIDENCE,
                    min_tracking_confidence=training_config.MIN_POSE_TRACKING_CONFIDENCE,
                )
                pose_landmarker = stack.enter_context(vision.PoseLandmarker.create_from_options(pose_options))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = preprocess_frame(frame)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                results = landmarker.detect(mp_image)

                # Wordle and Fingerspelling both read the static handshape
                # model's per-frame output; Fingerspelling auto-scores each
                # stable sign, Wordle only commits one on a SPACE press
                # (see _submit_wordle_letter()). Tutorial and Rally are
                # scored from a whole buffered dynamic-sign clip instead
                # (see below) - every mode uses exactly one of the two
                # pipelines, never both.
                wordle_active = self.screen == "playing" and (
                    self.mode == "wordle" or (self.mode == "both" and self.manager.tutorial.is_complete())
                )
                fingerspell_active = (
                    self.screen == "playing"
                    and self.mode == "fingerspell"
                    and not self.manager.fingerspelling.is_complete()
                )
                static_active = wordle_active or fingerspell_active
                dynamic_active = self.screen == "playing" and (
                    self.mode == "rally"
                    or (self.mode in {"tutorial", "both"} and not self.manager.tutorial.is_complete())
                )

                if results.hand_landmarks:
                    hand_landmarks = results.hand_landmarks[0]
                    raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
                    smoothed = self.landmark_smoother.smooth(raw)
                    features = self.feature_extractor.extract(smoothed)
                    detected_label, confidence = self.classifier.predict(features)
                    # PredictionSmoother is a blind majority vote over labels
                    # - it has no idea what confidence produced them - so a
                    # run of confident wrong guesses and a run of noisy
                    # low-confidence ones would count equally. Below the
                    # trained threshold, feed it "no reading" instead of the
                    # label, the same way the dynamic pipeline already
                    # discards low-confidence classifications.
                    if confidence < training_config.MIN_PREDICTION_CONFIDENCE:
                        detected_label = ""
                    stable_label = self.prediction_smoother.update(detected_label)
                    self.current_confidence = confidence
                    self.current_signal = stable_label.strip().upper() if stable_label else "UNKNOWN"

                    # Only score a transition into a new stable sign, not
                    # every frame it stays held, so one held sign yields one
                    # performance-log entry instead of dozens. Wordle reads
                    # this same live signal for display, but only commits a
                    # letter to its guess on a SPACE press - see
                    # _submit_wordle_letter() - not automatically here.
                    if fingerspell_active and stable_label and stable_label != self._last_stable_label:
                        self._handle_prediction(stable_label)
                    self._last_stable_label = stable_label

                    self._draw_hand_skeleton(frame, hand_landmarks)

                    if dynamic_active and pose_landmarker is not None:
                        pose_results = pose_landmarker.detect(mp_image)
                        if pose_results.pose_landmarks:
                            pose_arr = np.array(
                                [[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks[0]]
                            )
                            self._dynamic_buffer.append(extract_body_features(raw, pose_arr, dominant_side="right"))
                            self._dynamic_was_visible = True
                        elif self._dynamic_was_visible:
                            self._finish_dynamic_buffer()
                            self._dynamic_was_visible = False
                else:
                    self.current_signal = "UNKNOWN"
                    self.current_confidence = 0.0
                    self._last_stable_label = None
                    if dynamic_active:
                        if self._dynamic_was_visible:
                            self._finish_dynamic_buffer()
                            self._dynamic_was_visible = False
                    elif static_active:
                        self.status_message = "No hand detected. Move into frame."

                if self.screen == "playing" and self.mode == "rally":
                    timeout_result = self.manager.check_rally_timeout()
                    if timeout_result is not None:
                        self._dynamic_buffer = []
                        self._dynamic_was_visible = False
                        self._apply_rally_result(timeout_result)

                if self.screen == "start":
                    self._draw_start_screen(frame)
                else:
                    self._sync_lesson_timer()
                    self._sync_letter_timer()
                    tutorial_active = self.mode in {"tutorial", "both"} and not self.manager.tutorial.is_complete()

                    if self.mode == "rally":
                        prompt = self._rally_prompt()
                    elif tutorial_active:
                        prompt = self._tutorial_prompt()
                    elif fingerspell_active:
                        prompt = self._fingerspell_prompt()
                    elif self.mode in {"wordle", "both"} and self.manager.tutorial.is_complete():
                        prompt = self._wordle_prompt()
                    else:
                        prompt = "Game complete"

                    if tutorial_active:
                        self._draw_tutorial_demo(frame)
                    elif fingerspell_active:
                        self._draw_letter_demo(frame)
                    elif wordle_active:
                        self._draw_wordle_board(frame)

                    if dynamic_active:
                        detected_line = (
                            f"Buffering: {len(self._dynamic_buffer)} frames"
                            if self._dynamic_was_visible
                            else "Show your hand + upper body to sign"
                        )
                    else:
                        detected_line = f"Detected: {self.current_signal} ({self.current_confidence:.0%})"

                    self._draw_text(frame, f"Participant: {self.study.participant_id}", 30)
                    self._draw_text(frame, f"Mode: {self.mode.upper()}", 60)
                    self._draw_text(frame, prompt, 90)
                    self._draw_text(frame, detected_line, 120)
                    self._draw_text(frame, self.status_message, 150, color=(0, 255, 255), scale=0.6)
                    self._draw_text(
                        frame,
                        "T/W/G/B/R/F modes | SPACE = Wordle letter | N = next | Q = quit",
                        frame.shape[0] - 20,
                        color=(255, 255, 255),
                        scale=0.6,
                    )

                cv2.imshow("ASL Game Overlay", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                if key in {ord("n"), ord("N")}:
                    self._start_next_participant()
                    continue

                if key == ord(" "):
                    self._submit_wordle_letter()
                    continue

                next_mode = self._key_to_mode(key)
                if next_mode:
                    if self.screen == "start":
                        self._start_playing(next_mode)
                    else:
                        self._switch_mode(next_mode)

        cap.release()
        cv2.destroyAllWindows()

        summary = self.study.reset_for_next_player()
        if summary:
            print(
                f"Saved results for {summary['participant_id']}: "
                f"{summary['accuracy_percent']}% accuracy, "
                f"{summary['session_duration_seconds']}s -> {self.study.log_path}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the ASL game with a live webcam overlay.")
    parser.add_argument(
        "--mode",
        choices=["tutorial", "wordle", "both", "rally", "fingerspell"],
        default=None,
        help="Skip the start screen and launch directly in this mode.",
    )
    parser.add_argument(
        "--participant-id",
        default=None,
        help="Identifier for the first participant. Prompted for if omitted.",
    )
    args = parser.parse_args()

    participant_id = args.participant_id
    if participant_id is None:
        participant_id = input("Participant ID (blank to auto-generate): ").strip()

    GameOverlaySession(mode=args.mode, participant_id=participant_id).run()


if __name__ == "__main__":
    main()
