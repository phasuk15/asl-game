from __future__ import annotations

import argparse
import sys
import time
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

from game_mechanics.study_session import StudySession
from src.shared.features import FeatureExtractor
from src.shared.preprocessing import preprocess_frame
from src.shared.smoothing import LandmarkSmoother, PredictionSmoother
from src.static.classifier import SignClassifier


class GameOverlaySession:
    def __init__(self, mode: str | None = None, participant_id: str | None = None):
        # The session opens on a start screen where the player picks
        # Tutorial or Games; gameplay logic stays paused until they do, so
        # idle time spent reading the menu never counts toward response
        # times. Pass `mode` to skip the menu and launch straight into it
        # (used by the --mode CLI flag for scripted / non-interactive runs).
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

        self.wordle = self.manager.start_wordle_session()
        self.status_message = "Waiting for hand detection..."
        self.current_signal = "UNKNOWN"
        self.current_confidence = 0.0

        # Edge-detection so a held sign is only scored once, plus the
        # response-time clocks used to log how long each attempt took.
        self._last_stable_label: str | None = None
        self._current_lesson_word: str | None = None
        self._lesson_started_at = time.time()
        self._wordle_last_action_at = time.time()

        print(f"Started session for participant {self.study.participant_id}.")

        if mode:
            self._start_playing(mode)

    def _sync_lesson_timer(self) -> None:
        """Reset the response-time clock whenever the active lesson changes."""
        if self.manager.tutorial.is_complete():
            return
        lesson = self.manager.tutorial.current_lesson()
        if lesson.word != self._current_lesson_word:
            self._current_lesson_word = lesson.word
            self._lesson_started_at = time.time()

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
        self.status_message = f"Ready for participant {self.study.participant_id}."
        print(f"Started session for participant {self.study.participant_id}.")

    def _start_playing(self, mode: str) -> None:
        """Leave the start screen and begin gameplay in the chosen mode."""
        self.screen = "playing"
        self._switch_mode(mode)

    def _draw_start_screen(self, frame) -> None:
        self._draw_text(frame, f"Participant: {self.study.participant_id}", 40, scale=0.8)
        self._draw_text(frame, "Choose how to continue:", 80, color=(0, 255, 255))
        self._draw_text(frame, "Press T - Tutorial (learn the signs)", 115)
        self._draw_text(frame, "Press G - Games (Wordle)", 145)
        self._draw_text(frame, "Press B - Tutorial, then Games", 175)
        self._draw_text(
            frame,
            "Press N for next participant | Q to quit",
            frame.shape[0] - 20,
            color=(255, 255, 255),
            scale=0.6,
        )

    def _switch_mode(self, new_mode: str) -> None:
        if new_mode == self.mode:
            return

        self.mode = new_mode
        if self.mode == "tutorial":
            self.status_message = "Tutorial mode activated."
            self._current_lesson_word = None
        elif self.mode == "wordle":
            self.wordle = self.manager.start_wordle_session()
            self.status_message = "Wordle mode activated."
            self._wordle_last_action_at = time.time()
        elif self.mode == "both":
            self.status_message = "Tutorial + Wordle mode activated."
            self._current_lesson_word = None
            self._wordle_last_action_at = time.time()

    def _key_to_mode(self, key: int) -> str | None:
        mapping = {
            ord("t"): "tutorial",
            ord("w"): "wordle",
            ord("g"): "wordle",
            ord("b"): "both",
            ord("T"): "tutorial",
            ord("W"): "wordle",
            ord("G"): "wordle",
            ord("B"): "both",
        }
        return mapping.get(key)

    def _tutorial_prompt(self) -> str:
        lesson = self.manager.tutorial.current_lesson()
        return f"Tutorial: show {lesson.word}"

    def _wordle_prompt(self) -> str:
        return f"Wordle target: {self.wordle.target_word}"

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

    def _handle_prediction(self, label: str) -> None:
        label = label.strip().upper()
        self.current_signal = label

        if self.mode in {"tutorial", "both"} and not self.manager.tutorial.is_complete():
            lesson = self.manager.tutorial.current_lesson()
            if label == lesson.word:
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
                        self.previous_mode = "wordle"
                else:
                    self._current_lesson_word = next_lesson.word
                    self._lesson_started_at = time.time()
                    self.status_message = f"Next: show {next_lesson.word}"
                return

            self.status_message = f"Tutorial: show {lesson.word}"

        if self.mode in {"wordle", "both"}:
            if self.wordle.validate_guess(label):
                response_time = time.time() - self._wordle_last_action_at
                target_word = self.wordle.target_word
                feedback = self.manager.process_guess(label)
                self.study.record_wordle_guess(target_word, label, feedback["correct"], response_time)
                self._wordle_last_action_at = time.time()
                self.status_message = f"Guess {label}: {feedback['pattern']}"
                if feedback["correct"]:
                    self.status_message = "You solved the word!"
                    self.wordle = self.manager.start_wordle_session()
            else:
                self.status_message = f"Wordle: {self.wordle.target_word} (invalid guess: {label})"

    def run(self) -> None:
        if self.screen == "playing":
            print(f"Launching ASL game overlay in {self.mode} mode")
        else:
            print("Launching ASL game overlay. Choose Tutorial or Games from the start screen.")
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

        with vision.HandLandmarker.create_from_options(hand_options) as landmarker:
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

                if results.hand_landmarks:
                    hand_landmarks = results.hand_landmarks[0]
                    raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
                    smoothed = self.landmark_smoother.smooth(raw)
                    features = self.feature_extractor.extract(smoothed)
                    detected_label, confidence = self.classifier.predict(features)
                    stable_label = self.prediction_smoother.update(detected_label)
                    self.current_confidence = confidence
                    self.current_signal = stable_label.strip().upper() if stable_label else "UNKNOWN"

                    # Only score a transition into a new stable sign, not
                    # every frame it stays held, so one held sign yields one
                    # performance-log entry instead of dozens. The start
                    # screen ignores predictions entirely - nothing is
                    # scored until a mode has been chosen.
                    if self.screen == "playing" and stable_label and stable_label != self._last_stable_label:
                        self._handle_prediction(stable_label)
                    self._last_stable_label = stable_label

                    for lm in hand_landmarks:
                        x = int(lm.x * frame.shape[1])
                        y = int(lm.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                else:
                    self.current_signal = "UNKNOWN"
                    self.current_confidence = 0.0
                    self._last_stable_label = None
                    if self.screen == "playing":
                        self.status_message = "No hand detected. Move into frame."

                if self.screen == "start":
                    self._draw_start_screen(frame)
                else:
                    self._sync_lesson_timer()

                    if self.mode in {"tutorial", "both"} and not self.manager.tutorial.is_complete():
                        prompt = self._tutorial_prompt()
                    elif self.mode in {"wordle", "both"} and self.manager.tutorial.is_complete():
                        prompt = self._wordle_prompt()
                    else:
                        prompt = "Game complete"

                    self._draw_text(frame, f"Participant: {self.study.participant_id}", 30)
                    self._draw_text(frame, f"Mode: {self.mode.upper()}", 60)
                    self._draw_text(frame, prompt, 90)
                    self._draw_text(frame, f"Detected: {self.current_signal} ({self.current_confidence:.0%})", 120)
                    self._draw_text(frame, self.status_message, 150, color=(0, 255, 255), scale=0.6)
                    self._draw_text(
                        frame,
                        "Press T/W/B for modes | N for next participant | Q to quit",
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

                if self.screen == "start":
                    if key in {ord("t"), ord("T"), ord("w"), ord("W"), ord("g"), ord("G"), ord("b"), ord("B")}:
                        next_mode = self._key_to_mode(key)
                        if next_mode:
                            self._start_playing(next_mode)
                elif key in {ord("t"), ord("T"), ord("w"), ord("W"), ord("g"), ord("G"), ord("b"), ord("B")}:
                    next_mode = self._key_to_mode(key)
                    if next_mode:
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
        choices=["tutorial", "wordle", "both"],
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
