from __future__ import annotations

import argparse
import sys
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

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from game_mechanics.game_manager import GameManager
from src.shared.features import FeatureExtractor
from src.shared.preprocessing import preprocess_frame
from src.shared.smoothing import LandmarkSmoother, PredictionSmoother
from src.static.classifier import SignClassifier


class GameOverlaySession:
    def __init__(self, mode: str = "tutorial"):
        self.mode = mode
        self.manager = GameManager()
        self.previous_mode = mode
        self.landmark_smoother = LandmarkSmoother(window_size=5)
        self.prediction_smoother = PredictionSmoother(window_size=7)
        self.feature_extractor = FeatureExtractor()

        model_path = training_root / "src" / "static" / "models" / "sign_model.pkl"
        self.classifier = SignClassifier(model_path=str(model_path))

        self.wordle = self.manager.start_wordle_session()
        self.status_message = "Waiting for hand detection..."
        self.current_signal = "UNKNOWN"
        self.current_confidence = 0.0

    def _switch_mode(self, new_mode: str) -> None:
        if new_mode == self.mode:
            return

        self.mode = new_mode
        if self.mode == "tutorial":
            self.status_message = "Tutorial mode activated."
        elif self.mode == "wordle":
            self.wordle = self.manager.start_wordle_session()
            self.status_message = "Wordle mode activated."
        elif self.mode == "both":
            self.status_message = "Tutorial + Wordle mode activated."

    def _key_to_mode(self, key: int) -> str | None:
        mapping = {
            ord("t"): "tutorial",
            ord("w"): "wordle",
            ord("b"): "both",
            ord("T"): "tutorial",
            ord("W"): "wordle",
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
                self.manager.tutorial.mark_completed(lesson)
                self.status_message = f"Correct! Nice sign for {lesson.word}."
                next_lesson = self.manager.tutorial.advance()
                if next_lesson is None:
                    self.manager.progress.tutorial_complete = True
                    self.status_message = "Tutorial complete!"
                    if self.mode == "both":
                        self.mode = "wordle"
                        self.wordle = self.manager.start_wordle_session()
                        self.previous_mode = "wordle"
                else:
                    self.status_message = f"Next: show {next_lesson.word}"
                return

            self.status_message = f"Tutorial: show {lesson.word}"

        if self.mode in {"wordle", "both"}:
            if self.wordle.validate_guess(label):
                feedback = self.manager.process_guess(label)
                self.status_message = f"Guess {label}: {feedback['pattern']}"
                if feedback["correct"]:
                    self.status_message = "You solved the word!"
                    self.wordle = self.manager.start_wordle_session()
            else:
                self.status_message = f"Wordle: {self.wordle.target_word} (invalid guess: {label})"

    def run(self) -> None:
        print(f"Launching ASL game overlay in {self.mode} mode")
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
                    self._handle_prediction(stable_label)

                    for lm in hand_landmarks:
                        x = int(lm.x * frame.shape[1])
                        y = int(lm.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                else:
                    self.current_signal = "UNKNOWN"
                    self.status_message = "No hand detected. Move into frame."

                if self.mode in {"tutorial", "both"} and not self.manager.tutorial.is_complete():
                    prompt = self._tutorial_prompt()
                elif self.mode in {"wordle", "both"} and self.manager.tutorial.is_complete():
                    prompt = self._wordle_prompt()
                else:
                    prompt = "Game complete"

                self._draw_text(frame, f"Mode: {self.mode.upper()}", 30)
                self._draw_text(frame, prompt, 60)
                self._draw_text(frame, f"Detected: {self.current_signal} ({self.current_confidence:.0%})", 90)
                self._draw_text(frame, self.status_message, 120, color=(0, 255, 255), scale=0.6)
                self._draw_text(frame, "Press T/W/B for modes | Q to quit", frame.shape[0] - 20, color=(255, 255, 255), scale=0.6)

                cv2.imshow("ASL Game Overlay", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                if key in {ord("t"), ord("T"), ord("w"), ord("W"), ord("b"), ord("B")}:
                    next_mode = self._key_to_mode(key)
                    if next_mode:
                        self._switch_mode(next_mode)

        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the ASL game with a live webcam overlay.")
    parser.add_argument(
        "--mode",
        choices=["tutorial", "wordle", "both"],
        default="tutorial",
        help="Game mode to launch with the webcam overlay.",
    )
    args = parser.parse_args()

    GameOverlaySession(mode=args.mode).run()


if __name__ == "__main__":
    main()
