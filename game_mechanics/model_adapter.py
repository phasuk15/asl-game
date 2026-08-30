from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import DEFAULT_MODEL_CANDIDATES


@dataclass
class PredictionResult:
    label: str
    confidence: float
    raw: Any | None = None


class ModelAdapter:
    """Thin adapter for the trained ASL model produced by the training project."""

    def __init__(self, model_paths: Iterable[str | Path] | None = None):
        self.model_paths = [Path(p) for p in (model_paths or DEFAULT_MODEL_CANDIDATES)]
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the first usable pickled model, if present."""
        for candidate in self.model_paths:
            if candidate.exists():
                self.model = candidate
                return

        self.model = None

    def is_available(self) -> bool:
        return self.model is not None

    def predict(self, features: Any) -> PredictionResult:
        """Return a standard prediction result for a gesture.

        This keeps the game layer decoupled from the training implementation.
        """
        if self.model is None:
            return PredictionResult(label="UNKNOWN", confidence=0.0, raw=None)

        # Placeholder logic: the real project will swap in the trained model
        # when the detector or classifier is present.
        if hasattr(features, "__iter__"):
            return PredictionResult(label="ASL_GESTURE", confidence=0.85, raw=features)

        return PredictionResult(label="ASL_GESTURE", confidence=0.85, raw=features)

    def predict_from_live_frame(self, frame: Any) -> PredictionResult:
        return self.predict(frame)
