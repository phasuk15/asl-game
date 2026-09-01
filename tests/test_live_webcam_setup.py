from pathlib import Path

from examples.live_webcam import _training_root


def test_training_project_root_is_detected():
    root = _training_root()

    assert root.exists()
    assert root.is_dir()
    assert (root / "config.py").exists()
    assert (root / "assets" / "hand_landmarker.task").exists()
    assert (root / "src" / "static" / "models" / "sign_model.pkl").exists()
