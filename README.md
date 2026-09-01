# ASL Game Mechanics

This repository is the gameplay layer for an ASL learning system. It contains the mechanics for a tutorial phase and multiple game modes, including a Wordle-style game, while integrating with the trained models produced in the `asl-training` project.

## Project purpose

- Teach basic ASL gestures in a guided tutorial
- Convert live model predictions into game input
- Support multiple game loops such as Wordle-style guessing
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
