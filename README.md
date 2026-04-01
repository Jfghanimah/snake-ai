# snake-ai

A small AlphaZero-style toy project for Snake.

The project keeps the game logic intentionally simple and focuses on the training loop:

- `app/environment.py` contains the Snake environment.
- `app/alpha_zero/network.py` defines the policy-value network.
- `app/alpha_zero/mcts.py` runs Monte Carlo Tree Search.
- `app/alpha_zero/train.py` drives self-play, training, evaluation, and checkpoints.

## Setup

Create and activate a virtual environment from the repository root.

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux / Git Bash

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Requirements

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

The training stack currently depends on:

- `numpy`
- `torch`

## Run Training

From the repository root:

```bash
python -m app.alpha_zero.train
```

Training writes checkpoints to `checkpoints/` every 10 iterations.

## Notes

- This is a toy clone, not a full production implementation of AlphaZero.
- The project is single-player Snake, so the value target is learned from self-play outcomes rather than from an adversarial game.
- The current entrypoint is the training loop; older manual-control and heuristic-bot experiments have been removed to keep the repo focused.
