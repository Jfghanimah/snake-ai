"""
AlphaZero self-play training loop for Snake.

Run from the repo root:
    python -m app.alpha_zero.train

Requires:
    pip install torch numpy

The AlphaZero training cycle (per iteration):
  1. Self-play   — run MCTS-guided games; collect (state, π, z) tuples
  2. Train       — sample from replay buffer; minimise value MSE + policy cross-entropy
  3. Checkpoint  — save model weights every N iterations

Outcome labels (z):
  +1  snake wins (fills the board)
  -1  snake dies or times out without eating for too long
"""

import contextlib
import io
import os
import random
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from app.alpha_zero.mcts import MCTS, _clone, _silent_step, visit_counts_to_probs
from app.alpha_zero.network import PolicyValueNet, encode_state
from app.environment import Snake

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

BOARD_H = BOARD_W = 8

# Network
CHANNELS       = 64
NUM_RES_BLOCKS = 4

# MCTS
NUM_SIMULATIONS  = 25    # simulations per move (reduce to ~25 for fast testing)
C_PUCT           = 1.5
DIRICHLET_ALPHA  = 0.3    # concentration — lower = more random exploration
DIRICHLET_EPS    = 0.25   # fraction of noise mixed into root priors

# Self-play
GAMES_PER_ITER      = 25
TEMP_THRESHOLD      = 15  # use τ=1 for first N moves of each game, then τ→0
MAX_NO_FOOD_STEPS   = BOARD_H * BOARD_W * 2   # 128 — prevents looping
MAX_GAME_STEPS      = BOARD_H * BOARD_W * 20  # 1280 — hard ceiling

# Training
NUM_ITERS          = 100
BATCH_SIZE         = 256
TRAIN_STEPS        = 200
LR                 = 1e-3
WEIGHT_DECAY       = 1e-4   # L2 regularisation
REPLAY_MAXLEN      = 50_000
MIN_BUFFER_SIZE    = 1_000  # don't train until buffer has this many samples

CHECKPOINT_DIR = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

# Each entry: (encoded_state (3,H,W) float32, pi (4,) float32, z float32)
Example = Tuple[np.ndarray, np.ndarray, float]


class ReplayBuffer:
    def __init__(self, maxlen: int) -> None:
        self._buf: deque = deque(maxlen=maxlen)

    def extend(self, examples: List[Example]) -> None:
        self._buf.extend(examples)

    def sample(self, n: int) -> List[Example]:
        return random.sample(self._buf, min(n, len(self._buf)))

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------

def self_play_game(
    net: PolicyValueNet,
    mcts: MCTS,
) -> Tuple[List[Example], bool, int]:
    """
    Play one full game via MCTS self-play.

    Returns:
        examples  — list of (state, π, z) training tuples
        won       — True if the snake filled the board
        food_eaten — number of food pellets consumed
    """
    game = Snake(BOARD_H, BOARD_W)
    history: List[Tuple[np.ndarray, np.ndarray]] = []   # (state, π) without z

    move_num        = 0
    steps_no_food   = 0

    while not game.is_dead and not game.is_won:
        # Timeout: treat as death (encourages the network to avoid looping)
        if steps_no_food > MAX_NO_FOOD_STEPS or move_num > MAX_GAME_STEPS:
            game.is_dead = True
            break

        # Encode state BEFORE stepping
        state = encode_state(game)           # (3, H, W)

        # MCTS search → raw visit-count π
        pi_raw = mcts.run(game)

        # Apply temperature scheduling
        temp = 1.0 if move_num < TEMP_THRESHOLD else 0.0
        pi   = visit_counts_to_probs(pi_raw, temp)

        history.append((state, pi.copy()))

        # Sample action
        if pi.sum() < 1e-8:
            break   # completely boxed in
        action = int(np.random.choice(4, p=pi / pi.sum()))

        result = _silent_step(game, action)

        if result == "ate":
            steps_no_food = 0
        else:
            steps_no_food += 1
        move_num += 1

    # Assign outcome
    z          = 1.0 if game.is_won else -1.0
    food_eaten = game.length - 2   # snake starts at length 2

    examples: List[Example] = [
        (s.astype(np.float32), p.astype(np.float32), np.float32(z))
        for s, p in history
    ]
    return examples, game.is_won, food_eaten


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(
    net: PolicyValueNet,
    optimizer: optim.Optimizer,
    batch: List[Example],
) -> Tuple[float, float]:
    """
    One gradient update.  Returns (value_loss, policy_loss).

    Loss = MSE(v, z) + CrossEntropy(p, π)
    L2 regularisation is handled by the optimizer's weight_decay.

    Soft cross-entropy:  -(π · log p)  where π is the MCTS target distribution
    and p = softmax(logits).  This differs from nn.CrossEntropyLoss which
    expects hard integer labels.
    """
    states, pis, zs = zip(*batch)

    state_t = torch.tensor(np.stack(states), dtype=torch.float32, device=DEVICE)
    pi_t    = torch.tensor(np.stack(pis),    dtype=torch.float32, device=DEVICE)
    z_t     = torch.tensor(zs,               dtype=torch.float32, device=DEVICE).unsqueeze(1)

    net.train()
    logits, v = net(state_t)

    value_loss  = F.mse_loss(v, z_t)
    policy_loss = -(pi_t * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    loss        = value_loss + policy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(value_loss.item()), float(policy_loss.item())


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(net: PolicyValueNet, optimizer: optim.Optimizer, iteration: int) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"az_iter_{iteration:04d}.pt")
    torch.save(
        {
            "iteration":       iteration,
            "model_state":     net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )
    return path


def load_checkpoint(path: str) -> Tuple[PolicyValueNet, optim.Optimizer, int]:
    ckpt = torch.load(path, map_location=DEVICE)
    net  = PolicyValueNet(BOARD_H, BOARD_W, channels=CHANNELS, num_res_blocks=NUM_RES_BLOCKS).to(DEVICE)
    net.load_state_dict(ckpt["model_state"])
    opt  = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    opt.load_state_dict(ckpt["optimizer_state"])
    return net, opt, ckpt["iteration"]


# ---------------------------------------------------------------------------
# Evaluation (greedy play — no exploration noise, τ=0)
# ---------------------------------------------------------------------------

def evaluate(net: PolicyValueNet, num_games: int = 10) -> dict:
    """Play num_games greedily and report average food eaten and win rate."""
    mcts = MCTS(net, num_simulations=NUM_SIMULATIONS, c_puct=C_PUCT,
                dirichlet_alpha=0.0, dirichlet_eps=0.0)
    wins, foods = 0, []
    for _ in range(num_games):
        _, won, food = self_play_game(net, mcts)
        wins  += int(won)
        foods.append(food)
    return {"win_rate": wins / num_games, "avg_food": sum(foods) / len(foods)}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Board: {BOARD_H}×{BOARD_W}  |  ResNet: {NUM_RES_BLOCKS} blocks × {CHANNELS} ch")
    print(f"MCTS: {NUM_SIMULATIONS} sims/move  |  Replay buffer: {REPLAY_MAXLEN}")
    print()

    net       = PolicyValueNet(BOARD_H, BOARD_W, channels=CHANNELS, num_res_blocks=NUM_RES_BLOCKS).to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    mcts      = MCTS(net, num_simulations=NUM_SIMULATIONS, c_puct=C_PUCT,
                     dirichlet_alpha=DIRICHLET_ALPHA, dirichlet_eps=DIRICHLET_EPS)
    buffer    = ReplayBuffer(REPLAY_MAXLEN)

    for iteration in range(1, NUM_ITERS + 1):

        # ── Self-play ────────────────────────────────────────────────────────
        net.eval()
        total_food, wins = 0, 0
        for game_i in range(1, GAMES_PER_ITER + 1):
            print(f"\r[{iteration:03d}] self-play {game_i}/{GAMES_PER_ITER}  buf={len(buffer)}  ", end="", flush=True)
            examples, won, food = self_play_game(net, mcts)
            buffer.extend(examples)
            total_food += food
            wins       += int(won)

        # ── Training ─────────────────────────────────────────────────────────
        v_losses, p_losses = [], []
        if len(buffer) >= MIN_BUFFER_SIZE:
            for step_i in range(1, TRAIN_STEPS + 1):
                if step_i % 50 == 0 or step_i == 1:
                    print(f"\r[{iteration:03d}] training  {step_i}/{TRAIN_STEPS}                    ", end="", flush=True)
                batch = buffer.sample(BATCH_SIZE)
                vl, pl = train_step(net, optimizer, batch)
                v_losses.append(vl)
                p_losses.append(pl)

        # ── Logging ──────────────────────────────────────────────────────────
        avg_food = total_food / GAMES_PER_ITER
        win_rate = wins / GAMES_PER_ITER
        buf_size = len(buffer)

        if v_losses:
            print(
                f"\r[{iteration:03d}]  "
                f"food={avg_food:5.1f}  win={win_rate:.0%}  "
                f"buf={buf_size:6d}  "
                f"v_loss={sum(v_losses)/len(v_losses):.4f}  "
                f"p_loss={sum(p_losses)/len(p_losses):.4f}"
            )
        else:
            print(
                f"\r[{iteration:03d}]  "
                f"food={avg_food:5.1f}  win={win_rate:.0%}  "
                f"buf={buf_size:6d}  (collecting data...)"
            )

        # ── Checkpoint ───────────────────────────────────────────────────────
        if iteration % 10 == 0:
            path = save_checkpoint(net, optimizer, iteration)
            print(f"       → checkpoint saved: {path}")


if __name__ == "__main__":
    main()
