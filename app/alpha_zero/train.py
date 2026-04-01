"""
AlphaZero self-play training for Snake — concurrent multiprocessing design.

Architecture:
  N_WORKERS worker processes  — CPU-based self-play (MCTS + network inference)
  1 trainer process (main)    — GPU-based training on replay buffer samples

Data flow:
  workers  →  examples_queue  →  main (buffer → train)
  main     →  weights_queue   →  workers (updated weights every N games)

Run from the repo root:
    python -m app.alpha_zero.train

Requires:
    pip install torch numpy
"""

import os
import queue
import random
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim

from app.alpha_zero.mcts import MCTS, _silent_step, visit_counts_to_probs
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
NUM_SIMULATIONS = 25
C_PUCT          = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPS   = 0.25

# Self-play
N_WORKERS           = 4     # CPU worker processes for self-play
WEIGHT_UPDATE_EVERY = 20    # broadcast updated weights to workers every N games
GAMES_PER_ITER      = 25    # games between log lines
TEMP_THRESHOLD      = 15    # τ=1 for first N moves, then τ→0
MAX_NO_FOOD_STEPS   = BOARD_H * BOARD_W * 2
MAX_GAME_STEPS      = BOARD_H * BOARD_W * 20

# Training
BATCH_SIZE           = 512
TRAIN_STEPS_PER_GAME = 10   # gradient steps after receiving each game
LR                   = 1e-3
WEIGHT_DECAY         = 1e-4
REPLAY_MAXLEN        = 50_000
MIN_BUFFER_SIZE      = 1_000

CHECKPOINT_DIR = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Example = Tuple[np.ndarray, np.ndarray, float]   # (state, pi, z)

# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

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
# Self-play (runs inside worker processes)
# ---------------------------------------------------------------------------

def self_play_game(mcts: MCTS) -> Tuple[List[Example], bool, int]:
    """Play one full game. Returns (examples, won, food_eaten)."""
    game = Snake(BOARD_H, BOARD_W)
    history: List[Tuple[np.ndarray, np.ndarray]] = []
    move_num, steps_no_food = 0, 0

    while not game.is_dead and not game.is_won:
        if steps_no_food > MAX_NO_FOOD_STEPS or move_num > MAX_GAME_STEPS:
            game.is_dead = True
            break

        state  = encode_state(game)
        pi_raw = mcts.run(game)
        temp   = 1.0 if move_num < TEMP_THRESHOLD else 0.0
        pi     = visit_counts_to_probs(pi_raw, temp)

        history.append((state, pi.copy()))

        if pi.sum() < 1e-8:
            break
        action = int(np.random.choice(4, p=pi / pi.sum()))
        result = _silent_step(game, action)

        steps_no_food = 0 if result == "ate" else steps_no_food + 1
        move_num += 1

    z          = 1.0 if game.is_won else -1.0
    food_eaten = game.length - 2

    examples = [(s.astype(np.float32), p.astype(np.float32), np.float32(z))
                for s, p in history]
    return examples, game.is_won, food_eaten


def worker_fn(examples_q: mp.Queue, weights_q: mp.Queue) -> None:
    """
    Worker process: runs self-play games on CPU indefinitely.
    Pushes completed games to examples_q.
    Checks weights_q for updated network weights before each game.
    """
    net = PolicyValueNet(BOARD_H, BOARD_W, channels=CHANNELS, num_res_blocks=NUM_RES_BLOCKS)
    net.eval()

    mcts = MCTS(net, num_simulations=NUM_SIMULATIONS, c_puct=C_PUCT,
                dirichlet_alpha=DIRICHLET_ALPHA, dirichlet_eps=DIRICHLET_EPS)

    while True:
        # Pull latest weights if available (non-blocking)
        try:
            state_dict = weights_q.get_nowait()
            net.load_state_dict(state_dict)
            net.eval()
        except queue.Empty:
            pass

        examples, won, food = self_play_game(mcts)
        examples_q.put((examples, won, food))   # blocks if queue full (backpressure)

# ---------------------------------------------------------------------------
# Training step (runs in main process on GPU)
# ---------------------------------------------------------------------------

def train_step(
    net: PolicyValueNet,
    optimizer: optim.Optimizer,
    batch: List[Example],
) -> Tuple[float, float]:
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
    torch.save({"iteration": iteration,
                "model_state": net.state_dict(),
                "optimizer_state": optimizer.state_dict()}, path)
    return path


def load_checkpoint(path: str) -> Tuple[PolicyValueNet, optim.Optimizer, int]:
    ckpt = torch.load(path, map_location=DEVICE)
    net  = PolicyValueNet(BOARD_H, BOARD_W, channels=CHANNELS, num_res_blocks=NUM_RES_BLOCKS).to(DEVICE)
    net.load_state_dict(ckpt["model_state"])
    opt  = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    opt.load_state_dict(ckpt["optimizer_state"])
    return net, opt, ckpt["iteration"]

# ---------------------------------------------------------------------------
# Main — trainer process
# ---------------------------------------------------------------------------

def main() -> None:
    mp.set_start_method("spawn", force=True)

    print(f"Device: {DEVICE}")
    print(f"Workers: {N_WORKERS} (CPU self-play) + 1 trainer ({DEVICE.upper()})")
    print(f"Board: {BOARD_H}×{BOARD_W}  |  ResNet: {NUM_RES_BLOCKS} blocks × {CHANNELS} ch")
    print(f"MCTS: {NUM_SIMULATIONS} sims/move  |  Replay buffer: {REPLAY_MAXLEN}")
    print()

    # GPU network for training
    net       = PolicyValueNet(BOARD_H, BOARD_W, channels=CHANNELS, num_res_blocks=NUM_RES_BLOCKS).to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    buffer    = ReplayBuffer(REPLAY_MAXLEN)

    # IPC queues
    # examples_q: workers → trainer  (each item = one completed game)
    # weights_qs: trainer → workers  (one queue per worker, maxsize=1)
    examples_q = mp.Queue(maxsize=N_WORKERS * 4)
    weights_qs = [mp.Queue(maxsize=1) for _ in range(N_WORKERS)]

    # Seed workers with initial weights before they start
    init_weights = {k: v.cpu() for k, v in net.state_dict().items()}
    for wq in weights_qs:
        wq.put(init_weights)

    # Start worker processes
    workers = []
    for i in range(N_WORKERS):
        p = mp.Process(target=worker_fn, args=(examples_q, weights_qs[i]), daemon=True)
        p.start()
        workers.append(p)
    print(f"Started {N_WORKERS} self-play workers.\n")

    game_count = 0
    total_food, wins = 0, 0
    v_losses, p_losses = [], []

    try:
        while True:
            # ── Collect any completed games from workers ──────────────────────
            while True:
                try:
                    examples, won, food = examples_q.get(timeout=0.05)
                except queue.Empty:
                    break

                buffer.extend(examples)
                game_count += 1
                total_food += food
                wins       += int(won)
                print(f"\r  game={game_count}  buf={len(buffer)}  workers={sum(p.is_alive() for p in workers)}  ", end="", flush=True)

                # ── Train on the new data ─────────────────────────────────────
                if len(buffer) >= MIN_BUFFER_SIZE:
                    for _ in range(TRAIN_STEPS_PER_GAME):
                        batch = buffer.sample(BATCH_SIZE)
                        vl, pl = train_step(net, optimizer, batch)
                        v_losses.append(vl)
                        p_losses.append(pl)

                # ── Broadcast updated weights periodically ────────────────────
                if game_count % WEIGHT_UPDATE_EVERY == 0:
                    cpu_weights = {k: v.cpu() for k, v in net.state_dict().items()}
                    for wq in weights_qs:
                        try:
                            wq.get_nowait()   # drop stale weights if worker hasn't read yet
                        except queue.Empty:
                            pass
                        wq.put_nowait(cpu_weights)

                # ── Log every GAMES_PER_ITER games ────────────────────────────
                if game_count % GAMES_PER_ITER == 0:
                    iteration = game_count // GAMES_PER_ITER
                    avg_food  = total_food / GAMES_PER_ITER
                    win_rate  = wins / GAMES_PER_ITER
                    total_food, wins = 0, 0

                    if v_losses:
                        print(
                            f"\r[{iteration:03d}]  game={game_count}  "
                            f"food={avg_food:5.1f}  win={win_rate:.0%}  "
                            f"buf={len(buffer):6d}  "
                            f"v_loss={sum(v_losses)/len(v_losses):.4f}  "
                            f"p_loss={sum(p_losses)/len(p_losses):.4f}"
                        )
                        v_losses, p_losses = [], []
                    else:
                        print(
                            f"\r[{iteration:03d}]  game={game_count}  "
                            f"food={avg_food:5.1f}  win={win_rate:.0%}  "
                            f"buf={len(buffer):6d}  (collecting data...)"
                        )

                    if iteration % 10 == 0:
                        path = save_checkpoint(net, optimizer, iteration)
                        print(f"       → checkpoint saved: {path}")

    except KeyboardInterrupt:
        print("\nStopping workers...")
        for p in workers:
            p.terminate()
        print("Done.")


if __name__ == "__main__":
    main()
