"""
AlphaZero dual-head Policy-Value network for Snake.

Architecture (mirrors the AlphaZero paper):
  Input (B, 3, H, W)
    → stem conv block
    → N residual blocks  (shared body / "tower")
    → policy head  → (B, 4) logits
    → value  head  → (B, 1) tanh scalar in [-1, 1]

State encoding (3 channels):
  ch0  body   : map_value / length  (0 at empty/food, 1.0 at head)
  ch1  food   : 1.0 at food cell
  ch2  free   : 1.0 at empty cells
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.environment import Snake


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

def encode_state(snake: Snake) -> np.ndarray:
    """Return (3, H, W) float32 array representing the current game state."""
    ch0 = np.clip(snake.map, 0, None).astype(np.float32) / snake.length
    ch1 = (snake.map == -1).astype(np.float32)
    ch2 = (snake.map == 0).astype(np.float32)
    return np.stack([ch0, ch1, ch2])   # (3, H, W)


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Pre-activation residual block with two 3×3 convolutions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


# ---------------------------------------------------------------------------
# Dual-head network
# ---------------------------------------------------------------------------

class PolicyValueNet(nn.Module):
    """
    AlphaZero-style CNN with shared ResNet body and two output heads.

    Policy head : (B, num_actions) raw logits  — apply softmax + action mask outside
    Value  head : (B, 1)           tanh scalar
    """

    def __init__(
        self,
        board_h: int = 8,
        board_w: int = 8,
        in_channels: int = 3,
        channels: int = 64,
        num_res_blocks: int = 4,
        num_actions: int = 4,
    ) -> None:
        super().__init__()
        self.board_h = board_h
        self.board_w = board_w

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Residual tower
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])

        # Policy head: 2-filter 1×1 conv → flatten → FC(4)
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(2 * board_h * board_w, num_actions)

        # Value head: 1-filter 1×1 conv → flatten → FC(256) → FC(1) → tanh
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(board_h * board_w, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        h = self.tower(self.stem(x))

        # Policy (raw logits — caller handles masking and softmax)
        p = self.policy_conv(h).flatten(1)
        p = self.policy_fc(p)

        # Value
        v = self.value_conv(h).flatten(1)
        v = self.value_fc(v)

        return p, v

    @torch.no_grad()
    def predict(self, state: np.ndarray):
        """
        Single-state inference (used by MCTS).

        state : (3, H, W) float32 numpy array
        Returns: (probs (4,) numpy, value float)
            probs are raw softmax over all 4 actions (MCTS will mask + renorm)
        """
        self.eval()
        device = next(self.parameters()).device
        x = torch.from_numpy(state).unsqueeze(0).to(device)   # (1, 3, H, W)
        logits, v = self(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return probs, float(v.item())
