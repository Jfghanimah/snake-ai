"""
AlphaZero MCTS for Snake.

Key differences from classical MCTS:
  - No random rollouts.  The network's value head replaces them entirely.
  - Node selection uses the PUCT formula (polynomial upper confidence for trees):
        PUCT(a) = Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))
  - Dirichlet noise is mixed into the root's priors for exploration during self-play.
  - The tree is rebuilt from scratch on every move (no tree reuse).

Node storage:  nodes store only PUCT statistics (N, W, P, children).
               The game state is reconstructed by stepping a cloned Snake
               along the selected path, so no Snake objects live in nodes.
"""

import contextlib
import io
import math
from typing import Dict, List, Optional

import numpy as np

from app.alpha_zero.network import PolicyValueNet, encode_state
from app.environment import Snake


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _silent_step(snake: Snake, action: int):
    """Call snake.step() while suppressing all print output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return snake.step(action)


def _clone(snake: Snake) -> Snake:
    """
    Fast manual clone of a Snake — avoids __init__ / reset overhead.
    Only the numpy map needs .copy(); all other fields are scalars or tuples.
    """
    s = Snake.__new__(Snake)
    s.map_height = snake.map_height
    s.map_width  = snake.map_width
    s.map        = snake.map.copy()
    s.is_dead    = snake.is_dead
    s.is_won     = snake.is_won
    s.steps      = snake.steps
    s.length     = snake.length
    s.head_pos   = snake.head_pos   # tuple — immutable, no copy needed
    s.tail_pos   = snake.tail_pos
    s.food       = snake.food
    return s


def _valid_actions(snake: Snake) -> List[int]:
    """
    Actions that don't immediately wall or self-collide.
    Tail (map value == 1) is safe to enter because it will vacate before the head arrives.
    Food (map value == -1) is also safe (<= 1 catches it).
    """
    y, x = snake.head_pos
    H, W = snake.map_height, snake.map_width
    moves = {0: (y - 1, x), 1: (y + 1, x), 2: (y, x - 1), 3: (y, x + 1)}
    valid = []
    for action, (ny, nx) in moves.items():
        if 0 <= ny < H and 0 <= nx < W and snake.map[ny, nx] <= 1:
            valid.append(action)
    return valid


def _mask_and_renorm(probs: np.ndarray, valid: List[int]) -> np.ndarray:
    """Zero out invalid actions and renormalize.  Falls back to uniform if all priors are 0."""
    masked = np.zeros(4, dtype=np.float32)
    for a in valid:
        masked[a] = probs[a]
    total = masked.sum()
    if total < 1e-8:
        for a in valid:
            masked[a] = 1.0 / len(valid)
    else:
        masked /= total
    return masked


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class MCTSNode:
    """
    Lightweight node — stores only the statistics needed for PUCT selection.
    Game state is NOT stored here; it's reconstructed during tree traversal.
    """

    __slots__ = ("N", "W", "P", "children", "is_expanded", "is_terminal", "terminal_value")

    def __init__(self, prior: float = 1.0) -> None:
        self.N: int   = 0
        self.W: float = 0.0
        self.P: float = prior
        self.children: Dict[int, "MCTSNode"] = {}
        self.is_expanded:     bool           = False
        self.is_terminal:     bool           = False
        self.terminal_value:  Optional[float] = None

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def puct(self, parent_N: int, c_puct: float) -> float:
        return self.Q + c_puct * self.P * math.sqrt(parent_N) / (1 + self.N)


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

class MCTS:
    def __init__(
        self,
        net: PolicyValueNet,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
    ) -> None:
        self.net = net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, snake: Snake) -> np.ndarray:
        """
        Run MCTS from the given game state.

        Returns pi: (4,) float32 array of visit-count probabilities.
        Actions with zero visits (invalid or unexplored) have pi=0.
        """
        root = MCTSNode()
        root_value = self._expand(root, snake)

        if root.is_terminal or not root.children:
            # Already over or completely boxed in
            pi = np.zeros(4, dtype=np.float32)
            for a in root.children:
                pi[a] = 1.0 / len(root.children)
            return pi

        # Add Dirichlet noise to root's child priors (exploration)
        actions = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for i, a in enumerate(actions):
            c = root.children[a]
            c.P = (1 - self.dirichlet_eps) * c.P + self.dirichlet_eps * noise[i]

        # Run simulations
        for _ in range(self.num_simulations):
            sim = _clone(snake)
            self._simulate(root, sim)

        # Convert visit counts → policy vector
        pi = np.zeros(4, dtype=np.float32)
        total_visits = sum(c.N for c in root.children.values())
        if total_visits > 0:
            for a, child in root.children.items():
                pi[a] = child.N / total_visits
        return pi

    # ------------------------------------------------------------------
    # Internal simulation
    # ------------------------------------------------------------------

    def _simulate(self, root: MCTSNode, sim: Snake) -> None:
        """
        One full simulation:  selection → expansion/evaluation → backprop.
        `sim` is a clone of the root game state; it is stepped in-place.
        """
        node = root
        path: List[MCTSNode] = [root]

        # --- Selection: descend until an unexpanded leaf or terminal ---
        while node.is_expanded and not node.is_terminal and node.children:
            parent_N = node.N
            action = max(
                node.children,
                key=lambda a: node.children[a].puct(parent_N, self.c_puct),
            )
            _silent_step(sim, action)
            node = node.children[action]
            path.append(node)

        # --- Evaluation ---
        if node.is_terminal:
            value = node.terminal_value          # pre-computed at expansion
        elif sim.is_dead:
            value = -1.0
        elif sim.is_won:
            value = 1.0
        else:
            # Unexpanded leaf: ask the network, then expand
            value = self._expand(node, sim)

        # --- Backpropagation ---
        for n in path:
            n.N += 1
            n.W += value

    def _expand(self, node: MCTSNode, snake: Snake) -> float:
        """
        Evaluate `node` with the network and create its children.
        Returns the network's value estimate (or terminal value).
        """
        if node.is_expanded:
            return node.terminal_value if node.is_terminal else node.W / max(node.N, 1)

        # Terminal: already dead or won
        if snake.is_dead:
            node.is_terminal    = True
            node.terminal_value = -1.0
            node.is_expanded    = True
            return -1.0
        if snake.is_won:
            node.is_terminal    = True
            node.terminal_value = 1.0
            node.is_expanded    = True
            return 1.0

        valid = _valid_actions(snake)
        if not valid:
            node.is_terminal    = True
            node.terminal_value = -1.0
            node.is_expanded    = True
            return -1.0

        # Network inference
        probs, value = self.net.predict(encode_state(snake))
        masked = _mask_and_renorm(probs, valid)

        # Create one child per valid action.
        # Step a temporary clone to detect immediate terminals cheaply.
        for action in valid:
            child_sim = _clone(snake)
            _silent_step(child_sim, action)

            child = MCTSNode(prior=float(masked[action]))
            if child_sim.is_dead:
                child.is_terminal    = True
                child.terminal_value = -1.0
                child.is_expanded    = True
            elif child_sim.is_won:
                child.is_terminal    = True
                child.terminal_value = 1.0
                child.is_expanded    = True

            node.children[action] = child

        node.is_expanded = True
        return value


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def visit_counts_to_probs(pi: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply temperature to a (4,) visit-count probability vector.

    temperature = 1.0  → proportional to visit counts  (exploration)
    temperature ≈ 0    → argmax  (exploitation)
    """
    if temperature < 1e-6:
        out = np.zeros(4, dtype=np.float32)
        out[int(np.argmax(pi))] = 1.0
        return out
    powered = pi ** (1.0 / temperature)
    total = powered.sum()
    if total < 1e-8:
        return pi.copy()
    return (powered / total).astype(np.float32)
