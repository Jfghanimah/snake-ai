# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Exploring different snake game algorithms to fully complete the game as efficiently as possible.

## Setup

```bash
# Activate the virtual environment
source venv/Scripts/activate  # Windows (bash)

# Install dependencies (not yet in requirements.txt)
pip install numpy pygame
```

## Running

```bash
# Interactive GUI game (requires pygame)
python app/snakegame.py

# Command-line test mode (8x8 grid, no GUI required)
python app/env_test.py
```

### snakegame.py controls
- Arrow keys: manual snake control
- `Q`: single bot step
- `W`: enable continuous bot mode
- `E`: disable bot mode
- `A`: bot moves until it eats food

## Architecture

**[app/environment.py](app/environment.py)** — `Snake` class: core game engine
- Grid stored as a numpy 2D array (`map`); values: `0`=empty, `1+`=snake body (1=tail, increasing toward head), `-1`=food
- Coordinates are `(y, x)` with `(0,0)` at top-left
- Actions: `0`=up, `1`=down, `2`=left, `3`=right
- Win condition: snake length equals total grid cells

**[app/snake_bot.py](app/snake_bot.py)** — `SnakeBot` class: AI agent
- Primary strategy: A* to food, but only if a path back to the tail still exists afterward
- Fallback strategy: "worst path finding" — BFS that maximizes path length (keeps snake alive when food path is unsafe)
- `update_tail_map()`: BFS from the tail to compute which cells are reachable; used as safety constraint in all pathfinding
- A* heuristic: Manhattan distance + turn penalty

**[app/snakegame.py](app/snakegame.py)** — Pygame GUI renderer; integrates `Snake` + `SnakeBot`

**[app/env_test.py](app/env_test.py)** — CLI wrapper for manual testing without pygame
