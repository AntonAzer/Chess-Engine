# Python Chess Engine

A from-scratch chess engine with a Pygame GUI, an alpha-beta search core,
and an optional PyTorch neural network evaluator. Play White against the
engine (Black) with a click-to-move interface, a live evaluation bar, and
a move log.

```
python main.py
```

---

## Features

- **Rules & game state** via [`python-chess`](https://python-chess.readthedocs.io/)
  — move generation, legality, check/checkmate/stalemate, FEN, PGN-style
  move stack, etc.
- **Search**: negamax with alpha-beta pruning, plus the techniques that
  actually determine playing strength:
  - Transposition table (Zobrist-hashed, depth-preferring replacement)
  - Move ordering: TT move → MVV-LVA captures → killer moves → history
    heuristic
  - Null-move pruning (with a zugzwang guard for king/pawn endgames)
  - Principal Variation Search + Late Move Reductions
  - Reverse-futility and futility pruning
  - Check extensions (capped, to avoid runaway search)
  - Quiescence search over captures/promotions with delta pruning
  - Iterative deepening with progressive aspiration windows
  - Cheap in-search repetition detection (hash-set based, not the
    expensive `board.can_claim_draw()`)
- **Evaluation**: a handcrafted function (material, piece-square tables,
  bishop pair, rook-file bonuses, passed/doubled/isolated pawns, king
  safety, mobility) that works instantly with zero setup, optionally
  blended with a trained PyTorch network.
- **GUI**: Pygame board with click-to-move, legal-move highlighting,
  check highlighting, an evaluation bar, a move log, and a promotion
  picker. The engine thinks on a background thread so the window never
  freezes.

---

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Entry point — run this to play. |
| `gui.py` | Pygame board, input handling, game loop. |
| `engine.py` | Search (minimax/alpha-beta + all the pruning/ordering above) and the handcrafted evaluation function. |
| `eval_model.py` | PyTorch network architecture + FEN → tensor conversion. |
| `train_eval.py` | Standalone script to train and save the neural evaluator (`chess_eval.pth`). |
| `requirements.txt` | `python-chess`, `pygame`, `torch`. |

---

## Setup

```bash
pip install -r requirements.txt
python main.py
```

That's it — the engine runs immediately using the handcrafted evaluator.
No training or model file is required to play.

### Playing

- You are White (bottom of the board); the engine plays Black.
- Click a piece to select it — legal destination squares light up.
- Click a highlighted square to move there. Click the selected piece
  again (or an invalid square) to deselect.
- Promotions open a small in-window Q/R/B/N picker.
- Press **R** at any time to restart the game.
- The eval bar on the right shows who the engine currently thinks is
  ahead (from White's perspective); the move log lists moves in SAN.

### Command-line options

```bash
python main.py --depth 32 --time 20
```

| Flag | Default | Meaning |
|---|---|---|
| `--time` | `12.0` | Seconds the engine gets to think per move. **This is the main strength knob** — iterative deepening just keeps searching one ply deeper for as long as you allow. |
| `--depth` | `32` | Hard depth ceiling. Rarely reached in practice; `--time` runs out first. |

Rough guide for this search on a normal laptop CPU:

| `--time` | Depth reached | Feel |
|---|---|---|
| 3s | ~5–6 | fast, casual |
| 10s | ~7–8 | solid club-level tactics |
| 20s+ | ~9–10 | strongest practical setting |

There's no way to *guarantee* a specific rating — real strength depends on
your CPU, the opponent, and the time control, and would need to be
measured in actual rated games to know for sure. What these settings
maximize is search depth and quality within whatever time budget you give
it, in my case I see it around 1500 elo bot in defualt, if I increase time the depth with increase then elo.

---

## Training the neural evaluator (optional)

```bash
python train_eval.py
```

This generates random self-play positions, labels each one with a short
alpha-beta search (not just a static score — a shallow search catches
tactics a lookup table can't), trains `ChessEvalNet` to regress those
labels, and saves `chess_eval.pth` next to `engine.py`. The next time you
run `main.py`, the engine automatically detects the file and blends the
network's score in with the handcrafted evaluation.

Takes roughly 10–25 minutes on CPU with the default settings (mostly
spent on search-based labeling, not the training loop itself). Edit the
constants at the top of `train_eval.py` (`NUM_GAMES`, `EPOCHS`,
`LABEL_SEARCH_DEPTH`, etc.) to trade off quality for speed, or vice versa.

To go back to the handcrafted-only evaluator, just delete `chess_eval.pth`.

---

## Tuning / going further

- **`Evaluator(blend=...)`** in `engine.py` controls how much weight the
  NN score gets versus the handcrafted score (default `0.35`).
- The handcrafted evaluation's piece-square tables and bonus constants
  (bishop pair, rook-on-open-file, passed-pawn bonus, king safety, etc.)
  are all plain module-level constants near the top of `engine.py` if you
  want to hand-tune them.
- The single biggest remaining lever for strength is raw search speed:
  this is pure Python, so nodes/second is the ceiling. Rewriting the hot
  evaluation/move-generation path with Cython, Numba, or running under
  PyPy would let the same search techniques reach meaningfully greater
  depth in the same wall-clock time.

---

## Requirements

- Python 3.9+
- `python-chess`, `pygame`, `torch` (see `requirements.txt`)


## Next Development
I want to try LilaZero style in training this type of models and compare the results, the only limitation to get unreachable results is the hardware limitations,but the idea of the model playing itself is the key to get 
unhuman level of engines.
