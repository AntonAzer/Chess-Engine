"""
train_eval.py
-------------
Standalone script that:

  1. Generates a synthetic dataset of chess positions by playing random
     (capture-biased) games from the starting position.
  2. Labels each position by running a SHORT ALPHA-BETA SEARCH (via
     engine.SearchEngine, using the handcrafted evaluator) rather than
     just the static handcrafted evaluation. This matters: a plain static
     eval can't see "I hang my queen next move" -- a 2-3 ply search can.
     Training the network to imitate a shallow search (rather than a
     single static score) teaches it patterns a lookup-table PST can't
     express, which is what actually makes the NN a useful add-on instead
     of a duplicate of the handcrafted function it gets blended with.
  3. Converts every FEN into an 8x8x12 binary tensor (eval_model.py).
  4. Trains ChessEvalNet to regress the (scaled) search-based evaluation.
  5. Saves the trained weights to chess_eval.pth, which engine.py will
     automatically pick up on the next run.

Run it with:

    python train_eval.py

This takes roughly 10-25 minutes on a normal CPU with the default
settings below (dominated by step 2, the search-based labeling -- not
model training itself, which is fast). Turn NUM_GAMES down for a much
quicker (but lower quality) run while you're testing the pipeline, or up
for a stronger model.
"""

import random
import time
import os

import chess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from eval_model import ChessEvalNet, board_to_tensor
from engine import Evaluator, SearchEngine

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
NUM_GAMES = 220            # number of random self-play games to sample from
MAX_PLIES_PER_GAME = 40    # cap game length so we don't waste time on endgames
POSITIONS_PER_GAME = 5     # random positions sampled from each game
LABEL_SEARCH_DEPTH = 3     # ply depth of the labeling search (quality vs. speed)
LABEL_TIME_LIMIT = 1.5     # per-position time cap for the labeling search (s)
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
SEED = 42
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chess_eval.pth")

random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------------------------------
# Synthetic dataset generation
# --------------------------------------------------------------------------

def random_capture_biased_move(board: chess.Board) -> chess.Move:
    """Pick a random legal move, weighting captures higher so games explore
    tactically interesting (and thus more informative) positions."""
    moves = list(board.legal_moves)
    captures = [m for m in moves if board.is_capture(m)]
    if captures and random.random() < 0.35:
        return random.choice(captures)
    return random.choice(moves)


def generate_fens(num_games: int = NUM_GAMES):
    """Play random games and collect a diverse set of FENs (not yet
    labeled -- labeling with a search is the expensive part, done next)."""
    fens = []
    for game_idx in range(num_games):
        board = chess.Board()
        game_fens = []

        for _ply in range(MAX_PLIES_PER_GAME):
            if board.is_game_over():
                break
            move = random_capture_biased_move(board)
            board.push(move)
            game_fens.append(board.fen())

        if not game_fens:
            continue

        sample_count = min(POSITIONS_PER_GAME, len(game_fens))
        fens.extend(random.sample(game_fens, sample_count))

        if (game_idx + 1) % 50 == 0:
            print(f"  generated {game_idx + 1}/{num_games} games "
                  f"({len(fens)} positions so far)...")

    return fens


def label_positions(fens, search_depth: int = LABEL_SEARCH_DEPTH):
    """Label each FEN with a short alpha-beta search score (White's
    perspective, centipawns) using the same handcrafted evaluator the
    engine falls back on. This is what makes the resulting network worth
    blending in: it's distilling a few plies of tactical lookahead into a
    single forward pass, not just memorizing the static eval."""
    evaluator = Evaluator(blend=0.0)  # force pure handcrafted eval as the teacher
    searcher = SearchEngine(evaluator, max_depth=search_depth, time_limit=LABEL_TIME_LIMIT)

    labeled = []
    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        searcher.tt.clear()  # keep memory bounded across many positions
        _move, score, _depth, _nodes = searcher.choose_move(board)
        labeled.append((fen, score))

        if (i + 1) % 100 == 0:
            print(f"  labeled {i + 1}/{len(fens)} positions...")

    return labeled


class ChessPositionDataset(Dataset):
    """Wraps (fen, score) pairs as tensors ready for training."""

    def __init__(self, positions, scale: float):
        self.samples = []
        for fen, score in positions:
            board = chess.Board(fen)
            x = board_to_tensor(board)
            clipped = max(-scale, min(scale, score))
            y = torch.tensor([clipped / scale], dtype=torch.float32)
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------

def train():
    print("=" * 70)
    print("Chess Evaluation Network - Training")
    print("=" * 70)

    print(f"\n[1/5] Generating synthetic self-play positions "
          f"({NUM_GAMES} games)...")
    t0 = time.time()
    fens = generate_fens(NUM_GAMES)
    print(f"      -> {len(fens)} candidate positions in {time.time() - t0:.1f}s")

    print(f"\n[2/5] Labeling positions with a depth-{LABEL_SEARCH_DEPTH} search "
          f"(this is the slow step)...")
    t0 = time.time()
    positions = label_positions(fens)
    print(f"      -> {len(positions)} labeled positions in {time.time() - t0:.1f}s")

    model = ChessEvalNet()
    scale = model.SCALE

    print("\n[3/5] Building dataset / dataloader...")
    dataset = ChessPositionDataset(positions, scale=scale)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    print(f"      -> {train_size} train / {val_size} validation samples")

    print("\n[4/5] Training network...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
        train_loss = total_loss / max(1, train_size)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= max(1, val_size)

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            lr = optimizer.param_groups[0]["lr"]
            print(f"      epoch {epoch:3d}/{EPOCHS} | "
                  f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | lr={lr:.5f}")

    print(f"\n[5/5] Saving weights to '{OUTPUT_PATH}'...")
    torch.save(model.state_dict(), OUTPUT_PATH)
    print("      -> done!")

    print("\n" + "=" * 70)
    print("Training complete. engine.py will automatically load "
          "chess_eval.pth the next time it runs.")
    print("=" * 70)


if __name__ == "__main__":
    train()
