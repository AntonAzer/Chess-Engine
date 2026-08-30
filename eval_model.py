"""
eval_model.py
-------------
Defines the PyTorch neural network architecture used to evaluate chess
positions, plus a helper that converts a python-chess Board/FEN into an
8x8x12 binary tensor (one-hot plane per piece type/color).

The 12 planes are, in order:
    0: white pawn    6: black pawn
    1: white knight  7: black knight
    2: white bishop  8: black bishop
    3: white rook    9: black rook
    4: white queen  10: black queen
    5: white king   11: black king

The tensor is flattened to a 768-length vector before being fed to the
network (768 = 8 * 8 * 12).
"""

import chess
import torch
import torch.nn as nn
import numpy as np

# Maps a python-chess piece type (1-6) to its plane offset for the WHITE side.
_PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

INPUT_SIZE = 8 * 8 * 12  # 768


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Convert a python-chess Board into a flat 768-dim float tensor
    representing an 8x8x12 binary occupancy encoding.

    The board is always encoded from White's perspective (i.e. it does NOT
    flip for black-to-move); side-to-move is handled implicitly because the
    evaluation function negates the score for black-to-move nodes in the
    search (see engine.py).
    """
    planes = np.zeros((12, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)  # rank 8 -> row 0, rank 1 -> row 7
        col = chess.square_file(square)
        plane = _PIECE_TO_PLANE[piece.piece_type]
        if piece.color == chess.BLACK:
            plane += 6
        planes[plane, row, col] = 1.0

    flat = planes.reshape(-1)  # 768
    return torch.from_numpy(flat)


def fen_to_tensor(fen: str) -> torch.Tensor:
    """Convenience wrapper: FEN string -> flat 768-dim tensor."""
    board = chess.Board(fen)
    return board_to_tensor(board)


class ChessEvalNet(nn.Module):
    """
    A small fully-connected network that maps a 768-dim board encoding to a
    single scalar evaluation (in "centipawns from White's perspective",
    scaled down and passed through tanh so it stays in a bounded range that
    is easy to train).
    """

    def __init__(self, input_size: int = INPUT_SIZE, hidden1: int = 256, hidden2: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
        # Output is in [-1, 1]; caller scales by SCALE to get centipawns.
        self.SCALE = 2000.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def evaluate_board(self, board: chess.Board) -> float:
        """Return a centipawn-ish evaluation score from White's perspective."""
        self.eval()
        with torch.no_grad():
            x = board_to_tensor(board).unsqueeze(0)
            out = self.forward(x)
            return float(out.item()) * self.SCALE
