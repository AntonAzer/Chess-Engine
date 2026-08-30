"""
main.py
-------
Entry point. Run this file to play:

    python main.py

You play White; the engine plays Black. The engine will automatically use
a trained neural-network evaluation (chess_eval.pth) if present, otherwise
it falls back instantly to the handcrafted material + piece-square-table
evaluation -- no setup required to get started.

To train the neural evaluator yourself first, run:

    python train_eval.py

STRENGTH / TIME CONTROL
------------------------
Playing strength is governed almost entirely by `--time` (how many seconds
the engine gets to think per move), because iterative deepening just keeps
searching one ply deeper for as long as you let it. Rough guide with this
engine's search (alpha-beta + transposition table + null-move pruning +
late move reductions + PVS + quiescence):

    --time 3    ~depth 5-6   fast, casual-strength opponent
    --time 10   ~depth 7-8   solid club-level tactics
    --time 20+  ~depth 9-10  strongest practical setting on a laptop CPU

There's no way to *guarantee* a specific Elo number -- actual strength
depends on your CPU, the opponent, and time controls, and would need to be
measured with real games (e.g. against another rated engine) to know for
sure. What this configuration maximizes is depth and search quality within
whatever time budget you give it; if you want it stronger still, raise
--time, or run on a faster machine, or (bigger project) rewrite the hot
evaluation path in something like Cython/PyPy/Numba, since pure Python is
the actual speed ceiling here.

    python main.py --depth 32 --time 20
"""

import argparse
from gui import ChessGUI


def parse_args():
    parser = argparse.ArgumentParser(description="Play chess against a Python engine.")
    parser.add_argument("--depth", type=int, default=32,
                         help="Hard depth ceiling (rarely reached; iterative deepening "
                              "stops earlier based on --time). Default: 32")
    parser.add_argument("--time", type=float, default=12.0,
                         help="Time budget in seconds the engine gets per move -- the "
                              "main strength knob. Default: 12.0")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Starting Python Chess Engine...")
    print(f"  Search depth cap: {args.depth}")
    print(f"  Time per move:    {args.time}s")
    print("  You play White (bottom of board). Click a piece, then a highlighted square to move.")
    print("  Press 'R' at any time to restart the game.\n")

    gui = ChessGUI(engine_depth=args.depth, engine_time_limit=args.time)
    gui.run()


if __name__ == "__main__":
    main()
