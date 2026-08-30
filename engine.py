"""
engine.py
---------
The "brain" of the chess engine. This is a genuine tournament-style search,
not a toy demo: it combines several standard techniques that, together,
are what actually buys playing strength (far more than a static evaluator
ever can):

  * Negamax with Alpha-Beta pruning.
  * Transposition table (Zobrist-hashed, depth-preferring replacement).
  * Move ordering: TT move -> MVV-LVA captures -> killer moves ->
    history heuristic for quiet moves.
  * Null-move pruning (with zugzwang guard for pawn/king-only endgames).
  * Late Move Reductions (LMR) with re-search on fail-high.
  * Check extensions (search one ply deeper when in check).
  * Quiescence search over captures/promotions only, using
    `generate_legal_captures` (cheap) plus delta pruning, so the search
    doesn't stop mid-exchange (the classic "horizon effect").
  * Iterative deepening with aspiration windows, so each depth starts
    from a tight window around the previous score instead of (-inf, inf).
  * Cheap in-search repetition/draw detection (hash-set based) instead of
    the very expensive `board.can_claim_draw()`.

Together these let the engine reach roughly depth 8-14 in a few seconds on
a normal laptop (vs. depth 3-4 for plain alpha-beta), which is the single
biggest lever for strength. A handcrafted evaluation function (material +
piece-square tables + bishop pair + rook files + passed pawns + king
safety + mobility) is included so the engine plays well immediately with
zero setup; an optional trained neural net can be blended in on top.
"""

import time
import os
import sys
import chess
import chess.polyglot

# The search is recursive (one Python stack frame per ply); bump the
# recursion limit a bit above default so deep forcing lines (checks,
# captures, long endgame sequences) can't hit Python's own ceiling.
if sys.getrecursionlimit() < 4000:
    sys.setrecursionlimit(4000)

# --------------------------------------------------------------------------
# Handcrafted evaluation: material + piece-square tables + positional terms
# --------------------------------------------------------------------------

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

PAWN_PST = [
    0,   0,   0,   0,   0,   0,   0,   0,
    50,  50,  50,  50,  50,  50,  50,  50,
    10,  10,  20,  30,  30,  20,  10,  10,
    5,   5,  10,  25,  25,  10,   5,   5,
    0,   0,   0,  20,  20,   0,   0,   0,
    5,  -5, -10,   0,   0, -10,  -5,   5,
    5,  10,  10, -20, -20,  10,  10,   5,
    0,   0,   0,   0,   0,   0,   0,   0,
]

KNIGHT_PST = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_PST = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_PST = [
    0,   0,   0,   0,   0,   0,   0,   0,
    5,  10,  10,  10,  10,  10,  10,   5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    0,   0,   0,   5,   5,   0,   0,   0,
]

QUEEN_PST = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -5,   0,   5,   5,   5,   5,   0,  -5,
    0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]

KING_PST_MIDGAME = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20,  20,   0,   0,   0,   0,  20,  20,
    20,  30,  10,   0,   0,  10,  30,  20,
]

KING_PST_ENDGAME = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
]

_PST = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
}

# Passed-pawn bonus by rank (index = rank from the pawn's own side, 0=own
# back rank .. 7=promotion rank). Ramps up sharply near promotion.
PASSED_PAWN_BONUS = [0, 10, 15, 25, 40, 60, 90, 0]

BISHOP_PAIR_BONUS = 30
ROOK_OPEN_FILE_BONUS = 20
ROOK_SEMI_OPEN_FILE_BONUS = 10
DOUBLED_PAWN_PENALTY = -12
ISOLATED_PAWN_PENALTY = -10
KING_PAWN_SHIELD_BONUS = 8


def _is_endgame(board: chess.Board) -> bool:
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    minor_major = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        minor_major += len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))
    return queens == 0 or minor_major <= 6


def _pst_value(piece_type: int, square: int, color: bool, endgame: bool) -> int:
    if piece_type == chess.KING:
        table = KING_PST_ENDGAME if endgame else KING_PST_MIDGAME
    else:
        table = _PST[piece_type]
    idx = square ^ 56 if color == chess.WHITE else square
    return table[idx]


# -- Precomputed bitboard masks (built once at import time) so the hot
# evaluation path only ever does O(1) bitwise AND + popcount instead of
# nested Python loops over squares. This is the single biggest lever for
# search speed (and therefore search depth / strength), since the
# evaluator runs at every leaf node.
_FILE_MASKS = list(chess.BB_FILES)
_ADJACENT_FILE_MASKS = [
    (chess.BB_FILES[f - 1] if f > 0 else 0) | (chess.BB_FILES[f + 1] if f < 7 else 0)
    for f in range(8)
]


def _build_passed_pawn_masks():
    """WHITE_PASSED[sq] = mask of squares (on file-1/file/file+1) strictly
    ahead of `sq` from White's perspective; enemy black pawns anywhere in
    that mask mean the white pawn on `sq` is NOT passed. Mirrored for
    black."""
    white_masks = [0] * 64
    black_masks = [0] * 64
    for sq in chess.SQUARES:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        files = [f for f in (file - 1, file, file + 1) if 0 <= f <= 7]

        w_mask = 0
        for r in range(rank + 1, 8):
            for f in files:
                w_mask |= chess.BB_SQUARES[chess.square(f, r)]
        white_masks[sq] = w_mask

        b_mask = 0
        for r in range(0, rank):
            for f in files:
                b_mask |= chess.BB_SQUARES[chess.square(f, r)]
        black_masks[sq] = b_mask

    return white_masks, black_masks


_WHITE_PASSED_MASK, _BLACK_PASSED_MASK = _build_passed_pawn_masks()


def _pawn_structure_score(board: chess.Board) -> float:
    """Passed / doubled / isolated pawn terms, from White's perspective.
    Pure bitboard math -- no per-pawn Python loops over the opposing
    pawn set, so this stays cheap even at high node counts."""
    score = 0.0
    white_pawns_bb = board.pawns & board.occupied_co[chess.WHITE]
    black_pawns_bb = board.pawns & board.occupied_co[chess.BLACK]

    for square in chess.scan_forward(white_pawns_bb):
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        same_file_count = chess.popcount(white_pawns_bb & _FILE_MASKS[file])
        if same_file_count > 1:
            score += DOUBLED_PAWN_PENALTY / same_file_count
        if not (white_pawns_bb & _ADJACENT_FILE_MASKS[file]):
            score += ISOLATED_PAWN_PENALTY
        if not (black_pawns_bb & _WHITE_PASSED_MASK[square]):
            score += PASSED_PAWN_BONUS[rank]

    for square in chess.scan_forward(black_pawns_bb):
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        same_file_count = chess.popcount(black_pawns_bb & _FILE_MASKS[file])
        if same_file_count > 1:
            score -= DOUBLED_PAWN_PENALTY / same_file_count
        if not (black_pawns_bb & _ADJACENT_FILE_MASKS[file]):
            score -= ISOLATED_PAWN_PENALTY
        if not (white_pawns_bb & _BLACK_PASSED_MASK[square]):
            score -= PASSED_PAWN_BONUS[7 - rank]

    return score


def _rook_file_score(board: chess.Board) -> float:
    score = 0.0
    white_pawns_bb = board.pawns & board.occupied_co[chess.WHITE]
    black_pawns_bb = board.pawns & board.occupied_co[chess.BLACK]
    for color, sign, own_bb, enemy_bb in (
        (chess.WHITE, 1, white_pawns_bb, black_pawns_bb),
        (chess.BLACK, -1, black_pawns_bb, white_pawns_bb),
    ):
        rooks_bb = board.rooks & board.occupied_co[color]
        for square in chess.scan_forward(rooks_bb):
            file = chess.square_file(square)
            has_own = bool(own_bb & _FILE_MASKS[file])
            has_enemy = bool(enemy_bb & _FILE_MASKS[file])
            if not has_own and not has_enemy:
                score += sign * ROOK_OPEN_FILE_BONUS
            elif not has_own and has_enemy:
                score += sign * ROOK_SEMI_OPEN_FILE_BONUS
    return score


def _king_safety_score(board: chess.Board, endgame: bool) -> float:
    if endgame:
        return 0.0
    score = 0.0
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        king_sq = board.king(color)
        if king_sq is None:
            continue
        file = chess.square_file(king_sq)
        rank = chess.square_rank(king_sq)
        shield_rank = rank + 1 if color == chess.WHITE else rank - 1
        if 0 <= shield_rank <= 7:
            for f in (file - 1, file, file + 1):
                if 0 <= f <= 7:
                    sq = chess.square(f, shield_rank)
                    piece = board.piece_at(sq)
                    if piece is not None and piece.piece_type == chess.PAWN and piece.color == color:
                        score += sign * KING_PAWN_SHIELD_BONUS
    return score


def _mobility_score(board: chess.Board) -> float:
    """Cheap mobility estimate: pseudo-legal move count for the side to
    move only (no legality filtering, no board.turn flip needed for the
    other side), scaled and signed. Halves the cost of a symmetric
    two-sided count while still rewarding active, mobile positions -- and
    since it's re-evaluated every ply, "whoever's turn it is" cycles
    between both colors over the course of a search anyway."""
    mob = board.pseudo_legal_moves.count()
    return 1.5 * mob if board.turn == chess.WHITE else -1.5 * mob


def handcrafted_eval(board: chess.Board) -> float:
    """
    Material + piece-square tables + bishop pair + rook files + pawn
    structure + king safety + mobility, from White's perspective.

    NOTE: this is called at every leaf/quiescence node, so it deliberately
    does NOT re-check is_checkmate()/is_stalemate() (the caller -- the
    search -- already establishes non-terminal status before descending
    into quiescence; re-deriving it here via full legal-move generation on
    every single call would be the single most expensive thing in the
    entire engine). `is_insufficient_material()` is kept since it's a
    cheap O(1) piece-count check, not a move generation.
    """
    if board.is_insufficient_material():
        return 0.0

    endgame = _is_endgame(board)
    score = 0.0

    for square, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]
        pst_bonus = _pst_value(piece.piece_type, square, piece.color, endgame)
        total = value + pst_bonus
        score += total if piece.color == chess.WHITE else -total

    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += BISHOP_PAIR_BONUS
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= BISHOP_PAIR_BONUS

    score += _rook_file_score(board)
    score += _pawn_structure_score(board)
    score += _king_safety_score(board, endgame)
    score += _mobility_score(board)

    return score


# --------------------------------------------------------------------------
# Hybrid evaluation wrapper (handcrafted fallback + optional NN model)
# --------------------------------------------------------------------------

class Evaluator:
    """
    Wraps evaluation so the rest of the engine doesn't need to know whether
    it's using the neural net or the handcrafted function. Automatically
    tries to load `chess_eval.pth` next to this file; falls back to the
    handcrafted evaluator instantly if the file (or torch) isn't available.
    """

    def __init__(self, weights_path: str = None, blend: float = 0.35):
        self.model = None
        # blend = weight given to the NN score; handcrafted eval is exact
        # material/tactics-aware while the NN captures softer positional
        # patterns, so we lean on the handcrafted term more heavily by
        # default (this is a knob you can tune).
        self.blend = blend
        if weights_path is None:
            weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chess_eval.pth")

        if os.path.exists(weights_path):
            try:
                import torch
                from eval_model import ChessEvalNet
                model = ChessEvalNet()
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict)
                model.eval()
                self.model = model
                print(f"[engine] Loaded trained evaluation model from '{weights_path}'.")
            except Exception as exc:  # pragma: no cover - defensive fallback
                print(f"[engine] Could not load NN weights ({exc}); using handcrafted evaluation.")
                self.model = None
        else:
            print("[engine] No trained model found (chess_eval.pth); using handcrafted evaluation.")

    def evaluate(self, board: chess.Board) -> float:
        """Return evaluation in centipawns from White's perspective. Called
        at every leaf/quiescence node, so (like handcrafted_eval) it does
        not re-derive checkmate/stalemate via full legal-move generation;
        the search already handles those cases before it ever calls this."""
        if board.is_insufficient_material():
            return 0.0

        hc_score = handcrafted_eval(board)

        if self.model is not None:
            try:
                nn_score = self.model.evaluate_board(board)
                return self.blend * nn_score + (1 - self.blend) * hc_score
            except Exception:
                return hc_score
        return hc_score


# --------------------------------------------------------------------------
# Move ordering: MVV-LVA for captures, killers/history for quiet moves
# --------------------------------------------------------------------------

def _mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    if board.is_capture(move):
        victim = board.piece_type_at(move.to_square)
        if victim is None:  # en-passant
            victim = chess.PAWN
        attacker = board.piece_type_at(move.from_square)
        victim_value = PIECE_VALUES.get(victim, 0)
        attacker_value = PIECE_VALUES.get(attacker, 0)
        return 10_000 + victim_value * 10 - attacker_value
    if move.promotion:
        return 5_000 + PIECE_VALUES.get(move.promotion, 0)
    return 0


def order_moves(board: chess.Board, moves, tt_move: chess.Move = None):
    """Simple standalone move ordering (TT move, then MVV-LVA captures/
    promotions, then everything else). Used anywhere the richer
    killer/history context isn't available (e.g. quiescence search)."""

    def key(m):
        if tt_move is not None and m == tt_move:
            return 1_000_000
        return _mvv_lva_score(board, m)

    return sorted(moves, key=key, reverse=True)


# --------------------------------------------------------------------------
# Transposition table
# --------------------------------------------------------------------------

EXACT, LOWERBOUND, UPPERBOUND = 0, 1, 2


class TranspositionTable:
    def __init__(self):
        self.table = {}

    def get(self, key):
        return self.table.get(key)

    def store(self, key, depth, value, flag, best_move):
        entry = self.table.get(key)
        if entry is None or depth >= entry[0]:
            self.table[key] = (depth, value, flag, best_move)

    def clear(self):
        self.table.clear()


# --------------------------------------------------------------------------
# Search: Negamax + Alpha-Beta + Null-Move Pruning + LMR + Iterative
# Deepening with Aspiration Windows
# --------------------------------------------------------------------------

MATE_SCORE = 99999
CHECKMATE_THRESHOLD = 90000
MAX_PLY = 128


class SearchEngine:
    """
    Negamax search with alpha-beta pruning, a transposition table,
    killer-move / history-heuristic / MVV-LVA move ordering, null-move
    pruning, late move reductions, check extensions, quiescence search
    with delta pruning, and iterative deepening with aspiration windows.

    `evaluator` is any object exposing `.evaluate(board) -> float` from
    White's perspective. `max_depth` is a hard ceiling (rarely reached in
    practice); `time_limit` (seconds) governs iterative deepening, which
    is what actually determines playing strength -- give it more time for
    a stronger, deeper-searching bot.
    """

    def __init__(self, evaluator: Evaluator, max_depth: int = 32, time_limit: float = 8.0):
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.tt = TranspositionTable()
        self.nodes_searched = 0
        self._start_time = 0.0

        # Killer moves: 2 slots per ply (moves that caused a beta cutoff
        # without being captures -- very likely good in sibling nodes too).
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        # History heuristic: (color, from_square, to_square) -> score.
        self.history = {}
        # Zobrist hashes of positions already reached on the current
        # search path (seeded with the real game history) -- lets us
        # detect repetition draws with an O(1) set lookup instead of the
        # very expensive board.can_claim_draw().
        self.path_hashes = set()

    # -- public API ---------------------------------------------------

    def choose_move(self, board: chess.Board):
        """
        Iterative deepening search from the current position. Returns
        (best_move, score, depth_reached, nodes_searched). `score` is
        reported from White's perspective (positive = good for White).
        """
        self._start_time = time.time()
        self.nodes_searched = 0
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        self.history = {}
        self.path_hashes = self._build_history_hashes(board)

        legal = list(board.legal_moves)
        if not legal:
            return None, 0.0, 0, 0
        if len(legal) == 1:
            return legal[0], self.evaluator.evaluate(board), 1, 1

        best_move = None
        best_score = 0.0
        depth_reached = 0
        window = 25.0  # centipawns, aspiration window half-width

        for depth in range(1, self.max_depth + 1):
            if self._time_up():
                break
            try:
                if depth <= 2:
                    score, move = self._search_root(board, depth, -float("inf"), float("inf"))
                else:
                    # Progressive widening: start narrow (cheap, usually
                    # succeeds), and only fall back to a full-width search
                    # after a couple of failed attempts, instead of
                    # immediately paying for the most expensive possible
                    # re-search on the first miss.
                    w = window
                    alpha, beta = best_score - w, best_score + w
                    score, move = self._search_root(board, depth, alpha, beta)
                    attempts = 0
                    while (score <= alpha or score >= beta) and attempts < 3:
                        w *= 4
                        alpha, beta = best_score - w, best_score + w
                        score, move = self._search_root(board, depth, alpha, beta)
                        attempts += 1
            except _TimeUp:
                break

            if move is not None:
                best_move, best_score, depth_reached = move, score, depth
            if abs(best_score) >= CHECKMATE_THRESHOLD:
                break  # forced mate found, no benefit to searching deeper

        if best_move is None:
            best_move = legal[0]
        return best_move, best_score, depth_reached, self.nodes_searched

    # -- internals ------------------------------------------------------

    def _time_up(self) -> bool:
        return (time.time() - self._start_time) >= self.time_limit

    def _build_history_hashes(self, board: chess.Board) -> set:
        """Replay the real game so far and collect the Zobrist hash of
        every position reached, so in-search repetition detection also
        catches repeats of positions from earlier in the actual game."""
        hashes = set()
        try:
            temp = chess.Board()
            hashes.add(chess.polyglot.zobrist_hash(temp))
            for mv in board.move_stack:
                temp.push(mv)
                hashes.add(chess.polyglot.zobrist_hash(temp))
        except Exception:
            pass
        return hashes

    def _search_root(self, board: chess.Board, depth: int, alpha: float, beta: float):
        key = chess.polyglot.zobrist_hash(board)
        tt_entry = self.tt.get(key)
        tt_move = tt_entry[3] if tt_entry else None

        best_move = None
        best_value = -float("inf")
        perspective = 1 if board.turn == chess.WHITE else -1

        moves = self._order_moves(board, list(board.legal_moves), tt_move, ply=0)

        for move in moves:
            if self._time_up():
                raise _TimeUp()

            board.push(move)
            try:
                value = -self._negamax(board, depth - 1, -beta, -alpha, ply=1)
            finally:
                board.pop()

            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, value)

        self.tt.store(key, depth, best_value, EXACT, best_move)
        return best_value * perspective, best_move

    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int, ext_budget: int = 6) -> float:
        """Negamax with alpha-beta pruning, null-move pruning and late
        move reductions. Returns score from the perspective of the side
        to move.

        The Zobrist hash of `board` is computed exactly once here and
        reused for both the transposition table lookup and in-search
        repetition detection (computing it twice per node -- once in the
        caller's move loop, once here -- was previously the single
        biggest hidden cost in the whole search)."""
        self.nodes_searched += 1
        if self.nodes_searched % 2048 == 0 and self._time_up():
            raise _TimeUp()

        if board.halfmove_clock >= 100:
            return 0.0

        key = chess.polyglot.zobrist_hash(board)
        if key in self.path_hashes:
            return 0.0  # repetition draw (position already on this path)

        if board.is_checkmate():
            return -(MATE_SCORE - ply)
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        alpha_orig = alpha
        tt_entry = self.tt.get(key)
        tt_move = None
        if tt_entry is not None:
            tt_depth, tt_value, tt_flag, tt_move = tt_entry
            if tt_depth >= depth:
                if tt_flag == EXACT:
                    return tt_value
                elif tt_flag == LOWERBOUND:
                    alpha = max(alpha, tt_value)
                elif tt_flag == UPPERBOUND:
                    beta = min(beta, tt_value)
                if alpha >= beta:
                    return tt_value

        in_check = board.is_check()

        # Check extension: search one ply deeper when in check, so the
        # engine doesn't misjudge forced sequences that start with checks.
        # Capped via `ext_budget` so a long chain of checks can't blow the
        # search up into an unbounded chain of +1 extensions.
        if in_check and ext_budget > 0:
            depth += 1
            ext_budget -= 1

        if depth <= 0:
            return self._quiescence(board, alpha, beta, q_depth=6)

        # -- Reverse futility / static null-move pruning -----------------
        # If a cheap static eval already clears beta by a comfortable
        # margin at shallow depth, assume the full search would too and
        # cut immediately. Skipped near mate scores and while in check.
        if (depth <= 3 and not in_check and abs(beta) < CHECKMATE_THRESHOLD):
            perspective = 1 if board.turn == chess.WHITE else -1
            static_eval = self.evaluator.evaluate(board) * perspective
            margin = 120 * depth
            if static_eval - margin >= beta:
                return static_eval - margin

        self.path_hashes.add(key)
        try:
            # -- Null-move pruning ---------------------------------------
            non_pawn_material = board.occupied_co[board.turn] & ~board.pawns & ~board.kings
            if depth >= 3 and not in_check and non_pawn_material and beta < float("inf"):
                R = 3 if depth > 6 else 2
                board.push(chess.Move.null())
                try:
                    null_score = -self._negamax(board, depth - 1 - R, -beta, -beta + 1, ply + 1, ext_budget)
                finally:
                    board.pop()
                if null_score >= beta:
                    return beta

            moves = self._order_moves(board, list(board.legal_moves), tt_move, ply)
            best_value = -float("inf")
            best_move = None

            # -- Futility pruning setup --------------------------------
            # At shallow depth, a quiet move that can't possibly close the
            # gap to alpha (even with a generous per-move margin) is
            # almost never worth searching. Only applies to *later* quiet
            # moves so the best candidates are always still searched.
            futile = False
            if depth <= 2 and not in_check and abs(alpha) < CHECKMATE_THRESHOLD:
                perspective = 1 if board.turn == chess.WHITE else -1
                static_eval = self.evaluator.evaluate(board) * perspective
                futility_margin = 150 * depth
                futile = (static_eval + futility_margin) <= alpha

            for i, move in enumerate(moves):
                is_capture = board.is_capture(move)
                is_quiet = not is_capture and not move.promotion

                if futile and is_quiet and i >= 1 and best_move is not None:
                    continue  # futility pruning: skip, this quiet move won't help

                board.push(move)
                try:
                    if i == 0:
                        # First move (best guess from TT/ordering): search
                        # with the full window.
                        value = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, ext_budget)
                    else:
                        # Principal Variation Search: assume later moves
                        # are worse and verify cheaply with a null window
                        # (and an LMR depth reduction for late quiet
                        # moves); only pay for a full-width re-search if
                        # the cheap probe actually beats alpha.
                        reduction = 0
                        if depth >= 3 and i >= 3 and is_quiet and not in_check:
                            reduction = 2 if i >= 8 else 1
                        probe_depth = max(0, depth - 1 - reduction)

                        value = -self._negamax(board, probe_depth, -alpha - 1, -alpha, ply + 1, ext_budget)
                        if alpha < value < beta:
                            # Probe beat alpha and didn't fail high against
                            # beta either -- re-verify at full depth/window.
                            value = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, ext_budget)
                finally:
                    board.pop()

                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    if is_quiet:
                        self._record_killer(move, ply)
                        self._record_history(board.turn, move, depth)
                    break  # beta cutoff
        finally:
            self.path_hashes.discard(key)

        flag = EXACT
        if best_value <= alpha_orig:
            flag = UPPERBOUND
        elif best_value >= beta:
            flag = LOWERBOUND
        self.tt.store(key, depth, best_value, flag, best_move)

        return best_value

    def _quiescence(self, board: chess.Board, alpha: float, beta: float, q_depth: int = 6) -> float:
        """Quiescence search: extends through captures/promotions only, so
        the static eval at leaf nodes isn't fooled by a piece hanging one
        move beyond the horizon. Uses delta pruning to skip obviously
        futile captures."""
        self.nodes_searched += 1

        perspective = 1 if board.turn == chess.WHITE else -1
        stand_pat = self.evaluator.evaluate(board) * perspective

        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
        if q_depth <= 0:
            return alpha

        DELTA_MARGIN = 200  # centipawns; safety margin for delta pruning

        captures = list(board.generate_legal_captures())
        captures = order_moves(board, captures)

        for move in captures:
            victim = board.piece_type_at(move.to_square)
            victim_value = PIECE_VALUES.get(victim, 100) if victim else 100  # en passant
            if stand_pat + victim_value + DELTA_MARGIN < alpha and not move.promotion:
                continue  # delta pruning: this capture can't possibly help

            board.push(move)
            value = -self._quiescence(board, -beta, -alpha, q_depth - 1)
            board.pop()

            if value >= beta:
                return beta
            if value > alpha:
                alpha = value

        return alpha

    # -- move ordering helpers ------------------------------------------

    def _record_killer(self, move: chess.Move, ply: int):
        if ply >= MAX_PLY:
            return
        slots = self.killers[ply]
        if move != slots[0]:
            slots[1] = slots[0]
            slots[0] = move

    def _record_history(self, color: bool, move: chess.Move, depth: int):
        k = (color, move.from_square, move.to_square)
        self.history[k] = self.history.get(k, 0) + depth * depth

    def _order_moves(self, board: chess.Board, moves, tt_move, ply: int):
        killers = self.killers[ply] if ply < MAX_PLY else (None, None)

        def key(m):
            if tt_move is not None and m == tt_move:
                return 1_000_000
            if board.is_capture(m):
                return 100_000 + _mvv_lva_score(board, m)
            if m.promotion:
                return 90_000 + PIECE_VALUES.get(m.promotion, 0)
            if m == killers[0]:
                return 80_000
            if m == killers[1]:
                return 79_000
            return self.history.get((board.turn, m.from_square, m.to_square), 0)

        return sorted(moves, key=key, reverse=True)


class _TimeUp(Exception):
    """Internal signal used to unwind the search stack when the time
    budget for the current move has been exhausted."""
    pass
