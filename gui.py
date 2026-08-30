"""
gui.py
------
A clean pygame interface for playing chess against the engine.

  * You play White, the engine plays Black.
  * Click a piece to select it; legal destination squares light up.
  * Click a destination square to make the move (click the selected piece
    again, or an invalid square, to deselect).
  * Pawn promotions pop up a small in-window choice of Q/R/B/N.
  * An evaluation bar on the right shows who the engine thinks is better.
  * The engine "thinks" on a background thread so the window never freezes.
"""

import sys
import threading
import queue

import pygame
import chess

from engine import Evaluator, SearchEngine, MATE_SCORE, CHECKMATE_THRESHOLD

# --------------------------------------------------------------------------
# Layout / theme constants
# --------------------------------------------------------------------------

BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8
EVAL_BAR_WIDTH = 40
SIDE_PANEL_WIDTH = 260
MARGIN = 20
WINDOW_WIDTH = BOARD_SIZE + EVAL_BAR_WIDTH + SIDE_PANEL_WIDTH + MARGIN * 3
WINDOW_HEIGHT = BOARD_SIZE + MARGIN * 2

LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT_SELECTED = (246, 246, 105)
HIGHLIGHT_LEGAL = (106, 168, 79)
HIGHLIGHT_LAST_MOVE = (205, 210, 106)
HIGHLIGHT_CHECK = (220, 80, 80)
BG_COLOR = (30, 30, 34)
PANEL_TEXT = (230, 230, 230)
PANEL_SUBTEXT = (160, 160, 165)
EVAL_BAR_WHITE = (245, 245, 245)
EVAL_BAR_BLACK = (25, 25, 25)

UNICODE_PIECES = {
    "P": "\u2659", "N": "\u2658", "B": "\u2657", "R": "\u2656", "Q": "\u2655", "K": "\u2654",
    "p": "\u265F", "n": "\u265E", "b": "\u265D", "r": "\u265C", "q": "\u265B", "k": "\u265A",
}


class ChessGUI:
    def __init__(self, engine_depth: int = 32, engine_time_limit: float = 12.0,
                 human_color: bool = chess.WHITE):
        pygame.init()
        pygame.display.set_caption("Python Chess Engine - You (White) vs Engine (Black)")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_piece = pygame.font.SysFont("segoeuisymbol,dejavusans,arial", SQUARE_SIZE - 14)
        self.font_label = pygame.font.SysFont("arial", 14)
        self.font_title = pygame.font.SysFont("arial", 22, bold=True)
        self.font_body = pygame.font.SysFont("arial", 16)
        self.font_small = pygame.font.SysFont("arial", 13)

        self.board = chess.Board()
        self.human_color = human_color
        self.selected_square = None
        self.legal_targets = []
        self.last_move = None
        self.game_over_message = None

        self.evaluator = Evaluator()
        self.search_engine = SearchEngine(self.evaluator, max_depth=engine_depth,
                                           time_limit=engine_time_limit)

        self.current_eval = 0.0  # centipawns, White's perspective
        self.engine_thinking = False
        self.engine_stats = ""
        self._move_queue = queue.Queue()

        self.move_log = []  # list of SAN strings for the side panel

    # ----------------------------------------------------------------
    # Coordinate helpers
    # ----------------------------------------------------------------

    def _square_to_rect(self, square: int) -> pygame.Rect:
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        # Flip so White's home rank is at the bottom of the window.
        col = file
        row = 7 - rank
        x = MARGIN + col * SQUARE_SIZE
        y = MARGIN + row * SQUARE_SIZE
        return pygame.Rect(x, y, SQUARE_SIZE, SQUARE_SIZE)

    def _pixel_to_square(self, pos):
        x, y = pos
        x -= MARGIN
        y -= MARGIN
        if x < 0 or y < 0 or x >= BOARD_SIZE or y >= BOARD_SIZE:
            return None
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        file = col
        rank = 7 - row
        return chess.square(file, rank)

    # ----------------------------------------------------------------
    # Drawing
    # ----------------------------------------------------------------

    def draw_board(self):
        for square in chess.SQUARES:
            rect = self._square_to_rect(square)
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            is_light = (file + rank) % 2 == 1
            color = LIGHT_SQUARE if is_light else DARK_SQUARE

            if self.last_move and square in (self.last_move.from_square, self.last_move.to_square):
                color = HIGHLIGHT_LAST_MOVE
            if square == self.selected_square:
                color = HIGHLIGHT_SELECTED

            pygame.draw.rect(self.screen, color, rect)

            if square in self.legal_targets:
                center = rect.center
                if self.board.piece_at(square) is not None:
                    pygame.draw.circle(self.screen, HIGHLIGHT_LEGAL, center, SQUARE_SIZE // 2 - 4, width=5)
                else:
                    pygame.draw.circle(self.screen, HIGHLIGHT_LEGAL, center, SQUARE_SIZE // 7)

        # King-in-check highlight
        if self.board.is_check():
            king_square = self.board.king(self.board.turn)
            if king_square is not None:
                rect = self._square_to_rect(king_square)
                pygame.draw.rect(self.screen, HIGHLIGHT_CHECK, rect, width=5)

        # File/rank labels
        for i in range(8):
            file_label = chess.FILE_NAMES[i]
            label = self.font_label.render(file_label, True, PANEL_SUBTEXT)
            self.screen.blit(label, (MARGIN + i * SQUARE_SIZE + 4, MARGIN + BOARD_SIZE - 16))
            rank_label = str(8 - i)
            label = self.font_label.render(rank_label, True, PANEL_SUBTEXT)
            self.screen.blit(label, (MARGIN + 2, MARGIN + i * SQUARE_SIZE + 2))

    def draw_pieces(self):
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None:
                continue
            rect = self._square_to_rect(square)
            symbol = UNICODE_PIECES[piece.symbol()]
            fg = (250, 250, 250) if piece.color == chess.WHITE else (20, 20, 20)
            outline = (20, 20, 20) if piece.color == chess.WHITE else (250, 250, 250)
            # cheap outline effect for readability on any square color
            text_outline = self.font_piece.render(symbol, True, outline)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                r = text_outline.get_rect(center=(rect.centerx + dx, rect.centery + dy))
                self.screen.blit(text_outline, r)
            text = self.font_piece.render(symbol, True, fg)
            r = text.get_rect(center=rect.center)
            self.screen.blit(text, r)

    def draw_eval_bar(self):
        x = MARGIN + BOARD_SIZE + 10
        y = MARGIN
        h = BOARD_SIZE
        w = EVAL_BAR_WIDTH

        # Clamp displayed eval so the bar stays readable even for huge scores.
        score = self.current_eval
        if abs(score) >= CHECKMATE_THRESHOLD:
            frac = 0.98 if score > 0 else 0.02
        else:
            clamped = max(-1000, min(1000, score))
            frac = 0.5 + (clamped / 1000) * 0.5  # 0..1, 0.5 = even

        pygame.draw.rect(self.screen, EVAL_BAR_BLACK, (x, y, w, h))
        white_h = int(h * frac)
        pygame.draw.rect(self.screen, EVAL_BAR_WHITE, (x, y + (h - white_h), w, white_h))
        pygame.draw.rect(self.screen, (90, 90, 90), (x, y, w, h), width=2)

        # Numeric label
        if abs(score) >= CHECKMATE_THRESHOLD:
            label = "M"
        else:
            label = f"{score / 100:+.1f}"
        text = self.font_small.render(label, True, PANEL_TEXT)
        self.screen.blit(text, (x - 4, y + h + 4))

    def draw_side_panel(self):
        x = MARGIN + BOARD_SIZE + EVAL_BAR_WIDTH + 20
        y = MARGIN
        w = SIDE_PANEL_WIDTH

        title = self.font_title.render("Python Chess Engine", True, PANEL_TEXT)
        self.screen.blit(title, (x, y))
        y += 34

        subtitle = self.font_body.render("You: White   Engine: Black", True, PANEL_SUBTEXT)
        self.screen.blit(subtitle, (x, y))
        y += 30

        status = "Engine is thinking..." if self.engine_thinking else "Your move" \
            if self.board.turn == self.human_color and not self.board.is_game_over() else "—"
        if self.game_over_message:
            status = self.game_over_message
        status_surf = self.font_body.render(status, True, (255, 210, 90) if self.engine_thinking else PANEL_TEXT)
        self.screen.blit(status_surf, (x, y))
        y += 26

        if self.engine_stats:
            for line in self.engine_stats.split("\n"):
                stat_surf = self.font_small.render(line, True, PANEL_SUBTEXT)
                self.screen.blit(stat_surf, (x, y))
                y += 18
        y += 10

        pygame.draw.line(self.screen, (70, 70, 75), (x, y), (x + w, y), 1)
        y += 10

        move_hdr = self.font_body.render("Move log", True, PANEL_TEXT)
        self.screen.blit(move_hdr, (x, y))
        y += 24

        # Render move log, two-per-line as "N. e4 e5"
        max_lines = (WINDOW_HEIGHT - y - 60) // 18
        pairs = []
        for i in range(0, len(self.move_log), 2):
            white_move = self.move_log[i]
            black_move = self.move_log[i + 1] if i + 1 < len(self.move_log) else ""
            pairs.append(f"{i // 2 + 1}. {white_move}  {black_move}")
        visible = pairs[-max_lines:] if max_lines > 0 else []
        for line in visible:
            surf = self.font_small.render(line, True, PANEL_SUBTEXT)
            self.screen.blit(surf, (x, y))
            y += 18

        # Footer hint
        hint_y = WINDOW_HEIGHT - MARGIN - 40
        hint1 = self.font_small.render("Click a piece, then a highlighted", True, PANEL_SUBTEXT)
        hint2 = self.font_small.render("square to move. R = restart game.", True, PANEL_SUBTEXT)
        self.screen.blit(hint1, (x, hint_y))
        self.screen.blit(hint2, (x, hint_y + 18))

    def render(self):
        self.screen.fill(BG_COLOR)
        self.draw_board()
        self.draw_pieces()
        self.draw_eval_bar()
        self.draw_side_panel()
        pygame.display.flip()

    # ----------------------------------------------------------------
    # Promotion prompt (simple modal drawn on top of the board)
    # ----------------------------------------------------------------

    def _ask_promotion(self) -> int:
        options = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        labels = ["Q", "R", "B", "N"]
        box_w, box_h = 280, 90
        box_x = MARGIN + BOARD_SIZE // 2 - box_w // 2
        box_y = MARGIN + BOARD_SIZE // 2 - box_h // 2

        while True:
            pygame.draw.rect(self.screen, (50, 50, 55), (box_x, box_y, box_w, box_h), border_radius=8)
            pygame.draw.rect(self.screen, (200, 200, 200), (box_x, box_y, box_w, box_h), width=2, border_radius=8)
            prompt = self.font_body.render("Promote to:", True, PANEL_TEXT)
            self.screen.blit(prompt, (box_x + 15, box_y + 10))

            rects = []
            for i, label in enumerate(labels):
                r = pygame.Rect(box_x + 15 + i * 62, box_y + 40, 50, 40)
                pygame.draw.rect(self.screen, (90, 90, 95), r, border_radius=6)
                text = self.font_title.render(label, True, PANEL_TEXT)
                self.screen.blit(text, text.get_rect(center=r.center))
                rects.append(r)

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    for r, piece_type in zip(rects, options):
                        if r.collidepoint(event.pos):
                            return piece_type
            self.clock.tick(30)

    # ----------------------------------------------------------------
    # Human input handling
    # ----------------------------------------------------------------

    def handle_click(self, pos):
        if self.board.is_game_over() or self.board.turn != self.human_color or self.engine_thinking:
            return

        square = self._pixel_to_square(pos)
        if square is None:
            return

        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece is not None and piece.color == self.human_color:
                self.selected_square = square
                self.legal_targets = [m.to_square for m in self.board.legal_moves
                                       if m.from_square == square]
            return

        # A square is already selected.
        if square == self.selected_square:
            self.selected_square = None
            self.legal_targets = []
            return

        piece = self.board.piece_at(square)
        if piece is not None and piece.color == self.human_color:
            # Switch selection to the newly clicked piece.
            self.selected_square = square
            self.legal_targets = [m.to_square for m in self.board.legal_moves
                                   if m.from_square == square]
            return

        if square in self.legal_targets:
            move = chess.Move(self.selected_square, square)
            moving_piece = self.board.piece_at(self.selected_square)
            if moving_piece.piece_type == chess.PAWN and chess.square_rank(square) in (0, 7):
                promo = self._ask_promotion()
                move = chess.Move(self.selected_square, square, promotion=promo)

            if move in self.board.legal_moves:
                self.make_move(move)

        self.selected_square = None
        self.legal_targets = []

    def make_move(self, move: chess.Move):
        san = self.board.san(move)
        self.board.push(move)
        self.move_log.append(san)
        self.last_move = move
        self.check_game_over()
        if not self.board.is_game_over() and self.board.turn != self.human_color:
            self.start_engine_move()

    # ----------------------------------------------------------------
    # Engine move (runs on a background thread so the UI stays responsive)
    # ----------------------------------------------------------------

    def start_engine_move(self):
        self.engine_thinking = True
        self.engine_stats = "Searching..."
        thread = threading.Thread(target=self._engine_worker, daemon=True)
        thread.start()

    def _engine_worker(self):
        board_copy = self.board.copy()
        move, score, depth, nodes = self.search_engine.choose_move(board_copy)
        self._move_queue.put((move, score, depth, nodes))

    def poll_engine_result(self):
        try:
            move, score, depth, nodes = self._move_queue.get_nowait()
        except queue.Empty:
            return

        self.engine_thinking = False
        self.current_eval = score
        self.engine_stats = f"depth {depth} | {nodes} nodes\neval {score / 100:+.2f}"

        if move is not None and move in self.board.legal_moves:
            san = self.board.san(move)
            self.board.push(move)
            self.move_log.append(san)
            self.last_move = move
        self.check_game_over()

        if not self.board.is_game_over() and self.board.turn == self.human_color:
            # Update the eval bar for the human's turn as well, using a
            # quick static evaluation (fast, no search) for responsiveness.
            self.current_eval = self.evaluator.evaluate(self.board)

    def check_game_over(self):
        if self.board.is_checkmate():
            winner = "White" if self.board.turn == chess.BLACK else "Black"
            self.game_over_message = f"Checkmate! {winner} wins."
        elif self.board.is_stalemate():
            self.game_over_message = "Draw by stalemate."
        elif self.board.is_insufficient_material():
            self.game_over_message = "Draw: insufficient material."
        elif self.board.can_claim_draw():
            self.game_over_message = "Draw available (repetition/50-move)."

    def restart(self):
        self.board = chess.Board()
        self.selected_square = None
        self.legal_targets = []
        self.last_move = None
        self.game_over_message = None
        self.move_log = []
        self.current_eval = 0.0
        self.engine_stats = ""
        self.search_engine.tt.clear()

    # ----------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.restart()

            self.poll_engine_result()
            self.render()
            self.clock.tick(60)

        pygame.quit()
        sys.exit(0)
