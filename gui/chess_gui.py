import threading
import pygame
import chess

from engine.iterative import iterative_deepening
from engine.minimax import KILLER_MOVES, HISTORY_HEURISTIC
from engine.transposition import TT   # ✅ REAL transposition table



# ---------- CONSTANTS ----------
BOARD_WIDTH, HEIGHT = 640, 640
SIDEBAR_WIDTH = 220
WIDTH = BOARD_WIDTH + SIDEBAR_WIDTH
SQUARE_SIZE = BOARD_WIDTH // 8
CAPTURE_ICON_SIZE = 28

WHITE = (240, 217, 181)
BROWN = (181, 136, 99)
BLUE = (0, 120, 255)
GREEN = (0, 180, 0)
RED = (200, 0, 0)


# ---------- AI SETTINGS ----------
DIFFICULTY_SETTINGS = {
    "Easy":   {"time": 0.3, "max_depth": 2},
    "Medium": {"time": 1.0, "max_depth": 4},
    "Hard":   {"time": 2.5, "max_depth": 6},
}
CURRENT_DIFFICULTY = "Medium"


# ---------- INIT ----------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess AI")
clock = pygame.time.Clock()

font = pygame.font.SysFont(None, 36)
big_font = pygame.font.SysFont(None, 64)
sidebar_font = pygame.font.SysFont(None, 26)

board = chess.Board()
selected_square = None

capture_history = []


# ---------- SOUND ----------
pygame.mixer.init()

SOUNDS = {
    "move": pygame.mixer.Sound("assets/sounds/move.wav"),
    "capture": pygame.mixer.Sound("assets/sounds/capture.wav"),
    "promote": pygame.mixer.Sound("assets/sounds/promote.wav"),
    "check": pygame.mixer.Sound("assets/sounds/check.wav"),
}

for s in SOUNDS.values():
    s.set_volume(0.6)


def play_move_sound(board, move):
    if board.is_capture(move):
        SOUNDS["capture"].play()
    elif move.promotion is not None:
        SOUNDS["promote"].play()
    else:
        SOUNDS["move"].play()


# ---------- PROMOTION STATE ----------
promotion_from = None
promotion_square = None
promotion_color = None


# ---------- AI THREAD STATE ----------
ai_thinking = False
ai_move_result = None


# ---------- LOAD PIECES ----------
PIECE_IMAGES = {}

def load_piece(symbol, filename):
    PIECE_IMAGES[symbol] = pygame.transform.scale(
        pygame.image.load(f"assets/pieces/{filename}"),
        (SQUARE_SIZE, SQUARE_SIZE),
    )

for s, f in [
    ("P","wp.png"),("R","wr.png"),("N","wn.png"),
    ("B","wb.png"),("Q","wq.png"),("K","wk.png"),
    ("p","bp.png"),("r","br.png"),("n","bn.png"),
    ("b","bb.png"),("q","bq.png"),("k","bk.png")
]:
    load_piece(s, f)

CAPTURE_ICONS = {
    k: pygame.transform.scale(v, (CAPTURE_ICON_SIZE, CAPTURE_ICON_SIZE))
    for k, v in PIECE_IMAGES.items()
}


# ---------- DRAW ----------
def draw_board():
    for r in range(8):
        for c in range(8):
            color = WHITE if (r + c) % 2 == 0 else BROWN
            pygame.draw.rect(
                screen, color,
                (c*SQUARE_SIZE, r*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            )


def draw_pieces():
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            r = 7 - chess.square_rank(square)
            c = chess.square_file(square)
            screen.blit(
                PIECE_IMAGES[piece.symbol()],
                (c*SQUARE_SIZE, r*SQUARE_SIZE)
            )


def highlight_square(square):
    r = 7 - chess.square_rank(square)
    c = chess.square_file(square)
    pygame.draw.rect(
        screen, BLUE,
        (c*SQUARE_SIZE, r*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
        4
    )


def highlight_legal_moves(square):
    for move in board.legal_moves:
        if move.from_square == square:
            r = 7 - chess.square_rank(move.to_square)
            c = chess.square_file(move.to_square)
            center = (
                c*SQUARE_SIZE + SQUARE_SIZE//2,
                r*SQUARE_SIZE + SQUARE_SIZE//2
            )
            if board.piece_at(move.to_square):
                pygame.draw.circle(screen, RED, center, 10)
            else:
                pygame.draw.circle(screen, GREEN, center, 10)


def draw_capture_sidebar():
    sidebar_x = BOARD_WIDTH
    pygame.draw.rect(
        screen, (28, 28, 28),
        (sidebar_x, 0, SIDEBAR_WIDTH, HEIGHT)
    )

    title = sidebar_font.render("Captures", True, (220,220,220))
    screen.blit(title, (sidebar_x + 20, 15))

    y = 55
    for entry in capture_history[-15:]:
        screen.blit(CAPTURE_ICONS[entry["attacker"]], (sidebar_x + 20, y))
        x_text = sidebar_font.render("×", True, (230,230,230))
        screen.blit(x_text, (sidebar_x + 60, y + 2))
        screen.blit(CAPTURE_ICONS[entry["victim"]], (sidebar_x + 90, y))
        y += 32


def draw_status():
    if board.is_checkmate():
        screen.blit(big_font.render("CHECKMATE", True, RED),
                    (BOARD_WIDTH//2 - 140, HEIGHT//2))
    elif board.is_stalemate():
        screen.blit(big_font.render("STALEMATE", True, (80,80,80)),
                    (BOARD_WIDTH//2 - 120, HEIGHT//2))
    elif board.is_check():
        screen.blit(font.render("CHECK!", True, RED), (10, 10))

    diff = font.render(
        f"Difficulty: {CURRENT_DIFFICULTY}", True, (0,0,0)
    )
    screen.blit(diff, (BOARD_WIDTH - 260, 10))

    if ai_thinking:
        screen.blit(
            font.render("AI thinking...", True, (50,50,50)),
            (10, HEIGHT - 40)
        )


# ---------- AI ----------
def ai_think(board_copy):
    global ai_move_result, ai_thinking
    s = DIFFICULTY_SETTINGS[CURRENT_DIFFICULTY]

    # iterative_deepening now returns (move, score, depth) — unpack it
    move, _score, _depth = iterative_deepening(
        board_copy,
        max_time=s["time"],
        max_depth=s["max_depth"]
    )
    ai_move_result = move
    ai_thinking = False


def start_ai_turn():
    global ai_thinking
    if ai_thinking or board.is_game_over():
        return
    ai_thinking = True
    threading.Thread(
        target=ai_think,
        args=(board.copy(),),
        daemon=True
    ).start()


# ---------- RESET ----------
def reset_game():
    global board, selected_square
    global promotion_from, promotion_square, promotion_color
    global ai_thinking, ai_move_result, capture_history

    KILLER_MOVES.clear()
    HISTORY_HEURISTIC.clear()
    TT.clear()

    board = chess.Board()
    selected_square = None
    promotion_from = promotion_square = promotion_color = None
    ai_thinking = False
    ai_move_result = None
    capture_history.clear()


# ---------- MAIN LOOP ----------
running = True
while running:

    draw_board()

    if selected_square is not None:
        highlight_square(selected_square)

    draw_pieces()

    if selected_square is not None:
        highlight_legal_moves(selected_square)

    draw_status()
    draw_capture_sidebar()

    # AI move execution
    if not ai_thinking and ai_move_result:
        if board.is_capture(ai_move_result):
            capture_history.append({
                "attacker": board.piece_at(ai_move_result.from_square).symbol(),
                "victim": board.piece_at(ai_move_result.to_square).symbol()
            })

        play_move_sound(board, ai_move_result)
        board.push(ai_move_result)

        if board.is_check():
            SOUNDS["check"].play()

        ai_move_result = None

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1: CURRENT_DIFFICULTY = "Easy"
            if event.key == pygame.K_2: CURRENT_DIFFICULTY = "Medium"
            if event.key == pygame.K_3: CURRENT_DIFFICULTY = "Hard"
            if event.key == pygame.K_r: reset_game()

        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and board.turn == chess.WHITE
            and not ai_thinking
        ):
            x, y = event.pos

            if x >= BOARD_WIDTH:
                continue

            sq = chess.square(x//SQUARE_SIZE, 7-y//SQUARE_SIZE)

            if selected_square is None:
                if board.piece_at(sq):
                    selected_square = sq
            else:
                move = chess.Move(selected_square, sq)

                if move in board.legal_moves:
                    if board.is_capture(move):
                        capture_history.append({
                            "attacker": board.piece_at(selected_square).symbol(),
                            "victim": board.piece_at(sq).symbol()
                        })

                    play_move_sound(board, move)
                    board.push(move)

                    if board.is_check():
                        SOUNDS["check"].play()

                    selected_square = None
                    start_ai_turn()
                else:
                    selected_square = None

    pygame.display.flip()
    clock.tick(60)

pygame.quit()