import time
import chess

from engine.opening_book import get_opening_move
from engine.minimax import search
from engine.search_control import should_stop, start_search

# OPTIONAL: Endgame tablebase
try:
    import chess.syzygy
    TABLEBASE_AVAILABLE = True
except ImportError:
    TABLEBASE_AVAILABLE = False

# ============================
# ASPIRATION CONSTANTS
# ============================
INITIAL_WINDOW = 80
WINDOW_GROWTH = 2
MAX_WINDOW = 20000
MAX_RESEARCHES = 6
MAX_SCORE_JUMP = 400


def iterative_deepening(board, max_time=1.0, max_depth=6):

    """
    Iterative Deepening driver for Negamax search
    Uses:
    - Global time manager (search_control)
    - Aspiration windows
    - Opening book
    - Syzygy tablebase
    """

    # Start global timer (THIS controls stopping)
    start_search(max_time)

    best_move = None
    last_completed_move = None
    prev_score = 0

    # ==================================================
    # 1️⃣ OPENING BOOK
    # ==================================================
    book_move = get_opening_move(board)
    if book_move:
        return book_move

    # ==================================================
    # 2️⃣ ENDGAME TABLEBASE (Syzygy)
    # ==================================================
    if TABLEBASE_AVAILABLE:
        try:
            if len(board.piece_map()) <= 7:
                with chess.syzygy.open_tablebase("syzygy") as tablebase:
                    dtz = tablebase.probe_dtz(board)
                    if dtz is not None:
                        for move in board.legal_moves:
                            board.push(move)
                            new_dtz = tablebase.probe_dtz(board)
                            board.pop()
                            if new_dtz is not None and new_dtz < dtz:
                                return move
        except Exception:
            pass

    # ==================================================
    # 3️⃣ ITERATIVE DEEPENING LOOP
    # ==================================================
    for depth in range(1, max_depth + 1):

        if should_stop():
            break

        value = None
        move = None

        # -----------------------------
        # Depth 1–2 → Full Window
        # -----------------------------
        if depth <= 2:
            value, move = search(
                board,
                depth,
                -float("inf"),
                float("inf")
            )

        # -----------------------------
        # Depth ≥3 → Aspiration Search
        # -----------------------------
        else:

            # If near mate → disable aspiration
            if abs(prev_score) > 90000:
                alpha = -float("inf")
                beta = float("inf")
            else:
                window = INITIAL_WINDOW
                alpha = prev_score - window
                beta = prev_score + window

            researches = 0

            while True:

                if should_stop():
                    return best_move or last_completed_move

                value, move = search(
                    board,
                    depth,
                    alpha,
                    beta
                )

                # FAIL LOW
                if value <= alpha:
                    researches += 1
                    window *= WINDOW_GROWTH
                    alpha = prev_score - window

                # FAIL HIGH
                elif value >= beta:
                    researches += 1
                    window *= WINDOW_GROWTH
                    beta = prev_score + window

                else:
                    break

                # Window exploded → fallback full search
                if window > MAX_WINDOW or researches >= MAX_RESEARCHES:
                    value, move = search(
                        board,
                        depth,
                        -float("inf"),
                        float("inf")
                    )
                    break

        # Accept completed depth
        if should_stop():
            break

        if move is not None:
            last_completed_move = move
            best_move = move

            # Stabilize evaluation jumps
            if abs(value - prev_score) <= MAX_SCORE_JUMP:
                prev_score = value
            else:
                prev_score = (prev_score + value) // 2

    # ==================================================
    # FINAL SAFETY RETURN
    # ==================================================
    if best_move:
        return best_move

    if last_completed_move:
        return last_completed_move

    legal = list(board.legal_moves)
    if legal:
        return legal[0]

    return None