import time
import chess

from engine.opening_book import get_opening_move
from engine.minimax import minimax_alpha_beta
from engine.search_control import should_stop, start_search


# OPTIONAL: Endgame tablebase (safe import)
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
    from engine.transposition import TT
    TT.clear()
    """
    Iterative Deepening with:
    - Hard time control
    - Real aspiration windows (progressive widening)
    - Opening book
    - Syzygy tablebase
    - Fail-safe fallback move
    - Compatible with TT / Killer / LMR / Null Move / Threat extensions
    """

    # ===============================
    # START GLOBAL SEARCH TIMER
    # ===============================
    start_search(max_time)

    start_time = time.time()
    time_limit = start_time + max_time

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
    # 3️⃣ ITERATIVE DEEPENING
    # ==================================================
    for depth in range(1, max_depth + 1):

        if should_stop() or time.time() >= time_limit:
            break

        value = None
        move = None

        # -----------------------------
        # Depth 1–2 → Full window
        # -----------------------------
        if depth <= 2:
            value, move = minimax_alpha_beta(
                board,
                depth,
                -float("inf"),
                float("inf"),
                True,   # ⭐ FIX: always True (Negamax root)
                time_limit=time_limit
            )

        # -----------------------------
        # Depth ≥3 → REAL Aspiration
        # -----------------------------
        else:

            # MATE SCORES should NEVER use aspiration
            if abs(prev_score) > 90000:
                alpha = -float("inf")
                beta = float("inf")
            else:
                window = INITIAL_WINDOW
                alpha = prev_score - window
                beta = prev_score + window

            researches = 0

            while True:

                if should_stop() or time.time() >= time_limit:
                    return best_move or last_completed_move

                value, move = minimax_alpha_beta(
                    board,
                    depth,
                    alpha,
                    beta,
                    True,   # ⭐ FIX
                    time_limit=time_limit
                )

                # ---------------- FAIL LOW ----------------
                if value <= alpha:
                    researches += 1
                    window *= WINDOW_GROWTH
                    alpha = prev_score - window

                # ---------------- FAIL HIGH ----------------
                elif value >= beta:
                    researches += 1
                    window *= WINDOW_GROWTH
                    beta = prev_score + window

                else:
                    break

                # window exploded → fallback full search
                if window > MAX_WINDOW or researches >= MAX_RESEARCHES:
                    value, move = minimax_alpha_beta(
                        board,
                        depth,
                        -float("inf"),
                        float("inf"),
                        True,   # ⭐ FIX
                        time_limit=time_limit
                    )
                    break

        # -----------------------------
        # Accept completed depth
        # -----------------------------
        if should_stop():
            break

        if move is not None:
            last_completed_move = move
            best_move = move

            # evaluation shock protection
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