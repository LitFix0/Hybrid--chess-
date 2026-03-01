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
    Iterative Deepening driver for Negamax search.

    Returns:
        (move, score, depth_reached)  ← CHANGED: now returns a tuple
        score is in centipawns from the perspective of the side to move.
        depth_reached is the last fully completed search depth.

    Callers that only want the move can do:
        move, score, depth = iterative_deepening(board, ...)
    or unpack just the first element:
        move, *_ = iterative_deepening(board, ...)
    """

    start_search(max_time)

    best_move = None
    best_score = 0
    last_completed_move = None
    last_completed_score = 0
    last_completed_depth = 0
    prev_score = 0

    # ==================================================
    # 1. OPENING BOOK
    # ==================================================
    book_move = get_opening_move(board)
    if book_move:
        # Book moves have no search eval — return 0 as score, depth 0
        return book_move, 0, 0

    # ==================================================
    # 2. ENDGAME TABLEBASE (Syzygy)
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
                                return move, 0, 0
        except Exception:
            pass

    # ==================================================
    # 3. ITERATIVE DEEPENING LOOP
    # ==================================================
    for depth in range(1, max_depth + 1):

        if should_stop():
            break

        value = None
        move = None

        # Depth 1–2 → Full Window
        if depth <= 2:
            value, move = search(
                board,
                depth,
                -float("inf"),
                float("inf")
            )

        # Depth ≥ 3 → Aspiration Windows
        else:
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
                    # Return best completed depth result
                    final_move = best_move or last_completed_move
                    if final_move is None:
                        legal = list(board.legal_moves)
                        final_move = legal[0] if legal else None
                    return final_move, last_completed_score, last_completed_depth

                value, move = search(board, depth, alpha, beta)

                if value <= alpha:
                    researches += 1
                    window *= WINDOW_GROWTH
                    alpha = prev_score - window

                elif value >= beta:
                    researches += 1
                    window *= WINDOW_GROWTH
                    beta = prev_score + window

                else:
                    break

                if window > MAX_WINDOW or researches >= MAX_RESEARCHES:
                    value, move = search(board, depth, -float("inf"), float("inf"))
                    break

        if should_stop():
            break

        if move is not None:
            last_completed_move = move
            last_completed_score = value
            last_completed_depth = depth
            best_move = move
            best_score = value

            if abs(value - prev_score) <= MAX_SCORE_JUMP:
                prev_score = value
            else:
                prev_score = (prev_score + value) // 2

    # ==================================================
    # FINAL SAFETY RETURN
    # ==================================================
    final_move = best_move or last_completed_move
    if final_move is None:
        legal = list(board.legal_moves)
        final_move = legal[0] if legal else None

    return final_move, last_completed_score, last_completed_depth