import chess
from engine.see import static_exchange_evaluation

# ======================
# MATE SCORES
# ======================

MATE_SCORE = 100000
MATE_THRESHOLD = 90000

# ======================
# PIECE VALUES
# ======================

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

# ======================
# POSITIONAL CONSTANTS
# ======================

TEMPO_BONUS = 10
HANGING_DIVISOR = 8

# ======================
# PIECE SQUARE TABLES
# ======================

# (Your PST tables unchanged — keep them exactly as you wrote)

PAWN_TABLE = [...]
KNIGHT_TABLE = [...]
BISHOP_TABLE = [...]
ROOK_TABLE = [...]
QUEEN_TABLE = [...]
KING_MID_TABLE = [...]
KING_END_TABLE = [...]

PIECE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE
}

# ======================
# EVALUATION
# ======================

def evaluate_board(board: chess.Board):

    # --- TERMINAL POSITIONS ---
    if board.is_checkmate():
        return -MATE_SCORE

    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0

    # --- Detect Endgame ---
    total_material = 0
    for pt in PIECE_VALUES:
        if pt != chess.KING:
            total_material += (
                len(board.pieces(pt, chess.WHITE)) +
                len(board.pieces(pt, chess.BLACK))
            ) * PIECE_VALUES[pt]

    endgame = total_material <= 2400

    # --- Material + PST ---
    for pt in PIECE_VALUES:

        for sq in board.pieces(pt, chess.WHITE):
            score += PIECE_VALUES[pt]
            table = KING_END_TABLE if (pt == chess.KING and endgame) else \
                    KING_MID_TABLE if pt == chess.KING else PIECE_TABLES[pt]
            score += table[sq]

        for sq in board.pieces(pt, chess.BLACK):
            mirrored = chess.square_mirror(sq)
            score -= PIECE_VALUES[pt]
            table = KING_END_TABLE if (pt == chess.KING and endgame) else \
                    KING_MID_TABLE if pt == chess.KING else PIECE_TABLES[pt]
            score -= table[mirrored]

    # ======================
    # FIXED HANGING DETECTION
    # ======================

    # We evaluate captures for BOTH sides safely

    for move in board.generate_legal_captures():

        piece = board.piece_at(move.from_square)
        if piece is None:
            continue

        see_value = static_exchange_evaluation(board, move)

        if see_value < 0:
            penalty = (-see_value) // HANGING_DIVISOR

            # If WHITE piece is hanging → subtract
            if piece.color == chess.WHITE:
                score -= penalty
            else:
                score += penalty

    # --- TEMPO ---
    score += TEMPO_BONUS

    # --- Mate distance correction ---
    if score > MATE_THRESHOLD:
        score -= board.fullmove_number
    elif score < -MATE_THRESHOLD:
        score += board.fullmove_number

    # Always return from white perspective
    return score