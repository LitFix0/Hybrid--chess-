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

HANGING_DIVISOR = 8


# ==========================================================
# 64-SQUARE SAFE TABLE BUILDER  (prevents your crash forever)
# ==========================================================
def pst(table_8x8):
    """Flatten 8x8 board into 64-square list safely."""
    flat = []
    for row in table_8x8:
        flat.extend(row)
    assert len(flat) == 64
    return flat


# ======================
# PIECE SQUARE TABLES
# ======================

PAWN_TABLE = pst([
[ 0, 0, 0, 0, 0, 0, 0, 0],
[50,50,50,50,50,50,50,50],
[10,10,20,30,30,20,10,10],
[ 5, 5,10,25,25,10, 5, 5],
[ 0, 0, 0,20,20, 0, 0, 0],
[ 5,-5,-10,0, 0,-10,-5, 5],
[ 5,10,10,-20,-20,10,10, 5],
[ 0, 0, 0, 0, 0, 0, 0, 0],
])

KNIGHT_TABLE = pst([
[-50,-40,-30,-30,-30,-30,-40,-50],
[-40,-20,  0,  5,  5,  0,-20,-40],
[-30,  5, 10, 15, 15, 10,  5,-30],
[-30,  0, 15, 20, 20, 15,  0,-30],
[-30,  5, 15, 20, 20, 15,  5,-30],
[-30,  0, 10, 15, 15, 10,  0,-30],
[-40,-20,  0,  0,  0,  0,-20,-40],
[-50,-40,-30,-30,-30,-30,-40,-50],
])

BISHOP_TABLE = pst([
[-20,-10,-10,-10,-10,-10,-10,-20],
[-10,  5,  0,  0,  0,  0,  5,-10],
[-10, 10, 10, 10, 10, 10, 10,-10],
[-10,  0, 10, 10, 10, 10,  0,-10],
[-10,  5,  5, 10, 10,  5,  5,-10],
[-10,  0,  5, 10, 10,  5,  0,-10],
[-10,  0,  0,  0,  0,  0,  0,-10],
[-20,-10,-10,-10,-10,-10,-10,-20],
])

ROOK_TABLE = pst([
[ 0, 0, 5,10,10, 5, 0, 0],
[-5, 0, 0, 0, 0, 0, 0,-5],
[-5, 0, 0, 0, 0, 0, 0,-5],
[-5, 0, 0, 0, 0, 0, 0,-5],
[-5, 0, 0, 0, 0, 0, 0,-5],
[-5, 0, 0, 0, 0, 0, 0,-5],
[ 5,10,10,10,10,10,10, 5],
[ 0, 0, 0, 0, 0, 0, 0, 0],
])

QUEEN_TABLE = pst([
[-20,-10,-10, -5, -5,-10,-10,-20],
[-10,  0,  5,  0,  0,  0,  0,-10],
[-10,  5,  5,  5,  5,  5,  0,-10],
[ -5,  0,  5,  5,  5,  5,  0, -5],
[  0,  0,  5,  5,  5,  5,  0, -5],
[-10,  0,  5,  5,  5,  5,  0,-10],
[-10,  0,  0,  0,  0,  0,  0,-10],
[-20,-10,-10, -5, -5,-10,-10,-20],
])

KING_MID_TABLE = pst([
[-30,-40,-40,-50,-50,-40,-40,-30],
[-30,-40,-40,-50,-50,-40,-40,-30],
[-30,-40,-40,-50,-50,-40,-40,-30],
[-30,-40,-40,-50,-50,-40,-40,-30],
[-20,-30,-30,-40,-40,-30,-30,-20],
[-10,-20,-20,-20,-20,-20,-20,-10],
[ 20, 20,  0,  0,  0,  0, 20, 20],
[ 20, 30, 10,  0,  0, 10, 30, 20],
])

KING_END_TABLE = pst([
[-50,-40,-30,-20,-20,-30,-40,-50],
[-30,-20,-10,  0,  0,-10,-20,-30],
[-30,-10, 20, 30, 30, 20,-10,-30],
[-30,-10, 30, 40, 40, 30,-10,-30],
[-30,-10, 30, 40, 40, 30,-10,-30],
[-30,-10, 20, 30, 30, 20,-10,-30],
[-30,-30,  0,  0,  0,  0,-30,-30],
[-50,-30,-30,-30,-30,-30,-30,-50],
])


PIECE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE
}


# ======================
# EVALUATION FUNCTION
# ======================
def evaluate_board(board: chess.Board):

    if board.is_checkmate():
        return -MATE_SCORE

    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0

    # detect endgame
    total_material = 0
    for pt in PIECE_VALUES:
        if pt != chess.KING:
            total_material += (
                len(board.pieces(pt, chess.WHITE)) +
                len(board.pieces(pt, chess.BLACK))
            ) * PIECE_VALUES[pt]

    endgame = total_material <= 2400

    # material + PST
    for pt in PIECE_VALUES:

        table = KING_END_TABLE if (pt == chess.KING and endgame) else \
                KING_MID_TABLE if pt == chess.KING else PIECE_TABLES[pt]

        for sq in board.pieces(pt, chess.WHITE):
            score += PIECE_VALUES[pt]
            score += table[chess.square_mirror(sq)]

        for sq in board.pieces(pt, chess.BLACK):
            score -= PIECE_VALUES[pt]
            score -= table[sq]

    # hanging piece detection
    for move in board.legal_moves:
        see_value = static_exchange_evaluation(board, move)
        if see_value < 0:
            score -= (-see_value) // HANGING_DIVISOR

    return score if board.turn == chess.WHITE else -score