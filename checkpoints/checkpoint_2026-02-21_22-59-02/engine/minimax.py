import chess
from collections import defaultdict

from engine.evaluation import evaluate_board
from engine.see import static_exchange_evaluation
from engine.search_control import should_stop
from engine.transposition import TT, EXACT, LOWERBOUND, UPPERBOUND
from engine.training_data import save_position

MATE_SCORE = 100000
DELTA_MARGIN = 200

NULL_MOVE_REDUCTION = 2
NULL_MOVE_MARGIN = 150

LMR_REDUCTION = 1
LMR_MOVE_THRESHOLD = 5

KILLER_MOVES = defaultdict(lambda: [None, None])
HISTORY_HEURISTIC = defaultdict(int)


# ------------------------------
# Tactical move detection
# ------------------------------
def is_tactical_move(board, move):
    return board.is_capture(move) or board.gives_check(move) or move.promotion


# ------------------------------
# Move ordering
# ------------------------------
def ordered_moves(board, depth):

    moves = list(board.legal_moves)

    key = board._transposition_key
    entry = TT.get(key)
    tt_move = entry[3] if entry else None

    def score(move):
        piece = board.piece_at(move.from_square)
        history_key = (piece.piece_type, move.to_square) if piece else None

        s = 0

        if tt_move and move == tt_move:
            return 200000

        if board.is_capture(move):
            s += 10000 + static_exchange_evaluation(board, move)

        if board.gives_check(move):
            s += 9000

        if move in KILLER_MOVES[depth]:
            s += 8000

        if history_key:
            s += HISTORY_HEURISTIC[history_key]

        return s

    moves.sort(key=score, reverse=True)
    return moves


# ------------------------------
# Quiescence
# ------------------------------
def quiescence(board, alpha, beta):

    if should_stop():
        return evaluate_board(board)

    if board.is_checkmate():
        return -MATE_SCORE

    stand_pat = evaluate_board(board)

    # Save quiet positions only
    if not board.is_check():
        good_capture = False
        for m in board.legal_moves:
            if board.is_capture(m) and static_exchange_evaluation(board, m) >= 0:
                good_capture = True
                break
        if not good_capture:
            save_position(board, stand_pat)

    if stand_pat >= beta:
        return beta

    if stand_pat > alpha:
        alpha = stand_pat

    captures = [
        m for m in board.legal_moves
        if board.is_capture(m) and static_exchange_evaluation(board, m) >= 0
    ]

    captures.sort(key=lambda m: static_exchange_evaluation(board, m), reverse=True)

    for move in captures:
        board.push(move)
        score = -quiescence(board, -beta, -alpha)
        board.pop()

        if score >= beta:
            return beta

        if score > alpha:
            alpha = score

    return alpha


# ------------------------------
# NEGAMAX SEARCH
# ------------------------------
def negamax(board, depth, alpha, beta, ply=0):

    if should_stop():
        return evaluate_board(board), None

    if board.is_checkmate():
        return -MATE_SCORE + ply, None

    if board.is_stalemate() or board.is_insufficient_material():
        return 0, None

    if depth <= 0:
        return quiescence(board, alpha, beta), None

    alpha_original = alpha
    key = board._transposition_key

    # TT probe
    entry = TT.get(key)
    if entry:
        stored_depth, stored_value, stored_flag, stored_move = entry
        if stored_depth >= depth:
            if stored_flag == EXACT:
                return stored_value, stored_move
            elif stored_flag == LOWERBOUND:
                alpha = max(alpha, stored_value)
            elif stored_flag == UPPERBOUND:
                beta = min(beta, stored_value)
            if alpha >= beta:
                return stored_value, stored_move

    # Null move pruning
    if depth >= 3 and not board.is_check():
        static_eval = evaluate_board(board)
        if static_eval >= beta + NULL_MOVE_MARGIN:
            board.push(chess.Move.null())
            score, _ = negamax(board, depth - 1 - NULL_MOVE_REDUCTION, -beta, -beta + 1, ply + 1)
            board.pop()
            if -score >= beta:
                return beta, None

    best_move = None
    value = -999999

    for idx, move in enumerate(ordered_moves(board, depth)):

        board.push(move)

        reduction = 0
        if depth >= 3 and idx >= LMR_MOVE_THRESHOLD and not is_tactical_move(board, move):
            reduction = LMR_REDUCTION

        score, _ = negamax(board, depth - 1 - reduction, -beta, -alpha, ply + 1)
        score = -score

        board.pop()

        if score > value:
            value = score
            best_move = move

        alpha = max(alpha, score)

        if alpha >= beta:
            if not board.is_capture(move):
                killers = KILLER_MOVES[depth]
                killers[1] = killers[0]
                killers[0] = move

                piece = board.piece_at(move.from_square)
                if piece:
                    HISTORY_HEURISTIC[(piece.piece_type, move.to_square)] += depth * depth
            break

    # Store TT
    flag = EXACT
    if value <= alpha_original:
        flag = UPPERBOUND
    elif value >= beta:
        flag = LOWERBOUND

    TT[key] = (depth, value, flag, best_move)

    return value, best_move