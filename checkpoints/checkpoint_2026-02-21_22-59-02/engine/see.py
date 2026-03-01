import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


def static_exchange_evaluation(board, move):
    """
    Safe SEE implementation.
    Returns material gain/loss of a capture sequence.
    """

    # If not a capture, no gain
    if not board.is_capture(move):
        return 0

    # Handle EN PASSANT
    if board.is_en_passant(move):
        captured_square = chess.square(
            chess.square_file(move.to_square),
            chess.square_rank(move.from_square)
        )
        captured_piece = board.piece_at(captured_square)
    else:
        captured_piece = board.piece_at(move.to_square)

    # Safety check
    if captured_piece is None:
        return 0

    gain = PIECE_VALUES[captured_piece.piece_type]

    # simulate capture
    board.push(move)

    # opponent recaptures?
    attackers = board.attackers(not board.turn, move.to_square)

    if attackers:
        smallest = min(
            attackers,
            key=lambda sq: PIECE_VALUES[board.piece_at(sq).piece_type]
        )

        recapture = chess.Move(smallest, move.to_square)

        gain -= static_exchange_evaluation(board, recapture)

    board.pop()

    return gain
