import chess
import chess.polyglot

BOOK_PATH = "assets/book.bin"  # polyglot book

def get_opening_move(board):
    if board.fullmove_number > 10:
        return None

    try:
        with chess.polyglot.open_reader(BOOK_PATH) as reader:
            entry = reader.weighted_choice(board)
            return entry.move
    except:
        return None
