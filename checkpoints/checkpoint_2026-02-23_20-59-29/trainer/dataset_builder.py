import os
import chess
import threading

# ✅ Updated dataset path
DATASET_PATH = os.path.join("datasets", "raw_positions.txt")
os.makedirs("datasets", exist_ok=True)

LOCK = threading.Lock()

# Positions collected DURING one self-play game
CURRENT_GAME_POSITIONS = []

# Avoid duplicates across all games
SEEN_POSITIONS = set()

# Filters (important for NNUE quality)
MIN_PIECES = 6
MAX_PIECES = 32


def save_position(board: chess.Board, score: int):
    """
    Collect candidate positions during search.
    Labeling is done AFTER game finishes.
    """

    # Skip early opening theory
    if board.fullmove_number < 6:
        return

    piece_count = len(board.piece_map())

    # Avoid tablebase or trivial positions
    if piece_count < MIN_PIECES or piece_count > MAX_PIECES:
        return

    fen = board.fen()

    # Avoid duplicates
    if fen in SEEN_POSITIONS:
        return

    SEEN_POSITIONS.add(fen)

    # IMPORTANT: store side to move
    side_to_move = board.turn  # True = White, False = Black

    CURRENT_GAME_POSITIONS.append((fen, side_to_move))


def finalize_game(result: str):
    """
    Called after each self-play game ends.
    Assign correct perspective labels and save to the NEW dataset path.
    """

    # Determine winner
    if result == "1-0":
        winner = chess.WHITE
    elif result == "0-1":
        winner = chess.BLACK
    else:
        winner = None  # draw

    with LOCK:
        with open(DATASET_PATH, "a", encoding="utf-8") as f:

            for fen, stm in CURRENT_GAME_POSITIONS:

                # DRAW
                if winner is None:
                    label = 0.5

                # If side to move is the winner → good position
                elif stm == winner:
                    label = 1.0

                # If side to move is the loser → bad position
                else:
                    label = 0.0

                f.write(f"{fen}|{label}\n")

    # Clear positions for next game
    CURRENT_GAME_POSITIONS.clear()