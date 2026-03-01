import os
import chess
import random

from engine.iterative import iterative_deepening
from trainer.dataset_builder import finalize_game

GAMES_TO_PLAY = 50
MAX_MOVES_PER_GAME = 200      # prevents infinite games
THINK_TIME = 0.2              # seconds per move

# ✅ Fix: point to the correct dataset file
DATASET_PATH = os.path.join("datasets", "raw_positions.txt")
os.makedirs("datasets", exist_ok=True)


def log_position(fen):
    """Append a FEN position to raw_positions.txt"""
    with open(DATASET_PATH, "a") as f:
        f.write(f"{fen}\n")


def play_game(game_number):
    print(f"\nStarting Game {game_number}")

    board = chess.Board()
    move_counter = 0

    while not board.is_game_over() and move_counter < MAX_MOVES_PER_GAME:

        # Opening randomization (VERY IMPORTANT for RL dataset diversity)
        if board.fullmove_number <= 8:
            legal_moves = list(board.legal_moves)
            move = random.choice(legal_moves)
        else:
            move = iterative_deepening(board, max_time=THINK_TIME, max_depth=6)

        # Safety check
        if move is None or move not in board.legal_moves:
            print("Illegal move detected:", move)
            print("Board FEN:", board.fen())
            break

        board.push(move)
        move_counter += 1

        # ✅ Log every move (just FEN for now)
        log_position(board.fen())

    # If move cap reached → declare draw
    if move_counter >= MAX_MOVES_PER_GAME and not board.is_game_over():
        print("Move limit reached -> Declared Draw")
        result = "1/2-1/2"
    else:
        result = board.result()

    # Keep calling finalize_game if needed
    finalize_game(result)

    print(f"Game {game_number} finished in {move_counter} moves")
    print("Result:", result)
    print("-" * 50)


def main():
    for i in range(GAMES_TO_PLAY):
        play_game(i + 1)

    print("\nSelf-play finished.")
    print(f"Dataset saved to: {DATASET_PATH}")


if __name__ == "__main__":
    main()