import chess
import random

from engine.iterative import iterative_deepening
from engine.training_data import finalize_game

GAMES_TO_PLAY = 50
MAX_MOVES_PER_GAME = 200      # prevents infinite games
THINK_TIME = 0.2              # seconds per move


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

    # If move cap reached → declare draw
    if move_counter >= MAX_MOVES_PER_GAME and not board.is_game_over():
        print("Move limit reached -> Declared Draw")
        result = "1/2-1/2"
    else:
        result = board.result()

    # ⭐ CRITICAL PART (this labels all collected positions)
    finalize_game(result)

    print(f"Game {game_number} finished in {move_counter} moves")
    print("Result:", result)
    print("-" * 50)


def main():

    for i in range(GAMES_TO_PLAY):
        play_game(i + 1)

    print("\nSelf-play finished.")
    print("Dataset saved to: nnue_dataset.txt")


if __name__ == "__main__":
    main()