import chess
from engine.minimax import get_best_move

board = chess.Board()
DEPTH = 3

print("You are playing as WHITE")

while not board.is_game_over():
    print(board)

    if board.turn == chess.WHITE:
        move = input("Your move (e2e4): ")
        try:
            board.push_uci(move)
        except:
            print("Invalid move")
            continue
    else:
        ai_move = get_best_move(board, DEPTH)
        print("AI plays:", ai_move)
        board.push(ai_move)

print("Game over:", board.result())
