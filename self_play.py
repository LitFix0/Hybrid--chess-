"""
self_play.py — Position Generator for RL-NNUE Training
=======================================================

What this file does:
  - Plays engine vs engine games
  - Collects positions searched at depth >= MIN_SAVE_DEPTH
  - Backfills the game result onto every position after the game ends
  - Filters out: checkmates, repetitions, book moves (depth 0)
  - Writes one JSON record per line to datasets/raw_positions.jsonl

Output format (one JSON object per line):
  {
    "fen":        "rnbqkbnr/...",   # position BEFORE the move that was played
    "eval":       +42,              # centipawns from White's perspective
    "depth":      5,                # search depth that produced this eval
    "game_phase": "middlegame",     # "opening" | "middlegame" | "endgame"
    "move_number": 14,             # fullmove number
    "result":     "1/2-1/2"        # filled in after game ends
  }

Usage:
  python self_play.py
"""

import os
import json
import random
import chess

from engine.iterative import iterative_deepening
from engine.evaluation import PIECE_VALUES  # used for game_phase detection

# ============================
# CONFIG
# ============================
GAMES_TO_PLAY   = 100
MAX_MOVES       = 100      # hard cap — games rarely need more than this
THINK_TIME      = 0.5      # seconds per move  (higher = better evals = better data)
MAX_DEPTH       = 6

MIN_SAVE_DEPTH  = 4        # ⚠️ CRITICAL: never save shallow evals, they are noise
RANDOM_BOOK_PLY = 8        # first 8 half-moves random for opening diversity

DATASET_PATH    = os.path.join("datasets", "raw_positions.jsonl")
os.makedirs("datasets", exist_ok=True)

# Endgame material threshold (same logic as evaluation.py)
ENDGAME_THRESHOLD = 2400


# ============================
# GAME PHASE DETECTOR
# ============================
def get_game_phase(board: chess.Board) -> str:
    """
    Returns 'opening', 'middlegame', or 'endgame' based on material on board.
    Matches the same threshold used in evaluation.py so data is consistent.
    """
    total_material = 0
    for pt, value in PIECE_VALUES.items():
        if pt != chess.KING:
            total_material += (
                len(board.pieces(pt, chess.WHITE)) +
                len(board.pieces(pt, chess.BLACK))
            ) * value

    if board.fullmove_number <= 10:
        return "opening"
    elif total_material <= ENDGAME_THRESHOLD:
        return "endgame"
    else:
        return "middlegame"


# ============================
# POSITION FILTER
# ============================
def should_save(board: chess.Board, depth: int) -> bool:
    """
    Returns True only for positions worth saving.
    Bad positions corrupt NNUE training.
    """
    # Reject shallow searches — too noisy
    if depth < MIN_SAVE_DEPTH:
        return False

    # Reject checkmate / stalemate positions
    if board.is_checkmate() or board.is_stalemate():
        return False

    # Reject positions that are repetitions (draw by repetition)
    if board.is_repetition(2):
        return False

    # Reject insufficient material positions
    if board.is_insufficient_material():
        return False

    return True


# ============================
# EVAL NORMALIZER
# ============================
def to_white_perspective(eval_score: int, board: chess.Board) -> int:
    """
    iterative_deepening returns eval from the perspective of the side to move.
    We always store eval from White's perspective for consistency.
    """
    if board.turn == chess.WHITE:
        return eval_score
    else:
        return -eval_score


# ============================
# GAME WRITER
# ============================
def write_positions(positions: list, result: str):
    """
    Backfills the game result onto every position, then appends to dataset.
    Called once per game after the game ends.
    """
    with open(DATASET_PATH, "a") as f:
        for pos in positions:
            pos["result"] = result
            f.write(json.dumps(pos) + "\n")


# ============================
# SINGLE GAME
# ============================
def play_game(game_number: int):
    print(f"\n{'='*50}")
    print(f"  Game {game_number} / {GAMES_TO_PLAY}")
    print(f"{'='*50}")

    # Seed random with game number so each game gets a different opening
    # This is the clean fix for White bias — different openings = balanced results
    random.seed(game_number)

    board = chess.Board()
    move_count = 0

    # Buffer: positions collected this game (result added at the end)
    game_positions = []

    while not board.is_game_over() and move_count < MAX_MOVES:

        # ----- OPENING RANDOMIZATION -----
        # Random moves for the first few plies so we don't always see the same
        # opening positions. These are NOT saved (depth = 0).
        if board.ply() < RANDOM_BOOK_PLY:
            legal_moves = list(board.legal_moves)
            move = random.choice(legal_moves)
            board.push(move)
            move_count += 1
            continue

        # ----- ENGINE SEARCH -----
        # Capture the FEN *before* the move is made — that's the position we evaluate
        fen_before = board.fen()

        move, eval_score, depth_reached = iterative_deepening(
            board,
            max_time=THINK_TIME,
            max_depth=MAX_DEPTH
        )

        # Safety: reject illegal moves
        if move is None or move not in board.legal_moves:
            print(f"  ⚠️  Illegal move detected at move {move_count}: {move}")
            print(f"  FEN: {fen_before}")
            break

        # ----- COLLECT POSITION -----
        if should_save(board, depth_reached):
            white_eval = to_white_perspective(eval_score, board)
            record = {
                "fen":         fen_before,
                "eval":        white_eval,
                "depth":       depth_reached,
                "game_phase":  get_game_phase(board),
                "move_number": board.fullmove_number,
                "result":      None   # ← filled in after game ends
            }
            game_positions.append(record)

        # ----- MAKE MOVE -----
        board.push(move)
        move_count += 1

        # Progress indicator every 20 moves
        if move_count % 20 == 0:
            print(f"  Move {move_count} | Positions collected: {len(game_positions)}")

    # ----- DETERMINE RESULT -----
    if move_count >= MAX_MOVES and not board.is_game_over():
        result = "1/2-1/2"
        print(f"  Move cap reached → Draw")
    else:
        result = board.result()

    # ----- WRITE POSITIONS WITH RESULT -----
    write_positions(game_positions, result)

    print(f"  Result   : {result}")
    print(f"  Moves    : {move_count}")
    print(f"  Positions saved: {len(game_positions)}")

    return len(game_positions)


# ============================
# MAIN
# ============================
def main():
    print("\n🔧 RL-NNUE Position Generator")
    print(f"   Games     : {GAMES_TO_PLAY}")
    print(f"   Think time: {THINK_TIME}s/move")
    print(f"   Min depth : {MIN_SAVE_DEPTH}")
    print(f"   Output    : {DATASET_PATH}\n")

    total_positions = 0

    for i in range(GAMES_TO_PLAY):
        saved = play_game(i + 1)
        total_positions += saved

    print(f"\n{'='*50}")
    print(f"✅ Self-play complete!")
    print(f"   Total positions saved : {total_positions}")
    print(f"   Dataset location      : {DATASET_PATH}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()