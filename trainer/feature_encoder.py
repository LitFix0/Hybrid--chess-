"""
feature_encoder.py — HalfKP Feature Encoder
============================================

Converts a chess board position into a binary input vector for NNUE.

HalfKP feature set (same as Stockfish NNUE):
  For each non-king piece on the board, create a feature:
    (king_square, piece_type, piece_square, piece_color)

  This is computed TWICE:
    - Once from White's perspective (using White king square)
    - Once from Black's perspective (using Black king square, mirrored)

  Each half has: 64 king squares × 10 piece buckets × 64 piece squares
                 = 40,960 features

  Total input vector size: 40,960 × 2 = 81,920 binary values

Usage:
    from trainer.feature_encoder import encode_position, INPUT_SIZE
    vector = encode_position(board)   # returns list of 81920 ints (0 or 1)

    # Or get just the active feature indices (sparse, faster for training):
    white_features, black_features = get_active_features(board)
"""

import chess

# ============================
# CONSTANTS
# ============================

# 5 piece types (no king) × 2 colors = 10 buckets
NUM_PIECE_BUCKETS = 10

# 64 king squares × 10 piece buckets × 64 piece squares
HALFKP_SIZE = 64 * NUM_PIECE_BUCKETS * 64   # = 40,960

# Full input: White half + Black half
INPUT_SIZE = HALFKP_SIZE * 2                 # = 81,920

# Piece type → bucket index (0–4 for White, 5–9 for Black)
# King is excluded — it's the "anchor" of HalfKP, not a feature
PIECE_TO_INDEX = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.PAWN,   chess.BLACK): 5,
    (chess.KNIGHT, chess.BLACK): 6,
    (chess.BISHOP, chess.BLACK): 7,
    (chess.ROOK,   chess.BLACK): 8,
    (chess.QUEEN,  chess.BLACK): 9,
}


# ============================
# SQUARE MIRRORING
# ============================

def mirror_square(sq: int) -> int:
    """
    Mirror a square vertically (flip rank).
    Used for Black's perspective — Black's king on e1 should look
    the same as White's king on e8 from their own point of view.

    e.g. a1 (0) → a8 (56), e4 (28) → e5 (36)
    """
    return sq ^ 56


# ============================
# FEATURE INDEX CALCULATOR
# ============================

def halfkp_index(king_sq: int, piece_sq: int, piece_bucket: int) -> int:
    """
    Compute the index of a single HalfKP feature.

    Layout: king_sq * (NUM_PIECE_BUCKETS * 64) + piece_bucket * 64 + piece_sq

    This gives a unique index in [0, HALFKP_SIZE) for every
    (king_square, piece_type+color, piece_square) triple.
    """
    return king_sq * (NUM_PIECE_BUCKETS * 64) + piece_bucket * 64 + piece_sq


# ============================
# ACTIVE FEATURE EXTRACTOR
# ============================

def get_active_features(board: chess.Board):
    """
    Returns two lists of active feature indices:
        white_features — from White's perspective
        black_features — from Black's perspective

    These are SPARSE representations: instead of an 81,920-element
    binary vector, we return only the indices that are 1.
    Typically 20-30 active features per side (one per non-king piece).

    The neural network can use these directly via embedding lookup,
    which is much faster than dense matrix multiplication.
    """
    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)

    if white_king_sq is None or black_king_sq is None:
        # Shouldn't happen in legal positions, but safety check
        return [], []

    # Black's king square is mirrored so Black always "sees" from rank 1
    black_king_sq_mirrored = mirror_square(black_king_sq)

    white_features = []
    black_features = []

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        if piece.piece_type == chess.KING:
            continue  # kings are the anchor, not features

        bucket = PIECE_TO_INDEX.get((piece.piece_type, piece.color))
        if bucket is None:
            continue

        # White perspective: normal squares
        white_features.append(
            halfkp_index(white_king_sq, sq, bucket)
        )

        # Black perspective: mirror the piece square too
        black_features.append(
            halfkp_index(black_king_sq_mirrored, mirror_square(sq), bucket)
        )

    return white_features, black_features


# ============================
# DENSE VECTOR ENCODER
# ============================

def encode_position(board: chess.Board) -> list:
    """
    Convert a board position to a dense binary vector of size INPUT_SIZE (81,920).

    Returns a flat list of 0s and 1s.
    The first HALFKP_SIZE values are White's perspective.
    The second HALFKP_SIZE values are Black's perspective.

    Use this for inspection/debugging. For training, use get_active_features()
    which is much faster (sparse indices instead of full vector).
    """
    vector = [0] * INPUT_SIZE

    white_features, black_features = get_active_features(board)

    for idx in white_features:
        vector[idx] = 1

    for idx in black_features:
        vector[HALFKP_SIZE + idx] = 1

    return vector


# ============================
# VALIDATION / SELF-TEST
# ============================

def validate_encoder():
    """
    Quick sanity checks. Run this to verify the encoder is working.
    Prints results — no exceptions means everything is correct.
    """
    print("Running feature encoder validation...\n")

    # Test 1: starting position
    board = chess.Board()
    wf, bf = get_active_features(board)

    print(f"Starting position:")
    print(f"  White features active : {len(wf)}  (expected ~30)")
    print(f"  Black features active : {len(bf)}  (expected ~30)")
    assert len(wf) == len(bf), "White and Black should have same feature count in start pos"
    assert all(0 <= i < HALFKP_SIZE for i in wf), "White feature index out of range"
    assert all(0 <= i < HALFKP_SIZE for i in bf), "Black feature index out of range"

    # Test 2: dense vector
    vec = encode_position(board)
    assert len(vec) == INPUT_SIZE, f"Vector size wrong: {len(vec)} vs {INPUT_SIZE}"
    assert sum(vec) == len(wf) + len(bf), "Active feature count mismatch in dense vector"
    print(f"\nDense vector:")
    print(f"  Total size    : {len(vec):,}  (expected {INPUT_SIZE:,})")
    print(f"  Active (1s)   : {sum(vec)}  (expected {len(wf) + len(bf)})")

    # Test 3: after a move, features should change
    board.push_san("e4")
    wf2, bf2 = get_active_features(board)
    assert wf2 != wf, "Features should change after a move"
    print(f"\nAfter 1.e4:")
    print(f"  White features active : {len(wf2)}")
    print(f"  Black features active : {len(bf2)}")

    # Test 4: endgame position (few pieces)
    board = chess.Board("8/8/8/4k3/8/4K3/8/8 w - - 0 1")
    wf3, bf3 = get_active_features(board)
    print(f"\nKing vs King (only kings):")
    print(f"  White features active : {len(wf3)}  (expected 0)")
    print(f"  Black features active : {len(bf3)}  (expected 0)")
    assert len(wf3) == 0 and len(bf3) == 0, "Only kings → no HalfKP features"

    # Test 5: no duplicate indices
    board = chess.Board()
    wf, bf = get_active_features(board)
    assert len(wf) == len(set(wf)), "Duplicate White feature indices!"
    assert len(bf) == len(set(bf)), "Duplicate Black feature indices!"
    print(f"\nNo duplicate indices: ✅")

    print(f"\n✅ All validation checks passed!")
    print(f"   INPUT_SIZE = {INPUT_SIZE:,}")
    print(f"   HALFKP_SIZE = {HALFKP_SIZE:,}")


if __name__ == "__main__":
    validate_encoder()