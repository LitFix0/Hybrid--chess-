"""
nnue_eval.py — NNUE Evaluator for Engine Integration
=====================================================

Loads trained NNUE weights and provides evaluate_nnue(board) which
returns a centipawn score from the perspective of the side to move.

This is a drop-in replacement for evaluate_board() in evaluation.py.

Usage (automatic — called by evaluation.py):
    from engine.nnue_eval import evaluate_nnue, nnue_available
    if nnue_available():
        score = evaluate_nnue(board)
    else:
        score = evaluate_board(board)  # fallback
"""

import os
import torch
import chess

from trainer.feature_encoder import get_active_features, INPUT_SIZE, HALFKP_SIZE
from trainer.nnue_model import NNUE

# ============================
# CONFIG
# ============================
MODEL_PATH = os.path.join("models", "nnue_weights.pt")

# Denormalization: network outputs [-1, +1], engine expects centipawns
# Must match NORM_DIVISOR in train.py and dataset_builder.py
NORM_DIVISOR = 3000.0


# ============================
# MODEL LOADER (singleton)
# ============================
# Load once at import time — not on every evaluate call
_model = None
_device = None
_available = False


def _load_model():
    """Load weights once. Called automatically on first use."""
    global _model, _device, _available

    if not os.path.exists(MODEL_PATH):
        print(f"[nnue_eval] Weights not found at {MODEL_PATH} — using classical eval")
        _available = False
        return

    try:
        _device = torch.device("cpu")  # CPU is fine — inference is fast
        _model  = NNUE().to(_device)
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device, weights_only=True))
        _model.eval()  # disable dropout for inference
        _available = True
        print(f"[nnue_eval] ✅ Loaded NNUE weights from {MODEL_PATH}")
    except Exception as e:
        print(f"[nnue_eval] Failed to load weights: {e} — using classical eval")
        _available = False


# Load on import
_load_model()


def nnue_available() -> bool:
    """Returns True if NNUE weights loaded successfully."""
    return _available


# ============================
# INFERENCE
# ============================

def evaluate_nnue(board: chess.Board) -> int:
    """
    Evaluate a position using the NNUE network.

    Returns centipawns from the perspective of the side to move
    (positive = good for side to move, negative = bad).

    This matches the same convention as evaluate_board() in evaluation.py
    so it's a true drop-in replacement.
    """
    if not _available:
        raise RuntimeError("NNUE weights not loaded — check nnue_available() first")

    # Encode position
    white_feats, black_feats = get_active_features(board)

    x = torch.zeros(INPUT_SIZE, dtype=torch.float32)
    for i in white_feats:
        x[i] = 1.0
    for i in black_feats:
        x[HALFKP_SIZE + i] = 1.0

    # Forward pass
    with torch.no_grad():
        raw = _model(x.unsqueeze(0)).item()   # shape (1,1) → scalar

    # Denormalize: [-1, +1] → centipawns from White's perspective
    white_cp = int(raw * NORM_DIVISOR)

    # Convert to side-to-move perspective (same as evaluate_board)
    return white_cp if board.turn == chess.WHITE else -white_cp