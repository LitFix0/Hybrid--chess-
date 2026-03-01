"""
train.py — NNUE Training Script
================================

Reads clean_positions.jsonl, encodes each position with HalfKP,
trains the NNUE model to predict the centipawn eval, and saves weights.

Loss function: MSE on normalized eval (eval_norm in [-1, +1])
The network outputs centipawns, so we normalize before computing loss.

Run:
    python -m trainer.train

Output:
    models/nnue_weights.pt   ← load this in the engine
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import chess

from trainer.feature_encoder import get_active_features, INPUT_SIZE, HALFKP_SIZE
from trainer.nnue_model import NNUE

# ============================
# CONFIG
# ============================
CLEAN_PATH    = os.path.join("datasets", "clean_positions.jsonl")
MODEL_DIR     = "models"
MODEL_PATH    = os.path.join(MODEL_DIR, "nnue_weights.pt")

EPOCHS        = 100
BATCH_SIZE    = 64        # smaller batch = more gradient updates = better generalization
LEARNING_RATE = 5e-4      # lower LR — more careful learning
VAL_SPLIT     = 0.15      # 15% validation — more reliable signal with small dataset
EARLY_STOP    = 15        # stop if val loss doesn't improve for this many epochs

# Normalization divisor — must match dataset_builder.py MAX_EVAL_CP
NORM_DIVISOR  = 3000.0

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================
# DATASET
# ============================

class PositionDataset(Dataset):
    """
    Loads clean_positions.jsonl and encodes each position on-the-fly.

    Each sample:
        x : float tensor of shape (INPUT_SIZE,) — HalfKP binary features
        y : float tensor of shape (1,)           — eval_norm in [-1, +1]
    """

    def __init__(self, path: str):
        self.records = []

        print(f"  Loading dataset from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    # Use eval_norm if available, otherwise normalize raw eval
                    if "eval_norm" in r:
                        norm = float(r["eval_norm"])
                    else:
                        norm = max(-1.0, min(1.0, r["eval"] / NORM_DIVISOR))
                    self.records.append((r["fen"], norm))
                except (json.JSONDecodeError, KeyError):
                    continue

        print(f"  Loaded {len(self.records):,} positions")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fen, eval_norm = self.records[idx]

        board = chess.Board(fen)
        white_feats, black_feats = get_active_features(board)

        # Build dense float tensor
        x = torch.zeros(INPUT_SIZE, dtype=torch.float32)
        for i in white_feats:
            x[i] = 1.0
        for i in black_feats:
            x[HALFKP_SIZE + i] = 1.0

        y = torch.tensor([eval_norm], dtype=torch.float32)
        return x, y


# ============================
# TRAINING LOOP
# ============================

def train():
    print("\n🧠 NNUE Trainer")
    print(f"{'='*50}")

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device    : {device}")
    print(f"  Epochs    : {EPOCHS}")
    print(f"  Batch     : {BATCH_SIZE}")
    print(f"  LR        : {LEARNING_RATE}")
    print(f"{'='*50}\n")

    # ── Dataset ──
    dataset = PositionDataset(CLEAN_PATH)

    if len(dataset) == 0:
        print("❌ Dataset is empty. Run self_play.py and dataset_builder.py first.")
        return

    val_size   = max(1, int(len(dataset) * VAL_SPLIT))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Train samples : {train_size:,}")
    print(f"  Val samples   : {val_size:,}\n")

    # ── Model ──
    model     = NNUE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_loss  = float("inf")
    best_epoch     = 0
    no_improve     = 0   # epochs since last val improvement

    # ── Training ──
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)

        train_loss /= train_size

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item() * len(x)

        val_loss /= val_size
        scheduler.step(val_loss)

        elapsed = time.time() - t0

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            no_improve    = 0
            torch.save(model.state_dict(), MODEL_PATH)
            saved = " ← saved"
        else:
            no_improve += 1
            saved = ""

        print(
            f"  Epoch {epoch:3d}/{EPOCHS} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"{elapsed:.1f}s{saved}"
        )

        # Early stopping
        if no_improve >= EARLY_STOP:
            print(f"\n  Early stop — no improvement for {EARLY_STOP} epochs")
            break

    print(f"\n{'='*50}")
    print(f"✅ Training complete!")
    print(f"   Best val loss : {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"   Model saved   : {MODEL_PATH}")
    print(f"{'='*50}\n")


# ============================
# ENTRY POINT
# ============================

if __name__ == "__main__":
    train()