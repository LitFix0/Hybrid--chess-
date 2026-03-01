"""
nnue_model.py — NNUE Neural Network (PyTorch)
==============================================

Architecture:
    HalfKP input (81,920 sparse binary features)
        ↓
    Linear(81920 → 256) + ClippedReLU
        ↓
    Linear(256 → 32)    + ClippedReLU
        ↓
    Linear(32 → 1)      → eval in centipawns

ClippedReLU is used instead of standard ReLU because it bounds
activations to [0, 1], which improves quantization later if you
ever want to run this in C++.

The network is intentionally small — it runs inside alpha-beta search
millions of times per second, so speed matters more than capacity.
"""

import torch
import torch.nn as nn

from trainer.feature_encoder import INPUT_SIZE  # 81,920


# ============================
# CLIPPED RELU
# ============================

class ClippedReLU(nn.Module):
    """Clamps activations to [0, 1]. Bounds gradients, aids quantization."""
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


# ============================
# NNUE MODEL
# ============================

class NNUE(nn.Module):
    """
    Small NNUE network.

    Input  : sparse HalfKP feature vector (size 81,920)
    Output : single float — predicted eval in centipawns

    Forward pass takes a dense float tensor of shape (batch, INPUT_SIZE).
    During training this comes from the dataset loader.
    During inference (in engine) this comes from feature_encoder.encode_position().
    """

    def __init__(self):
        super().__init__()

        # Smaller network — prevents overfitting on small datasets
        # 81920 → 64 → 32 → 1  (~5.2M params vs 21M before)
        self.fc1     = nn.Linear(INPUT_SIZE, 64)
        self.cr1     = ClippedReLU()
        self.drop1   = nn.Dropout(p=0.3)   # randomly zero 30% of neurons during training

        self.fc2     = nn.Linear(64, 32)
        self.cr2     = ClippedReLU()
        self.drop2   = nn.Dropout(p=0.2)

        self.fc3     = nn.Linear(32, 1)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch_size, INPUT_SIZE) float tensor
        returns : (batch_size, 1) float tensor — predicted eval in centipawns
        """
        x = self.drop1(self.cr1(self.fc1(x)))
        x = self.drop2(self.cr2(self.fc2(x)))
        x = self.fc3(x)
        return x

    def predict(self, x: torch.Tensor) -> float:
        """
        Single-position inference. Returns eval as a Python float (centipawns).
        x : (1, INPUT_SIZE) or (INPUT_SIZE,) float tensor
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return self.forward(x).item()


# ============================
# QUICK MODEL SUMMARY
# ============================

def print_model_summary():
    model = NNUE()
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nNNUE Model Summary")
    print(f"{'='*40}")
    print(f"  fc1 : {INPUT_SIZE:,} → 64   (weights: {INPUT_SIZE * 64:,})")
    print(f"  cr1 : ClippedReLU + Dropout(0.3)")
    print(f"  fc2 : 64 → 32         (weights: {64 * 32:,})")
    print(f"  cr2 : ClippedReLU + Dropout(0.2)")
    print(f"  fc3 : 32 → 1          (weights: {32:,})")
    print(f"{'='*40}")
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable:,}")

    # Quick forward pass test
    dummy = torch.zeros(4, INPUT_SIZE)
    out = model(dummy)
    print(f"\n  Forward pass test : input {tuple(dummy.shape)} → output {tuple(out.shape)} ✅")


if __name__ == "__main__":
    print_model_summary()