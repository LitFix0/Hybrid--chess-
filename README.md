# ♟ Hybrid Chess — RL-NNUE Engine

A self-learning chess engine built from scratch in Python. Uses Negamax + Alpha-Beta search with Iterative Deepening, and a NNUE neural network evaluator trained through self-play reinforcement learning. The engine generates its own training data, trains on it, and gets stronger each cycle.


---

## 🚀 Run

```bash
python play.py
```

**Controls:**
| Key | Action |
|-----|--------|
| `1` | Easy difficulty |
| `2` | Medium difficulty |
| `3` | Hard difficulty |
| `R` | Reset game |

---

## 🧠 How It Works

### Search Engine
The engine uses classical chess search techniques:
- **Negamax** with Alpha-Beta pruning
- **Iterative Deepening** with Aspiration Windows
- **Move ordering** — TT move, captures (SEE), killer moves, history heuristic
- **Null move pruning** and **Late Move Reduction (LMR)**
- **Quiescence search** to resolve tactical positions
- **Transposition Table** for caching previously searched positions
- **Opening Book** and **Syzygy endgame tablebase** support

### NNUE Evaluator
Instead of hand-crafted evaluation, the engine uses a neural network:

- **Feature set:** HalfKP — each non-king piece is encoded as a `(king_square, piece_type, piece_square)` triple, computed from both White and Black's perspective
- **Input size:** 81,920 binary features
- **Architecture:** `81920 → 64 → 32 → 1` with ClippedReLU and Dropout
- **Output:** centipawn evaluation from the side to move's perspective

### Reinforcement Learning Loop
```
Self-Play → Position Dataset → Clean & Normalize → Train NNUE → Stronger Engine → Repeat
```
1. Engine plays against itself and records every position searched at depth ≥ 4
2. Dataset is cleaned: duplicates removed, extreme evals filtered, draws capped at 40%, positions balanced
3. NNUE is trained on `(position, eval)` pairs via MSE loss in PyTorch
4. New weights are loaded into the engine automatically
5. Stronger engine generates better training data next cycle

---

## 🗂 Project Structure

```
CHESS_AI/
├── engine/
│   ├── minimax.py          # Negamax + Alpha-Beta search
│   ├── iterative.py        # Iterative deepening driver
│   ├── evaluation.py       # Eval toggle (NNUE / classical)
│   ├── nnue_eval.py        # NNUE inference
│   ├── transposition.py    # Transposition table
│   ├── search_control.py   # Time management
│   ├── see.py              # Static Exchange Evaluation
│   ├── tablebase.py        # Syzygy tablebase
│   └── opening_book.py     # Opening book
│
├── trainer/
│   ├── feature_encoder.py  # HalfKP feature encoder
│   ├── nnue_model.py       # PyTorch NNUE model
│   ├── train.py            # Training script
│   └── dataset_builder.py  # Data cleaning pipeline
│
├── datasets/
│   ├── raw_positions.jsonl # Self-play output
│   └── clean_positions.jsonl # Cleaned training data
│
├── models/
│   └── nnue_weights.pt     # Trained weights
│
├── gui/
│   └── chess_gui.py        # Pygame GUI
│
├── play.py                 # Entry point
└── requirements.txt
```

---

## 🔁 Training Your Own Model

```bash
# 1. Generate self-play data
python self_play.py

# 2. Clean and balance the dataset
python -m trainer.dataset_builder

# 3. Train the NNUE
python -m trainer.train
```

More games = better data = stronger engine. Run step 1 multiple times to accumulate data before retraining.

---

## 📦 Installation

```bash
git clone https://github.com/LitFix0/hybrid-chess.git
cd hybrid-chess
pip install -r requirements.txt
python play.py
```

**Requirements:**
```
python-chess
pygame
torch
```

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Search | Pure Python (Negamax + Alpha-Beta) |
| Neural Network | PyTorch |
| Chess Logic | python-chess |
| GUI | pygame |
| Feature Encoding | HalfKP (custom implementation) |

---

## 📌 Key Concepts

`Negamax` `Alpha-Beta Pruning` `NNUE` `HalfKP` `Iterative Deepening` `Aspiration Windows` `Bootstrapped Reinforcement Learning` `Self-Play` `Transposition Table` `Quiescence Search`
