"""
dataset_builder.py — NNUE Dataset Cleaner
==========================================

Two responsibilities:

1. STUBS (save_position / finalize_game)
   - minimax.py still imports and calls these
   - They are now no-ops — position saving happens in self_play.py
   - Do NOT remove them or the engine will crash on import

2. CLEANING PIPELINE (build_clean_dataset)
   - Reads raw_positions.jsonl written by self_play.py
   - Removes duplicates
   - Removes positions with extreme evals (likely mate scores)
   - Normalizes centipawn eval → [-1.0, +1.0] float
   - Balances White-favored vs Black-favored positions
   - Writes clean_positions.jsonl ready for NNUE training

Run standalone to clean your dataset:
    python -m trainer.dataset_builder
"""

import os
import json
import random
import chess

# ============================
# PATHS
# ============================
RAW_PATH   = os.path.join("datasets", "raw_positions.jsonl")
CLEAN_PATH = os.path.join("datasets", "clean_positions.jsonl")
os.makedirs("datasets", exist_ok=True)


# ============================
# STUBS — keep minimax.py happy
# These are intentional no-ops.
# Position saving now lives entirely in self_play.py
# ============================

def save_position(board: chess.Board, score: int):
    """
    No-op stub. Called from minimax.py quiescence search.
    Position collection is handled by self_play.py instead.
    """
    pass


def finalize_game(result: str):
    """
    No-op stub. Called from self_play.py after each game.
    Result is now written directly by self_play.py via write_positions().
    """
    pass


# ============================
# CLEANING CONFIG
# ============================

# Positions with |eval| above this are likely near-mate — too noisy for NNUE
MAX_EVAL_CP     = 3000

# Normalization range: maps [-MAX_EVAL_CP, +MAX_EVAL_CP] → [-1.0, +1.0]
NORM_DIVISOR    = float(MAX_EVAL_CP)

# If White-favored / Black-favored ratio exceeds this, undersample the majority
MAX_BALANCE_RATIO = 1.5


# ============================
# CLEANING PIPELINE
# ============================

def load_raw(path: str) -> list:
    """Load all records from a .jsonl file. Skips malformed lines."""
    records = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                skipped += 1
    if skipped:
        print(f"  ⚠️  Skipped {skipped} malformed lines")
    return records


def remove_duplicates(records: list) -> list:
    """Keep only the first occurrence of each FEN."""
    seen = set()
    unique = []
    for r in records:
        fen = r.get("fen", "")
        if fen and fen not in seen:
            seen.add(fen)
            unique.append(r)
    return unique


def filter_extreme_evals(records: list) -> list:
    """
    Remove positions with |eval| > MAX_EVAL_CP.
    These are near-mate positions — their eval is unstable and misleading.
    """
    return [r for r in records if abs(r.get("eval", 0)) <= MAX_EVAL_CP]


def filter_missing_result(records: list) -> list:
    """Remove records where result was never filled in (shouldn't happen, but safety check)."""
    valid_results = {"1-0", "0-1", "1/2-1/2"}
    return [r for r in records if r.get("result") in valid_results]


def normalize_eval(records: list) -> list:
    """
    Add a 'eval_norm' field: centipawns mapped to [-1.0, +1.0].
    Clamps values that exceed MAX_EVAL_CP just in case.

    Formula:  eval_norm = clamp(eval / MAX_EVAL_CP, -1.0, 1.0)
    """
    for r in records:
        raw = r.get("eval", 0)
        r["eval_norm"] = max(-1.0, min(1.0, raw / NORM_DIVISOR))
    return records


# Max fraction of dataset that draw positions can occupy
MAX_DRAW_FRACTION = 0.40


def balance_positions(records: list) -> list:
    """
    Two-stage balancing:

    Stage 1 — Cap draws at MAX_DRAW_FRACTION of the dataset.
    Self-play engines naturally draw a lot. We don't want 70% of training
    data to be drawn positions — the network will learn everything is equal.

    Stage 2 — Balance White-favored vs Black-favored positions.
    If one side has more than MAX_BALANCE_RATIO × the other, undersample
    the majority.
    """
    draw_results    = {"1/2-1/2"}
    decisive        = [r for r in records if r.get("result") not in draw_results]
    draws           = [r for r in records if r.get("result") in draw_results]

    # Stage 1: cap draws
    max_draws = int(len(decisive) * (MAX_DRAW_FRACTION / (1 - MAX_DRAW_FRACTION)))
    if len(draws) > max_draws:
        draws = random.sample(draws, max_draws)
        print(f"  Draw cap applied      : kept {len(draws)} draws (max {MAX_DRAW_FRACTION:.0%} of dataset)")

    records = decisive + draws

    # Stage 2: balance White vs Black favored
    white_favored = [r for r in records if r.get("eval_norm", 0) > 0]
    black_favored = [r for r in records if r.get("eval_norm", 0) < 0]
    neutral       = [r for r in records if r.get("eval_norm", 0) == 0]

    w, b = len(white_favored), len(black_favored)

    if w > 0 and b > 0:
        ratio = w / b if w > b else b / w
        if ratio > MAX_BALANCE_RATIO:
            if w > b:
                white_favored = random.sample(white_favored, int(b * MAX_BALANCE_RATIO))
            else:
                black_favored = random.sample(black_favored, int(w * MAX_BALANCE_RATIO))

    balanced = white_favored + black_favored + neutral
    random.shuffle(balanced)
    return balanced


def write_clean(records: list, path: str):
    """Write cleaned records to a .jsonl file."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ============================
# STATS PRINTER
# ============================

def print_stats(records: list, label: str):
    if not records:
        print(f"  {label}: 0 records")
        return

    evals = [r.get("eval", 0) for r in records]
    results = {}
    phases  = {}

    for r in records:
        res = r.get("result", "?")
        ph  = r.get("game_phase", "?")
        results[res] = results.get(res, 0) + 1
        phases[ph]   = phases.get(ph, 0) + 1

    print(f"\n  {label}: {len(records):,} positions")
    print(f"    Eval range : {min(evals):+d} to {max(evals):+d} cp")
    print(f"    Results    : {results}")
    print(f"    Phases     : {phases}")


# ============================
# MAIN CLEANING ENTRY POINT
# ============================

def build_clean_dataset(raw_path=RAW_PATH, clean_path=CLEAN_PATH):
    """
    Full cleaning pipeline. Call this after self_play.py finishes.

    Steps:
      1. Load raw .jsonl
      2. Remove duplicates
      3. Remove extreme evals (near-mate noise)
      4. Remove records without a valid result
      5. Normalize eval → [-1.0, +1.0]
      6. Balance white/black favored positions
      7. Write clean_positions.jsonl
    """
    print("\n🧹 Dataset Builder — Cleaning Pipeline")
    print(f"   Input  : {raw_path}")
    print(f"   Output : {clean_path}\n")

    if not os.path.exists(raw_path):
        print(f"  ❌ Raw dataset not found: {raw_path}")
        print(f"     Run self_play.py first to generate it.")
        return

    # Step 1: Load
    records = load_raw(raw_path)
    print_stats(records, "Raw")

    # Step 2: Remove duplicates
    records = remove_duplicates(records)
    print(f"  After deduplication   : {len(records):,}")

    # Step 3: Remove extreme evals
    records = filter_extreme_evals(records)
    print(f"  After eval filter     : {len(records):,}  (|eval| <= {MAX_EVAL_CP}cp)")

    # Step 4: Remove missing results
    records = filter_missing_result(records)
    print(f"  After result filter   : {len(records):,}")

    # Step 5: Normalize
    records = normalize_eval(records)

    # Step 6: Balance
    before_balance = len(records)
    records = balance_positions(records)
    print(f"  After balancing       : {len(records):,}  (was {before_balance:,})")

    # Step 7: Write
    write_clean(records, clean_path)
    print_stats(records, "Clean")

    print(f"\n  ✅ Saved to: {clean_path}\n")


# ============================
# RUN STANDALONE
# ============================
if __name__ == "__main__":
    build_clean_dataset()