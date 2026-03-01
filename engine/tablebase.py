import chess
import chess.syzygy
import os

TABLEBASE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "assets",
    "syzygy"
)

tb = None

if os.path.exists(TABLEBASE_PATH):
    try:
        tb = chess.syzygy.open_tablebase(TABLEBASE_PATH)
        print("✅ Syzygy tablebases loaded")
    except Exception as e:
        print("⚠️ Failed to load tablebases:", e)
        tb = None
else:
    print("ℹ️ No Syzygy tablebases found, continuing without them")


def probe_tablebase(board):
    if tb is None:
        return None

    if board.is_game_over():
        return None

    # Syzygy works only up to 7 pieces
    if len(board.piece_map()) > 7:
        return None

    try:
        wdl = tb.probe_wdl(board)
        return wdl
    except Exception:
        return None
