import os
import shutil
from datetime import datetime

# ---------------------------
# CONFIG
# ---------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BACKUP_ROOT = os.path.join(PROJECT_ROOT, "checkpoints")

# folders we DO NOT want to backup
IGNORE_FOLDERS = {
    "checkpoints",
    "__pycache__",
    ".git",
    ".idea",
    ".vscode"
}

# files we DO NOT want
IGNORE_FILES = {
    "nnue_dataset.txt",
    "games.pgn"
}

# ---------------------------
# CREATE CHECKPOINT
# ---------------------------

def create_checkpoint():

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_folder = os.path.join(BACKUP_ROOT, f"checkpoint_{timestamp}")

    os.makedirs(backup_folder, exist_ok=True)

    print("Creating checkpoint:", backup_folder)

    for root, dirs, files in os.walk(PROJECT_ROOT):

        # skip unwanted folders
        dirs[:] = [d for d in dirs if d not in IGNORE_FOLDERS]

        # relative path
        rel_path = os.path.relpath(root, PROJECT_ROOT)
        target_dir = os.path.join(backup_folder, rel_path)

        os.makedirs(target_dir, exist_ok=True)

        for file in files:

            # only .py files
            if not file.endswith(".py"):
                continue

            if file in IGNORE_FILES:
                continue

            src_file = os.path.join(root, file)
            dst_file = os.path.join(target_dir, file)

            shutil.copy2(src_file, dst_file)
            print("Saved:", os.path.join(rel_path, file))

    print("\nCheckpoint complete.")


if __name__ == "__main__":
    create_checkpoint()