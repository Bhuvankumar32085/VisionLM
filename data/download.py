"""Data download entrypoint alias for scripts/download_dataset.py."""

from pathlib import Path
import sys

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.download_dataset import main

if __name__ == "__main__":
    main()
