"""Entry point; all study writes are confined to this directory."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from hpe_sync.cli import main

if __name__ == "__main__":
    main()
