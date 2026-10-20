from pathlib import Path
import sys

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "src"))
sys.path.insert(0, str(STUDY.parents[1] / "src"))
