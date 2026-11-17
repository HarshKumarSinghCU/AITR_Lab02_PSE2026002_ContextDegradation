"""Run the factorial experiment from the repository root."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_pipeline import main

if __name__ == "__main__":
    main()
