"""Run the automated evaluation suite."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluation.evaluator import run_evaluation

if __name__ == "__main__":
    run_evaluation()
