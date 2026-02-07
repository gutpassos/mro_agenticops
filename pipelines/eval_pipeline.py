"""Continuous evaluation pipeline."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import main as evaluate


def run_evaluation_pipeline():
    """Execute evaluation with ground truth."""
    print("=== MRO AgenticOps - Evaluation Pipeline ===\n")
    
    print("Running RAG evaluation metrics...")
    evaluate()
    
    print("\n✓ Evaluation completed. Check monitoring/reports/")


if __name__ == "__main__":
    run_evaluation_pipeline()
