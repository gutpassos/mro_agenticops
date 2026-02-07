"""Pipeline orchestration scripts."""
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess import main as preprocess
from src.train import main as train


def run_ingestion_pipeline():
    """Execute full ingestion: preprocess -> train."""
    print("=== MRO AgenticOps - Ingestion Pipeline ===\n")
    
    print("[1/2] Preprocessing PDFs...")
    preprocess()
    
    print("\n[2/2] Training embeddings and building FAISS index...")
    train()
    
    print("\n✓ Ingestion pipeline completed successfully!")


if __name__ == "__main__":
    run_ingestion_pipeline()
