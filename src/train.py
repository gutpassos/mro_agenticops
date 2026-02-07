"""Train vector index or models for RAG."""
from pathlib import Path
from typing import List, Dict, Any
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss


def load_chunks(interim_dir: Path) -> List[Dict[str, Any]]:
    """Load processed chunks from JSON."""
    chunks_file = interim_dir / "chunks.json"
    
    if not chunks_file.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_file}\n"
            "Run preprocess.py first!"
        )
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    return chunks


def generate_embeddings(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    device: str = "cpu"
) -> np.ndarray:
    """
    Generate embeddings for texts using sentence-transformers.
    
    Args:
        texts: List of text strings
        model_name: Model name from sentence-transformers
        batch_size: Batch size for encoding
        device: 'cpu' or 'cuda'
        
    Returns:
        Numpy array of embeddings (n_texts, embedding_dim)
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Important for cosine similarity
    )
    
    return embeddings


def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str = "flat"
) -> faiss.Index:
    """
    Build FAISS index from embeddings.
    
    Args:
        embeddings: Numpy array of embeddings
        index_type: 'flat' (exact) or 'ivf' (approximate)
        
    Returns:
        FAISS index
    """
    dimension = embeddings.shape[1]
    n_vectors = embeddings.shape[0]
    
    print(f"\nBuilding FAISS index ({index_type})...")
    print(f"  Dimension: {dimension}")
    print(f"  Vectors: {n_vectors}")
    
    if index_type == "flat":
        # Exact search with L2 distance (for normalized vectors = cosine)
        index = faiss.IndexFlatIP(dimension)  # IP = Inner Product
    elif index_type == "ivf":
        # Approximate search with IVF (Inverted File Index)
        n_clusters = min(int(np.sqrt(n_vectors)), 100)
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
        
        print(f"  Training IVF with {n_clusters} clusters...")
        index.train(embeddings)
    else:
        raise ValueError(f"Unknown index_type: {index_type}")
    
    # Add vectors to index
    print("  Adding vectors to index...")
    index.add(embeddings)
    
    print(f"✓ Index built with {index.ntotal} vectors")
    
    return index


def save_artifacts(
    index: faiss.Index,
    chunks: List[Dict],
    embeddings: np.ndarray,
    output_dir: Path
) -> None:
    """Save FAISS index, chunks metadata, and embeddings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    index_path = output_dir / "faiss.index"
    faiss.write_index(index, str(index_path))
    print(f"✓ Saved FAISS index: {index_path}")
    
    # Save chunks metadata
    metadata_path = output_dir / "chunks_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved chunks metadata: {metadata_path}")
    
    # Save embeddings (for debugging/reindexing)
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"✓ Saved embeddings: {embeddings_path}")


def main(
    interim_dir: Path = None,
    artifacts_dir: Path = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    device: str = "cpu",
    index_type: str = "flat"
) -> None:
    """
    Main training pipeline: embeddings + FAISS index.
    
    Args:
        interim_dir: Directory with processed chunks
        artifacts_dir: Output directory for artifacts
        model_name: Sentence-transformers model
        batch_size: Batch size for encoding
        device: 'cpu' or 'cuda'
        index_type: 'flat' or 'ivf'
    """
    if interim_dir is None:
        base = Path(__file__).resolve().parents[1]
        interim_dir = base / "data" / "interim"
    
    if artifacts_dir is None:
        base = Path(__file__).resolve().parents[1]
        artifacts_dir = base / "data" / "processed"
    
    print("=== Training Pipeline ===\n")
    
    # 1. Load chunks
    print("[1/3] Loading chunks...")
    chunks = load_chunks(interim_dir)
    texts = [chunk["text"] for chunk in chunks]
    print(f"  Loaded {len(chunks)} chunks\n")
    
    # 2. Generate embeddings
    print("[2/3] Generating embeddings...")
    embeddings = generate_embeddings(
        texts,
        model_name=model_name,
        batch_size=batch_size,
        device=device
    )
    print(f"  Embeddings shape: {embeddings.shape}\n")
    
    # 3. Build FAISS index
    print("[3/3] Building FAISS index...")
    index = build_faiss_index(embeddings, index_type=index_type)
    
    # 4. Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(index, chunks, embeddings, artifacts_dir)
    
    print("\n✓ Training complete!")
    print(f"✓ Artifacts saved to: {artifacts_dir}")


if __name__ == "__main__":
    main()
