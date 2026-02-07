"""Retrieval engine with FAISS vector store."""
from pathlib import Path
from typing import List, Dict, Any
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class VectorRetriever:
    """FAISS-based vector retriever for semantic search."""
    
    def __init__(self, index_path: Path, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize retriever with FAISS index.
        
        Args:
            index_path: Path to FAISS index file
            embedding_model: Sentence-transformers model name
        """
        self.index_path = Path(index_path)
        self.embedding_model_name = embedding_model
        self.index = None
        self.metadata = None
        self.encoder = None
        
    def load_index(self) -> None:
        """Load FAISS index and metadata from disk."""
        # Load FAISS index
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        
        print(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        print(f"✓ Loaded index with {self.index.ntotal} vectors")
        
        # Load metadata
        metadata_path = self.index_path.parent / "chunks_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f"✓ Loaded {len(self.metadata)} chunk metadata")
        
        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.encoder = SentenceTransformer(self.embedding_model_name)
        print("✓ Retriever ready!")
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query string."""
        if self.encoder is None:
            self.load_index()
        
        # Encode and normalize (same as training)
        embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant chunks for a query.
        
        Args:
            query: User query string
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of dicts with 'text', 'score', 'metadata'
        """
        if self.index is None:
            self.load_index()
        
        # 1. Embed query
        query_embedding = self.embed_query(query)
        
        # 2. Search FAISS index
        # Returns distances (Inner Product = cosine for normalized vectors)
        # and indices of nearest neighbors
        distances, indices = self.index.search(query_embedding, top_k)
        
        # 3. Filter by threshold and build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # Convert to similarity score (IP distance is already similarity for normalized)
            score = float(dist)
            
            if score >= similarity_threshold:
                chunk = self.metadata[idx]
                results.append({
                    "text": chunk["text"],
                    "score": score,
                    "metadata": chunk.get("metadata", {})
                })
        
        return results
    
    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: Original query
            candidates: List of retrieved candidates
            
        Returns:
            Reranked list
        """
        # TODO: Implement cross-encoder reranking
        # For now, just return as-is (already sorted by FAISS score)
        return candidates


class HybridRetriever(VectorRetriever):
    """Combines semantic (FAISS) + keyword (BM25) retrieval."""
    
    def __init__(self, *args, bm25_weight: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.bm25_weight = bm25_weight
        self.bm25_index = None
        
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict]:
        """Hybrid retrieval with score fusion."""
        # TODO: Implement hybrid retrieval
        # 1. Get semantic results
        semantic_results = super().retrieve(query, top_k * 2, **kwargs)
        
        # 2. Get BM25 results (when implemented)
        # bm25_results = self._bm25_search(query, top_k * 2)
        
        # 3. Reciprocal rank fusion
        # For now, just return semantic results
        return semantic_results[:top_k]
